"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "Donut Embezzler"
AGENT_NAME = "Holey Donut"


# ===== YOUR CODE HERE =====
#
# Strategy overview
# -----------------
# * Equity engine: a numpy Monte-Carlo simulation of the rest of the hand
#   against sampled opponent holdings. A "pair x triple" lookup table (built
#   once at import time from evaluator's table, so it agrees with
#   omaha_score) scores thousands of Omaha hands per millisecond.
# * Opponent model: counts how often the opponent opens, re-raises, bets,
#   raises, calls and folds (per street). Sampled opponent hands are then
#   weighted by those frequencies: "they bet 30% of the time, so their
#   betting range is roughly the top 30% of hands on this board". Equity is
#   therefore measured against a range, not against a random hand.
# * Decisions: expected value of fold / check-call / raise-half / raise-pot
#   from the range equity, pot odds and the opponent's observed fold
#   frequency (fold equity). Pure bluffs are rate-limited.
# * Time: the number of samples adapts to the time bank; every decision is
#   wrapped so that a bug can never produce an illegal action.

import math
import sys
import time
from itertools import chain, combinations

import numpy as np

from evaluator import get_table

_CAT = 1 << 20                      # one hand category, in score units
_NT = 7140                          # C(36, 3): number of board triples
_BIN = np.zeros((37, 6), dtype=np.int64)
for _n in range(37):
    for _k in range(6):
        _BIN[_n, _k] = math.comb(_n, _k)
_B1, _B2, _B3 = _BIN[:, 1], _BIN[:, 2], _BIN[:, 3]
_PAIRS = np.array(list(combinations(range(5), 2)), dtype=np.int64)
_TRIPLES = {k: np.array(list(combinations(range(k), 3)), dtype=np.int64) for k in (3, 4, 5)}
_AR9 = np.arange(9)
_AR4 = np.arange(4)
_POP = np.array([bin(i).count("1") for i in range(512)], dtype=np.int64)
# straight windows as 9-bit rank masks (6..A = bits 0..8); last one is the wheel A6789
_WINDOWS = np.array([sum(1 << r for r in range(lo, lo + 5)) for lo in range(5)] + [(1 << 8) | 0b1111],
                    dtype=np.int64)
# linear preflop-strength proxy, fitted offline against Monte-Carlo equity (R^2 = 0.89)
_PRE_COEF = np.array([0.3232, -0.0429, -0.0119, 0.4685, 0.0423, 0.3164, 0.0585, -0.021, 0.0014,
                      -0.0033, -0.0407, 0.0364, 0.0745, 0.983, -0.0332, -0.6587])


def _hand_idx(holes: np.ndarray) -> np.ndarray:
    """(N,5) holdings (any order) -> unique index 0..376991 (colex of the sorted cards)."""
    s = np.sort(holes, axis=1)
    return _B1[s[:, 0]] + _B2[s[:, 1]] + _B3[s[:, 2]] + _BIN[s[:, 3], 4] + _BIN[s[:, 4], 5]


def _build_tables():
    """Build every lookup table once (about one second):

    _TBL        score rank of (2 hole cards, 3 board cards), by pair_idx * _NT + triple_idx.
                Scores are compressed to dense int16 ranks (same order as omaha_score).
    _VAL        dense rank -> category + position inside the category (float).
    _PRE_PCT    per 5-card holding: preflop strength percentile (0..1).
    _RMASK      per holding: 9-bit mask of ranks present.
    _SCNT       per holding: number of cards of each suit.
    """
    combos = np.fromiter(chain.from_iterable(combinations(range(36), 5)), dtype=np.int64,
                         count=376992 * 5).reshape(-1, 5)
    masks = (np.int64(1) << combos).sum(axis=1)
    table = get_table()
    scores = np.fromiter((table[m] for m in masks.tolist()), dtype=np.int64, count=len(masks))
    uniq, dense = np.unique(scores, return_inverse=True)
    dense = dense.astype(np.int16)
    cat = uniq >> 20
    val = np.empty(uniq.size, dtype=np.float64)
    for c in range(9):
        sel = np.nonzero(cat == c)[0]
        val[sel] = c + np.arange(sel.size) / max(1, sel.size)
    tbl = np.zeros(630 * _NT, dtype=np.int16)
    for pi in combinations(range(5), 2):
        ti = [k for k in range(5) if k not in pi]
        a, b = combos[:, pi[0]], combos[:, pi[1]]
        c, d, e = combos[:, ti[0]], combos[:, ti[1]], combos[:, ti[2]]
        tbl[(_B1[a] + _B2[b]) * _NT + _B1[c] + _B2[d] + _B3[e]] = dense
    idx = _hand_idx(combos)
    strength = np.empty(combos.shape[0], dtype=np.float64)
    for lo in range(0, combos.shape[0], 50000):
        strength[lo:lo + 50000] = _preflop_strength(combos[lo:lo + 50000])
    order = np.argsort(strength, kind="stable")
    pct_by_row = np.empty(combos.shape[0], dtype=np.float32)
    pct_by_row[order] = np.arange(combos.shape[0], dtype=np.float32) / combos.shape[0]
    pre_pct = np.empty(combos.shape[0], dtype=np.float32)
    pre_pct[idx] = pct_by_row
    ranks = combos >> 2
    rmask = np.empty(combos.shape[0], dtype=np.int64)
    rmask[idx] = ((ranks[:, :, None] == _AR9).any(1) << _AR9).sum(1)
    scnt = np.empty((combos.shape[0], 4), dtype=np.int8)
    scnt[idx] = ((combos & 3)[:, :, None] == _AR4).sum(1)
    straight_rank = int(np.searchsorted(uniq, 3 << 20))
    return tbl, val, pre_pct, rmask, scnt, straight_rank


def _pair_idx(cards: np.ndarray) -> np.ndarray:
    """(..., 5) hole cards -> (..., 10) indices of the 10 two-card combinations."""
    a = cards[..., _PAIRS[:, 0]]
    b = cards[..., _PAIRS[:, 1]]
    return _B1[np.minimum(a, b)] + _B2[np.maximum(a, b)]


def _triple_idx(board: np.ndarray) -> np.ndarray:
    """(..., k) board cards -> (..., C(k,3)) indices of the three-card combinations."""
    t = _TRIPLES[board.shape[-1]]
    a, b, c = board[..., t[:, 0]], board[..., t[:, 1]], board[..., t[:, 2]]
    lo = np.minimum(np.minimum(a, b), c)
    hi = np.maximum(np.maximum(a, b), c)
    return _B1[lo] + _B2[a + b + c - lo - hi] + _B3[hi]


def _best(pidx: np.ndarray, tidx: np.ndarray) -> np.ndarray:
    """pidx (10,) or (N,10) or (1,10); tidx (N,T) or (1,T) -> best Omaha score
    rank per row (N,). Either argument may be a single row that broadcasts."""
    if pidx.ndim == 1:
        pidx = pidx[None, :]
    idx = pidx[:, :, None] * _NT + tidx[:, None, :]          # (N, 10, T)
    return _TBL[idx].reshape(idx.shape[0], -1).max(axis=1)


def _preflop_strength(holes: np.ndarray) -> np.ndarray:
    """(N,5) holdings -> approximate preflop equity vs a random hand (N,)."""
    r = holes >> 2
    s = holes & 3
    cnt_r = (r[:, :, None] == _AR9).sum(1)
    cnt_s = (s[:, :, None] == _AR4).sum(1)
    rmask = ((cnt_r > 0) << _AR9).sum(1)
    pr = np.where(cnt_r == 2, _AR9, -1).max(1)
    ace_s = ((r == 8)[:, :, None] & (s[:, :, None] == _AR4)).any(1)
    king_s = ((r == 7)[:, :, None] & (s[:, :, None] == _AR4)).any(1)
    wc = _POP[rmask[:, None] & _WINDOWS[None, :]]
    adj = (rmask & (rmask >> 1)) & 0xFF
    f = np.stack([
        r.sum(1) / 40.0,
        r.max(1) / 8.0,
        cnt_r[:, 8] / 2.0,
        (cnt_r == 2).sum(1) / 2.0,
        np.where(pr >= 0, (pr + 1) / 9.0, 0.0),
        (cnt_r >= 3).any(1).astype(float),
        (cnt_s >= 2).sum(1) / 2.0,
        (cnt_s >= 3).any(1).astype(float),
        (ace_s & (cnt_s >= 2)).any(1).astype(float),
        (king_s & (cnt_s >= 2) & ~ace_s).any(1).astype(float),
        wc.max(1) / 5.0,
        (wc >= 3).sum(1) / 6.0,
        (wc >= 4).sum(1) / 6.0,
        (cnt_r > 0).sum(1) / 5.0,
        _POP[adj] / 4.0,
        np.ones(holes.shape[0]),
    ], axis=1)
    return f @ _PRE_COEF


_TBL, _VAL, _PRE_PCT, _RMASK, _SCNT, _STRAIGHT_RANK = _build_tables()


def _board_strength(pidx: np.ndarray, hidx: np.ndarray, board: np.ndarray) -> np.ndarray:
    """Made hand (in category units) of each sampled holding on the board seen
    so far, plus a draw bonus before the river. Only used to RANK the
    opponent's possible holdings. pidx: (N,10) pair indices, hidx: (N,) hand indices."""
    k = board.shape[0]
    made = _best(pidx, _triple_idx(board[None, :]))
    val = _VAL[made]
    if k == 5:
        return val
    bs = np.bincount(board & 3, minlength=4)                            # suit counts on board
    # Omaha flush uses exactly two hole cards. A draw (one card to come) is two
    # hole cards of a suit and exactly two of that suit on the board. Two hole
    # cards plus three board cards is a made flush (already in `val`). Two hole
    # cards plus one board card is only a backdoor and gets no draw bonus.
    # One hole card plus three board cards is not a flush draw at all.
    sc = _SCNT[hidx]
    flush_draw = ((sc >= 2) & (bs[None, :] == 2)).any(1)
    bmask = 0
    for x in board.tolist():
        bmask |= 1 << (x >> 2)
    hole_mask = _RMASK[hidx][:, None]
    windows = _WINDOWS[None, :]
    board_mask = np.int64(bmask)
    hw = _POP[hole_mask & windows]
    bw = _POP[board_mask & windows]
    # Union count: a rank on both the holding and the board is one rank, not two.
    union = _POP[(hole_mask | board_mask) & windows]
    # A one-card straight draw needs two hole ranks and two board ranks already
    # in the window, with at least four distinct ranks between them. Three or
    # more hole ranks plus fewer than two board ranks can reach union 4 and
    # still have no 2-hole + 3-board straight (6c7d8h9s6d on Ac Kd Qh Ks).
    # Union 3 is a two-card backdoor and gets no bonus.
    straight_draw = ((hw >= 2) & (bw >= 2) & (union >= 4)).any(1) & (made < _STRAIGHT_RANK)
    scale = 1.0 if k == 3 else 0.6
    return val + scale * (1.3 * flush_draw + 0.8 * straight_draw)


def _percentile(v: np.ndarray) -> np.ndarray:
    """Rank of each value inside the sample, scaled to 0..1."""
    order = np.argsort(v, kind="stable")
    out = np.empty(v.size, dtype=np.float64)
    out[order] = np.arange(v.size, dtype=np.float64) / v.size
    return out


# Random permutations, drawn once, so that sampling cards is a table lookup
# instead of a sort. One pool per deck size (36 - 5 hole - 0/3/4/5 board).
_POOL_ROWS = 40000
_PERM_POOL = {k: np.random.default_rng(12345 + k).permuted(
    np.tile(np.arange(k, dtype=np.int8), (_POOL_ROWS, 1)), axis=1) for k in (31, 28, 27, 26)}


def _make_pool(rng, hole, board, m: int) -> dict:
    """Draw m candidate opponent holdings, each with its own run-out of the board.

    The pool is cheap to rank (which hands fit the opponent's actions) and the
    expensive part, simulating the run-out, is only done for the holdings that
    get selected (see _fill_wins). On the river, ranking and scoring coincide."""
    hole_a = np.array(hole, dtype=np.int64)
    board_a = np.array(board, dtype=np.int64)
    known = set(hole) | set(board)
    deck = np.array([c for c in range(36) if c not in known], dtype=np.int64)
    need = 10 - len(board)
    perms = _PERM_POOL.get(deck.size)
    if perms is None:
        pick = np.argpartition(rng.random((m, deck.size)), need - 1, axis=1)[:, :need]
    else:
        pick = perms[rng.integers(0, _POOL_ROWS, m), :need]
    draws = deck[pick]                                   # (m, need): 5 opponent cards, then the run-out
    opp = draws[:, :5]
    hidx = _hand_idx(opp)
    pool = {"m": m, "draws": draws, "board": board_a, "my_pidx": _pair_idx(hole_a),
            "pidx": _pair_idx(opp), "hidx": hidx,
            "pct": {0: _PRE_PCT[hidx].astype(np.float64)},
            "win": np.full(m, np.nan)}
    if len(board) == 5:
        tidx = _triple_idx(board_a[None, :])
        my = _best(pool["my_pidx"], tidx)
        op = _best(pool["pidx"], tidx)
        pool["win"] = (my > op).astype(np.float64) + 0.5 * (my == op)
        pool["pct"][3] = _percentile(op.astype(np.float64))
    return pool


def _extend_pool(rng, pool: dict, hole, m: int) -> None:
    """Add m more candidates to a pool (same hole cards and board).

    Preflop percentiles come from the global ``_PRE_PCT`` table, so they stay
    comparable when concatenated. Street percentiles are ranks inside one
    sample. They are dropped and recomputed on the combined pool the next
    time ``_street_pct`` is asked, instead of being concatenated batch-wise.
    """
    more = _make_pool(rng, hole, pool["board"], m)
    pre = np.concatenate([pool["pct"][0], more["pct"][0]])
    for key in ("draws", "pidx", "hidx", "win"):
        pool[key] = np.concatenate([pool[key], more[key]])
    pool["m"] += m
    pool["pct"] = {0: pre}


def _street_pct(pool: dict, street: int) -> np.ndarray:
    """Opponent strength percentile on a street (computed on first use)."""
    pct = pool["pct"].get(street)
    if pct is None:
        k = {1: 3, 2: 4, 3: 5}[street]
        pct = _percentile(_board_strength(pool["pidx"], pool["hidx"], pool["board"][:k]))
        pool["pct"][street] = pct
    return pct


def _fill_wins(pool: dict, idx: np.ndarray) -> None:
    """Simulate the run-out for the candidates idx and store our win share."""
    draws = pool["draws"][idx]
    k = pool["board"].size
    full = np.empty((idx.size, 5), dtype=np.int64)
    full[:, :k] = pool["board"]
    full[:, k:] = draws[:, 5:]
    tidx = _triple_idx(full)
    my = _best(pool["my_pidx"], tidx)
    op = _best(pool["pidx"][idx], tidx)
    pool["win"][idx] = (my > op).astype(np.float64) + 0.5 * (my == op)


# ----- opponent statistics ---------------------------------------------------

_PRIORS = {"pre_open": 0.5, "pre_re": 0.12, "pre_fold": 0.3, "bet": 0.45, "raise": 0.12,
           "fold_s": 0.3, "fold_b": 0.42}
_SD_KINDS = ("open_raise", "reraise", "call_raise", "bet", "raise", "call")


def _fresh_stats() -> dict:
    st = {k: [0.0, 0.0] for k in ("pre_open", "pre_re", "pre_fold")}
    for k in ("bet", "raise", "fold_s", "fold_b"):      # fold_s / fold_b: facing a small / big bet
        st[k] = {1: [0.0, 0.0], 2: [0.0, 0.0], 3: [0.0, 0.0], "all": [0.0, 0.0]}
    st["sd"] = {}       # (kind, street) -> [sum of strength percentiles, count], from showdowns
    return st


def _size_class(bet: float, pot_before: float) -> str:
    """'s' for bets up to about two thirds of the pot, 'b' for bigger ones."""
    return "b" if bet > 0.65 * max(1.0, pot_before) else "s"


def _count(st: dict, street: int, kind: str, size: str = "s") -> None:
    def pre(key, hit):
        st[key][0] += hit
        st[key][1] += 1

    def post(key, hit):
        for s in (street, "all"):
            st[key][s][0] += hit
            st[key][s][1] += 1

    if kind == "open_raise":
        pre("pre_open", 1)
    elif kind == "open_pass":
        pre("pre_open", 0)
    elif kind == "reraise":
        pre("pre_re", 1)
        pre("pre_fold", 0)
    elif kind == "call_raise":
        pre("pre_re", 0)
        pre("pre_fold", 0)
    elif kind == "fold":
        if street == 0:
            pre("pre_fold", 1)
            pre("pre_re", 0)
        else:
            post("fold_" + size, 1)
            post("raise", 0)
    elif kind == "bet":
        post("bet", 1)
    elif kind == "check":
        post("bet", 0)
    elif kind == "raise":
        post("raise", 1)
        post("fold_" + size, 0)
    elif kind == "call":
        post("raise", 0)
        post("fold_" + size, 0)


def _decay_stats(st: dict, factor: float) -> None:
    """Fade every counter so that recent hands matter most (adaptive opponents)."""
    for key, val in st.items():
        if key == "sd":
            for pair in val.values():
                pair[0] *= factor
                pair[1] *= factor
        elif isinstance(val, dict):
            for pair in val.values():
                pair[0] *= factor
                pair[1] *= factor
        else:
            val[0] *= factor
            val[1] *= factor


def _f_pre(st: dict, key: str) -> float:
    k, n = st[key]
    return (k + 6.0 * _PRIORS[key]) / (n + 6.0)


def _f_post(st: dict, key: str, street: int) -> float:
    k, n = st[key]["all"]
    pooled = (k + 6.0 * _PRIORS[key]) / (n + 6.0)
    ks, ns = st[key][street]
    return (ks + 8.0 * pooled) / (ns + 8.0)


def _x_for(st: dict, street: int, kind: str, size: str) -> float:
    """Fraction of hands (strongest first) the opponent takes this action with.
    Starts from how often they do it, corrected by what they showed down."""
    if kind == "open_raise":
        x = max(0.15, _f_pre(st, "pre_open"))
    elif kind == "reraise":
        x = max(0.08, _f_pre(st, "pre_re"))
    elif kind == "call_raise":
        x = max(0.2, 1.0 - _f_pre(st, "pre_fold"))
    elif kind == "bet":
        x = max(0.15, _f_post(st, "bet", street))
    elif kind == "raise":
        x = max(0.08, _f_post(st, "raise", street))
    else:   # call
        x = max(0.2, 1.0 - _f_post(st, "fold_" + size, street))
    sd = st["sd"].get((kind, street))
    if sd is not None and sd[1] > 0:
        # a "top x" range has mean percentile 1 - x/2; hands seen at showdown
        # are biased towards strength, so this only gets partial weight
        x_obs = min(1.0, max(0.05, 2.0 * (1.0 - sd[0] / sd[1])))
        wgt = _P["sd_weight"][street] * sd[1] / (sd[1] + 12.0)
        x = (1.0 - wgt) * x + wgt * x_obs
    return x


def _replay(obs: Observation, memory: dict) -> list:
    """Walk this hand's history. Returns [(street, kind, bet size class)] for
    every opponent action and records the ones not seen before in the statistics."""
    cur = memory["cur"]
    st = memory["stats"]
    bb, sb, stack0 = memory["bb"], memory["sb"], memory["stack0"]
    if obs.is_button:
        tot, put = [sb, bb], [sb, bb]       # index 0 = me, 1 = opponent
    else:
        tot, put = [bb, sb], [bb, sb]
    street, raises = 0, 0
    acts = []
    for i, rec in enumerate(obs.action_history):
        if int(rec.street) != street:
            street, raises = int(rec.street), 0
            tot = [0, 0]
        a = 1 if not rec.is_me else 0
        o = 1 - a
        to_call = tot[o] - tot[a]
        is_raise = rec.action in (Action.RAISE_HALF, Action.RAISE_POT)
        can_raise = raises < memory["raise_cap"] and (stack0 - put[a]) > to_call and (stack0 - put[o]) > 0
        if a == 1:
            kind = None
            size = _size_class(to_call, put[0] + put[1] - to_call) if to_call > 0 else "s"
            if rec.action == Action.FOLD:
                kind = "fold"
            elif street == 0:
                if to_call > 0 and tot[o] > bb:
                    kind = "reraise" if is_raise else "call_raise"
                elif is_raise:
                    kind = "open_raise"
                elif can_raise:
                    kind = "open_pass"
            else:
                if to_call > 0:
                    kind = "raise" if is_raise else "call"
                elif is_raise:
                    kind = "bet"
                elif can_raise:
                    kind = "check"
            if kind is not None:
                acts.append((street, kind, size))
                if i >= cur["hist_n"]:
                    _count(st, street, kind, size)
        tot[a] = rec.street_total
        put[a] += rec.amount
        if is_raise:
            raises += 1
    cur["hist_n"] = len(obs.action_history)
    return acts


def _opp_pct_on_board(memory: dict, opp: np.ndarray, board: np.ndarray, hole) -> float:
    """Strength percentile of a revealed opponent hand on a board, against
    random holdings (the same yardstick the range model uses)."""
    known = set(hole) | set(board.tolist())
    deck = np.array([c for c in range(36) if c not in known], dtype=np.int64)
    perms = _PERM_POOL.get(deck.size)
    if perms is None:
        return 0.5
    cand = deck[perms[memory["rng"].integers(0, _POOL_ROWS, 160), :5]]
    ref = _board_strength(_pair_idx(cand), _hand_idx(cand), board)
    mine = _board_strength(_pair_idx(opp[None, :]), _hand_idx(opp[None, :]), board)[0]
    return float((ref < mine).mean() + 0.5 * (ref == mine).mean())


def _calibrate(memory: dict, result: HandResult, acts: list, hole) -> None:
    """After a showdown: note how strong the opponent's hand was on each
    street they bet, raised or called on."""
    st = memory["stats"]
    opp = np.array(result.opp_hole, dtype=np.int64)
    board = np.array(result.board, dtype=np.int64)
    pcts = {0: float(_PRE_PCT[_hand_idx(opp[None, :])[0]])}
    for street, kind, _size in acts:
        if kind not in _SD_KINDS:
            continue
        if street not in pcts:
            k = {1: 3, 2: 4, 3: 5}[street]
            if board.size < k:
                continue
            pcts[street] = _opp_pct_on_board(memory, opp, board[:k], hole)
        sd = st["sd"].setdefault((kind, street), [0.0, 0.0])
        sd[0] += pcts[street]
        sd[1] += 1


def _top(pct: np.ndarray, x: float) -> np.ndarray:
    """Weight that keeps the top x fraction of hands (soft edge, small floor)."""
    floor = _P["top_floor"]
    return floor + (1.0 - floor) / (1.0 + np.exp(-(pct - (1.0 - x)) / _P["tau"]))


def _cap(pct: np.ndarray, x: float) -> np.ndarray:
    """Weight that mildly removes the top x fraction (they would have bet)."""
    return 1.0 - _P["cap_depth"] / (1.0 + np.exp(-(pct - (1.0 - x)) / _P["tau"]))


def _range_weights(pool: dict, acts: list, st: dict, now: int) -> np.ndarray:
    w = np.ones(pool["m"])
    if not _P["narrow"]:
        return w
    for street, kind, size in acts:
        if kind == "check" and street != now:
            continue        # a check on an earlier street says little; skip the work
        p = pool["pct"][0] if street == 0 else _street_pct(pool, street)
        if kind == "open_pass":
            w = w * _cap(p, _f_pre(st, "pre_open"))
        elif kind == "check":
            w = w * _cap(p, _f_post(st, "bet", street))
        elif kind in _SD_KINDS:
            w = w * _top(p, _x_for(st, street, kind, size))
    return w


# ----- time management ---------------------------------------------------------

_TRACE = None                       # a list here records every decision (local debugging only)
_BASE_M = (500, 400, 400, 400)      # candidate opponent holdings per street (preflop, flop, turn, river)
_BASE_K = 240                       # holdings simulated per decision
_BIG_K = 400                        # ... when the decision is big
_OVERHEAD = 0.0004                  # seconds per request not visible from inside (JSON, HTTP), unloaded
_REF_DRAW = (0.12e-3, 1.4e-6)       # draw cost on the development machine: fixed + per sample
_BANK_SHARE = 0.70                  # fraction of the time bank we plan to use (margin for slow machines)


def _load(memory: dict) -> float:
    """How much slower than the development machine we are running (>= 0.7)."""
    return min(6.0, max(0.7, memory["load"]))


def _spent(memory: dict) -> float:
    """Estimate of the time the judge has charged so far: our own measured
    time plus a per-request overhead that grows with machine load."""
    return memory["time_used"] + memory["calls"] * _OVERHEAD * _load(memory)


def _timed_fill(memory: dict, pool: dict, idx: np.ndarray) -> None:
    """_fill_wins, timed, to estimate how loaded the machine is."""
    t = time.perf_counter()
    _fill_wins(pool, idx)
    load = (time.perf_counter() - t) / (_REF_DRAW[0] + _REF_DRAW[1] * idx.size)
    memory["load"] = 0.9 * memory["load"] + 0.1 * load if memory["load"] else load
    k = pool["board"].size
    memory["draws"][k] = memory["draws"].get(k, 0) + 1
    memory["drawn"][k] = memory["drawn"].get(k, 0) + idx.size


def _range_equity(memory: dict, pool: dict, w: np.ndarray, k: int) -> tuple:
    """Equity against the weighted pool: resample k holdings in proportion to
    their weight and simulate only those. Returns (equity, distinct holdings)."""
    m = pool["m"]
    if w.min() == w.max():
        sel = np.arange(min(m, k))
    else:
        sel = memory["rng"].choice(m, size=k, p=w / w.sum())
    uniq = np.unique(sel)
    todo = uniq[np.isnan(pool["win"][uniq])]
    if todo.size:
        _timed_fill(memory, pool, todo)
    return float(pool["win"][sel].mean()), int(uniq.size)


def _new_hand_accounting(memory: dict) -> None:
    """Update the per-hand cost estimate and the sample multiplier."""
    spent = _spent(memory)
    cost = spent - memory["hand_mark"]
    memory["hand_mark"] = spent
    if memory["hands"] < 3:
        return
    memory["hand_cost"] = 0.93 * memory["hand_cost"] + 0.07 * cost if memory["hand_cost"] else cost
    remaining_hands = max(1, memory["num_hands"] - memory["hands"])
    allowed = (_BANK_SHARE * memory["bank"] - spent) / remaining_hands
    ratio = allowed / max(memory["hand_cost"], 1e-5)
    memory["scale"] = min(1.6, max(0.45, memory["scale"] * min(1.25, max(0.7, ratio))))


_FIXED_SCALE = None                 # local testing only: pin the sample multiplier, ignore the clock


def _scale(memory: dict) -> float:
    if _FIXED_SCALE is not None:
        return _FIXED_SCALE
    if _spent(memory) > 0.88 * memory["bank"]:
        return 0.15
    return max(0.45, min(1.6, memory["scale"]))


def _time_ok(memory: dict, share: float) -> bool:
    return _FIXED_SCALE is not None or _spent(memory) < share * memory["bank"]


def _cheap_decide(obs: Observation) -> Action:
    """Emergency decision without simulation (only when the bank is nearly empty)."""
    legal = obs.legal_actions
    hole = np.array(obs.hole_cards, dtype=np.int64)
    if obs.street == 0:
        pct = float(_PRE_PCT[_hand_idx(hole[None, :])[0]])
        if pct > 0.85 and Action.RAISE_POT in legal:
            return Action.RAISE_POT
        if obs.to_call == 0 or obs.to_call <= 2 * obs.big_blind or pct > 0.6:
            return Action.CHECK_CALL
        return Action.FOLD
    board = np.array(obs.board, dtype=np.int64)
    cat = int(_VAL[_best(_pair_idx(hole), _triple_idx(board[None, :]))[0]])
    if cat >= 4 and Action.RAISE_POT in legal:
        return Action.RAISE_POT
    if obs.to_call == 0:
        return Action.RAISE_HALF if cat >= 2 and Action.RAISE_HALF in legal else Action.CHECK_CALL
    if cat >= 2 or (cat >= 1 and obs.to_call * 3 <= obs.pot):
        return Action.CHECK_CALL
    return Action.FOLD


# ----- the decision ---------------------------------------------------------------

# Tunable strategy parameters (kept in one place so they can be swept in tests).
_P = {
    "fold_k": (0.12, 0.3, 0.35, 0.4),   # how much weaker the hands an opponent folds are, per street
    "realise": (0.0, 0.05, 0.03, 0.0),  # equity lost when our bet is called on early streets
    "oop_pen": 0.02,                    # equity penalty out of position before the river
    "call_margin": (0.0, 0.05, 0.04, 0.02),  # extra equity needed to call, per street (future bets)
    "decay": 0.995,                     # per hand: old opponent statistics fade (window ~200 hands)
    "raise_margin": 0.03,               # x pot: prefer the passive line on ties
    "noise": 0.02,                      # x pot: random EV jitter (mixing)
    "bluff_lo": 0.3,                    # equity below which a bet counts as a bluff
    "bluff_f0": 0.3, "bluff_slope": 2.0, "bluff_max": 0.75,
    "junk": 0.37,                       # button open-folds below this preflop equity
    "reraise_coef": 1.5,
    "size_f": (0.7, 0.4),               # preflop: fold probability scaling with raise size
    "narrow": 1.0,                      # 1 = weight opponent hands by their actions, 0 = random hand
    "top_floor": 0.04, "tau": 0.06, "cap_depth": 0.5,
    "sd_weight": (0.6, 0.4, 0.6, 0.8),  # trust in showdown calibration, per street
}


def _raise_errors() -> bool:
    """Dev/tests set AGENT_RAISE_ERRORS=1. The tournament leaves it unset."""
    return os.environ.get("AGENT_RAISE_ERRORS") == "1"


def _record_failure(memory: dict, where: str) -> None:
    """Count a failure, then re-raise in dev or log a traceback on stderr.

    The traceback is not put into the action JSON. ``choose_action`` still
    returns a legal action when this does not re-raise. Nothing from the
    environment is written.
    """
    counts = memory.setdefault("errors", {"choose_action": 0, "on_hand_end": 0})
    counts[where] = int(counts.get(where, 0)) + 1
    if _raise_errors():
        raise
    print(f"agent {where} failed", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)


def _equity_when_called(e: float, f: float, street: int) -> float:
    """Our equity against the part of the range that continues after a bet."""
    f = min(f, 0.9)
    e_fold = e + _P["fold_k"][street] * (1.0 - e)
    return min(1.0, max(0.0, (e - f * e_fold) / (1.0 - f)))


def _decide(obs: Observation, memory: dict) -> Action:
    cur = memory["cur"]
    st = memory["stats"]
    rng = memory["rng"]
    street = int(obs.street)
    if cur["hand_index"] != obs.hand_index:
        cur.clear()
        cur.update({"hand_index": obs.hand_index, "hist_n": 0, "pools": {}, "my_last": None})
        _new_hand_accounting(memory)
    if _FIXED_SCALE is None and _spent(memory) > 0.96 * memory["bank"]:
        # Still parse history. Skipping _replay and jumping hist_n drops
        # opponent actions forever, because later calls only count i >= hist_n.
        acts = _replay(obs, memory)
        best = _cheap_decide(obs)
        if best not in obs.legal_actions:
            best = Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
        cur["my_last"] = (street, best, obs.to_call, obs.is_button, "s")
        cur["acts"] = acts
        cur["hole"] = obs.hole_cards
        return best
    acts = _replay(obs, memory)

    # --- equity against the opponent's (modelled) range ---
    scale = _scale(memory)
    stack0 = memory["stack0"]
    big = obs.to_call >= 0.15 * stack0 or obs.pot >= 0.4 * stack0   # rare, and they decide the match
    pool = cur["pools"].get(street)
    if pool is None:
        pool = _make_pool(rng, obs.hole_cards, obs.board, int(_BASE_M[street] * scale))
        cur["pools"][street] = pool
    w = _range_weights(pool, acts, st, street)
    k = int((_BIG_K if big else _BASE_K) * scale)
    # Narrow range: grow the pool until enough distinct holdings fit the opponent's actions.
    want = int((1.5 if big else 0.6) * k)
    max_m = int((2400 if big else 1000) * min(1.0, scale))
    while pool["m"] < max_m and _time_ok(memory, 0.8):
        n_eff = w.sum() ** 2 / max(1e-9, (w * w).sum())
        if n_eff >= want:
            break
        _extend_pool(rng, pool, obs.hole_cards, min(pool["m"], max_m - pool["m"]))
        w = _range_weights(pool, acts, st, street)
    equity, ess = _range_equity(memory, pool, w, k)

    # --- expected values of the legal actions ---
    legal = obs.legal_actions
    pot, to_call = obs.pot, obs.to_call
    if street == 0:
        g_raise = _f_pre(st, "pre_re")
    else:
        g_raise = _f_post(st, "raise", street)
    p = _P
    e_adj = equity - p["call_margin"][street]
    if not obs.is_button and street < 3:
        e_adj -= p["oop_pen"]                           # out of position: equity realises worse
    realise = p["realise"][street]                      # called bets on early streets
    evs = {}
    if Action.FOLD in legal:
        evs[Action.FOLD] = 0.0
    if to_call == 0:
        evs[Action.CHECK_CALL] = e_adj * pot
    else:
        c = min(to_call, obs.my_stack)
        evs[Action.CHECK_CALL] = e_adj * (pot + c) - c
    fas, sizes = {}, {}
    for a in (Action.RAISE_HALF, Action.RAISE_POT):
        if a not in legal:
            continue
        r = obs.raise_to[a] - obs.my_bet                # chips we add
        size = (r - to_call) / max(1.0, pot + to_call)  # raise size relative to the pot
        sizes[a] = _size_class(r - to_call, pot + to_call)
        if street == 0:
            fa = _f_pre(st, "pre_fold") * (p["size_f"][0] + p["size_f"][1] * min(1.0, size))
        else:
            fa = _f_post(st, "fold_" + sizes[a], street)
        fas[a] = fa = min(0.95, fa)
        ec = max(0.0, _equity_when_called(equity, fa, street) - realise)
        call_amt = min(r - to_call, obs.opp_stack)
        pot_after = pot + r + call_amt
        ev = fa * pot + (1.0 - fa) * (ec * pot_after - r)
        ev += (1.0 - fa) * g_raise * (ec - 0.5) * p["reraise_coef"] * r   # re-raises help only when strong
        ev -= p["raise_margin"] * pot                                      # prefer the passive line on ties
        evs[a] = ev
    noise = p["noise"] * pot
    best = max(evs, key=lambda k: evs[k] + rng.uniform(-noise, noise))

    # --- strategic guards ---
    aggressive = best in (Action.RAISE_HALF, Action.RAISE_POT)
    if aggressive and equity < p["bluff_lo"]:
        bluff_rate = min(p["bluff_max"], max(0.0, (fas[best] - p["bluff_f0"]) * p["bluff_slope"]))
        if rng.random() > bluff_rate:
            passive = {k: v for k, v in evs.items() if k in (Action.FOLD, Action.CHECK_CALL)}
            best = max(passive, key=passive.get)
    first_in = street == 0 and obs.is_button and not obs.action_history
    if first_in and best == Action.CHECK_CALL and equity < p["junk"]:
        best = Action.FOLD                              # junk: don't even limp
    if best not in legal:
        best = Action.CHECK_CALL if to_call == 0 else Action.FOLD
    if _TRACE is not None:                              # debugging aid for local analysis only
        _TRACE.append((obs.hand_index, street, round(equity, 3), int(ess), pool["m"],
                       {a.name: round(v, 1) for a, v in evs.items()}, best.name, pot, to_call,
                       [k for _, k, _ in acts]))
    cur["my_last"] = (street, best, to_call, obs.is_button, sizes.get(best, "s"))
    cur["acts"] = acts
    cur["hole"] = obs.hole_cards
    cur["hist_n"] = len(obs.action_history)
    return best


def new_half(config: AgentConfig, memory: dict) -> None:
    """Called at the start of each half-match (1000 hands).

    `memory` is a fresh, empty dict every half. Store anything you want to
    remember between hands in it, e.g. memory["opp_raises"] = 0.
    `config` holds the match settings (starting_stack, big_blind, ...).

    In the tournament, each half runs in a brand-new copy of your server,
    so nothing you learn in one half carries over to the other half.
    """
    memory["hands"] = 0
    memory["calls"] = 0
    memory["time_used"] = 0.0
    memory["scale"] = 1.0
    memory["load"] = 0.0
    memory["draws"] = {}
    memory["drawn"] = {}
    memory["hand_cost"] = 0.0
    memory["hand_mark"] = 0.0
    memory["bank"] = float(config.time_bank_seconds)
    memory["num_hands"] = int(config.num_hands)
    memory["bb"] = int(config.big_blind)
    memory["sb"] = int(config.small_blind)
    memory["stack0"] = int(config.starting_stack)
    memory["raise_cap"] = int(config.raise_cap)
    memory["rng"] = np.random.default_rng()
    memory["stats"] = _fresh_stats()
    memory["cur"] = {"hand_index": -1, "hist_n": 0, "pools": {}, "my_last": None}
    memory["errors"] = {"choose_action": 0, "on_hand_end": 0}


def choose_action(obs: Observation, memory: dict) -> Action:
    """Called every time it is your turn. Return one of obs.legal_actions.

    What you can see in obs:
        hand_index          which hand of the half this is (0, 1, 2, ...)
        street              Street.PREFLOP, Street.FLOP, Street.TURN or Street.RIVER
        hole_cards          your 5 cards as ints, sorted ascending (0-35)
        board               board (community) cards seen so far (0, 3, 4 or 5 ints)
        pot                 total chips both players put in this hand
        to_call             chips you must add to match the opponent
        my_bet, opp_bet     chips you,opponent put in on THIS street
        my_stack, opp_stack chips each player still has behind
        is_button           True if you are the button (small blind)
        legal_actions       the Actions you may return right now
        raise_to            {RAISE_HALF: total, RAISE_POT: total} for legal raises
        min_raise_to        smallest normal raise total (all-ins may be lower)
        raises_this_street  raises so far this street (max 4)
        action_history      this hand's actions so far; each has
                            street, is_me, action, amount, street_total
        big_blind           size of the big blind in chips

    Actions: Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT
    Cards are ints: card_rank(c) is 0 (six) .. 8 (ace), card_suit(c) is 0..3,
    and cards_to_str(obs.hole_cards) prints them like "6h 7c 9s Kd Ah".
    Once all 5 board cards are out, omaha_score(obs.hole_cards, obs.board)
    scores your hand (higher is better).
    """
    t0 = time.perf_counter()
    try:
        action = _decide(obs, memory)
    except Exception:
        action = None
        _record_failure(memory, "choose_action")
    finally:
        memory["time_used"] = memory.get("time_used", 0.0) + (time.perf_counter() - t0)
        memory["calls"] = memory.get("calls", 0) + 1
    if action not in obs.legal_actions:
        action = Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Called once for every finished hand. Optional: learn from what happened.

    To save time, the judge sends finished hands along with your next
    decision request, so this always runs BEFORE choose_action() sees the
    next hand. (The last hands of a half arrive at the end of the half.)

    result has: hand_index, my_net (chips won or lost), pot,
    went_to_showdown, board (cards revealed), opp_hole (opponent's cards,
    only after a showdown, else None), winner ('me', 'opp' or 'split').
    """
    t0 = time.perf_counter()
    try:
        memory["hands"] = memory.get("hands", 0) + 1
        cur = memory["cur"]
        st = memory["stats"]
        _decay_stats(st, _P["decay"])
        # The opponent's last action of a hand never shows up in an
        # observation (the hand is over), so infer it from the result.
        if cur.get("hand_index") == result.hand_index and cur.get("my_last") is not None:
            street, action, to_call_then, was_button, size = cur["my_last"]
            acts = list(cur.get("acts", []))
            my_aggr = action in (Action.RAISE_HALF, Action.RAISE_POT)
            if not result.went_to_showdown:
                if result.winner == "me" and my_aggr:
                    _count(st, street, "fold", size)    # opponent folded to my bet
            elif my_aggr:
                # Postflop "call" updates fold/raise tables that have no street 0.
                # A preflop all-in that is called uses the preflop call counters.
                kind = "call_raise" if street == 0 else "call"
                _count(st, street, kind, size)
                acts.append((street, kind, size))
            elif action == Action.CHECK_CALL and to_call_then == 0 and street == 3 and not was_button:
                _count(st, street, "check")             # opponent checked back the river
            if result.went_to_showdown and result.opp_hole is not None and _P["sd_weight"][3] > 0:
                _calibrate(memory, result, acts, cur.get("hole", ()))
        elif cur.get("hand_index") != result.hand_index:
            if result.winner == "me" and not result.went_to_showdown:
                _count(st, 0, "open_pass")              # opponent open-folded the button
    except Exception:
        _record_failure(memory, "on_hand_end")
    finally:
        memory["time_used"] = memory.get("time_used", 0.0) + (time.perf_counter() - t0)

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
