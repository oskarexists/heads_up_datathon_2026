"""Review regressions. These check the engine rules and lock observed bot defects.

Run from the repo root: python -m pytest -q test_review.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

import agent
from evaluator import (
    FLUSH,
    FULL_HOUSE,
    STRAIGHT,
    THREE_OF_A_KIND,
    cards_from_str,
    category_of,
    omaha_score,
    score_five,
)
from game import Action, AgentConfig, Deal, Game, button_for_deal, make_deal
from review_adversaries import ADVERSARIES

# Rank bits: 0=6 ... 8=ace. Straight windows, wheel last (A-6-7-8-9).
_WINDOWS = [set(range(lo, lo + 5)) for lo in range(5)] + [{8, 0, 1, 2, 3}]


def _card(rank: int, suit: int) -> int:
    return (rank << 2) | suit


def test_short_deck_ranking_rules():
    flush = score_five(cards_from_str("Ah Kh Qh Jh 9h"))
    full = score_five(cards_from_str("As Ad Ac Ks Kd"))
    trips = score_five(cards_from_str("As Ad Ac Ks Qd"))
    straight = score_five(cards_from_str("Ts 9d 8c 7h 6s"))
    wheel = score_five(cards_from_str("As 9d 8c 7h 6s"))
    six_high = score_five(cards_from_str("Ts 9d 8c 7h 6c"))
    assert category_of(flush) == FLUSH
    assert category_of(full) == FULL_HOUSE
    assert category_of(trips) == THREE_OF_A_KIND
    assert category_of(straight) == STRAIGHT
    assert flush > full
    assert trips > straight
    assert category_of(wheel) == STRAIGHT
    assert wheel < six_high


def test_agent_rank_table_matches_omaha_score_order():
    rng = np.random.default_rng(7)
    for _ in range(40):
        cards = rng.choice(36, size=20, replace=False)
        hole, board = cards[:5], cards[5:10]
        hole2, board2 = cards[10:15], cards[15:20]
        a = int(agent._best(agent._pair_idx(np.array(hole)), agent._triple_idx(np.array(board)[None, :]))[0])
        b = int(agent._best(agent._pair_idx(np.array(hole2)), agent._triple_idx(np.array(board2)[None, :]))[0])
        ra, rb = omaha_score(hole.tolist(), board.tolist()), omaha_score(hole2.tolist(), board2.tolist())
        assert (a > b) == (ra > rb)
        assert (a == b) == (ra == rb)


def test_replay_pot_and_to_call_match_the_engine():
    """The bot's blind-aware history walk matches Game at each decision for seat 0."""
    for hand in range(25):
        game = Game(make_deal("review-replay", hand), button_for_deal(hand), hand)
        while not game.is_terminal:
            obs = game.observation(game.to_act)
            if game.to_act == 0:
                bb, sb = 2, 1
                if obs.is_button:
                    tot, put = [sb, bb], [sb, bb]
                else:
                    tot, put = [bb, sb], [bb, sb]
                street = 0
                for rec in obs.action_history:
                    if int(rec.street) != street:
                        street = int(rec.street)
                        tot = [0, 0]
                    actor = 0 if rec.is_me else 1
                    tot[actor] = rec.street_total
                    put[actor] += rec.amount
                assert tot[1] - tot[0] == obs.to_call
                assert put[0] + put[1] == obs.pot
            legal = obs.legal_actions
            if hand % 3 == 0 and Action.RAISE_POT in legal:
                action = Action.RAISE_POT
            elif hand % 3 == 1 and Action.RAISE_HALF in legal:
                action = Action.RAISE_HALF
            elif Action.FOLD in legal and hand % 5 == 0:
                action = Action.FOLD
            else:
                action = Action.CHECK_CALL
            game.apply(action)


def _real_straight_draw(hole: np.ndarray, board: np.ndarray) -> bool:
    hole_ranks = [int(c) >> 2 for c in hole.tolist()]
    board_ranks = [int(c) >> 2 for c in board.tolist()]
    distinct = sorted(set(hole_ranks))
    for i, rank_a in enumerate(distinct):
        for rank_b in distinct[i + 1:]:
            for window in _WINDOWS:
                if rank_a in window and rank_b in window:
                    union = {rank_a, rank_b} | {r for r in board_ranks if r in window}
                    if len(union) >= 4:
                        return True
    return False


def _draw_bonus(hole: np.ndarray, board: np.ndarray) -> float:
    """Flop/turn draw bonus in category units (strength minus made-hand value)."""
    row = hole[None, :]
    made = float(agent._VAL[agent._best(agent._pair_idx(hole), agent._triple_idx(board[None, :]))[0]])
    strength = float(agent._board_strength(agent._pair_idx(row), agent._hand_idx(row), board)[0])
    return strength - made


def test_draw_bonus_uses_two_hole_cards_and_no_double_count():
    """Flush and straight bonuses follow exactly-two-hole Omaha draws.

    Made flushes and backdoors get no flush-draw bonus. A rank on both the
    holding and the board counts once. On the flop the scale is 1, so a pure
    flush draw adds 1.3 and a pure straight draw adds 0.8.
    """
    # Ah Ac Ad Kd Qd on 9h 8h 7h: one heart in the hole, so no flush draw.
    one_heart = np.array([_card(8, 2), _card(8, 0), _card(8, 1), _card(7, 1), _card(6, 1)])
    monotone = np.array([_card(3, 2), _card(2, 2), _card(1, 2)])
    assert _real_straight_draw(one_heart, monotone) is False
    assert abs(_draw_bonus(one_heart, monotone)) < 1e-9

    # Qh Jh Qc Jd Qd. Two hearts, ranks Q/J only.
    flush_hole = np.array([_card(6, 2), _card(5, 2), _card(6, 0), _card(5, 1), _card(6, 1)])
    # Kh Kd 6h: exactly two hearts. T-A union is Q,J,K (3), so no straight draw.
    two_hearts = np.array([_card(7, 2), _card(7, 1), _card(0, 2)])
    assert _real_straight_draw(flush_hole, two_hearts) is False
    assert abs(_draw_bonus(flush_hole, two_hearts) - 1.3) < 1e-9

    # Kh 6h 9h is a made flush. The straight-draw gate is off once a straight is made.
    made_flush_board = np.array([_card(7, 2), _card(0, 2), _card(3, 2)])
    assert abs(_draw_bonus(flush_hole, made_flush_board)) < 1e-9

    # Kh 6c 7d: one heart (backdoor) and no four-rank window.
    backdoor_board = np.array([_card(7, 2), _card(0, 0), _card(1, 1)])
    assert _real_straight_draw(flush_hole, backdoor_board) is False
    assert abs(_draw_bonus(flush_hole, backdoor_board)) < 1e-9

    # 9c 8d Ac Ad Kc on 9h 8c Kd. Shared 9 and 8 used to count twice
    # (hole 2 + board 2). The union in 8-Q is {8,9}, which is not a draw.
    overlap = np.array([_card(3, 0), _card(2, 1), _card(8, 0), _card(8, 1), _card(7, 0)])
    overlap_board = np.array([_card(3, 2), _card(2, 0), _card(7, 1)])
    assert _real_straight_draw(overlap, overlap_board) is False
    assert abs(_draw_bonus(overlap, overlap_board)) < 1e-9

    # Tc Jc Ad Kd 6s on 9h 8d 8c. 8-Q holds {8,9,T,J}: two hole ranks, four
    # distinct, and the paired board is not yet a straight.
    straight_hole = np.array([_card(4, 0), _card(5, 0), _card(8, 1), _card(7, 1), _card(0, 3)])
    straight_board = np.array([_card(3, 2), _card(2, 1), _card(2, 0)])
    assert _real_straight_draw(straight_hole, straight_board) is True
    assert abs(_draw_bonus(straight_hole, straight_board) - 0.8) < 1e-9


def test_extended_pool_percentiles_are_ranks_in_the_full_sample():
    """Street percentiles are recomputed on the combined pool. Preflop stays global."""
    rng = np.random.default_rng(0)
    hole = [0, 1, 2, 3, 4]
    board = [8, 9, 10]
    pool = agent._make_pool(rng, hole, board, 30)
    pre = pool["pct"][0].copy()
    agent._extend_pool(rng, pool, hole, 30)
    assert pool["m"] == 60
    assert 1 not in pool["pct"]
    assert np.allclose(pool["pct"][0][:30], pre)
    assert np.allclose(pool["pct"][0], agent._PRE_PCT[pool["hidx"]])
    ranked = agent._street_pct(pool, 1)
    strength = agent._board_strength(pool["pidx"], pool["hidx"], pool["board"][:3])
    assert np.allclose(ranked, agent._percentile(strength))


def test_decide_exception_does_not_escape_choose_action():
    memory: dict = {}
    agent.new_half(AgentConfig(), memory)
    obs = Game(make_deal(1, 0), 0).observation(0)
    real = agent._decide

    def broken(obs, memory):
        raise RuntimeError("review probe")

    agent._decide = broken
    try:
        action = agent.choose_action(obs, memory)
    finally:
        agent._decide = real
    assert action in obs.legal_actions
    assert action == (Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD)


def test_river_nuts_are_not_folded():
    hero = tuple(sorted(cards_from_str("As Ks 9c 8d 7c")))
    villain = tuple(sorted(cards_from_str("Ad Kd Qc Jh Th")))
    board = tuple(cards_from_str("Qs Js Ts 6h 6d"))
    assert len(set(hero + villain + board)) == 15
    game = Game(Deal(hole_cards=(hero, villain), board=board), button_seat=1, hand_index=3)
    # Button limps, BB checks, then both check until the river. The button then bets.
    while not game.is_terminal and game.street < 3:
        game.apply(Action.CHECK_CALL)
    assert game.street == 3
    # Non-button (hero, seat 0) acts first. Check, then let the button bet.
    assert game.to_act == 0
    game.apply(Action.CHECK_CALL)
    assert Action.RAISE_POT in game.legal_actions()
    game.apply(Action.RAISE_POT)
    obs = game.observation(0)
    assert obs.to_call > 0 and Action.FOLD in obs.legal_actions
    memory: dict = {}
    agent.new_half(AgentConfig(), memory)
    memory["rng"] = np.random.default_rng(123)
    action = agent.choose_action(obs, memory)
    assert action in obs.legal_actions
    assert action != Action.FOLD


def test_adversaries_stay_legal_for_fifty_hands():
    for name, cls in ADVERSARIES.items():
        bot = cls()
        memory: dict = {}
        bot.new_half(AgentConfig(), memory)
        for hand in range(50):
            game = Game(make_deal(f"adv-{name}", hand), button_for_deal(hand), hand)
            while not game.is_terminal:
                obs = game.observation(game.to_act)
                action = bot.choose_action(obs, memory)
                assert action in obs.legal_actions, name
                game.apply(action)
            bot.on_hand_end(game.hand_result(0), memory)


def test_mixed_actions_do_not_throw_inside_decide():
    memory: dict = {}
    agent.new_half(AgentConfig(num_hands=20), memory)
    memory["rng"] = np.random.default_rng(4)
    for hand in range(20):
        game = Game(make_deal("review-gauntlet", hand), button_for_deal(hand), hand)
        while not game.is_terminal:
            obs = game.observation(game.to_act)
            if game.to_act == 0:
                action = agent._decide(obs, memory)
                assert action in obs.legal_actions
            else:
                action = Action.RAISE_POT if Action.RAISE_POT in obs.legal_actions and hand % 2 == 0 else Action.CHECK_CALL
            game.apply(action)
        agent.on_hand_end(game.hand_result(0), memory)
