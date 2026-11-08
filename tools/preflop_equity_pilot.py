"""Suit-orbit map and a resumable full preflop equity table.

Equity is versus a uniformly random opposing 5-card hand and a random 5-card
board. Ties count as 0.5. It is a Monte Carlo estimate, not exact equity, and
it is not equity against an adapted range.

Each world draws a fresh permutation of the 31 remaining cards from its own
NumPy Generator. Nothing reads agent._PERM_POOL.

The canonical key of a 5-card holding is the lexicographic minimum of the
sorted card ids over all 24 suit permutations. That is constant on suit
orbits. Burnside's count is (C(36,5) + 6*16560 + 8*450) / 24 = 19998.

    python tools/preflop_equity_pilot.py verify
    python tools/preflop_equity_pilot.py run
"""

from __future__ import annotations

import json
import os
import sys
import time
from itertools import combinations, permutations
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "review_results" / "preflop_equity"
MASTER = 20261107
HOLDINGS = 376992
N_ORBITS = 19998
BURNSIDE_NUM = HOLDINGS + 6 * 16560 + 8 * 450  # divisible by 24
STAGE_N = (1024, 8192, 38416)
BATCH = 8192


def _suit_perms() -> np.ndarray:
    return np.array(list(permutations(range(4))), dtype=np.int8)


SUIT_PERMS = _suit_perms()
_CARDS = np.arange(36, dtype=np.int16)
# CARD_MAP[p, c] is card c after suit permutation p. perm[s] is the new suit.
CARD_MAP = (
    (((_CARDS >> 2) << 2)[None, :] | SUIT_PERMS[:, _CARDS & 3].astype(np.int16)).astype(np.int16)
)
assert CARD_MAP.shape == (24, 36)


def pack5(cards: np.ndarray) -> np.ndarray:
    """Pack sorted rows of card ids into a base-36 integer. Digits are 0..35."""
    c = np.asarray(cards, dtype=np.int32)
    return (((c[..., 0] * 36 + c[..., 1]) * 36 + c[..., 2]) * 36 + c[..., 3]) * 36 + c[..., 4]


def all_combos() -> np.ndarray:
    """Every 5-card holding, combinations() order, shape (376992, 5)."""
    combos = np.array(list(combinations(range(36), 5)), dtype=np.int16)
    if combos.shape != (HOLDINGS, 5):
        raise RuntimeError(f"expected {(HOLDINGS, 5)}, got {combos.shape}")
    return combos


def image_keys(combos: np.ndarray) -> np.ndarray:
    """(N, 5) cards -> (24, N) packed sorted images, one per suit permutation."""
    combos = np.asarray(combos, dtype=np.int16)
    mapped = CARD_MAP[:, combos]
    mapped.sort(axis=-1)
    keys = pack5(mapped.reshape(-1, 5)).reshape(mapped.shape[0], mapped.shape[1])
    return keys


def canon_key(cards) -> tuple[int, ...]:
    """Lex-min sorted image of one holding under the 24 suit permutations."""
    arr = np.asarray([int(c) for c in cards], dtype=np.int16).reshape(1, 5)
    keys = image_keys(arr)
    which = int(np.argmin(keys[:, 0]))
    mapped = np.sort(CARD_MAP[which, arr[0]])
    return tuple(int(c) for c in mapped)


def composition_table() -> np.ndarray:
    """comp[g, p] = h means CARD_MAP[g] ∘ CARD_MAP[p] = CARD_MAP[h] on all 36 cards."""
    comp = np.empty((24, 24), dtype=np.int8)
    for g in range(24):
        for p in range(24):
            composed = CARD_MAP[g, CARD_MAP[p]]
            hit = np.flatnonzero(np.all(CARD_MAP == composed, axis=1))
            if hit.size != 1:
                raise RuntimeError(f"suit maps are not a group at {(g, p)}")
            comp[g, p] = int(hit[0])
    return comp


def build_orbit_map(combos: np.ndarray | None = None):
    """Return combos, per-holding orbit id, per-orbit size, canonical rep, image keys."""
    if combos is None:
        combos = all_combos()
    keys = image_keys(combos)
    orbit_key = keys.min(axis=0)
    uniq, inverse, counts = np.unique(orbit_key, return_inverse=True, return_counts=True)
    which = np.argmin(keys, axis=0)
    images = CARD_MAP[which[:, None], combos]
    images.sort(axis=1)
    # First holding of each orbit id, in np.unique order, is a representative.
    order = np.argsort(inverse, kind="stable")
    inv_sorted = inverse[order]
    starts = np.flatnonzero(np.diff(inv_sorted, prepend=-1))
    reps = images[order[starts]]
    return combos, inverse.astype(np.int32), counts.astype(np.int16), reps.astype(np.int16), keys, orbit_key


def verify_exhaustive(combos=None) -> dict:
    """Fail unless the map is the 19998 suit orbits and every perm preserves ids.

    The check covers all 376992 holdings and all 24 permutations, twice:
    the group action reorders the 24 materialized image keys, and a second
    call recomputes the lex-min on the permuted cards.
    """
    t0 = time.perf_counter()
    if canon_key([0, 1, 4, 8, 12]) != canon_key([1, 0, 5, 9, 13]):
        raise AssertionError("suit swap 0<->1 changed the canonical key")
    if BURNSIDE_NUM != N_ORBITS * 24:
        raise AssertionError(f"Burnside numerator {BURNSIDE_NUM} is not {N_ORBITS}*24")
    combos, inverse, counts, reps, keys, orbit_key = build_orbit_map(combos)
    if int(inverse.max()) + 1 != N_ORBITS or counts.shape != (N_ORBITS,):
        raise AssertionError(f"orbit count {counts.shape[0]} != {N_ORBITS}")
    if int(counts.sum()) != HOLDINGS:
        raise AssertionError("orbit sizes do not cover every holding")
    if not np.all((24 % counts.astype(np.int32)) == 0):
        raise AssertionError("an orbit size does not divide 24")
    # Every member's lex-min image is that orbit's representative.
    which = np.argmin(keys, axis=0)
    images = CARD_MAP[which[:, None], combos]
    images.sort(axis=1)
    if not np.array_equal(images, reps[inverse]):
        raise AssertionError("lex-min image is not constant on an orbit")
    comp = composition_table()
    for p in range(24):
        if not np.array_equal(keys[comp[:, p]].min(axis=0), orbit_key):
            raise AssertionError(f"group reorder changed orbit ids at perm {p}")
        imaged = CARD_MAP[p][combos]
        recomputed = image_keys(imaged).min(axis=0)
        if not np.array_equal(recomputed, orbit_key):
            raise AssertionError(f"direct recompute changed orbit ids at perm {p}")
    # The verified failing pair sits in one orbit.
    pair_rows = []
    for hand in ([0, 1, 4, 8, 12], [1, 0, 5, 9, 13]):
        row = np.flatnonzero(np.all(combos == np.asarray(sorted(hand), dtype=np.int16), axis=1))
        if row.size != 1:
            raise AssertionError(f"holding {hand} missing from the census")
        pair_rows.append(int(inverse[int(row[0])]))
    if pair_rows[0] != pair_rows[1]:
        raise AssertionError("counterexample holdings landed in different orbits")
    seconds = time.perf_counter() - t0
    return {
        "orbits": N_ORBITS,
        "holdings": HOLDINGS,
        "seconds": seconds,
        "min_orbit_size": int(counts.min()),
        "max_orbit_size": int(counts.max()),
        "reps": reps,
        "counts": counts,
        "inverse": inverse,
    }


def remaining_deck(hole: np.ndarray) -> np.ndarray:
    mask = np.ones(36, dtype=bool)
    mask[np.asarray(hole, dtype=np.int64)] = False
    deck = np.flatnonzero(mask).astype(np.int16)
    if deck.shape != (31,):
        raise RuntimeError("a holding must be 5 distinct cards")
    return deck


def sample_deals(deck: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n independent uniform permutations. Columns 0:5 opponent, 5:10 board."""
    work = np.empty((n, deck.shape[0]), dtype=np.int16)
    work[:] = deck
    rng.permuted(work, axis=1, out=work)
    return work[:, :10]


def count_wins_ties(hole: np.ndarray, deals: np.ndarray) -> tuple[int, int]:
    """Vectorized dense-rank comparison. Order must match omaha_score."""
    import agent

    hole64 = np.asarray(hole, dtype=np.int64).reshape(1, 5)
    deals64 = np.asarray(deals, dtype=np.int64)
    hero_p = agent._pair_idx(hole64)
    opp_p = agent._pair_idx(deals64[:, :5])
    tidx = agent._triple_idx(deals64[:, 5:10])
    hero = agent._best(hero_p, tidx)
    opp = agent._best(opp_p, tidx)
    return int(np.count_nonzero(hero > opp)), int(np.count_nonzero(hero == opp))


def sample_counts(hole: np.ndarray, n: int, rng: np.random.Generator, batch: int = BATCH) -> tuple[int, int]:
    deck = remaining_deck(hole)
    wins = ties = 0
    done = 0
    while done < n:
        m = min(batch, n - done)
        deals = sample_deals(deck, m, rng)
        w, t = count_wins_ties(hole, deals)
        wins += w
        ties += t
        done += m
    return wins, ties


def orbit_rng(orbit_id: int, stage: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([MASTER, int(orbit_id), int(stage)]))


def parity_mismatches(n_deals: int = 500, n_holes: int = 4, seed: int = 11) -> tuple[int, int]:
    """Random worlds where dense _best disagrees with evaluator.omaha_score."""
    from evaluator import omaha_score

    rng = np.random.default_rng(seed)
    bad = 0
    checked = 0
    for _ in range(n_holes):
        hole = np.sort(rng.choice(36, size=5, replace=False).astype(np.int16))
        deck = remaining_deck(hole)
        local = np.random.default_rng(int(rng.integers(1, 2**31 - 1)))
        deals = sample_deals(deck, n_deals, local)
        import agent

        hole64 = hole.astype(np.int64).reshape(1, 5)
        deals64 = deals.astype(np.int64)
        hero = agent._best(agent._pair_idx(hole64), agent._triple_idx(deals64[:, 5:10]))
        opp = agent._best(agent._pair_idx(deals64[:, :5]), agent._triple_idx(deals64[:, 5:10]))
        hole_t = tuple(int(c) for c in hole.tolist())
        for i in range(n_deals):
            board = tuple(int(c) for c in deals[i, 5:10].tolist())
            other = tuple(int(c) for c in deals[i, :5].tolist())
            hs = omaha_score(hole_t, board)
            os_ = omaha_score(other, board)
            if hs > os_:
                exp = 1
            elif hs < os_:
                exp = -1
            else:
                exp = 0
            got = 1 if hero[i] > opp[i] else (-1 if hero[i] < opp[i] else 0)
            bad += int(got != exp)
            checked += 1
    return bad, checked


def _atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_slice(orbit_ids, reps, n: int, stage: int, ckpt: Path, batch: int = BATCH, max_new: int | None = None) -> dict:
    """Accumulate wins/ties for these orbits. Resume skips rows already done."""
    ids = np.asarray(orbit_ids, dtype=np.int32)
    reps = np.asarray(reps, dtype=np.int16)
    rows = ids.shape[0]
    wins = np.zeros(rows, dtype=np.int32)
    ties = np.zeros(rows, dtype=np.int32)
    done = np.zeros(rows, dtype=np.bool_)
    if ckpt.exists():
        prev = np.load(ckpt)
        if (
            np.array_equal(prev["orbit_id"], ids)
            and int(prev["n"]) == int(n)
            and int(prev["stage"]) == int(stage)
        ):
            wins = prev["wins"].astype(np.int32, copy=False)
            ties = prev["ties"].astype(np.int32, copy=False)
            done = prev["done"].astype(np.bool_, copy=False)
    fresh = 0
    t0 = time.perf_counter()
    for i in range(rows):
        if done[i]:
            continue
        if max_new is not None and fresh >= max_new:
            break
        w, t = sample_counts(reps[i], n, orbit_rng(int(ids[i]), stage), batch)
        wins[i] = w
        ties[i] = t
        done[i] = True
        fresh += 1
        if fresh % 20 == 0:
            _atomic_savez(ckpt, orbit_id=ids, wins=wins, ties=ties, done=done, n=np.int32(n), stage=np.int32(stage))
    _atomic_savez(ckpt, orbit_id=ids, wins=wins, ties=ties, done=done, n=np.int32(n), stage=np.int32(stage))
    return {
        "done": int(done.sum()),
        "rows": int(rows),
        "fresh": fresh,
        "seconds": time.perf_counter() - t0,
        "wins": wins,
        "ties": ties,
        "finished": bool(done.all()),
    }


def _worker(payload: dict) -> None:
    ckpt = Path(payload["ckpt"])
    prog = Path(payload["progress"])
    ids = np.asarray(payload["orbit_ids"], dtype=np.int32)
    reps = np.asarray(payload["reps"], dtype=np.int16)
    t0 = time.perf_counter()
    # Save often enough that a killed worker loses at most 20 orbits.
    rows = ids.shape[0]
    wins = np.zeros(rows, dtype=np.int32)
    ties = np.zeros(rows, dtype=np.int32)
    done = np.zeros(rows, dtype=np.bool_)
    n = int(payload["n"])
    stage = int(payload["stage"])
    if ckpt.exists():
        prev = np.load(ckpt)
        if np.array_equal(prev["orbit_id"], ids) and int(prev["n"]) == n and int(prev["stage"]) == stage:
            wins = prev["wins"].astype(np.int32, copy=True)
            ties = prev["ties"].astype(np.int32, copy=True)
            done = prev["done"].astype(np.bool_, copy=True)
    fresh = 0
    for i in range(rows):
        if done[i]:
            continue
        w, t = sample_counts(reps[i], n, orbit_rng(int(ids[i]), stage), int(payload["batch"]))
        wins[i] = w
        ties[i] = t
        done[i] = True
        fresh += 1
        if fresh % 20 == 0:
            _atomic_savez(ckpt, orbit_id=ids, wins=wins, ties=ties, done=done, n=np.int32(n), stage=np.int32(stage))
            _write_json(prog, {
                "pid": os.getpid(),
                "done": int(done.sum()),
                "rows": rows,
                "seconds": time.perf_counter() - t0,
                "samples": int(done.sum()) * n,
            })
    _atomic_savez(ckpt, orbit_id=ids, wins=wins, ties=ties, done=done, n=np.int32(n), stage=np.int32(stage))
    _write_json(prog, {
        "pid": os.getpid(),
        "done": int(done.sum()),
        "rows": rows,
        "seconds": time.perf_counter() - t0,
        "samples": int(done.sum()) * n,
        "finished": True,
    })


def stage_increment(stage: int) -> int:
    """New worlds per orbit in this stage. Stages partition the table; they do not overlap."""
    if stage < 0 or stage >= len(STAGE_N):
        raise IndexError(f"stage {stage}")
    prev = 0 if stage == 0 else int(STAGE_N[stage - 1])
    inc = int(STAGE_N[stage]) - prev
    if inc <= 0:
        raise RuntimeError(f"stage {stage} increment {inc}")
    return inc


def unique_completed_worlds(stage: int, orbits_done: int, *, n_this: int) -> int:
    """Count each finished world once.

    Earlier stages are complete before this stage starts, so they contribute
    N_ORBITS times their own increment. This stage contributes one increment
    per finished orbit.

    ``n_this`` is that increment. The finished snapshot passes ``n_this=0``
    because the stage target is already the cumulative ``n_cum``. Those worlds
    are still only this stage's increment, taken from the schedule. Using
    ``N_ORBITS * (n_cum - 0)`` and then adding the worker samples counts the
    last stage twice (1372662720 worlds, 178.6755%, instead of 768243168).
    """
    if stage < 0 or stage >= len(STAGE_N):
        raise ValueError(f"stage {stage}")
    if orbits_done < 0 or orbits_done > N_ORBITS:
        raise ValueError(f"orbits_done {orbits_done}")
    scheduled = stage_increment(stage)
    if int(n_this) not in (0, scheduled):
        raise ValueError(f"stage {stage} increment {n_this} is not 0 or {scheduled}")
    prior = sum(N_ORBITS * stage_increment(i) for i in range(stage))
    return prior + int(orbits_done) * scheduled


def _aggregate_progress(workers: int, stage: int, n_this: int, n_cum: int, t0: float, extra: dict) -> dict:
    done = 0
    samples_stage = 0
    alive = 0
    for wid in range(workers):
        path = OUT / "chunks" / f"s{stage}" / f"w{wid:02d}.progress.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        done += int(payload.get("done", 0))
        samples_stage += int(payload.get("samples", 0))
        alive += 1
    elapsed = max(1e-6, time.perf_counter() - t0)
    samples_done = unique_completed_worlds(stage, done, n_this=n_this)
    scheduled = stage_increment(stage)
    expected_stage = done * scheduled
    full = N_ORBITS * STAGE_N[-1]
    rate = samples_done / max(1e-6, time.time() - extra["started_unix"]) if samples_done else samples_stage / elapsed
    remaining = max(0, full - samples_done)
    payload = {
        "pid": os.getpid(),
        "stage_index": stage,
        "stage_samples_target": n_cum,
        "stage_increment": n_this,
        "orbits_done_this_stage": done,
        "orbits_total": N_ORBITS,
        "workers_reporting": alive,
        "workers": workers,
        "samples_done": samples_done,
        "samples_stage_reported": samples_stage,
        "samples_stage_unique": expected_stage,
        "samples_target_full": full,
        "percent_full": 100.0 * samples_done / full,
        "worlds_definition": "finished orbits times that stage's increment, summed across stages; stages do not overlap",
        "percent_stage": 100.0 * done / N_ORBITS,
        "samples_per_second": rate,
        "eta_seconds_full": remaining / rate if rate > 0 else None,
        "elapsed_seconds": time.time() - extra["started_unix"],
        "updated_unix": time.time(),
        **extra,
    }
    _write_json(OUT / "progress.json", payload)
    return payload


def run_table(workers: int | None = None, batch: int = BATCH) -> None:
    import multiprocessing as mp

    workers = int(os.cpu_count() if workers is None else workers)
    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "run.log"
    started = time.time()
    info = verify_exhaustive()
    reps = info["reps"]
    _atomic_savez(
        OUT / "orbit_map.npz",
        reps=reps,
        counts=info["counts"],
        holding_orbit=info["inverse"],
    )
    bad, checked = parity_mismatches()
    if bad:
        raise SystemExit(f"FAILED parity mismatches {bad} of {checked}")
    # One-core rate, discarded from the table. Same sampler the workers use.
    bench_n = 1024
    bench_orbits = 4
    t1 = time.perf_counter()
    for i in range(bench_orbits):
        sample_counts(reps[i], bench_n, orbit_rng(10_000_000 + i, 99), batch)
    per_s = (bench_orbits * bench_n) / (time.perf_counter() - t1)
    extra = {
        "started_unix": started,
        "parity_checked": checked,
        "parity_mismatches": bad,
        "one_core_samples_per_second": per_s,
        "orbits": N_ORBITS,
        "canon_seconds": info["seconds"],
        "status": "running",
    }
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(
            f"verify orbits {N_ORBITS} in {info['seconds']:.2f}s "
            f"parity {bad}/{checked} one_core_samples_per_s {per_s:.1f} workers {workers}\n"
        )
        log.flush()
    ctx = mp.get_context("fork")
    cumulative = 0
    for stage, target in enumerate(STAGE_N):
        increment = target - cumulative
        (OUT / "chunks" / f"s{stage}").mkdir(parents=True, exist_ok=True)
        edges = np.linspace(0, N_ORBITS, workers + 1, dtype=np.int64)
        procs = []
        for wid in range(workers):
            lo, hi = int(edges[wid]), int(edges[wid + 1])
            payload = {
                "orbit_ids": np.arange(lo, hi, dtype=np.int32),
                "reps": np.ascontiguousarray(reps[lo:hi]),
                "n": increment,
                "stage": stage,
                "batch": batch,
                "ckpt": str(OUT / "chunks" / f"s{stage}" / f"w{wid:02d}.npz"),
                "progress": str(OUT / "chunks" / f"s{stage}" / f"w{wid:02d}.progress.json"),
            }
            proc = ctx.Process(target=_worker, args=(payload,), name=f"eq-s{stage}-w{wid}")
            proc.start()
            procs.append(proc)
        t_stage = time.perf_counter()
        while any(p.is_alive() for p in procs):
            snap = _aggregate_progress(workers, stage, increment, target, t_stage, extra)
            time.sleep(2)
            # Keep the last snap.
            _ = snap
        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                extra["status"] = "failed"
                _aggregate_progress(workers, stage, increment, target, t_stage, extra)
                raise SystemExit(f"FAILED worker exit {proc.exitcode} stage {stage}")
        wins = np.zeros(N_ORBITS, dtype=np.int64)
        ties = np.zeros(N_ORBITS, dtype=np.int64)
        for wid in range(workers):
            z = np.load(OUT / "chunks" / f"s{stage}" / f"w{wid:02d}.npz")
            if not np.all(z["done"]):
                raise SystemExit(f"FAILED incomplete chunk s{stage} w{wid}")
            if int(z["n"]) != increment:
                raise SystemExit(f"FAILED chunk n {int(z['n'])} != {increment}")
            ids = z["orbit_id"]
            wins[ids] = z["wins"]
            ties[ids] = z["ties"]
        _atomic_savez(
            OUT / f"stage_{target}.npz",
            wins=wins.astype(np.int32),
            ties=ties.astype(np.int32),
            n=np.int32(increment),
            stage=np.int32(stage),
        )
        cumulative = target
        line = (
            f"STAGE_DONE target {target} increment {increment} "
            f"seconds {time.perf_counter() - t_stage:.1f}\n"
        )
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(line)
            log.flush()
        print(line, flush=True)
    # Merge stages into one accumulator per orbit.
    total_wins = np.zeros(N_ORBITS, dtype=np.int64)
    total_ties = np.zeros(N_ORBITS, dtype=np.int64)
    total_n = 0
    for target in STAGE_N:
        z = np.load(OUT / f"stage_{target}.npz")
        total_wins += z["wins"]
        total_ties += z["ties"]
        total_n += int(z["n"])
    if total_n != STAGE_N[-1]:
        raise SystemExit(f"FAILED merged n {total_n}")
    losses = total_n - total_wins - total_ties
    if np.any(losses < 0):
        raise SystemExit("FAILED negative losses")
    mean = (total_wins + 0.5 * total_ties) / total_n
    _atomic_savez(
        OUT / "equity_table.npz",
        reps=reps,
        counts=info["counts"],
        wins=total_wins.astype(np.int32),
        ties=total_ties.astype(np.int32),
        losses=losses.astype(np.int32),
        n=np.int32(total_n),
        mean=mean.astype(np.float64),
    )
    extra["status"] = "done"
    extra["merged_n"] = total_n
    # n_this=0 is the finished snapshot (cumulative target, no new increment
    # argument). unique_completed_worlds still counts the last stage once.
    final = _aggregate_progress(workers, len(STAGE_N) - 1, 0, total_n, time.perf_counter(), extra)
    if final["samples_done"] != N_ORBITS * total_n:
        raise SystemExit(
            f"FAILED unique worlds {final['samples_done']} != {N_ORBITS * total_n}"
        )
    print(f"DONE orbits {N_ORBITS} n {total_n} worlds {final['samples_done']}", flush=True)


def completed_stage_counts() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sum finished stage files. Each file stores that stage's increment only."""
    wins = np.zeros(N_ORBITS, dtype=np.int64)
    ties = np.zeros(N_ORBITS, dtype=np.int64)
    n_each = np.zeros(N_ORBITS, dtype=np.int64)
    reps = np.load(OUT / "orbit_map.npz")["reps"]
    merged = OUT / "equity_table.npz"
    if merged.exists():
        z = np.load(merged)
        return (
            z["wins"].astype(np.int64),
            z["ties"].astype(np.int64),
            np.full(N_ORBITS, int(z["n"]), dtype=np.int64),
            z["reps"].astype(np.int16),
        )
    paths = sorted(OUT.glob("stage_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        raise RuntimeError("no completed equity stage")
    for path in paths:
        z = np.load(path)
        wins += z["wins"].astype(np.int64)
        ties += z["ties"].astype(np.int64)
        n_each += int(z["n"])
        if np.any(z["wins"] + z["ties"] > int(z["n"])):
            raise RuntimeError(f"wins+ties exceed n in {path.name}")
    return wins, ties, n_each, reps


def tie_average_percentile(values: np.ndarray) -> np.ndarray:
    """Percentile in [0, 1). Equal values share the average rank. Higher is better."""
    values = np.asarray(values, dtype=np.float64)
    return weighted_tie_percentile(values, np.ones(values.shape[0], dtype=np.float64))


def weighted_tie_percentile(means: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Multiplicity-weighted percentile of each orbit mean. Higher is better.

    A tie is a group of equal means. Every orbit in that group gets the average
    of the hand-ranks the group's hands occupy, so the result does not depend
    on argsort stability or on suit order. ``counts`` are orbit sizes.
    """
    means = np.asarray(means, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if means.shape != counts.shape or means.ndim != 1:
        raise ValueError("means and counts must be aligned 1-d arrays")
    if np.any(counts < 0):
        raise ValueError("negative multiplicity")
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("empty population")
    order = np.argsort(means, kind="mergesort")
    sorted_means = means[order]
    sorted_counts = counts[order]
    pct = np.empty(means.shape[0], dtype=np.float64)
    start = 0
    weight_before = 0.0
    n = int(means.shape[0])
    while start < n:
        end = start + 1
        while end < n and sorted_means[end] == sorted_means[start]:
            end += 1
        weight = float(sorted_counts[start:end].sum())
        avg_rank = weight_before + (weight - 1.0) / 2.0
        pct[order[start:end]] = avg_rank / total
        weight_before += weight
        start = end
    return pct


def equity_summary() -> dict:
    """Coverage and pointwise standard errors for every finished orbit."""
    wins, ties, n_each, _reps = completed_stage_counts()
    if np.any(n_each <= 0):
        raise RuntimeError("an orbit has no finished samples")
    n = n_each.astype(np.float64)
    mean = (wins + 0.5 * ties) / n
    second = (wins + 0.25 * ties) / n
    var = np.maximum(0.0, second - mean * mean)
    # Sample variance with ddof=1 when n>1.
    var_u = var * n / np.maximum(1.0, n - 1.0)
    se = np.sqrt(var_u / n)
    half = 1.96 * se
    return {
        "orbits": int(n_each.size),
        "n_min": int(n_each.min()),
        "n_median": float(np.median(n_each)),
        "n_max": int(n_each.max()),
        "coverage": 1.0,
        "mean_min": float(mean.min()),
        "mean_median": float(np.median(mean)),
        "mean_max": float(mean.max()),
        "se_min": float(se.min()),
        "se_median": float(np.median(se)),
        "se_max": float(se.max()),
        "halfwidth95_min": float(half.min()),
        "halfwidth95_median": float(np.median(half)),
        "halfwidth95_max": float(half.max()),
        "worst_case_halfwidth95": float(1.96 * 0.5 / np.sqrt(n_each.min())),
        "pointwise_target_halfwidth": 0.005,
        "target_met": bool(np.all(half <= 0.005)),
    }


def publish_equity_lookup() -> dict:
    """Write the opt-in percentile. Does not change the default proxy files."""
    import agent

    wins, ties, n_each, reps = completed_stage_counts()
    summary = equity_summary()
    if summary["n_min"] != summary["n_max"]:
        raise RuntimeError("refusing to publish uneven orbit coverage")
    n = int(summary["n_min"])
    mean = (wins + 0.5 * ties) / n
    orb = np.load(OUT / "orbit_map.npz")
    holding_orbit = orb["holding_orbit"]
    counts = np.asarray(orb["counts"], dtype=np.int64)
    if holding_orbit.shape != (HOLDINGS,):
        raise RuntimeError("orbit map does not cover every holding")
    if counts.shape != mean.shape or int(counts.sum()) != HOLDINGS:
        raise RuntimeError("orbit multiplicities are not aligned with the means")
    # Percentile over the uniform hand, weighted by orbit size. Suit images share
    # a mean, so they share this value. Equal means across orbits share it too.
    pct_orbit = weighted_tie_percentile(mean, counts.astype(np.float64))
    pct_row = pct_orbit[holding_orbit].astype(np.float32)
    combos = all_combos().astype(np.int64)
    hidx = agent._hand_idx(combos)
    if hidx.shape != (HOLDINGS,) or int(hidx.min()) != 0 or int(hidx.max()) != HOLDINGS - 1:
        raise RuntimeError("hand index does not cover 0..376991")
    if len(np.unique(hidx)) != HOLDINGS:
        raise RuntimeError("hand index is not one-to-one")
    if np.array_equal(hidx, np.arange(HOLDINGS)):
        raise RuntimeError("combination row matched hand index; the scatter would hide a bug")
    pct = np.empty(HOLDINGS, dtype=np.float32)
    pct[hidx] = pct_row
    if not np.array_equal(pct[hidx], pct_row):
        raise RuntimeError("hand-index scatter did not round-trip")
    pair = []
    for hand in ([0, 1, 4, 8, 12], [1, 0, 5, 9, 13]):
        row = agent._hand_idx(np.asarray(sorted(hand), dtype=np.int64).reshape(1, 5))
        pair.append(float(pct[int(row[0])]))
    if pair[0] != pair[1]:
        raise RuntimeError("suit-swapped holdings received different percentiles")
    dest = agent._LOOKUP_DIR
    arrays = {
        "equity_pct": np.ascontiguousarray(pct),
        "equity_mean": np.ascontiguousarray(mean.astype(np.float64)),
        "equity_wins": np.ascontiguousarray(wins.astype(np.int32)),
        "equity_ties": np.ascontiguousarray(ties.astype(np.int32)),
        "equity_n": np.ascontiguousarray(n_each.astype(np.int32)),
    }
    manifest = {"format": agent._EQUITY_FORMAT, "samples_per_orbit": n, "arrays": {}}
    for name, arr in arrays.items():
        path = dest / f"{name}.npy"
        with open(path, "wb") as handle:
            np.save(handle, arr, allow_pickle=False)
        manifest["arrays"][name] = {
            "file": f"{name}.npy",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "nbytes": int(arr.nbytes),
            "sha256": agent._sha256_file(path),
        }
    (dest / "equity_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"samples_per_orbit": n, "suit_swap_percentile": pair[0], "summary": summary}


HELDOUT_STAGE = 100  # not a table-building stage (those are 0, 1, 2)


def _mc_se(wins: np.ndarray, ties: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Unbiased SE of the win+0.5*tie mean, and the mean. Not an exact equity."""
    wins = np.asarray(wins, dtype=np.float64)
    ties = np.asarray(ties, dtype=np.float64)
    n = float(n)
    mean = (wins + 0.5 * ties) / n
    second = (wins + 0.25 * ties) / n
    var = np.maximum(0.0, second - mean * mean)
    var_u = var * n / max(1.0, n - 1.0)
    return np.sqrt(var_u / n), mean


def equity_error_metrics(
    fresh: np.ndarray,
    table: np.ndarray,
    formula: np.ndarray,
    counts: np.ndarray,
    se_fresh: np.ndarray,
    se_table: np.ndarray,
) -> dict:
    """Plain and multiplicity-weighted errors. Fresh means are Monte Carlo, not truth."""
    fresh = np.asarray(fresh, dtype=np.float64)
    table = np.asarray(table, dtype=np.float64)
    formula = np.asarray(formula, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    se_fresh = np.asarray(se_fresh, dtype=np.float64)
    se_table = np.asarray(se_table, dtype=np.float64)
    weight = counts / counts.sum()

    def _pack(err: np.ndarray, name: str) -> dict:
        return {
            f"{name}_rmse": float(np.sqrt(np.mean(err * err))),
            f"{name}_mae": float(np.mean(np.abs(err))),
            f"{name}_rmse_weighted": float(np.sqrt(np.sum(weight * err * err))),
            f"{name}_mae_weighted": float(np.sum(weight * np.abs(err))),
        }

    noise_var = se_fresh * se_fresh + se_table * se_table
    out = {}
    out.update(_pack(fresh - table, "fresh_vs_table"))
    out.update(_pack(formula - fresh, "formula_vs_fresh"))
    out.update(_pack(formula - table, "formula_vs_table"))
    out["noise_floor_rmse"] = float(np.sqrt(np.mean(noise_var)))
    out["noise_floor_rmse_weighted"] = float(np.sqrt(np.sum(weight * noise_var)))
    return out


def _heldout_worker(payload: dict) -> None:
    ids = np.asarray(payload["orbit_ids"], dtype=np.int32)
    reps = np.asarray(payload["reps"], dtype=np.int16)
    n = int(payload["n"])
    stage = int(payload["stage"])
    batch = int(payload["batch"])
    wins = np.zeros(ids.shape[0], dtype=np.int32)
    ties = np.zeros(ids.shape[0], dtype=np.int32)
    prog = Path(payload["progress"])
    t0 = time.perf_counter()
    for i in range(ids.shape[0]):
        w, t = sample_counts(reps[i], n, orbit_rng(int(ids[i]), stage), batch)
        wins[i] = w
        ties[i] = t
        if (i + 1) % 5 == 0 or i + 1 == ids.shape[0]:
            _write_json(prog, {
                "done": i + 1,
                "rows": int(ids.shape[0]),
                "samples": (i + 1) * n,
                "seconds": time.perf_counter() - t0,
                "finished": i + 1 == ids.shape[0],
            })
    _atomic_savez(
        Path(payload["ckpt"]),
        orbit_id=ids,
        wins=wins,
        ties=ties,
        n=np.int32(n),
        stage=np.int32(stage),
    )


def run_heldout(
    n_orbits: int = 1000,
    n_fresh: int = 38416,
    workers: int | None = None,
    stage: int = HELDOUT_STAGE,
    select_seed: int = 424242,
) -> dict:
    """Independent worlds on true orbits. Stage must not be a generation stage.

    Generation streams are SeedSequence([20261107, orbit_id, stage]) for
    stage in {0, 1, 2}. This uses ``stage`` (default 100) and a separate
    SeedSequence to choose which orbits. The fresh means are another Monte
    Carlo estimate, not exact equity and not a replay of the table.
    """
    import multiprocessing as mp

    if int(stage) in range(len(STAGE_N)):
        raise RuntimeError(f"held-out stage {stage} collides with table generation")
    wins, ties, n_each, reps = completed_stage_counts()
    if np.any(n_each != n_each[0]):
        raise RuntimeError("held-out check wants equal n")
    n_table = int(n_each[0])
    table_mean = (wins + 0.5 * ties) / n_table
    counts = np.load(OUT / "orbit_map.npz")["counts"].astype(np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([MASTER, int(select_seed), int(stage)]))
    chosen = np.sort(rng.choice(N_ORBITS, size=int(n_orbits), replace=False).astype(np.int32))
    workers = int(os.cpu_count() if workers is None else workers)
    workers = max(1, min(workers, int(chosen.shape[0])))
    work = Path("/tmp") / f"poker-heldout-{os.getpid()}"
    if work.exists():
        raise RuntimeError(f"{work} already exists")
    work.mkdir(parents=True)
    progress_path = OUT / "heldout_progress.json"
    started = time.time()
    _write_json(progress_path, {
        "status": "running",
        "orbits": int(n_orbits),
        "samples_per_orbit": int(n_fresh),
        "stage": int(stage),
        "workers": workers,
        "samples_done": 0,
        "samples_target": int(n_orbits) * int(n_fresh),
        "started_unix": started,
    })
    edges = np.linspace(0, chosen.shape[0], workers + 1, dtype=np.int64)
    ctx = mp.get_context("fork")
    procs = []
    for wid in range(workers):
        lo, hi = int(edges[wid]), int(edges[wid + 1])
        payload = {
            "orbit_ids": chosen[lo:hi],
            "reps": np.ascontiguousarray(reps[chosen[lo:hi]]),
            "n": int(n_fresh),
            "stage": int(stage),
            "batch": BATCH,
            "ckpt": str(work / f"w{wid:02d}.npz"),
            "progress": str(work / f"w{wid:02d}.progress.json"),
        }
        proc = ctx.Process(target=_heldout_worker, args=(payload,), name=f"heldout-w{wid}")
        proc.start()
        procs.append(proc)
    while any(p.is_alive() for p in procs):
        samples = 0
        orbits_done = 0
        for wid in range(workers):
            path = work / f"w{wid:02d}.progress.json"
            if not path.exists():
                continue
            snap = json.loads(path.read_text(encoding="utf-8"))
            samples += int(snap.get("samples", 0))
            orbits_done += int(snap.get("done", 0))
        _write_json(progress_path, {
            "status": "running",
            "orbits_done": orbits_done,
            "orbits": int(n_orbits),
            "samples_done": samples,
            "samples_target": int(n_orbits) * int(n_fresh),
            "elapsed_seconds": time.time() - started,
            "stage": int(stage),
            "workers": workers,
        })
        time.sleep(2)
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            _write_json(progress_path, {"status": "failed", "exit": proc.exitcode})
            raise SystemExit(f"FAILED held-out worker exit {proc.exitcode}")
    fresh_wins = np.zeros(chosen.shape[0], dtype=np.int64)
    fresh_ties = np.zeros(chosen.shape[0], dtype=np.int64)
    for wid in range(workers):
        z = np.load(work / f"w{wid:02d}.npz")
        if int(z["n"]) != int(n_fresh) or int(z["stage"]) != int(stage):
            raise RuntimeError("held-out chunk header does not match the request")
        ids = z["orbit_id"]
        pos = {int(oid): i for i, oid in enumerate(chosen.tolist())}
        for local, oid in enumerate(ids.tolist()):
            fresh_wins[pos[int(oid)]] = int(z["wins"][local])
            fresh_ties[pos[int(oid)]] = int(z["ties"][local])
    se_fresh, fresh_mean = _mc_se(fresh_wins, fresh_ties, int(n_fresh))
    se_table, _ = _mc_se(wins[chosen], ties[chosen], n_table)
    table_sel = table_mean[chosen]
    import agent

    reps64 = reps[chosen].astype(np.int64)
    formula = agent._preflop_features(reps64) @ agent._PRE_COEF
    # The frozen formula is a function of rank and suit-count features. A suit
    # swap of the representative must not move it. If it does, the orbit mean
    # is the wrong comparison target.
    suit = reps64 & 3
    suit = np.where(suit == 0, 1, np.where(suit == 1, 0, suit))
    swapped = np.sort(((reps64 >> 2) << 2) | suit, axis=1)
    formula_swap = agent._preflop_features(swapped) @ agent._PRE_COEF
    formula_suit_gap = float(np.max(np.abs(formula - formula_swap)))
    if formula_suit_gap != 0.0:
        raise RuntimeError(f"formula is not suit-invariant, max gap {formula_suit_gap}")
    metrics = equity_error_metrics(
        fresh_mean, table_sel, formula, counts[chosen].astype(np.float64), se_fresh, se_table
    )
    metrics["worst_case_se_one_mean"] = float(0.5 / np.sqrt(n_fresh))
    metrics["worst_case_se_difference"] = float(0.5 * np.sqrt(1.0 / n_fresh + 1.0 / n_table))
    summary = {
        "orbits": int(n_orbits),
        "fresh_samples_per_orbit": int(n_fresh),
        "table_samples_per_orbit": n_table,
        "stage": int(stage),
        "select_seed": int(select_seed),
        "select_stream": [MASTER, int(select_seed), int(stage)],
        "world_stream": "SeedSequence([20261107, orbit_id, stage])",
        "generation_stages": [0, 1, 2],
        "disjoint_from_generation": True,
        "formula": "frozen _PRE_COEF dot _preflop_features; not refit",
        "formula_suit_max_abs_gap": formula_suit_gap,
        "fresh_means_are": "independent Monte Carlo estimates, not exact equity and not the table labels",
        "multiplicity_weight": "orbit size among the sampled orbits",
        "elapsed_seconds": time.time() - started,
        **metrics,
    }
    _atomic_savez(
        OUT / "heldout_orbits.npz",
        orbit_id=chosen,
        counts=counts[chosen].astype(np.int32),
        fresh_wins=fresh_wins.astype(np.int32),
        fresh_ties=fresh_ties.astype(np.int32),
        fresh_n=np.int32(n_fresh),
        table_wins=wins[chosen].astype(np.int32),
        table_ties=ties[chosen].astype(np.int32),
        table_n=np.int32(n_table),
        formula=formula.astype(np.float64),
    )
    _write_json(OUT / "heldout_summary.json", summary)
    _write_json(progress_path, {**summary, "status": "done", "samples_done": int(n_orbits) * int(n_fresh)})
    # Worker scratch lives under /tmp. The compact npz above is the validation record.
    for path in work.iterdir():
        path.unlink()
    work.rmdir()
    return summary


def heldout_check(n_orbits: int = 128, n_fresh: int = 4096, seed: int = 424242) -> dict:
    """Serial fresh worlds, disjoint from the table seeds, on a fixed orbit sample."""
    wins, ties, n_each, reps = completed_stage_counts()
    if np.any(n_each != n_each[0]):
        raise RuntimeError("held-out check wants equal n")
    n_table = int(n_each[0])
    table_mean = (wins + 0.5 * ties) / n_table
    rng = np.random.default_rng(np.random.SeedSequence([MASTER, int(seed), HELDOUT_STAGE]))
    chosen = np.sort(rng.choice(N_ORBITS, size=n_orbits, replace=False))
    fresh_wins = np.empty(n_orbits, dtype=np.int64)
    fresh_ties = np.empty(n_orbits, dtype=np.int64)
    for i, oid in enumerate(chosen):
        w, t = sample_counts(reps[int(oid)], n_fresh, orbit_rng(int(oid), HELDOUT_STAGE), BATCH)
        fresh_wins[i] = w
        fresh_ties[i] = t
    se_fresh, fresh = _mc_se(fresh_wins, fresh_ties, n_fresh)
    se_table, _ = _mc_se(wins[chosen], ties[chosen], n_table)
    ref = table_mean[chosen]
    import agent

    proxy = agent._preflop_features(reps[chosen].astype(np.int64)) @ agent._PRE_COEF
    counts = np.load(OUT / "orbit_map.npz")["counts"].astype(np.float64)[chosen]
    metrics = equity_error_metrics(fresh, ref, proxy, counts, se_fresh, se_table)
    return {
        "orbits": int(n_orbits),
        "fresh_samples": int(n_fresh),
        "table_samples": n_table,
        "seed_stage": HELDOUT_STAGE,
        "note": "Stage 100 is not a table-building stage. These worlds are not the table labels.",
        **metrics,
    }


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "verify":
        info = verify_exhaustive()
        bad, checked = parity_mismatches()
        print(json.dumps({
            "orbits": info["orbits"],
            "canon_seconds": info["seconds"],
            "min_orbit_size": info["min_orbit_size"],
            "max_orbit_size": info["max_orbit_size"],
            "parity_mismatches": bad,
            "parity_checked": checked,
        }, indent=2))
        if bad:
            raise SystemExit(1)
        return
    if cmd == "run":
        run_table()
        return
    if cmd == "report":
        print(json.dumps({"summary": equity_summary(), "heldout": heldout_check()}, indent=2))
        return
    if cmd == "publish":
        print(json.dumps(publish_equity_lookup(), indent=2))
        return
    if cmd == "heldout":
        n_orbits = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        n_fresh = int(sys.argv[3]) if len(sys.argv) > 3 else STAGE_N[-1]
        print(json.dumps(run_heldout(n_orbits=n_orbits, n_fresh=n_fresh), indent=2))
        return
    raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
