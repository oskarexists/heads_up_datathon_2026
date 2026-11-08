"""Duplicate-match harness and one-CPU HTTP timing for the review.

In-process matches call agent.choose_action with a new memory dict each half.
The hero RNG is replaced with a seeded Generator so the run repeats. The
tournament bot leaves that RNG unseeded.

Paired result, matching judge_engine.combine_halves:

    paired_bb[d] = (hero_chips_half1[d] + hero_chips_half2[d]) / big_blind
    mean = sum(paired_bb) / n
    SE = stdev(paired_bb, ddof=1) / sqrt(n)
    CI = mean ± 1.96 * SE

n is the number of deals, not the number of hands. Half 1 puts the hero in
seat 0; half 2 puts the hero in seat 1 on the same make_deal(seed, d) cards.
Button seat stays deal_index % 2.

Usage (from the repo root, 1 CPU):

    taskset -c 0 python review_harness.py --deals 1000 --seed review-20261107

    taskset -c 0 python review_harness.py --http-half 1000 --http-opponent calling_station
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import math
import os
import statistics
import subprocess
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

import agent
from game import BIG_BLIND, CHOP_Z, Action, AgentConfig, Game, button_for_deal, make_deal
from judge_engine import agent1_seat_for_half, decide_verdict
from review_adversaries import ADVERSARIES

CI_FORMULA = (
    "paired_bb[d] = (hero_net_chips_half1[d] + hero_net_chips_half2[d]) / big_blind; "
    "mean = sum(paired_bb)/n; SE = sample_stdev(paired_bb, ddof=1)/sqrt(n); "
    "CI = mean ± 1.96*SE. n = deals. This is the judge's paired-deal interval, "
    "not an interval on independent single hands."
)


def _seed_int(text: str) -> int:
    return zlib.adler32(text.encode()) & 0x7FFFFFFF


def play_half(opponent: str, match_seed: str, half: int, num_deals: int, decide_exc: list) -> dict:
    """Play one half. Hero memory is new. Returns hero chip nets and internal timing."""
    hero_seat = agent1_seat_for_half(half)
    hero_mem: dict = {}
    agent.new_half(AgentConfig(num_hands=num_deals), hero_mem)
    hero_mem["rng"] = np.random.default_rng(_seed_int(f"{match_seed}:{half}:{opponent}:hero"))
    villain = ADVERSARIES[opponent]()
    vill_mem: dict = {}
    villain.new_half(AgentConfig(num_hands=num_deals), vill_mem)

    real_decide = agent._decide

    def wrapped(obs, memory):
        try:
            return real_decide(obs, memory)
        except Exception:
            decide_exc[0] += 1
            raise

    agent._decide = wrapped
    nets: list[int] = []
    illegal = 0
    try:
        for deal_index in range(num_deals):
            game = Game(make_deal(match_seed, deal_index), button_for_deal(deal_index), deal_index)
            while not game.is_terminal:
                seat = game.to_act
                obs = game.observation(seat)
                if seat == hero_seat:
                    action = agent.choose_action(obs, hero_mem)
                    if action not in obs.legal_actions:
                        illegal += 1
                        action = Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
                else:
                    action = villain.choose_action(obs, vill_mem)
                    if action not in obs.legal_actions:
                        raise RuntimeError(f"{opponent} returned illegal {action} from {obs.legal_actions}")
                game.apply(action)
            result = game.result()
            nets.append(result.nets[hero_seat])
            agent.on_hand_end(game.hand_result(hero_seat), hero_mem)
            villain.on_hand_end(game.hand_result(1 - hero_seat), vill_mem)
    finally:
        agent._decide = real_decide

    return {
        "nets": nets,
        "illegal": illegal,
        "hero_internal_time_s": hero_mem.get("time_used", 0.0),
        "hero_calls": hero_mem.get("calls", 0),
        "hero_scale": hero_mem.get("scale", None),
        "hero_load": hero_mem.get("load", None),
        "hero_hands": hero_mem.get("hands", 0),
    }


def summarize_pair(half1: list[int], half2: list[int]) -> dict:
    n = len(half1)
    if len(half2) != n:
        raise ValueError("halves differ in length")
    paired_chips = [half1[i] + half2[i] for i in range(n)]
    paired_bb = [c / BIG_BLIND for c in paired_chips]
    mean = sum(paired_bb) / n if n else 0.0
    std = statistics.stdev(paired_bb) if n > 1 else 0.0
    se = std / math.sqrt(n) if n else 0.0
    ci = (mean - CHOP_Z * se, mean + CHOP_Z * se)
    verdict, points, reason = decide_verdict(ci[0], ci[1])
    return {
        "n_deals": n,
        "mean_bb_per_deal": mean,
        "std_bb": std,
        "se_bb": se,
        "ci95": [ci[0], ci[1]],
        "z": CHOP_Z,
        "verdict_if_this_were_the_match": verdict,
        "reason": reason,
        "points": points,
        "half1_chips": sum(half1),
        "half2_chips": sum(half2),
        "paired_bb": paired_bb,
        "half1_chips_by_deal": half1,
        "half2_chips_by_deal": half2,
    }


def run_matches(opponents: list[str], match_seed: str, num_deals: int) -> dict:
    rows = []
    summaries = []
    decide_exc = [0]
    t0 = time.perf_counter()
    for name in opponents:
        print(f"match {name} deals={num_deals} seed={match_seed}", flush=True)
        t1 = time.perf_counter()
        before = decide_exc[0]
        h1 = play_half(name, match_seed, 1, num_deals, decide_exc)
        h2 = play_half(name, match_seed, 2, num_deals, decide_exc)
        stats = summarize_pair(h1["nets"], h2["nets"])
        elapsed = time.perf_counter() - t1
        print(
            f"  {name}: {stats['mean_bb_per_deal']:+.3f} bb/deal "
            f"CI [{stats['ci95'][0]:+.3f}, {stats['ci95'][1]:+.3f}] "
            f"in {elapsed:.1f}s exc {decide_exc[0] - before}",
            flush=True,
        )
        for i, bb in enumerate(stats["paired_bb"]):
            rows.append({
                "opponent": name,
                "deal_index": i,
                "half1_chips": stats["half1_chips_by_deal"][i],
                "half2_chips": stats["half2_chips_by_deal"][i],
                "paired_bb": round(bb, 6),
            })
        summaries.append({
            "opponent": name,
            "n_deals": num_deals,
            "match_seed": match_seed,
            "hero_rng": "np.random.default_rng(adler32(seed:half:opponent:hero)) after new_half",
            "mean_bb_per_deal": stats["mean_bb_per_deal"],
            "std_bb": stats["std_bb"],
            "se_bb": stats["se_bb"],
            "ci95": stats["ci95"],
            "z": CHOP_Z,
            "verdict_if_this_were_the_match": stats["verdict_if_this_were_the_match"],
            "reason": stats["reason"],
            "half1_hero_chips": stats["half1_chips"],
            "half2_hero_chips": stats["half2_chips"],
            "illegal_actions": h1["illegal"] + h2["illegal"],
            "decide_exceptions": decide_exc[0] - before,
            "wall_seconds": elapsed,
            "internal_time_half1_s": h1["hero_internal_time_s"],
            "internal_time_half2_s": h2["hero_internal_time_s"],
            "hero_calls_half1": h1["hero_calls"],
            "hero_calls_half2": h2["hero_calls"],
            "scale_end_half1": h1["hero_scale"],
            "scale_end_half2": h2["hero_scale"],
            "load_end_half1": h1["hero_load"],
            "load_end_half2": h2["hero_load"],
        })
    return {
        "kind": "in_process_duplicate",
        "hero": "agent.py Holey Donut",
        "base_note": "memory dict replaced each half; process globals are the lookup tables",
        "match_seed": match_seed,
        "num_deals": num_deals,
        "big_blind": BIG_BLIND,
        "ci_formula": CI_FORMULA,
        "seat_normalization": (
            "half 1 hero is seat 0; half 2 hero is seat 1; "
            "both halves use make_deal(match_seed, deal_index) and button deal_index % 2; "
            "paired chips add the hero's net from each seat"
        ),
        "wall_seconds": time.perf_counter() - t0,
        "decide_exceptions_total": decide_exc[0],
        "opponents": summaries,
        "rows": rows,
    }


def _wait_ready(port: int, proc: subprocess.Popen, timeout: float) -> float:
    deadline = time.perf_counter() + timeout
    start = time.perf_counter()
    last = ""
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"bot on {port} exited {proc.returncode} before ready; last={last}")
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
        try:
            conn.request("GET", "/")
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status == 200:
                return time.perf_counter() - start
            last = raw[:200].decode("utf-8", "replace")
        except Exception as exc:
            last = str(exc)
        finally:
            conn.close()
        time.sleep(0.05)
    raise RuntimeError(f"bot on {port} not ready in {timeout}s; last={last}")


def _post(conn: http.client.HTTPConnection, path: str, body: dict) -> tuple[int, dict, float]:
    payload = json.dumps(body)
    start = time.perf_counter()
    conn.request("POST", path, body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read()
    elapsed = time.perf_counter() - start
    data = json.loads(raw) if raw else {}
    return resp.status, data, elapsed


def run_http_half(num_deals: int, opponent: str, match_seed: str, port: int) -> dict:
    """One fresh server process, hero in seat 0, judge-like charged bank.

    Charged seconds are sum(max(0, request_latency - 0.001)) over /new-half,
    /act, and /hand-end. Health-check time is startup only and is not charged.
    The opponent runs in-process, which does not add to the hero's HTTP time.
    """
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    log_path = Path(f"/tmp/holey-donut-http-{port}.log")
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        ["taskset", "-c", "0", sys.executable, str(ROOT / "agent.py")],
        cwd=str(ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    villain = ADVERSARIES[opponent]()
    vill_mem: dict = {}
    requests = []
    try:
        startup = _wait_ready(port, proc, 20.0)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
        status, data, dt = _post(conn, "/new-half", {"config": AgentConfig(num_hands=num_deals).to_dict()})
        requests.append({"path": "/new-half", "hand": None, "status": status, "seconds": dt})
        if status != 200:
            raise RuntimeError(f"new-half failed: {status} {data}")
        villain.new_half(AgentConfig(num_hands=num_deals), vill_mem)
        pending: list[dict] = []
        hero_nets: list[int] = []
        errors = 0
        for deal_index in range(num_deals):
            game = Game(make_deal(match_seed, deal_index), button_for_deal(deal_index), deal_index)
            while not game.is_terminal:
                seat = game.to_act
                obs = game.observation(seat)
                if seat == 0:
                    status, data, dt = _post(
                        conn, "/act", {"observation": obs.to_dict(), "hand_results": pending}
                    )
                    pending = []
                    requests.append({
                        "path": "/act",
                        "hand": deal_index,
                        "street": int(obs.street),
                        "status": status,
                        "seconds": dt,
                    })
                    if status != 200 or not isinstance(data, dict) or "action" not in data:
                        errors += 1
                        action = Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
                    else:
                        errors += len(data.get("hand_end_errors") or [])
                        try:
                            action = Action[data["action"]]
                        except Exception:
                            errors += 1
                            action = Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
                        if action not in obs.legal_actions:
                            errors += 1
                            action = Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
                else:
                    action = villain.choose_action(obs, vill_mem)
                game.apply(action)
            hero_nets.append(game.result().nets[0])
            pending.append(game.hand_result(0).to_dict())
            villain.on_hand_end(game.hand_result(1), vill_mem)
        if pending:
            status, data, dt = _post(conn, "/hand-end", {"results": pending})
            requests.append({"path": "/hand-end", "hand": None, "status": status, "seconds": dt})
            if isinstance(data, dict):
                errors += len(data.get("hand_end_errors") or [])
        conn.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    charged = [max(0.0, r["seconds"] - 0.001) for r in requests]
    latencies = [r["seconds"] for r in requests]
    act_lat = [r["seconds"] for r in requests if r["path"] == "/act"]
    return {
        "kind": "http_half_trace",
        "opponent": opponent,
        "hero_seat": 0,
        "half": 1,
        "match_seed": match_seed,
        "num_deals": num_deals,
        "cpu": "taskset -c 0",
        "thread_env": "OMP/MKL/OPENBLAS/NUMEXPR=1",
        "startup_seconds_to_get": startup,
        "startup_limit_seconds": 15.0,
        "startup_is_not_charged_to_the_bank": True,
        "hero_chips": sum(hero_nets),
        "errors": errors,
        "requests": len(requests),
        "act_requests": len(act_lat),
        "wall_sum_s": sum(latencies),
        "charged_bank_s": sum(charged),
        "charge_rule": "sum(max(0, latency-0.001)) over /new-half, /act, and /hand-end",
        "bank_limit_s": 8.0,
        "request_max_s": max(latencies) if latencies else 0.0,
        "act_max_s": max(act_lat) if act_lat else 0.0,
        "act_p50_s": statistics.median(act_lat) if act_lat else 0.0,
        "requests_over_1s": sum(1 for s in latencies if s > 1.0),
        "note": (
            "Latencies are client-observed HTTP times for a live agent.py process. "
            "They include JSON and the server. An in-process perf_counter around "
            "choose_action is smaller than this. The opponent did not use HTTP."
        ),
        "trace": [
            {
                "i": i,
                "path": r["path"],
                "hand": r["hand"],
                "street": r.get("street"),
                "seconds": round(r["seconds"], 6),
                "charged_s": round(max(0.0, r["seconds"] - 0.001), 6),
                "status": r["status"],
            }
            for i, r in enumerate(requests)
        ],
    }


def write_outputs(out_dir: Path, payload: dict, http_payload: dict | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = payload["rows"]
    compact = {k: v for k, v in payload.items() if k != "rows"}
    (out_dir / "duplicate_summary.json").write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "duplicate_paired.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["opponent", "deal_index", "half1_chips", "half2_chips", "paired_bb"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    if http_payload is not None:
        (out_dir / "http_half_trace.json").write_text(json.dumps(http_payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review duplicate matches and optional HTTP timing")
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--seed", default="review-20261107")
    parser.add_argument("--opponents", default="maniac,calling_station,min_raise,exploiter")
    parser.add_argument("--out", default=str(ROOT / "review_results"))
    parser.add_argument("--http-half", type=int, default=0, help="if >0, also time one HTTP half of this many deals")
    parser.add_argument("--http-opponent", default="calling_station")
    parser.add_argument("--http-port", type=int, default=5918)
    parser.add_argument("--skip-inprocess", action="store_true")
    args = parser.parse_args(argv)
    opponents = [p.strip() for p in args.opponents.split(",") if p.strip()]
    unknown = [p for p in opponents if p not in ADVERSARIES]
    if unknown:
        print(f"unknown opponents {unknown}; choose from {sorted(ADVERSARIES)}", file=sys.stderr)
        return 2
    if args.http_opponent not in ADVERSARIES:
        print(f"unknown http opponent {args.http_opponent}", file=sys.stderr)
        return 2

    if args.skip_inprocess:
        payload = {"kind": "in_process_duplicate", "opponents": [], "rows": [], "ci_formula": CI_FORMULA}
    else:
        payload = run_matches(opponents, args.seed, args.deals)
    http_payload = None
    if args.http_half > 0:
        print(f"http half deals={args.http_half} opponent={args.http_opponent}", flush=True)
        http_payload = run_http_half(args.http_half, args.http_opponent, args.seed, args.http_port)
        print(
            f"  startup {http_payload['startup_seconds_to_get']:.2f}s "
            f"charged {http_payload['charged_bank_s']:.3f}s "
            f"max {http_payload['request_max_s']:.4f}s errors {http_payload['errors']}",
            flush=True,
        )
    write_outputs(Path(args.out), payload, http_payload)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
