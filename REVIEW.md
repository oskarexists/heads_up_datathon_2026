# Adversarial review: Holey Donut

## Scope

- Repo branch: `adversary-review-20261107`
- Reviewed commit: `5571729ae285bd2e534e0b09c77a83a95ad2899e` (`v5: stronger folded-hands-are-weak assumption`)
- Bot: `PARTICIPANT = "Donut Embezzler"`, `AGENT_NAME = "Holey Donut"` in `agent.py`
- Official starter used for comparison: `https://github.com/tamu-datathon-org/heads-up-bundle` at `c83ab6eeb7660a2fd1fae7df224315d41fe985a9` (cloned to `/tmp/poker-starter-review-20261107`)
- Rules brief: `../poker-brief.md`. Where the brief and the code disagree, `game.py` in the starter is the source of truth. Oskar's `game.py` is byte-identical to that file.
- This review did not change `agent.py`. No strategy rewrite. No push, no PR, no contest submission.
- Python 3.12.3, numpy 2.5.4, temporary venv `/tmp/poker-venv-review`. Host: AMD Ryzen 7 3700X. Timed runs used `taskset -c 0` and `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`.

## What matches the official starter

`sha256sum` is identical for `game.py`, `evaluator.py`, `judge_engine.py`, `test_agent.py`, `requirements-dev.txt`, `docker/Dockerfile.base`, `docker/Dockerfile.agent`, `tools/check_build.sh`, and `tools/smoke_test.py`.

`agent.py` server plumbing from the `# ===== SERVER PLUMBING` marker to EOF is identical to the starter (125 lines). The only product changes are the strategy block and `numpy` added to `requirements.txt`. The base image already installs numpy, so that pin is redundant. FastAPI `0.156.1` and uvicorn `0.54.0` are unchanged.

Confirmed from `game.py` / `evaluator.py` and `test_review.py`:

- 36-card deck, 5 hole cards, exactly 2 hole + 3 board (`omaha_score`)
- Flush beats a full house; trips beat a straight; wheel A-6-7-8-9 is a nine-high straight and loses to 6-7-8-9-T
- Legal actions are only `FOLD`, `CHECK_CALL`, `RAISE_HALF`, `RAISE_POT`, and only those in `obs.legal_actions`. `RAISE_HALF` is the smaller pot-limit size (engine may bump it to the minimum raise or to all-in)
- Blinds are 1/2, stacks 80, max 4 raises per street, blinds are not action-history entries
- Duplicate seats: half 1 agent1 is seat 0, half 2 agent1 is seat 1, same `make_deal(seed, deal_index)`, button is `deal_index % 2`
- Time: startup health check 15s and not charged; bank 8s per half; charge `max(0, latency - 0.001)` per `/new-half`, `/act`, and `/hand-end`; 1s per request; 5 errors forfeit the half
- `new_half` clears `MEMORY`. This bot keeps learned state in that dict. Import-time tables are deterministic and are not updated during play

## Ranked findings

### 1. Confirmed: flush-draw bonus ignores the two-hole-card rule

`agent.py:216` in `_board_strength`:

```python
flush_draw = ((np.minimum(_SCNT[hidx], 2) + np.minimum(bs, 3)[None, :]) == 4).any(1)
```

`min(hole_suit, 2) + min(board_suit, 3) == 4` is true for one hole card and three board cards. An Omaha flush needs two hole cards of that suit, so a one-card holding is not a flush draw. The bonus added on the flop is `1.0 * 1.3` category units (`agent.py:224-225`), which is larger than a one-pair score. The bonus is used only to rank sampled opponent hands, not to score showdown.

Measured on 2000 deals from `make_deal("freq", i)`: 81/2000 flops (4.1%) are monotone, and 304/2000 turns (15.2%) already have three cards of one suit. On those boards a large share of random five-card holdings have exactly one card of the suit and get the false bonus.

Reproduction: `test_draw_bonus_flags_hands_that_cannot_draw_with_two_hole_cards` (Ah plus four non-hearts on 9h8h7h).

Recommended fix, not applied: count a flush draw only when the holding has at least two cards of the suit and the board has exactly two. Do not add the bonus when the board already has three (that is a made flush if two hole cards match, and the made-hand score already sees it). Re-run the duplicate harness before keeping the change. It will move postflop ranges.

### 2. Confirmed: straight-draw bonus has the same hole-card hole

`agent.py:222-223` flags a straight draw when `min(hole_ranks_in_window, 2) + min(board_ranks_in_window, 3) >= 4` and the made hand is below a straight. One hole rank plus three board ranks qualifies. The counterexample in the same test is aces, kings, and a queen on 7c 8d 9h: no pair of hole ranks forms a four-card straight window with the board, and the function still flags a draw (the wheel window treats the ace as one rank beside 7-8-9).

The sum also double-counts a rank that is on both the hole cards and the board. That is a second false-positive path in the same expression.

Recommended fix, not applied: require two distinct hole ranks inside the window, and count the size of the union with the board ranks. Same caveat: this is a ranker change, not a showdown fix.

### 3. Confirmed: extended pools keep batch-local percentiles

`_extend_pool` (`agent.py:275-282`) builds a new sample, ranks street strength inside that sample only, and concatenates those percentiles onto the old ones. Old rows are not re-ranked. Preflop percentiles come from the global `_PRE_PCT` table and do stay comparable. Flop, turn, and river percentiles do not.

`test_extended_pool_percentiles_stay_inside_each_batch` extends a 30-row flop pool by 30 and checks that the stored vector differs from `_percentile` of the combined sample by more than 0.05.

This runs when the weighted range is narrower than the target. Against an opponent that pot-raises whenever legal, `_extend_pool` was called 121 times across 250 hero decisions in 80 hands (`make_deal("ext", ...)`). Facing aggression is exactly when the range model is supposed to matter.

Recommended fix, not applied: after concatenating `draws`, `pidx`, `hidx`, and `win`, keep the absolute preflop percentile and delete cached street percentiles so `_street_pct` recomputes them on the full pool. That adds ranking work on the turns that already extend the pool, so time it on one CPU before adopting it.

### 4. Confirmed behavior, low tournament risk: exceptions never reach the judge

`choose_action` (`agent.py:824-830`) turns any `_decide` exception into check, or fold if facing a bet. `on_hand_end` (`agent.py:870`) swallows its own exceptions, so a learning bug does not increment the judge's error counter. Five real errors forfeit the half, so the wrapper is deliberate.

`test_decide_exception_does_not_escape_choose_action` locks the fallback. Across the 8,000-hand duplicate run below, the wrapper around `_decide` saw 0 exceptions, and the HTTP half reported 0 protocol errors. Nothing in this review is currently taking the fallback. The risk is a future bug that check/folds silently while `pytest` still looks clean if it only checks HTTP status.

### 5. Checked, and not bugs

- Dense Omaha ranks from `_best` agreed with `omaha_score` ordering on 300 random pairs in the diagnostic and 40 pairs in `test_agent_rank_table_matches_omaha_score_order`. Unique five-card scores: 1404, so the int16 table does not overflow.
- The history walk's reconstructed pot and `to_call` matched `Game` on mixed pot raises, half raises, and folds (`test_replay_pot_and_to_call_match_the_engine`, 25 hands; diagnostic 190/190 decisions).
- A river straight flush (As Ks with Qs Js Ts on board, hero to call a pot bet) was not folded (`test_river_nuts_are_not_folded`).
- Startup to `GET /` on a cold process: **1.57s** (limit 15s). Import of `agent.py` on one CPU was 1.47s in the diagnostic.
- No illegal action was returned in the duplicate matches or the existing `test_agent.py` gauntlet.

### Strategy hypotheses (not confirmed defects)

- `fold_k` (`agent.py:637` and `_equity_when_called` at `agent.py:655-659`) treats folded hands as weaker than the current equity. That can drive equity-when-called to 0. It did not stop the bot from beating a calling station, which is the opponent that punishes missed value. That is evidence against "the bot never value-bets," not proof the constant is well calibrated.
- The bluff cap, junk open-fold (`equity < 0.37` on the button), and showdown range calibration are policy choices. This review did not isolate them.
- `new_half` uses `np.random.default_rng()` with no seed (`agent.py:789`). Tournament runs are stochastic. The harness below replaces `memory["rng"]` after `new_half` so the published numbers repeat. Repeating the HTTP server will not repeat those chips.
- The internal clock adds `0.4ms * load` per call and does not subtract the judge's 1ms grace (`agent.py:540-553`). On this host the bot's own `time_used` finished near 4s while the charged HTTP bank was 2.03s, and `scale` sat on the 1.6 cap. The bot is leaving sample budget unused here. A slower contest CPU could go the other way. `load` at the end of the halves was about 0.6 to 1.3 versus the author's reference draw, so this machine is in the same band as the development machine, not a proof about the contest VM.

## Duplicate matches

Command:

```bash
cd /home/alexn/Documents/code/datathon_2026/poker-bot-tournament/oskar-review
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  taskset -c 0 /tmp/poker-venv-review/bin/python -u review_harness.py \
  --deals 1000 --seed review-20261107 --http-half 1000 --http-opponent calling_station
```

Each opponent is 1000 deals, two halves, fresh `memory` from `new_half` each half. Hero RNG seed is `adler32(f"{match_seed}:{half}:{opponent}:hero")`. Wall time for all four matches was 32.2s. `_decide` exceptions: 0. Illegal actions: 0.

Seat normalization, same as `judge_engine.combine_halves`:

```text
paired_bb[d] = (hero_net_chips_half1[d] + hero_net_chips_half2[d]) / big_blind
mean = sum(paired_bb) / n
SE = sample_stdev(paired_bb, ddof=1) / sqrt(n)
CI = mean ± 1.96 * SE
```

`n = 1000` deals, not 2000 hands. `big_blind = 2`. Half 1 hero is seat 0. Half 2 hero is seat 1. Positive means the hero won chips across the two seats of that deal. The mean is the sum of the two seatings in bb per deal, which is what the judge calls bb/deal. It is about twice the average of a single seating.

The maniac CI was recomputed from `review_results/duplicate_paired.csv` and matched the summary to the last stored digit.

| Opponent | Mean bb/deal | SD | SE | 95% CI | Hero chips half1, half2 | Internal `time_used` half1, half2 |
|---|---:|---:|---:|---|---:|---:|
| maniac (always `RAISE_POT`, else call) | +6.785 | 24.682 | 0.781 | [+5.255, +8.315] | +6884, +6686 | 3.36s, 3.27s |
| calling station (always `CHECK_CALL`) | +12.666 | 17.013 | 0.538 | [+11.611, +13.720] | +13417, +11914 | 3.94s, 3.91s |
| min-raise prober (always `RAISE_HALF` when legal) | +7.994 | 22.639 | 0.716 | [+6.591, +9.397] | +9093, +6895 | 3.84s, 4.13s |
| exploiter (half-pot probe, then pot-steal or value-bet) | +6.202 | 25.001 | 0.791 | [+4.652, +7.752] | +6494, +5910 | 3.38s, 3.81s |

All four intervals are entirely above 0, so `decide_verdict` would score a win. That is not a tournament result. These four bots are weak on purpose. Raw rows: `review_results/duplicate_paired.csv`. Aggregates: `review_results/duplicate_summary.json`.

No fixed candidate was run. The ranker bugs above are real, and this bot still won these matches with the bugs present. A patch would change postflop equity estimates without a measured gain.

## Time bank (one CPU, real HTTP)

`review_harness.py --http-half 1000` started a cold `agent.py` under `taskset -c 0` on port 5918, waited for `GET /`, then played 1000 deals with the hero in seat 0 against an in-process calling station. Finished hands were queued and posted with the next `/act`, then `/hand-end`, which is the judge's pattern. The opponent was not on HTTP, so opponent think time is not in these numbers.

| Measurement | Value |
|---|---|
| Startup until `GET /` 200 | 1.57s (limit 15s; not charged) |
| Requests (`/new-half` + `/act` + `/hand-end`) | 3966 (3964 acts) |
| Wall sum of those requests | 5.973s |
| Charged bank `sum(max(0, latency-0.001))` | **2.030s** of 8s |
| Max request | **6.14ms** (limit 1s) |
| Act median | 1.36ms |
| Requests over 1s | 0 |
| Protocol errors | 0 |
| Hero chips this single half | +13554 (not a paired result) |

Trace: `review_results/http_half_trace.json`.

In-process `time_used` on the seeded calling-station half 1 was 3.94s. HTTP wall on the unseeded half was 5.97s. Charged bank was 2.03s because most of the wall time sits inside the first free millisecond (3966 requests times 1ms is 3.97s). An in-process timer underestimates HTTP wall and, on this host, overestimates the bank the judge would charge. Do not treat either number as the contest VM. Passing this trace is not tournament compliance: the image was not smoke-tested under the match's read-only filesystem, and the contest CPU is unknown.

`scale` ended at 1.6 on seven of eight in-process halves and 1.37 on one. The emergency check/fold path (`spent > 0.96 * bank`) was not used.

## Tests

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  taskset -c 0 /tmp/poker-venv-review/bin/python -m pytest -q
```

Result: **21 passed**, 1 warning, 21.2s on the last run (an earlier run of the same suite was 17.9s). The warning is Starlette's deprecation notice inside FastAPI's `TestClient` (`httpx` vs `httpx2`). It is not from the bot.

The 21 tests are the starter `test_agent.py` suite (legal actions over hundreds of hands, both seats, aggressive opponent, memory reset, token, malformed body, crash reporting, `PORT`) plus `test_review.py` (ranking rules, table agreement, history replay, the two ranker defects, pool percentiles, exception fallback, river nuts, adversary legality, a 20-hand `_decide` gauntlet).

## `tools/check_build.sh`

Docker is installed and `docker info` succeeded. There is no cached `poker-agent-base` image. `./tools/check_build.sh` was started and got as far as `docker build` of `docker/Dockerfile.base`. The log `/tmp/poker-check-build-20261107.log` stopped printing once pip began `torch` from `https://download.pytorch.org/whl/cpu` (wheel listed as 209.5 MB). The build process was still alive, at low CPU, more than 15 minutes later. Buildkit was not writing further log lines (stdout is not a TTY).

At the time of this report the build had been running for 17 minutes and was still inside that `pip install torch`. No later step was printed. **`check_build` did not finish and is not a pass.** The smoke test (15s startup, sample hands, 1s actions, read-only container) was not run. The HTTP timing above is the substitute, and it is weaker than `check_build.sh`.

## Sample limits

- One deal seed, `review-20261107`. Another seed will move the means.
- 1000 paired deals per opponent. The CI is the judge's paired-deal interval. It is not an independent-hand CI and it was not computed on 2000 separate hands.
- Hero and the exploiter both adapt inside a half, so deals are not strictly independent. The iid standard error is a bit optimistic. The formula is still the one the contest uses.
- Opponents are four scripts. Beating them does not place the bot in the field.
- In-process matches reset memory, not the OS process. For this bot those are the same learning state, because nothing outside `memory` is updated during a half. Startup cost is paid once per process, not once per half, in the in-process runs. The HTTP trace used a fresh process.
- Published in-process chips use a seeded RNG. The file that will be submitted does not.
- Frequency counts for monotone boards used 2000 deals, seed `freq`. The extend-pool count used 80 hands against a pot-raiser.

## Reproducing a single defect

```bash
taskset -c 0 /tmp/poker-venv-review/bin/python -m pytest -q test_review.py -k "draw or extended or nuts or replay"
```
