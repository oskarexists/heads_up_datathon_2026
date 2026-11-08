# Adversarial review: Holey Donut

## Scope

- Repo branch: `adversary-review-20261107`
- Reviewed commit: `5571729ae285bd2e534e0b09c77a83a95ad2899e` (`v5: stronger folded-hands-are-weak assumption`)
- Bot: `PARTICIPANT = "Donut Embezzler"`, `AGENT_NAME = "Holey Donut"` in `agent.py`
- Official starter used for comparison: `https://github.com/tamu-datathon-org/heads-up-bundle` at `c83ab6eeb7660a2fd1fae7df224315d41fe985a9` (cloned to `/tmp/poker-starter-review-20261107`)
- Rules brief: `../poker-brief.md`. Where the brief and the code disagree, `game.py` in the starter is the source of truth. Oskar's `game.py` is byte-identical to that file.
- The first review commit left `agent.py` unchanged. A continuation, authorized by Alex, patches only the three confirmed ranker bugs below. No other strategy rewrite. No push, no PR, no contest submission.
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

Unpatched `agent.py:216` (`5571729`) in `_board_strength`:

```python
flush_draw = ((np.minimum(_SCNT[hidx], 2) + np.minimum(bs, 3)[None, :]) == 4).any(1)
```

`min(hole_suit, 2) + min(board_suit, 3) == 4` is true for one hole card and three board cards. An Omaha flush needs two hole cards of that suit, so a one-card holding is not a flush draw. The bonus added on the flop is `1.0 * 1.3` category units (`agent.py:224-225`), which is larger than a one-pair score. The bonus is used only to rank sampled opponent hands, not to score showdown.

Measured on 2000 deals from `make_deal("freq", i)`: 81/2000 flops (4.1%) are monotone, and 304/2000 turns (15.2%) already have three cards of one suit. On those boards a large share of random five-card holdings have exactly one card of the suit and get the false bonus.

The old reproduction was `test_draw_bonus_flags_hands_that_cannot_draw_with_two_hole_cards` (Ah plus four non-hearts on 9h8h7h).

**Patched.** A flush-draw bonus now requires at least two hole cards of that suit and exactly two board cards of that suit (`agent.py` `_board_strength`). Two hole cards plus three board cards are a made flush and get no extra bonus. Two hole cards plus one board card are a backdoor and get no bonus. One hole card plus three board cards gets nothing. `test_draw_bonus_uses_two_hole_cards_and_no_double_count` asserts the flop bonus is 1.3 only for the real draw.

### 2. Confirmed: straight-draw bonus has the same hole-card hole

Unpatched `agent.py:222-223` flags a straight draw when `min(hole_ranks_in_window, 2) + min(board_ranks_in_window, 3) >= 4` and the made hand is below a straight. One hole rank plus three board ranks qualifies. The counterexample is aces, kings, and a queen on 7c 8d 9h: no pair of hole ranks forms a four-card straight window with the board, and the function still flags a draw (the wheel window treats the ace as one rank beside 7-8-9).

The sum also double-counts a rank that is on both the hole cards and the board.

**Patched, then corrected.** The first patch required two distinct hole ranks and a union of at least 4, and stopped double-counting a shared rank. That rule is not enough. Hole `6c 7d 8h 9s 6d` on the turn `Ac Kd Qh Ks` has four hole ranks in 6-T and those ranks plus the ace in the wheel, so `hw >= 2` and `union >= 4` fired. The turn scale is 0.6, so the bonus was `0.48` (`_board_strength` 3.087 versus made `_VAL` 2.607). An exhaustive check of all 27 remaining rivers and every 2-hole + 3-board subset finds no straight and no straight flush. Three hole ranks and fewer than two board ranks cannot supply the three board cards a straight needs.

The rule now also requires at least two distinct board ranks in that same window. Union 3 stays a backdoor with no bonus. The made-hand gate is unchanged. `test_straight_draw_requires_two_board_ranks_in_the_window` locks the reported hand at bonus 0 and zero straight outs, and locks a real turn draw (two board ranks, four-rank union, not yet a straight) at `0.48`. The flop double-count case is still bonus 0 and the real flop draw is still 0.8. The head-to-head in `review_results/patch_campaign/` is the first patch only. It does not measure this correction. The later section does.

### 3. Confirmed: extended pools keep batch-local percentiles

Unpatched `_extend_pool` (`agent.py:275-282` at `5571729`) builds a new sample, ranks street strength inside that sample only, and concatenates those percentiles onto the old ones. Old rows are not re-ranked. Preflop percentiles come from the global `_PRE_PCT` table and do stay comparable. Flop, turn, and river percentiles do not.

The old test extended a 30-row flop pool by 30 and checked that the stored vector differed from `_percentile` of the combined sample by more than 0.05.

This runs when the weighted range is narrower than the target. Against an opponent that pot-raises whenever legal, `_extend_pool` was called 121 times across 250 hero decisions in 80 hands (`make_deal("ext", ...)`). Facing aggression is exactly when the range model is supposed to matter.

**Patched.** Concatenation keeps `draws`, `pidx`, `hidx`, `win`, and the global preflop percentile. Street percentile keys are dropped. The next `_street_pct` ranks the combined pool. `test_extended_pool_percentiles_are_ranks_in_the_full_sample` asserts that. The rerank cost is in the patch section below.

### 3b. Confirmed, and fixed: the cheap path dropped opponent actions

Independent audit of `5571729` (`../independent-final.log`): if `_spent` is over 96% of the bank, `_decide` used to set `hist_n` to the end of the history and return before `_replay`. Later decisions only count `i >= hist_n`, so those opponent actions never entered `stats`. This runs only after the bot has already switched to `_cheap_decide`. It does not itself return an illegal action. `choose_action` still falls back to check, or fold if chips are owed.

**Patched.** The cheap path now calls `_replay` before `_cheap_decide`, so each new opponent action is counted once. A later call, including one that has left the cheap path, does not count it again. `test_emergency_path_counts_opponent_actions_once` forces `time_used` to 97% of the bank, checks an opening raise is stored as one `pre_open` observation, and checks the count stays 1.

### 4. Exceptions stay off the action JSON, and they are now counted

Tournament default (`AGENT_RAISE_ERRORS` unset): `choose_action` and `on_hand_end` log `traceback.print_exc` to stderr, increment `memory["errors"]`, and still return a legal action (check, or fold if facing a bet) or finish the hand-end call. Dev and tests set `AGENT_RAISE_ERRORS=1` to re-raise after the counter increments. `new_half` resets both counters. The stderr line does not dump the environment. A successful `/act` body stays `{"action": ..., "hand_end_errors": []}` with no traceback. Tests: `test_failures_log_or_reraise_and_reset_each_half`, `test_action_json_has_no_traceback_when_decide_fails`.

### 4b. Confirmed behavior, low tournament risk: the judge still does not see a raised error by default

`choose_action` (the `try` around `_decide`) turns any exception into check, or fold if facing a bet. `on_hand_end` swallows its own exceptions, so a learning bug does not increment the judge's error counter. Five real errors forfeit the half, so the wrapper is deliberate. After the ranker patch those handlers are `agent.py:838` and `agent.py:886`.

`test_decide_exception_does_not_escape_choose_action` locks the fallback. Across the 8,000-hand duplicate run below, the wrapper around `_decide` saw 0 exceptions, and the HTTP half reported 0 protocol errors. Nothing in this review is currently taking the fallback. The risk is a future bug that check/folds silently while `pytest` still looks clean if it only checks HTTP status.

### 5. Checked, and not bugs

- Dense Omaha ranks from `_best` agreed with `omaha_score` ordering on 300 random pairs in the diagnostic and 40 pairs in `test_agent_rank_table_matches_omaha_score_order`. Unique five-card scores: 1404, so the int16 table does not overflow.
- The history walk's reconstructed pot and `to_call` matched `Game` on mixed pot raises, half raises, and folds (`test_replay_pot_and_to_call_match_the_engine`, 25 hands; diagnostic 190/190 decisions).
- A river straight flush (As Ks with Qs Js Ts on board, hero to call a pot bet) was not folded (`test_river_nuts_are_not_folded`).
- Startup to `GET /` on a cold process: **1.57s** (limit 15s). Import of `agent.py` on one CPU was 1.47s in the diagnostic.
- No illegal action was returned in the duplicate matches or the existing `test_agent.py` gauntlet.

### Strategy hypotheses (not confirmed defects)

- `fold_k` (`_P` and `_equity_when_called`) treats folded hands as weaker than the current equity. That can drive equity-when-called toward 0. It did not stop the unpatched bot from beating a calling station, which is the opponent that punishes missed value. That is evidence against "the bot never value-bets," not proof the constant is well calibrated. The design section below works an example.
- The bluff cap, junk open-fold (`equity < 0.37` on the button), and showdown range calibration are policy choices. This review did not isolate them.
- `new_half` uses `np.random.default_rng()` with no seed. Tournament runs are stochastic. The harness replaces `memory["rng"]` after `new_half` so the published numbers repeat. Repeating the HTTP server will not repeat those chips.
- The internal clock adds `0.0004s * load` per call (`_OVERHEAD`) and does not subtract the judge's 1ms grace. On this host the unpatched bot's own `time_used` finished near 4s while the charged HTTP bank was 2.03s, and `scale` sat on the 1.6 cap. The bot is leaving sample budget unused here. A slower contest CPU could go the other way. `load` at the end of the halves was about 0.6 to 1.3 versus the author's reference draw, so this machine is in the same band as the development machine, not a proof about the contest VM.

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

No fixed candidate was run in that pass. The ranker bugs above are real, and the unpatched bot still won these matches. The next section measures the patched bot against that same binary (`git show 5571729:agent.py`). Those files are untouched.

## Patched bot versus the original

Command (one CPU, existing venv, baseline outside the repo):

```bash
cd /home/alexn/Documents/code/datathon_2026/poker-bot-tournament/oskar-review
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  taskset -c 0 /tmp/poker-venv-review/bin/python -u review_harness.py \
  --deals 1000 --seed review-patch-20261107 \
  --baseline /tmp/agent_original_5571729.py \
  --hero-label patched --baseline-label original \
  --opponents mono_board,extension_press,fold_bait,maniac,calling_station,min_raise,exploiter \
  --out review_results/patch_campaign
```

`/tmp/agent_original_5571729.py` is `git show 5571729:agent.py` (1002 lines). The working tree is the patched bot. Each half calls `new_half` on a fresh dict, then replaces `memory["rng"]`. Head-to-head seeds are `adler32(f"{seed}:{half}:patched")` and `adler32(f"{seed}:{half}:original")`. Script matches use `adler32(f"{seed}:{half}:{opponent}:{hero_label}")`, so the patched-versus-script rows and the original-versus-script rows are not a paired comparison of the two heroes. The only paired comparison of the two bots is the head-to-head. Positive means the patched bot gained chips across the two seats. Same CI formula as above, `n = 1000`, `big_blind = 2`.

The CI was recomputed from the CSV `paired_bb` column with `statistics.stdev` (`ddof=1`) and `1.96 * SE`. It matched `patch_comparison.json` on every row. Chip sums divided by `2 * 1000` matched the means, including the head-to-head mean of 0.1525. Illegal actions: 0. Decide exceptions: 0.

Rerank microbench, same process, 30 repeats, flop strength on 400 rows versus 800 rows: 0.0989 ms versus 0.1381 ms. Extra cost of ranking the old rows again: **0.039 ms** per extension of a 400-row pool to 800. The earlier count was 121 extensions in 80 hands. That is well under a millisecond per hand on this host, next to the Monte Carlo. It is not a time-bank problem here.

| Matchup | Mean bb/deal | SD | SE | 95% CI | Chips half1, half2 | Interval includes 0 |
|---|---:|---:|---:|---|---:|---|
| patched vs original | +0.1525 | 14.344 | 0.454 | [-0.737, +1.042] | +1783, -1478 | yes |

The two seats point in opposite directions (+1783 and -1478). `decide_verdict` would call this a chop. **This is not a demonstrated win-rate improvement.**

Script matches, same seed, hero named in the row. These probes and the four earlier scripts all lost to both bots. They are targeted probes, not proven exploits.

| Opponent | Patched mean | Patched 95% CI | Original mean | Original 95% CI |
|---|---:|---|---:|---|
| mono_board (pot-raise on monotone or connected flops, else call) | +9.589 | [+8.269, +10.909] | +9.541 | [+8.236, +10.845] |
| extension_press (call preflop, pot-raise every postflop decision) | +10.314 | [+8.737, +11.890] | +10.932 | [+9.372, +12.491] |
| fold_bait (fold for 40 hands, then pot when checked to and never fold) | +8.213 | [+6.694, +9.731] | +8.354 | [+6.833, +9.875] |
| maniac | +6.287 | [+4.794, +7.780] | +6.218 | [+4.677, +7.759] |
| calling station | +12.220 | [+11.206, +13.233] | +12.191 | [+11.141, +13.240] |
| min-raise | +8.109 | [+6.697, +9.520] | +8.322 | [+6.923, +9.720] |
| exploiter | +6.216 | [+4.698, +7.734] | +6.361 | [+4.872, +7.849] |

Every one of those fourteen intervals is entirely above 0, so each script match would be a hero win under `decide_verdict`. The gaps between patched and original on a given script are a few tenths of a bb/deal and the hero RNGs differ, so those gaps are not a paired effect. Raw rows: `review_results/patch_campaign/patched_vs_original.csv`, `patched_vs_probes.csv`, `original_vs_probes.csv`. Aggregates: `review_results/patch_campaign/patch_comparison.json`.

Those rows are the first draw patch (`hw >= 2` and `union >= 4`, no board-rank minimum). Do not read them as the result of the two-board-rank correction.

## Straight-fix versus the original

The corrected straight rule, the cheap-path history parse, the preflop showdown-call fix, and the error counters are in this tree. Baseline is still `git show 5571729:agent.py`. Same paired formula, `n = 1000`, fresh `new_half` memory, seeded RNG. Hero label `straightfix`, so this is not the same RNG stream as the table above. Positive means the corrected bot gained across both seats. Raw files are under `review_results/straight_fix_campaign/` and do not replace `patch_campaign/`.

A first run of this command was discarded and not kept. Called preflop all-ins reached `on_hand_end` with street 0, `_count(..., "call", ...)` indexed a postflop table, and the traceback aborted the rest of that hand's update. Those hands are now recorded as `call_raise`. The rerun log has zero `agent on_hand_end failed` lines and zero `agent choose_action failed` lines.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  taskset -c 0 /tmp/poker-venv-review/bin/python -u review_harness.py \
  --deals 1000 --seed review-straightfix-20261107 \
  --baseline /tmp/agent_original_5571729.py \
  --hero-label straightfix --baseline-label original \
  --opponents "" --out review_results/straight_fix_campaign
```

CI recomputed from `patched_vs_original.csv` with `statistics.stdev` (`ddof=1`) and `1.96 * SE`. It matches the JSON. Chip sum `(21 + 1182) / 2 / 1000 = 0.6015`. Illegal actions: 0. Decide exceptions: 0. Wall 14.8s.

| Matchup | Mean bb/deal | SD | SE | 95% CI | Chips half1, half2 | Interval includes 0 |
|---|---:|---:|---:|---|---:|---|
| straight-fix vs original `5571729` | +0.6015 | 14.231 | 0.450 | [-0.281, +1.484] | +21, +1182 | yes |

`decide_verdict` would call this a chop. **This is not a demonstrated win-rate improvement.** The first patch's chop and this chop are different seeds and different code. Script probes were not re-run after the two-board-rank correction. Offline pair-by-triple files and the preflop orbit table were not built in this chunk.

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

Result after the straight-draw correction, the cheap-path history fix, and the preflop showdown-call fix:

```text
26 passed, 1 warning in 17.63s
```

The warning is Starlette's deprecation notice inside FastAPI's `TestClient` (`httpx` vs `httpx2`). It is not from the bot. The unpatched suite was 21 passed. The five additions are the corrected draw tests' replacement coverage, the emergency-path count, loud errors, action JSON, and the preflop showdown call.

The 26 tests are the starter `test_agent.py` suite (legal actions over hundreds of hands, both seats, aggressive opponent, memory reset, token, malformed body, crash reporting, `PORT`) plus `test_review.py`. The ranker tests assert the corrected bonuses, including two board ranks for a straight draw, and the full-pool percentile. They no longer assert the defects. Adversary legality covers the three probes.

## `tools/check_build.sh`

Docker is installed and `docker info` succeeded. There was no cached `poker-agent-base` image, so `./tools/check_build.sh` built `docker/Dockerfile.base` from scratch. Pip did install CPU `torch==2.15.0+cpu` (209.5 MB wheel, about 126 kB/s). While BuildKit was exporting the finished base layers, Docker stopped the build:

```text
#12 exporting layers 8.9s done
#12 CANCELED
ERROR: failed to build: failed to solve: Canceled: context canceled
CHECK FAILED: the base image did not build (this is not your bot's fault; tell the organizers)
EXIT:1
```

Log: `/tmp/poker-check-build-20261107.log`. The script's own message says the failure is the base image, not the bot. The agent image was never built, and the in-container smoke test never ran. The coordinator intentionally canceled that build after it had stalled for more than 19 minutes. The log line `Canceled: context canceled` is that cancellation, not a spontaneous defect found in the bot or in Docker's layer export. That run is **unfinished/terminated, not a pass.**

Later, on commit `527d20a32e91e8b71b36eda6f567d6917e7e847a`, the coordinator ran official `tools/check_build.sh`. Log: `../check-build-527d20a.log`. The base image was cached. The bot image added **6 MB** (cap 3072 MB). The smoke container is how that script always starts the bot: `--read-only`, `--network none`, `--tmpfs /tmp`, `--cap-drop ALL`, image user `10001` (non-root). `GET /` answered in **2.0 s**. Five `/act` calls, slowest **22 ms**. `SMOKE TEST PASSED`. `CHECK_BUILD_EXIT=0`. That pass is for `527d20a` before the lookup files below. It is not a pass for later commits until `check_build` is run again.

## Shipped lookup tables

`lookup_tables/` holds `tbl`, `val`, `pre_pct`, `rmask`, `scnt`, and `straight_rank` as NumPy `.npy` files plus `manifest.json` (`holey-donut-lookup-v1`). No pickle. Startup checks format, dtype, shape, nbytes, and sha256 of each file, then `np.load(..., allow_pickle=False)`. A mismatch raises. Loading does not write. `AGENT_BUILD_TABLES=1` is only for the generator.

Regenerate:

```bash
/tmp/poker-venv-review/bin/python tools/generate_lookup_tables.py
```

Measured on this host, one CPU: rebuild **0.848 s**, file reload **0.016 s**, fresh `import agent` **3.666 s** (that import still runs `evaluator.warmup()`, which builds the showdown dict). Tracemalloc peak for that import **81.6 MB**. Uncompressed array bytes **15,039,512** including the 8-byte straight-rank vector. The five policy arrays the bot already kept in memory are **15,039,504** bytes (14.34 MiB). Largest file `tbl.npy` is 9.0 MB, under 100 MB. No Git LFS.

`test_shipped_lookup_matches_a_fresh_build_and_is_read_only` rebuilds in the test process and requires bit equality with the loaded arrays, and it checks that a tampered file fails the hash. Same arrays means the same percentile order and the same actions for a fixed RNG. Pytest on the pushed lookup commit `87fde9e`: **27 passed**. Official `check_build` on that exact snapshot passed: startup **1.0 s**, image **+36 MB**, log `../check-build-87fde9.log`. That pass does not cover the equity-table default below. The candidate that loads the table has its own `check_build`, in the preflop section.

## Ten seeds against the original

The coordinator ran 10 seeds × 1000 paired deals of stable `87fde9e` versus `5571729` in `../multi-seed-87fde9`. This tree did not rerun or edit that campaign. Snapshot: `review_results/multi_seed_87fde9/aggregate.json`.

Pooled mean **+0.1994** bb/deal on 10,000 paired deals. Pooled iid judge 95% CI **[-0.082309, +0.481109]**. Seed-cluster interval (10 seeds, t with 9 df) **[-0.091631, +0.490431]**. Both include 0. **This is not a demonstrated win-rate gain.** Per-seed means in that file run from −0.149 to +1.2175. The printed seed-09 interval is entirely above 0; the other nine printed intervals include 0. Illegal actions 0. Decide exceptions 0. Runtime policy matches `527d20a` except that startup loads the lookup files.

## Preflop suit orbits

A first-appearance suit label is not a suit orbit. It sorted cards by id, so suit 0 was renamed before suit 1, and `canon_key([0,1,4,8,12])` was `(0,1,4,8,12)` while the suit-0↔1 image `[1,0,5,9,13]` was `(0,1,5,9,13)`. The census that reported **35,334** orbits, and the 323×2048 pilot metrics computed from it, are **not suit-disjoint and are not evidence**. Those files were deleted. Nothing below uses them.

The key is now the lexicographic minimum of the sorted 5-card id under all 24 suit permutations. Burnside's count is `(376992 + 6*16560 + 8*450) / 24 = 19998`. `test_suit_orbits_match_burnside_and_every_suit_perm` builds all `C(36,5)` holdings and checks both of these, exhaustively: reordering the 24 image keys by the group action preserves the orbit id, and recomputing the lex-min on every permuted holding preserves it too. The counterexample pair is one orbit. Orbit sizes divide 24 and sum to 376992.

Equity samples are fresh permutations of the 31-card deck (`Generator.permuted`), one stream per `(orbit, stage)` from `SeedSequence([20261107, orbit, stage])`. They do not read `_PERM_POOL`. Ties are 0.5. Showdown uses the shipped dense pair×triple table (`agent._best`). `test_dense_lookup_matches_omaha_score_on_random_deals` requires the win/tie/loss to match `evaluator.omaha_score` on 600 random worlds. The table job repeats that check on 2000 worlds before it samples, and it refuses to start if any world disagrees.

The full table is staged, all 19998 orbits each time: 1024 samples, then 8192, then 38416. 38416 is the worst-case pointwise n for a 95% half-width of 0.005 (`1.96 * 0.5 / sqrt(n)`). That is not a simultaneous band and not an absolute-error guarantee. The job writes `review_results/preflop_equity/progress.json` and atomic per-worker checkpoints under `chunks/`. A finished stage is `stage_{n}.npz` with that stage's increment only. `test_equity_checkpoint_resume_matches_one_shot` checks that a stopped run and a different batch size reproduce the same counts. Checkpoints stay on disk and are not part of the artifact commit.

The finished `progress.json` had counted the last stage twice. Worker files for stage 2 store `30224` new worlds per finished orbit (`38416 - 8192`). The final snapshot passed an increment of 0, and the old total added `N_ORBITS * 38416` to those worker samples: `1372662720` worlds, `178.6755518533944%`. The repair counts each stage's increment once: `19998 * 38416 = 768243168` unique worlds, `100%`. `test_progress_unique_worlds_do_not_double_count_final_stage` builds that `n_this=0` snapshot on a toy schedule and requires 160, not 240. `test_saved_equity_progress_counts_each_world_once` locks the repaired file. The job's own elapsed time is still **330.06 s**. The rate implied by the unique count is about **2.328e6** worlds/s. The double-counted total is kept on the file as `samples_done_double_counted_final_stage` so the percent is not a clamp of a wrong numerator.

Pointwise estimated 95% half-widths on the merged table: min **0.00298939**, median **0.00485860**, max **0.0050000114**. The strict `half <= 0.005` flag is false because the max sits `1.14e-11` over 0.005. `n` is exactly 38416, which is the n where `1.96 * 0.5 / sqrt(n) = 0.005`. That is still not a simultaneous or absolute guarantee.

## Equity table in the bot

`tools/preflop_equity_pilot.py publish` writes `lookup_tables/equity_{pct,mean,wins,ties,n}.npy` and `equity_manifest.json` (`holey-donut-equity-v1`). Numeric `.npy`, `allow_pickle=False`, no pickle, no Git LFS. Startup checks format, dtype, shape, nbytes, and sha256. Loading does not write. A bad hash or version raises (`test_equity_manifest_rejects_a_bad_hash_and_version`). There is no runtime generation.

The percentile is the multiplicity-weighted rank of the orbit means. Orbit size is the weight, so a size-24 orbit counts 24 times and a smaller orbit counts its own size. Equal means, including every suit image of one orbit, share the average rank of that tie. The value does not follow stable-sort order inside the tie (`test_weighted_percentile_uses_multiplicity_not_suit_order`). The array is scattered with `_hand_idx` (colex of the sorted cards). Combination-lex row is not that index: only four of the 376992 rows match, and the publish step refuses to continue if they are identical. `test_published_equity_table_is_suit_invariant_and_hashed` checks all 24 suit images of every holding.

`AGENT_PREFLOP` selects the source. `equity` (the default) uses `equity_pct.npy`. `proxy` uses the shipped formula percentile `_PRE_PCT`. `use_preflop` rebinds the process and sets the env var. `test_preflop_sources_stay_selectable_and_default_is_the_table` starts a fresh interpreter with the variable unset and again with `AGENT_PREFLOP=proxy`. The default was switched to the table only after the loader tests passed. Postflop ranking, the showdown sampler, and the rest of the policy are unchanged. `_make_pool` reads `_RANGE_PCT`, which is whichever source was selected.

## Held-out worlds against the frozen formula

`heldout 1000 38416` drew 1000 orbits with `SeedSequence([20261107, 424242, 100])` and, for each, 38416 new deals from `SeedSequence([20261107, orbit_id, 100])`. Generation used stages 0, 1, and 2 only. Stage 100 is not one of them. Opponent 5 and board 5 are a uniform permutation of the other 31 cards. Ties count 0.5. Showdown is the dense 2+3 table. These fresh means are another Monte Carlo estimate. They are not exact equity and they are not a replay of the table.

On those 1000 orbits the fresh-minus-table RMSE is **0.00359486** (MAE **0.00289074**). Multiplicity-weighted RMSE **0.00362420** (MAE **0.00292371**). The sampling noise floor, `sqrt(mean(se_fresh^2 + se_table^2))` from the outcome variances, is **0.00349484** (weighted **0.00349677**). The observed RMSE is 1.029 times that floor. Worst-case SE of a difference of two independent means at n=38416 is `0.5 * sqrt(2/38416) = 0.00360769`. The table matches a disjoint sample at the noise floor. That is not a proof that either mean equals the true equity.

The frozen formula (`_PRE_COEF` dot `_preflop_features`, not refit) against the same fresh means has RMSE **0.0152750** and MAE **0.0118428** (weighted RMSE **0.0142493**, MAE **0.0112123**), about 4.37 times the noise floor. A suit swap of each representative moves the formula by 0. Against the noisy fresh means on these 1000 orbits, that RMSE is an R^2 of about **0.951**. The source comment's R^2 of 0.89 is a different fit statement and was not remeasured as a claim about that comment. Formula-versus-table on the same orbits (RMSE **0.0152543**) describes the fit to the stored labels. It is not a second held-out truth. Records: `heldout_summary.json`, `heldout_orbits.npz`. Elapsed **20.8 s**.

## Ten seeds, table versus proxy

Hero is this tree with `AGENT_PREFLOP=equity` (`preflop_source` is `equity_table`). Baseline is `../verification-87fde9/agent.py`, the `87fde9e` snapshot: same basic fixes, its own lookup files, and the shipped proxy only. The harness aborts unless those two source names are the ones just stated. The percentile arrays are not equal. Each seed is one process, `taskset -c 0`, 1000 paired deals, two halves, a new memory dict and an 8.0 s bank each half. Hero internal time was **3.971 s to 4.232 s** per half. Decide exceptions **0**. Illegal actions **0**.

Pooled mean **−0.0086** bb/deal on 10,000 paired deals (20,000 hands). Pooled iid judge 95% CI **[−0.312768, +0.295568]**. Seed-cluster 95% interval (10 seed means, t with 9 df, critical value 2.262157) **[−0.407072, +0.389872]**. Both include 0. **This is not a demonstrated win-rate change.** Per-seed means: −0.7265, −0.0805, +0.1035, −0.654, −0.245, −0.1175, +0.9115, −0.364, +0.2325, +0.854. Raw rows: `review_results/table_vs_proxy/paired_all.csv` and `seed-XX/patched_vs_original.csv`. Summary: `aggregate.json`.

The match harness no longer writes the old sentence that called `check_build` an unfinished current failure. That earlier base-image cancellation stays history in the section above. This harness does not measure `check_build`.

Official `./tools/check_build.sh` on this candidate, base image cached: **EXIT 0**. Image addition **38 MB** (cap 3072 MB). The smoke container is `--read-only`, `--network none`, `--tmpfs /tmp`, `--cap-drop ALL`, user `10001`. `GET /` in **1.0 s**. Five `/act` calls, slowest **31 ms**. `SMOKE TEST PASSED`. Log: `/tmp/check-build-equity-table.log`.

Pytest after the default switch: **38 passed**, 1 Starlette deprecation warning, 25.7 s. The 38 tests are the previous 27 plus the orbit, checkpoint, equity-loader, progress, percentile, and seed-cluster checks.

Reproduce:

```bash
/tmp/poker-venv-review/bin/python -m pytest -q
/tmp/poker-venv-review/bin/python tools/preflop_equity_pilot.py publish
/tmp/poker-venv-review/bin/python tools/preflop_equity_pilot.py heldout 1000 38416
# one seed; repeat 01..10. Do not set AGENT_PREFLOP in the shell.
taskset -c 0 /tmp/poker-venv-review/bin/python review_harness.py \
  --baseline ../verification-87fde9/agent.py --opponents '' --deals 1000 \
  --seed table-proxy-20261107-01 --hero-label equity-table \
  --baseline-label shipped-proxy --hero-preflop equity \
  --expect-hero-source equity_table --expect-baseline-source shipped_proxy \
  --out review_results/table_vs_proxy/seed-01
./tools/check_build.sh
```

Limits of this comparison: the held-out set is 1000 of 19998 orbits, not every orbit. Both Monte Carlo means have error of a few thousandths; agreement at the noise floor is not exact equity. The formula's 0.015 RMSE is on the equity scale (full-table orbit means run from 0.099 to 0.693), and the paired match is the test of whether that difference changes bb/deal. Ten seeds do not pin a difference as small as 0.01 bb/deal. In-process halves reset `memory` and the 8 s bank; they do not open a new HTTP server per half. The smoke test is the read-only startup measurement. The 10-seed campaign of `87fde9e` versus `5571729` was not rerun. Worker checkpoints under `review_results/preflop_equity/chunks/` were kept and left untracked.

The coordinator independently rechecked commit `05e830fe0e0eb2617185849a06c8a8570516f8cd`: 38 tests, `check_build` startup 1 s and image addition 38 MB, and runtime hashes equal to the expanded-campaign snapshot. That commit is pushed. The notes below are documentation of campaigns already finished. They do not change the bot, the generator, the harness, or the default (`AGENT_PREFLOP` unset still loads the equity table; `AGENT_PREFLOP=proxy` still loads the shipped formula percentile).

## Expanded table campaign

The coordinator ran this. It was not rerun here. Snapshot: `review_results/expanded_table_campaign/`. Runtime files are byte-identical to `05e830f` (`runtime_snapshot.json`, 0 hash mismatches on recheck). Hero source was forced to `equity_table`. Baselines were asserted `shipped_proxy`. One CPU per process, pinned to distinct CPUs 1–7. Observed decide exceptions 0. Illegal actions 0. Each primary row is 10 seeds × 1000 paired deals (10,000 paired deals, 20,000 hands). Nine primary matchups are 90,000 paired deals. Positive mean means the table bot gained chips. Intervals are per matchup and are not adjusted for multiple comparisons. The pooled interval treats paired deals as iid and uses z=1.96. The seed-cluster interval uses the 10 seed means and t with 9 df.

| Opponent | Mean bb/deal | Pooled 95% CI | Seed-cluster 95% CI |
| --- | ---: | --- | --- |
| calling_station | +12.33000 | [+11.999491, +12.660509] | [+11.902522, +12.757478] |
| exploiter | +5.42085 | [+4.930226, +5.911474] | [+4.754627, +6.087073] |
| extension_press | +10.72620 | [+10.233408, +11.218992] | [+10.274512, +11.177888] |
| fold_bait | +8.61515 | [+8.139471, +9.090829] | [+8.190226, +9.040074] |
| maniac | +5.52935 | [+5.040465, +6.018235] | [+4.857982, +6.200718] |
| min_raise | +7.78220 | [+7.324472, +8.239928] | [+7.245098, +8.319302] |
| mono_board | +10.07335 | [+9.663529, +10.483171] | [+9.533404, +10.613296] |
| original557 | −0.22365 | [−0.525252, +0.077952] | [−0.415375, −0.031925] |
| patched_formula87 | +0.24415 | [−0.065341, +0.553641] | [−0.057453, +0.545753] |

The seven script rows are all positive, and both of their intervals sit above 0, so those scripts lost chips to the table bot on this design. That is not evidence about a strong field. The same seven scripts also lost to the patched formula bot; those rows are `additional_formula_probe_matchups` in `aggregate.json` and are not part of the nine-row matrix.

`original557` is the row that needs both intervals read together. The mean is **−0.22365** bb/deal (the table bot lost chips). The pooled interval **[−0.525252, +0.077952]** includes 0. The seed-cluster interval **[−0.415375, −0.031925]** does not: it lies entirely below 0. The deal-level interval and the seed-mean interval disagree about whether 0 is inside the band. The cluster band on the negative side is not a win for the table. **This does not show that the table beats the original.** `5571729` still has its original silent error path. A harness count of zero decide exceptions does not prove that original `on_hand_end` never failed quietly. No runnable small-model opponent exists. An earlier MLP pilot used an invalid canonicalization and was not deployed, so it is not a result.

`patched_formula87` in this matrix uses the expanded seeds `table-expanded-20261107-01` … `10`, mean **+0.24415**, and both intervals include 0. That is a different draw from the earlier `table-proxy-20261107` ten seeds (mean −0.0086, both intervals include 0). The two ten-seed sets are disjoint. Combining them, already computed in `combined_table_vs_formula_20seeds.json`, is the check against quoting only one of them: **20 seeds, 20,000 paired deals, 40,000 hands, mean +0.117775**. Pooled 95% CI **[−0.099196, +0.334746]**. Seed-cluster 95% CI (t with 19 df) **[−0.115287, +0.350837]**. Both include 0. **Neither the separate tens nor the combined twenty shows a win-rate gain of the table over the patched formula.**

Raw primary rows, 160,000 paired records: `paired_all.csv`. Commands: `completed_jobs.json`. Per-seed logs were not copied.

## Independent preflop validation

The coordinator also ran a second 1,000-orbit check, copied to `review_results/independent_preflop_validation/`. It is not a rerun of the held-out section above. Selection seed `997114291`, world namespace `91988771`, 16 workers, 18.1 s. Each of 1,000 true orbits got 38,416 new worlds: **38,416,000** additional worlds. The new worlds use a different RNG namespace from table generation. The orbits themselves were not held out of the table. The stored table still covers all 19,998 orbits. The frozen formula is the `87fde9e` coefficient vector; its original training sample is unknown.

Against those independent Monte Carlo means, table RMSE is **0.0033948475553** (0.3395 percentage points) and formula RMSE is **0.014768157975** (1.4768 percentage points). Multiplicity-weighted RMSE is **0.003370387633** for the table and **0.013934511438** for the formula. A noise-adjusted table RMSE estimate, subtracting estimated validation variance only, is **0.0023283**. That adjustment is an estimate, not a confidence guarantee, and it does not remove table-generation variance. The independent means are Monte Carlo targets, not exact equity. These figures sit next to the earlier held-out RMSE (table 0.003595, formula 0.015275). They do not promote the table to an exact equity oracle, and they do not by themselves imply a win-rate gain. Per-orbit rows: `per_orbit.csv`. Summary: `summary.json`.

## Sample limits

- One deal seed for the unpatched table (`review-20261107`) and one for the patch table (`review-patch-20261107`). Another seed will move the means.
- 1000 paired deals per opponent. The CI is the judge's paired-deal interval. It is not an independent-hand CI and it was not computed on 2000 separate hands.
- Hero and the exploiter both adapt inside a half, so deals are not strictly independent. The iid standard error is a bit optimistic. The formula is still the one the contest uses.
- Opponents are scripts. The three new ones are targeted probes (monotone or connected pot-raise, postflop pot pressure, fold-then-reverse). All three lost to both the patched bot and the original. That does not prove the ideas are harmless against a stronger opponent, and it does not place the bot in the field. The head-to-head interval includes 0.
- In-process matches reset memory, not the OS process. For this bot those are the same learning state, because nothing outside `memory` is updated during a half. Startup cost is paid once per process, not once per half, in the in-process runs. The HTTP trace used a fresh process.
- Published in-process chips use a seeded RNG. The file that will be submitted does not.
- Frequency counts for monotone boards used 2000 deals, seed `freq`. The extend-pool count used 80 hands against a pot-raiser.

## How the bot actually decides

Line numbers are the patched `agent.py`. Constants that the patch did not touch match `5571729`.

Import loads a dense pair-by-triple rank table from `lookup_tables/` (numeric `.npy`, `allow_pickle=False`). `tools/generate_lookup_tables.py` is what calls `_build_tables` against `evaluator.get_table()`. Startup does not rebuild it. The default preflop label is now the equity-table percentile (`AGENT_PREFLOP=equity`, or unset). `AGENT_PREFLOP=proxy` selects the 16-coefficient formula percentile instead. `_PRE_COEF` is applied in `_preflop_strength`. The comment in source says it was fitted offline to Monte Carlo equity with R^2 = 0.89. The held-out section reports a different comparison and does not replace that comment. The sixteen features, in order, are: rank sum / 40, max rank / 8, ace count / 2, pair count / 2, highest pair rank, a trips flag, number of suits with at least two cards / 2, a three-suited flag, an ace-suited flag, a king-suited flag, the fullest straight window / 5, count of windows with at least 3 ranks / 6, count with at least 4 / 6, distinct ranks / 5, adjacent-rank popcount / 4, and an intercept. `_PRE_PCT` is that score's percentile over every 5-card holding. The equity percentile is the multiplicity-weighted rank of the Monte Carlo orbit means, stored in `equity_pct.npy`. Neither number is conditioned on a specific opponent combo. `_make_pool` removes the hero's cards and the board from the deck before it draws an opponent holding, so the Monte Carlo does apply that card removal. The weight on a sampled hand is `_RANGE_PCT` of the opponent's five cards (the selected preflop source), or `_board_strength` of that holding plus the board. Removing dead cards and pricing suit blockers against a continuing range are different things.

Postflop, `_make_pool` draws opponent holdings from the cards that are not in the hero's hand or on the board, plus a private run-out, from a fixed permutation table (`_PERM_POOL`, seed `12345 + deck size`). `_board_strength` ranks those holdings by the made Omaha category plus, before the river, 1.3 for a flush draw and 0.8 for a straight draw. A straight draw also needs two distinct board ranks in the window. The scale is 1.0 on the flop and 0.6 on the turn. That rank is turned into a within-sample percentile by `_percentile` / `_street_pct`. It is not the showdown score. Showdown equity is a separate Monte Carlo: `_fill_wins` deals the stored run-out and compares `_best` for hero and opponent.

`_range_weights` (`agent.py:533-547`) multiplies a soft top-percentile weight (`_top`) for bets, raises, and calls, and a soft cap (`_cap`) for checks. The cutoff `x` comes from `_x_for` (`agent.py:411-433`): the observed frequency of that action, with a floor (0.15 / 0.08 / 0.20 depending on the action), then a shrink toward showdown calibration. Frequency is a Beta-style average with the priors in `_PRIORS` (`agent.py:327-328`): 6 pseudo-hands preflop (`_f_pre`) and a street count shrunk toward 8 pseudo-hands of the pooled postflop rate (`_f_post`). Counters decay by `_P["decay"] = 0.995` each hand (`_decay_stats`).

Showdown calibration (`_calibrate`, `agent.py:502-520`) runs only when `went_to_showdown` and `opp_hole` is present. It records the revealed hand's percentile on each street where the opponent bet, raised, or called. `_x_for` treats a top-`x` range as having mean percentile `1 - x/2`, inverts the observed mean, and mixes that in with weight `sd_weight[street] * n / (n + 12)`. `sd_weight` is `(0.6, 0.4, 0.6, 0.8)` for streets 0..3. Folded hands never enter this average. The source comment says showdown hands are biased strong and that the mix is only partial. That bias is why the weight exists. It does not remove the selection.

EV (`_decide`, `agent.py:715-753`). Fold is 0. A check is `equity * pot` after subtracting `call_margin` `(0, 0.05, 0.04, 0.02)` and, out of position before the river, `oop_pen = 0.02`. A call of `c` chips is `equity * (pot + c) - c` with the same haircut. A raise uses an estimated fold probability `fa`: preflop `pre_fold` times `0.7 + 0.4 * min(1, size)`; postflop the empirical `fold_s` or `fold_b` rate. Size class is `b` when the bet exceeds 0.65 of the pot (`_size_class`), else `s`. Equity when called is

```text
e_fold = e + fold_k[street] * (1 - e)
e_called = clip((e - fa * e_fold) / (1 - fa), 0, 1)
```

with `fa` capped at 0.90 inside the helper and at 0.95 before it, and `fold_k = (0.12, 0.30, 0.35, 0.40)`. The assumption is that folded hands are `fold_k` of the way from current equity up to 1, so the hands that continue are worse than the average. Realization then subtracts `(0, 0.05, 0.03, 0)` on a called bet. A guessed re-raise rate moves EV by `(e_called - 0.5) * 1.5 * chips` (`reraise_coef`). A `raise_margin` of 0.03 pot prefers checking when the raise is only a little better. Noise of 0.02 pot breaks exact ties.

Bluff cap (`agent.py:757-763`): if the chosen action is a raise and equity is below `bluff_lo = 0.30`, the bot bluffs with probability `min(0.75, max(0, (fa - 0.30) * 2.0))` and otherwise takes the better of fold and check/call. Button open-folds when equity is below `junk = 0.37` and the raw choice was a limp.

Sizing is only `RAISE_HALF` and `RAISE_POT`, because those are the only raise actions the engine returns. The bot scores both when both are legal and takes the higher EV. It has no third size.

Time (`agent.py:550-646`). Target samples are `_BASE_M = (500, 400, 400, 400)` holdings and `_BASE_K = 240` simulated run-outs (`_BIG_K = 400` when the pot or the bet is large relative to the 80-chip stack). `scale` moves between 0.45 and 1.6 from a per-hand cost estimate, aiming to spend `_BANK_SHARE = 0.70` of the 8s bank. Below 0.15 scale when 88% of the bank looks spent, and `_cheap_decide` (no simulation) above 96%. `choose_action` and `on_hand_end` catch exceptions and check/fold or ignore them. `new_half` draws an unseeded `default_rng()`.

The patch does not change any of those constants. It only changes which holdings sit in the top percentile postflop, and it makes an extended pool's street percentile a rank in the combined sample.

## Criticism of the design

These are arguments against the choices above. Each one says what a strong opponent can do, what is source evidence, what is still a hypothesis, and how to test it. The ranker patch does not answer them. Wins against the four scripts in the first table are not evidence that the design is sound.

### A scalar preflop proxy drops hand-class interactions

A strong opponent keeps hand classes (double-suited rundowns, suited aces with a dangler, pair plus suit) and changes the continuing range when hero's cards block a nut suit. This proxy assigns one number from rank counts, suit counts, and straight-window occupancy. Two holdings with the same counts and the same window popcount get the same `_PRE_PCT` even when their suits block different boards. There is no term for a specific opponent combo. The sampler is separate: opponent hands are drawn from the cards hero does not hold (`_make_pool`, `known = hole | board`). Do not describe the bot as having no card removal.

Evidence: the feature list and `_PRE_COEF` are the whole preflop strength label (`agent.py` `_preflop_strength`). The R^2 = 0.89 comment, if taken at face value, already says about 11% of equity variance versus a random hand is missing, and that fit does not measure order against a tight continuing range. That second gap is a hypothesis. The calling-station win does not speak to it. A station does not fold when a suit is blocked.

Test: rank all `C(36,5)` holdings by `_PRE_PCT` and by enumerated equity versus a uniform hand and versus the top 20% of `_PRE_PCT`. Report pairwise inversions and the equity gap between holdings that share a percentile but differ by one suit. A replacement worth trying is a class table or a small equity lookup, still computed at import, keyed by the sorted rank-and-suit pattern.

### Turning a frequency into "the top x percent" is not a range

`_x_for` sets `x` from how often the action happens, then `_top` keeps a prefix of the same strength percentile. A check applies `_cap`, which downweights that same prefix. The continuing range is always an upper segment of one scalar, plus a floor of 0.04 so weak hands never quite die.

A strong opponent's continuing range is not that prefix. A polar pot-raise is the nuts and air, with a gap in the middle. A prefix model prices the air as if it were the hands just below the value cutoff, or, when `x` is large, prices the air as if it were inside the top. Examples that can actually occur: on a monotone flop the three board cards are three distinct ranks, so no 2+3 full house exists. The split is trips (a pair in the hole matching a board rank) versus a flush draw that is only a pair or high card plus the 1.3 bonus. On the turn, three of one suit plus a paired side card can make a full house, and a flush is made only when the hole has two of that suit. On the river a flush outranks a full house in `_VAL`, and there is no draw bonus, so the river prefix criticism is about nuts and air, not about full houses ranking above flushes.

Evidence: `_top`, `_cap`, and `_range_weights` implement exactly that prefix (`agent.py:522-547`). The three patched bugs were false positions inside this scalar. Fixing the draw rules makes the scalar less wrong. It does not add a second mode. The 121 pool extensions in 80 hands against a pot-raiser show the weights are narrow enough to matter. They do not show the shape is right. The monotone/connected pot-raise probe lost to both bots (patched +9.589 bb/deal, CI [+8.269, +10.909]; original +9.541, CI [+8.236, +10.845]). The postflop pot-pressure probe also lost (patched +10.314, CI [+8.737, +11.890]; original +10.932, CI [+9.372, +12.491]). Neither is a proven exploit. Whether a polar opponent who mixes nuts and air beats this hero is still a hypothesis.

Test: add a weight that is high on the top 10% and the bottom 25% and near zero in between, and measure paired bb against this hero. Separately, on a fixed monotone flop, compare the model's mean equity to enumerated equity against the hands that actually have a flush draw or a made flush.

### Position and size should change the shape, not just a penalty

The only position term in the EV is `oop_pen = 0.02` equity before the river. Stats have no button/blind key. Size enters as two fold bins split at 0.65 pot, and as a linear preflop multiplier `0.7 + 0.4 * size`. The same top-percentile cutoff is used for a min-raise and a pot raise except for that frequency.

A strong opponent opens the button wide and defends the blind wider against a small size than against a pot size, and the pot-size continue is polar. This hero uses one prefix either way, then subtracts two hundredths of equity when it is out of position.

Evidence: `_P["oop_pen"]`, `_size_class`, and the single `x` in `_x_for`. `was_button` is stored on the hero's last action and used only to infer a river check-back. There is no measurement in the first match table against a position-split range. The min-raise and maniac rows show the unpatched hero beat both a small size and a pot size. That is evidence the hero is not indifferent to size against those scripts. It is not evidence the 0.02 penalty or the two-bin fold model is calibrated. Calling the penalty "too small" is a hypothesis.

Test: a button that always opens, and a blind that continues the top 50% of an enumerated equity ranking against `RAISE_HALF` and a polar 15% plus air against `RAISE_POT`. Report hero bb by the size it faces. A concrete alternative is separate fold and continue frequencies for button and blind, which the observation already provides via `is_button` on the opponent's actions during `_replay`.

### Showdown calibration only sees hands that were not folded

`_calibrate` never observes a fold. `_x_for` then moves `x` toward `2 * (1 - mean showdown percentile)`. A strong opponent chooses which hands reach showdown. It can call with the nuts and fold the rest, so the shown mean percentile is very high and `x_obs` becomes very small, and the hero starts treating ordinary calls as the nuts. It can also check down weak hands when the hero checks, which pulls the shown mean the other way. The shrink `n / (n + 12)` and `sd_weight` slow this down. They do not distinguish "strong because they called" from "strong because they chose to show".

Evidence: the gate is `went_to_showdown and opp_hole is not None` (`agent.py:881-882`) and the inversion is `agent.py:428-432`. The source itself calls the showdown sample biased. The numerical effect on this hero's bb/deal was not isolated in the first review. The fold-bait probe below is a partial test of the frequency side, not of the showdown inversion. Treat the exploit magnitude as a hypothesis.

Test: two controls that reach showdown at the same rate. One calls only with the nuts. One calls any pair. Log `x` before and after `_calibrate` and the hero's bet frequency. An alternative is to drop showdown inversion, or to record the percentile only as a check against the frequency model and refuse to move `x` by more than a capped amount per half.

### Equity-when-called is a scalar, and future streets are three constants

Take `e = 0.40`, `fa = 0.50`, river `fold_k = 0.40`. Then `e_fold = 0.40 + 0.40 * 0.60 = 0.64`, and `e_called = (0.40 - 0.50 * 0.64) / 0.50 = 0.16`. The same formula at `e = 0.55`, `fa = 0.40`, flop `fold_k = 0.30` gives `e_called = 0.43` before the 0.05 realization haircut. Nothing in that arithmetic looks at the board, the stack (80 chips, blinds 2), or how many streets remain. `call_margin` and `realise` are constants per street. A draw that is behind now and ahead if it gets there is charged the same haircut as a made hand.

A strong opponent calls and raises with hands whose equity is in the run-out, and barrels later streets. This hero, once called, does not resimulate the opponent's future betting. It subtracts a constant and, on the next street, builds a new pool. That next pool is better than nothing. The raise EV on the current street still uses the constant.

Evidence: the formula is `_equity_when_called` and the `realise` / `call_margin` tuples in `_P`. The calling-station result (unpatched mean +12.666 bb/deal, CI entirely above 0) is evidence that hero still bets when `fa` is near 0, because a station does not fold and `fold_k` then cannot manufacture fold equity. It is not a measurement of `e_called` error. The worked numbers above are algebra, not match results.

Test: on enumerated rivers, compute equity against the top `x` fraction of the patched strength order and subtract `_equity_when_called`. Report bias by street and by `fa`. A replacement is to use the weighted pool's equity on the subset above the cutoff, which the bot already sampled, instead of the closed form.

### Fold frequencies can be trained and then reversed

Priors are 6 or 8 pseudo-hands and the decay is 0.995, about a 200-hand window. After a few dozen hands of pure folds, `fa` sits near the prior-updated rate and then falls slowly. A strong opponent folds to bets for the first part of the half, then calls or raises. While `fa` is still high, the bluff formula increases bluff probability (it rises with `fa`), and `_equity_when_called` marks the continuing hands as weak. The opponent is no longer folding, so both pieces are wrong in the direction that makes the hero bluff off chips.

Evidence: `_f_post`, `_decay_stats`, and the bluff formula. This is the mechanism. The fold-bait probe (40 hands of folding, then pot-raise and never fold) is a crude version. Measured on seed `review-patch-20261107`, n=1000: patched hero +8.213 bb/deal, CI [+6.694, +9.731]; original +8.354, CI [+6.833, +9.875]. Both intervals are entirely above zero, so this probe did not beat either bot. The mechanism remains in the source. A stronger opponent would switch as soon as the hero's bluff rate rises, and would call rather than pot-raise, which this probe does not do.

Test: sweep the training length and the post-training action (call versus pot). Plot hero paired bb against training length at one seed, then repeat on a second seed before calling it an exploit.

### The bluff cap is a published function of the fold estimate

Below equity 0.30 the bot bluffs with probability `min(0.75, max(0, (fa - 0.30) * 2))` and otherwise gives up. Above 0.30 it bets whenever the EV says so. A strong opponent does not need hero's cards to use this. It controls `fa` by folding and calling, and it can call more often on dry boards, where hero's equity when behind is more often under 0.30, and fold on wet boards, where hero's draw equity clears 0.30 and the hand is value. The cap never depends on board texture except through the equity estimate.

Evidence: the threshold, slope, and cap are `_P["bluff_lo"]`, `bluff_f0`, `bluff_slope`, `bluff_max`. No match in this review isolates bluff-catching. Predictability is a hypothesis about opponents who track bet frequency, not a measured bb loss.

Test: log, inside the harness, how often hero pots when equity `< 0.30`, binned by the current `fa`. Compare that rate to the formula. An alternative is a mix that depends on the draw bonus and on whether the bet is the first of the street, with the rate capped per texture rather than per `fa` alone.

### Two pot-limit sizes, and no plan that assigns hands to them

The engine allows only `RAISE_HALF` and `RAISE_POT` (and half can be bumped to the minimum raise or to all-in). Asking for more sizes is illegal. Inside those two, the bot computes an EV for each from the same equity and the same fold bin and adds 0.02 pot of noise. It does not reserve the pot size for a polar subset and the half size for a merged one. A strong opponent pots with nuts and air and uses the smaller size as a probe, and calls the hero's pot bets that are driven by a medium scalar plus noise.

Evidence: the action set is `game.py` (identical to the starter). The choice rule is the EV loop in `_decide`. The unpatched hero's edge versus the maniac (+6.785) and versus the min-raiser (+7.994) shows it can answer both sizes against scripts that do not switch. It does not show a size-to-range plan. Whether mixing the two sizes by one scalar loses chips to a size-aware caller is a hypothesis.

Test: force the patched hero to pot only when equity is above 0.65 or below 0.30, and to use half-pot otherwise, and play that variant against the same opponents and against the patched hero. Keep the three ranker fixes in both.

## Independent audit

Source: `../independent-final.log`, written against commit `5571729` only. It rechecked the two draw bugs and the emergency-history path. It did not time requests, fuzz illegal actions, or play a duplicate match.

What it executed: the showdown evaluator (flush beats a full house, trips beat a straight, the wheel loses to 6-7-8-9-T, `omaha_score` matches an exhaustive 2+3 search, a royal on the board is not awarded, four sixes with two kings is a full house, 24 straight flushes). Action-history replay on 160 hands and 535 opponent actions, including blind initialization. The straight windows include A-6-7-8-9; the straight bug was the popcount add, not a missing wheel.

Findings it confirmed, and what happened to them: the flush and straight ranker bugs and the batch percentile bug are the patches above. The emergency `hist_n` skip is finding 3b and is fixed. Impact it stated, and which still holds: the draw bugs change opponent ranking used by `_top`, not the pot awarded at showdown. The history skip changes the model only after `_cheap_decide` is already active.

Not claimed from that log: a timeout defect, an illegal action, or a measured bb/deal.

## Reproducing a single defect

```bash
taskset -c 0 /tmp/poker-venv-review/bin/python -m pytest -q test_review.py -k "draw or extended or nuts or replay"
```
