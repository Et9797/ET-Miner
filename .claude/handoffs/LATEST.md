# Session Handoff: ET-Miner, 2026-10-18 18:06

**Branch:** `fix/pool-fragmentation` @ `cc14c6f` (in sync with origin)
**Tree:** clean
**Focus:** open optimization items after PR #35; two follow-up PRs open, council-reviewed and fixed
**Reason:** end of session

## Verify on Resume
Run `git status --short --branch` and `git rev-parse --short HEAD`. HEAD should be `cc14c6f` on `fix/pool-fragmentation`, tree clean. The other open branch is `perf/two-gpu-teardown` @ `1a74cd0`. If the owner merged #36/#37 meanwhile, start from an up-to-date `main`.

## Current State
- **PR #36** (`fix/pool-fragmentation`, https://github.com/Et9797/ET-Miner/pull/36): open, council 4/4 APPROVE. Fixes REPORT Findings 3: release CuPy's cached pool blocks before every row-split run.
- **PR #37** (`perf/two-gpu-teardown`, https://github.com/Et9797/ET-Miner/pull/37): open, council 4/4 APPROVE. Names the 2-GPU tail (NCCL destroy) and moves the destroy to a background thread.
- The council's points 1–3 (report contradiction, per-test GPU markers, wording/erratum) are fixed and pushed on both branches: `cc14c6f` (#36), `1a74cd0` (#37).
- `git merge-tree` of both branches is clean, and the merged `REPORT.md` reads consistently.
- Nothing is in flight. No timed run is running.
- Waiting on the owner: merge #36 and #37, then pick the next item.

## Next Steps
1. Owner merges #36 and #37 (in either order). Afterwards, on `main`, run `tests/test_nccl_teardown.py`, `tests/test_row_split_e2e.py::TestPoolReuse` and the tier chain on the merged tree (council flag: the combination was checked only by the auditor's scratch merge, 99 passed).
2. The next open item, recommended to the owner: **M2 dispatch statistic** (REPORT Findings 4). Replace r3 by (R2 + ρ²·T3)/(slots × words). It needs its own sweep (`bench/moonshots/k3_crossover.py`) and a PROTOCOL first.
3. Further items, from the old handoff list:
   - M7 per-level switch (Findings 5);
   - deep-param register cost;
   - lex row order on real data;
   - M5b permuted `ids_to_bitvec`;
   - host fallback builder;
   - unserved column read;
   - `KEEP_ROWS_BYTES`;
   - cleanups.
4. Open council flags (nice, owner's call):
   - `_release_cached_blocks` is the third copy of `gpu/utils.py:113 cleanup_gpu_memory` / `streaming/multi_gpu.py:93`. Those two also free the pinned pool, so reuse changes behaviour; at least document why.
   - The `bitvecs=` call site (`core/apriori.py:647` on #36) has no test.
   - `TestPoolReuse[host]` passes spuriously after a failed `[device]` (a traceback holds GPU memory).
   - SON GPU routes do not release the pool (unverified whether they need it).
   - `bench/level_split.py` `_TimedComm.destroy` sets `_destroyed` before success.
   - Pending destroys hold ~16 MiB/device each in tight loops (bounded).
   - New Findings 12: `count_tiled_fused` sizes its output at min(2²⁴, candidates) × 12 B regardless of the budget.

## Session Instructions
- Owner decisions 2026-10-18:
  - X0 `use_generator_pruning` default stays **off** (closed);
  - **no 1-GPU dispatch** for small work (`n_gpus` stays the exact device count);
  - NCCL fix = **background destroy thread** (not caching communicators, not `abort()`);
  - for #37: one PR with naming + fix together.
- Campaign rules still bind:
  - one branch per idea off `main`;
  - PROTOCOL committed before any timed run;
  - ≥3 reps rep-major, fresh process per config, same box only;
  - no tracked-file edits/commits during a timed run;
  - knobs only as `ET_MINER_*` pins;
  - GPU budget ≤ 1 GPU-h per idea (#37 used 0.244);
  - no PR merge without the owner.
- Chat in Dutch, plain language. Files, code and commits in English.
- Reports: numbered lists tagged (must-fix)/(should)/(nice).

## Decisions & Rationale
- **Pool release in `core/apriori.py`, at three sites before the first GPU allocation** (bitvecs=, before `build_device_input`, before the host `_apriori_row_split_multi_gpu`). The release inside the miner would come too late: `build_device_input` and `shard_prebuilt_bitvecs` allocate first. Only the run's devices are touched, so no context is created on unused GPUs. The pinned pool is not freed (the finding is device memory only).
- **NCCL destroy hand-off after the level loop's try/finally, normal path only** (`gpu/row_split.py:1261`). After an error the comms go with the frame, as before, so a hung collective cannot strand a background thread. The thread is non-daemon, so the interpreter joins it before CUDA's atexit handlers run.
- **Errata, not silent edits, for protocols after their run** (`PROTOCOL-TEARDOWN.md` "554 rows" → erratum: 404 unique).
- **REPORT.md follow-ups go in an appended "After the merge" section** so parallel PRs do not conflict on the Findings lines.

## Dead Ends
- Do not use `abort()` for NCCL teardown: 0.54–0.56 s, slower than `destroy()`.
- Do not use `gh pr view`/`gh pr edit` for PR bodies: they fail with a GraphQL "Projects (classic) is being deprecated" error. Use `gh api repos/Et9797/ET-Miner/pulls/<n> --jq .body` and `gh api -X PATCH ... -F body=@file`.
- Do not rebase branches whose commits stamp campaign rows (`8d7d317`, `09d08fd`): the rows' `rev` would become unreachable.
- Do not trust `TestPoolReuse` results right after another test failed with OOM: the pytest traceback keeps GPU memory alive.

## Key Findings
- 2-GPU tail = NCCL communicator destroy. NCCL 2.31.2's proxy thread polls every 500 ms when idle, so the destroy waits ≈ 500 ms − (idle since last NCCL call mod 500). Before the fix: 0.319 s median, 0.028–0.520 s. After: `tail_nccl` 0.
- Pool OOM repro: a second `apriori(stress_k2, max_length=2)` raised "Out of memory allocating 8,750,000,128 bytes"; the cached block is exactly the bitvectors (35000 × 31250 × 8 B).
- The 2-GPU reduce is the next fixed cost on small regimes (0.07–0.09 s; the first all-reduce on new comms is ~0.07 s). NCCL first init is 0.6 s, but the bench warm-up pays it.

## Test Status
- `perf/two-gpu-teardown` @ `09d08fd`:
  - tier: 91 passed;
  - `-m "not slow" -k "not test_support_00"`: 1625 passed;
  - gpu+slow: 17 passed;
  - no-CuPy: 895/559;
  - selfcheck READY.
- At `1a74cd0`: affected tests 25 passed; no-CuPy 896 passed, 563 skipped; ruff clean.
- `fix/pool-fragmentation` @ `8ee633a`:
  - tier: 91 passed;
  - not-slow: 1617 passed;
  - gpu+slow: 17 passed;
  - no-CuPy: 893/556 (with `-k "not test_support_00"`);
  - selfcheck READY.
  - `cc14c6f` is a report-only change.

## What Was Done
- `core/apriori.py` `_release_cached_blocks` + 3 call sites; `tests/test_row_split_e2e.py::TestPoolReuse` (500k rows, limit held + 1.75 × cached).
- `gpu/nccl.py` `_destroy_nccl`; `gpu/row_split.py:1258-1262` hand-off; `tests/test_nccl_teardown.py` (6 tests).
- Bench changes:
  - `bench/level_split.py` `_TimedComm`, `tail_nccl`, `nccl_destroys`;
  - `--mode moonshots-teardown(-after)` (`bench/consolidation_matrix.py`, `bench/runner.py`);
  - `bench/moonshots/tail_table.py`.
- Reports:
  - `bench/moonshots/PROTOCOL-TEARDOWN.md`, `REPORT-TEARDOWN.md`, `REPORT.md`;
  - results in `bench/results/2026-10-18-moonshots-teardown{,-after}/`;
  - CHANGELOG entries.
- Council review of both PRs (4/4 APPROVE, round 1).
- Memory: `moonshots-campaign.md` updated with the owner decisions and progress.

## Commits This Session
- `cc14c6f` Moonshots report: Findings 2 leaves the tail's status to its follow-up (#36)
- `1a74cd0` Council review of PR #37: wording, erratum, per-test GPU markers (#37)
- `46166ad` Teardown report, Amendment 1: every 2-GPU regime faster by the wait (#37)
- `09d08fd` GPU route: destroy the NCCL communicators off the call's path (#37)
- `a3028c2` Teardown report: the 2-GPU tail is the NCCL destroy (#37)
- `8d7d317` bench: time the NCCL destroy in the 2-GPU tail; teardown protocol (#37)
- `f17650b` handoff: 2026-10-18 13:36, after PR #35 (#36)
- `8ee633a` GPU route: release the pool's cached blocks before a run (#36)

## Key Files
| File | Role |
|------|------|
| `src/et_miner/core/apriori.py` (#36) | `_release_cached_blocks` and its call sites |
| `src/et_miner/gpu/nccl.py`, `gpu/row_split.py` (#37) | `_destroy_nccl`, hand-off after the level loop |
| `bench/level_split.py` | span phases incl. `tail_nccl` |
| `bench/moonshots/REPORT.md` | all open items; "After the merge" section (#37) |
| `bench/moonshots/{PROTOCOL,REPORT}-TEARDOWN.md` | teardown protocol + Amendment 1, report |
| `tests/test_nccl_teardown.py`, `tests/test_row_split_e2e.py` | new regression tests |

## Extra Context
- Box: 2× RTX 3060 12 GB, EPYC 7282, NCCL 2.31.2, CuPy 14.1.1; campaigns pin threads 16 and `NCCL_P2P_DISABLE=1`.
- The scratch diagnostics are in the session scratchpad (`repro_pool.py`, `nccl_destroy.py`, `tail_diag.py`); they are not in the repo.
