# The 2-GPU tail: measurement protocol

`REPORT.md` Findings 2: on two GPUs the time from the last level's callback to
the return is 0.03–0.54 s, and it decides the 2-GPU medians of the small
regimes. This protocol names it. It changes no line of `src/`: the tree is
`main` at `59e991d` plus the level split's new phase. Fixed before the first
timed run, on `perf/two-gpu-teardown`.

## What is known before the run

- **The campaigns' rows on this box** (`../results/2026-10-1*-moonshots-*`,
  EPYC 7282 rows only, 554 per GPU count). `tail − frame` has a median of 0.001 s
  on one GPU (max 0.027) and 0.302 s on two (quartiles 0.26 and 0.39, range
  0.033–0.541).
- **What exists only on two GPUs** is the pair of NCCL communicators.
  `row_split._apriori_row_split_multi_gpu` never destroys them itself: they go
  when its last reference does, at the return, inside the tail.
- **An indicative check** (scratch scripts, not campaign rows, NCCL 2.31.2,
  `NCCL_P2P_DISABLE=1`). Destroying two communicators after one all-reduce
  takes about 500 ms minus the idle time since the last NCCL call, modulo
  500 ms. The first rank's destroy returns at once; the second one waits.

  | idle before the destroy | destroy, 3 reps |
  |---|---|
  | 0 ms | 0.449–0.467 s |
  | 100 ms | 0.366–0.383 s |
  | 250 ms | 0.206–0.218 s |
  | 400 ms | 0.055–0.076 s |
  | 600 ms | 0.353–0.388 s |
  | 900 ms | 0.045–0.080 s |
  | 1200 ms | 0.238–0.271 s |

  On smoke with two GPUs (4 fresh processes) the destroy was the whole tail:
  0.356–0.382 s of a 0.55–0.57 s wall.

## The new phase

`bench/level_split.py` wraps each communicator that `row_split._init_nccl`
returns. The wrapper passes every method through, and when the miner drops it,
it destroys the communicator and records the time as `tail_nccl`.
`tail_other` becomes `tail − frame − tail_nccl`. The communicators are
destroyed where they were before, at the miner's return, in the same order.
`tests/test_moonshots_bench.py::test_level_split_times_the_nccl_destroy`
checks that the result is unchanged on one GPU and on two.

## Box and budget

- The M1 box: 2× RTX 3060, EPYC 7282, thread pools at 16,
  `NCCL_P2P_DISABLE=1`, on a shared host.
- Budget ≤ 1 GPU-hour. Phase 3's 30 two-GPU rows took 229 s of subprocess
  time (0.13 GPU-hours), so the campaign runs with `--max-gpu-hours 0.2`.

## Configs, repetitions, recording

- Every `MOONSHOT_REGIMES` regime on two GPUs, every knob unset
  (`--mode moonshots-teardown`, arm `teardown`). One GPU is left out: it builds
  no communicator, and its tail is the 554 rows above.
- 3 reps rep-major, a fresh process per config, warm-up as before.
- The level split, peak memory, throttle reasons, logged fallbacks and
  signatures are recorded. No tracked-file edits or commits during the run.

## Reading (no decision on a fix)

1. Every row carries its regime's signature from Phase 3's 2-GPU rows
   (`--reference`). A mismatch or a logged fallback is a must-fix.
2. **The tail is named** as the communicators' destroy if every config's
   median `tail_other` is at most 0.027 s, the one-GPU maximum above.
   Otherwise the remainder is reported as unnamed, with the configs it shows
   in.
3. Moving the destroy off the call's path is a separate change. It needs its
   own arm and protocol, and the owner's word.

## Commands

```bash
OUT=bench/results/2026-10-18-moonshots-teardown
uv run python bench/runner.py --mode moonshots-teardown --out $OUT --max-gpu-hours 0.2
uv run python bench/moonshots/tail_table.py $OUT/raw.jsonl \
  --reference bench/results/2026-10-18-moonshots-phase3/final.jsonl > $OUT/tail.md
uv run python bench/moonshots/split_table.py $OUT/raw.jsonl > $OUT/split.md
```

## Amendment 1: the destroy off the call's path (owner, 2026-10-18)

Fixed after the run above (`REPORT-TEARDOWN.md`) and before the after-run.

### Change

Once its levels are done, `row_split._apriori_row_split_multi_gpu` hands its
communicators to `nccl._destroy_nccl`. That function destroys them on a
thread of its own (`et-miner-nccl-destroy`), while the call builds its
result frame and returns. The thread is not a daemon: the interpreter waits
for it at exit, before CUDA's exit handlers run. After an error the miner
starts no thread, and the communicators go with its frame as before. No
knob, no result change. `tests/test_nccl_teardown.py` covers:
- the hand-off;
- three calls in a row, each init overlapping the previous destroy;
- the error path;
- a process that exits right after a call.

The level split now counts as `tail_nccl` only a destroy on the call's own
thread. A destroy on another thread is outside the call.

### Measurement

- The "before" rows are the run above: `raw.jsonl` at `8d7d317`, the arm
  renamed `before`.
- The "after" rows are the same configs (`--mode moonshots-teardown-after`,
  arm `after`), 3 reps rep-major at the change's revision, with
  `--max-gpu-hours 0.2`.
- Same box, same pins, fresh process per config. The two runs are hours
  apart, so the shared host's load may differ; it is recorded.

### Decision (rules 1 and 2, `decide.py --arms before after` over both files)

1. Every after row carries its regime's signature, and no row logs a
   fallback.
2. Every after row has `tail_nccl` = 0 and `tail_other` at most 0.027 s.
3. The change stays unless some regime loses by rule 2 (≥ 10 % slower,
   disjoint ranges, ≥ 1 s). It removes a wait, so a loss would point to a
   defect, for example the destroy thread competing with the result frame's
   build. Rule 2's 1 s cannot be met by a wait of at most 0.5 s. The per-regime
   change of the median wall is reported as measured, against the before
   rows' `tail_nccl`.

### Commands

```bash
OUT=bench/results/2026-10-18-moonshots-teardown-after
uv run python bench/runner.py --mode moonshots-teardown-after --out $OUT --max-gpu-hours 0.2
python3 -c "import json; [print(json.dumps({**r, 'id': r['id'].replace('-teardown#', '-before#')})) for r in map(json.loads, open('bench/results/2026-10-18-moonshots-teardown/raw.jsonl'))]" > $OUT/before.jsonl
cat $OUT/before.jsonl $OUT/raw.jsonl > $OUT/compare.jsonl
uv run python bench/moonshots/decide.py $OUT/compare.jsonl --arms before after > $OUT/decision.md
uv run python bench/moonshots/tail_table.py $OUT/raw.jsonl \
  --reference bench/results/2026-10-18-moonshots-phase3/final.jsonl > $OUT/tail.md
```
