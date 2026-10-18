# X0 — Count inference by default: measurement protocol

X0 is a table, not a code change: the whole complete-lattice matrix with
`use_generator_pruning` off and on, for the owner to decide the default. The
branch `perf/moonshot-x0` stacks on M2 (and M1), so the table measures the
tree the default would ship in. The default flips only on the owner's word.

## Box and budget

- The M1/M2 box: 2× RTX 3060 12 GB, EPYC 7282, thread pools pinned to 16,
  `NCCL_P2P_DISABLE=1`, a shared host.
- Budget: the brief's ~0.1 GPU-hours was for one GPU. The table also runs on
  two, and the campaign runs with `--max-gpu-hours 0.4`.

## Arms and regimes

| arm | `apriori(...)` |
|---|---|
| `plain` | `use_generator_pruning=False` (today's default) |
| `infer` | `use_generator_pruning=True` |

The regimes are smoke, deepk, skew, or003, or002, dsl, oom2ml3 and sk2ml3, on
one GPU and on two (`X0_REGIMES`). Free sets are left out: inference does
not apply to them. Every other knob is unset: the input layer and the K=3
kernel are their dispatch.

## Repetitions and recording

Fresh process per config, warm-up, 3 reps rep-major, medians with [min, max].
The level split, peak memory, throttle reasons, logged fallbacks and
signatures are recorded as in M1 and M2. No tracked-file edits during a run.

## Reading (`decide.py --arms plain infer`)

1. The two arms must give one signature per regime (inference is exact). A
   mismatch is a must-fix.
2. Per regime, rule 2 says whether `infer` wins (≥ 10 % lower median,
   disjoint ranges, ≥ 1 s saved) or loses (the same against it). Anything else
   is a tie.
3. The report shows the table and names the regimes where `infer` loses. The
   default stays `False` until the owner decides.

## Commands

```bash
OUT=bench/results/2026-10-17-moonshots-x0
uv run python bench/runner.py --mode moonshots-x0 --out $OUT --max-gpu-hours 0.4
uv run python bench/moonshots/decide.py $OUT/raw.jsonl --arms plain infer > $OUT/decision.md
uv run python bench/moonshots/split_table.py $OUT/raw.jsonl > $OUT/split.md
```
