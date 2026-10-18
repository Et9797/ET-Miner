# Phase 3 polish: measurement protocol

Three overhead removals that the M1 and M5b reports listed as (should). None
changes a bit of any bitvector, count or result. On `perf/moonshot-polish`,
stacked on M5b. Fixed before the after-run.

## Changes

1. **The span check reads the chunks' entries** (M1 finding 1). The device
   input layer took the id range from Polars' `list.min` / `list.max` per row
   (`cpu_miner._int_ids`, 1.0 s on dsl). Each chunk's smallest and largest item
   now come from the entries it already reads in place, and from the
   non-null exploded entries for a chunk with nulls. That is the same range,
   nulls aside, as before. A chunk with null items keeps its nulls as a mask
   until the copy writes the null id.
2. **The lex order by two stable radix sorts** (M5b finding 1). `cp.lexsort`
   over the two keys took 0.25 s on dsl. Now: `argsort(key_lo)`, then a stable
   `argsort(key_hi)` through it. That is the same permutation, because both
   sorts are stable.
3. **Lazy tile occupancy** (M5b finding 2). Each device's occupancy is built
   at its first per-candidate or group launch, not before K=1. sk2ml3 counts
   K=3 row-wise and never builds it. The level split times it as `occupancy`
   in the level that builds it.

## Measurement

- The "before" rows are M5b's `lex` arm: `../results/2026-10-17-moonshots-m5b/raw.jsonl`,
  3 reps at `797b54c`, on this box.
- The "after" rows are the same configs (`--mode moonshots-polish`, the arm
  renamed `after`), 3 reps rep-major at the polish revision, run with
  `--max-gpu-hours 0.3`.
- Same box, same pins, fresh process per config. The two campaigns run hours
  apart, so the shared host's load may differ between them; that is recorded.

## Decision (rules 1 and 2, `decide.py --arms before after` over both files)

- Every after row must carry its regime's signature.
- The polish stays unless some regime loses by rule 2 (≥ 10 % slower, disjoint
  ranges, ≥ 1 s). It removes work, so a loss would point to a defect.

## Commands

```bash
OUT=bench/results/2026-10-18-moonshots-polish
uv run python bench/runner.py --mode moonshots-polish --out $OUT --max-gpu-hours 0.3
python3 -c "import json,sys; [print(json.dumps({**r, 'id': r['id'].replace('-lex#', '-before#')})) for r in map(json.loads, open('bench/results/2026-10-17-moonshots-m5b/raw.jsonl')) if '-lex#' in r['id']]" > $OUT/before.jsonl
cat $OUT/before.jsonl $OUT/raw.jsonl > $OUT/compare.jsonl
uv run python bench/moonshots/decide.py $OUT/compare.jsonl --arms before after > $OUT/decision.md
```
