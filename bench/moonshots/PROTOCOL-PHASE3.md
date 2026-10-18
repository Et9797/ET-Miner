# Phase 3: the reduced matrix against the baseline, on this box

Phase 0 ran on another box (1× A4000, Pentium G5420), so its numbers are not
compared here. Both trees are measured on this box with the same harness, the
same pins and the same configs instead:

- **baseline**: `src/` of `perf/moonshot-baseline` (`c0f0fd2`: `main`'s
  `src/` plus the `deep_k20` preset), checked out over the final tree's
  `src/` only (`git checkout c0f0fd2 -- src`). Its rows are stamped
  `<final sha>-dirty.<digest>`. The harness is the final tree's;
  `level_split.py` skips the phases of modules the baseline lacks.
- **final**: the stacked tree on `perf/moonshot-polish`: M1, M2, X0's table
  (default unchanged), M7, M5b and the polish.

## Box, budget, pins

- The M1 box (2× RTX 3060, EPYC 7282), thread pools at 16,
  `NCCL_P2P_DISABLE=1`, every `ET_MINER_*` knob unset (each tree's own
  dispatch), `use_generator_pruning` as each regime sets it.
- Budget: ≤ 0.8 GPU-hours for both runs (`--max-gpu-hours 0.25` final,
  `0.55` baseline).

## Matrix (`--mode moonshots-phase3`)

Every moonshot regime on one GPU and on two, 3 reps rep-major, fresh process
per config, warm-up, the level split, medians with [min, max]. The final
tree runs first, then the baseline (no tracked edits during either run except
the documented `src/` checkout between them; `git checkout HEAD -- src`
restores it after).

## Reading

`decide.py --arms baseline final` over both files, ids relabelled. Rules 1
and 2 per regime:
- every regime must carry one signature across the two trees (a mismatch is
  a must-fix of the whole stack);
- the table reports where the stack wins, ties and loses.

## Commands

```bash
OUT=bench/results/2026-10-18-moonshots-phase3
uv run python bench/runner.py --mode moonshots-phase3 --out $OUT/final --max-gpu-hours 0.25
git checkout c0f0fd2 -- src
uv run python bench/runner.py --mode moonshots-phase3 --out $OUT/baseline --max-gpu-hours 0.55
git checkout HEAD -- src
python3 -c "import json; [print(json.dumps({**r, 'id': r['id'].replace('-phase3#', '-baseline#')})) for r in map(json.loads, open('$OUT/baseline/raw.jsonl'))]" > $OUT/baseline.jsonl
python3 -c "import json; [print(json.dumps({**r, 'id': r['id'].replace('-phase3#', '-final#')})) for r in map(json.loads, open('$OUT/final/raw.jsonl'))]" > $OUT/final.jsonl
cat $OUT/baseline.jsonl $OUT/final.jsonl > $OUT/compare.jsonl
uv run python bench/moonshots/decide.py $OUT/compare.jsonl --arms baseline final > $OUT/decision.md
```
