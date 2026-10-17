# M7 — Deduction bounds as count inference: measurement protocol

The design is `SPEC.md` § M7 (and its amendment); the code is
`_src/_subset_index.cu`'s `_deduce`, `subset_index.py`'s deep levels and
`row_split.py`'s `_complete_levels`, on `perf/moonshot-m7`. That branch
stacks on X0, M2 and M1, so the tree is the device input layer with
row-wise K=3 by dispatch. Fixed before the first timed run.

## Why it is re-opened

Phase 1 dropped M7 on the Pentium box's walls (1.65 s of 21.54 s, 7.7 %).
After M1, dsl with count inference takes 5.8 s on this box, and its K=6–10
levels are 3.1 s of it. The bounds settle 72–100 % of their counted
candidates (`SPEC.md` § M7), so the stake now passes rule 2's thresholds. An
indicative run at this tree gave 5.84 → 3.65 s.

## Box and budget

- The M1 box (2× RTX 3060, EPYC 7282, thread pools at 16,
  `NCCL_P2P_DISABLE=1`, a shared host).
- Budget ≤ 1 GPU-hour; the campaign runs with `--max-gpu-hours 0.45`.

## Arms

Every config mines with `use_generator_pruning=True`: the bounds exist only
under count inference.

| arm | pin | K≥3 subset test |
|---|---|---|
| `pascal` | `ET_MINER_DEDUCTION=0` | today's: prune, and Pascal inference for a non-free subset |
| `bounds` | `ET_MINER_DEDUCTION=1` | also the deduction bounds of depth ≤ 3 (skip on u < min_count, infer on l = u) |
| `final` | unset | the dispatch: bounds wherever levels k−2 and k−3 are complete and the three indexes fit a quarter of free VRAM |

Every other knob is unset.

## Regimes, repetitions, recording

- `X0_REGIMES` (smoke, deepk, skew, or003, or002, dsl, oom2ml3, sk2ml3), on
  one GPU and on two.
- `pascal` and `bounds` run 3 reps rep-major. The `final` configs run once,
  after them: they are the final check.
- Fresh process per config, warm-up as before, medians with [min, max].
- The level split, peak memory, throttle reasons, logged fallbacks and
  signatures are recorded. No tracked-file edits during a run.

## Decision rule (rules 1, 2, 4, 5; `decide.py --arms pascal bounds`)

1. Every row of a regime, final rows included, carries one signature. A
   mismatch or an unallowed fallback is a must-fix.
2. `bounds` wins a regime with a ≥ 10 % lower median, disjoint ranges and
   ≥ 1 s saved.
4. If `bounds` wins some regimes and loses none, the default stays on (the
   dispatch above). If it loses some, a dispatch rule from facts known before
   the level (the candidates' K, the level sizes) must leave those regimes on
   Pascal alone, or the default goes off.
5. If it wins nowhere, the bounds go.

The final rows must match the `bounds` arm's walls within its range in the
regimes it wins.

## Commands

```bash
OUT=bench/results/2026-10-17-moonshots-m7
uv run python bench/runner.py --mode moonshots-m7 --out $OUT --max-gpu-hours 0.45
uv run python bench/moonshots/decide.py $OUT/raw.jsonl --arms pascal bounds > $OUT/decision.md
uv run python bench/moonshots/split_table.py $OUT/raw.jsonl > $OUT/split.md
```
