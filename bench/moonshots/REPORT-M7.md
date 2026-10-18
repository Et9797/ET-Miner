# M7 — Deduction bounds as count inference: report

**Decision: the bounds stay on under count inference.** They win dsl by rule 2
on one GPU and on two, and lose no regime. The final check (the knob unset)
runs them in every regime and reproduces the `bounds` arm's walls. No
signature mismatch, no logged fallback. The bounds act only with
`use_generator_pruning=True`, so the default output and default runs are
unchanged.

Protocol: `PROTOCOL-M7.md`. Data: `../results/2026-10-17-moonshots-m7/`
(`decision.md`, `split.md`, `raw.jsonl`), at `9abcd99` on `perf/moonshot-m7`
(stacked on X0, M2 and M1).

## What changed

- `_src/_subset_index.cu`: when Pascal inference leaves a candidate of at
  most 24 items, `_classify_candidate` hands it to `_deduce` (`__noinline__`).
  `_deduce` reads levels k−2 and k−3 from the index and takes:
  - the lower bound from every pair;
  - the upper bound from every triple, stopping once they meet or the upper
    bound falls below min_count.

  An upper bound below min_count skips the candidate. Equal bounds give its
  count, written by one device.
- Every K≥3 counting kernel (per-candidate, group, tiled dense and fused, CSR)
  takes the deep levels through `SUBSET_DEEP_PARAMS` / `SUBSET_DEEP_ARGS`.
- `subset_index.py` uploads the deep levels; `row_split.py` keeps the last
  three complete levels (`_complete_levels`) in complete-lattice runs with
  inference.
- `ET_MINER_DEDUCTION=0|1` pins it. Unset, it is on wherever the levels exist
  and the three indexes fit a quarter of free VRAM.
- Tests (`tests/test_deduction.py`):
  - the classifier against a host reference, on zeroed bitvectors, over data
    where the bounds both skip candidates and settle them with equal bounds;
  - mining with the bounds on and off against the oracle, every kernel pinned,
    forced chunks, ESCO, one and two GPUs, an item in every row, empty rows,
    ties.
- Tier legs: count inference with the bounds on and off, on 1 and 2 GPUs.

## Results (2× RTX 3060 / EPYC 7282, median of 3, seconds; every config with count inference)

| regime | GPUs | Pascal only [min, max] | with bounds [min, max] | bounds/Pascal | rule 2 |
|---|---|---|---|---|---|
| dsl | 1 | 5.83 [5.77, 5.84] | 3.65 [3.61, 3.66] | 0.625 | **bounds win** (−2.18 s) |
| dsl | 2 | 4.83 [4.30, 4.84] | 3.33 [3.32, 3.35] | 0.689 | **bounds win** (−1.50 s) |
| or002 | 1 / 2 | 0.29 / 1.08 | 0.37 / 1.08 | 1.28 / 1.00 | tie (+0.08 s on one GPU) |
| sk2ml3, oom2ml3, deepk, skew, or003, smoke | 1 / 2 | | | 0.96–1.02 | tie |

dsl on one GPU, per level (Pascal → bounds): K=6 0.68 → 0.34, K=7 0.94 →
0.27, K=8 0.78 → 0.11, K=9 0.49 → 0.09, K=10 0.18 → 0.06. That is 3.07 →
0.87 s over K=6–10, the levels where the bounds settle 72–100 % of the
candidates (`SPEC.md` § M7).

Final check: 16 of 16 ok, signatures consistent. dsl 3.65 / 3.34 s (one /
two GPUs) against the `bounds` arm's 3.65 / 3.33.

Budget used: 0.301 GPU-hours.

Together with M1, M2 and X0's default flip (the owner's decision), dsl with
count inference has gone from 25.2 s on the Pentium box (Phase 0) to 3.65 s
here; the boxes differ, so that is context, not a comparison.

## Findings

1. **(should)** On or002 (one GPU) the bounds cost 0.08 s: 0.29 → 0.37 s,
   disjoint ranges. Its K=5–7 levels have many candidates the bounds leave
   undecided; K=5 goes 0.053 → 0.077 s and K=6 0.022 → 0.042 s. That is under
   rule 2's 1 s, so the default stays. A per-level dispatch would remove it,
   by turning the bounds off for level k+1 when they settled few of level k's
   candidates. It needs the kernels to report how many they settled.
2. **(nice)** `SUBSET_BOUNDS_MAX_K = 24` bounds the local arrays (a k-item
   candidate holds C(k, 2) counts). Above it Pascal inference alone applies.
   No regime reaches K=24.
