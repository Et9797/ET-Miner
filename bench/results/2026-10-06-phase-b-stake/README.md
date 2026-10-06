# Phase B stakes, offline (2026-10-06)

What O4 and O5 of `bench/optimizations/SPEC.md` could gain, computed on the CPU
from the complete lattice dumps of the candidate-waste run
(`bench/results/2026-10-05-candidate-waste/*-C1-split.lattice.*`, not
committed, regenerable with `bench/runner.py --mode waste`). No GPU time.

Run from this directory:

```
uv run python o4_stake.py ../2026-10-05-candidate-waste/{deepk,dsl,skew,or002,or003,smoke}-C1-split.lattice > o4_stake.txt
uv run python k3_cost.py ../2026-10-05-candidate-waste/oom2ml3/oom2ml3-C1-split.lattice \
    ../2026-10-05-candidate-waste/sk2ml3-C1-split.lattice > k3_cost.txt
uv run python k4_cost.py ../2026-10-05-candidate-waste/oom2ml3/oom2ml3-C1-split.lattice \
    ../2026-10-05-candidate-waste/sk2ml3-C1-split.lattice > k4_cost.txt
```

## O4: tidset bytes of the K≥3 levels (`o4_stake.txt`)

| workload | K≥3 tidsets | inferred survivors | identical tidsets once |
|---|---|---|---|
| deep_k | 952 MB | 84.9 % | 7.4 % |
| deep_sparse_large | 1.23 TB | 61.8 % | 18.1 % |
| skewed_rows | 1.2 GB | 7.8 % | 89.1 % |
| online_retail 0.002 | 131 MB | 3.2 % | 99.9 % |
| online_retail 0.003 | 20 MB | 0.1 % | 100.0 % |

- "Identical tidsets once" (itemsets with the same closure share one tidset)
  is the least any sharing layout can store.
- On deep_sparse_large it still leaves 49.5 GB at K=5 and 30–42 GB at K=6–7:
  ESCO cannot run the middle levels on a 16 GB card with any sharing. The
  fit-checked transition is at K=14 (`../2026-10-06-esco-fit/`); ideal sharing
  could move it to K=10 or K=11 (level 9 holds 13.0 GB once shared, level 10
  5.0 GB).
- deep_k fits without sharing: ESCO's peak there is 331 MB (phase A,
  `../2026-10-05-optimizations/raw.jsonl`, `deepk-C1-esco-base`).
- Time: materialization, the step O4 shortens, takes at most 0.22 s per run on
  the measured ESCO workloads (phase A, deepk on one GPU), below rule 2's 1 s.

## O5: K=3 work on the explosion workloads (`k3_cost.txt`)

| workload | mean K=2 count | N/32 | conversion | K=3 after the subset test | sparse / dense bytes |
|---|---|---|---|---|---|
| oom_regression to K=3 (500K) | 39 | 15,625 | 0.07 GB | 15,817,779 | 3.8 % |
| stress_k2 to K=3 (2M) | 103 | 62,500 | 0.68 GB | 144,951,409 | 3.1 % |

- Dense bytes count every surviving candidate's ceil(N/64) words without tile
  reuse; sparse bytes are a merge of both parent tidsets. Both are models, not
  measurements: the tiled kernel reuses staged rows and skips tile-pairs that
  are entirely prunable, and the CSR kernel enumerates every generated
  candidate (11.8 B on stress_k2) before it intersects.
- Measured dense K=3 level on this box (phase A final check, defaults):
  stress_k2 16.96 s on one GPU, 11.84 s on two; oom_regression 0.99 s and
  0.88 s. ESCO never ran on either.

## O5: one level deeper (`k4_cost.txt`)

Whether the explosion workloads mined to K=4 would be a further regime:

| workload | K=4 generated | after the subset test | dense bytes | sparse merge bytes |
|---|---|---|---|---|
| oom_regression | 48,003,920 | 170,021 | 1.06e10 | 3.08e8 |
| stress_k2 | 3,135,256,758 | 4,677,419 | 1.17e12 | 4.40e10 |

- oom_regression's K=4 holds tens of milliseconds of dense work: nothing for
  rule 2's 1 s.
- stress_k2's K=4 holds 1/31 of its K=3 dense bytes but 27 % of its generated
  candidates: mostly enumeration, the risk K=3 already carries. The stake stays
  at K=3.

Matrix density (nnz / (N × vocabulary)), from `datasets/synth/*.json`: stress_k2
6.8e-4, oom_regression 6.6e-4, deep_k 3.2e-3, skewed_rows 4.0e-3, smoke
4.2e-3, deep_sparse_large 1.2e-2. The two calibration workloads are the
sparsest in the suite, with K=2 tidsets at 5e-5 and 8e-5 of N.
