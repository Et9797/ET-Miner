# Consolidation campaign report

488 config-reps, 459 ok. Box time 4.16 h; GPU-hours (process time × devices used) 4.49.

Cells: median [min, max] in seconds over the ok reps (a rep count in parentheses when it is not 3);
`>N (timeout)` is a lower bound; `FAIL` gives the status. Rule 2 needs the winner's median ≥ 10% below
the best alternative, disjoint [min, max] ranges, and ≥ 1 s saved.

## Signatures

| regime | signatures | configs |
|---|---|---|
| ('deep_k', 0.02, None, False) | 1 | 72 |
| ('deep_k', 0.02, None, True) | 1 | 6 |
| ('deep_sparse_large', 0.015, None, False) | 1 | 34 |
| ('deep_sparse_large', 0.015, None, True) | 1 | 6 |
| ('online_retail', 0.002, None, False) | 1 | 48 |
| ('online_retail', 0.002, None, True) | 1 | 6 |
| ('online_retail', 0.003, None, False) | 1 | 60 |
| ('online_retail', 0.005, None, False) | 1 | 30 |
| ('oom_regression', 3e-05, 2, False) | 1 | 17 |
| ('skewed_rows', 0.02, None, False) | 1 | 63 |
| ('smoke', 0.01, None, False) | 1 | 62 |
| ('smoke', 0.01, None, True) | 1 | 6 |
| ('stress_k2', 1.5e-05, 2, False) | 1 | 16 |
| ('stress_k2', 1.5e-05, 3, False) | 1 | 3 |
| ('wide_vocab', 0.004, None, False) | 1 | 30 |

## Non-ok config-reps

- `dsl-A1-legacy-auto#r0`: error: OutOfMemoryError: Out of memory allocating 2,385,000,448 bytes (allocated so far: 10,372,472,320 bytes).
- `dsl-A1-legacy-auto#r1`: error: OutOfMemoryError: Out of memory allocating 2,385,000,448 bytes (allocated so far: 10,379,384,832 bytes).
- `dsl-A1-legacy-auto#r2`: error: OutOfMemoryError: Out of memory allocating 2,385,000,448 bytes (allocated so far: 10,387,515,392 bytes).
- `dsl-A1-shared-auto#r0`: error: OutOfMemoryError: Out of memory allocating 2,385,000,448 bytes (allocated so far: 10,374,873,600 bytes).
- `dsl-A1-shared-auto#r1`: error: OutOfMemoryError: Out of memory allocating 2,385,000,448 bytes (allocated so far: 10,374,873,600 bytes).
- `dsl-A1-shared-auto#r2`: error: OutOfMemoryError: Out of memory allocating 2,385,000,448 bytes (allocated so far: 10,374,873,600 bytes).
- `dsl-C1-legacy-auto#r0`: error: OutOfMemoryError: Out of memory allocating 2,385,000,448 bytes (allocated so far: 10,374,873,600 bytes).
- `dsl-C1-legacy-auto#r1`: error: OutOfMemoryError: Out of memory allocating 2,385,000,448 bytes (allocated so far: 10,374,873,600 bytes).
- `dsl-C1-legacy-auto#r2`: error: OutOfMemoryError: Out of memory allocating 2,385,000,448 bytes (allocated so far: 10,374,873,600 bytes).
- `dsl-C1-legacy-k3#r0`: error: OutOfMemoryError: Out of memory allocating 5,385,686,016 bytes (allocated so far: 7,693,704,704 bytes).
- `dsl-C1-legacy-k3#r1`: error: OutOfMemoryError: Out of memory allocating 5,385,686,016 bytes (allocated so far: 7,693,704,704 bytes).
- `dsl-C1-legacy-k3#r2`: error: OutOfMemoryError: Out of memory allocating 5,385,686,016 bytes (allocated so far: 7,693,704,704 bytes).
- `dsl-C1-shared-auto#r0`: error: OutOfMemoryError: Out of memory allocating 2,385,000,448 bytes (allocated so far: 10,374,873,600 bytes).
- `dsl-C1-shared-auto#r1`: error: OutOfMemoryError: Out of memory allocating 2,385,000,448 bytes (allocated so far: 10,374,873,600 bytes).
- `dsl-C1-shared-auto#r2`: error: OutOfMemoryError: Out of memory allocating 2,385,000,448 bytes (allocated so far: 10,374,873,600 bytes).
- `dsl-C1-shared-k3#r0`: error: OutOfMemoryError: Out of memory allocating 5,385,686,016 bytes (allocated so far: 7,693,704,704 bytes).
- `dsl-C1-shared-k3#r1`: error: OutOfMemoryError: Out of memory allocating 5,385,686,016 bytes (allocated so far: 7,693,704,704 bytes).
- `dsl-C1-shared-k3#r2`: error: OutOfMemoryError: Out of memory allocating 5,385,686,016 bytes (allocated so far: 7,693,704,704 bytes).
- `dsl-D1-gpu#r0`: timeout
- `dsl-E2#r0`: timeout
- `sk2ml3-A1-legacy#r0`: skipped: sk2ml2-A1-legacy K=2 median 453.2s is more than 2.0x the 1-GPU best 47.2s
- `sk2ml3-A1-legacy#r1`: skipped: sk2ml2-A1-legacy K=2 median 453.2s is more than 2.0x the 1-GPU best 47.2s
- `sk2ml3-A1-legacy#r2`: skipped: sk2ml2-A1-legacy K=2 median 453.2s is more than 2.0x the 1-GPU best 47.2s
- `sk2ml3-B1#r0`: skipped: sk2ml2-B1 K=2 median 450.0s is more than 2.0x the 1-GPU best 47.2s
- `sk2ml3-B1#r1`: skipped: sk2ml2-B1 K=2 median 450.0s is more than 2.0x the 1-GPU best 47.2s
- `sk2ml3-B1#r2`: skipped: sk2ml2-B1 K=2 median 449.9s is more than 2.0x the 1-GPU best 47.2s
- `sk2ml3-C1-shared#r0`: skipped: sk2ml2-C1-shared K=2 median 451.0s is more than 2.0x the 1-GPU best 47.2s
- `sk2ml3-C1-shared#r1`: skipped: sk2ml2-C1-shared K=2 median 451.0s is more than 2.0x the 1-GPU best 47.2s
- `sk2ml3-C1-shared#r2`: skipped: sk2ml2-C1-shared K=2 median 451.0s is more than 2.0x the 1-GPU best 47.2s

## Decision points

### DP1 — in-core miner (fastest variant per miner, wall s)

| regime | A | B | C | winner (rule 2) |
|---|---|---|---|---|
| smoke | 0.03 [0.03, 0.03] | 0.04 [0.04, 0.04] | 0.04 [0.04, 0.09] | none |
| deepk | 0.50 [0.50, 0.52] | 0.48 [0.47, 0.49] | 0.43 [0.43, 0.44] | none |
| skew | 0.66 [0.66, 0.67] | 0.66 [0.66, 0.67] | 0.59 [0.59, 0.62] | none |
| oom2 | 10.34 [10.27, 10.41] | 84.93 [84.93, 84.96] | 8.22 [8.21, 8.24] | C |
| sk2ml2 | 56.21 [56.20, 56.37] | 460.20 [460.10, 460.22] | 453.57 [453.55, 453.61] | A |
| sk2ml3 | 734.66 [733.72, 737.58] | FAIL: skipped: sk2ml2-B1 K=2 median 449.9s is more than 2.0x the 1; skipped: sk2ml2-B1 K=2 median 450.0s is more than 2.0x the 1 | FAIL: skipped: sk2ml2-C1-shared K=2 median 451.0s is more than 2.0 | A |
| dsl | 29.82 [29.66, 29.96] | 26.98 [26.88, 27.00] | 24.19 [23.95, 24.27] | C |
| wide | 0.16 [0.16, 0.16] | 0.26 [0.26, 0.26] | 0.16 [0.15, 0.16] | none |
| or005 | 0.13 [0.13, 0.14] | 0.17 [0.16, 0.20] | 0.07 [0.07, 0.11] | none |
| or003 | 0.47 [0.45, 0.49] | 0.49 [0.48, 0.49] | 0.11 [0.11, 0.12] | none |
| or002 | 3.15 [3.12, 3.18] | 2.66 [2.63, 2.71] | 0.35 [0.35, 0.39] | C |

### DP2 — K=2 kernel within A1 (K=2 level s)

| regime | A1-shared | A1-legacy | winner (rule 2) |
|---|---|---|---|
| smoke | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | none |
| deepk | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | none |
| skew | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | none |
| oom2 | 7.99 [7.98, 8.01] | 83.06 [83.03, 83.06] | A1-shared |
| sk2ml2 | 47.21 [47.11, 47.37] | 453.16 [453.16, 453.18] | A1-shared |
| sk2ml3 | 45.09 [45.02, 45.22] | FAIL: skipped: sk2ml2-A1-legacy K=2 median 453.2s is more than 2.0 | A1-shared |
| dsl | 0.07 [0.07, 0.07] | 0.04 [0.04, 0.04] | none |
| wide | 0.02 [0.02, 0.02] | 0.10 [0.10, 0.10] | none |
| or005 | 0.01 [0.01, 0.01] | 0.05 [0.05, 0.05] | none |
| or003 | 0.04 [0.04, 0.04] | 0.11 [0.11, 0.11] | none |
| or002 | 0.11 [0.11, 0.11] | 0.20 [0.20, 0.20] | none |

### DP2 — K=2 kernel within C1 (K=2 level s)

| regime | C1-shared | C1-legacy | winner (rule 2) |
|---|---|---|---|
| smoke | 0.00 [0.00, 0.02] | 0.00 [0.00, 0.02] | none |
| deepk | 0.01 [0.01, 0.01] | 0.00 [0.00, 0.00] | none |
| skew | 0.01 [0.01, 0.01] | 0.00 [0.00, 0.00] | none |
| oom2 | 7.64 [7.62, 7.64] | 82.49 [82.49, 82.49] | C1-shared |
| sk2ml2 | 450.99 [450.97, 451.05] | — | none |
| sk2ml3 | FAIL: skipped: sk2ml2-C1-shared K=2 median 451.0s is more than 2.0 | — | none |
| dsl | 0.07 [0.07, 0.09] | 0.05 [0.05, 0.05] | none |
| wide | 0.02 [0.02, 0.02] | 0.11 [0.11, 0.11] | none |
| or005 | 0.00 [0.00, 0.00] | 0.05 [0.05, 0.05] | none |
| or003 | 0.01 [0.01, 0.01] | 0.09 [0.09, 0.09] | none |
| or002 | 0.01 [0.01, 0.01] | 0.13 [0.13, 0.13] | none |

### DP2 — K=2 kernel within C2 (K=2 level s)

| regime | C2-shared | C2-legacy | winner (rule 2) |
|---|---|---|---|
| smoke | 0.03 [0.03, 0.03] (1) | 0.03 [0.03, 0.03] (1) | none |
| deepk | 0.03 [0.03, 0.03] (1) | 0.03 [0.03, 0.03] (1) | none |
| skew | 0.03 [0.03, 0.03] (1) | 0.03 [0.03, 0.03] (1) | none |
| oom2 | 4.27 [4.27, 4.27] (1) | 42.11 [42.11, 42.11] (1) | C2-shared |
| sk2ml2 | 25.60 [25.60, 25.60] (1) | 328.71 [328.71, 328.71] (1) | C2-shared |

### DP3 — K≥3 counting (Σ K≥3 level s; B = count_k3plus_gpu_resident)

| regime | A1-shared | A1-legacy | B1 | C1-shared | C1-legacy | C1-tiny0 | winner (rule 2) |
|---|---|---|---|---|---|---|---|
| smoke | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | none |
| deepk | 0.13 [0.13, 0.13] | 0.06 [0.05, 0.06] | 0.03 [0.03, 0.03] | 0.03 [0.03, 0.03] | 0.02 [0.02, 0.03] | 0.12 [0.12, 0.12] | none |
| skew | 0.08 [0.08, 0.08] | 0.11 [0.10, 0.11] | 0.07 [0.07, 0.07] | 0.12 [0.11, 0.18] | 0.06 [0.06, 0.07] | 0.06 [0.05, 0.06] | none |
| oom2 | n/a | n/a | n/a | n/a | n/a | — | none |
| sk2ml2 | n/a | n/a | n/a | n/a | — | — | none |
| sk2ml3 | 675.65 [674.52, 677.96] | FAIL: skipped: sk2ml2-A1-legacy K=2 median 453.2s is more than 2.0 | FAIL: skipped: sk2ml2-B1 K=2 median 449.9s is more than 2.0x the 1; skipped: sk2ml2-B1 K=2 median 450.0s is more than 2.0x the 1 | FAIL: skipped: sk2ml2-C1-shared K=2 median 451.0s is more than 2.0 | — | — | A1-shared |
| dsl | 108.04 [108.02, 108.10] | 17.97 [17.93, 18.05] | 14.47 [14.46, 14.47] | 76.39 [76.38, 76.43] | 14.63 [14.58, 14.74] | 105.82 [105.78, 105.82] | none |
| wide | 0.00 [0.00, 0.00] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | none |
| or005 | 0.01 [0.01, 0.01] | 0.03 [0.03, 0.03] | 0.02 [0.02, 0.02] | 0.03 [0.03, 0.03] | 0.03 [0.03, 0.04] | 0.01 [0.01, 0.01] | none |
| or003 | 0.12 [0.12, 0.13] | 0.28 [0.28, 0.28] | 0.07 [0.07, 0.07] | 0.04 [0.04, 0.04] | 0.10 [0.10, 0.10] | 0.04 [0.04, 0.04] | none |
| or002 | 1.26 [1.24, 1.26] | 2.61 [2.59, 2.61] | 0.38 [0.38, 0.45] | 0.29 [0.29, 0.32] | 0.61 [0.61, 0.64] | 0.27 [0.26, 0.28] | none |

### DP3 — K≥3 counting within A1

| regime | A1-shared | A1-legacy | winner (rule 2) |
|---|---|---|---|
| smoke | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | none |
| deepk | 0.13 [0.13, 0.13] | 0.06 [0.05, 0.06] | none |
| skew | 0.08 [0.08, 0.08] | 0.11 [0.10, 0.11] | none |
| oom2 | n/a | n/a | none |
| sk2ml2 | n/a | n/a | none |
| sk2ml3 | 675.65 [674.52, 677.96] | FAIL: skipped: sk2ml2-A1-legacy K=2 median 453.2s is more than 2.0 | A1-shared |
| dsl | 108.04 [108.02, 108.10] | 17.97 [17.93, 18.05] | A1-legacy |
| wide | 0.00 [0.00, 0.00] | 0.01 [0.01, 0.01] | none |
| or005 | 0.01 [0.01, 0.01] | 0.03 [0.03, 0.03] | none |
| or003 | 0.12 [0.12, 0.13] | 0.28 [0.28, 0.28] | none |
| or002 | 1.26 [1.24, 1.26] | 2.61 [2.59, 2.61] | A1-shared |

### DP3 — K≥3 counting within C1

| regime | C1-shared | C1-legacy | C1-tiny0 | winner (rule 2) |
|---|---|---|---|---|
| smoke | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | none |
| deepk | 0.03 [0.03, 0.03] | 0.02 [0.02, 0.03] | 0.12 [0.12, 0.12] | none |
| skew | 0.12 [0.11, 0.18] | 0.06 [0.06, 0.07] | 0.06 [0.05, 0.06] | none |
| oom2 | n/a | n/a | — | none |
| sk2ml2 | n/a | — | — | none |
| sk2ml3 | FAIL: skipped: sk2ml2-C1-shared K=2 median 451.0s is more than 2.0 | — | — | none |
| dsl | 76.39 [76.38, 76.43] | 14.63 [14.58, 14.74] | 105.82 [105.78, 105.82] | C1-legacy |
| wide | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | none |
| or005 | 0.03 [0.03, 0.03] | 0.03 [0.03, 0.04] | 0.01 [0.01, 0.01] | none |
| or003 | 0.04 [0.04, 0.04] | 0.10 [0.10, 0.10] | 0.04 [0.04, 0.04] | none |
| or002 | 0.29 [0.29, 0.32] | 0.61 [0.61, 0.64] | 0.27 [0.26, 0.28] | none |

### DP4 — what tiled cannot serve (wall s)

C1-shared: tiny groups, mega-groups and multi-chunk K=2 on the per-candidate kernel; C1-tiny0: tiny groups tiled too; C1-legacy: everything per-candidate.

| regime | C1-shared | C1-tiny0 | C1-legacy | winner (rule 2) |
|---|---|---|---|---|
| smoke | 0.04 [0.04, 0.09] | 0.04 [0.04, 0.09] | 0.04 [0.04, 0.09] | none |
| deepk | 0.45 [0.45, 0.45] | 0.53 [0.53, 0.54] | 0.43 [0.43, 0.44] | none |
| skew | 0.67 [0.63, 0.73] | 0.60 [0.59, 0.60] | 0.59 [0.59, 0.62] | none |
| oom2 | 8.22 [8.21, 8.24] | — | 83.09 [83.09, 83.09] | C1-shared |
| sk2ml2 | 453.57 [453.55, 453.61] | — | — | none |
| sk2ml3 | FAIL: skipped: sk2ml2-C1-shared K=2 median 451.0s is more than 2.0 | — | — | none |
| dsl | 86.00 [85.95, 86.01] | 115.33 [115.23, 115.36] | 24.19 [23.95, 24.27] | C1-legacy |
| wide | 0.16 [0.15, 0.16] | 0.16 [0.15, 0.17] | 0.26 [0.25, 0.27] | none |
| or005 | 0.09 [0.09, 0.13] | 0.07 [0.07, 0.11] | 0.14 [0.14, 0.18] | none |
| or003 | 0.12 [0.12, 0.13] | 0.11 [0.11, 0.12] | 0.25 [0.25, 0.28] | none |
| or002 | 0.38 [0.37, 0.44] | 0.35 [0.35, 0.39] | 0.82 [0.82, 0.87] | none |

### DP5 — layout on C1-shared (wall s)

| regime | C1-shared | C1-shared-auto | C1-shared-k3 | winner (rule 2) |
|---|---|---|---|---|
| smoke | 0.04 [0.04, 0.09] | 0.06 [0.06, 0.12] | 0.04 [0.04, 0.10] | none |
| deepk | 0.45 [0.45, 0.45] | 0.59 [0.59, 0.62] | 0.79 [0.79, 0.80] | none |
| skew | 0.67 [0.63, 0.73] | 0.77 [0.76, 0.78] | 1.12 [1.12, 1.14] | none |
| oom2 | 8.22 [8.21, 8.24] | — | — | none |
| sk2ml2 | 453.57 [453.55, 453.61] | — | — | none |
| sk2ml3 | FAIL: skipped: sk2ml2-C1-shared K=2 median 451.0s is more than 2.0 | — | — | none |
| dsl | 86.00 [85.95, 86.01] | FAIL: error: OutOfMemoryError: Out of memory allocating 2,385,000, | FAIL: error: OutOfMemoryError: Out of memory allocating 5,385,686, | C1-shared |
| wide | 0.16 [0.15, 0.16] | — | — | none |
| or005 | 0.09 [0.09, 0.13] | — | — | none |
| or003 | 0.12 [0.12, 0.13] | 0.17 [0.17, 0.17] | 0.17 [0.14, 0.17] | none |
| or002 | 0.38 [0.37, 0.44] | 0.47 [0.46, 0.51] | 0.46 [0.45, 0.50] | none |

### DP5 — layout on A1-shared (wall s)

| regime | A1-shared | A1-shared-auto | winner (rule 2) |
|---|---|---|---|
| smoke | 0.03 [0.03, 0.03] | 0.06 [0.06, 0.12] | none |
| deepk | 0.58 [0.57, 0.59] | 0.64 [0.62, 0.67] | none |
| skew | 0.66 [0.66, 0.67] | 0.84 [0.83, 0.87] | none |
| oom2 | 10.34 [10.27, 10.41] | — | none |
| sk2ml2 | 56.21 [56.20, 56.37] | — | none |
| sk2ml3 | 734.66 [733.72, 737.58] | — | none |
| dsl | 119.76 [119.72, 119.84] | FAIL: error: OutOfMemoryError: Out of memory allocating 2,385,000, | A1-shared |
| wide | 0.16 [0.16, 0.16] | — | none |
| or005 | 0.13 [0.13, 0.14] | — | none |
| or003 | 0.47 [0.45, 0.49] | 0.46 [0.45, 0.49] | none |
| or002 | 3.15 [3.12, 3.18] | 2.64 [2.60, 2.66] | none |

### DP5 — layout on C1-legacy (wall s)

| regime | C1-legacy | C1-legacy-auto | C1-legacy-k3 | winner (rule 2) |
|---|---|---|---|---|
| smoke | 0.04 [0.04, 0.09] | 0.06 [0.06, 0.12] | 0.04 [0.04, 0.13] | none |
| deepk | 0.43 [0.43, 0.44] | 0.58 [0.58, 0.60] | 0.80 [0.79, 0.80] | none |
| skew | 0.59 [0.59, 0.62] | 0.72 [0.71, 0.76] | 1.10 [1.10, 1.14] | none |
| oom2 | 83.09 [83.09, 83.09] | — | — | none |
| dsl | 24.19 [23.95, 24.27] | FAIL: error: OutOfMemoryError: Out of memory allocating 2,385,000, | FAIL: error: OutOfMemoryError: Out of memory allocating 5,385,686, | C1-legacy |
| wide | 0.26 [0.25, 0.27] | — | — | none |
| or005 | 0.14 [0.14, 0.18] | — | — | none |
| or003 | 0.25 [0.25, 0.28] | 0.23 [0.23, 0.23] | 0.22 [0.21, 0.22] | none |
| or002 | 0.82 [0.82, 0.87] | 0.58 [0.57, 0.67] | 0.56 [0.56, 0.61] | none |

### DP5 — layout on A1-legacy (wall s)

| regime | A1-legacy | A1-legacy-auto | winner (rule 2) |
|---|---|---|---|
| smoke | 0.03 [0.03, 0.04] | 0.06 [0.06, 0.12] | none |
| deepk | 0.50 [0.50, 0.52] | 0.63 [0.62, 0.68] | none |
| skew | 0.68 [0.68, 0.68] | 0.87 [0.84, 0.87] | none |
| oom2 | 85.37 [85.37, 85.40] | — | none |
| sk2ml2 | 462.11 [462.07, 462.26] | — | none |
| sk2ml3 | FAIL: skipped: sk2ml2-A1-legacy K=2 median 453.2s is more than 2.0 | — | none |
| dsl | 29.82 [29.66, 29.96] | FAIL: error: OutOfMemoryError: Out of memory allocating 2,385,000,; error: OutOfMemoryError: Out of memory allocating 2,385,000,; error: OutOfMemoryError: Out of memory allocating 2,385,000, | A1-legacy |
| wide | 0.26 [0.26, 0.26] | — | none |
| or005 | 0.17 [0.17, 0.21] | — | none |
| or003 | 0.69 [0.68, 0.72] | 0.52 [0.52, 0.54] | none |
| or002 | 4.57 [4.55, 4.62] | 2.71 [2.69, 2.73] | A1-legacy-auto |

### DP6 — multi-GPU (Σ K≥2 level s)

| regime | C2-shared | C2-legacy | Asplit2-shared | Bsplit2 | winner (rule 2) |
|---|---|---|---|---|---|
| smoke | 0.04 [0.04, 0.04] (1) | 0.04 [0.04, 0.04] (1) | — | — | none |
| deepk | 0.07 [0.07, 0.07] (1) | 0.07 [0.07, 0.07] (1) | — | — | none |
| skew | 0.18 [0.18, 0.18] (1) | 0.16 [0.16, 0.16] (1) | — | — | none |
| oom2 | 4.27 [4.27, 4.27] (1) | 42.11 [42.11, 42.11] (1) | — | — | C2-shared |
| sk2ml2 | 25.60 [25.60, 25.60] (1) | 328.71 [328.71, 328.71] (1) | — | — | C2-shared |

C1 vs C2 (wall s), for scaling:

### DP6 — row-split scaling

| regime | C1-shared | C2-shared | C1-legacy | C2-legacy | winner (rule 2) |
|---|---|---|---|---|---|
| smoke | 0.04 [0.04, 0.09] | 0.66 [0.66, 0.66] (1) | 0.04 [0.04, 0.09] | 0.60 [0.60, 0.60] (1) | none |
| deepk | 0.45 [0.45, 0.45] | 0.92 [0.92, 0.92] (1) | 0.43 [0.43, 0.44] | 0.90 [0.90, 0.90] (1) | none |
| skew | 0.67 [0.63, 0.73] | 0.99 [0.99, 0.99] (1) | 0.59 [0.59, 0.62] | 1.03 [1.03, 1.03] (1) | none |
| oom2 | 8.22 [8.21, 8.24] | 5.07 [5.07, 5.07] (1) | 83.09 [83.09, 83.09] | 43.12 [43.12, 43.12] (1) | C2-shared |
| sk2ml2 | 453.57 [453.55, 453.61] | 28.45 [28.45, 28.45] (1) | — | 331.27 [331.27, 331.27] (1) | C2-shared |
| sk2ml3 | FAIL: skipped: sk2ml2-C1-shared K=2 median 451.0s is more than 2.0 | — | — | — | none |
| dsl | 86.00 [85.95, 86.01] | — | 24.19 [23.95, 24.27] | — | C1-legacy |
| wide | 0.16 [0.15, 0.16] | — | 0.26 [0.25, 0.27] | — | none |
| or005 | 0.09 [0.09, 0.13] | — | 0.14 [0.14, 0.18] | — | none |
| or003 | 0.12 [0.12, 0.13] | — | 0.25 [0.25, 0.28] | — | none |
| or002 | 0.38 [0.37, 0.44] | — | 0.82 [0.82, 0.87] | — | none |

### DP7 — SON (wall s; per-pass s from the rep with the median wall)

| regime | config | wall | pass 1 | pass 2 | chunks |
|---|---|---|---|---|---|
| dsl | D1-resident | 194.75 [194.75, 194.75] (1) | 58.462 | 132.551 | 4 |
| dsl | D1-gpu | >600 (timeout) | — | — | — |
| dsl | E2 | >600 (timeout) | — | — | — |
| deepk | D1-resident | 3.69 [3.36, 3.76] | 1.951 | 1.701 | 4 |
| deepk | D1-gpu | 22.64 [22.31, 22.67] | 15.76 | 6.839 | 4 |
| deepk | E2 | 22.79 [22.79, 22.79] (1) | 15.226 | 7.515 | 4 |
| deepk | D1-cpu | 12.21 [11.98, 12.26] | 8.664 | 3.507 | 4 |

### DP8 — CPU tier (wall s)

| regime | F-polars | F-sparse | F-sparse-norust | F-auto | winner (rule 2) |
|---|---|---|---|---|---|
| smoke | 0.18 [0.18, 0.19] | 0.41 [0.40, 0.42] | 0.79 [0.78, 0.80] | 0.19 [0.17, 0.19] | none |
| deepk | 2.64 [2.44, 2.67] | 5.46 [5.15, 5.58] | 62.20 [61.03, 63.17] | 2.61 [2.50, 2.73] | none |
| skew | 3.52 [3.31, 3.52] | 6.71 [6.54, 7.00] | 148.49 [147.64, 148.87] | 3.49 [3.35, 3.55] | none |
| wide | 40.08 [39.93, 40.84] | 41.59 [41.21, 42.06] | 46.21 [46.04, 46.89] | 41.47 [40.85, 41.60] | none |
| or005 | 16.14 [15.58, 16.21] | 16.81 [16.42, 17.12] | 33.09 [31.27, 33.89] | 17.02 [16.73, 17.19] | none |
| or003 | 44.77 [44.53, 44.77] | 39.81 [39.04, 39.97] | 228.06 [203.84, 228.96] | 39.43 [39.35, 39.71] | none |

### DP8 — R1: CPU K>2 counting (Σ K≥3 level s)

| regime | F-sparse | F-sparse-norust | winner (rule 2) |
|---|---|---|---|
| smoke | 0.13 [0.12, 0.13] | 0.55 [0.52, 0.56] | none |
| deepk | 3.25 [3.11, 3.35] | 60.09 [59.03, 60.99] | F-sparse |
| skew | 4.08 [3.93, 4.20] | 145.87 [144.93, 146.15] | F-sparse |
| wide | 1.03 [1.00, 1.04] | 5.84 [5.72, 5.95] | F-sparse |
| or005 | 1.32 [1.31, 1.32] | 17.28 [15.88, 18.60] | F-sparse |
| or003 | 9.52 [9.26, 9.56] | 197.14 [173.90, 198.71] | F-sparse |

### DP9 — host roles on C1-shared (wall s)

| regime | C1-shared | C1-shared-norust | C1-shared-noprune | winner (rule 2) |
|---|---|---|---|---|
| smoke | 0.04 [0.04, 0.09] | 0.04 [0.04, 0.10] | 0.04 [0.04, 0.09] | none |
| deepk | 0.45 [0.45, 0.45] | 0.48 [0.48, 0.49] | 0.45 [0.45, 0.45] | none |
| skew | 0.67 [0.63, 0.73] | 0.70 [0.68, 0.74] | 0.78 [0.77, 0.82] | none |
| oom2 | 8.22 [8.21, 8.24] | — | — | none |
| sk2ml2 | 453.57 [453.55, 453.61] | — | — | none |
| sk2ml3 | FAIL: skipped: sk2ml2-C1-shared K=2 median 451.0s is more than 2.0 | — | — | none |
| dsl | 86.00 [85.95, 86.01] | 87.53 [87.35, 87.65] | 86.27 [86.20, 86.64] | none |
| wide | 0.16 [0.15, 0.16] | — | — | none |
| or005 | 0.09 [0.09, 0.13] | — | — | none |
| or003 | 0.12 [0.12, 0.13] | 0.34 [0.33, 0.34] | 0.10 [0.10, 0.14] | none |
| or002 | 0.38 [0.37, 0.44] | 2.46 [2.43, 2.54] | 0.22 [0.21, 0.24] | none |

### DP9 — host roles on C1-legacy (wall s)

| regime | C1-legacy | C1-legacy-norust | C1-legacy-noprune | winner (rule 2) |
|---|---|---|---|---|
| smoke | 0.04 [0.04, 0.09] | 0.04 [0.04, 0.09] | 0.04 [0.04, 0.11] | none |
| deepk | 0.43 [0.43, 0.44] | 0.46 [0.46, 0.47] | 0.45 [0.44, 0.45] | none |
| skew | 0.59 [0.59, 0.62] | 0.63 [0.62, 0.72] | 0.67 [0.66, 0.71] | none |
| oom2 | 83.09 [83.09, 83.09] | — | — | none |
| dsl | 24.19 [23.95, 24.27] | 25.47 [25.47, 25.66] | 24.50 [24.41, 24.53] | none |
| wide | 0.26 [0.25, 0.27] | — | — | none |
| or005 | 0.14 [0.14, 0.18] | — | — | none |
| or003 | 0.25 [0.25, 0.28] | 0.47 [0.47, 0.52] | 0.23 [0.23, 0.28] | none |
| or002 | 0.82 [0.82, 0.87] | 2.84 [2.82, 2.88] | 0.64 [0.64, 0.70] | none |

### DP9 — free-set runs (R4) (wall s)

| regime | C1-legacy-free | C1-legacy-free-norust | winner (rule 2) |
|---|---|---|---|
| smoke | 0.03 [0.03, 0.09] | 0.04 [0.04, 0.09] | none |
| deepk | 0.43 [0.42, 0.44] | 0.44 [0.43, 0.44] | none |
| dsl | 15.50 [15.49, 15.55] | 16.65 [16.54, 16.68] | none |
| or002 | 0.85 [0.84, 0.90] | 3.39 [3.38, 3.39] | C1-legacy-free |

### DP10 — survivor filter on C2-shared (wall s)

| regime | C2-shared | C2-shared-filter-cupy | C2-shared-filter-cpu | winner (rule 2) |
|---|---|---|---|---|
| smoke | 0.66 [0.66, 0.66] (1) | — | — | none |
| deepk | 0.92 [0.92, 0.92] (1) | — | — | none |
| skew | 0.99 [0.99, 0.99] (1) | — | — | none |
| oom2 | 5.07 [5.07, 5.07] (1) | — | — | none |
| sk2ml2 | 28.45 [28.45, 28.45] (1) | 22.94 [22.94, 22.94] (1) | 24.94 [24.94, 24.94] (1) | none |

### DP10 — survivor filter on C2-legacy (wall s)

| regime | C2-legacy | C2-legacy-filter-cupy | C2-legacy-filter-cpu | winner (rule 2) |
|---|---|---|---|---|
| smoke | 0.60 [0.60, 0.60] (1) | — | — | none |
| deepk | 0.90 [0.90, 0.90] (1) | — | — | none |
| skew | 1.03 [1.03, 1.03] (1) | — | — | none |
| oom2 | 43.12 [43.12, 43.12] (1) | — | — | none |
| sk2ml2 | 331.27 [331.27, 331.27] (1) | — | — | none |
| dsl | — | 17.85 [17.85, 17.85] (1) | 17.89 [17.89, 17.89] (1) | none |

### DP10 — row balance (wall s)

| regime | C2-legacy | C2-legacy-balance-nnz | winner (rule 2) |
|---|---|---|---|
| smoke | 0.60 [0.60, 0.60] (1) | — | none |
| deepk | 0.90 [0.90, 0.90] (1) | — | none |
| skew | 1.03 [1.03, 1.03] (1) | 1.04 [1.04, 1.04] (1) | none |
| oom2 | 43.12 [43.12, 43.12] (1) | — | none |
| sk2ml2 | 331.27 [331.27, 331.27] (1) | — | none |
| dsl | — | 18.36 [18.36, 18.36] (1) | none |

### DP9 — per-call microbench, Rust vs fallback (s, median of reps)

| call | size | Rust | fallback | fallback / Rust |
|---|---|---|---|---|
| deep_sparse_large/r2_k10.npz | rows=88,698, k=10 | 0.0026 | 0.0168 | 6.5× |
| deep_sparse_large/r2_k11.npz | rows=53,910, k=11 | 0.0011 | 0.0087 | 7.6× |
| deep_sparse_large/r2_k12.npz | rows=25,554, k=12 | 0.0006 | 0.0039 | 6.5× |
| deep_sparse_large/r2_k13.npz | rows=9,336, k=13 | 0.0003 | 0.0014 | 5.4× |
| deep_sparse_large/r2_k14.npz | rows=2,564, k=14 | 0.0001 | 0.0004 | 5.5× |
| deep_sparse_large/r2_k15.npz | rows=504, k=15 | 0.0000 | 0.0001 | 9.8× |
| deep_sparse_large/r2_k16.npz | rows=64, k=16 | 0.0000 | 0.0001 | 10.2× |
| deep_sparse_large/r2_k3.npz | rows=1,058, k=3 | 0.0000 | 0.0001 | 9.1× |
| deep_sparse_large/r2_k4.npz | rows=6,043, k=4 | 0.0002 | 0.0005 | 2.1× |
| deep_sparse_large/r2_k5.npz | rows=21,430, k=5 | 0.0005 | 0.0021 | 4.4× |
| deep_sparse_large/r2_k6.npz | rows=51,243, k=6 | 0.0010 | 0.0062 | 6.1× |
| deep_sparse_large/r2_k7.npz | rows=88,374, k=7 | 0.0017 | 0.0122 | 7.1× |
| deep_sparse_large/r2_k8.npz | rows=114,509, k=8 | 0.0023 | 0.0198 | 8.6× |
| deep_sparse_large/r2_k9.npz | rows=114,174, k=9 | 0.0023 | 0.0215 | 9.2× |
| deep_sparse_large/r3_k10.npz | candidates=70,690, groups=21,245, prev_rows=88,698, k=10 | 0.0137 | 0.2186 | 15.9× |
| deep_sparse_large/r3_k11.npz | candidates=30,802, groups=12,346, prev_rows=53,910, k=11 | 0.0066 | 0.1229 | 18.6× |
| deep_sparse_large/r3_k12.npz | candidates=10,336, groups=5,382, prev_rows=25,554, k=12 | 0.0031 | 0.0462 | 15.1× |
| deep_sparse_large/r3_k13.npz | candidates=2,652, groups=1,723, prev_rows=9,336, k=13 | 0.0012 | 0.0148 | 12.5× |
| deep_sparse_large/r3_k14.npz | candidates=504, groups=388, prev_rows=2,564, k=14 | 0.0004 | 0.0034 | 9.5× |
| deep_sparse_large/r3_k15.npz | candidates=64, groups=56, prev_rows=504, k=15 | 0.0001 | 0.0005 | 3.9× |
| deep_sparse_large/r3_k16.npz | candidates=4, groups=4, prev_rows=64, k=16 | 0.0001 | 0.0001 | 0.9× |
| deep_sparse_large/r3_k3.npz | candidates=15,673, groups=55, prev_rows=1,058, k=3 | 0.0002 | 0.0015 | 7.0× |
| deep_sparse_large/r3_k4.npz | candidates=43,877, groups=671, prev_rows=6,043, k=4 | 0.0011 | 0.0183 | 16.3× |
| deep_sparse_large/r3_k5.npz | candidates=92,935, groups=3,410, prev_rows=21,430, k=5 | 0.0035 | 0.0652 | 18.6× |
| deep_sparse_large/r3_k6.npz | candidates=148,891, groups=10,001, prev_rows=51,243, k=6 | 0.0078 | 0.1413 | 18.2× |
| deep_sparse_large/r3_k7.npz | candidates=181,999, groups=19,489, prev_rows=88,374, k=7 | 0.0136 | 0.2390 | 17.6× |
| deep_sparse_large/r3_k8.npz | candidates=171,479, groups=27,015, prev_rows=114,509, k=8 | 0.0171 | 0.3064 | 18.0× |
| deep_sparse_large/r3_k9.npz | candidates=125,178, groups=27,628, prev_rows=114,174, k=9 | 0.0162 | 0.2975 | 18.3× |
| deep_sparse_large-free/r2_k10.npz | rows=11,800, k=10 | 0.0002 | 0.0019 | 11.7× |
| deep_sparse_large-free/r2_k11.npz | rows=2,500, k=11 | 0.0000 | 0.0004 | 11.1× |
| deep_sparse_large-free/r2_k3.npz | rows=1,058, k=3 | 0.0000 | 0.0001 | 8.4× |
| deep_sparse_large-free/r2_k4.npz | rows=6,043, k=4 | 0.0000 | 0.0006 | 12.7× |
| deep_sparse_large-free/r2_k5.npz | rows=19,327, k=5 | 0.0002 | 0.0021 | 13.1× |
| deep_sparse_large-free/r2_k6.npz | rows=40,364, k=6 | 0.0004 | 0.0052 | 13.0× |
| deep_sparse_large-free/r2_k7.npz | rows=52,788, k=7 | 0.0006 | 0.0077 | 12.7× |
| deep_sparse_large-free/r2_k8.npz | rows=44,635, k=8 | 0.0006 | 0.0069 | 12.3× |
| deep_sparse_large-free/r2_k9.npz | rows=26,372, k=9 | 0.0003 | 0.0042 | 12.2× |
| deep_sparse_large-free/r3_k10.npz | candidates=4,967, groups=2,954, prev_rows=17,514, k=10 | 0.0019 | 0.0230 | 12.0× |
| deep_sparse_large-free/r3_k11.npz | candidates=401, groups=319, prev_rows=4,967, k=11 | 0.0005 | 0.0031 | 6.5× |
| deep_sparse_large-free/r3_k3.npz | candidates=15,673, groups=55, prev_rows=1,058, k=3 | 0.0002 | 0.0015 | 6.3× |
| deep_sparse_large-free/r3_k4.npz | candidates=43,877, groups=671, prev_rows=6,043, k=4 | 0.0013 | 0.0178 | 14.2× |
| deep_sparse_large-free/r3_k5.npz | candidates=85,191, groups=3,005, prev_rows=21,430, k=5 | 0.0035 | 0.0604 | 17.5× |
| deep_sparse_large-free/r3_k6.npz | candidates=116,734, groups=7,801, prev_rows=46,966, k=6 | 0.0069 | 0.1137 | 16.5× |
| deep_sparse_large-free/r3_k7.npz | candidates=98,724, groups=12,126, prev_rows=71,154, k=7 | 0.0091 | 0.1452 | 16.0× |
| deep_sparse_large-free/r3_k8.npz | candidates=51,062, groups=11,905, prev_rows=68,838, k=8 | 0.0083 | 0.1349 | 16.3× |
| deep_sparse_large-free/r3_k9.npz | candidates=18,114, groups=7,404, prev_rows=42,668, k=9 | 0.0046 | 0.0692 | 15.2× |
| deep_sparse_large-free/r4_k10.npz | rows=4,967, prev_rows=17,514, k=10 | 0.0008 | 0.0178 | 22.4× |
| deep_sparse_large-free/r4_k11.npz | rows=401, prev_rows=4,967, k=11 | 0.0002 | 0.0022 | 12.7× |
| deep_sparse_large-free/r4_k2.npz | rows=1,058, prev_rows=85, k=2 | 0.0001 | 0.0006 | 6.1× |
| deep_sparse_large-free/r4_k3.npz | rows=6,043, prev_rows=1,058, k=3 | 0.0003 | 0.0055 | 21.3× |
| deep_sparse_large-free/r4_k4.npz | rows=21,430, prev_rows=6,043, k=4 | 0.0010 | 0.0248 | 25.8× |
| deep_sparse_large-free/r4_k5.npz | rows=46,966, prev_rows=21,430, k=5 | 0.0026 | 0.0707 | 27.4× |
| deep_sparse_large-free/r4_k6.npz | rows=71,154, prev_rows=46,966, k=6 | 0.0046 | 0.1259 | 27.1× |
| deep_sparse_large-free/r4_k7.npz | rows=68,838, prev_rows=71,154, k=7 | 0.0054 | 0.1445 | 27.0× |
| deep_sparse_large-free/r4_k8.npz | rows=42,668, prev_rows=68,838, k=8 | 0.0041 | 0.1078 | 26.0× |
| deep_sparse_large-free/r4_k9.npz | rows=17,514, prev_rows=42,668, k=9 | 0.0022 | 0.0558 | 25.6× |
| stress_k2/r2_k3.npz | rows=1,660,332, k=3 | 0.0138 | 0.0831 | 6.0× |
| stress_k2/r3_k3.npz | candidates=11,796,796,856, groups=784, prev_rows=1,660,332, k=3 | 487.2035 | >300.0000 | >0.6× |
| stress_k2-free/r4_k2.npz | rows=1,660,332, prev_rows=35,000, k=2 | 0.0387 | 0.9498 | 24.6× |

## All configs

| config | wall s | K=2 s | K≥3 s | peak VRAM MB | peak RSS MB | throttled | status |
|---|---|---|---|---|---|---|---|
| deepk-A1-legacy | 0.50 [0.50, 0.52] | 0.00 [0.00, 0.00] | 0.06 [0.05, 0.06] | 144 | 1040 |  | ok |
| deepk-A1-legacy-auto | 0.63 [0.62, 0.68] | 0.00 [0.00, 0.00] | 0.18 [0.18, 0.21] | 324 | 1142 |  | ok |
| deepk-A1-shared | 0.58 [0.57, 0.59] | 0.00 [0.00, 0.00] | 0.13 [0.13, 0.13] | 228 | 1113 |  | ok |
| deepk-A1-shared-auto | 0.64 [0.62, 0.67] | 0.00 [0.00, 0.00] | 0.18 [0.18, 0.21] | 394 | 1136 |  | ok |
| deepk-B1 | 0.48 [0.47, 0.49] | 0.00 [0.00, 0.00] | 0.03 [0.03, 0.03] | 144 | 1143 |  | ok |
| deepk-C1-legacy | 0.43 [0.43, 0.44] | 0.00 [0.00, 0.00] | 0.02 [0.02, 0.03] | 144 | 867 |  | ok |
| deepk-C1-legacy-auto | 0.58 [0.58, 0.60] | 0.00 [0.00, 0.00] | 0.10 [0.10, 0.10] | 156 | 1213 |  | ok |
| deepk-C1-legacy-free | 0.43 [0.42, 0.44] | 0.00 [0.00, 0.00] | 0.01 [0.01, 0.01] | 144 | 860 |  | ok |
| deepk-C1-legacy-free-norust | 0.44 [0.43, 0.44] | 0.00 [0.00, 0.00] | 0.02 [0.01, 0.02] | 144 | 834 |  | ok |
| deepk-C1-legacy-k3 | 0.80 [0.79, 0.80] | 0.00 [0.00, 0.00] | 0.17 [0.16, 0.17] | 392 | 1207 |  | ok |
| deepk-C1-legacy-noprune | 0.45 [0.44, 0.45] | 0.00 [0.00, 0.00] | 0.03 [0.03, 0.03] | 144 | 854 |  | ok |
| deepk-C1-legacy-norust | 0.46 [0.46, 0.47] | 0.00 [0.00, 0.00] | 0.04 [0.04, 0.04] | 144 | 840 |  | ok |
| deepk-C1-shared | 0.45 [0.45, 0.45] | 0.01 [0.01, 0.01] | 0.03 [0.03, 0.03] | 144 | 857 |  | ok |
| deepk-C1-shared-auto | 0.59 [0.59, 0.62] | 0.01 [0.01, 0.01] | 0.10 [0.10, 0.13] | 324 | 1215 |  | ok |
| deepk-C1-shared-k3 | 0.79 [0.79, 0.80] | 0.01 [0.01, 0.01] | 0.16 [0.16, 0.16] | 392 | 1213 |  | ok |
| deepk-C1-shared-noprune | 0.45 [0.45, 0.45] | 0.01 [0.01, 0.01] | 0.03 [0.03, 0.03] | 144 | 838 |  | ok |
| deepk-C1-shared-norust | 0.48 [0.48, 0.49] | 0.01 [0.01, 0.01] | 0.05 [0.05, 0.06] | 144 | 850 |  | ok |
| deepk-C1-tiny0 | 0.53 [0.53, 0.54] | 0.01 [0.01, 0.01] | 0.12 [0.12, 0.12] | 144 | 1176 |  | ok |
| deepk-C2-legacy | 0.90 [0.90, 0.90] (1) | 0.03 [0.03, 0.03] (1) | 0.04 [0.04, 0.04] (1) | 244 | 1444 |  | ok |
| deepk-C2-shared | 0.92 [0.92, 0.92] (1) | 0.03 [0.03, 0.03] (1) | 0.03 [0.03, 0.03] (1) | 244 | 1446 |  | ok |
| deepk-D1-cpu | 12.21 [11.98, 12.26] | n/a | n/a | 140 | 673 |  | ok |
| deepk-D1-gpu | 22.64 [22.31, 22.67] | n/a | n/a | 214 | 764 |  | ok |
| deepk-D1-resident | 3.69 [3.36, 3.76] | n/a | n/a | 146 | 786 |  | ok |
| deepk-E2 | 22.79 [22.79, 22.79] (1) | n/a | n/a | 164 | 936 |  | ok |
| deepk-F-auto | 2.61 [2.50, 2.73] | 0.19 [0.17, 0.20] | 1.24 [1.24, 1.28] | 36 | 555 |  | ok |
| deepk-F-polars | 2.64 [2.44, 2.67] | 0.17 [0.17, 0.18] | 1.24 [1.19, 1.25] | 36 | 567 |  | ok |
| deepk-F-sparse | 5.46 [5.15, 5.58] | 0.92 [0.91, 0.93] | 3.25 [3.11, 3.35] | 36 | 1370 |  | ok |
| deepk-F-sparse-norust | 62.20 [61.03, 63.17] | 0.91 [0.90, 0.94] | 60.09 [59.03, 60.99] | 36 | 1480 |  | ok |
| dsl-A1-legacy | 29.82 [29.66, 29.96] | 0.04 [0.04, 0.04] | 17.97 [17.93, 18.05] | 2216 | 11348 | [0] | ok |
| dsl-A1-legacy-auto | FAIL: error: OutOfMemoryError: Out of memory allocating 2,385,000,; error: OutOfMemoryError: Out of memory allocating 2,385,000,; error: OutOfMemoryError: Out of memory allocating 2,385,000, | FAIL: error: OutOfMemoryError: Out of memory allocating 2,385,000,; error: OutOfMemoryError: Out of memory allocating 2,385,000,; error: OutOfMemoryError: Out of memory allocating 2,385,000, | FAIL: error: OutOfMemoryError: Out of memory allocating 2,385,000,; error: OutOfMemoryError: Out of memory allocating 2,385,000,; error: OutOfMemoryError: Out of memory allocating 2,385,000, | 0 | 11479 |  | error: OutOfMemoryError: Out of memory a, error: OutOfMemoryError: Out of memory a, error: OutOfMemoryError: Out of memory a |
| dsl-A1-shared | 119.76 [119.72, 119.84] | 0.07 [0.07, 0.07] | 108.04 [108.02, 108.10] | 2216 | 11329 | [0] | ok |
| dsl-A1-shared-auto | FAIL: error: OutOfMemoryError: Out of memory allocating 2,385,000, | FAIL: error: OutOfMemoryError: Out of memory allocating 2,385,000, | FAIL: error: OutOfMemoryError: Out of memory allocating 2,385,000, | 0 | 11458 |  | error: OutOfMemoryError: Out of memory a |
| dsl-B1 | 26.98 [26.88, 27.00] | 0.04 [0.04, 0.04] | 14.47 [14.46, 14.47] | 2216 | 11485 | [0] | ok |
| dsl-C1-legacy | 24.19 [23.95, 24.27] | 0.05 [0.05, 0.05] | 14.63 [14.58, 14.74] | 2216 | 11584 | [0] | ok |
| dsl-C1-legacy-auto | FAIL: error: OutOfMemoryError: Out of memory allocating 2,385,000, | FAIL: error: OutOfMemoryError: Out of memory allocating 2,385,000, | FAIL: error: OutOfMemoryError: Out of memory allocating 2,385,000, | 0 | 11627 |  | error: OutOfMemoryError: Out of memory a |
| dsl-C1-legacy-free | 15.50 [15.49, 15.55] | 0.05 [0.05, 0.05] | 6.01 [6.00, 6.07] | 2216 | 11546 | [0] | ok |
| dsl-C1-legacy-free-norust | 16.65 [16.54, 16.68] | 0.05 [0.05, 0.05] | 7.14 [7.12, 7.23] | 2216 | 11444 | [0] | ok |
| dsl-C1-legacy-k3 | FAIL: error: OutOfMemoryError: Out of memory allocating 5,385,686, | FAIL: error: OutOfMemoryError: Out of memory allocating 5,385,686, | FAIL: error: OutOfMemoryError: Out of memory allocating 5,385,686, | 0 | 11494 |  | error: OutOfMemoryError: Out of memory a |
| dsl-C1-legacy-noprune | 24.50 [24.41, 24.53] | 0.05 [0.05, 0.05] | 14.84 [14.82, 14.86] | 2216 | 11414 | [0] | ok |
| dsl-C1-legacy-norust | 25.47 [25.47, 25.66] | 0.05 [0.05, 0.05] | 15.98 [15.97, 16.04] | 2216 | 11532 | [0] | ok |
| dsl-C1-shared | 86.00 [85.95, 86.01] | 0.07 [0.07, 0.09] | 76.39 [76.38, 76.43] | 2216 | 11430 | [0] | ok |
| dsl-C1-shared-auto | FAIL: error: OutOfMemoryError: Out of memory allocating 2,385,000, | FAIL: error: OutOfMemoryError: Out of memory allocating 2,385,000, | FAIL: error: OutOfMemoryError: Out of memory allocating 2,385,000, | 0 | 11475 |  | error: OutOfMemoryError: Out of memory a |
| dsl-C1-shared-k3 | FAIL: error: OutOfMemoryError: Out of memory allocating 5,385,686, | FAIL: error: OutOfMemoryError: Out of memory allocating 5,385,686, | FAIL: error: OutOfMemoryError: Out of memory allocating 5,385,686, | 0 | 11444 |  | error: OutOfMemoryError: Out of memory a |
| dsl-C1-shared-noprune | 86.27 [86.20, 86.64] | 0.07 [0.07, 0.07] | 76.75 [76.65, 77.00] | 2216 | 11307 | [0] | ok |
| dsl-C1-shared-norust | 87.53 [87.35, 87.65] | 0.07 [0.07, 0.07] | 77.92 [77.81, 78.07] | 2216 | 11486 | [0] | ok |
| dsl-C1-tiny0 | 115.33 [115.23, 115.36] | 0.07 [0.07, 0.07] | 105.82 [105.78, 105.82] | 2216 | 11470 | [0] | ok |
| dsl-C2-legacy-balance-nnz | 18.36 [18.36, 18.36] (1) | 0.05 [0.05, 0.05] (1) | 8.58 [8.58, 8.58] (1) | 1238 | 12022 | [0, 1] | ok |
| dsl-C2-legacy-filter-cpu | 17.89 [17.89, 17.89] (1) | 0.05 [0.05, 0.05] (1) | 8.42 [8.42, 8.42] (1) | 1236 | 11814 | [0, 1] | ok |
| dsl-C2-legacy-filter-cupy | 17.85 [17.85, 17.85] (1) | 0.06 [0.06, 0.06] (1) | 8.08 [8.08, 8.08] (1) | 1238 | 11788 | [0, 1] | ok |
| dsl-D1-gpu | >600 (timeout) | >600 (timeout) | >600 (timeout) | 0 | 0 |  | timeout |
| dsl-D1-resident | 194.75 [194.75, 194.75] (1) | n/a | n/a | 680 | 6732 |  | ok |
| dsl-E2 | >600 (timeout) | >600 (timeout) | >600 (timeout) | 0 | 0 |  | timeout |
| oom2-A1-legacy | 85.37 [85.37, 85.40] | 83.06 [83.03, 83.06] | n/a | 2270 | 1296 | [0] | ok |
| oom2-A1-shared | 10.34 [10.27, 10.41] | 7.99 [7.98, 8.01] | n/a | 2270 | 1291 | [0] | ok |
| oom2-B1 | 84.93 [84.93, 84.96] | 82.34 [82.31, 82.34] | n/a | 2270 | 1349 | [0] | ok |
| oom2-C1-legacy | 83.09 [83.09, 83.09] | 82.49 [82.49, 82.49] | n/a | 3652 | 1353 | [0] | ok |
| oom2-C1-shared | 8.22 [8.21, 8.24] | 7.64 [7.62, 7.64] | n/a | 3652 | 1352 | [0] | ok |
| oom2-C2-legacy | 43.12 [43.12, 43.12] (1) | 42.11 [42.11, 42.11] (1) | n/a | 2852 | 1485 | [0, 1] | ok |
| oom2-C2-shared | 5.07 [5.07, 5.07] (1) | 4.27 [4.27, 4.27] (1) | n/a | 2852 | 1575 | [0, 1] | ok |
| or002-A1-legacy | 4.57 [4.55, 4.62] | 0.20 [0.20, 0.20] | 2.61 [2.59, 2.61] | 364 | 731 |  | ok |
| or002-A1-legacy-auto | 2.71 [2.69, 2.73] | 0.20 [0.20, 0.20] | 0.74 [0.73, 0.74] | 248 | 835 |  | ok |
| or002-A1-shared | 3.15 [3.12, 3.18] | 0.11 [0.11, 0.11] | 1.26 [1.24, 1.26] | 270 | 726 |  | ok |
| or002-A1-shared-auto | 2.64 [2.60, 2.66] | 0.11 [0.11, 0.12] | 0.75 [0.74, 0.75] | 248 | 826 |  | ok |
| or002-B1 | 2.66 [2.63, 2.71] | 0.12 [0.12, 0.12] | 0.38 [0.38, 0.45] | 364 | 876 |  | ok |
| or002-C1-legacy | 0.82 [0.82, 0.87] | 0.13 [0.13, 0.13] | 0.61 [0.61, 0.64] | 154 | 699 |  | ok |
| or002-C1-legacy-auto | 0.58 [0.57, 0.67] | 0.13 [0.13, 0.13] | 0.36 [0.35, 0.38] | 234 | 768 |  | ok |
| or002-C1-legacy-free | 0.85 [0.84, 0.90] | 0.13 [0.13, 0.14] | 0.65 [0.64, 0.65] | 154 | 679 |  | ok |
| or002-C1-legacy-free-norust | 3.39 [3.38, 3.39] | 0.16 [0.16, 0.17] | 3.15 [3.14, 3.15] | 168 | 685 |  | ok |
| or002-C1-legacy-k3 | 0.56 [0.56, 0.61] | 0.13 [0.13, 0.13] | 0.34 [0.34, 0.35] | 144 | 766 |  | ok |
| or002-C1-legacy-noprune | 0.64 [0.64, 0.70] | 0.13 [0.13, 0.13] | 0.44 [0.43, 0.47] | 154 | 710 |  | ok |
| or002-C1-legacy-norust | 2.84 [2.82, 2.88] | 0.13 [0.13, 0.13] | 2.62 [2.61, 2.68] | 168 | 696 |  | ok |
| or002-C1-shared | 0.38 [0.37, 0.44] | 0.01 [0.01, 0.01] | 0.29 [0.29, 0.32] | 154 | 707 |  | ok |
| or002-C1-shared-auto | 0.47 [0.46, 0.51] | 0.01 [0.01, 0.01] | 0.37 [0.36, 0.37] | 152 | 741 |  | ok |
| or002-C1-shared-k3 | 0.46 [0.45, 0.50] | 0.01 [0.01, 0.01] | 0.36 [0.34, 0.36] | 144 | 741 |  | ok |
| or002-C1-shared-noprune | 0.22 [0.21, 0.24] | 0.01 [0.01, 0.01] | 0.12 [0.12, 0.13] | 154 | 713 |  | ok |
| or002-C1-shared-norust | 2.46 [2.43, 2.54] | 0.01 [0.01, 0.01] | 2.37 [2.35, 2.39] | 168 | 695 |  | ok |
| or002-C1-tiny0 | 0.35 [0.35, 0.39] | 0.01 [0.01, 0.01] | 0.27 [0.26, 0.28] | 154 | 704 |  | ok |
| or003-A1-legacy | 0.69 [0.68, 0.72] | 0.11 [0.11, 0.11] | 0.28 [0.28, 0.28] | 226 | 517 |  | ok |
| or003-A1-legacy-auto | 0.52 [0.52, 0.54] | 0.11 [0.11, 0.11] | 0.12 [0.12, 0.13] | 152 | 552 |  | ok |
| or003-A1-shared | 0.47 [0.45, 0.49] | 0.04 [0.04, 0.04] | 0.12 [0.12, 0.13] | 190 | 519 |  | ok |
| or003-A1-shared-auto | 0.46 [0.45, 0.49] | 0.04 [0.04, 0.04] | 0.12 [0.12, 0.12] | 152 | 554 |  | ok |
| or003-B1 | 0.49 [0.48, 0.49] | 0.08 [0.08, 0.08] | 0.07 [0.07, 0.07] | 226 | 602 |  | ok |
| or003-C1-legacy | 0.25 [0.25, 0.28] | 0.09 [0.09, 0.09] | 0.10 [0.10, 0.10] | 152 | 556 |  | ok |
| or003-C1-legacy-auto | 0.23 [0.23, 0.23] | 0.08 [0.08, 0.08] | 0.08 [0.08, 0.08] | 144 | 548 |  | ok |
| or003-C1-legacy-k3 | 0.22 [0.21, 0.22] | 0.08 [0.08, 0.08] | 0.06 [0.06, 0.06] | 142 | 583 |  | ok |
| or003-C1-legacy-noprune | 0.23 [0.23, 0.28] | 0.08 [0.08, 0.09] | 0.08 [0.08, 0.09] | 144 | 552 |  | ok |
| or003-C1-legacy-norust | 0.47 [0.47, 0.52] | 0.09 [0.09, 0.09] | 0.32 [0.31, 0.33] | 152 | 563 |  | ok |
| or003-C1-shared | 0.12 [0.12, 0.13] | 0.01 [0.01, 0.01] | 0.04 [0.04, 0.04] | 144 | 551 |  | ok |
| or003-C1-shared-auto | 0.17 [0.17, 0.17] | 0.01 [0.01, 0.01] | 0.08 [0.08, 0.08] | 144 | 543 |  | ok |
| or003-C1-shared-k3 | 0.17 [0.14, 0.17] | 0.01 [0.01, 0.01] | 0.06 [0.06, 0.07] | 144 | 583 |  | ok |
| or003-C1-shared-noprune | 0.10 [0.10, 0.14] | 0.01 [0.01, 0.01] | 0.03 [0.03, 0.03] | 152 | 557 |  | ok |
| or003-C1-shared-norust | 0.34 [0.33, 0.34] | 0.01 [0.01, 0.01] | 0.26 [0.26, 0.27] | 144 | 540 |  | ok |
| or003-C1-tiny0 | 0.11 [0.11, 0.12] | 0.01 [0.01, 0.01] | 0.04 [0.04, 0.04] | 144 | 542 |  | ok |
| or003-F-auto | 39.43 [39.35, 39.71] | 27.94 [27.81, 28.30] | 9.57 [9.38, 9.74] | 36 | 1524 |  | ok |
| or003-F-polars | 44.77 [44.53, 44.77] | 26.35 [26.02, 26.36] | 16.53 [16.50, 16.73] | 36 | 1139 |  | ok |
| or003-F-sparse | 39.81 [39.04, 39.97] | 28.10 [28.04, 28.34] | 9.52 [9.26, 9.56] | 36 | 1502 |  | ok |
| or003-F-sparse-norust | 228.06 [203.84, 228.96] | 28.31 [28.23, 28.65] | 197.14 [173.90, 198.71] | 36 | 2451 |  | ok |
| or005-A1-legacy | 0.17 [0.17, 0.21] | 0.05 [0.05, 0.05] | 0.03 [0.03, 0.03] | 152 | 487 |  | ok |
| or005-A1-shared | 0.13 [0.13, 0.14] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 144 | 483 |  | ok |
| or005-B1 | 0.17 [0.16, 0.20] | 0.04 [0.04, 0.04] | 0.02 [0.02, 0.02] | 152 | 564 |  | ok |
| or005-C1-legacy | 0.14 [0.14, 0.18] | 0.05 [0.05, 0.05] | 0.03 [0.03, 0.04] | 152 | 552 |  | ok |
| or005-C1-shared | 0.09 [0.09, 0.13] | 0.00 [0.00, 0.00] | 0.03 [0.03, 0.03] | 152 | 552 |  | ok |
| or005-C1-tiny0 | 0.07 [0.07, 0.11] | 0.00 [0.00, 0.00] | 0.01 [0.01, 0.01] | 152 | 552 |  | ok |
| or005-F-auto | 17.02 [16.73, 17.19] | 14.36 [14.35, 14.54] | 1.31 [1.30, 1.33] | 36 | 935 |  | ok |
| or005-F-polars | 16.14 [15.58, 16.21] | 13.37 [13.12, 13.45] | 1.44 [1.40, 1.46] | 36 | 736 |  | ok |
| or005-F-sparse | 16.81 [16.42, 17.12] | 14.32 [13.81, 14.50] | 1.32 [1.31, 1.32] | 36 | 949 |  | ok |
| or005-F-sparse-norust | 33.09 [31.27, 33.89] | 14.29 [14.10, 14.62] | 17.28 [15.88, 18.60] | 36 | 955 |  | ok |
| sk2ml2-A1-legacy | 462.11 [462.07, 462.26] | 453.16 [453.16, 453.18] | n/a | 8870 | 3105 | [0] | ok |
| sk2ml2-A1-shared | 56.21 [56.20, 56.37] | 47.21 [47.11, 47.37] | n/a | 8870 | 3191 | [0] | ok |
| sk2ml2-B1 | 460.20 [460.10, 460.22] | 449.93 [449.92, 450.00] | n/a | 8870 | 3234 | [0] | ok |
| sk2ml2-C1-shared | 453.57 [453.55, 453.61] | 450.99 [450.97, 451.05] | n/a | 9924 | 3087 | [0] | ok |
| sk2ml2-C2-legacy | 331.27 [331.27, 331.27] (1) | 328.71 [328.71, 328.71] (1) | n/a | 6772 | 3612 | [0, 1] | ok |
| sk2ml2-C2-shared | 28.45 [28.45, 28.45] (1) | 25.60 [25.60, 25.60] (1) | n/a | 6750 | 3687 | [0, 1] | ok |
| sk2ml2-C2-shared-filter-cpu | 24.94 [24.94, 24.94] (1) | 22.10 [22.10, 22.10] (1) | n/a | 6750 | 5073 | [0, 1] | ok |
| sk2ml2-C2-shared-filter-cupy | 22.94 [22.94, 22.94] (1) | 20.18 [20.18, 20.18] (1) | n/a | 6750 | 3537 | [0, 1] | ok |
| sk2ml3-A1-legacy | FAIL: skipped: sk2ml2-A1-legacy K=2 median 453.2s is more than 2.0 | FAIL: skipped: sk2ml2-A1-legacy K=2 median 453.2s is more than 2.0 | FAIL: skipped: sk2ml2-A1-legacy K=2 median 453.2s is more than 2.0 | 0 | 0 |  | skipped: sk2ml2-A1-legacy K=2 median 453 |
| sk2ml3-A1-shared | 734.66 [733.72, 737.58] | 45.09 [45.02, 45.22] | 675.65 [674.52, 677.96] | 0 | 4094 |  | ok |
| sk2ml3-B1 | FAIL: skipped: sk2ml2-B1 K=2 median 449.9s is more than 2.0x the 1; skipped: sk2ml2-B1 K=2 median 450.0s is more than 2.0x the 1 | FAIL: skipped: sk2ml2-B1 K=2 median 449.9s is more than 2.0x the 1; skipped: sk2ml2-B1 K=2 median 450.0s is more than 2.0x the 1 | FAIL: skipped: sk2ml2-B1 K=2 median 449.9s is more than 2.0x the 1; skipped: sk2ml2-B1 K=2 median 450.0s is more than 2.0x the 1 | 0 | 0 |  | skipped: sk2ml2-B1 K=2 median 449.9s is , skipped: sk2ml2-B1 K=2 median 450.0s is  |
| sk2ml3-C1-shared | FAIL: skipped: sk2ml2-C1-shared K=2 median 451.0s is more than 2.0 | FAIL: skipped: sk2ml2-C1-shared K=2 median 451.0s is more than 2.0 | FAIL: skipped: sk2ml2-C1-shared K=2 median 451.0s is more than 2.0 | 0 | 0 |  | skipped: sk2ml2-C1-shared K=2 median 451 |
| skew-A1-legacy | 0.68 [0.68, 0.68] | 0.00 [0.00, 0.00] | 0.11 [0.10, 0.11] | 242 | 1269 |  | ok |
| skew-A1-legacy-auto | 0.87 [0.84, 0.87] | 0.00 [0.00, 0.00] | 0.28 [0.28, 0.29] | 242 | 1262 |  | ok |
| skew-A1-shared | 0.66 [0.66, 0.67] | 0.00 [0.00, 0.00] | 0.08 [0.08, 0.08] | 242 | 1364 |  | ok |
| skew-A1-shared-auto | 0.84 [0.83, 0.87] | 0.00 [0.00, 0.00] | 0.24 [0.24, 0.27] | 242 | 1281 |  | ok |
| skew-B1 | 0.66 [0.66, 0.67] | 0.00 [0.00, 0.00] | 0.07 [0.07, 0.07] | 242 | 1365 |  | ok |
| skew-C1-legacy | 0.59 [0.59, 0.62] | 0.00 [0.00, 0.00] | 0.06 [0.06, 0.07] | 158 | 1346 |  | ok |
| skew-C1-legacy-auto | 0.72 [0.71, 0.76] | 0.00 [0.00, 0.00] | 0.14 [0.14, 0.18] | 158 | 1326 |  | ok |
| skew-C1-legacy-k3 | 1.10 [1.10, 1.14] | 0.00 [0.00, 0.00] | 0.28 [0.28, 0.29] | 366 | 1380 |  | ok |
| skew-C1-legacy-noprune | 0.67 [0.66, 0.71] | 0.00 [0.00, 0.00] | 0.12 [0.12, 0.18] | 160 | 1335 |  | ok |
| skew-C1-legacy-norust | 0.63 [0.62, 0.72] | 0.00 [0.00, 0.00] | 0.09 [0.08, 0.15] | 158 | 1347 |  | ok |
| skew-C1-shared | 0.67 [0.63, 0.73] | 0.01 [0.01, 0.01] | 0.12 [0.11, 0.18] | 160 | 1320 |  | ok |
| skew-C1-shared-auto | 0.77 [0.76, 0.78] | 0.01 [0.01, 0.01] | 0.17 [0.17, 0.20] | 160 | 1340 |  | ok |
| skew-C1-shared-k3 | 1.12 [1.12, 1.14] | 0.01 [0.01, 0.01] | 0.29 [0.28, 0.29] | 608 | 1469 |  | ok |
| skew-C1-shared-noprune | 0.78 [0.77, 0.82] | 0.01 [0.01, 0.01] | 0.23 [0.22, 0.29] | 160 | 1358 |  | ok |
| skew-C1-shared-norust | 0.70 [0.68, 0.74] | 0.01 [0.01, 0.01] | 0.16 [0.14, 0.19] | 160 | 1348 |  | ok |
| skew-C1-tiny0 | 0.60 [0.59, 0.60] | 0.01 [0.01, 0.01] | 0.06 [0.05, 0.06] | 158 | 1351 |  | ok |
| skew-C2-legacy | 1.03 [1.03, 1.03] (1) | 0.03 [0.03, 0.03] (1) | 0.13 [0.13, 0.13] (1) | 246 | 1588 |  | ok |
| skew-C2-legacy-balance-nnz | 1.04 [1.04, 1.04] (1) | 0.03 [0.03, 0.03] (1) | 0.12 [0.12, 0.12] (1) | 244 | 1670 |  | ok |
| skew-C2-shared | 0.99 [0.99, 0.99] (1) | 0.03 [0.03, 0.03] (1) | 0.15 [0.15, 0.15] (1) | 246 | 1601 |  | ok |
| skew-F-auto | 3.49 [3.35, 3.55] | 0.20 [0.19, 0.21] | 1.71 [1.68, 1.73] | 36 | 650 |  | ok |
| skew-F-polars | 3.52 [3.31, 3.52] | 0.20 [0.20, 0.21] | 1.74 [1.69, 1.75] | 36 | 631 |  | ok |
| skew-F-sparse | 6.71 [6.54, 7.00] | 1.14 [1.12, 1.15] | 4.08 [3.93, 4.20] | 36 | 1648 |  | ok |
| skew-F-sparse-norust | 148.49 [147.64, 148.87] | 1.13 [1.11, 1.13] | 145.87 [144.93, 146.15] | 36 | 1685 |  | ok |
| smoke-A1-legacy | 0.03 [0.03, 0.04] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 148 | 442 |  | ok |
| smoke-A1-legacy-auto | 0.06 [0.06, 0.12] | 0.00 [0.00, 0.00] | 0.03 [0.03, 0.09] | 148 | 477 |  | ok |
| smoke-A1-shared | 0.03 [0.03, 0.03] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 148 | 443 |  | ok |
| smoke-A1-shared-auto | 0.06 [0.06, 0.12] | 0.00 [0.00, 0.00] | 0.03 [0.03, 0.09] | 148 | 446 |  | ok |
| smoke-B1 | 0.04 [0.04, 0.04] | 0.00 [0.00, 0.00] | 0.01 [0.01, 0.01] | 148 | 533 |  | ok |
| smoke-C1-legacy | 0.04 [0.04, 0.09] | 0.00 [0.00, 0.02] | 0.01 [0.01, 0.01] | 148 | 514 |  | ok |
| smoke-C1-legacy-auto | 0.06 [0.06, 0.12] | 0.00 [0.00, 0.02] | 0.03 [0.03, 0.03] | 148 | 514 |  | ok |
| smoke-C1-legacy-free | 0.03 [0.03, 0.09] | 0.00 [0.00, 0.06] | 0.00 [0.00, 0.00] | 148 | 514 |  | ok |
| smoke-C1-legacy-free-norust | 0.04 [0.04, 0.09] | 0.00 [0.00, 0.02] | 0.01 [0.01, 0.01] | 148 | 515 |  | ok |
| smoke-C1-legacy-k3 | 0.04 [0.04, 0.13] | 0.00 [0.00, 0.02] | 0.01 [0.01, 0.01] | 148 | 545 |  | ok |
| smoke-C1-legacy-noprune | 0.04 [0.04, 0.11] | 0.00 [0.00, 0.08] | 0.01 [0.01, 0.01] | 148 | 514 |  | ok |
| smoke-C1-legacy-norust | 0.04 [0.04, 0.09] | 0.00 [0.00, 0.05] | 0.01 [0.01, 0.01] | 148 | 513 |  | ok |
| smoke-C1-shared | 0.04 [0.04, 0.09] | 0.00 [0.00, 0.02] | 0.01 [0.01, 0.01] | 148 | 514 |  | ok |
| smoke-C1-shared-auto | 0.06 [0.06, 0.12] | 0.00 [0.00, 0.02] | 0.03 [0.03, 0.03] | 148 | 515 |  | ok |
| smoke-C1-shared-k3 | 0.04 [0.04, 0.10] | 0.00 [0.00, 0.02] | 0.01 [0.01, 0.01] | 148 | 544 |  | ok |
| smoke-C1-shared-noprune | 0.04 [0.04, 0.09] | 0.00 [0.00, 0.02] | 0.01 [0.01, 0.01] | 148 | 513 |  | ok |
| smoke-C1-shared-norust | 0.04 [0.04, 0.10] | 0.00 [0.00, 0.00] | 0.01 [0.01, 0.01] | 148 | 511 |  | ok |
| smoke-C1-tiny0 | 0.04 [0.04, 0.09] | 0.00 [0.00, 0.02] | 0.01 [0.01, 0.01] | 148 | 514 |  | ok |
| smoke-C2-legacy | 0.60 [0.60, 0.60] (1) | 0.03 [0.03, 0.03] (1) | 0.01 [0.01, 0.01] (1) | 224 | 906 |  | ok |
| smoke-C2-shared | 0.66 [0.66, 0.66] (1) | 0.03 [0.03, 0.03] (1) | 0.01 [0.01, 0.01] (1) | 222 | 901 |  | ok |
| smoke-F-auto | 0.19 [0.17, 0.19] | 0.09 [0.09, 0.10] | 0.03 [0.03, 0.03] | 36 | 330 |  | ok |
| smoke-F-polars | 0.18 [0.18, 0.19] | 0.09 [0.09, 0.10] | 0.03 [0.03, 0.03] | 36 | 316 |  | ok |
| smoke-F-sparse | 0.41 [0.40, 0.42] | 0.13 [0.13, 0.14] | 0.13 [0.12, 0.13] | 36 | 323 |  | ok |
| smoke-F-sparse-norust | 0.79 [0.78, 0.80] | 0.15 [0.15, 0.15] | 0.55 [0.52, 0.56] | 36 | 376 |  | ok |
| wide-A1-legacy | 0.26 [0.26, 0.26] | 0.10 [0.10, 0.10] | 0.01 [0.01, 0.01] | 144 | 590 |  | ok |
| wide-A1-shared | 0.16 [0.16, 0.16] | 0.02 [0.02, 0.02] | 0.00 [0.00, 0.00] | 144 | 586 |  | ok |
| wide-B1 | 0.26 [0.26, 0.26] | 0.10 [0.10, 0.10] | 0.01 [0.01, 0.01] | 144 | 658 |  | ok |
| wide-C1-legacy | 0.26 [0.25, 0.27] | 0.11 [0.11, 0.11] | 0.01 [0.01, 0.01] | 144 | 610 |  | ok |
| wide-C1-shared | 0.16 [0.15, 0.16] | 0.02 [0.02, 0.02] | 0.01 [0.01, 0.01] | 144 | 622 |  | ok |
| wide-C1-tiny0 | 0.16 [0.15, 0.17] | 0.02 [0.02, 0.02] | 0.01 [0.01, 0.01] | 144 | 631 |  | ok |
| wide-F-auto | 41.47 [40.85, 41.60] | 36.20 [35.98, 36.27] | 1.01 [0.99, 1.01] | 36 | 1759 |  | ok |
| wide-F-polars | 40.08 [39.93, 40.84] | 35.72 [35.62, 35.86] | 0.39 [0.37, 0.40] | 36 | 1126 |  | ok |
| wide-F-sparse | 41.59 [41.21, 42.06] | 36.17 [35.70, 36.64] | 1.03 [1.00, 1.04] | 36 | 1750 |  | ok |
| wide-F-sparse-norust | 46.21 [46.04, 46.89] | 36.29 [35.69, 36.51] | 5.84 [5.72, 5.95] | 36 | 1727 |  | ok |
