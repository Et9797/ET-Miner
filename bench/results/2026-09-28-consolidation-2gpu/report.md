# Consolidation campaign report

82 config-reps, 72 ok. Box time 1.72 h; GPU-hours (process time × devices used) 3.44.

Cells: median [min, max] in seconds over the ok reps (a rep count in parentheses when it is not 3);
`>N (timeout)` is a lower bound; `FAIL` gives the status. Rule 2 needs the winner's median ≥ 10% below
the best alternative, disjoint [min, max] ranges, and ≥ 1 s saved.

## Signatures

| regime | signatures | configs |
|---|---|---|
| ('deep_k', 0.02, None, False) | 1 | 9 |
| ('deep_sparse_large', 0.015, None, False) | 1 | 15 |
| ('oom_regression', 3e-05, 2, False) | 1 | 12 |
| ('skewed_rows', 0.02, None, False) | 1 | 9 |
| ('smoke', 0.01, None, False) | 1 | 6 |
| ('stress_k2', 1.5e-05, 2, False) | 1 | 18 |
| ('stress_k2', 1.5e-05, 3, False) | 1 | 3 |

## Non-ok config-reps

- `dsl-E2#r0`: timeout
- `sk2ml3-Asplit2-shared#r0`: skipped: sk2ml2-Asplit2-shared K=2 median 227.4s is more than 2.0x the 2-GPU best 16.0s
- `sk2ml3-Asplit2-shared#r1`: skipped: sk2ml2-Asplit2-shared K=2 median 225.6s is more than 2.0x the 2-GPU best 16.0s
- `sk2ml3-Asplit2-shared#r2`: skipped: sk2ml2-Asplit2-shared K=2 median 225.1s is more than 2.0x the 2-GPU best 16.0s
- `sk2ml3-Bsplit2#r0`: skipped: sk2ml2-Bsplit2 K=2 median 224.7s is more than 2.0x the 2-GPU best 16.0s
- `sk2ml3-Bsplit2#r1`: skipped: sk2ml2-Bsplit2 K=2 median 223.6s is more than 2.0x the 2-GPU best 16.0s
- `sk2ml3-Bsplit2#r2`: skipped: sk2ml2-Bsplit2 K=2 median 224.7s is more than 2.0x the 2-GPU best 16.0s
- `sk2ml3-C2-legacy#r0`: skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0x the 2-GPU best 15.0s
- `sk2ml3-C2-legacy#r1`: skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0x the 2-GPU best 15.5s
- `sk2ml3-C2-legacy#r2`: skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0x the 2-GPU best 16.0s

## Decision points

### DP1 — in-core miner (fastest variant per miner, wall s)

| regime | A | B | C | winner (rule 2) |
|---|---|---|---|---|

### DP2 — K=2 kernel within A1 (K=2 level s)

| regime | A1-shared | A1-legacy | winner (rule 2) |
|---|---|---|---|

### DP2 — K=2 kernel within C1 (K=2 level s)

| regime | C1-shared | C1-legacy | winner (rule 2) |
|---|---|---|---|

### DP2 — K=2 kernel within C2 (K=2 level s)

| regime | C2-shared | C2-legacy | winner (rule 2) |
|---|---|---|---|
| smoke | 0.03 [0.03, 0.04] | 0.04 [0.04, 0.04] | none |
| deepk | 0.04 [0.03, 0.04] | 0.03 [0.03, 0.04] | none |
| skew | 0.04 [0.03, 0.05] | 0.05 [0.04, 0.08] | none |
| oom2 | 3.40 [3.38, 3.41] | 35.35 [35.31, 35.36] | C2-shared |
| sk2ml2 | 17.22 [17.15, 17.23] | 188.87 [188.86, 188.87] | C2-shared |
| sk2ml3 | 15.99 [15.93, 16.00] | FAIL: skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0; skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0; skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0 | C2-shared |
| dsl | 0.08 [0.08, 0.08] | 0.05 [0.05, 0.06] | none |

### DP3 — K≥3 counting (Σ K≥3 level s; B = count_k3plus_gpu_resident)

| regime | A1-shared | A1-legacy | B1 | C1-shared | C1-legacy | C1-tiny0 | winner (rule 2) |
|---|---|---|---|---|---|---|---|

### DP3 — K≥3 counting within A1

| regime | A1-shared | A1-legacy | winner (rule 2) |
|---|---|---|---|

### DP3 — K≥3 counting within C1

| regime | C1-shared | C1-legacy | C1-tiny0 | winner (rule 2) |
|---|---|---|---|---|

### DP4 — what tiled cannot serve (wall s)

C1-shared: tiny groups, mega-groups and multi-chunk K=2 on the per-candidate kernel; C1-tiny0: tiny groups tiled too; C1-legacy: everything per-candidate.

| regime | C1-shared | C1-tiny0 | C1-legacy | winner (rule 2) |
|---|---|---|---|---|

### DP5 — layout on C1-shared (wall s)

| regime | C1-shared | C1-shared-auto | C1-shared-k3 | winner (rule 2) |
|---|---|---|---|---|

### DP5 — layout on A1-shared (wall s)

| regime | A1-shared | A1-shared-auto | winner (rule 2) |
|---|---|---|---|

### DP5 — layout on C1-legacy (wall s)

| regime | C1-legacy | C1-legacy-auto | C1-legacy-k3 | winner (rule 2) |
|---|---|---|---|---|

### DP5 — layout on A1-legacy (wall s)

| regime | A1-legacy | A1-legacy-auto | winner (rule 2) |
|---|---|---|---|

### DP6 — multi-GPU (Σ K≥2 level s)

| regime | C2-shared | C2-legacy | Asplit2-shared | Bsplit2 | winner (rule 2) |
|---|---|---|---|---|---|
| smoke | 0.04 [0.04, 0.05] | 0.05 [0.04, 0.05] | — | — | none |
| deepk | 0.07 [0.07, 0.07] | 0.06 [0.06, 0.07] | — | — | none |
| skew | 0.20 [0.19, 0.21] | 0.16 [0.15, 0.17] | — | — | none |
| oom2 | 3.40 [3.38, 3.41] | 35.35 [35.31, 35.36] | 37.25 [37.20, 37.26] | 36.39 [36.35, 36.40] | C2-shared |
| sk2ml2 | 17.22 [17.15, 17.23] | 188.87 [188.86, 188.87] | 225.08 [223.73, 227.38] | 224.67 [222.45, 227.63] | C2-shared |
| sk2ml3 | 781.87 [779.92, 785.05] | FAIL: skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0; skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0; skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0 | FAIL: skipped: sk2ml2-Asplit2-shared K=2 median 225.1s is more tha; skipped: sk2ml2-Asplit2-shared K=2 median 225.6s is more tha; skipped: sk2ml2-Asplit2-shared K=2 median 227.4s is more tha | FAIL: skipped: sk2ml2-Bsplit2 K=2 median 223.6s is more than 2.0x ; skipped: sk2ml2-Bsplit2 K=2 median 224.7s is more than 2.0x  | C2-shared |
| dsl | 27.69 [27.67, 27.94] | 5.65 [5.63, 5.65] | — | — | C2-legacy |

C1 vs C2 (wall s), for scaling:

### DP6 — row-split scaling

| regime | C1-shared | C2-shared | C1-legacy | C2-legacy | winner (rule 2) |
|---|---|---|---|---|---|
| smoke | — | 0.66 [0.64, 2.60] | — | 0.60 [0.60, 0.66] | none |
| deepk | — | 0.94 [0.93, 0.96] | — | 0.93 [0.93, 0.95] | none |
| skew | — | 1.06 [1.05, 1.06] | — | 1.07 [1.06, 1.09] | none |
| oom2 | — | 4.09 [4.09, 4.10] | — | 36.13 [36.13, 36.13] | C2-shared |
| sk2ml2 | — | 20.00 [19.98, 20.05] | — | 191.70 [191.67, 191.74] | C2-shared |
| sk2ml3 | — | 784.87 [782.82, 787.80] | — | FAIL: skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0; skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0; skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0 | C2-shared |
| dsl | — | 40.23 [40.21, 40.30] | — | 17.91 [17.90, 18.41] | C2-legacy |

### DP7 — SON (wall s; per-pass s from the rep with the median wall)

| regime | config | wall | pass 1 | pass 2 | chunks |
|---|---|---|---|---|---|
| dsl | E2 | >600 (timeout) | — | — | — |
| deepk | E2 | 22.64 [22.46, 22.73] | 15.362 | 7.222 | 4 |

### DP8 — CPU tier (wall s)

| regime | F-polars | F-sparse | F-sparse-norust | F-auto | winner (rule 2) |
|---|---|---|---|---|---|

### DP8 — R1: CPU K>2 counting (Σ K≥3 level s)

| regime | F-sparse | F-sparse-norust | winner (rule 2) |
|---|---|---|---|

### DP9 — host roles on C1-shared (wall s)

| regime | C1-shared | C1-shared-norust | C1-shared-noprune | winner (rule 2) |
|---|---|---|---|---|

### DP9 — host roles on C1-legacy (wall s)

| regime | C1-legacy | C1-legacy-norust | C1-legacy-noprune | winner (rule 2) |
|---|---|---|---|---|

### DP9 — free-set runs (R4) (wall s)

| regime | C1-legacy-free | C1-legacy-free-norust | winner (rule 2) |
|---|---|---|---|

### DP10 — survivor filter on C2-shared (wall s)

| regime | C2-shared | C2-shared-filter-cupy | C2-shared-filter-cpu | winner (rule 2) |
|---|---|---|---|---|
| smoke | 0.66 [0.64, 2.60] | — | — | none |
| deepk | 0.94 [0.93, 0.96] | — | — | none |
| skew | 1.06 [1.05, 1.06] | — | — | none |
| oom2 | 4.09 [4.09, 4.10] | — | — | none |
| sk2ml2 | 20.00 [19.98, 20.05] | 19.05 [18.06, 19.05] | 19.99 [19.00, 20.07] | none |
| sk2ml3 | 784.87 [782.82, 787.80] | — | — | none |
| dsl | 40.23 [40.21, 40.30] | — | — | none |

### DP10 — survivor filter on C2-legacy (wall s)

| regime | C2-legacy | C2-legacy-filter-cupy | C2-legacy-filter-cpu | winner (rule 2) |
|---|---|---|---|---|
| smoke | 0.60 [0.60, 0.66] | — | — | none |
| deepk | 0.93 [0.93, 0.95] | — | — | none |
| skew | 1.07 [1.06, 1.09] | — | — | none |
| oom2 | 36.13 [36.13, 36.13] | — | — | none |
| sk2ml2 | 191.70 [191.67, 191.74] | — | — | none |
| sk2ml3 | FAIL: skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0; skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0; skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0 | — | — | none |
| dsl | 17.91 [17.90, 18.41] | 18.49 [17.50, 19.23] | 18.83 [17.96, 19.05] | none |

### DP10 — row balance (wall s)

| regime | C2-legacy | C2-legacy-balance-nnz | winner (rule 2) |
|---|---|---|---|
| smoke | 0.60 [0.60, 0.66] | — | none |
| deepk | 0.93 [0.93, 0.95] | — | none |
| skew | 1.07 [1.06, 1.09] | 1.07 [1.07, 1.07] | none |
| oom2 | 36.13 [36.13, 36.13] | — | none |
| sk2ml2 | 191.70 [191.67, 191.74] | — | none |
| sk2ml3 | FAIL: skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0; skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0; skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0 | — | none |
| dsl | 17.91 [17.90, 18.41] | 18.14 [18.02, 18.80] | none |

## All configs

| config | wall s | K=2 s | K≥3 s | peak VRAM MB | peak RSS MB | throttled | status |
|---|---|---|---|---|---|---|---|
| deepk-C2-legacy | 0.93 [0.93, 0.95] | 0.03 [0.03, 0.04] | 0.03 [0.03, 0.03] | 309 | 1446 |  | ok |
| deepk-C2-shared | 0.94 [0.93, 0.96] | 0.04 [0.03, 0.04] | 0.04 [0.03, 0.04] | 309 | 1459 |  | ok |
| deepk-E2 | 22.64 [22.46, 22.73] | n/a | n/a | 213 | 981 |  | ok |
| dsl-C2-legacy | 17.91 [17.90, 18.41] | 0.05 [0.05, 0.06] | 5.59 [5.58, 5.59] | 1311 | 11396 | [0, 1] | ok |
| dsl-C2-legacy-balance-nnz | 18.14 [18.02, 18.80] | 0.05 [0.05, 0.06] | 5.48 [5.47, 5.50] | 1325 | 11396 | [0, 1] | ok |
| dsl-C2-legacy-filter-cpu | 18.83 [17.96, 19.05] | 0.05 [0.05, 0.07] | 5.46 [5.45, 5.49] | 1311 | 12320 | [0, 1] | ok |
| dsl-C2-legacy-filter-cupy | 18.49 [17.50, 19.23] | 0.05 [0.05, 0.06] | 5.53 [5.42, 5.54] | 1325 | 11398 | [0, 1] | ok |
| dsl-C2-shared | 40.23 [40.21, 40.30] | 0.08 [0.08, 0.08] | 27.62 [27.59, 27.86] | 1311 | 11396 | [0, 1] | ok |
| dsl-E2 | >600 (timeout) | >600 (timeout) | >600 (timeout) | 0 | 0 |  | timeout |
| oom2-Asplit2-shared | 39.58 [39.54, 39.60] | 37.25 [37.20, 37.26] | n/a | 2557 | 4581 | [0, 1] | ok |
| oom2-Bsplit2 | 39.01 [38.99, 39.03] | 36.39 [36.35, 36.40] | n/a | 2557 | 4664 | [0, 1] | ok |
| oom2-C2-legacy | 36.13 [36.13, 36.13] | 35.35 [35.31, 35.36] | n/a | 2917 | 1552 | [0, 1] | ok |
| oom2-C2-shared | 4.09 [4.09, 4.10] | 3.40 [3.38, 3.41] | n/a | 2917 | 1600 | [0, 1] | ok |
| sk2ml2-Asplit2-shared | 234.27 [232.85, 236.40] | 225.08 [223.73, 227.38] | n/a | 9185 | 26274 | [0, 1] | ok |
| sk2ml2-Bsplit2 | 234.94 [232.77, 237.90] | 224.67 [222.45, 227.63] | n/a | 9185 | 26264 | [0, 1] | ok |
| sk2ml2-C2-legacy | 191.70 [191.67, 191.74] | 188.87 [188.86, 188.87] | n/a | 6837 | 3579 | [0, 1] | ok |
| sk2ml2-C2-shared | 20.00 [19.98, 20.05] | 17.22 [17.15, 17.23] | n/a | 6815 | 3574 | [0, 1] | ok |
| sk2ml2-C2-shared-filter-cpu | 19.99 [19.00, 20.07] | 17.10 [16.39, 17.18] | n/a | 6815 | 5152 | [0, 1] | ok |
| sk2ml2-C2-shared-filter-cupy | 19.05 [18.06, 19.05] | 16.00 [15.01, 16.06] | n/a | 6815 | 3526 | [0, 1] | ok |
| sk2ml3-Asplit2-shared | FAIL: skipped: sk2ml2-Asplit2-shared K=2 median 225.1s is more tha; skipped: sk2ml2-Asplit2-shared K=2 median 225.6s is more tha; skipped: sk2ml2-Asplit2-shared K=2 median 227.4s is more tha | FAIL: skipped: sk2ml2-Asplit2-shared K=2 median 225.1s is more tha; skipped: sk2ml2-Asplit2-shared K=2 median 225.6s is more tha; skipped: sk2ml2-Asplit2-shared K=2 median 227.4s is more tha | FAIL: skipped: sk2ml2-Asplit2-shared K=2 median 225.1s is more tha; skipped: sk2ml2-Asplit2-shared K=2 median 225.6s is more tha; skipped: sk2ml2-Asplit2-shared K=2 median 227.4s is more tha | 0 | 0 |  | skipped: sk2ml2-Asplit2-shared K=2 media, skipped: sk2ml2-Asplit2-shared K=2 media, skipped: sk2ml2-Asplit2-shared K=2 media |
| sk2ml3-Bsplit2 | FAIL: skipped: sk2ml2-Bsplit2 K=2 median 223.6s is more than 2.0x ; skipped: sk2ml2-Bsplit2 K=2 median 224.7s is more than 2.0x  | FAIL: skipped: sk2ml2-Bsplit2 K=2 median 223.6s is more than 2.0x ; skipped: sk2ml2-Bsplit2 K=2 median 224.7s is more than 2.0x  | FAIL: skipped: sk2ml2-Bsplit2 K=2 median 223.6s is more than 2.0x ; skipped: sk2ml2-Bsplit2 K=2 median 224.7s is more than 2.0x  | 0 | 0 |  | skipped: sk2ml2-Bsplit2 K=2 median 223.6, skipped: sk2ml2-Bsplit2 K=2 median 224.7 |
| sk2ml3-C2-legacy | FAIL: skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0; skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0; skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0 | FAIL: skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0; skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0; skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0 | FAIL: skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0; skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0; skipped: sk2ml2-C2-legacy K=2 median 188.9s is more than 2.0 | 0 | 0 |  | skipped: sk2ml2-C2-legacy K=2 median 188, skipped: sk2ml2-C2-legacy K=2 median 188, skipped: sk2ml2-C2-legacy K=2 median 188 |
| sk2ml3-C2-shared | 784.87 [782.82, 787.80] | 15.99 [15.93, 16.00] | 765.87 [763.99, 769.05] | 11527 | 3568 | [0, 1] | ok |
| skew-C2-legacy | 1.07 [1.06, 1.09] | 0.05 [0.04, 0.08] | 0.12 [0.07, 0.12] | 311 | 1611 |  | ok |
| skew-C2-legacy-balance-nnz | 1.07 [1.07, 1.07] | 0.04 [0.04, 0.09] | 0.12 [0.08, 0.13] | 309 | 1664 |  | ok |
| skew-C2-shared | 1.06 [1.05, 1.06] | 0.04 [0.03, 0.05] | 0.16 [0.15, 0.17] | 311 | 1636 |  | ok |
| smoke-C2-legacy | 0.60 [0.60, 0.66] | 0.04 [0.04, 0.04] | 0.01 [0.01, 0.01] | 289 | 903 |  | ok |
| smoke-C2-shared | 0.66 [0.64, 2.60] | 0.03 [0.03, 0.04] | 0.01 [0.01, 0.01] | 305 | 1242 |  | ok |
