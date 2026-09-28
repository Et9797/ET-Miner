# Consolidation campaign report

59 config-reps, 59 ok. Box time 0.57 h; GPU-hours (process time × devices used) 0.57.

Cells: median [min, max] in seconds over the ok reps (a rep count in parentheses when it is not 3);
`>N (timeout)` is a lower bound; `FAIL` gives the status. Rule 2 needs the winner's median ≥ 10% below
the best alternative, disjoint [min, max] ranges, and ≥ 1 s saved.

## Signatures

| regime | signatures | configs |
|---|---|---|
| ('deep_k', 0.02, None, False) | 1 | 9 |
| ('deep_sparse_large', 0.015, None, False) | 1 | 10 |
| ('online_retail', 0.002, None, False) | 1 | 9 |
| ('online_retail', 0.003, None, False) | 1 | 6 |
| ('online_retail', 0.005, None, False) | 1 | 3 |
| ('oom_regression', 3e-05, 2, False) | 1 | 3 |
| ('oom_regression', 3e-05, 3, False) | 1 | 6 |
| ('skewed_rows', 0.02, None, False) | 1 | 3 |
| ('smoke', 0.01, None, False) | 1 | 3 |
| ('stress_k2', 1.5e-05, 2, False) | 1 | 3 |
| ('stress_k2', 1.5e-05, 3, False) | 1 | 1 |
| ('wide_vocab', 0.004, None, False) | 1 | 3 |

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

C1 vs C2 (wall s), for scaling:

### DP6 — row-split scaling

| regime | C1-shared | C2-shared | C1-legacy | C2-legacy | winner (rule 2) |
|---|---|---|---|---|---|

### DP7 — SON (wall s; per-pass s from the rep with the median wall)

| regime | config | wall | pass 1 | pass 2 | chunks |
|---|---|---|---|---|---|

### DP8 — CPU tier (wall s)

| regime | F-polars | F-sparse | F-sparse-norust | F-auto | winner (rule 2) |
|---|---|---|---|---|---|
| deepk | — | — | — | 2.66 [2.61, 2.96] | none |
| or003 | — | — | — | 39.80 [38.97, 42.97] | none |

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

### DP10 — survivor filter on C2-legacy (wall s)

| regime | C2-legacy | C2-legacy-filter-cupy | C2-legacy-filter-cpu | winner (rule 2) |
|---|---|---|---|---|

### DP10 — row balance (wall s)

| regime | C2-legacy | C2-legacy-balance-nnz | winner (rule 2) |
|---|---|---|---|

## All configs

| config | wall s | K=2 s | K≥3 s | peak VRAM MB | peak RSS MB | throttled | status |
|---|---|---|---|---|---|---|---|
| deepk-C1 | 0.46 [0.45, 0.46] | 0.01 [0.01, 0.01] | 0.03 [0.03, 0.03] | 0 | 713 |  | ok |
| deepk-D1 | 3.49 [3.37, 4.01] | n/a | n/a | 0 | 680 |  | ok |
| deepk-F-auto | 2.66 [2.61, 2.96] | 0.18 [0.17, 0.20] | 1.26 [1.25, 1.37] | 0 | 566 |  | ok |
| dsl-C1 | 23.65 [23.63, 24.47] | 0.07 [0.07, 0.07] | 13.99 [13.93, 14.01] | 0 | 11584 |  | ok |
| dsl-C1-percand | 24.36 [24.16, 24.61] | 0.05 [0.05, 0.05] | 14.61 [14.60, 14.75] | 0 | 11485 |  | ok |
| dsl-C1-tiled | 115.35 [115.30, 115.70] | 0.07 [0.07, 0.07] | 105.66 [105.47, 105.68] | 0 | 11671 |  | ok |
| dsl-D1 | 191.50 [191.50, 191.50] (1) | n/a | n/a | 0 | 6729 |  | ok |
| oom2-C1 | 7.76 [7.71, 7.82] | 7.17 [7.12, 7.19] | n/a | 0 | 1363 |  | ok |
| oom2ml3-C1 | 36.83 [36.71, 37.00] | 7.27 [7.23, 7.28] | 28.96 [28.89, 29.14] | 0 | 1376 |  | ok |
| oom2ml3-C1-tiled | 36.99 [36.95, 37.03] | 7.29 [7.28, 7.30] | 29.09 [29.05, 29.10] | 0 | 1357 |  | ok |
| or002-C1 | 0.18 [0.17, 0.18] | 0.01 [0.01, 0.01] | 0.09 [0.09, 0.10] | 0 | 709 |  | ok |
| or002-C1-percand | 0.63 [0.62, 0.66] | 0.13 [0.13, 0.15] | 0.42 [0.42, 0.43] | 0 | 706 |  | ok |
| or002-C1-tiled | 0.19 [0.19, 0.20] | 0.01 [0.01, 0.01] | 0.11 [0.11, 0.11] | 0 | 720 |  | ok |
| or003-C1 | 0.10 [0.10, 0.10] | 0.01 [0.01, 0.01] | 0.02 [0.02, 0.03] | 0 | 533 |  | ok |
| or003-F-auto | 39.80 [38.97, 42.97] | 28.43 [27.46, 30.40] | 9.60 [9.59, 10.28] | 0 | 1529 |  | ok |
| or005-C1 | 0.07 [0.07, 0.08] | 0.00 [0.00, 0.00] | 0.01 [0.01, 0.01] | 0 | 530 |  | ok |
| sk2ml2-C1 | 44.21 [43.27, 44.68] | 41.27 [40.67, 41.60] | n/a | 0 | 3112 |  | ok |
| sk2ml3-C1 | 711.57 [711.57, 711.57] (1) | 42.88 [42.88, 42.88] (1) | 666.08 [666.08, 666.08] (1) | 0 | 3055 |  | ok |
| skew-C1 | 0.61 [0.60, 0.64] | 0.01 [0.01, 0.01] | 0.06 [0.05, 0.06] | 0 | 1416 |  | ok |
| smoke-C1 | 0.04 [0.04, 0.04] | 0.00 [0.00, 0.00] | 0.01 [0.01, 0.01] | 0 | 507 |  | ok |
| wide-C1 | 0.16 [0.15, 0.17] | 0.02 [0.02, 0.02] | 0.01 [0.01, 0.01] | 0 | 547 |  | ok |
