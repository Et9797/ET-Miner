# Consolidation campaign report

78 config-reps, 78 ok. Box time 0.62 h; GPU-hours (process time × devices used) 0.76.

Cells: median [min, max] in seconds over the ok reps (a rep count in parentheses when it is not 3);
`>N (timeout)` is a lower bound; `FAIL` gives the status. Rule 2 needs the winner's median ≥ 10% below
the best alternative, disjoint [min, max] ranges, and ≥ 1 s saved.

## Signatures

| regime | signatures | configs |
|---|---|---|
| ('deep_k', 0.02, None, False) | 1 | 15 |
| ('deep_sparse_large', 0.015, None, False) | 1 | 13 |
| ('online_retail', 0.002, None, False) | 1 | 9 |
| ('online_retail', 0.003, None, False) | 1 | 6 |
| ('online_retail', 0.005, None, False) | 1 | 3 |
| ('oom_regression', 3e-05, 2, False) | 1 | 6 |
| ('oom_regression', 3e-05, 3, False) | 1 | 6 |
| ('skewed_rows', 0.02, None, False) | 1 | 3 |
| ('smoke', 0.01, None, False) | 1 | 6 |
| ('stress_k2', 1.5e-05, 2, False) | 1 | 6 |
| ('stress_k2', 1.5e-05, 3, False) | 1 | 2 |
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
| deepk | E2 | 2.89 [2.88, 2.93] | 1.445 | 1.389 | 4 |

### DP8 — CPU tier (wall s)

| regime | F-polars | F-sparse | F-sparse-norust | F-auto | winner (rule 2) |
|---|---|---|---|---|---|
| deepk | — | — | — | 2.38 [2.35, 2.38] | none |
| or003 | — | — | — | 38.89 [38.62, 39.00] | none |

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
| deepk-C1 | 0.49 [0.48, 0.49] | 0.01 [0.01, 0.01] | 0.03 [0.03, 0.04] | 193 | 844 |  | ok |
| deepk-C2 | 0.93 [0.92, 0.95] | 0.03 [0.03, 0.04] | 0.04 [0.04, 0.04] | 309 | 1488 |  | ok |
| deepk-D1 | 3.21 [3.03, 3.32] | n/a | n/a | 213 | 702 | [0] | ok |
| deepk-E2 | 2.89 [2.88, 2.93] | n/a | n/a | 193 | 840 |  | ok |
| deepk-F-auto | 2.38 [2.35, 2.38] | 0.17 [0.17, 0.19] | 1.19 [1.18, 1.19] | 29 | 565 |  | ok |
| dsl-C1 | 22.94 [22.08, 22.99] | 0.07 [0.07, 0.07] | 10.64 [10.49, 10.69] | 2265 | 10684 | [0] | ok |
| dsl-C1-percand | 22.56 [22.48, 23.09] | 0.03 [0.03, 0.03] | 11.07 [11.00, 11.08] | 2265 | 10852 | [0] | ok |
| dsl-C1-tiled | 94.28 [93.53, 94.37] | 0.07 [0.07, 0.10] | 82.47 [82.06, 82.66] | 2265 | 10852 | [0] | ok |
| dsl-C2 | 17.21 [16.92, 17.24] | 0.08 [0.08, 0.10] | 5.24 [5.22, 5.24] | 1325 | 11398 | [0, 1] | ok |
| dsl-D1 | 174.72 [174.72, 174.72] (1) | n/a | n/a | 715 | 6660 | [0] | ok |
| oom2-C1 | 6.05 [5.84, 6.07] | 5.44 [5.23, 5.45] | n/a | 3701 | 1192 | [0] | ok |
| oom2-C2 | 4.09 [4.08, 4.09] | 3.36 [3.35, 3.39] | n/a | 2917 | 1513 | [0, 1] | ok |
| oom2ml3-C1 | 28.04 [27.30, 28.16] | 5.55 [5.51, 5.66] | 21.92 [21.02, 21.97] | 10186 | 1188 | [0] | ok |
| oom2ml3-C1-tiled | 29.08 [28.64, 29.29] | 5.54 [5.53, 5.54] | 22.92 [22.50, 23.13] | 9927 | 1183 | [0] | ok |
| or002-C1 | 0.20 [0.19, 0.20] | 0.01 [0.01, 0.01] | 0.08 [0.07, 0.08] | 203 | 718 |  | ok |
| or002-C1-percand | 0.51 [0.50, 0.51] | 0.09 [0.09, 0.09] | 0.31 [0.31, 0.31] | 203 | 692 |  | ok |
| or002-C1-tiled | 0.21 [0.20, 0.21] | 0.01 [0.01, 0.01] | 0.10 [0.08, 0.10] | 203 | 715 |  | ok |
| or003-C1 | 0.13 [0.13, 0.13] | 0.01 [0.01, 0.01] | 0.02 [0.02, 0.02] | 201 | 554 |  | ok |
| or003-F-auto | 38.89 [38.62, 39.00] | 28.18 [27.99, 28.48] | 9.09 [8.96, 9.15] | 29 | 1477 |  | ok |
| or005-C1 | 0.12 [0.11, 0.12] | 0.00 [0.00, 0.00] | 0.01 [0.01, 0.01] | 201 | 550 |  | ok |
| sk2ml2-C1 | 35.06 [31.83, 35.22] | 32.39 [29.06, 32.58] | n/a | 10877 | 3232 | [0] | ok |
| sk2ml2-C2 | 17.99 [17.99, 18.00] | 15.38 [15.37, 15.40] | n/a | 6815 | 3552 | [0, 1] | ok |
| sk2ml3-C1 | 584.07 [584.07, 584.07] (1) | 29.51 [29.51, 29.51] (1) | 551.88 [551.88, 551.88] (1) | 15205 | 3189 | [0] | ok |
| sk2ml3-C2 | 306.32 [306.32, 306.32] (1) | 15.23 [15.23, 15.23] (1) | 288.12 [288.12, 288.12] (1) | 11557 | 3432 | [0, 1] | ok |
| skew-C1 | 0.61 [0.60, 0.70] | 0.01 [0.01, 0.01] | 0.05 [0.05, 0.11] | 209 | 1253 |  | ok |
| smoke-C1 | 0.09 [0.09, 0.09] | 0.00 [0.00, 0.01] | 0.01 [0.01, 0.01] | 197 | 511 |  | ok |
| smoke-C2 | 0.63 [0.59, 0.65] | 0.03 [0.03, 0.03] | 0.01 [0.01, 0.01] | 289 | 912 |  | ok |
| wide-C1 | 0.16 [0.16, 0.16] | 0.01 [0.01, 0.01] | 0.01 [0.01, 0.01] | 193 | 654 |  | ok |
