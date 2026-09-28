# Consolidation campaign report

12 config-reps, 12 ok. Box time 0.64 h; GPU-hours (process time × devices used) 0.64.

Cells: median [min, max] in seconds over the ok reps (a rep count in parentheses when it is not 3);
`>N (timeout)` is a lower bound; `FAIL` gives the status. Rule 2 needs the winner's median ≥ 10% below
the best alternative, disjoint [min, max] ranges, and ≥ 1 s saved.

## Signatures

| regime | signatures | configs |
|---|---|---|
| ('oom_regression', 3e-05, 3, False) | 1 | 12 |

## Decision points

### DP1 — in-core miner (fastest variant per miner, wall s)

| regime | A | B | C | winner (rule 2) |
|---|---|---|---|---|
| oom2ml3 | 41.28 [41.11, 41.31] | — | 101.25 [100.33, 102.08] | A |

### DP2 — K=2 kernel within A1 (K=2 level s)

| regime | A1-shared | A1-legacy | winner (rule 2) |
|---|---|---|---|
| oom2ml3 | 8.61 [8.59, 8.62] | — | none |

### DP2 — K=2 kernel within C1 (K=2 level s)

| regime | C1-shared | C1-legacy | winner (rule 2) |
|---|---|---|---|
| oom2ml3 | 7.61 [7.58, 7.62] | 82.49 [82.48, 82.50] | C1-shared |

### DP2 — K=2 kernel within C2 (K=2 level s)

| regime | C2-shared | C2-legacy | winner (rule 2) |
|---|---|---|---|

### DP3 — K≥3 counting (Σ K≥3 level s; B = count_k3plus_gpu_resident)

| regime | A1-shared | A1-legacy | B1 | C1-shared | C1-legacy | C1-tiny0 | winner (rule 2) |
|---|---|---|---|---|---|---|---|
| oom2ml3 | 29.95 [29.79, 30.01] | — | — | 93.08 [92.13, 93.88] | 492.21 [491.18, 492.72] | — | A1-shared |

### DP3 — K≥3 counting within A1

| regime | A1-shared | A1-legacy | winner (rule 2) |
|---|---|---|---|
| oom2ml3 | 29.95 [29.79, 30.01] | — | none |

### DP3 — K≥3 counting within C1

| regime | C1-shared | C1-legacy | C1-tiny0 | winner (rule 2) |
|---|---|---|---|---|
| oom2ml3 | 93.08 [92.13, 93.88] | 492.21 [491.18, 492.72] | — | C1-shared |

### DP4 — what tiled cannot serve (wall s)

C1-shared: tiny groups, mega-groups and multi-chunk K=2 on the per-candidate kernel; C1-tiny0: tiny groups tiled too; C1-legacy: everything per-candidate.

| regime | C1-shared | C1-tiny0 | C1-legacy | winner (rule 2) |
|---|---|---|---|---|
| oom2ml3 | 101.25 [100.33, 102.08] | — | 575.27 [574.26, 575.80] | C1-shared |

### DP5 — layout on C1-shared (wall s)

| regime | C1-shared | C1-shared-auto | C1-shared-k3 | winner (rule 2) |
|---|---|---|---|---|
| oom2ml3 | 101.25 [100.33, 102.08] | — | — | none |

### DP5 — layout on A1-shared (wall s)

| regime | A1-shared | A1-shared-auto | winner (rule 2) |
|---|---|---|---|
| oom2ml3 | 41.28 [41.11, 41.31] | — | none |

### DP5 — layout on C1-legacy (wall s)

| regime | C1-legacy | C1-legacy-auto | C1-legacy-k3 | winner (rule 2) |
|---|---|---|---|---|
| oom2ml3 | 575.27 [574.26, 575.80] | — | — | none |

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
| oom2ml3 | 101.25 [100.33, 102.08] | — | 575.27 [574.26, 575.80] | — | C1-shared |

### DP7 — SON (wall s; per-pass s from the rep with the median wall)

| regime | config | wall | pass 1 | pass 2 | chunks |
|---|---|---|---|---|---|

### DP8 — CPU tier (wall s)

| regime | F-polars | F-sparse | F-sparse-norust | F-auto | winner (rule 2) |
|---|---|---|---|---|---|

### DP8 — R1: CPU K>2 counting (Σ K≥3 level s)

| regime | F-sparse | F-sparse-norust | winner (rule 2) |
|---|---|---|---|

### DP9 — host roles on C1-shared (wall s)

| regime | C1-shared | C1-shared-norust | C1-shared-noprune | winner (rule 2) |
|---|---|---|---|---|
| oom2ml3 | 101.25 [100.33, 102.08] | — | 37.62 [37.48, 37.65] | C1-shared-noprune |

### DP9 — host roles on C1-legacy (wall s)

| regime | C1-legacy | C1-legacy-norust | C1-legacy-noprune | winner (rule 2) |
|---|---|---|---|---|
| oom2ml3 | 575.27 [574.26, 575.80] | — | — | none |

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
| oom2ml3-A1-shared | 41.28 [41.11, 41.31] | 8.61 [8.59, 8.62] | 29.95 [29.79, 30.01] | 0 | 1293 |  | ok |
| oom2ml3-C1-legacy | 575.27 [574.26, 575.80] | 82.49 [82.48, 82.50] | 492.21 [491.18, 492.72] | 0 | 1365 |  | ok |
| oom2ml3-C1-shared | 101.25 [100.33, 102.08] | 7.61 [7.58, 7.62] | 93.08 [92.13, 93.88] | 0 | 1374 |  | ok |
| oom2ml3-C1-shared-noprune | 37.62 [37.48, 37.65] | 7.51 [7.49, 7.52] | 29.51 [29.40, 29.53] | 0 | 1360 |  | ok |
