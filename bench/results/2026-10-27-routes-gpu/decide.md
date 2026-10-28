# Decisions (bench/routes/decide.py)

## Rule 1

- none

## G3

**rows**: stays (rule 3: the full dispatch wins a regime); rule 11: oom2 (C1), oom2 (C2), sk2ml2 (C1), sk2ml2 (C2)

| regime | full | without | full wins | without wins |
|---|---|---|---|---|
| oom2 (C1) | 0.30 [0.27, 0.34] | 8.51 [8.50, 8.51] | True | False |
| oom2 (C2) | 0.66 [0.66, 0.66] | 4.75 [4.74, 4.78] | True | False |
| or0001k2 (C1) | 0.51 [0.50, 0.51] | 0.51 [0.50, 0.53] | False | False |
| or002 (C1) | 0.28 [0.27, 0.28] | 0.28 [0.28, 0.29] | False | False |
| sk2ml2 (C1) | 1.10 [1.10, 1.14] | 45.18 [45.10, 45.21] | True | False |
| sk2ml2 (C2) | 1.32 [1.29, 1.46] | 23.38 [23.37, 23.39] | True | False |
| wide (C1) | 0.07 [0.07, 0.10] | 0.07 [0.07, 0.07] | False | False |

**percand**: goes (rule 3): neither wins anywhere

| regime | full | without | full wins | without wins |
|---|---|---|---|---|
| deepk15 (C1) | 0.11 [0.11, 0.12] | 0.13 [0.13, 0.13] | False | False |

**tiled**: stays (rule 3: the full dispatch wins a regime); rule 11: oom2-bv (C1), oom2-bv (C2), or0001k2 (C1), or002 (C1), wide (C1)

| regime | full | without | full wins | without wins |
|---|---|---|---|---|
| dsl (C1) | 3.95 [3.90, 4.00] | 3.97 [3.92, 4.03] | False | False |
| dsl (C2) | 3.26 [3.08, 3.28] | 3.13 [3.13, 3.42] | False | False |
| dslxl (C1) | 10.42 [10.05, 10.45] | 10.33 [10.26, 10.39] | False | False |
| oom2-bv (C1) | 8.47 [8.44, 8.50] | 82.56 [82.55, 82.57] | True | False |
| oom2-bv (C2) | 5.22 [5.22, 5.23] | 42.37 [42.34, 42.38] | True | False |
| or0001k2 (C1) | 0.51 [0.50, 0.51] | 0.84 [0.84, 0.85] | False | False |
| or002 (C1) | 0.28 [0.27, 0.28] | 0.41 [0.41, 0.44] | False | False |
| wide (C1) | 0.07 [0.07, 0.10] | 0.17 [0.16, 0.19] | False | False |

**fused**: stays (rule 3: the full dispatch wins a regime); rule 11: sk2ml2-bv (C1)

| regime | full | without | full wins | without wins |
|---|---|---|---|---|
| sk2ml2-bv (C1) | 44.94 [44.85, 45.24] | 451.79 [451.77, 451.83] | True | False |

**shared**: goes (rule 3): neither wins anywhere

| regime | full | without | full wins | without wins |
|---|---|---|---|---|
| dsl (C1) | 5.02 [4.99, 5.03] | 4.87 [4.82, 4.95] | False | False |

## G5

**percand**: stays (rule 3: the full dispatch wins a regime); rule 11: deepk20 (C1), dsl (C1), dsl (C2), dsl-bv (C1), dsl-bv (C2), dsl-noprune (C1), dslxl (C1)

| regime | full | without | full wins | without wins |
|---|---|---|---|---|
| deepk (C1) | 0.17 [0.16, 0.17] | 0.18 [0.18, 0.20] | False | False |
| deepk20 (C1) | 4.96 [4.89, 4.96] | 8.94 [8.89, 9.07] | True | False |
| dsl (C1) | 3.95 [3.90, 4.00] | 6.87 [6.85, 6.96] | True | False |
| dsl (C2) | 3.26 [3.08, 3.28] | 4.63 [4.52, 4.70] | True | False |
| dsl-bv (C1) | 8.92 [8.88, 8.92] | 23.84 [23.83, 23.92] | True | False |
| dsl-bv (C2) | 5.13 [5.13, 5.14] | 12.55 [12.46, 12.59] | True | False |
| dsl-free (C1) | 2.84 [2.71, 2.89] | 3.33 [3.32, 3.34] | False | False |
| dsl-infer (C1) | 2.49 [2.48, 2.52] | 2.55 [2.54, 2.67] | False | False |
| dsl-noprune (C1) | 3.71 [3.69, 3.71] | 6.47 [6.45, 6.53] | True | False |
| dslxl (C1) | 10.42 [10.05, 10.45] | 18.93 [18.89, 18.97] | True | False |
| oom2ml3-bv (C1) | 10.05 [10.02, 10.22] | 10.06 [10.02, 10.06] | False | False |
| oom2ml3-bv (C2) | 6.52 [6.52, 6.84] | 6.46 [6.46, 6.49] | False | False |
| oom2ml3-noprune (C1) | 27.21 [27.18, 27.22] | 27.29 [27.25, 27.39] | False | False |
| or002 (C1) | 0.28 [0.27, 0.28] | 0.26 [0.25, 0.27] | False | False |
| or002 (C2) | 0.57 [0.56, 0.58] | 0.52 [0.51, 0.52] | False | False |
| or003 (C1) | 0.11 [0.10, 0.12] | 0.09 [0.08, 0.09] | False | False |
| sk2ml3-bv (C1) | 72.78 [72.64, 73.00] | 72.80 [72.74, 72.97] | False | False |
| sk2ml3-bv (C2) | FAIL (error: OutOfMemoryError: Out of memory allocating 4,375,000,) | FAIL (error: OutOfMemoryError: Out of memory allocating 4,375,000,) | False | False |
| skew (C1) | 0.17 [0.17, 0.18] | 0.17 [0.17, 0.18] | False | False |

**group**: stays (rule 3: the full dispatch wins a regime); rule 11: dsl (C1), dsl (C2), dsl-bv (C1), dsl-bv (C2), dsl-free (C1), dsl-noprune (C1), dslxl (C1)

| regime | full | without | full wins | without wins |
|---|---|---|---|---|
| deepk (C1) | 0.17 [0.16, 0.17] | 0.17 [0.15, 0.18] | False | False |
| deepk20 (C1) | 4.96 [4.89, 4.96] | 5.54 [5.41, 5.55] | False | False |
| dsl (C1) | 3.95 [3.90, 4.00] | 5.92 [5.87, 5.99] | True | False |
| dsl (C2) | 3.26 [3.08, 3.28] | 4.12 [4.06, 4.17] | False | False |
| dsl-bv (C1) | 8.92 [8.88, 8.92] | 13.59 [13.56, 13.62] | True | False |
| dsl-bv (C2) | 5.13 [5.13, 5.14] | 7.46 [7.45, 7.48] | True | False |
| dsl-free (C1) | 2.84 [2.71, 2.89] | 3.62 [3.59, 3.62] | False | False |
| dsl-infer (C1) | 2.49 [2.48, 2.52] | 3.00 [3.00, 3.07] | False | False |
| dsl-noprune (C1) | 3.71 [3.69, 3.71] | 5.82 [5.80, 5.89] | True | False |
| dslxl (C1) | 10.42 [10.05, 10.45] | 16.46 [16.44, 16.65] | True | False |
| oom2ml3-bv (C1) | 10.05 [10.02, 10.22] | 10.07 [10.06, 10.10] | False | False |
| oom2ml3-bv (C2) | 6.52 [6.52, 6.84] | 6.45 [6.45, 6.47] | False | False |
| oom2ml3-noprune (C1) | 27.21 [27.18, 27.22] | 27.32 [27.32, 27.34] | False | False |
| or002 (C1) | 0.28 [0.27, 0.28] | 0.30 [0.27, 0.31] | False | False |
| or002 (C2) | 0.57 [0.56, 0.58] | 0.56 [0.55, 0.56] | False | False |
| or003 (C1) | 0.11 [0.10, 0.12] | 0.09 [0.09, 0.09] | False | False |
| sk2ml3-bv (C1) | 72.78 [72.64, 73.00] | 72.89 [72.86, 73.00] | False | False |
| sk2ml3-bv (C2) | FAIL (error: OutOfMemoryError: Out of memory allocating 4,375,000,) | FAIL (error: OutOfMemoryError: Out of memory allocating 4,375,000,) | False | False |
| skew (C1) | 0.17 [0.17, 0.18] | 0.18 [0.17, 0.18] | False | False |

**tiled**: stays (rule 3: the full dispatch wins a regime); rule 11: oom2ml3-bv (C1), oom2ml3-bv (C2), oom2ml3-noprune (C1), or002 (C1), or002 (C2), or003 (C1), sk2ml3-bv (C1)

| regime | full | without | full wins | without wins |
|---|---|---|---|---|
| deepk (C1) | 0.17 [0.16, 0.17] | 0.17 [0.16, 0.19] | False | False |
| deepk20 (C1) | 4.96 [4.89, 4.96] | 4.88 [4.78, 4.94] | False | False |
| dsl (C1) | 3.95 [3.90, 4.00] | 3.94 [3.85, 3.94] | False | False |
| dsl (C2) | 3.26 [3.08, 3.28] | 3.09 [3.03, 3.24] | False | False |
| dsl-bv (C1) | 8.92 [8.88, 8.92] | 8.88 [8.87, 8.90] | False | False |
| dsl-bv (C2) | 5.13 [5.13, 5.14] | 5.11 [5.11, 5.11] | False | False |
| dsl-free (C1) | 2.84 [2.71, 2.89] | 2.71 [2.69, 2.77] | False | False |
| dsl-infer (C1) | 2.49 [2.48, 2.52] | 2.48 [2.48, 2.52] | False | False |
| dsl-noprune (C1) | 3.71 [3.69, 3.71] | 3.57 [3.55, 3.65] | False | False |
| dslxl (C1) | 10.42 [10.05, 10.45] | 10.13 [10.09, 10.21] | False | False |
| oom2ml3-bv (C1) | 10.05 [10.02, 10.22] | 79.48 [79.43, 79.73] | True | False |
| oom2ml3-bv (C2) | 6.52 [6.52, 6.84] | 74.64 [74.58, 74.87] | True | False |
| oom2ml3-noprune (C1) | 27.21 [27.18, 27.22] | 209.23 [208.85, 211.06] | True | False |
| or002 (C1) | 0.28 [0.27, 0.28] | 0.69 [0.60, 0.70] | False | False |
| or002 (C2) | 0.57 [0.56, 0.58] | 0.88 [0.86, 0.89] | False | False |
| or003 (C1) | 0.11 [0.10, 0.12] | 0.18 [0.17, 0.18] | False | False |
| sk2ml3-bv (C1) | 72.78 [72.64, 73.00] | 333.95 [333.84, 334.51] | True | False |
| sk2ml3-bv (C2) | FAIL (error: OutOfMemoryError: Out of memory allocating 4,375,000,) | FAIL (error: OutOfMemoryError: Out of memory allocating 4,375,000,) | False | False |
| skew (C1) | 0.17 [0.17, 0.18] | 0.17 [0.16, 0.17] | False | False |

**fused**: stays (rule 3: the full dispatch wins a regime); rule 11: sk2ml3-bv (C1)

| regime | full | without | full wins | without wins |
|---|---|---|---|---|
| sk2ml3-bv (C1) | 72.78 [72.64, 73.00] | 293.69 [293.33, 293.99] | True | False |

## C1

**gram**: stays (rule 3: the full dispatch wins a regime); rule 11: or0001k2 (F1), or003 (F1), or005 (F1), wide (F1)

| regime | full | without | full wins | without wins |
|---|---|---|---|---|
| deepk (F1) | 2.68 [2.54, 2.80] | 2.66 [2.53, 2.76] | False | False |
| deepk20 (F1) | 65.41 [65.02, 67.18] | 63.46 [63.07, 65.56] | False | False |
| dsl (F1) | FAIL (timeout) | FAIL (timeout) | False | False |
| or0001k2 (F1) | 0.74 [0.69, 0.75] | 2.81 [2.81, 2.88] | True | False |
| or002 (F1) | 6.58 [6.44, 7.00] | 7.02 [6.89, 7.25] | False | False |
| or003 (F1) | 1.38 [1.35, 1.47] | 2.00 [1.82, 2.01] | False | False |
| or005 (F1) | 0.39 [0.39, 0.47] | 0.69 [0.64, 0.71] | False | False |
| skew (F1) | 2.63 [2.59, 2.71] | 2.57 [2.56, 2.58] | False | False |
| smoke (F1) | 0.09 [0.09, 0.11] | 0.11 [0.10, 0.11] | False | False |
| wide (F1) | 0.39 [0.38, 0.45] | 2.76 [2.62, 2.85] | True | False |

**bitvec**: goes (rule 3): neither wins anywhere

| regime | full | without | full wins | without wins |
|---|---|---|---|---|
| deepk (F1) | 2.68 [2.54, 2.80] | 2.85 [2.79, 2.89] | False | False |
| deepk20 (F1) | 65.41 [65.02, 67.18] | 65.16 [64.50, 65.88] | False | False |
| dsl (F1) | FAIL (timeout) | FAIL (timeout) | False | False |
| or0001k2 (F1) | 0.74 [0.69, 0.75] | 0.72 [0.65, 0.75] | False | False |
| or002 (F1) | 6.58 [6.44, 7.00] | 6.32 [6.20, 6.34] | False | False |
| or003 (F1) | 1.38 [1.35, 1.47] | 1.30 [1.28, 1.47] | False | False |
| or005 (F1) | 0.39 [0.39, 0.47] | 0.38 [0.36, 0.45] | False | False |
| skew (F1) | 2.63 [2.59, 2.71] | 2.82 [2.63, 2.88] | False | False |
| smoke (F1) | 0.09 [0.09, 0.11] | 0.10 [0.09, 0.10] | False | False |
| wide (F1) | 0.39 [0.38, 0.45] | 0.42 [0.38, 0.44] | False | False |

## C2

**popcount**: stays (rule 3: the full dispatch wins a regime); rule 11: deepk (F1), or002 (F1), or003 (F1), or005 (F1), skew (F1), smoke (F1)

| regime | full | without | full wins | without wins |
|---|---|---|---|---|
| deepk (F1) | 2.68 [2.54, 2.80] | 27.43 [27.36, 27.64] | True | False |
| deepk20 (F1) | 65.41 [65.02, 67.18] | FAIL (timeout) | True | False |
| dsl (F1) | FAIL (timeout) | FAIL (timeout) | False | False |
| or002 (F1) | 6.58 [6.44, 7.00] | 37.65 [37.25, 37.75] | True | False |
| or003 (F1) | 1.38 [1.35, 1.47] | 5.81 [5.76, 5.91] | True | False |
| or005 (F1) | 0.39 [0.39, 0.47] | 1.33 [1.24, 1.37] | False | False |
| skew (F1) | 2.63 [2.59, 2.71] | 16.81 [16.70, 16.95] | True | False |
| smoke (F1) | 0.09 [0.09, 0.11] | 0.15 [0.15, 0.19] | False | False |
| wide (F1) | 0.39 [0.38, 0.45] | 0.44 [0.42, 0.49] | False | False |

**proj**: goes (rule 3): neither wins anywhere

| regime | full | without | full wins | without wins |
|---|---|---|---|---|
| deepk (F1) | 2.68 [2.54, 2.80] | 2.21 [2.14, 2.22] | False | False |
| deepk20 (F1) | 65.41 [65.02, 67.18] | 64.92 [64.64, 66.37] | False | False |
| dsl (F1) | FAIL (timeout) | FAIL (timeout) | False | False |
| or002 (F1) | 6.58 [6.44, 7.00] | 5.67 [5.49, 6.20] | False | False |
| or003 (F1) | 1.38 [1.35, 1.47] | 1.15 [1.13, 1.32] | False | False |
| or005 (F1) | 0.39 [0.39, 0.47] | 0.37 [0.31, 0.39] | False | False |
| skew (F1) | 2.63 [2.59, 2.71] | 1.79 [1.75, 1.84] | False | False |
| smoke (F1) | 0.09 [0.09, 0.11] | 0.07 [0.07, 0.08] | False | False |
| wide (F1) | 0.39 [0.38, 0.45] | 0.38 [0.37, 0.39] | False | False |

**son-percand**: stays (rule 3: the full dispatch wins a regime)

| regime | full | without | full wins | without wins |
|---|---|---|---|---|
| deepk-son (D1) | 3.77 [3.76, 3.82] | 3.75 [3.57, 3.76] | False | False |
| dsl-son (D1) | 1091.21 [1085.16, 1092.05] | 1089.57 [1088.05, 1090.65] | False | False |
| or005-son (D1) | 5.72 [5.45, 6.25] | 6.99 [6.62, 6.99] | True | False |
| skew-son (D1) | 18.83 [18.80, 18.98] | 18.97 [18.50, 19.07] | False | False |
| wide-son (D1) | 1.06 [1.06, 1.08] | 1.04 [1.02, 1.09] | False | False |

**son-groups**: stays (rule 3: the full dispatch wins a regime)

| regime | full | without | full wins | without wins |
|---|---|---|---|---|
| deepk-son (D1) | 3.77 [3.76, 3.82] | 3.80 [3.77, 3.95] | False | False |
| dsl-son (D1) | 1091.21 [1085.16, 1092.05] | 1090.02 [1089.21, 1092.07] | False | False |
| or005-son (D1) | 5.72 [5.45, 6.25] | 5.77 [5.71, 5.88] | False | False |
| skew-son (D1) | 18.83 [18.80, 18.98] | 21.84 [21.83, 22.11] | True | False |
| wide-son (D1) | 1.06 [1.06, 1.08] | 1.01 [0.99, 1.07] | False | False |

## G1

- **device**: wins dsl, dsl-str, dsl-wide, dslxl, sk2ml3
- **host**: wins no regime (rule 4: goes, subject to rule 9); BLOCKED by rule 7 (deepk: peak VRAM 1.44x without host; skew: peak VRAM 1.31x without host)
- **rebuilt**: wins no regime (rule 4: goes, subject to rule 9); BLOCKED by rule 7 (deepk: peak VRAM 1.30x without rebuilt; dsl-wide: peak VRAM 1.28x without rebuilt; or002: peak RSS 1.29x without rebuilt; or002-str: peak RSS 1.28x without rebuilt; skew: peak VRAM 1.31x without rebuilt)

| regime | device | host | rebuilt | winner |
|---|---|---|---|---|
| deepk | C1-default: 0.17 [0.16, 0.17] | C1-host: 1.02 [0.99, 1.04] | C1-rebuilt: 0.60 [0.58, 0.60] | none |
| dsl | C1-default: 3.95 [3.90, 4.00] | C1-host: 28.00 [27.93, 31.59] | C1-rebuilt: 20.59 [20.47, 22.79] | device |
| dsl-str | C1-coded: 18.30 [18.01, 18.50] | C1-host: 30.56 [30.45, 32.91] | C1-rebuilt: 56.28 [55.90, 57.49] | device |
| dsl-wide | C1-coded: 10.70 [10.49, 10.98] | C1-host: 29.05 [28.46, 30.38] | C1-rebuilt: 34.33 [33.29, 36.06] | device |
| dslxl | C1-default: 10.42 [10.05, 10.45] | C1-host: 99.55 [99.09, 124.49] | C1-rebuilt: 77.63 [68.64, 132.89] | device |
| oom2ml3 | C1-default: 0.36 [0.36, 0.38] | C1-host: 1.71 [1.68, 1.72] | C1-rebuilt: 1.12 [1.03, 1.22] | none |
| or002 | C1-default: 0.28 [0.27, 0.28] | C1-host: 0.43 [0.38, 0.45] | C1-rebuilt: 0.37 [0.32, 0.39] | none |
| or002-str | C1-coded: 0.40 [0.39, 0.51] | C1-host: 0.43 [0.43, 0.45] | C1-rebuilt: 0.54 [0.53, 0.64] | none |
| sk2ml3 | C1-default: 1.72 [1.71, 1.74] | C1-host: 5.99 [5.97, 6.00] | C1-rebuilt: 3.51 [3.43, 3.53] | device |
| skew | C1-default: 0.17 [0.17, 0.18] | C1-host: 1.36 [1.31, 1.36] | C1-rebuilt: 0.69 [0.68, 0.69] | none |

## G2

- **lex**: wins dsl, dsl-free, dslxl
- **keep**: wins no regime (rule 4: goes, subject to rule 9); BLOCKED by rule 7 (dsl: peak VRAM 1.74x without keep; dslxl: peak VRAM 1.79x without keep)

| regime | lex | keep | winner |
|---|---|---|---|
| deepk | C1-default: 0.17 [0.16, 0.17] | C1-keep: 0.15 [0.15, 0.17] | none |
| deepk20 | C1-default: 4.96 [4.89, 4.96] | C1-keep: 4.09 [4.04, 4.11] | none |
| dsl | C1-default: 3.95 [3.90, 4.00] | C1-keep: 10.03 [10.02, 10.04] | lex |
| dsl-free | C1-default: 2.84 [2.71, 2.89] | C1-keep: 4.63 [4.59, 4.64] | lex |
| dsl-infer | C1-default: 2.49 [2.48, 2.52] | C1-keep: 3.02 [2.99, 3.02] | none |
| dslxl | C1-default: 10.42 [10.05, 10.45] | C1-keep: 28.28 [28.21, 28.48] | lex |
| oom2ml3 | C1-default: 0.36 [0.36, 0.38] | C1-keep: 0.36 [0.35, 0.36] | none |
| or002 | C1-default: 0.28 [0.27, 0.28] | C1-keep: 0.28 [0.27, 0.29] | none |
| or003 | C1-default: 0.11 [0.10, 0.12] | C1-keep: 0.10 [0.09, 0.13] | none |
| or005 | C1-default: 0.06 [0.06, 0.06] | C1-keep: 0.05 [0.05, 0.06] | none |
| sk2ml3 | C1-default: 1.72 [1.71, 1.74] | C1-keep: 1.69 [1.68, 1.80] | none |
| skew | C1-default: 0.17 [0.17, 0.18] | C1-keep: 0.17 [0.17, 0.17] | none |
| smoke | C1-default: 0.05 [0.05, 0.06] | C1-keep: 0.05 [0.04, 0.05] | none |
| wide | C1-default: 0.07 [0.07, 0.10] | C1-keep: 0.07 [0.07, 0.07] | none |

## G6

- **dense**: wins dsl, oom2ml3, sk2ml3
- **esco-auto**: wins no regime (rule 4: goes, subject to rule 9)
- **esco3**: wins no regime (rule 4: goes, subject to rule 9)

| regime | dense | esco-auto | esco3 | winner |
|---|---|---|---|---|
| deepk | C1-default: 0.17 [0.16, 0.17] | C1-esco-auto: 0.32 [0.30, 0.33] | C1-esco3: 0.48 [0.47, 0.48] | none |
| dsl | C1-default: 3.95 [3.90, 4.00] | C1-esco-auto: 7.93 [7.84, 7.99] | C1-esco3: FAIL (error: TidsetFitError: dense→sparse transition: the tidsets ) | dense |
| oom2ml3 | C1-default: 0.36 [0.36, 0.38] | C1-esco-auto: 18.68 [18.64, 19.05] | C1-esco3: 18.71 [18.69, 18.74] | dense |
| or002 | C1-default: 0.28 [0.27, 0.28] | C1-esco-auto: 0.48 [0.45, 0.49] | C1-esco3: 0.45 [0.45, 0.46] | none |
| or003 | C1-default: 0.11 [0.10, 0.12] | C1-esco-auto: 0.12 [0.12, 0.12] | C1-esco3: 0.12 [0.12, 0.12] | none |
| sk2ml3 | C1-default: 1.72 [1.71, 1.74] | C1-esco-auto: 136.00 [135.62, 136.29] | C1-esco3: 135.99 [135.66, 136.16] | dense |
| skew | C1-default: 0.17 [0.17, 0.18] | C1-esco-auto: 0.31 [0.29, 0.31] | C1-esco3: 0.59 [0.57, 0.59] | none |

## C3

- **cpu**: wins deepk20, or0001k2, or002, or003
- **rust**: wins deepk, dsl, skew

| regime | cpu | rust | winner |
|---|---|---|---|
| deepk | F1-default: 2.68 [2.54, 2.80] | R1-default: 0.89 [0.88, 0.93] | rust |
| deepk20 | F1-default: 65.41 [65.02, 67.18] | R1-default: FAIL (timeout) | cpu |
| dsl | F1-default: FAIL (timeout) | R1-default: 226.83 [225.75, 227.70] | rust |
| or0001k2 | F1-default: 0.74 [0.69, 0.75] | R1-default: 18.10 [17.82, 18.12] | cpu |
| or002 | F1-default: 6.58 [6.44, 7.00] | R1-default: 93.61 [93.37, 95.65] | cpu |
| or003 | F1-default: 1.38 [1.35, 1.47] | R1-default: 3.31 [3.28, 3.40] | cpu |
| or005 | F1-default: 0.39 [0.39, 0.47] | R1-default: 0.37 [0.35, 0.41] | none |
| skew | F1-default: 2.63 [2.59, 2.71] | R1-default: 1.04 [1.00, 1.05] | rust |
| smoke | F1-default: 0.09 [0.09, 0.11] | R1-default: 0.03 [0.03, 0.03] | none |
| wide | F1-default: 0.39 [0.38, 0.45] | R1-default: 0.60 [0.58, 0.66] | none |

## S1

- **cpu**: wins no regime; stays (rule 9: SON is never removed as a strategy)
- **boolean**: wins no regime (rule 4: goes, subject to rule 9); BLOCKED by rule 7 (dsl-son: peak VRAM 1.89x without boolean; dslxl-son: peak VRAM 2.02x without boolean)
- **ported**: wins deepk-son, dsl-son, dslxl-son

| regime | cpu | boolean | ported | winner |
|---|---|---|---|---|
| deepk-son | D1-default: 3.77 [3.76, 3.82] | D1-boolean: 3.71 [3.68, 3.76] | D1-ported: 0.43 [0.36, 0.46] | ported |
| dsl-son | D1-default: 1091.21 [1085.16, 1092.05] | D1-boolean: 176.25 [175.65, 176.42] | D1-ported: 8.08 [8.04, 8.20] | ported |
| dslxl-son | — | D1-boolean: 504.07 [486.86, 506.98] | D1-ported: 14.87 [14.74, 14.96] | ported |

## S2

- **boolean**: wins no regime (rule 4: goes, subject to rule 9); BLOCKED by rule 7 (deepk-son (2 GPUs): peak VRAM 1.84x without boolean)
- **ported**: wins deepk-son (2 GPUs), dsl-son (2 GPUs), dslxl-son (2 GPUs)

| regime | boolean | ported | winner |
|---|---|---|---|
| deepk-son (2 GPUs) | E2-boolean: 3.29 [3.17, 3.38] | E2-ported: 1.15 [1.11, 1.23] | ported |
| dsl-son (2 GPUs) | E2-boolean: 118.75 [118.29, 118.79] | E2-ported: 7.93 [7.70, 8.21] | ported |
| dslxl-son (2 GPUs) | E2-boolean: 305.89 [298.90, 306.49] | E2-ported: 12.68 [12.50, 12.94] | ported |

## G7

- **prune**: wins oom2ml3 (C1), sk2ml3 (C1)
- **noprune**: wins no regime (rule 4)

| regime | prune | noprune | winner |
|---|---|---|---|
| deepk (C1) | 0.17 [0.16, 0.17] | 0.17 [0.15, 0.17] | none |
| dsl (C1) | 3.95 [3.90, 4.00] | 3.71 [3.69, 3.71] | none |
| oom2ml3 (C1) | 0.36 [0.36, 0.38] | 27.21 [27.18, 27.22] | prune |
| or002 (C1) | 0.28 [0.27, 0.28] | 0.21 [0.18, 0.21] | none |
| sk2ml3 (C1) | 1.72 [1.71, 1.74] | 628.70 [627.25, 629.65] | prune |

