# Decisions (bench/routes/decide.py)

## Rule 1

- none

## G3

**rows**: no regime ran

**percand**: no regime ran

**tiled**: no regime ran

**fused**: no regime ran

**shared**: no regime ran

## G5

**percand**: no regime ran

**group**: no regime ran

**tiled**: no regime ran

**fused**: no regime ran

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

- **device**: wins no regime (rule 4: goes, subject to rule 9)
- **host**: wins no regime (rule 4: goes, subject to rule 9)
- **rebuilt**: wins no regime (rule 4: goes, subject to rule 9)

## G2

- **lex**: wins no regime (rule 4: goes, subject to rule 9)
- **keep**: wins no regime (rule 4: goes, subject to rule 9)

## G6

- **dense**: wins no regime (rule 4: goes, subject to rule 9)
- **esco-auto**: wins no regime (rule 4: goes, subject to rule 9)
- **esco3**: wins no regime (rule 4: goes, subject to rule 9)

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
- **boolean**: wins no regime (rule 4: goes, subject to rule 9)
- **ported**: wins no regime (rule 4: goes, subject to rule 9)

## S2

- **boolean**: wins no regime (rule 4: goes, subject to rule 9)
- **ported**: wins no regime (rule 4: goes, subject to rule 9)

## G7

- **prune**: wins no regime (rule 4)
- **noprune**: wins no regime (rule 4)

