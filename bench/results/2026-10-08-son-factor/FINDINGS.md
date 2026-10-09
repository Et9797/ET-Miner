# SON's `local_support_factor`: Phase F1

Protocol: `bench/cpu/PROTOCOL.md`, Amendment 8. Harness:
`bench/cpu/son_stakes.py --matrix --arms built,incore --factors
0.8,0.9,0.95,1.0 --reps 3`, cap 600 s, 4 chunks, the S0/S1 box (`env.txt`),
seven workloads (Amendment 3's six and dslk2). One run at `de29318`:
`raw.jsonl`, 2026-10-08 23:57 to 2026-10-09 00:26.

All 210 rows are ok and carry `de29318` with a clean tree. The gate passed
before the run: ruff; 833 passed without the slow tests and the four
`test_support_00*` (on the working tree whose `src` and `tests` became
`de29318`); on `de29318` itself, the tier-equivalence, SON, streaming and
free-set suites, 74 passed.

`stakes.jsonl` holds the untimed stakes the amendment cites; `diag.jsonl` the
T1 profile per workload and factor (00:26–00:31, after the run).

## Correctness

One signature per workload across the four factors and `incore`
(`son_stakes.py --check`). The profile's union, bound and exact counts equal
the stakes' at every factor.

## Results

`built` medians over 3 reps with [min, max], in seconds.

| regime | f = 0.8 | f = 0.9 | f = 0.95 | f = 1.0 | `incore` |
|---|---|---|---|---|---|
| smoke T1 | 0.10 [0.09, 0.10] | 0.12 [0.12, 0.12] | 0.12 [0.12, 0.12] | 0.13 [0.13, 0.13] | 0.07 [0.07, 0.07] |
| smoke T4 | 0.09 [0.09, 0.10] | 0.12 [0.12, 0.12] | 0.12 [0.12, 0.12] | 0.13 [0.12, 0.13] | 0.07 [0.07, 0.07] |
| deepk T1 | 3.31 [3.28, 3.36] | 2.29 [2.28, 2.30] | 2.24 [2.23, 2.24] | 2.48 [2.47, 2.49] | 1.75 [1.75, 1.77] |
| deepk T4 | 2.84 [2.79, 2.84] | 2.10 [2.10, 2.11] | 2.07 [2.04, 2.07] | 2.22 [2.21, 2.25] | 1.48 [1.48, 1.49] |
| skew T1 | 22.12 [22.05, 22.22] | 21.51 [21.50, 21.51] | 22.51 [22.43, 22.51] | 23.86 [23.85, 24.03] | 2.32 [2.30, 2.32] |
| skew T4 | 10.53 [10.48, 10.58] | 10.47 [10.43, 10.49] | 11.06 [11.00, 11.10] | 11.77 [11.71, 11.77] | 1.58 [1.57, 1.59] |
| wide T1 | 1.13 [1.11, 1.13] | 1.02 [1.02, 1.02] | 0.96 [0.96, 0.97] | 0.90 [0.89, 0.90] | 0.76 [0.76, 0.77] |
| wide T4 | 0.55 [0.55, 0.55] | 0.48 [0.48, 0.51] | 0.46 [0.46, 0.49] | 0.44 [0.44, 0.46] | 0.34 [0.34, 0.34] |
| or005 T1 | 60.28 [60.18, 60.31] | 7.95 [7.94, 8.03] | 3.75 [3.72, 3.76] | 3.07 [3.06, 3.09] | 0.41 [0.41, 0.41] |
| or005 T4 | 59.17 [58.85, 59.63] | 7.71 [7.55, 7.75] | 3.49 [3.48, 3.51] | 2.85 [2.84, 2.87] | 0.26 [0.26, 0.26] |
| or0001k2 T1 | 2.75 [2.75, 2.77] | 2.76 [2.76, 2.76] | 2.75 [2.75, 2.78] | 2.76 [2.75, 2.77] | 0.79 [0.79, 0.79] |
| or0001k2 T4 | 2.84 [2.83, 2.84] | 2.84 [2.83, 2.86] | 2.84 [2.83, 2.86] | 2.84 [2.83, 2.84] | 0.54 [0.54, 0.54] |
| dslk2 T1 | 11.96 [11.91, 12.03] | 11.66 [11.55, 11.74] | 11.58 [11.52, 11.60] | 19.91 [19.78, 20.09] | 11.61 [11.55, 11.70] |
| dslk2 T4 | 8.50 [8.49, 8.62] | 8.47 [8.45, 8.51] | 8.43 [8.38, 8.44] | 14.79 [14.79, 14.83] | 8.76 [8.63, 8.78] |

Geometric mean of the 14 `built` medians: 0.8 3.46 s, 0.9 2.49 s, 0.95
2.23 s, 1.0 2.40 s.

Peak RSS (`ru_maxrss_mb`) against 0.9: 0.95 0.82–1.08×, 1.0 0.76–1.01×,
0.8 0.99–3.19× (or005: 256 → 817 MB at T1, 282 → 853 MB at T4).

## Rules

1. **Exactness: met** at every factor; no error and no cap hit.
2. **Qualifies.**
   - **0.95 qualifies.** It meets rule 2 on or005 at T1 (7.95 → 3.75 s,
     2.12×, 4.19 s saved) and T4 (7.71 → 3.49 s, 2.21×, 4.22 s), with no
     loss and RSS at most 1.08× (wide T1).
   - **1.0 does not.** It wins or005 (2.58× and 2.71×, 4.87 s each) but loses
     dslk2 (+71 % and +75 %, +8.26 and +6.32 s), skew (+11 % and +12 %,
     +2.35 and +1.30 s) and smoke T1 and T4 (+12 % and +11 %, both +0.01 s,
     under the 0.1 s floor, so no loss).
   - **0.8 does not.** It loses or005 (+659 % and +667 %, about +52 s),
     deepk (+44 % and +35 %, +1.02 and +0.74 s) and wide T1 (+11 %,
     +0.11 s), and its RSS exceeds 1.25× on or005 (3.19× and 3.02×).
3. **Recommendation: 0.95**, the only qualifying factor.
4. **Noise control: met.** or0001k2's four `built` medians agree within
   0.2 % at both thread settings; every `incore` [min, max] lies within 10 %
   of its median (widest: dslk2 T4, 8.63–8.78 against 8.76).
5. The owner decides the default.

What 0.95 costs against 0.9, under rule 2's thresholds: skew +4.6 % at T1
(+1.00 s) and +5.7 % at T4 (+0.59 s); smoke +2.9–3.9 % (under 0.01 s).

## Decision (owner)

The default is 0.95, in `apriori_streaming` and
`apriori_streaming_multi_gpu`. The GPU passes take the same default
unmeasured.

## Where the time went

Diagnostic, outside the rules: `son_phases.py`, T1, one run per workload and
factor (`diag.jsonl`). Seconds per profile phase; pass 1 includes the union's
merges and its finish. *counted* is what pass 2 counts.

| workload | f | pass 1 | pass 2 | union | counted |
|---|---|---|---|---|---|
| deepk | 0.9 | 2.29 | 0.00 | 13,070 | 0 |
| deepk | 0.95 | 2.22 | 0.00 | 12,981 | 0 |
| deepk | 1.0 | 1.90 | 0.59 | 9,888 | 1,065 |
| skew | 0.8 | 16.51 | 5.67 | 285,457 | 61,014 |
| skew | 0.9 | 13.18 | 8.39 | 221,111 | 97,121 |
| skew | 0.95 | 11.74 | 10.67 | 197,154 | 127,404 |
| skew | 1.0 | 10.39 | 13.44 | 175,148 | 173,512 |
| or005 | 0.8 | 60.48 | 0.07 | 7,254,832 | 16,913 |
| or005 | 0.9 | 7.93 | 0.12 | 1,125,261 | 32,840 |
| or005 | 0.95 | 3.44 | 0.27 | 445,722 | 75,929 |
| or005 | 1.0 | 2.39 | 0.67 | 277,701 | 184,191 |
| dslk2 | 0.9 | 11.59 | 0.00 | 1,192 | 0 |
| dslk2 | 0.95 | 11.51 | 0.00 | 1,152 | 0 |
| dslk2 | 1.0 | 11.44 | 8.39 | 1,147 | 19 |

- **or005** is pass 1 at every factor: its cost follows the union, 7.25M
  local itemsets at 0.8 against 278K at 1.0. Pass 2 stays under 0.7 s.
- **skew** trades the two passes almost one for one: from 0.8 to 1.0, pass 1
  falls by 6.1 s and pass 2 rises by 7.8 s.
- **dslk2 at 1.0** spends 8.4 s in pass 2 on 19 candidates: pass 2 reads and
  maps all four 5M-row chunks whatever it counts. Below 1.0 every candidate
  within the bound has its exact count from pass 1, and pass 2 reads nothing.
  deepk is the same at a smaller scale (0.59 s at 1.0).
- **or0001k2** does the same work at every factor (local min_count 1).

## Not measured

- Factors between 0.95 and 1.0. The sweep holds the four pre-registered
  values.
- The GPU passes, which read the same factor without the bound; this box has
  no GPU.

## Levers not taken here

- Counting a candidate only in the chunks that did not emit it (handoff
  item 3) shrinks pass 2 at higher factors. On dslk2 at 1.0 it would skip
  only the chunks with nothing left to count: its 19 candidates need 30
  (candidate, chunk) pairs over 4 chunks, and the cost is reading and
  mapping a chunk, not counting in it.
