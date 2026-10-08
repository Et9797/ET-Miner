# SON's partition upper bound: Phase B1

Protocol: `bench/cpu/PROTOCOL.md`, Amendment 5. Harness:
`bench/cpu/son_stakes.py --matrix --arms built,incore --reps 3`, cap 600 s,
4 chunks, `local_support_factor` 0.9, the S0/S1 box. Two runs, back to back:

- *base* at `e892d8a` (main after PR #28): `base.jsonl`, `base-env.txt`,
  19:35–19:42;
- *bound* at `740404f`: `raw.jsonl`, `env.txt`, 19:43–19:49.

All 144 rows are ok and carry a clean tree digest; the two env files differ
only in `rev`. The gate passed on `740404f` before the run (ruff; 791 passed
without the slow tests and the four `test_support_00*`; the tier-equivalence,
SON, streaming and free-set suites, 57 passed).

## Correctness

One signature per workload across base, bound and S1.

## Results

Medians over 3 reps with [min, max]. *ratio* is base / bound.

| regime | base `built` | bound `built` | ratio | saved | peak RSS base → bound | `incore` drift |
|---|---|---|---|---|---|---|
| smoke T1 | 0.15 s [0.15, 0.15] | 0.14 s [0.14, 0.14] | 1.11× | 0.02 s | 130 → 130 MB | −1.7 % |
| smoke T4 | 0.15 s [0.14, 0.15] | 0.13 s [0.13, 0.14] | 1.12× | 0.02 s | 136 → 136 MB | −2.0 % |
| deepk T1 | 4.74 s [4.70, 4.87] | 2.59 s [2.56, 2.62] | 1.83× | 2.15 s | 286 → 284 MB | −1.8 % |
| deepk T4 | 4.25 s [4.17, 4.34] | 2.26 s [2.24, 2.27] | 1.88× | 1.99 s | 331 → 335 MB | −0.9 % |
| skew T1 | 31.16 s [31.02, 31.27] | 22.31 s [22.20, 22.37] | 1.40× | 8.85 s | 379 → 370 MB | +1.2 % |
| skew T4 | 15.00 s [14.96, 15.02] | 11.02 s [10.98, 11.05] | 1.36× | 3.98 s | 435 → 435 MB | −0.1 % |
| wide T1 | 1.16 s [1.13, 1.20] | 1.15 s [1.14, 1.16] | 1.01× | 0.01 s | 235 → 235 MB | +0.5 % |
| wide T4 | 0.58 s [0.57, 0.62] | 0.57 s [0.54, 0.58] | 1.01× | 0.00 s | 221 → 220 MB | −0.4 % |
| or005 T1 | 13.89 s [13.79, 13.90] | 8.08 s [8.00, 8.10] | 1.72× | 5.81 s | 257 → 259 MB | −0.0 % |
| or005 T4 | 13.38 s [13.37, 13.50] | 7.76 s [7.68, 7.79] | 1.72× | 5.62 s | 281 → 286 MB | +1.0 % |
| or0001k2 T1 | 7.89 s [7.86, 7.91] | 2.81 s [2.81, 2.83] | 2.81× | 5.08 s | 768 → 579 MB | −0.1 % |
| or0001k2 T4 | 7.98 s [7.94, 7.99] | 2.87 s [2.86, 2.88] | 2.78× | 5.12 s | 854 → 672 MB | −0.8 % |

Base reproduces S1's `built` medians within 1.2 % in every regime.

## Rules

1. **Exactness: met.**
2. **Go: met.** Rule 2 holds in eight regimes: deepk, skew, or005 and
   or0001k2 at T1 and T4, 1.36–2.81× and 2.0–8.9 s saved. No regime is
   slower; smoke and wide gain at most 0.02 s, under the 1 s floor. Peak RSS
   is 0.75–1.02× of base's.
3. **Drift control: met.** `incore` agrees within −2.0 % to +1.2 %.
4. No regression to report.

## Where the time went

Diagnostic, outside the rules: `son_phases.py`, T1, one run per workload
(`base-diag.jsonl`, `diag.jsonl`). Seconds per profile phase; pass 1
includes the union's merges and its finish.

| workload | base pass 1 | base pass 2 | bound pass 1 | bound pass 2 | counted in pass 2 / union |
|---|---|---|---|---|---|
| smoke | 0.10 | 0.05 | 0.10 | 0.04 | 11 / 924 |
| deepk | 2.63 | 2.17 | 2.70 | 0.00 | 0 / 13,070 |
| skew | 13.79 | 17.81 | 14.02 | 8.87 | 97,121 / 221,111 |
| wide | 1.01 | 0.15 | 1.05 | 0.13 | 228 / 5,262 |
| or005 | 8.42 | 5.54 | 8.04 | 0.14 | 32,840 / 1,125,261 |
| or0001k2 | 6.20 | 1.79 | 2.82 | 0.00 | 0 / 5,421,817 |

- **The bound** takes or005's pass 2 from 5.5 to 0.14 s. Pass 2 reads no
  chunk on deepk and or0001k2: every candidate within the bound has its exact
  count from pass 1. On skew it counts 44 % of the union and halves pass 2.
- **The union** no longer hashes. or0001k2's pass 1 falls by 3.4 s, although
  each row now carries two int32 sums; its peak RSS falls by 180–190 MB
  because pass 2 no longer builds a Gram over 5.4M pairs.
- **Pass 1 is now most of SON**: 61 % of skew's wall time, 97–100 % of
  deepk's, or005's and or0001k2's.

## What SON costs against the in-core route

SON is now 1.3–28× slower than mining in core, against 1.4–48× at base
(deepk 1.26–1.38×, skew 6.3–8.4×, or005 18.6–27.7×, or0001k2 3.4–5.1×). What is
left is mostly or005's pass 1: four chunks at 0.9 × the threshold mine 1.13M
local itemsets, against 10,488 frequent ones.

## Levers not taken here

- Count a candidate only in the chunks that did not emit it: 290,827
  (candidate, chunk) pairs on skew against 388,484 (−25 %), and 78,644 on or005
  against 131,360 (`stakes.jsonl`, `pairs_to_count`). Each union row would have
  to carry the chunks that emitted it.
- The factor: at 1.0 the bound keeps 67–100 % of a smaller union on five
  workloads (or005 67.4 %, wide 95.5 %, smoke 99.3 %, skew 99.9 %, deepk
  100 %), because the slack is larger and fewer counts are exact. or0001k2 is
  the same at both factors (49.4 %, all exact): its local min_count is 1
  either way, so no chunk has slack. A lower factor might make pass 2 cheaper
  at a cost to pass 1. Changing it is the owner's call.
