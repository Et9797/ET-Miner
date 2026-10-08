# The input layer: Phases I1 and I2

Protocol: `bench/cpu/PROTOCOL.md`, Amendment 6. Harness:
`bench/cpu/son_stakes.py --matrix --arms built,incore --reps 3`, cap 600 s,
4 chunks, the S0/S1 box. Three runs, back to back:

- *base* at `c973ed3` (main after PR #29) with the input commit's harness
  file, so its rows carry `c973ed3+dirty`: `base.jsonl`, `base-env.txt`,
  21:09–21:18;
- *input* at `7fddd05`: `raw.jsonl`, `env.txt`, 21:18–21:26;
- *base again*, as base: `base2.jsonl`, `base2-env.txt`, 21:26–21:35.

All 252 rows are ok. The env files differ only in `rev`. The gate passed on
`7fddd05` before the run: ruff; 828 passed without the slow tests and the four
`test_support_00*`; the tier-equivalence, SON, streaming and free-set suites,
59 passed.

`stages.jsonl` is the diagnostic the amendment cites (`input_stages.py` at
`c973ed3`).

## Correctness

One signature per workload across the three files (`son_stakes.py --check`).

## Results

Medians over 3 reps with [min, max]. *ratio* is base / input; *drift* is base
again against base.

| regime | arm | base | input | ratio | saved | peak RSS base → input | drift |
|---|---|---|---|---|---|---|---|
| smoke T1 | incore | 0.08 s [0.08, 0.09] | 0.06 s [0.06, 0.07] | 1.27× | 0.02 s | 146 → 189 MB (1.30×) | −3.3 % |
| smoke T1 | built | 0.13 s [0.13, 0.14] | 0.11 s [0.11, 0.11] | 1.20× | 0.02 s | 130 → 185 MB (1.42×) | +2.9 % |
| smoke T4 | incore | 0.07 s [0.07, 0.08] | 0.06 s [0.06, 0.06] | 1.15× | 0.01 s | 161 → 207 MB (1.28×) | −1.4 % |
| smoke T4 | built | 0.13 s [0.13, 0.13] | 0.11 s [0.11, 0.12] | 1.14× | 0.02 s | 136 → 189 MB (1.39×) | −0.2 % |
| deepk T1 | incore | 2.05 s [2.04, 2.11] | 1.75 s [1.74, 1.76] | 1.18× | 0.31 s | 363 → 390 MB (1.08×) | −0.0 % |
| deepk T1 | built | 2.57 s [2.56, 2.59] | 2.21 s [2.20, 2.21] | 1.16× | 0.36 s | 284 → 326 MB (1.15×) | +0.1 % |
| deepk T4 | incore | 1.63 s [1.62, 1.70] | 1.48 s [1.48, 1.49] | 1.10× | 0.15 s | 395 → 432 MB (1.09×) | +0.0 % |
| deepk T4 | built | 2.25 s [2.24, 2.26] | 2.05 s [2.03, 2.07] | 1.10× | 0.20 s | 334 → 366 MB (1.10×) | −0.8 % |
| skew T1 | incore | 2.65 s [2.63, 2.66] | 2.30 s [2.29, 2.32] | 1.15× | 0.35 s | 406 → 443 MB (1.09×) | +0.0 % |
| skew T1 | built | 22.12 s [22.05, 22.32] | 21.52 s [21.37, 21.53] | 1.03× | 0.61 s | 379 → 438 MB (1.16×) | +0.4 % |
| skew T4 | incore | 1.71 s [1.71, 1.71] | 1.54 s [1.53, 1.56] | 1.11× | 0.17 s | 460 → 497 MB (1.08×) | +1.2 % |
| skew T4 | built | 10.84 s [10.82, 10.91] | 10.41 s [10.39, 10.46] | 1.04× | 0.42 s | 435 → 502 MB (1.15×) | +0.3 % |
| wide T1 | incore | 0.83 s [0.83, 0.83] | 0.75 s [0.75, 0.75] | 1.11× | 0.08 s | 271 → 310 MB (1.14×) | +0.1 % |
| wide T1 | built | 1.12 s [1.12, 1.14] | 1.00 s [1.00, 1.00] | 1.11× | 0.11 s | 235 → 298 MB (1.26×) | +0.5 % |
| wide T4 | incore | 0.37 s [0.37, 0.37] | 0.32 s [0.32, 0.32] | 1.16× | 0.05 s | 270 → 303 MB (1.12×) | −0.3 % |
| wide T4 | built | 0.55 s [0.53, 0.55] | 0.50 s [0.47, 0.50] | 1.10× | 0.05 s | 225 → 257 MB (1.14×) | +3.5 % |
| or005 T1 | incore | 0.44 s [0.43, 0.44] | 0.40 s [0.40, 0.40] | 1.08× | 0.03 s | 192 → 231 MB (1.20×) | −0.0 % |
| or005 T1 | built | 8.13 s [8.04, 8.14] | 7.95 s [7.93, 8.06] | 1.02× | 0.19 s | 256 → 311 MB (1.21×) | +0.6 % |
| or005 T4 | incore | 0.28 s [0.28, 0.28] | 0.26 s [0.26, 0.26] | 1.08× | 0.02 s | 200 → 228 MB (1.14×) | +0.3 % |
| or005 T4 | built | 7.76 s [7.67, 7.80] | 7.79 s [7.73, 7.80] | 1.00× | −0.03 s | 286 → 337 MB (1.18×) | −0.9 % |
| or0001k2 T1 | incore | 0.82 s [0.82, 0.82] | 0.79 s [0.79, 0.79] | 1.05× | 0.04 s | 336 → 375 MB (1.12×) | −0.3 % |
| or0001k2 T1 | built | 2.79 s [2.78, 2.79] | 2.76 s [2.75, 2.77] | 1.01× | 0.03 s | 578 → 635 MB (1.10×) | +0.3 % |
| or0001k2 T4 | incore | 0.56 s [0.55, 0.56] | 0.54 s [0.54, 0.54] | 1.04× | 0.02 s | 383 → 419 MB (1.09×) | −1.0 % |
| or0001k2 T4 | built | 2.87 s [2.86, 2.88] | 2.87 s [2.84, 2.87] | 1.00× | 0.00 s | 673 → 730 MB (1.08×) | +0.0 % |
| dslk2 T1 | incore | 17.47 s [17.41, 17.54] | 11.06 s [11.06, 11.06] | 1.58× | 6.41 s | 4963 → 5003 MB (1.01×) | −0.2 % |
| dslk2 T1 | built | 16.52 s [16.43, 16.58] | 10.57 s [10.57, 10.62] | 1.56× | 5.94 s | 2886 → 2950 MB (1.02×) | +0.2 % |
| dslk2 T4 | incore | 12.47 s [12.45, 12.50] | 8.21 s [8.17, 8.25] | 1.52× | 4.26 s | 4966 → 5005 MB (1.01×) | −1.1 % |
| dslk2 T4 | built | 11.07 s [11.02, 11.09] | 7.46 s [7.44, 7.49] | 1.48× | 3.61 s | 2926 → 2978 MB (1.02×) | −0.2 % |

## Rules

1. **Exactness: met.**
2. **Go: not met.**
   - Time: rule 2 holds in four regimes, all of them dslk2: `incore` 1.58× at
     T1 (6.41 s saved) and 1.52× at T4 (4.26 s), `built` 1.56× (5.94 s) and
     1.48× (3.61 s). No regime is slower under rule 2's 10 % and 0.1 s:
     or005 `built` T4 loses 0.03 s (0.4 %); elsewhere the change saves up
     to 0.61 s, below the 1 s floor.
   - Memory: `ru_maxrss_mb` exceeds 1.25 × base's in five regimes: smoke in
     all four (1.28–1.42×, +43–55 MB) and wide `built` T1 (1.26×, +63 MB).
     Every regime rose, by 27–67 MB.
3. **Drift control: met.** Base again agrees with base within −3.3 % to
   +3.5 %.
4. No timing regression to report.

## Why the memory rose

Reading the Arrow buffers (`Series.to_arrow`) imports pyarrow, which the CPU
route did not load before. Measured in a fresh process: 80 MB of RSS after
importing et_miner, 107 MB after `to_arrow` on a two-row Series. That fixed
cost is most of smoke's 0.06–0.19 s run.

Amendment 7 removes it: the integer path takes its values from Polars'
explode instead (`to_numpy`), keeping the bincount and the lookup table.

# Phase I2

Protocol: Amendment 7. The integer path reads the exploded values through
Polars (`to_numpy`) instead of the Arrow buffers. Two runs, back to back:

- *base again* at `c973ed3` with the input commit's harness file (unchanged
  from `7fddd05`'s), so its rows carry `c973ed3+dirty`: `base3.jsonl`,
  `base3-env.txt`, 21:42–21:51;
- *input* at `b328cc1`: `raw2.jsonl`, `env2.txt`, 21:51–21:59.

All 168 rows are ok. The env files differ only in `rev`, and `base3-env.txt`
equals `base-env.txt`. The gate passed on `b328cc1` before the run: ruff; 828
passed without the slow tests and the four `test_support_00*`; the
tier-equivalence, SON, streaming and free-set suites, 59 passed.

## Correctness

One signature per workload across all five files (`son_stakes.py --check`).

## Results

Medians over 3 reps with [min, max]. *ratio* is base again / input; *drift*
is I1's base against base again.

| regime | arm | base again | input | ratio | saved | peak RSS base → input | drift |
|---|---|---|---|---|---|---|---|
| smoke T1 | incore | 0.08 s [0.08, 0.08] | 0.07 s [0.07, 0.07] | 1.19× | 0.01 s | 145 → 134 MB (0.92×) | +0.5 % |
| smoke T1 | built | 0.14 s [0.13, 0.14] | 0.12 s [0.11, 0.12] | 1.18× | 0.02 s | 131 → 126 MB (0.96×) | −2.6 % |
| smoke T4 | incore | 0.08 s [0.07, 0.08] | 0.07 s [0.07, 0.07] | 1.10× | 0.01 s | 162 → 151 MB (0.93×) | −2.5 % |
| smoke T4 | built | 0.13 s [0.13, 0.13] | 0.12 s [0.12, 0.12] | 1.08× | 0.01 s | 136 → 130 MB (0.96×) | +1.4 % |
| deepk T1 | incore | 2.05 s [2.05, 2.06] | 1.76 s [1.75, 1.78] | 1.17× | 0.29 s | 363 → 347 MB (0.96×) | +0.3 % |
| deepk T1 | built | 2.58 s [2.55, 2.61] | 2.30 s [2.30, 2.32] | 1.12× | 0.28 s | 284 → 282 MB (0.99×) | −0.2 % |
| deepk T4 | incore | 1.62 s [1.61, 1.64] | 1.50 s [1.49, 1.50] | 1.09× | 0.13 s | 399 → 378 MB (0.95×) | +0.0 % |
| deepk T4 | built | 2.24 s [2.24, 2.28] | 2.11 s [2.11, 2.13] | 1.06× | 0.14 s | 335 → 321 MB (0.96×) | +0.4 % |
| skew T1 | incore | 2.63 s [2.63, 2.65] | 2.33 s [2.32, 2.33] | 1.13× | 0.30 s | 408 → 390 MB (0.96×) | +0.6 % |
| skew T1 | built | 22.22 s [22.11, 22.24] | 21.58 s [21.50, 21.61] | 1.03× | 0.64 s | 382 → 370 MB (0.97×) | −0.4 % |
| skew T4 | incore | 1.72 s [1.72, 1.73] | 1.59 s [1.58, 1.60] | 1.08× | 0.13 s | 462 → 443 MB (0.96×) | −0.7 % |
| skew T4 | built | 10.86 s [10.79, 10.88] | 10.57 s [10.56, 10.61] | 1.03× | 0.29 s | 433 → 433 MB (1.00×) | −0.2 % |
| wide T1 | incore | 0.83 s [0.83, 0.83] | 0.77 s [0.77, 0.77] | 1.08× | 0.06 s | 267 → 258 MB (0.97×) | +0.6 % |
| wide T1 | built | 1.12 s [1.11, 1.14] | 1.02 s [1.01, 1.02] | 1.10× | 0.10 s | 235 → 242 MB (1.03×) | −0.4 % |
| wide T4 | incore | 0.37 s [0.37, 0.37] | 0.33 s [0.33, 0.33] | 1.11× | 0.04 s | 269 → 252 MB (0.93×) | +0.8 % |
| wide T4 | built | 0.55 s [0.55, 0.56] | 0.49 s [0.49, 0.49] | 1.13× | 0.06 s | 218 → 204 MB (0.94×) | −0.7 % |
| or005 T1 | incore | 0.43 s [0.43, 0.44] | 0.41 s [0.41, 0.41] | 1.06× | 0.03 s | 192 → 180 MB (0.94×) | +0.2 % |
| or005 T1 | built | 8.00 s [7.97, 8.02] | 8.02 s [8.01, 8.10] | 1.00× | −0.02 s | 257 → 256 MB (1.00×) | +1.6 % |
| or005 T4 | incore | 0.28 s [0.28, 0.29] | 0.26 s [0.26, 0.26] | 1.05× | 0.01 s | 200 → 183 MB (0.92×) | +0.4 % |
| or005 T4 | built | 7.62 s [7.61, 7.83] | 7.63 s [7.62, 7.64] | 1.00× | −0.01 s | 288 → 281 MB (0.98×) | +1.9 % |
| or0001k2 T1 | incore | 0.82 s [0.82, 0.83] | 0.79 s [0.79, 0.80] | 1.04× | 0.03 s | 336 → 324 MB (0.96×) | +0.3 % |
| or0001k2 T1 | built | 2.79 s [2.78, 2.79] | 2.78 s [2.75, 2.78] | 1.00× | 0.00 s | 578 → 576 MB (0.99×) | +0.0 % |
| or0001k2 T4 | incore | 0.56 s [0.56, 0.56] | 0.54 s [0.54, 0.55] | 1.03× | 0.01 s | 382 → 368 MB (0.96×) | +0.8 % |
| or0001k2 T4 | built | 2.85 s [2.85, 2.86] | 2.85 s [2.84, 2.88] | 1.00× | 0.00 s | 672 → 669 MB (1.00×) | +0.7 % |
| dslk2 T1 | incore | 17.46 s [17.42, 17.51] | 11.61 s [11.60, 11.62] | 1.50× | 5.85 s | 4961 → 4956 MB (1.00×) | +0.1 % |
| dslk2 T1 | built | 16.54 s [16.45, 16.57] | 11.73 s [11.61, 11.78] | 1.41× | 4.81 s | 2882 → 2898 MB (1.01×) | −0.1 % |
| dslk2 T4 | incore | 12.47 s [12.37, 12.54] | 8.69 s [8.62, 8.77] | 1.43× | 3.78 s | 4968 → 4953 MB (1.00×) | −0.1 % |
| dslk2 T4 | built | 11.07 s [11.05, 11.07] | 8.53 s [8.47, 8.56] | 1.30× | 2.53 s | 2924 → 2924 MB (1.00×) | +0.0 % |

## Rules

1. **Exactness: met.**
2. **Go: met.** Rule 2 holds in the four dslk2 regimes: `incore` 1.50× at T1
   (5.85 s saved) and 1.43× at T4 (3.78 s), `built` 1.41× (4.81 s) and 1.30×
   (2.53 s). No regime is slower under rule 2's 10 % and 0.1 s: or005
   `built` loses 0.02 s (0.2 %) at T1 and at T4. Elsewhere the change saves
   up to 0.64 s (skew `built` T1), below the 1 s floor. Peak RSS is
   0.92–1.03× of base's.
3. **Drift control: met.** I1's base agrees with base again within −2.6 % to
   +1.9 %.
4. No regression to report.

## Against I1

Without pyarrow, peak RSS falls by 43–70 MB against I1's input and lies
below base's in 25 of 28 regimes (the others: wide `built` T1 +7 MB, dslk2
`built` +16 MB at T1 and +0 MB at T4). The explode costs time on dslk2:
`incore` 11.06 → 11.61 s at T1 and 8.21 → 8.69 s at T4, `built` 10.57 →
11.73 s and 7.46 → 8.53 s. The other regimes move by at most 0.16 s.
