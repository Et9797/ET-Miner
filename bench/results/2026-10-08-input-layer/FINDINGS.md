# The input layer: Phase I1

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
     1.48× (3.61 s). No regime is slower; elsewhere the change saves
     0.00–0.61 s, below the 1 s floor.
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
