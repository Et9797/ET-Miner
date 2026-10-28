# Phase 4 (routes-final) against Phase 2, by base_id

Rows: `raw.jsonl` here (195 rows, 65 configs x 3, stamped `af85dad`, 19:22-21:38 on 2x RTX 3060 12 GB)
against `../2026-10-27-routes-cpu/raw.jsonl` and `../2026-10-27-routes-gpu/raw.jsonl` (Phase 2, same
`base_id`s, no pins). Wall = median over the ok reps; VRAM (largest device) and RSS = max over the ok reps.
sig = `bench/routes/decide.py::_sig` (n_itemsets, sum_counts, itemset_hash) of every ok rep.

**Result.** Every one of the 64 configs both phases ran ok mines the same lattice (one signature per
config, equal to Phase 2's). No config got slower beyond noise: P4/P2 is 0.57-1.15, and every ratio
above 1.07 is a sub-second config (deepk15 0.11 -> 0.13 s, skew-C1 0.17 -> 0.20 s). The one fall is
skew-son-D1-default (18.8 -> 10.7 s, C2). sk2ml3-bv-C2-default fails as in Phase 2, now with an
upfront `MemoryError` (2aa6b72) instead of an `OutOfMemoryError` mid-run.

Memory, read with care:
- VRAM is sampled by `nvidia-smi` every 0.5 s (`bench/child_run.py:63-99`), so short peaks are
  missed at random: dsl-wide-C1-coded peaks 1229 / 1465 / 2555 MB in Phase 2 and 2469 / 4603 / 4807 MB
  here, on the same path (identical logs) and in the same time. A max-over-reps difference of
  this kind is sampling, not a regression.
- RSS is sampled the same way but rises consistently on the CPU route with free-sets or count
  inference: dsl-free-F1 5138-5330 -> 6935-6973 MB, dsl-infer-F1 5226-5491 -> 7020-7068 MB
  (+1.6-1.8 GB, 1.3x, every rep). See REPORT.md for its attribution.

The commit message of `964c087` says the run ended "at about 21:55"; it ended at 21:38.

| base_id | P2 wall s | P4 wall s | P4/P2 | P2 VRAM MB | P4 VRAM MB | P2 RSS MB | P4 RSS MB | sig | notes |
|---|---|---|---|---|---|---|---|---|---|
| deepk-C1-default | 0.17 | 0.17 | 1.02 | 191 | 183 | 858 | 858 | same |  |
| deepk-F1-default | 2.68 | 2.31 | 0.86 | 4 | 4 | 592 | 627 | same |  |
| deepk-R1-default | 0.89 | 0.88 | 0.99 | 4 | 4 | 853 | 901 | same |  |
| deepk-son-D1-default | 3.77 | 3.58 | 0.95 | 4 | 4 | 560 | 560 | same |  |
| deepk-son-D1-ported | 0.43 | 0.38 | 0.89 | 157 | 157 | 852 | 854 | same |  |
| deepk-son-E2-ported | 1.15 | 1.24 | 1.07 | 281 | 267 | 1394 | 1415 | same |  |
| deepk15-C1-default | 0.11 | 0.13 | 1.15 | 205 | 191 | 858 | 859 | same |  |
| deepk20-C1-default | 4.96 | 4.94 | 1.00 | 177 | 177 | 2566 | 2553 | same |  |
| deepk20-F1-default | 65.41 | 63.87 | 0.98 | 4 | 4 | 2322 | 2327 | same |  |
| dsl-C1-default | 3.95 | 3.95 | 1.00 | 2555 | 2469 | 3522 | 3521 | same |  |
| dsl-C2-default | 3.26 | 3.24 | 0.99 | 1419 | 1375 | 3731 | 3731 | same |  |
| dsl-R1-default | 226.83 | 224.55 | 0.99 | 4 | 4 | 8044 | 8085 | same |  |
| dsl-bv-C1-default | 8.92 | 8.95 | 1.00 | 2205 | 2205 | 10309 | 10309 | same |  |
| dsl-bv-C2-default | 5.13 | 5.13 | 1.00 | 2273 | 2273 | 10690 | 10689 | same |  |
| dsl-free-C1-default | 2.84 | 2.91 | 1.03 | 2555 | 2469 | 3444 | 3445 | same |  |
| dsl-free-F1-default | 272.28 | 263.21 | 0.97 | 4 | 4 | 5330 | 6973 | same |  |
| dsl-infer-C1-default | 2.49 | 2.49 | 1.00 | 2555 | 2469 | 3529 | 3528 | same |  |
| dsl-infer-F1-default | 277.82 | 270.32 | 0.97 | 4 | 4 | 5491 | 7068 | same |  |
| dsl-son-D1-default | 1091.21 | 1084.89 | 0.99 | 4 | 4 | 3897 | 4230 | same |  |
| dsl-son-D1-ported | 8.08 | 8.12 | 1.00 | 1237 | 725 | 3228 | 3182 | same |  |
| dsl-son-E2-ported | 7.93 | 7.99 | 1.01 | 375 | 527 | 3664 | 3653 | same |  |
| dsl-str-C1-coded | 18.30 | 18.35 | 1.00 | 2555 | 4371 | 14074 | 14067 | same |  |
| dsl-wide-C1-coded | 10.70 | 10.75 | 1.00 | 2555 | 4807 | 11114 | 10748 | same |  |
| dslxl-C1-default | 10.42 | 10.45 | 1.00 | 7375 | 7113 | 8462 | 8514 | same |  |
| dslxl-son-D1-ported | 14.87 | 14.58 | 0.98 | 3435 | 1889 | 7444 | 7446 | same |  |
| dslxl-son-E2-ported | 12.68 | 12.89 | 1.02 | 1119 | 1085 | 7833 | 7831 | same |  |
| oom2-C1-default | 0.30 | 0.29 | 0.99 | 1923 | 1923 | 901 | 928 | same |  |
| oom2-C2-default | 0.66 | 0.65 | 0.98 | 3097 | 1117 | 1338 | 1339 | same |  |
| oom2-bv-C1-default | 8.47 | 8.47 | 1.00 | 3641 | 3641 | 1279 | 1266 | same |  |
| oom2-bv-C2-default | 5.22 | 5.15 | 0.99 | 4687 | 4687 | 1668 | 1666 | same |  |
| oom2ml3-C1-default | 0.36 | 0.37 | 1.04 | 1971 | 1971 | 968 | 969 | same |  |
| oom2ml3-bv-C1-default | 10.05 | 10.08 | 1.00 | 7865 | 7927 | 1262 | 1281 | same |  |
| oom2ml3-bv-C2-default | 6.52 | 6.47 | 0.99 | 8221 | 8283 | 1928 | 1664 | same |  |
| or0001k2-C1-default | 0.51 | 0.52 | 1.02 | 237 | 237 | 1631 | 1613 | same |  |
| or0001k2-F1-default | 0.74 | 0.66 | 0.90 | 4 | 4 | 1328 | 1330 | same |  |
| or0001k2-R1-default | 18.10 | 18.35 | 1.01 | 4 | 4 | 1958 | 1945 | same |  |
| or002-C1-default | 0.28 | 0.27 | 0.96 | 147 | 163 | 903 | 904 | same |  |
| or002-C2-default | 0.57 | 0.54 | 0.94 | 249 | 253 | 1318 | 1318 | same |  |
| or002-F1-default | 6.58 | 5.68 | 0.86 | 4 | 4 | 676 | 766 | same |  |
| or002-R1-default | 93.61 | 93.79 | 1.00 | 4 | 4 | 710 | 704 | same |  |
| or002-str-C1-coded | 0.40 | 0.41 | 1.04 | 145 | 145 | 933 | 931 | same |  |
| or003-C1-default | 0.11 | 0.09 | 0.84 | 143 | 157 | 715 | 715 | same |  |
| or003-F1-default | 1.38 | 1.27 | 0.92 | 4 | 4 | 461 | 565 | same |  |
| or003-R1-default | 3.31 | 3.31 | 1.00 | 4 | 4 | 374 | 376 | same |  |
| or005-C1-default | 0.06 | 0.06 | 1.05 | 141 | 141 | 716 | 724 | same |  |
| or005-F1-default | 0.39 | 0.36 | 0.92 | 4 | 4 | 326 | 327 | same |  |
| or005-R1-default | 0.37 | 0.33 | 0.89 | 4 | 4 | 334 | 336 | same |  |
| or005-son-D1-default | 5.72 | 4.98 | 0.87 | 4 | 4 | 426 | 455 | same |  |
| sk2ml2-C1-default | 1.10 | 1.10 | 1.00 | 10505 | 10505 | 1700 | 1700 | same |  |
| sk2ml2-C2-default | 1.32 | 1.41 | 1.07 | 7007 | 7007 | 2045 | 2051 | same |  |
| sk2ml2-bv-C1-default | 44.94 | 44.91 | 1.00 | 8671 | 8671 | 2843 | 2881 | same |  |
| sk2ml3-C1-default | 1.72 | 1.72 | 1.00 | 10513 | 10513 | 2134 | 2134 | same |  |
| sk2ml3-bv-C1-default | 72.78 | 72.89 | 1.00 | 11901 | 11901 | 2852 | 2882 | same |  |
| sk2ml3-bv-C2-default | - | - | - | - | - | - | - | - | P4 0/3 ok; P4 bad: ['error: MemoryError: bitvecs= on 2 GPUs: device 0 needs 4.32 ', 'skipped: rep 0 error: MemoryError: bitvecs= on 2 GPUs: devic', 'skipped: rep 0 error: MemoryError: bitvecs= on 2 GPUs: devic'] |
| skew-C1-default | 0.17 | 0.20 | 1.12 | 195 | 219 | 871 | 880 | same |  |
| skew-F1-default | 2.63 | 2.10 | 0.80 | 4 | 4 | 692 | 770 | same |  |
| skew-R1-default | 1.04 | 1.04 | 1.00 | 4 | 4 | 778 | 866 | same |  |
| skew-son-D1-default | 18.83 | 10.65 | 0.57 | 4 | 4 | 731 | 778 | same |  |
| smoke-C1-default | 0.05 | 0.05 | 0.98 | 135 | 135 | 706 | 707 | same |  |
| smoke-F1-default | 0.09 | 0.07 | 0.70 | 4 | 4 | 312 | 318 | same |  |
| smoke-R1-default | 0.03 | 0.03 | 0.91 | 4 | 4 | 300 | 301 | same |  |
| wide-C1-default | 0.07 | 0.08 | 1.01 | 203 | 203 | 723 | 722 | same |  |
| wide-F1-default | 0.39 | 0.37 | 0.97 | 4 | 4 | 349 | 347 | same |  |
| wide-R1-default | 0.60 | 0.65 | 1.07 | 4 | 4 | 357 | 358 | same |  |
| wide-son-D1-default | 1.06 | 1.02 | 0.96 | 4 | 4 | 414 | 405 | same |  |

signature mismatches: none
no Phase 2 row: none
