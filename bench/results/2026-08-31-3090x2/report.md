# GPU campaign report

Runs: 28 ok, 0 failed/timeout. Raw data: `raw.jsonl`, env: `env.txt`.

## Wall time by config (median over reps)

| config | preset | variant | filter | gpus | reps | median s | min s | peak VRAM MB | throttled |
|---|---|---|---|---|---|---|---|---|---|
| deepk-density-auto | deep_k | legacy | compact | 2 | 1 | 10.4 | 10.4 | 416 |  |
| deepk-legacy-1g | deep_k | legacy | compact | 1 | 1 | 0.6 | 0.6 | 352 | ⚠ |
| deepk-legacy-2g | deep_k | legacy | compact | 2 | 1 | 1.3 | 1.3 | 416 |  |
| deepk-nonccl | deep_k | legacy | compact | 2 | 1 | 0.9 | 0.9 | 310 |  |
| deepk-prefilter-off | deep_k | legacy | compact | 1 | 1 | 0.6 | 0.6 | 352 |  |
| deepk-shared-1g | deep_k | shared | compact | 1 | 3 | 0.6 | 0.6 | 352 | ⚠ |
| deepk-shared-2g | deep_k | shared | compact | 2 | 3 | 1.4 | 1.3 | 416 |  |
| deepk-single-prefilter-on | deep_k | legacy | compact | 1 | 1 | 0.6 | 0.6 | 352 | ⚠ |
| skew-nnz | skewed_rows | legacy | compact | 2 | 2 | 1.5 | 1.5 | 418 |  |
| skew-rows | skewed_rows | legacy | compact | 2 | 2 | 1.4 | 1.4 | 416 |  |
| stressk2-filter-compact | stress_k2 | legacy | compact | 2 | 1 | 89.6 | 89.6 | 6918 | ⚠ |
| stressk2-filter-cpu | stress_k2 | legacy | cpu | 2 | 1 | 90.2 | 90.2 | 6920 | ⚠ |
| stressk2-filter-cupy | stress_k2 | legacy | cupy | 2 | 1 | 89.7 | 89.7 | 6998 | ⚠ |
| stressk2-legacy-1g | stress_k2 | legacy | compact | 1 | 1 | 2238.8 | 2238.8 | 17340 | ⚠ |
| stressk2-legacy-2g | stress_k2 | legacy | compact | 2 | 1 | 1996.7 | 1996.7 | 16576 | ⚠ |
| stressk2-shared-1g | stress_k2 | shared | compact | 1 | 3 | 253.3 | 253.1 | 17340 | ⚠ |
| stressk2-shared-2g | stress_k2 | shared | compact | 2 | 3 | 122.3 | 122.3 | 16576 | ⚠ |
| twophase-smoke | smoke | legacy | compact | 2 | 1 | 1.6 | 1.6 | 408 | ⚠ |

## Kernel variant A/B (legacy → shared, median wall)

| preset | gpus | legacy s | shared s | speedup |
|---|---|---|---|---|
| deep_k | 2 | 1.3 | 1.4 | 0.97× |
| stress_k2 | 2 | 1996.7 | 122.3 | 16.32× |

## Per-level timings (slowest run per preset)

**deep_k** (deepk-density-auto#r0, 10.431s):

| K | candidates | frequent | ms |
|---|---|---|---|
| 1 | 112 | 112 | 27 |
| 2 | 0 | 471 | 46 |
| 3 | 0 | 901 | 5 |
| 4 | 0 | 1,407 | 6 |
| 5 | 0 | 1,854 | 3145 |
| 6 | 0 | 1,848 | 2553 |
| 7 | 0 | 1,320 | 1817 |
| 8 | 0 | 660 | 902 |
| 9 | 0 | 220 | 374 |
| 10 | 0 | 44 | 65 |
| 11 | 0 | 4 | 14 |

**skewed_rows** (skew-nnz#r1, 1.517s):

| K | candidates | frequent | ms |
|---|---|---|---|
| 1 | 118 | 118 | 26 |
| 2 | 0 | 717 | 47 |
| 3 | 0 | 1,928 | 7 |
| 4 | 0 | 2,932 | 16 |
| 5 | 0 | 2,683 | 34 |
| 6 | 0 | 1,458 | 40 |
| 7 | 0 | 443 | 22 |
| 8 | 0 | 67 | 5 |
| 9 | 0 | 4 | 2 |

**smoke** (twophase-smoke#r0, 1.582s):

| K | candidates | frequent | ms |
|---|---|---|---|
| 1 | 118 | 118 | 252 |
| 2 | 0 | 290 | 48 |
| 3 | 0 | 202 | 120 |
| 4 | 0 | 22 | 7 |
| 5 | 0 | 0 | 0 |
| 1 | 118 | 118 | 2 |
| 2 | 0 | 290 | 34 |
| 3 | 0 | 202 | 87 |
| 4 | 0 | 22 | 5 |
| 5 | 0 | 0 | 0 |

**stress_k2** (stressk2-legacy-1g#r0, 2238.846s):

| K | candidates | frequent | ms |
|---|---|---|---|
| 1 | 35,000 | 35,000 | 72 |
| 2 | 612,482,500 | 1,660,332 | 175580 |
| 3 | 1,301,153 | 1,301,153 | 2046663 |
