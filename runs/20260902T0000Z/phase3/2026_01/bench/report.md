# GPU campaign report

Runs: 26 ok, 1 failed/timeout. Raw data: `raw.jsonl`, env: `env.txt`.

## Wall time by config (median over reps)

| config | preset | variant | filter | gpus | reps | median s | min s | peak VRAM MB | throttled |
|---|---|---|---|---|---|---|---|---|---|
| deepk-density-auto | deep_k | legacy | compact | 2 | 1 | 2.1 | 2.1 | 587 |  |
| deepk-density-auto-1g | deep_k | legacy | compact | 1 | 1 | 0.9 | 0.9 | 31 |  |
| deepk-legacy-1g | deep_k | legacy | compact | 1 | 1 | 0.8 | 0.8 | 71 | ⚠ |
| deepk-legacy-2g | deep_k | legacy | compact | 2 | 1 | 2.1 | 2.1 | 427 | ⚠ |
| deepk-nonccl | deep_k | legacy | compact | 2 | 1 | 1.2 | 1.2 | 311 |  |
| deepk-shared-1g | deep_k | shared | compact | 1 | 3 | 0.8 | 0.8 | 335 | ⚠ |
| deepk-shared-2g | deep_k | shared | compact | 2 | 3 | 2.1 | 2.1 | 427 | ⚠ |
| skew-nnz | skewed_rows | legacy | compact | 2 | 2 | 2.3 | 2.3 | 429 | ⚠ |
| skew-rows | skewed_rows | legacy | compact | 2 | 2 | 2.3 | 2.3 | 427 | ⚠ |
| stressk2-filter-compact | stress_k2 | legacy | compact | 2 | 1 | 91.4 | 91.4 | 6931 | ⚠ |
| stressk2-filter-cpu | stress_k2 | legacy | cpu | 2 | 1 | 93.8 | 93.8 | 8452 | ⚠ |
| stressk2-filter-cupy | stress_k2 | legacy | cupy | 2 | 1 | 91.9 | 91.9 | 7009 | ⚠ |
| stressk2-legacy-2g | stress_k2 | legacy | compact | 2 | 1 | 2001.9 | 2001.9 | 16587 | ⚠ |
| stressk2-shared-1g | stress_k2 | shared | compact | 1 | 3 | 269.7 | 269.2 | 17339 | ⚠ |
| stressk2-shared-2g | stress_k2 | shared | compact | 2 | 3 | 128.4 | 128.4 | 16587 | ⚠ |
| twophase-smoke | smoke | legacy | compact | 2 | 1 | 2.2 | 2.2 | 419 | ⚠ |

## Kernel variant A/B (legacy → shared, median wall)

| preset | gpus | legacy s | shared s | speedup |
|---|---|---|---|---|
| deep_k | 2 | 2.1 | 2.1 | 1.02× |
| stress_k2 | 2 | 2001.9 | 128.4 | 15.59× |

## Per-level timings (slowest run per preset)

**deep_k** (deepk-legacy-2g#r0, 2.128s):

| K | candidates | frequent | ms |
|---|---|---|---|
| 1 | 112 | 112 | 59 |
| 2 | 0 | 471 | 114 |
| 3 | 0 | 901 | 13 |
| 4 | 0 | 1,407 | 12 |
| 5 | 0 | 1,854 | 8 |
| 6 | 0 | 1,848 | 6 |
| 7 | 0 | 1,320 | 6 |
| 8 | 0 | 660 | 5 |
| 9 | 0 | 220 | 5 |
| 10 | 0 | 44 | 5 |
| 11 | 0 | 4 | 4 |

**skewed_rows** (skew-nnz#r1, 2.332s):

| K | candidates | frequent | ms |
|---|---|---|---|
| 1 | 118 | 118 | 53 |
| 2 | 0 | 717 | 106 |
| 3 | 0 | 1,928 | 12 |
| 4 | 0 | 2,932 | 32 |
| 5 | 0 | 2,683 | 67 |
| 6 | 0 | 1,458 | 80 |
| 7 | 0 | 443 | 43 |
| 8 | 0 | 67 | 11 |
| 9 | 0 | 4 | 5 |

**smoke** (twophase-smoke#r0, 2.234s):

| K | candidates | frequent | ms |
|---|---|---|---|
| 1 | 118 | 118 | 393 |
| 2 | 0 | 290 | 95 |
| 3 | 3,748 | 202 | 40 |
| 4 | 142 | 22 | 5 |
| 5 | 0 | 0 | 0 |
| 1 | 118 | 118 | 3 |
| 2 | 0 | 290 | 68 |
| 3 | 3,748 | 202 | 12 |
| 4 | 142 | 22 | 5 |
| 5 | 0 | 0 | 0 |

**stress_k2** (stressk2-legacy-2g#r0, 2001.852s):

| K | candidates | frequent | ms |
|---|---|---|---|
| 1 | 35,000 | 35,000 | 67 |
| 2 | 0 | 1,660,332 | 86570 |
| 3 | 0 | 1,310,438 | 1910758 |

## Failures / timeouts

- `stressk2-legacy-1g#r0`: timeout
