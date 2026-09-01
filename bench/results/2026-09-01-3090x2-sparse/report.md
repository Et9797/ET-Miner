# GPU campaign report

Runs: 13 ok, 0 failed/timeout. Raw data: `raw.jsonl`, env: `env.txt`.

## Wall time by config (median over reps)

| config | preset | variant | filter | gpus | reps | median s | min s | peak VRAM MB | throttled |
|---|---|---|---|---|---|---|---|---|---|
| deepk-density-auto | deep_k | legacy | compact | 2 | 1 | 1.3 | 1.3 | 408 |  |
| deepk-density-auto-1g | deep_k | legacy | compact | 1 | 1 | 0.7 | 0.7 | 680 |  |
| deepk-legacy-1g | deep_k | legacy | compact | 1 | 1 | 0.6 | 0.6 | 352 | ⚠ |
| deepk-legacy-2g | deep_k | legacy | compact | 2 | 1 | 1.3 | 1.3 | 416 | ⚠ |
| deepk-nonccl | deep_k | legacy | compact | 2 | 1 | 0.9 | 0.9 | 310 | ⚠ |
| deepk-prefilter-off | deep_k | legacy | compact | 1 | 1 | 0.6 | 0.6 | 352 | ⚠ |
| deepk-shared-1g | deep_k | shared | compact | 1 | 3 | 0.6 | 0.6 | 352 | ⚠ |
| deepk-shared-2g | deep_k | shared | compact | 2 | 3 | 1.4 | 1.3 | 416 | ⚠ |
| deepk-single-prefilter-on | deep_k | legacy | compact | 1 | 1 | 0.6 | 0.6 | 352 | ⚠ |

## Kernel variant A/B (legacy → shared, median wall)

| preset | gpus | legacy s | shared s | speedup |
|---|---|---|---|---|
| deep_k | 2 | 1.3 | 1.4 | 0.99× |

## Per-level timings (slowest run per preset)

**deep_k** (deepk-shared-2g#r2, 1.382s):

| K | candidates | frequent | ms |
|---|---|---|---|
| 1 | 112 | 112 | 25 |
| 2 | 0 | 471 | 50 |
| 3 | 0 | 901 | 7 |
| 4 | 0 | 1,407 | 9 |
| 5 | 0 | 1,854 | 6 |
| 6 | 0 | 1,848 | 3 |
| 7 | 0 | 1,320 | 3 |
| 8 | 0 | 660 | 2 |
| 9 | 0 | 220 | 2 |
| 10 | 0 | 44 | 2 |
| 11 | 0 | 4 | 2 |
