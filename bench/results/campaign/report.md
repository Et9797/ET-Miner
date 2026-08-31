# GPU campaign report

Runs: 6 ok, 0 failed/timeout. Raw data: `raw.jsonl`, env: `env.txt`.

## Wall time by config (median over reps)

| config | preset | variant | filter | gpus | reps | median s | min s | peak VRAM MB | throttled |
|---|---|---|---|---|---|---|---|---|---|
| smoke-legacy-1g | smoke | legacy | compact | 1 | 1 | 0.2 | 0.2 | 4 |  |
| smoke-legacy-2g | smoke | legacy | compact | 2 | 1 | 1.0 | 1.0 | 408 |  |
| smoke-shared-1g | smoke | shared | compact | 1 | 1 | 0.2 | 0.2 | 4 |  |
| smoke-shared-2g | smoke | shared | compact | 2 | 1 | 1.0 | 1.0 | 380 |  |
| stressk2ml2-legacy-2g | stress_k2 | legacy | compact | 2 | 1 | 89.7 | 89.7 | 6942 | ⚠ |
| stressk2ml2-shared-2g | stress_k2 | shared | compact | 2 | 1 | 10.6 | 10.6 | 6920 | ⚠ |

## Kernel variant A/B (legacy → shared, median wall)

| preset | gpus | legacy s | shared s | speedup |
|---|---|---|---|---|
| smoke | 1 | 0.2 | 0.2 | 1.07× |
| smoke | 2 | 1.0 | 1.0 | 0.99× |
| stress_k2 | 2 | 89.7 | 10.6 | 8.44× |

## Per-level timings (slowest run per preset)

**smoke** (smoke-shared-2g#r0, 1.011s):

| K | candidates | frequent | ms |
|---|---|---|---|
| 1 | 118 | 118 | 27 |
| 2 | 0 | 290 | 48 |
| 3 | 0 | 202 | 4 |
| 4 | 0 | 63 | 5 |
| 5 | 0 | 18 | 3 |
| 6 | 0 | 3 | 2 |

**stress_k2** (stressk2ml2-legacy-2g#r0, 89.674s):

| K | candidates | frequent | ms |
|---|---|---|---|
| 1 | 35,000 | 35,000 | 47 |
| 2 | 0 | 1,660,332 | 86417 |
