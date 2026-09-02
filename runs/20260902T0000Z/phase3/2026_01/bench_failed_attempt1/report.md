# GPU campaign report

Runs: 0 ok, 27 failed/timeout. Raw data: `raw.jsonl`, env: `env.txt`.

## Wall time by config (median over reps)

| config | preset | variant | filter | gpus | reps | median s | min s | peak VRAM MB | throttled |
|---|---|---|---|---|---|---|---|---|---|

## Kernel variant A/B (legacy → shared, median wall)

| preset | gpus | legacy s | shared s | speedup |
|---|---|---|---|---|

## Per-level timings (slowest run per preset)

## Failures / timeouts

- `stressk2-filter-compact#r0`: no-result (rc=1)
- `stressk2-filter-cupy#r0`: no-result (rc=1)
- `stressk2-filter-cpu#r0`: no-result (rc=1)
- `deepk-nonccl#r0`: no-result (rc=1)
- `deepk-density-auto#r0`: no-result (rc=1)
- `deepk-density-auto-1g#r0`: no-result (rc=1)
- `twophase-smoke#r0`: no-result (rc=1)
- `skew-rows#r0`: no-result (rc=1)
- `skew-nnz#r0`: no-result (rc=1)
- `skew-rows#r1`: no-result (rc=1)
- `skew-nnz#r1`: no-result (rc=1)
- `stressk2-legacy-1g#r0`: no-result (rc=1)
- `stressk2-legacy-2g#r0`: no-result (rc=1)
- `stressk2-shared-1g#r0`: no-result (rc=1)
- `deepk-legacy-1g#r0`: no-result (rc=1)
- `deepk-shared-1g#r0`: no-result (rc=1)
- `stressk2-shared-2g#r0`: no-result (rc=1)
- `deepk-legacy-2g#r0`: no-result (rc=1)
- `deepk-shared-2g#r0`: no-result (rc=1)
- `stressk2-shared-1g#r1`: no-result (rc=1)
- `deepk-shared-1g#r1`: no-result (rc=1)
- `stressk2-shared-2g#r1`: no-result (rc=1)
- `deepk-shared-2g#r1`: no-result (rc=1)
- `stressk2-shared-1g#r2`: no-result (rc=1)
- `deepk-shared-1g#r2`: no-result (rc=1)
- `stressk2-shared-2g#r2`: no-result (rc=1)
- `deepk-shared-2g#r2`: no-result (rc=1)
