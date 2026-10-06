# ESCO fit check on deep_sparse_large (2026-10-06)

Where `sparse_from_k="auto"` transitions on deep_sparse_large once the
conversion checks that the tidsets fit (`TidsetFitError`, commit `3fa9962`),
and the peak VRAM it then reaches. One rep per config: no timing claims.

Run: `uv run python bench/results/2026-10-06-esco-fit/run.py` (2x RTX A4000,
`NCCL_P2P_DISABLE=1`, box idle). The run used the same script with an absolute
`REPO` path. Budget 0.1 GPU-hour, used 0.037. Evidence: `raw.jsonl`, `env.txt`.

| config | transition | peak VRAM per GPU (MB) |
|---|---|---|
| dsl 1 GPU, auto | K=14 | 12,421 |
| dsl 1 GPU, auto, count inference | K=14 | 12,421 |
| dsl 2 GPUs, auto | K=14 | 4,317 / 4,311 |
| dsl 2 GPUs, auto, count inference | K=14 | 4,317 / 4,311 |

- K=6 to K=13 stay dense: the conversion bound falls from 251.8 GB to
  35.8 GB, against 16.46 GB available. Before the check, the K=6 conversion
  ran out of memory (phase A calibration: the K=5 tidsets are 125.6 GB).
- K=14 converts 2,564 itemsets (4.6 GB of tidsets); ESCO then mines 504, 64
  and 4 frequent itemsets.
- The peak comes from the conversion (tidsets twice plus the AND batch) and
  is identical with and without count inference, so O4 has nothing to save on
  this workload. Dense reached 2,267 MB on one GPU in phase A
  (`../2026-10-05-optimizations/raw.jsonl`, `dsl-C1-base`).
- `peak_vram_mb` samples nvidia-smi every 0.5 s and is a lower bound.
