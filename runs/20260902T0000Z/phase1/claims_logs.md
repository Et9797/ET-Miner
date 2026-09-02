# Phase 1 — claims extracted from surviving log / result files

Generated 2026-09-02 by the log-claims subagent. Repo root: `/root/projects/ET-Miner` (all `file` cells in Section A are relative to it).
Scope: every log-like or result-like file that survived in the repo and in `/root` (home). NOT judged for correctness — extraction only.
Paper sources (`paper/*.tex`, `paper/*.md`) were inventoried (Section B) but deliberately NOT extracted here; they are covered by the `claims_tex.md` / `claims_reviews_*.md` subagents.

**Headline:** the ONLY surviving AlphaFold result artifacts are the two `results_214m/*.json` files (2026-02-19, 1K vocab, 76,890,945 transactions), the decoded K=15–19 itemset dump, and hard-coded numbers in the notebook / GLOSSARY / README / RUNBOOK. No mining log (`*.log`, `nohup.out`, `experiment_log_*.txt`), no per-K parquet, no `itemsets_214m_godmode.parquet`, no `item_mapping_214m.parquet`, no `transactions_214m*.parquet`, and no `experiment_full_campaign_*.json` exist anywhere on this machine. The only real logs on the box (`bench/results/**/raw.jsonl`, `env.txt`, `report.md`, `FINDINGS.md`) are the synthetic-preset GPU benchmark campaign on 2× RTX 3090 (2026-08-31 / 2026-09-01), which the paper may cite for kernel throughput but which is not AlphaFold data.

## SECTION A — claim rows

Categories: deterministic | hardware-dependent | method-parameter | external-fact | software. Line refs: `L<n>` = 1-based line in file; `cell N (md/code) L<n>` for the notebook; `(file)` = whole-file property; `(derived)` = value computed by the extractor from the file's own numbers (for cross-checking, not a claim in the file).

| ID | file | line | value | unit | category | quoted context (≤ 25 words, verbatim) |
|---|---|---|---|---|---|---|
| L-001 | `bench/results/2026-08-31-3090x2/env.txt` | (file) | mtime 2026-09-01 22:47 UTC; size 2194 B; git-added 6f789d8 2026-08-31 22:39:15 +0000 | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-002 | `bench/results/2026-08-31-3090x2/report.md` | (file) | mtime 2026-09-01 22:47 UTC; size 3192 B; git-added 6f789d8 2026-08-31 22:39:15 +0000 | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-003 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | (file) | mtime 2026-09-01 22:47 UTC; size 4634 B; git-added 6f789d8 2026-08-31 22:39:15 +0000 (follow-up note b1f147e) | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-004 | `bench/results/2026-08-31-3090x2/raw.jsonl` | (file) | mtime 2026-09-01 22:47 UTC; size 33942 B; git-added 6f789d8 2026-08-31 22:39:15 +0000 | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-005 | `bench/results/2026-09-01-3090x2-sparse/env.txt` | (file) | mtime 2026-09-01 22:47 UTC; size 2194 B; git-added b1f147e 2026-09-01 20:11:16 +0000 | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-006 | `bench/results/2026-09-01-3090x2-sparse/report.md` | (file) | mtime 2026-09-01 22:47 UTC; size 1551 B; git-added b1f147e 2026-09-01 20:11:16 +0000 | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-007 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | (file) | mtime 2026-09-01 22:47 UTC; size 4375 B; git-added b1f147e 2026-09-01 20:11:16 +0000 | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-008 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | (file) | mtime 2026-09-01 22:47 UTC; size 18932 B; git-added b1f147e 2026-09-01 20:11:16 +0000 | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-009 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | (file) | mtime 2026-09-01 22:48 UTC; size 1110 B; git-added 65d9098 2026-08-31 21:46:33 +0200 (Et9797) | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-010 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | (file) | mtime 2026-09-01 22:48 UTC; size 4676 B; git-added 65d9098 2026-08-31 21:46:33 +0200 (Et9797) | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-011 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | (file) | mtime 2026-09-01 22:48 UTC; size 4749057 B; git-added 65d9098 2026-08-31 21:46:33 +0200 (Et9797) | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-012 | `applications/alphafold/results_214m/GLOSSARY.md` | (file) | mtime 2026-09-01 22:48 UTC; size 17193 B; git-added 65d9098 2026-08-31 21:46:33 +0200 (Et9797) | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-013 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | (file) | mtime 2026-09-01 22:48 UTC; size 111866 B; git-added 65d9098 2026-08-31 21:46:33 +0200 (Et9797) | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-014 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | (file) | mtime 2026-09-01 22:48 UTC; size 7790 B; git-added 65d9098 2026-08-31 21:46:33 +0200 (Et9797) | timestamp | software | filesystem mtime = git checkout time, not original creation time |
| L-015 | `bench/results/2026-08-31-3090x2/env.txt` | 2 | 5a8e59f8493622f623eae97daed02d460ec80389 | git sha | software | $ git rev-parse HEAD → 5a8e59f8493622f623eae97daed02d460ec80389 |
| L-016 | `bench/results/2026-08-31-3090x2/env.txt` | 6 | Mon Aug 31 20:52:51 2026 | timestamp | hardware-dependent | nvidia-smi banner: Mon Aug 31 20:52:51 2026 |
| L-017 | `bench/results/2026-08-31-3090x2/env.txt` | 8 | 580.159.03 | driver version | software | NVIDIA-SMI 580.159.03 Driver Version: 580.159.03 CUDA Version: 13.0 |
| L-018 | `bench/results/2026-08-31-3090x2/env.txt` | 8 | 13.0 | CUDA version | software | CUDA Version: 13.0 |
| L-019 | `bench/results/2026-08-31-3090x2/env.txt` | 14 | NVIDIA GeForce RTX 3090 (GPU 0, bus 01:00.0) | GPU model | hardware-dependent | 0  NVIDIA GeForce RTX 3090  On  00000000:01:00.0 Off |
| L-020 | `bench/results/2026-08-31-3090x2/env.txt` | 15 | 24576 | MiB VRAM | hardware-dependent | 1MiB / 24576MiB |
| L-021 | `bench/results/2026-08-31-3090x2/env.txt` | 15 | 360 | W power cap | hardware-dependent | 123W / 360W |
| L-022 | `bench/results/2026-08-31-3090x2/env.txt` | 18 | NVIDIA GeForce RTX 3090 (GPU 1, bus 82:00.0) | GPU model | hardware-dependent | 1  NVIDIA GeForce RTX 3090  On  00000000:82:00.0 Off |
| L-023 | `bench/results/2026-08-31-3090x2/env.txt` | 19 | 24576 | MiB VRAM | hardware-dependent | 1MiB / 24576MiB |
| L-024 | `bench/results/2026-08-31-3090x2/env.txt` | 32 | (no output captured) | — | software | $ /root/projects/ET-Miner/.venv/bin/python3 -m pip freeze — section is empty in file |
| L-025 | `bench/results/2026-08-31-3090x2/report.md` | 3 | 28 ok / 0 failed | runs | deterministic | Runs: 28 ok, 0 failed/timeout. |
| L-026 | `bench/results/2026-08-31-3090x2/report.md` | 9 | 10.4 | s median wall | hardware-dependent | deepk-density-auto \| deep_k \| legacy \| compact \| 2 \| 1 \| 10.4 \| 10.4 \| 416 |
| L-027 | `bench/results/2026-08-31-3090x2/report.md` | 9 | 10.4 | s min wall | hardware-dependent | deepk-density-auto: min s 10.4 |
| L-028 | `bench/results/2026-08-31-3090x2/report.md` | 9 | 416 | MB peak VRAM | hardware-dependent | deepk-density-auto: peak VRAM MB 416; throttled no |
| L-029 | `bench/results/2026-08-31-3090x2/report.md` | 9 | gpus=2; reps=1; variant=legacy; filter=compact; preset=deep_k | params | method-parameter | deepk-density-auto \| deep_k \| legacy \| compact \| 2 \| 1 |
| L-030 | `bench/results/2026-08-31-3090x2/report.md` | 10 | 0.6 | s median wall | hardware-dependent | deepk-legacy-1g \| deep_k \| legacy \| compact \| 1 \| 1 \| 0.6 \| 0.6 \| 352 |
| L-031 | `bench/results/2026-08-31-3090x2/report.md` | 10 | 0.6 | s min wall | hardware-dependent | deepk-legacy-1g: min s 0.6 |
| L-032 | `bench/results/2026-08-31-3090x2/report.md` | 10 | 352 | MB peak VRAM | hardware-dependent | deepk-legacy-1g: peak VRAM MB 352; throttled yes |
| L-033 | `bench/results/2026-08-31-3090x2/report.md` | 10 | gpus=1; reps=1; variant=legacy; filter=compact; preset=deep_k | params | method-parameter | deepk-legacy-1g \| deep_k \| legacy \| compact \| 1 \| 1 |
| L-034 | `bench/results/2026-08-31-3090x2/report.md` | 11 | 1.3 | s median wall | hardware-dependent | deepk-legacy-2g \| deep_k \| legacy \| compact \| 2 \| 1 \| 1.3 \| 1.3 \| 416 |
| L-035 | `bench/results/2026-08-31-3090x2/report.md` | 11 | 1.3 | s min wall | hardware-dependent | deepk-legacy-2g: min s 1.3 |
| L-036 | `bench/results/2026-08-31-3090x2/report.md` | 11 | 416 | MB peak VRAM | hardware-dependent | deepk-legacy-2g: peak VRAM MB 416; throttled no |
| L-037 | `bench/results/2026-08-31-3090x2/report.md` | 11 | gpus=2; reps=1; variant=legacy; filter=compact; preset=deep_k | params | method-parameter | deepk-legacy-2g \| deep_k \| legacy \| compact \| 2 \| 1 |
| L-038 | `bench/results/2026-08-31-3090x2/report.md` | 12 | 0.9 | s median wall | hardware-dependent | deepk-nonccl \| deep_k \| legacy \| compact \| 2 \| 1 \| 0.9 \| 0.9 \| 310 |
| L-039 | `bench/results/2026-08-31-3090x2/report.md` | 12 | 0.9 | s min wall | hardware-dependent | deepk-nonccl: min s 0.9 |
| L-040 | `bench/results/2026-08-31-3090x2/report.md` | 12 | 310 | MB peak VRAM | hardware-dependent | deepk-nonccl: peak VRAM MB 310; throttled no |
| L-041 | `bench/results/2026-08-31-3090x2/report.md` | 12 | gpus=2; reps=1; variant=legacy; filter=compact; preset=deep_k | params | method-parameter | deepk-nonccl \| deep_k \| legacy \| compact \| 2 \| 1 |
| L-042 | `bench/results/2026-08-31-3090x2/report.md` | 13 | 0.6 | s median wall | hardware-dependent | deepk-prefilter-off \| deep_k \| legacy \| compact \| 1 \| 1 \| 0.6 \| 0.6 \| 352 |
| L-043 | `bench/results/2026-08-31-3090x2/report.md` | 13 | 0.6 | s min wall | hardware-dependent | deepk-prefilter-off: min s 0.6 |
| L-044 | `bench/results/2026-08-31-3090x2/report.md` | 13 | 352 | MB peak VRAM | hardware-dependent | deepk-prefilter-off: peak VRAM MB 352; throttled no |
| L-045 | `bench/results/2026-08-31-3090x2/report.md` | 13 | gpus=1; reps=1; variant=legacy; filter=compact; preset=deep_k | params | method-parameter | deepk-prefilter-off \| deep_k \| legacy \| compact \| 1 \| 1 |
| L-046 | `bench/results/2026-08-31-3090x2/report.md` | 14 | 0.6 | s median wall | hardware-dependent | deepk-shared-1g \| deep_k \| shared \| compact \| 1 \| 3 \| 0.6 \| 0.6 \| 352 |
| L-047 | `bench/results/2026-08-31-3090x2/report.md` | 14 | 0.6 | s min wall | hardware-dependent | deepk-shared-1g: min s 0.6 |
| L-048 | `bench/results/2026-08-31-3090x2/report.md` | 14 | 352 | MB peak VRAM | hardware-dependent | deepk-shared-1g: peak VRAM MB 352; throttled yes |
| L-049 | `bench/results/2026-08-31-3090x2/report.md` | 14 | gpus=1; reps=3; variant=shared; filter=compact; preset=deep_k | params | method-parameter | deepk-shared-1g \| deep_k \| shared \| compact \| 1 \| 3 |
| L-050 | `bench/results/2026-08-31-3090x2/report.md` | 15 | 1.4 | s median wall | hardware-dependent | deepk-shared-2g \| deep_k \| shared \| compact \| 2 \| 3 \| 1.4 \| 1.3 \| 416 |
| L-051 | `bench/results/2026-08-31-3090x2/report.md` | 15 | 1.3 | s min wall | hardware-dependent | deepk-shared-2g: min s 1.3 |
| L-052 | `bench/results/2026-08-31-3090x2/report.md` | 15 | 416 | MB peak VRAM | hardware-dependent | deepk-shared-2g: peak VRAM MB 416; throttled no |
| L-053 | `bench/results/2026-08-31-3090x2/report.md` | 15 | gpus=2; reps=3; variant=shared; filter=compact; preset=deep_k | params | method-parameter | deepk-shared-2g \| deep_k \| shared \| compact \| 2 \| 3 |
| L-054 | `bench/results/2026-08-31-3090x2/report.md` | 16 | 0.6 | s median wall | hardware-dependent | deepk-single-prefilter-on \| deep_k \| legacy \| compact \| 1 \| 1 \| 0.6 \| 0.6 \| 352 |
| L-055 | `bench/results/2026-08-31-3090x2/report.md` | 16 | 0.6 | s min wall | hardware-dependent | deepk-single-prefilter-on: min s 0.6 |
| L-056 | `bench/results/2026-08-31-3090x2/report.md` | 16 | 352 | MB peak VRAM | hardware-dependent | deepk-single-prefilter-on: peak VRAM MB 352; throttled yes |
| L-057 | `bench/results/2026-08-31-3090x2/report.md` | 16 | gpus=1; reps=1; variant=legacy; filter=compact; preset=deep_k | params | method-parameter | deepk-single-prefilter-on \| deep_k \| legacy \| compact \| 1 \| 1 |
| L-058 | `bench/results/2026-08-31-3090x2/report.md` | 17 | 1.5 | s median wall | hardware-dependent | skew-nnz \| skewed_rows \| legacy \| compact \| 2 \| 2 \| 1.5 \| 1.5 \| 418 |
| L-059 | `bench/results/2026-08-31-3090x2/report.md` | 17 | 1.5 | s min wall | hardware-dependent | skew-nnz: min s 1.5 |
| L-060 | `bench/results/2026-08-31-3090x2/report.md` | 17 | 418 | MB peak VRAM | hardware-dependent | skew-nnz: peak VRAM MB 418; throttled no |
| L-061 | `bench/results/2026-08-31-3090x2/report.md` | 17 | gpus=2; reps=2; variant=legacy; filter=compact; preset=skewed_rows | params | method-parameter | skew-nnz \| skewed_rows \| legacy \| compact \| 2 \| 2 |
| L-062 | `bench/results/2026-08-31-3090x2/report.md` | 18 | 1.4 | s median wall | hardware-dependent | skew-rows \| skewed_rows \| legacy \| compact \| 2 \| 2 \| 1.4 \| 1.4 \| 416 |
| L-063 | `bench/results/2026-08-31-3090x2/report.md` | 18 | 1.4 | s min wall | hardware-dependent | skew-rows: min s 1.4 |
| L-064 | `bench/results/2026-08-31-3090x2/report.md` | 18 | 416 | MB peak VRAM | hardware-dependent | skew-rows: peak VRAM MB 416; throttled no |
| L-065 | `bench/results/2026-08-31-3090x2/report.md` | 18 | gpus=2; reps=2; variant=legacy; filter=compact; preset=skewed_rows | params | method-parameter | skew-rows \| skewed_rows \| legacy \| compact \| 2 \| 2 |
| L-066 | `bench/results/2026-08-31-3090x2/report.md` | 19 | 89.6 | s median wall | hardware-dependent | stressk2-filter-compact \| stress_k2 \| legacy \| compact \| 2 \| 1 \| 89.6 \| 89.6 \| 6918 |
| L-067 | `bench/results/2026-08-31-3090x2/report.md` | 19 | 89.6 | s min wall | hardware-dependent | stressk2-filter-compact: min s 89.6 |
| L-068 | `bench/results/2026-08-31-3090x2/report.md` | 19 | 6918 | MB peak VRAM | hardware-dependent | stressk2-filter-compact: peak VRAM MB 6918; throttled yes |
| L-069 | `bench/results/2026-08-31-3090x2/report.md` | 19 | gpus=2; reps=1; variant=legacy; filter=compact; preset=stress_k2 | params | method-parameter | stressk2-filter-compact \| stress_k2 \| legacy \| compact \| 2 \| 1 |
| L-070 | `bench/results/2026-08-31-3090x2/report.md` | 20 | 90.2 | s median wall | hardware-dependent | stressk2-filter-cpu \| stress_k2 \| legacy \| cpu \| 2 \| 1 \| 90.2 \| 90.2 \| 6920 |
| L-071 | `bench/results/2026-08-31-3090x2/report.md` | 20 | 90.2 | s min wall | hardware-dependent | stressk2-filter-cpu: min s 90.2 |
| L-072 | `bench/results/2026-08-31-3090x2/report.md` | 20 | 6920 | MB peak VRAM | hardware-dependent | stressk2-filter-cpu: peak VRAM MB 6920; throttled yes |
| L-073 | `bench/results/2026-08-31-3090x2/report.md` | 20 | gpus=2; reps=1; variant=legacy; filter=cpu; preset=stress_k2 | params | method-parameter | stressk2-filter-cpu \| stress_k2 \| legacy \| cpu \| 2 \| 1 |
| L-074 | `bench/results/2026-08-31-3090x2/report.md` | 21 | 89.7 | s median wall | hardware-dependent | stressk2-filter-cupy \| stress_k2 \| legacy \| cupy \| 2 \| 1 \| 89.7 \| 89.7 \| 6998 |
| L-075 | `bench/results/2026-08-31-3090x2/report.md` | 21 | 89.7 | s min wall | hardware-dependent | stressk2-filter-cupy: min s 89.7 |
| L-076 | `bench/results/2026-08-31-3090x2/report.md` | 21 | 6998 | MB peak VRAM | hardware-dependent | stressk2-filter-cupy: peak VRAM MB 6998; throttled yes |
| L-077 | `bench/results/2026-08-31-3090x2/report.md` | 21 | gpus=2; reps=1; variant=legacy; filter=cupy; preset=stress_k2 | params | method-parameter | stressk2-filter-cupy \| stress_k2 \| legacy \| cupy \| 2 \| 1 |
| L-078 | `bench/results/2026-08-31-3090x2/report.md` | 22 | 2238.8 | s median wall | hardware-dependent | stressk2-legacy-1g \| stress_k2 \| legacy \| compact \| 1 \| 1 \| 2238.8 \| 2238.8 \| 17340 |
| L-079 | `bench/results/2026-08-31-3090x2/report.md` | 22 | 2238.8 | s min wall | hardware-dependent | stressk2-legacy-1g: min s 2238.8 |
| L-080 | `bench/results/2026-08-31-3090x2/report.md` | 22 | 17340 | MB peak VRAM | hardware-dependent | stressk2-legacy-1g: peak VRAM MB 17340; throttled yes |
| L-081 | `bench/results/2026-08-31-3090x2/report.md` | 22 | gpus=1; reps=1; variant=legacy; filter=compact; preset=stress_k2 | params | method-parameter | stressk2-legacy-1g \| stress_k2 \| legacy \| compact \| 1 \| 1 |
| L-082 | `bench/results/2026-08-31-3090x2/report.md` | 23 | 1996.7 | s median wall | hardware-dependent | stressk2-legacy-2g \| stress_k2 \| legacy \| compact \| 2 \| 1 \| 1996.7 \| 1996.7 \| 16576 |
| L-083 | `bench/results/2026-08-31-3090x2/report.md` | 23 | 1996.7 | s min wall | hardware-dependent | stressk2-legacy-2g: min s 1996.7 |
| L-084 | `bench/results/2026-08-31-3090x2/report.md` | 23 | 16576 | MB peak VRAM | hardware-dependent | stressk2-legacy-2g: peak VRAM MB 16576; throttled yes |
| L-085 | `bench/results/2026-08-31-3090x2/report.md` | 23 | gpus=2; reps=1; variant=legacy; filter=compact; preset=stress_k2 | params | method-parameter | stressk2-legacy-2g \| stress_k2 \| legacy \| compact \| 2 \| 1 |
| L-086 | `bench/results/2026-08-31-3090x2/report.md` | 24 | 253.3 | s median wall | hardware-dependent | stressk2-shared-1g \| stress_k2 \| shared \| compact \| 1 \| 3 \| 253.3 \| 253.1 \| 17340 |
| L-087 | `bench/results/2026-08-31-3090x2/report.md` | 24 | 253.1 | s min wall | hardware-dependent | stressk2-shared-1g: min s 253.1 |
| L-088 | `bench/results/2026-08-31-3090x2/report.md` | 24 | 17340 | MB peak VRAM | hardware-dependent | stressk2-shared-1g: peak VRAM MB 17340; throttled yes |
| L-089 | `bench/results/2026-08-31-3090x2/report.md` | 24 | gpus=1; reps=3; variant=shared; filter=compact; preset=stress_k2 | params | method-parameter | stressk2-shared-1g \| stress_k2 \| shared \| compact \| 1 \| 3 |
| L-090 | `bench/results/2026-08-31-3090x2/report.md` | 25 | 122.3 | s median wall | hardware-dependent | stressk2-shared-2g \| stress_k2 \| shared \| compact \| 2 \| 3 \| 122.3 \| 122.3 \| 16576 |
| L-091 | `bench/results/2026-08-31-3090x2/report.md` | 25 | 122.3 | s min wall | hardware-dependent | stressk2-shared-2g: min s 122.3 |
| L-092 | `bench/results/2026-08-31-3090x2/report.md` | 25 | 16576 | MB peak VRAM | hardware-dependent | stressk2-shared-2g: peak VRAM MB 16576; throttled yes |
| L-093 | `bench/results/2026-08-31-3090x2/report.md` | 25 | gpus=2; reps=3; variant=shared; filter=compact; preset=stress_k2 | params | method-parameter | stressk2-shared-2g \| stress_k2 \| shared \| compact \| 2 \| 3 |
| L-094 | `bench/results/2026-08-31-3090x2/report.md` | 26 | 1.6 | s median wall | hardware-dependent | twophase-smoke \| smoke \| legacy \| compact \| 2 \| 1 \| 1.6 \| 1.6 \| 408 |
| L-095 | `bench/results/2026-08-31-3090x2/report.md` | 26 | 1.6 | s min wall | hardware-dependent | twophase-smoke: min s 1.6 |
| L-096 | `bench/results/2026-08-31-3090x2/report.md` | 26 | 408 | MB peak VRAM | hardware-dependent | twophase-smoke: peak VRAM MB 408; throttled yes |
| L-097 | `bench/results/2026-08-31-3090x2/report.md` | 26 | gpus=2; reps=1; variant=legacy; filter=compact; preset=smoke | params | method-parameter | twophase-smoke \| smoke \| legacy \| compact \| 2 \| 1 |
| L-098 | `bench/results/2026-08-31-3090x2/report.md` | 32 | 0.97× | speedup shared vs legacy (deep_k, 2 GPUs) | hardware-dependent | deep_k \| 2 \| 1.3 \| 1.4 \| 0.97× |
| L-099 | `bench/results/2026-08-31-3090x2/report.md` | 33 | 16.32× | speedup shared vs legacy (stress_k2, 2 GPUs) | hardware-dependent | stress_k2 \| 2 \| 1996.7 \| 122.3 \| 16.32× |
| L-100 | `bench/results/2026-08-31-3090x2/report.md` | 37 | 10.431 | s (deepk-density-auto#r0) | hardware-dependent | **deep_k** (deepk-density-auto#r0, 10.431s): |
| L-101 | `bench/results/2026-08-31-3090x2/report.md` | 41 | K1: candidates=112; frequent=112 | count | deterministic | deep_k per-level: \| 1 \| 112 \| 112 \| 27 \| |
| L-102 | `bench/results/2026-08-31-3090x2/report.md` | 41 | 27 | ms (K=1, deepk-density-auto#r0) | hardware-dependent | \| 1 \| 112 \| 112 \| 27 \| |
| L-103 | `bench/results/2026-08-31-3090x2/report.md` | 42 | K2: candidates=0; frequent=471 | count | deterministic | deep_k per-level: \| 2 \| 0 \| 471 \| 46 \| |
| L-104 | `bench/results/2026-08-31-3090x2/report.md` | 42 | 46 | ms (K=2, deepk-density-auto#r0) | hardware-dependent | \| 2 \| 0 \| 471 \| 46 \| |
| L-105 | `bench/results/2026-08-31-3090x2/report.md` | 43 | K3: candidates=0; frequent=901 | count | deterministic | deep_k per-level: \| 3 \| 0 \| 901 \| 5 \| |
| L-106 | `bench/results/2026-08-31-3090x2/report.md` | 43 | 5 | ms (K=3, deepk-density-auto#r0) | hardware-dependent | \| 3 \| 0 \| 901 \| 5 \| |
| L-107 | `bench/results/2026-08-31-3090x2/report.md` | 44 | K4: candidates=0; frequent=1407 | count | deterministic | deep_k per-level: \| 4 \| 0 \| 1,407 \| 6 \| |
| L-108 | `bench/results/2026-08-31-3090x2/report.md` | 44 | 6 | ms (K=4, deepk-density-auto#r0) | hardware-dependent | \| 4 \| 0 \| 1,407 \| 6 \| |
| L-109 | `bench/results/2026-08-31-3090x2/report.md` | 45 | K5: candidates=0; frequent=1854 | count | deterministic | deep_k per-level: \| 5 \| 0 \| 1,854 \| 3145 \| |
| L-110 | `bench/results/2026-08-31-3090x2/report.md` | 45 | 3145 | ms (K=5, deepk-density-auto#r0) | hardware-dependent | \| 5 \| 0 \| 1,854 \| 3145 \| |
| L-111 | `bench/results/2026-08-31-3090x2/report.md` | 46 | K6: candidates=0; frequent=1848 | count | deterministic | deep_k per-level: \| 6 \| 0 \| 1,848 \| 2553 \| |
| L-112 | `bench/results/2026-08-31-3090x2/report.md` | 46 | 2553 | ms (K=6, deepk-density-auto#r0) | hardware-dependent | \| 6 \| 0 \| 1,848 \| 2553 \| |
| L-113 | `bench/results/2026-08-31-3090x2/report.md` | 47 | K7: candidates=0; frequent=1320 | count | deterministic | deep_k per-level: \| 7 \| 0 \| 1,320 \| 1817 \| |
| L-114 | `bench/results/2026-08-31-3090x2/report.md` | 47 | 1817 | ms (K=7, deepk-density-auto#r0) | hardware-dependent | \| 7 \| 0 \| 1,320 \| 1817 \| |
| L-115 | `bench/results/2026-08-31-3090x2/report.md` | 48 | K8: candidates=0; frequent=660 | count | deterministic | deep_k per-level: \| 8 \| 0 \| 660 \| 902 \| |
| L-116 | `bench/results/2026-08-31-3090x2/report.md` | 48 | 902 | ms (K=8, deepk-density-auto#r0) | hardware-dependent | \| 8 \| 0 \| 660 \| 902 \| |
| L-117 | `bench/results/2026-08-31-3090x2/report.md` | 49 | K9: candidates=0; frequent=220 | count | deterministic | deep_k per-level: \| 9 \| 0 \| 220 \| 374 \| |
| L-118 | `bench/results/2026-08-31-3090x2/report.md` | 49 | 374 | ms (K=9, deepk-density-auto#r0) | hardware-dependent | \| 9 \| 0 \| 220 \| 374 \| |
| L-119 | `bench/results/2026-08-31-3090x2/report.md` | 50 | K10: candidates=0; frequent=44 | count | deterministic | deep_k per-level: \| 10 \| 0 \| 44 \| 65 \| |
| L-120 | `bench/results/2026-08-31-3090x2/report.md` | 50 | 65 | ms (K=10, deepk-density-auto#r0) | hardware-dependent | \| 10 \| 0 \| 44 \| 65 \| |
| L-121 | `bench/results/2026-08-31-3090x2/report.md` | 51 | K11: candidates=0; frequent=4 | count | deterministic | deep_k per-level: \| 11 \| 0 \| 4 \| 14 \| |
| L-122 | `bench/results/2026-08-31-3090x2/report.md` | 51 | 14 | ms (K=11, deepk-density-auto#r0) | hardware-dependent | \| 11 \| 0 \| 4 \| 14 \| |
| L-123 | `bench/results/2026-08-31-3090x2/report.md` | 53 | 1.517 | s (skew-nnz#r1) | hardware-dependent | **skewed_rows** (skew-nnz#r1, 1.517s): |
| L-124 | `bench/results/2026-08-31-3090x2/report.md` | 57 | K1: candidates=118; frequent=118 | count | deterministic | skewed_rows per-level: \| 1 \| 118 \| 118 \| 26 \| |
| L-125 | `bench/results/2026-08-31-3090x2/report.md` | 57 | 26 | ms (K=1, skew-nnz#r1) | hardware-dependent | \| 1 \| 118 \| 118 \| 26 \| |
| L-126 | `bench/results/2026-08-31-3090x2/report.md` | 58 | K2: candidates=0; frequent=717 | count | deterministic | skewed_rows per-level: \| 2 \| 0 \| 717 \| 47 \| |
| L-127 | `bench/results/2026-08-31-3090x2/report.md` | 58 | 47 | ms (K=2, skew-nnz#r1) | hardware-dependent | \| 2 \| 0 \| 717 \| 47 \| |
| L-128 | `bench/results/2026-08-31-3090x2/report.md` | 59 | K3: candidates=0; frequent=1928 | count | deterministic | skewed_rows per-level: \| 3 \| 0 \| 1,928 \| 7 \| |
| L-129 | `bench/results/2026-08-31-3090x2/report.md` | 59 | 7 | ms (K=3, skew-nnz#r1) | hardware-dependent | \| 3 \| 0 \| 1,928 \| 7 \| |
| L-130 | `bench/results/2026-08-31-3090x2/report.md` | 60 | K4: candidates=0; frequent=2932 | count | deterministic | skewed_rows per-level: \| 4 \| 0 \| 2,932 \| 16 \| |
| L-131 | `bench/results/2026-08-31-3090x2/report.md` | 60 | 16 | ms (K=4, skew-nnz#r1) | hardware-dependent | \| 4 \| 0 \| 2,932 \| 16 \| |
| L-132 | `bench/results/2026-08-31-3090x2/report.md` | 61 | K5: candidates=0; frequent=2683 | count | deterministic | skewed_rows per-level: \| 5 \| 0 \| 2,683 \| 34 \| |
| L-133 | `bench/results/2026-08-31-3090x2/report.md` | 61 | 34 | ms (K=5, skew-nnz#r1) | hardware-dependent | \| 5 \| 0 \| 2,683 \| 34 \| |
| L-134 | `bench/results/2026-08-31-3090x2/report.md` | 62 | K6: candidates=0; frequent=1458 | count | deterministic | skewed_rows per-level: \| 6 \| 0 \| 1,458 \| 40 \| |
| L-135 | `bench/results/2026-08-31-3090x2/report.md` | 62 | 40 | ms (K=6, skew-nnz#r1) | hardware-dependent | \| 6 \| 0 \| 1,458 \| 40 \| |
| L-136 | `bench/results/2026-08-31-3090x2/report.md` | 63 | K7: candidates=0; frequent=443 | count | deterministic | skewed_rows per-level: \| 7 \| 0 \| 443 \| 22 \| |
| L-137 | `bench/results/2026-08-31-3090x2/report.md` | 63 | 22 | ms (K=7, skew-nnz#r1) | hardware-dependent | \| 7 \| 0 \| 443 \| 22 \| |
| L-138 | `bench/results/2026-08-31-3090x2/report.md` | 64 | K8: candidates=0; frequent=67 | count | deterministic | skewed_rows per-level: \| 8 \| 0 \| 67 \| 5 \| |
| L-139 | `bench/results/2026-08-31-3090x2/report.md` | 64 | 5 | ms (K=8, skew-nnz#r1) | hardware-dependent | \| 8 \| 0 \| 67 \| 5 \| |
| L-140 | `bench/results/2026-08-31-3090x2/report.md` | 65 | K9: candidates=0; frequent=4 | count | deterministic | skewed_rows per-level: \| 9 \| 0 \| 4 \| 2 \| |
| L-141 | `bench/results/2026-08-31-3090x2/report.md` | 65 | 2 | ms (K=9, skew-nnz#r1) | hardware-dependent | \| 9 \| 0 \| 4 \| 2 \| |
| L-142 | `bench/results/2026-08-31-3090x2/report.md` | 67 | 1.582 | s (twophase-smoke#r0) | hardware-dependent | **smoke** (twophase-smoke#r0, 1.582s): |
| L-143 | `bench/results/2026-08-31-3090x2/report.md` | 71 | K1 (phase 1): candidates=118; frequent=118 | count | deterministic | smoke two-phase per-level: \| 1 \| 118 \| 118 \| 252 \| |
| L-144 | `bench/results/2026-08-31-3090x2/report.md` | 71 | 252 | ms (K=1, phase 1, twophase-smoke#r0) | hardware-dependent | \| 1 \| 118 \| 118 \| 252 \| |
| L-145 | `bench/results/2026-08-31-3090x2/report.md` | 72 | K2 (phase 1): candidates=0; frequent=290 | count | deterministic | smoke two-phase per-level: \| 2 \| 0 \| 290 \| 48 \| |
| L-146 | `bench/results/2026-08-31-3090x2/report.md` | 72 | 48 | ms (K=2, phase 1, twophase-smoke#r0) | hardware-dependent | \| 2 \| 0 \| 290 \| 48 \| |
| L-147 | `bench/results/2026-08-31-3090x2/report.md` | 73 | K3 (phase 1): candidates=0; frequent=202 | count | deterministic | smoke two-phase per-level: \| 3 \| 0 \| 202 \| 120 \| |
| L-148 | `bench/results/2026-08-31-3090x2/report.md` | 73 | 120 | ms (K=3, phase 1, twophase-smoke#r0) | hardware-dependent | \| 3 \| 0 \| 202 \| 120 \| |
| L-149 | `bench/results/2026-08-31-3090x2/report.md` | 74 | K4 (phase 1): candidates=0; frequent=22 | count | deterministic | smoke two-phase per-level: \| 4 \| 0 \| 22 \| 7 \| |
| L-150 | `bench/results/2026-08-31-3090x2/report.md` | 74 | 7 | ms (K=4, phase 1, twophase-smoke#r0) | hardware-dependent | \| 4 \| 0 \| 22 \| 7 \| |
| L-151 | `bench/results/2026-08-31-3090x2/report.md` | 75 | K5 (phase 1): candidates=0; frequent=0 | count | deterministic | smoke two-phase per-level: \| 5 \| 0 \| 0 \| 0 \| |
| L-152 | `bench/results/2026-08-31-3090x2/report.md` | 75 | 0 | ms (K=5, phase 1, twophase-smoke#r0) | hardware-dependent | \| 5 \| 0 \| 0 \| 0 \| |
| L-153 | `bench/results/2026-08-31-3090x2/report.md` | 76 | K1 (phase 2): candidates=118; frequent=118 | count | deterministic | smoke two-phase per-level: \| 1 \| 118 \| 118 \| 2 \| |
| L-154 | `bench/results/2026-08-31-3090x2/report.md` | 76 | 2 | ms (K=1, phase 2, twophase-smoke#r0) | hardware-dependent | \| 1 \| 118 \| 118 \| 2 \| |
| L-155 | `bench/results/2026-08-31-3090x2/report.md` | 77 | K2 (phase 2): candidates=0; frequent=290 | count | deterministic | smoke two-phase per-level: \| 2 \| 0 \| 290 \| 34 \| |
| L-156 | `bench/results/2026-08-31-3090x2/report.md` | 77 | 34 | ms (K=2, phase 2, twophase-smoke#r0) | hardware-dependent | \| 2 \| 0 \| 290 \| 34 \| |
| L-157 | `bench/results/2026-08-31-3090x2/report.md` | 78 | K3 (phase 2): candidates=0; frequent=202 | count | deterministic | smoke two-phase per-level: \| 3 \| 0 \| 202 \| 87 \| |
| L-158 | `bench/results/2026-08-31-3090x2/report.md` | 78 | 87 | ms (K=3, phase 2, twophase-smoke#r0) | hardware-dependent | \| 3 \| 0 \| 202 \| 87 \| |
| L-159 | `bench/results/2026-08-31-3090x2/report.md` | 79 | K4 (phase 2): candidates=0; frequent=22 | count | deterministic | smoke two-phase per-level: \| 4 \| 0 \| 22 \| 5 \| |
| L-160 | `bench/results/2026-08-31-3090x2/report.md` | 79 | 5 | ms (K=4, phase 2, twophase-smoke#r0) | hardware-dependent | \| 4 \| 0 \| 22 \| 5 \| |
| L-161 | `bench/results/2026-08-31-3090x2/report.md` | 80 | K5 (phase 2): candidates=0; frequent=0 | count | deterministic | smoke two-phase per-level: \| 5 \| 0 \| 0 \| 0 \| |
| L-162 | `bench/results/2026-08-31-3090x2/report.md` | 80 | 0 | ms (K=5, phase 2, twophase-smoke#r0) | hardware-dependent | \| 5 \| 0 \| 0 \| 0 \| |
| L-163 | `bench/results/2026-08-31-3090x2/report.md` | 82 | 2238.846 | s (stressk2-legacy-1g#r0) | hardware-dependent | **stress_k2** (stressk2-legacy-1g#r0, 2238.846s): |
| L-164 | `bench/results/2026-08-31-3090x2/report.md` | 86 | K1: candidates=35,000; frequent=35,000 | count | deterministic | stress_k2 per-level: \| 1 \| 35,000 \| 35,000 \| 72 \| |
| L-165 | `bench/results/2026-08-31-3090x2/report.md` | 86 | 72 | ms (K=1, stressk2-legacy-1g#r0) | hardware-dependent | \| 1 \| 35,000 \| 35,000 \| 72 \| |
| L-166 | `bench/results/2026-08-31-3090x2/report.md` | 87 | K2: candidates=612,482,500; frequent=1,660,332 | count | deterministic | stress_k2 per-level: \| 2 \| 612,482,500 \| 1,660,332 \| 175580 \| |
| L-167 | `bench/results/2026-08-31-3090x2/report.md` | 87 | 175580 | ms (K=2, stressk2-legacy-1g#r0) | hardware-dependent | \| 2 \| 612,482,500 \| 1,660,332 \| 175580 \| |
| L-168 | `bench/results/2026-08-31-3090x2/report.md` | 88 | K3: candidates=1,301,153; frequent=1,301,153 | count | deterministic | stress_k2 per-level: \| 3 \| 1,301,153 \| 1,301,153 \| 2046663 \| |
| L-169 | `bench/results/2026-08-31-3090x2/report.md` | 88 | 2046663 | ms (K=3, stressk2-legacy-1g#r0) | hardware-dependent | \| 3 \| 1,301,153 \| 1,301,153 \| 2046663 \| |
| L-170 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 1 | 2×RTX 3090; 2026-08-31 | hardware/date | hardware-dependent | # GPU campaign findings — 2×RTX 3090, 2026-08-31 |
| L-171 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 4 | 2× RTX 3090 24 GB (sm_86) | GPU | hardware-dependent | Box: 2× RTX 3090 24 GB (sm_86, driver 580.159.03, CuPy 14.1.1, no P2P — NCCL over SHM), vast.ai. |
| L-172 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 5 | 580.159.03 / CuPy 14.1.1 | driver / library | software | driver 580.159.03, CuPy 14.1.1, no P2P — NCCL over SHM |
| L-173 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 10 | 7/7 | tier-equivalence legs | deterministic | **Tier-equivalence chain: 7/7.** Tier 1 Polars == Tier 2 Rust == single-GPU == multi-GPU legacy == shared multi-GPU == efficient-apriori |
| L-174 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 13 | 114 passed / 0 failed | tests | software | **Full GPU test suite: 114 passed / 0 failed** — first-ever on-device run |
| L-175 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 22 | 8.8× / 16.3× | speedup | hardware-dependent | ## Finding 1 — shared/tiled kernel: 8.8× / 16.3× on the stress workload |
| L-176 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 24 | 2,000,000 transactions × 35,000 items; min_count 30; K=3 | preset params | method-parameter | `stress_k2`: 2M transactions × 35K items, min_count 30, mined to K=3 — |
| L-177 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 25 | 612M K=2 candidates; ~76 billion K=3 candidates | count | deterministic | 612M K=2 candidates plus **~76 billion** K=3 candidates, every one exactly counted (no sampling) |
| L-178 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 30 | 2238.8 s legacy → 253.3 s shared (±1 s over 3 reps); 8.8× | s / speedup (1× 3090 fused) | hardware-dependent | \| 1× 3090 (fused) \| 2238.8 s \| 253.3 s (±1 s over 3 reps) \| **8.8×** \| |
| L-179 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 31 | 1996.7 s legacy → 122.3 s shared; 16.3× | s / speedup (2× 3090 row-split) | hardware-dependent | \| 2× 3090 (dense row-split) \| 1996.7 s \| 122.3 s \| **16.3×** \| |
| L-180 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 33 | ≈300M counts/s (1 GPU); ≈630M counts/s (2 GPUs) | support counts per second | hardware-dependent | ≈300M exact support counts/s on one 3090, ≈630M/s on two. |
| L-181 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 36 | 0.97–1.07× | speedup (deep_k/smoke, shared vs legacy) | hardware-dependent | On small/deep data (`deep_k`, `smoke`) shared is neutral (0.97–1.07×) |
| L-182 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 37 | 64 | pairs (sub-64-pair groups routed to legacy kernel) | method-parameter | sub-64-pair groups route to the legacy kernel by design |
| L-183 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 43 | 9,285 fewer K=3 itemsets (−0.7%) | count | deterministic | mined **9,285 fewer K=3 itemsets (−0.7% of that level)** than the seven exact configs |
| L-184 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 45 | 0.7×min_count | prefilter reject threshold | method-parameter | rejects candidates whose sampled estimate is < 0.7×min_count *without exact recount* |
| L-185 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 57 | 10.4 s vs 1.3 s | s (density-auto vs dense) | hardware-dependent | took 10.4 s vs 1.3 s dense: post-transition levels (K=5–9) spend seconds |
| L-186 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 60 | n/32 | crossover (mean support) | method-parameter | The n/32 crossover optimizes memory, not yet time, at this scale. |
| L-187 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 66 | ~2.5 s per level | s (host tidset rebuild) | hardware-dependent | the host tidset rebuild, not the pair loop, cost ~2.5 s per level |
| L-188 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 67 | 1.34 s | s (deepk-density-auto, 2026-09-01 rerun) | hardware-dependent | `deepk-density-auto` runs in 1.34 s (== dense) with the identical signature |
| L-189 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 73 | 89.6 / 89.7 / 90.2 s | s (filter compact/cupy/cpu) | hardware-dependent | **Filter impls** (compact/cupy/cpu) indistinguishable at this scale (89.6/89.7/90.2 s) |
| L-190 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 74 | 2.4 GB | full-array D2H (cpu filter) | hardware-dependent | the cpu impl's full-array D2H is only 2.4 GB here |
| L-191 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 75 | 80 GB | D2H scale (not reachable on box) | method-parameter | structurally required at the 80 GB-D2H scale this box cannot reach |
| L-192 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 76 | 0.9 s vs 1.3 s | s (NCCL fallback vs NCCL) | hardware-dependent | **NCCL fallback** (staged D2D): works, 0.9 s vs 1.3 s with NCCL on tiny data |
| L-193 | `bench/results/2026-08-31-3090x2/FINDINGS.md` | 79 | 1.5 vs 1.4 s | s (nnz vs rows balance) | hardware-dependent | `nnz` shows no win over `rows` on the clustered preset (1.5 vs 1.4 s) |
| L-194 | `bench/results/2026-09-01-3090x2-sparse/env.txt` | 2 | b8a5032b3e2c492dbc9e185ec24f0aa6b0206b4f | git sha | software | $ git rev-parse HEAD → b8a5032b3e2c492dbc9e185ec24f0aa6b0206b4f |
| L-195 | `bench/results/2026-09-01-3090x2-sparse/env.txt` | 6 | Tue Sep  1 20:09:15 2026 | timestamp | hardware-dependent | nvidia-smi banner: Tue Sep  1 20:09:15 2026 |
| L-196 | `bench/results/2026-09-01-3090x2-sparse/env.txt` | 8 | 580.159.03 / CUDA 13.0 | driver / CUDA | software | NVIDIA-SMI 580.159.03 Driver Version: 580.159.03 CUDA Version: 13.0 |
| L-197 | `bench/results/2026-09-01-3090x2-sparse/env.txt` | 14 | NVIDIA GeForce RTX 3090 (GPU 0) | GPU model | hardware-dependent | 0  NVIDIA GeForce RTX 3090  On  00000000:01:00.0 Off |
| L-198 | `bench/results/2026-09-01-3090x2-sparse/env.txt` | 15 | 24576 | MiB VRAM | hardware-dependent | 1MiB / 24576MiB (49W / 360W, 46C, P5) |
| L-199 | `bench/results/2026-09-01-3090x2-sparse/env.txt` | 18 | NVIDIA GeForce RTX 3090 (GPU 1) | GPU model | hardware-dependent | 1  NVIDIA GeForce RTX 3090  On  00000000:82:00.0 Off |
| L-200 | `bench/results/2026-09-01-3090x2-sparse/env.txt` | 32 | (no output captured) | — | software | pip freeze section empty (file otherwise identical to 2026-08-31 env.txt) |
| L-201 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 3 | 13 ok / 0 failed | runs | deterministic | Runs: 13 ok, 0 failed/timeout. |
| L-202 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 9 | 1.3 | s median wall | hardware-dependent | deepk-density-auto \| deep_k \| legacy \| compact \| 2 \| 1 \| 1.3 \| 1.3 \| 408 |
| L-203 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 9 | 1.3 | s min wall | hardware-dependent | deepk-density-auto: min s 1.3 |
| L-204 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 9 | 408 | MB peak VRAM | hardware-dependent | deepk-density-auto: peak VRAM MB 408; throttled no |
| L-205 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 9 | gpus=2; reps=1; variant=legacy; preset=deep_k | params | method-parameter | deepk-density-auto \| deep_k \| legacy \| compact \| 2 \| 1 |
| L-206 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 10 | 0.7 | s median wall | hardware-dependent | deepk-density-auto-1g \| deep_k \| legacy \| compact \| 1 \| 1 \| 0.7 \| 0.7 \| 680 |
| L-207 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 10 | 0.7 | s min wall | hardware-dependent | deepk-density-auto-1g: min s 0.7 |
| L-208 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 10 | 680 | MB peak VRAM | hardware-dependent | deepk-density-auto-1g: peak VRAM MB 680; throttled no |
| L-209 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 10 | gpus=1; reps=1; variant=legacy; preset=deep_k | params | method-parameter | deepk-density-auto-1g \| deep_k \| legacy \| compact \| 1 \| 1 |
| L-210 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 11 | 0.6 | s median wall | hardware-dependent | deepk-legacy-1g \| deep_k \| legacy \| compact \| 1 \| 1 \| 0.6 \| 0.6 \| 352 |
| L-211 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 11 | 0.6 | s min wall | hardware-dependent | deepk-legacy-1g: min s 0.6 |
| L-212 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 11 | 352 | MB peak VRAM | hardware-dependent | deepk-legacy-1g: peak VRAM MB 352; throttled yes |
| L-213 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 11 | gpus=1; reps=1; variant=legacy; preset=deep_k | params | method-parameter | deepk-legacy-1g \| deep_k \| legacy \| compact \| 1 \| 1 |
| L-214 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 12 | 1.3 | s median wall | hardware-dependent | deepk-legacy-2g \| deep_k \| legacy \| compact \| 2 \| 1 \| 1.3 \| 1.3 \| 416 |
| L-215 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 12 | 1.3 | s min wall | hardware-dependent | deepk-legacy-2g: min s 1.3 |
| L-216 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 12 | 416 | MB peak VRAM | hardware-dependent | deepk-legacy-2g: peak VRAM MB 416; throttled yes |
| L-217 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 12 | gpus=2; reps=1; variant=legacy; preset=deep_k | params | method-parameter | deepk-legacy-2g \| deep_k \| legacy \| compact \| 2 \| 1 |
| L-218 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 13 | 0.9 | s median wall | hardware-dependent | deepk-nonccl \| deep_k \| legacy \| compact \| 2 \| 1 \| 0.9 \| 0.9 \| 310 |
| L-219 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 13 | 0.9 | s min wall | hardware-dependent | deepk-nonccl: min s 0.9 |
| L-220 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 13 | 310 | MB peak VRAM | hardware-dependent | deepk-nonccl: peak VRAM MB 310; throttled yes |
| L-221 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 13 | gpus=2; reps=1; variant=legacy; preset=deep_k | params | method-parameter | deepk-nonccl \| deep_k \| legacy \| compact \| 2 \| 1 |
| L-222 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 14 | 0.6 | s median wall | hardware-dependent | deepk-prefilter-off \| deep_k \| legacy \| compact \| 1 \| 1 \| 0.6 \| 0.6 \| 352 |
| L-223 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 14 | 0.6 | s min wall | hardware-dependent | deepk-prefilter-off: min s 0.6 |
| L-224 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 14 | 352 | MB peak VRAM | hardware-dependent | deepk-prefilter-off: peak VRAM MB 352; throttled yes |
| L-225 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 14 | gpus=1; reps=1; variant=legacy; preset=deep_k | params | method-parameter | deepk-prefilter-off \| deep_k \| legacy \| compact \| 1 \| 1 |
| L-226 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 15 | 0.6 | s median wall | hardware-dependent | deepk-shared-1g \| deep_k \| shared \| compact \| 1 \| 3 \| 0.6 \| 0.6 \| 352 |
| L-227 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 15 | 0.6 | s min wall | hardware-dependent | deepk-shared-1g: min s 0.6 |
| L-228 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 15 | 352 | MB peak VRAM | hardware-dependent | deepk-shared-1g: peak VRAM MB 352; throttled yes |
| L-229 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 15 | gpus=1; reps=3; variant=shared; preset=deep_k | params | method-parameter | deepk-shared-1g \| deep_k \| shared \| compact \| 1 \| 3 |
| L-230 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 16 | 1.4 | s median wall | hardware-dependent | deepk-shared-2g \| deep_k \| shared \| compact \| 2 \| 3 \| 1.4 \| 1.3 \| 416 |
| L-231 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 16 | 1.3 | s min wall | hardware-dependent | deepk-shared-2g: min s 1.3 |
| L-232 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 16 | 416 | MB peak VRAM | hardware-dependent | deepk-shared-2g: peak VRAM MB 416; throttled yes |
| L-233 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 16 | gpus=2; reps=3; variant=shared; preset=deep_k | params | method-parameter | deepk-shared-2g \| deep_k \| shared \| compact \| 2 \| 3 |
| L-234 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 17 | 0.6 | s median wall | hardware-dependent | deepk-single-prefilter-on \| deep_k \| legacy \| compact \| 1 \| 1 \| 0.6 \| 0.6 \| 352 |
| L-235 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 17 | 0.6 | s min wall | hardware-dependent | deepk-single-prefilter-on: min s 0.6 |
| L-236 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 17 | 352 | MB peak VRAM | hardware-dependent | deepk-single-prefilter-on: peak VRAM MB 352; throttled yes |
| L-237 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 17 | gpus=1; reps=1; variant=legacy; preset=deep_k | params | method-parameter | deepk-single-prefilter-on \| deep_k \| legacy \| compact \| 1 \| 1 |
| L-238 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 23 | 0.99× | speedup shared vs legacy (deep_k, 2 GPUs) | hardware-dependent | \| deep_k \| 2 \| 1.3 \| 1.4 \| 0.99× \| |
| L-239 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 27 | 1.382 | s (deepk-shared-2g#r2) | hardware-dependent | **deep_k** (deepk-shared-2g#r2, 1.382s): |
| L-240 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 31 | K1: candidates=112; frequent=112 | count | deterministic | deep_k per-level: \| 1 \| 112 \| 112 \| 25 \| |
| L-241 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 31 | 25 | ms (K=1, deepk-shared-2g#r2) | hardware-dependent | \| 1 \| 112 \| 112 \| 25 \| |
| L-242 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 32 | K2: candidates=0; frequent=471 | count | deterministic | deep_k per-level: \| 2 \| 0 \| 471 \| 50 \| |
| L-243 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 32 | 50 | ms (K=2, deepk-shared-2g#r2) | hardware-dependent | \| 2 \| 0 \| 471 \| 50 \| |
| L-244 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 33 | K3: candidates=0; frequent=901 | count | deterministic | deep_k per-level: \| 3 \| 0 \| 901 \| 7 \| |
| L-245 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 33 | 7 | ms (K=3, deepk-shared-2g#r2) | hardware-dependent | \| 3 \| 0 \| 901 \| 7 \| |
| L-246 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 34 | K4: candidates=0; frequent=1407 | count | deterministic | deep_k per-level: \| 4 \| 0 \| 1,407 \| 9 \| |
| L-247 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 34 | 9 | ms (K=4, deepk-shared-2g#r2) | hardware-dependent | \| 4 \| 0 \| 1,407 \| 9 \| |
| L-248 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 35 | K5: candidates=0; frequent=1854 | count | deterministic | deep_k per-level: \| 5 \| 0 \| 1,854 \| 6 \| |
| L-249 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 35 | 6 | ms (K=5, deepk-shared-2g#r2) | hardware-dependent | \| 5 \| 0 \| 1,854 \| 6 \| |
| L-250 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 36 | K6: candidates=0; frequent=1848 | count | deterministic | deep_k per-level: \| 6 \| 0 \| 1,848 \| 3 \| |
| L-251 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 36 | 3 | ms (K=6, deepk-shared-2g#r2) | hardware-dependent | \| 6 \| 0 \| 1,848 \| 3 \| |
| L-252 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 37 | K7: candidates=0; frequent=1320 | count | deterministic | deep_k per-level: \| 7 \| 0 \| 1,320 \| 3 \| |
| L-253 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 37 | 3 | ms (K=7, deepk-shared-2g#r2) | hardware-dependent | \| 7 \| 0 \| 1,320 \| 3 \| |
| L-254 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 38 | K8: candidates=0; frequent=660 | count | deterministic | deep_k per-level: \| 8 \| 0 \| 660 \| 2 \| |
| L-255 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 38 | 2 | ms (K=8, deepk-shared-2g#r2) | hardware-dependent | \| 8 \| 0 \| 660 \| 2 \| |
| L-256 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 39 | K9: candidates=0; frequent=220 | count | deterministic | deep_k per-level: \| 9 \| 0 \| 220 \| 2 \| |
| L-257 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 39 | 2 | ms (K=9, deepk-shared-2g#r2) | hardware-dependent | \| 9 \| 0 \| 220 \| 2 \| |
| L-258 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 40 | K10: candidates=0; frequent=44 | count | deterministic | deep_k per-level: \| 10 \| 0 \| 44 \| 2 \| |
| L-259 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 40 | 2 | ms (K=10, deepk-shared-2g#r2) | hardware-dependent | \| 10 \| 0 \| 44 \| 2 \| |
| L-260 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 41 | K11: candidates=0; frequent=4 | count | deterministic | deep_k per-level: \| 11 \| 0 \| 4 \| 2 \| |
| L-261 | `bench/results/2026-09-01-3090x2-sparse/report.md` | 41 | 2 | ms (K=11, deepk-shared-2g#r2) | hardware-dependent | \| 11 \| 0 \| 4 \| 2 \| |
| L-262 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 1 | 2×RTX 3090; 2026-09-01 | hardware/date | hardware-dependent | # Sparse-CSR follow-up — 2×RTX 3090, 2026-09-01 |
| L-263 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 4 | 2× RTX 3090 24 GB, sm_86, driver 580.159.03, CuPy 14.1.1 | GPU/driver | hardware-dependent | Same box as the 2026-08-31 campaign (2× RTX 3090 24 GB, sm_86, driver 580.159.03, CuPy 14.1.1, NCCL over SHM) |
| L-264 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 11 | 9/9 | tier-equivalence legs | deterministic | Tier-equivalence chain 9/9 — now extended with single-GPU and multi-GPU sparse CSR legs |
| L-265 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 12 | sparse_from_k=3 | param (smoke preset) | method-parameter | (`sparse_from_k=3` on the smoke preset; CLAUDE.md) |
| L-266 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 13 | 13 | deep_k configs | deterministic | All 13 deep_k configs (dense legacy/shared × 1/2 GPUs, NCCL fallback, prefilter on/off, density-auto × 1/2 GPUs) |
| L-267 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 15 | 8841 | n_itemsets (deep_k) | deterministic | produce the campaign's exact signature: `n_itemsets=8841`, `sum_counts=266261275`, |
| L-268 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 15 | 266261275 | sum_counts (deep_k) | deterministic | `sum_counts=266261275` |
| L-269 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 16 | 4d2c8d28bcd33cd6… | itemset_hash prefix (deep_k) | deterministic | `itemset_hash=4d2c8d28bcd33cd6…` — bit-identical to dense mining. |
| L-270 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 26 | 10.4 s vs 1.3 s | s | hardware-dependent | The campaign measured `deepk-density-auto` at 10.4 s vs 1.3 s dense |
| L-271 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 28 | ~30 ms per level | ms (pair building) | hardware-dependent | pair building was ~30 ms per level, while ~2.5 s |
| L-272 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 29 | ~2.5 s per level (84%) | s / share (host tidset rebuild) | hardware-dependent | ~2.5 s per level (84%) went to rebuilding survivors' tidsets on the host |
| L-273 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 34 | 10.431 s → 1.343 s; legacy-2g 1.338 s; shared-2g 1.35 s; 1.00× | s / ratio (deepk-density-auto 2 GPUs) | hardware-dependent | \| `deepk-density-auto` (2 GPUs) \| 10.431 s \| **1.343 s** \| legacy-2g 1.338 s, shared-2g 1.35 s → **1.00×** \| |
| L-274 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 35 | 0.710 s; legacy-1g 0.584 s; 1.22× | s / ratio (deepk-density-auto-1g) | hardware-dependent | \| `deepk-density-auto-1g` (new) \| — \| **0.710 s** \| legacy-1g 0.584 s → 1.22× \| |
| L-275 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 37 | K=5 | sparse transition level (deep_k) | method-parameter | Per-level, 2 GPUs (ms; the sparse path runs from K=5): |
| L-276 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 41 | K5: frequent=1854; ms 08-31=3145; ms 09-01=48 (transition + first-use NVRTC compile); dense legacy-2g ms=3.6 | count / ms | hardware-dependent | \| 5 \| 1,854 \| 3145 \| 48 (transition + first-use NVRTC compile) \| 3.6 \| |
| L-277 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 42 | K6: frequent=1848; ms 08-31=2553; ms 09-01=4.1; dense legacy-2g ms=2.4 | count / ms | hardware-dependent | \| 6 \| 1,848 \| 2553 \| 4.1 \| 2.4 \| |
| L-278 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 43 | K7: frequent=1320; ms 08-31=1817; ms 09-01=3.5; dense legacy-2g ms=2.5 | count / ms | hardware-dependent | \| 7 \| 1,320 \| 1817 \| 3.5 \| 2.5 \| |
| L-279 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 44 | K8: frequent=660; ms 08-31=902; ms 09-01=3.0; dense legacy-2g ms=3.2 | count / ms | hardware-dependent | \| 8 \| 660 \| 902 \| 3.0 \| 3.2 \| |
| L-280 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 45 | K9: frequent=220; ms 08-31=374; ms 09-01=2.9; dense legacy-2g ms=2.4 | count / ms | hardware-dependent | \| 9 \| 220 \| 374 \| 2.9 \| 2.4 \| |
| L-281 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 46 | K10: frequent=44; ms 08-31=65; ms 09-01=2.7; dense legacy-2g ms=2.2 | count / ms | hardware-dependent | \| 10 \| 44 \| 65 \| 2.7 \| 2.2 \| |
| L-282 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 47 | K11: frequent=4; ms 08-31=14; ms 09-01=2.7; dense legacy-2g ms=1.9 | count / ms | hardware-dependent | \| 11 \| 4 \| 14 \| 2.7 \| 1.9 \| |
| L-283 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 49 | ~700× | per-level speedup vs host rebuild | hardware-dependent | Sparse levels are now single-digit milliseconds — ~700× faster per level than the host rebuild |
| L-284 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 51 | 406/408 MB (dense 414/416 MB) | MB peak VRAM | hardware-dependent | Peak VRAM 406/408 MB (dense 414/416 MB) |
| L-285 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 64 | 156 MB tidsets vs 168 MB bitvecs | MB at transition (deep_k) | deterministic | at deep_k scale it saves almost nothing (156 MB tidsets vs 168 MB bitvecs at the transition) |
| L-286 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 76 | ~11–15 ms sparse vs ~2–11 ms dense (K=6–9, 1 GPU) | ms per level | hardware-dependent | The single-GPU sparse levels cost ~11–15 ms each vs ~2–11 ms dense on one 3090 (K=6–9) |
| L-287 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 80 | 0.7 s | s wall (1-GPU density-auto) | hardware-dependent | Not worth tuning at a 0.7 s wall. |
| L-288 | `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 81 | 0.90 s vs 1.34 s | s (nonccl vs NCCL) | hardware-dependent | `deepk-nonccl` (0.90 s) is faster than NCCL (1.34 s) on this tiny data |
| L-289 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 1 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=3; min_support=None; two_phase=False; timeout_s=3600 | params | method-parameter | "id": "stressk2-legacy-1g#r0", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-legacy-1g_r0.result.json" |
| L-290 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 1 | 2238.846 | s wall | hardware-dependent | stressk2-legacy-1g#r0: "status": "ok", "wall_s": 2238.846 |
| L-291 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 1 | GPU0=17340; GPU1=4 | MB peak VRAM | hardware-dependent | stressk2-legacy-1g#r0: "peak_vram_mb": {"0": 17340, "1": 4} |
| L-292 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 1 | 0x0000000000000001, 0x0000000000000004 | nvml throttle flags | hardware-dependent | stressk2-legacy-1g#r0: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004"] |
| L-293 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 1 | 2996485 | n_itemsets | deterministic | stressk2-legacy-1g#r0: "motifs_ok": true, "n_itemsets": 2996485 |
| L-294 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 1 | 314055259 | sum_counts | deterministic | stressk2-legacy-1g#r0: "sum_counts": 314055259 |
| L-295 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 1 | 0b3e8434997fd3b1ec1e95357857b6d575988d0f4d32f478710738887828aca6 | sha256 itemset_hash | deterministic | stressk2-legacy-1g#r0: "itemset_hash": "0b3e8434997fd3b1…" |
| L-296 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 1 | K1 cand=35000 freq=35000; K2 cand=612482500 freq=1660332; K3 cand=1301153 freq=1301153 | per-level candidates/frequent | deterministic | stressk2-legacy-1g#r0: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-297 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 1 | K1=71.9; K2=175580.2; K3=2046663.2 | ms per level | hardware-dependent | stressk2-legacy-1g#r0: per-level "ms" values (rounded to 0.1) |
| L-298 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 2 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-legacy-1g#r0", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-legacy-1g_r0.result.json" |
| L-299 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 2 | 0.603 | s wall | hardware-dependent | deepk-legacy-1g#r0: "status": "ok", "wall_s": 0.603 |
| L-300 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 2 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-legacy-1g#r0: "peak_vram_mb": {"0": 352, "1": 4} |
| L-301 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 2 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-legacy-1g#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-302 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 2 | 8841 | n_itemsets | deterministic | deepk-legacy-1g#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-303 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 2 | 266261275 | sum_counts | deterministic | deepk-legacy-1g#r0: "sum_counts": 266261275 |
| L-304 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 2 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-legacy-1g#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-305 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 2 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-legacy-1g#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-306 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 2 | K1=11.7; K2=3.4; K3=6.0; K4=7.9; K5=10.4; K6=12.3; K7=9.0; K8=5.0; K9=2.1; K10=0.8; K11=0.4 | ms per level | hardware-dependent | deepk-legacy-1g#r0: per-level "ms" values (rounded to 0.1) |
| L-307 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 3 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'ET_MINER_FILTER_IMPL': 'compact'}; n_gpus=2; sparse_from_k=None; max_length=2; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "stressk2-filter-compact#r0", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-filter-compact_r0.result.json" |
| L-308 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 3 | 89.571 | s wall | hardware-dependent | stressk2-filter-compact#r0: "status": "ok", "wall_s": 89.571 |
| L-309 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 3 | GPU0=6918; GPU1=6918 | MB peak VRAM | hardware-dependent | stressk2-filter-compact#r0: "peak_vram_mb": {"0": 6918, "1": 6918} |
| L-310 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 3 | 0x0000000000000001, 0x0000000000000004 | nvml throttle flags | hardware-dependent | stressk2-filter-compact#r0: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004"] |
| L-311 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 3 | 1695332 | n_itemsets | deterministic | stressk2-filter-compact#r0: "motifs_ok": true, "n_itemsets": 1695332 |
| L-312 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 3 | 218250884 | sum_counts | deterministic | stressk2-filter-compact#r0: "sum_counts": 218250884 |
| L-313 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 3 | 8d989bfcc6e5c745f67108a9b76aacdbe7ae7929ae1df9a0b96a5d2f9dfba006 | sha256 itemset_hash | deterministic | stressk2-filter-compact#r0: "itemset_hash": "8d989bfcc6e5c745…" |
| L-314 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 3 | K1 cand=35000 freq=35000; K2 cand=0 freq=1660332 | per-level candidates/frequent | deterministic | stressk2-filter-compact#r0: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-315 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 3 | K1=40.9; K2=86424.3 | ms per level | hardware-dependent | stressk2-filter-compact#r0: per-level "ms" values (rounded to 0.1) |
| L-316 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 4 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'ET_MINER_FILTER_IMPL': 'cupy'}; n_gpus=2; sparse_from_k=None; max_length=2; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "stressk2-filter-cupy#r0", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-filter-cupy_r0.result.json" |
| L-317 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 4 | 89.665 | s wall | hardware-dependent | stressk2-filter-cupy#r0: "status": "ok", "wall_s": 89.665 |
| L-318 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 4 | GPU0=6998; GPU1=6920 | MB peak VRAM | hardware-dependent | stressk2-filter-cupy#r0: "peak_vram_mb": {"0": 6998, "1": 6920} |
| L-319 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 4 | 0x0000000000000001, 0x0000000000000004 | nvml throttle flags | hardware-dependent | stressk2-filter-cupy#r0: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004"] |
| L-320 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 4 | 1695332 | n_itemsets | deterministic | stressk2-filter-cupy#r0: "motifs_ok": true, "n_itemsets": 1695332 |
| L-321 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 4 | 218250884 | sum_counts | deterministic | stressk2-filter-cupy#r0: "sum_counts": 218250884 |
| L-322 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 4 | 8d989bfcc6e5c745f67108a9b76aacdbe7ae7929ae1df9a0b96a5d2f9dfba006 | sha256 itemset_hash | deterministic | stressk2-filter-cupy#r0: "itemset_hash": "8d989bfcc6e5c745…" |
| L-323 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 4 | K1 cand=35000 freq=35000; K2 cand=0 freq=1660332 | per-level candidates/frequent | deterministic | stressk2-filter-cupy#r0: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-324 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 4 | K1=41.4; K2=86467.9 | ms per level | hardware-dependent | stressk2-filter-cupy#r0: per-level "ms" values (rounded to 0.1) |
| L-325 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 5 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'ET_MINER_FILTER_IMPL': 'cpu'}; n_gpus=2; sparse_from_k=None; max_length=2; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "stressk2-filter-cpu#r0", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-filter-cpu_r0.result.json" |
| L-326 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 5 | 90.196 | s wall | hardware-dependent | stressk2-filter-cpu#r0: "status": "ok", "wall_s": 90.196 |
| L-327 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 5 | GPU0=6920; GPU1=6920 | MB peak VRAM | hardware-dependent | stressk2-filter-cpu#r0: "peak_vram_mb": {"0": 6920, "1": 6920} |
| L-328 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 5 | 0x0000000000000001, 0x0000000000000004 | nvml throttle flags | hardware-dependent | stressk2-filter-cpu#r0: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004"] |
| L-329 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 5 | 1695332 | n_itemsets | deterministic | stressk2-filter-cpu#r0: "motifs_ok": true, "n_itemsets": 1695332 |
| L-330 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 5 | 218250884 | sum_counts | deterministic | stressk2-filter-cpu#r0: "sum_counts": 218250884 |
| L-331 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 5 | 8d989bfcc6e5c745f67108a9b76aacdbe7ae7929ae1df9a0b96a5d2f9dfba006 | sha256 itemset_hash | deterministic | stressk2-filter-cpu#r0: "itemset_hash": "8d989bfcc6e5c745…" |
| L-332 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 5 | K1 cand=35000 freq=35000; K2 cand=0 freq=1660332 | per-level candidates/frequent | deterministic | stressk2-filter-cpu#r0: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-333 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 5 | K1=52.0; K2=86704.7 | ms per level | hardware-dependent | stressk2-filter-cpu#r0: per-level "ms" values (rounded to 0.1) |
| L-334 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 6 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'ET_MINER_DISABLE_NCCL': '1'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-nonccl#r0", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-nonccl_r0.result.json" |
| L-335 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 6 | 0.863 | s wall | hardware-dependent | deepk-nonccl#r0: "status": "ok", "wall_s": 0.863 |
| L-336 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 6 | GPU0=310; GPU1=310 | MB peak VRAM | hardware-dependent | deepk-nonccl#r0: "peak_vram_mb": {"0": 310, "1": 310} |
| L-337 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 6 | (none) | nvml throttle flags | hardware-dependent | deepk-nonccl#r0: "throttle_reasons": [] |
| L-338 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 6 | 8841 | n_itemsets | deterministic | deepk-nonccl#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-339 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 6 | 266261275 | sum_counts | deterministic | deepk-nonccl#r0: "sum_counts": 266261275 |
| L-340 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 6 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-nonccl#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-341 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 6 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-nonccl#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-342 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 6 | K1=20.4; K2=12.8; K3=4.3; K4=4.4; K5=3.1; K6=2.2; K7=2.3; K8=2.2; K9=1.8; K10=2.1; K11=1.7 | ms per level | hardware-dependent | deepk-nonccl#r0: per-level "ms" values (rounded to 0.1) |
| L-343 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 7 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy'}; n_gpus=2; sparse_from_k=auto; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-density-auto#r0", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-density-auto_r0.result.json" |
| L-344 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 7 | 10.431 | s wall | hardware-dependent | deepk-density-auto#r0: "status": "ok", "wall_s": 10.431 |
| L-345 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 7 | GPU0=416; GPU1=416 | MB peak VRAM | hardware-dependent | deepk-density-auto#r0: "peak_vram_mb": {"0": 416, "1": 416} |
| L-346 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 7 | (none) | nvml throttle flags | hardware-dependent | deepk-density-auto#r0: "throttle_reasons": [] |
| L-347 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 7 | 8841 | n_itemsets | deterministic | deepk-density-auto#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-348 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 7 | 266261275 | sum_counts | deterministic | deepk-density-auto#r0: "sum_counts": 266261275 |
| L-349 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 7 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-density-auto#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-350 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 7 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-density-auto#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-351 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 7 | K1=26.5; K2=45.8; K3=5.3; K4=5.5; K5=3145.4; K6=2552.8; K7=1817.4; K8=902.1; K9=373.7; K10=65.2; K11=13.6 | ms per level | hardware-dependent | deepk-density-auto#r0: per-level "ms" values (rounded to 0.1) |
| L-352 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 8 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'CUDA_VISIBLE_DEVICES': '0', 'ET_MINER_DISABLE_PREFILTER': '1'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-prefilter-off#r0", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-prefilter-off_r0.result.json" |
| L-353 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 8 | 0.609 | s wall | hardware-dependent | deepk-prefilter-off#r0: "status": "ok", "wall_s": 0.609 |
| L-354 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 8 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-prefilter-off#r0: "peak_vram_mb": {"0": 352, "1": 4} |
| L-355 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 8 | (none) | nvml throttle flags | hardware-dependent | deepk-prefilter-off#r0: "throttle_reasons": [] |
| L-356 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 8 | 8841 | n_itemsets | deterministic | deepk-prefilter-off#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-357 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 8 | 266261275 | sum_counts | deterministic | deepk-prefilter-off#r0: "sum_counts": 266261275 |
| L-358 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 8 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-prefilter-off#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-359 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 8 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-prefilter-off#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-360 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 8 | K1=11.4; K2=3.4; K3=6.0; K4=7.9; K5=12.4; K6=11.5; K7=9.0; K8=5.2; K9=2.1; K10=0.8; K11=0.4 | ms per level | hardware-dependent | deepk-prefilter-off#r0: per-level "ms" values (rounded to 0.1) |
| L-361 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 9 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-single-prefilter-on#r0", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-single-prefilter-on_r0.result.json" |
| L-362 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 9 | 0.592 | s wall | hardware-dependent | deepk-single-prefilter-on#r0: "status": "ok", "wall_s": 0.592 |
| L-363 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 9 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-single-prefilter-on#r0: "peak_vram_mb": {"0": 352, "1": 4} |
| L-364 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 9 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-single-prefilter-on#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-365 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 9 | 8841 | n_itemsets | deterministic | deepk-single-prefilter-on#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-366 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 9 | 266261275 | sum_counts | deterministic | deepk-single-prefilter-on#r0: "sum_counts": 266261275 |
| L-367 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 9 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-single-prefilter-on#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-368 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 9 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-single-prefilter-on#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-369 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 9 | K1=11.4; K2=2.9; K3=5.9; K4=7.8; K5=10.4; K6=12.2; K7=8.8; K8=4.9; K9=2.2; K10=0.8; K11=0.4 | ms per level | hardware-dependent | deepk-single-prefilter-on#r0: per-level "ms" values (rounded to 0.1) |
| L-370 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 10 | preset=smoke; env={'ET_MINER_KERNEL_VARIANT': 'legacy'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=True; timeout_s=1800 | params | method-parameter | "id": "twophase-smoke#r0", "preset": "smoke" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/twophase-smoke_r0.result.json" |
| L-371 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 10 | 1.582 | s wall | hardware-dependent | twophase-smoke#r0: "status": "ok", "wall_s": 1.582 |
| L-372 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 10 | GPU0=408; GPU1=408 | MB peak VRAM | hardware-dependent | twophase-smoke#r0: "peak_vram_mb": {"0": 408, "1": 408} |
| L-373 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 10 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | twophase-smoke#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-374 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 10 | 632 | n_itemsets | deterministic | twophase-smoke#r0: "motifs_ok": true, "n_itemsets": 632 |
| L-375 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 10 | 1049580 | sum_counts | deterministic | twophase-smoke#r0: "sum_counts": 1049580 |
| L-376 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 10 | ea17ea26fd0e44f735b62a5699d1227477965905ff34068203ac513aaee6deee | sha256 itemset_hash | deterministic | twophase-smoke#r0: "itemset_hash": "ea17ea26fd0e44f7…" |
| L-377 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 10 | K1 cand=118 freq=118; K2 cand=0 freq=290; K3 cand=0 freq=202; K4 cand=0 freq=22; K5 cand=0 freq=0; K1 cand=118 freq=118; K2 cand=0 freq=290; K3 cand=0 freq=202; K4 cand=0 freq=22; K5 cand=0 freq=0 | per-level candidates/frequent | deterministic | twophase-smoke#r0: "levels": [{"k": 1, "n_candidates": 118, "n_frequent": 118 … |
| L-378 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 10 | K1=251.7; K2=48.4; K3=119.5; K4=6.9; K5=0.1; K1=1.5; K2=33.9; K3=87.1; K4=5.1; K5=0.1 | ms per level | hardware-dependent | twophase-smoke#r0: per-level "ms" values (rounded to 0.1) |
| L-379 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 11 | preset=skewed_rows; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'ET_MINER_ROW_BALANCE': 'rows'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "skew-rows#r0", "preset": "skewed_rows" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/skew-rows_r0.result.json" |
| L-380 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 11 | 1.435 | s wall | hardware-dependent | skew-rows#r0: "status": "ok", "wall_s": 1.435 |
| L-381 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 11 | GPU0=416; GPU1=416 | MB peak VRAM | hardware-dependent | skew-rows#r0: "peak_vram_mb": {"0": 416, "1": 416} |
| L-382 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 11 | (none) | nvml throttle flags | hardware-dependent | skew-rows#r0: "throttle_reasons": [] |
| L-383 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 11 | 10350 | n_itemsets | deterministic | skew-rows#r0: "motifs_ok": true, "n_itemsets": 10350 |
| L-384 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 11 | 346834073 | sum_counts | deterministic | skew-rows#r0: "sum_counts": 346834073 |
| L-385 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 11 | 75262446a1c2b29bbc15e6d36afe23300984e6d46f8483511f055f1a510c2405 | sha256 itemset_hash | deterministic | skew-rows#r0: "itemset_hash": "75262446a1c2b29b…" |
| L-386 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 11 | K1 cand=118 freq=118; K2 cand=0 freq=717; K3 cand=0 freq=1928; K4 cand=0 freq=2932; K5 cand=0 freq=2683; K6 cand=0 freq=1458; K7 cand=0 freq=443; K8 cand=0 freq=67; K9 cand=0 freq=4 | per-level candidates/frequent | deterministic | skew-rows#r0: "levels": [{"k": 1, "n_candidates": 118, "n_frequent": 118 … |
| L-387 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 11 | K1=25.7; K2=48.0; K3=6.1; K4=14.7; K5=32.7; K6=41.6; K7=20.8; K8=5.9; K9=2.1 | ms per level | hardware-dependent | skew-rows#r0: per-level "ms" values (rounded to 0.1) |
| L-388 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 12 | preset=skewed_rows; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'ET_MINER_ROW_BALANCE': 'nnz'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "skew-nnz#r0", "preset": "skewed_rows" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/skew-nnz_r0.result.json" |
| L-389 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 12 | 1.488 | s wall | hardware-dependent | skew-nnz#r0: "status": "ok", "wall_s": 1.488 |
| L-390 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 12 | GPU0=414; GPU1=418 | MB peak VRAM | hardware-dependent | skew-nnz#r0: "peak_vram_mb": {"0": 414, "1": 418} |
| L-391 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 12 | (none) | nvml throttle flags | hardware-dependent | skew-nnz#r0: "throttle_reasons": [] |
| L-392 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 12 | 10350 | n_itemsets | deterministic | skew-nnz#r0: "motifs_ok": true, "n_itemsets": 10350 |
| L-393 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 12 | 346834073 | sum_counts | deterministic | skew-nnz#r0: "sum_counts": 346834073 |
| L-394 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 12 | 75262446a1c2b29bbc15e6d36afe23300984e6d46f8483511f055f1a510c2405 | sha256 itemset_hash | deterministic | skew-nnz#r0: "itemset_hash": "75262446a1c2b29b…" |
| L-395 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 12 | K1 cand=118 freq=118; K2 cand=0 freq=717; K3 cand=0 freq=1928; K4 cand=0 freq=2932; K5 cand=0 freq=2683; K6 cand=0 freq=1458; K7 cand=0 freq=443; K8 cand=0 freq=67; K9 cand=0 freq=4 | per-level candidates/frequent | deterministic | skew-nnz#r0: "levels": [{"k": 1, "n_candidates": 118, "n_frequent": 118 … |
| L-396 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 12 | K1=26.6; K2=48.2; K3=6.2; K4=16.0; K5=33.7; K6=40.6; K7=22.8; K8=5.5; K9=2.1 | ms per level | hardware-dependent | skew-nnz#r0: per-level "ms" values (rounded to 0.1) |
| L-397 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 13 | preset=skewed_rows; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'ET_MINER_ROW_BALANCE': 'rows'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "skew-rows#r1", "preset": "skewed_rows" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/skew-rows_r1.result.json" |
| L-398 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 13 | 1.441 | s wall | hardware-dependent | skew-rows#r1: "status": "ok", "wall_s": 1.441 |
| L-399 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 13 | GPU0=416; GPU1=416 | MB peak VRAM | hardware-dependent | skew-rows#r1: "peak_vram_mb": {"0": 416, "1": 416} |
| L-400 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 13 | (none) | nvml throttle flags | hardware-dependent | skew-rows#r1: "throttle_reasons": [] |
| L-401 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 13 | 10350 | n_itemsets | deterministic | skew-rows#r1: "motifs_ok": true, "n_itemsets": 10350 |
| L-402 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 13 | 346834073 | sum_counts | deterministic | skew-rows#r1: "sum_counts": 346834073 |
| L-403 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 13 | 75262446a1c2b29bbc15e6d36afe23300984e6d46f8483511f055f1a510c2405 | sha256 itemset_hash | deterministic | skew-rows#r1: "itemset_hash": "75262446a1c2b29b…" |
| L-404 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 13 | K1 cand=118 freq=118; K2 cand=0 freq=717; K3 cand=0 freq=1928; K4 cand=0 freq=2932; K5 cand=0 freq=2683; K6 cand=0 freq=1458; K7 cand=0 freq=443; K8 cand=0 freq=67; K9 cand=0 freq=4 | per-level candidates/frequent | deterministic | skew-rows#r1: "levels": [{"k": 1, "n_candidates": 118, "n_frequent": 118 … |
| L-405 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 13 | K1=26.4; K2=49.0; K3=6.7; K4=15.4; K5=32.7; K6=38.5; K7=23.9; K8=5.1; K9=2.1 | ms per level | hardware-dependent | skew-rows#r1: per-level "ms" values (rounded to 0.1) |
| L-406 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 14 | preset=skewed_rows; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'ET_MINER_ROW_BALANCE': 'nnz'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "skew-nnz#r1", "preset": "skewed_rows" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/skew-nnz_r1.result.json" |
| L-407 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 14 | 1.517 | s wall | hardware-dependent | skew-nnz#r1: "status": "ok", "wall_s": 1.517 |
| L-408 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 14 | GPU0=414; GPU1=418 | MB peak VRAM | hardware-dependent | skew-nnz#r1: "peak_vram_mb": {"0": 414, "1": 418} |
| L-409 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 14 | (none) | nvml throttle flags | hardware-dependent | skew-nnz#r1: "throttle_reasons": [] |
| L-410 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 14 | 10350 | n_itemsets | deterministic | skew-nnz#r1: "motifs_ok": true, "n_itemsets": 10350 |
| L-411 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 14 | 346834073 | sum_counts | deterministic | skew-nnz#r1: "sum_counts": 346834073 |
| L-412 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 14 | 75262446a1c2b29bbc15e6d36afe23300984e6d46f8483511f055f1a510c2405 | sha256 itemset_hash | deterministic | skew-nnz#r1: "itemset_hash": "75262446a1c2b29b…" |
| L-413 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 14 | K1 cand=118 freq=118; K2 cand=0 freq=717; K3 cand=0 freq=1928; K4 cand=0 freq=2932; K5 cand=0 freq=2683; K6 cand=0 freq=1458; K7 cand=0 freq=443; K8 cand=0 freq=67; K9 cand=0 freq=4 | per-level candidates/frequent | deterministic | skew-nnz#r1: "levels": [{"k": 1, "n_candidates": 118, "n_frequent": 118 … |
| L-414 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 14 | K1=25.7; K2=46.6; K3=6.6; K4=16.5; K5=34.0; K6=39.9; K7=21.9; K8=5.0; K9=2.0 | ms per level | hardware-dependent | skew-nnz#r1: per-level "ms" values (rounded to 0.1) |
| L-415 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 15 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'legacy'}; n_gpus=2; sparse_from_k=None; max_length=3; min_support=None; two_phase=False; timeout_s=3600 | params | method-parameter | "id": "stressk2-legacy-2g#r0", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-legacy-2g_r0.result.json" |
| L-416 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 15 | 1996.726 | s wall | hardware-dependent | stressk2-legacy-2g#r0: "status": "ok", "wall_s": 1996.726 |
| L-417 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 15 | GPU0=16576; GPU1=16576 | MB peak VRAM | hardware-dependent | stressk2-legacy-2g#r0: "peak_vram_mb": {"0": 16576, "1": 16576} |
| L-418 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 15 | 0x0000000000000001, 0x0000000000000004, 0x0000000000000020, 0x0000000000000024 | nvml throttle flags | hardware-dependent | stressk2-legacy-2g#r0: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004", "0x0000000000000020", "0x0000000000000024"] |
| L-419 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 15 | 3005770 | n_itemsets | deterministic | stressk2-legacy-2g#r0: "motifs_ok": true, "n_itemsets": 3005770 |
| L-420 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 15 | 314350393 | sum_counts | deterministic | stressk2-legacy-2g#r0: "sum_counts": 314350393 |
| L-421 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 15 | a6d53e9a5e1b44a78dc01b5d23a8ebfdd0e800c0c716bef8a476f0ba0540fe2a | sha256 itemset_hash | deterministic | stressk2-legacy-2g#r0: "itemset_hash": "a6d53e9a5e1b44a7…" |
| L-422 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 15 | K1 cand=35000 freq=35000; K2 cand=0 freq=1660332; K3 cand=0 freq=1310438 | per-level candidates/frequent | deterministic | stressk2-legacy-2g#r0: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-423 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 15 | K1=42.5; K2=86439.6; K3=1906847.2 | ms per level | hardware-dependent | stressk2-legacy-2g#r0: per-level "ms" values (rounded to 0.1) |
| L-424 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 16 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'shared', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=3; min_support=None; two_phase=False; timeout_s=3600 | params | method-parameter | "id": "stressk2-shared-1g#r0", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-shared-1g_r0.result.json" |
| L-425 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 16 | 255.303 | s wall | hardware-dependent | stressk2-shared-1g#r0: "status": "ok", "wall_s": 255.303 |
| L-426 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 16 | GPU0=17340; GPU1=4 | MB peak VRAM | hardware-dependent | stressk2-shared-1g#r0: "peak_vram_mb": {"0": 17340, "1": 4} |
| L-427 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 16 | 0x0000000000000001, 0x0000000000000004, 0x0000000000000020, 0x0000000000000024 | nvml throttle flags | hardware-dependent | stressk2-shared-1g#r0: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004", "0x0000000000000020", "0x0000000000000024"] |
| L-428 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 16 | 3005770 | n_itemsets | deterministic | stressk2-shared-1g#r0: "motifs_ok": true, "n_itemsets": 3005770 |
| L-429 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 16 | 314350393 | sum_counts | deterministic | stressk2-shared-1g#r0: "sum_counts": 314350393 |
| L-430 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 16 | a6d53e9a5e1b44a78dc01b5d23a8ebfdd0e800c0c716bef8a476f0ba0540fe2a | sha256 itemset_hash | deterministic | stressk2-shared-1g#r0: "itemset_hash": "a6d53e9a5e1b44a7…" |
| L-431 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 16 | K1 cand=35000 freq=35000; K2 cand=612482500 freq=1660332; K3 cand=1310438 freq=1310438 | per-level candidates/frequent | deterministic | stressk2-shared-1g#r0: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-432 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 16 | K1=73.3; K2=18561.1; K3=220167.6 | ms per level | hardware-dependent | stressk2-shared-1g#r0: per-level "ms" values (rounded to 0.1) |
| L-433 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 17 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-1g#r0", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-shared-1g_r0.result.json" |
| L-434 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 17 | 0.627 | s wall | hardware-dependent | deepk-shared-1g#r0: "status": "ok", "wall_s": 0.627 |
| L-435 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 17 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-shared-1g#r0: "peak_vram_mb": {"0": 352, "1": 4} |
| L-436 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 17 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-shared-1g#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-437 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 17 | 8841 | n_itemsets | deterministic | deepk-shared-1g#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-438 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 17 | 266261275 | sum_counts | deterministic | deepk-shared-1g#r0: "sum_counts": 266261275 |
| L-439 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 17 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-1g#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-440 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 17 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-1g#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-441 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 17 | K1=11.4; K2=6.3; K3=5.3; K4=7.6; K5=11.2; K6=15.0; K7=13.2; K8=9.2; K9=4.7; K10=3.1; K11=2.9 | ms per level | hardware-dependent | deepk-shared-1g#r0: per-level "ms" values (rounded to 0.1) |
| L-442 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 18 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'shared'}; n_gpus=2; sparse_from_k=None; max_length=3; min_support=None; two_phase=False; timeout_s=3600 | params | method-parameter | "id": "stressk2-shared-2g#r0", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-shared-2g_r0.result.json" |
| L-443 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 18 | 122.814 | s wall | hardware-dependent | stressk2-shared-2g#r0: "status": "ok", "wall_s": 122.814 |
| L-444 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 18 | GPU0=16576; GPU1=16576 | MB peak VRAM | hardware-dependent | stressk2-shared-2g#r0: "peak_vram_mb": {"0": 16576, "1": 16576} |
| L-445 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 18 | 0x0000000000000001, 0x0000000000000004 | nvml throttle flags | hardware-dependent | stressk2-shared-2g#r0: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004"] |
| L-446 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 18 | 3005770 | n_itemsets | deterministic | stressk2-shared-2g#r0: "motifs_ok": true, "n_itemsets": 3005770 |
| L-447 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 18 | 314350393 | sum_counts | deterministic | stressk2-shared-2g#r0: "sum_counts": 314350393 |
| L-448 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 18 | a6d53e9a5e1b44a78dc01b5d23a8ebfdd0e800c0c716bef8a476f0ba0540fe2a | sha256 itemset_hash | deterministic | stressk2-shared-2g#r0: "itemset_hash": "a6d53e9a5e1b44a7…" |
| L-449 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 18 | K1 cand=35000 freq=35000; K2 cand=0 freq=1660332; K3 cand=0 freq=1310438 | per-level candidates/frequent | deterministic | stressk2-shared-2g#r0: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-450 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 18 | K1=45.2; K2=7215.2; K3=111994.4 | ms per level | hardware-dependent | stressk2-shared-2g#r0: per-level "ms" values (rounded to 0.1) |
| L-451 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 19 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-legacy-2g#r0", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-legacy-2g_r0.result.json" |
| L-452 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 19 | 1.322 | s wall | hardware-dependent | deepk-legacy-2g#r0: "status": "ok", "wall_s": 1.322 |
| L-453 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 19 | GPU0=414; GPU1=416 | MB peak VRAM | hardware-dependent | deepk-legacy-2g#r0: "peak_vram_mb": {"0": 414, "1": 416} |
| L-454 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 19 | (none) | nvml throttle flags | hardware-dependent | deepk-legacy-2g#r0: "throttle_reasons": [] |
| L-455 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 19 | 8841 | n_itemsets | deterministic | deepk-legacy-2g#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-456 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 19 | 266261275 | sum_counts | deterministic | deepk-legacy-2g#r0: "sum_counts": 266261275 |
| L-457 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 19 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-legacy-2g#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-458 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 19 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-legacy-2g#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-459 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 19 | K1=26.0; K2=44.2; K3=6.0; K4=5.5; K5=4.2; K6=2.6; K7=2.4; K8=2.2; K9=2.2; K10=2.0; K11=2.0 | ms per level | hardware-dependent | deepk-legacy-2g#r0: per-level "ms" values (rounded to 0.1) |
| L-460 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 20 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-2g#r0", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-shared-2g_r0.result.json" |
| L-461 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 20 | 1.391 | s wall | hardware-dependent | deepk-shared-2g#r0: "status": "ok", "wall_s": 1.391 |
| L-462 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 20 | GPU0=414; GPU1=416 | MB peak VRAM | hardware-dependent | deepk-shared-2g#r0: "peak_vram_mb": {"0": 414, "1": 416} |
| L-463 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 20 | (none) | nvml throttle flags | hardware-dependent | deepk-shared-2g#r0: "throttle_reasons": [] |
| L-464 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 20 | 8841 | n_itemsets | deterministic | deepk-shared-2g#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-465 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 20 | 266261275 | sum_counts | deterministic | deepk-shared-2g#r0: "sum_counts": 266261275 |
| L-466 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 20 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-2g#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-467 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 20 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-2g#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-468 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 20 | K1=26.4; K2=50.6; K3=6.7; K4=8.3; K5=6.0; K6=2.7; K7=2.5; K8=2.8; K9=2.1; K10=4.5; K11=2.0 | ms per level | hardware-dependent | deepk-shared-2g#r0: per-level "ms" values (rounded to 0.1) |
| L-469 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 21 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'shared', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=3; min_support=None; two_phase=False; timeout_s=3600 | params | method-parameter | "id": "stressk2-shared-1g#r1", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-shared-1g_r1.result.json" |
| L-470 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 21 | 253.074 | s wall | hardware-dependent | stressk2-shared-1g#r1: "status": "ok", "wall_s": 253.074 |
| L-471 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 21 | GPU0=17340; GPU1=4 | MB peak VRAM | hardware-dependent | stressk2-shared-1g#r1: "peak_vram_mb": {"0": 17340, "1": 4} |
| L-472 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 21 | 0x0000000000000001, 0x0000000000000004 | nvml throttle flags | hardware-dependent | stressk2-shared-1g#r1: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004"] |
| L-473 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 21 | 3005770 | n_itemsets | deterministic | stressk2-shared-1g#r1: "motifs_ok": true, "n_itemsets": 3005770 |
| L-474 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 21 | 314350393 | sum_counts | deterministic | stressk2-shared-1g#r1: "sum_counts": 314350393 |
| L-475 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 21 | a6d53e9a5e1b44a78dc01b5d23a8ebfdd0e800c0c716bef8a476f0ba0540fe2a | sha256 itemset_hash | deterministic | stressk2-shared-1g#r1: "itemset_hash": "a6d53e9a5e1b44a7…" |
| L-476 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 21 | K1 cand=35000 freq=35000; K2 cand=612482500 freq=1660332; K3 cand=1310438 freq=1310438 | per-level candidates/frequent | deterministic | stressk2-shared-1g#r1: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-477 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 21 | K1=72.9; K2=18099.5; K3=218647.6 | ms per level | hardware-dependent | stressk2-shared-1g#r1: per-level "ms" values (rounded to 0.1) |
| L-478 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 22 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-1g#r1", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-shared-1g_r1.result.json" |
| L-479 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 22 | 0.639 | s wall | hardware-dependent | deepk-shared-1g#r1: "status": "ok", "wall_s": 0.639 |
| L-480 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 22 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-shared-1g#r1: "peak_vram_mb": {"0": 352, "1": 4} |
| L-481 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 22 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-shared-1g#r1: "throttle_reasons": ["0x0000000000000001"] |
| L-482 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 22 | 8841 | n_itemsets | deterministic | deepk-shared-1g#r1: "motifs_ok": true, "n_itemsets": 8841 |
| L-483 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 22 | 266261275 | sum_counts | deterministic | deepk-shared-1g#r1: "sum_counts": 266261275 |
| L-484 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 22 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-1g#r1: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-485 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 22 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-1g#r1: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-486 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 22 | K1=11.4; K2=6.3; K3=5.3; K4=7.7; K5=11.4; K6=14.1; K7=13.3; K8=9.2; K9=4.7; K10=3.1; K11=3.0 | ms per level | hardware-dependent | deepk-shared-1g#r1: per-level "ms" values (rounded to 0.1) |
| L-487 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 23 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'shared'}; n_gpus=2; sparse_from_k=None; max_length=3; min_support=None; two_phase=False; timeout_s=3600 | params | method-parameter | "id": "stressk2-shared-2g#r1", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-shared-2g_r1.result.json" |
| L-488 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 23 | 122.257 | s wall | hardware-dependent | stressk2-shared-2g#r1: "status": "ok", "wall_s": 122.257 |
| L-489 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 23 | GPU0=16576; GPU1=16576 | MB peak VRAM | hardware-dependent | stressk2-shared-2g#r1: "peak_vram_mb": {"0": 16576, "1": 16576} |
| L-490 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 23 | 0x0000000000000001, 0x0000000000000004 | nvml throttle flags | hardware-dependent | stressk2-shared-2g#r1: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004"] |
| L-491 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 23 | 3005770 | n_itemsets | deterministic | stressk2-shared-2g#r1: "motifs_ok": true, "n_itemsets": 3005770 |
| L-492 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 23 | 314350393 | sum_counts | deterministic | stressk2-shared-2g#r1: "sum_counts": 314350393 |
| L-493 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 23 | a6d53e9a5e1b44a78dc01b5d23a8ebfdd0e800c0c716bef8a476f0ba0540fe2a | sha256 itemset_hash | deterministic | stressk2-shared-2g#r1: "itemset_hash": "a6d53e9a5e1b44a7…" |
| L-494 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 23 | K1 cand=35000 freq=35000; K2 cand=0 freq=1660332; K3 cand=0 freq=1310438 | per-level candidates/frequent | deterministic | stressk2-shared-2g#r1: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-495 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 23 | K1=54.9; K2=7204.6; K3=111955.2 | ms per level | hardware-dependent | stressk2-shared-2g#r1: per-level "ms" values (rounded to 0.1) |
| L-496 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 24 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-2g#r1", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-shared-2g_r1.result.json" |
| L-497 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 24 | 1.316 | s wall | hardware-dependent | deepk-shared-2g#r1: "status": "ok", "wall_s": 1.316 |
| L-498 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 24 | GPU0=414; GPU1=416 | MB peak VRAM | hardware-dependent | deepk-shared-2g#r1: "peak_vram_mb": {"0": 414, "1": 416} |
| L-499 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 24 | (none) | nvml throttle flags | hardware-dependent | deepk-shared-2g#r1: "throttle_reasons": [] |
| L-500 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 24 | 8841 | n_itemsets | deterministic | deepk-shared-2g#r1: "motifs_ok": true, "n_itemsets": 8841 |
| L-501 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 24 | 266261275 | sum_counts | deterministic | deepk-shared-2g#r1: "sum_counts": 266261275 |
| L-502 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 24 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-2g#r1: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-503 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 24 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-2g#r1: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-504 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 24 | K1=26.3; K2=48.9; K3=6.2; K4=9.3; K5=5.4; K6=2.5; K7=2.5; K8=2.2; K9=2.0; K10=2.0; K11=2.2 | ms per level | hardware-dependent | deepk-shared-2g#r1: per-level "ms" values (rounded to 0.1) |
| L-505 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 25 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'shared', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=3; min_support=None; two_phase=False; timeout_s=3600 | params | method-parameter | "id": "stressk2-shared-1g#r2", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-shared-1g_r2.result.json" |
| L-506 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 25 | 253.285 | s wall | hardware-dependent | stressk2-shared-1g#r2: "status": "ok", "wall_s": 253.285 |
| L-507 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 25 | GPU0=17340; GPU1=4 | MB peak VRAM | hardware-dependent | stressk2-shared-1g#r2: "peak_vram_mb": {"0": 17340, "1": 4} |
| L-508 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 25 | 0x0000000000000001, 0x0000000000000004 | nvml throttle flags | hardware-dependent | stressk2-shared-1g#r2: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004"] |
| L-509 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 25 | 3005770 | n_itemsets | deterministic | stressk2-shared-1g#r2: "motifs_ok": true, "n_itemsets": 3005770 |
| L-510 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 25 | 314350393 | sum_counts | deterministic | stressk2-shared-1g#r2: "sum_counts": 314350393 |
| L-511 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 25 | a6d53e9a5e1b44a78dc01b5d23a8ebfdd0e800c0c716bef8a476f0ba0540fe2a | sha256 itemset_hash | deterministic | stressk2-shared-1g#r2: "itemset_hash": "a6d53e9a5e1b44a7…" |
| L-512 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 25 | K1 cand=35000 freq=35000; K2 cand=612482500 freq=1660332; K3 cand=1310438 freq=1310438 | per-level candidates/frequent | deterministic | stressk2-shared-1g#r2: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-513 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 25 | K1=73.9; K2=18136.5; K3=218798.1 | ms per level | hardware-dependent | stressk2-shared-1g#r2: per-level "ms" values (rounded to 0.1) |
| L-514 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 26 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-1g#r2", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-shared-1g_r2.result.json" |
| L-515 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 26 | 0.603 | s wall | hardware-dependent | deepk-shared-1g#r2: "status": "ok", "wall_s": 0.603 |
| L-516 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 26 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-shared-1g#r2: "peak_vram_mb": {"0": 352, "1": 4} |
| L-517 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 26 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-shared-1g#r2: "throttle_reasons": ["0x0000000000000001"] |
| L-518 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 26 | 8841 | n_itemsets | deterministic | deepk-shared-1g#r2: "motifs_ok": true, "n_itemsets": 8841 |
| L-519 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 26 | 266261275 | sum_counts | deterministic | deepk-shared-1g#r2: "sum_counts": 266261275 |
| L-520 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 26 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-1g#r2: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-521 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 26 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-1g#r2: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-522 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 26 | K1=11.3; K2=6.3; K3=5.3; K4=7.6; K5=11.2; K6=13.9; K7=13.2; K8=9.2; K9=4.7; K10=3.1; K11=3.0 | ms per level | hardware-dependent | deepk-shared-1g#r2: per-level "ms" values (rounded to 0.1) |
| L-523 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 27 | preset=stress_k2; env={'ET_MINER_KERNEL_VARIANT': 'shared'}; n_gpus=2; sparse_from_k=None; max_length=3; min_support=None; two_phase=False; timeout_s=3600 | params | method-parameter | "id": "stressk2-shared-2g#r2", "preset": "stress_k2" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/stressk2-shared-2g_r2.result.json" |
| L-524 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 27 | 122.333 | s wall | hardware-dependent | stressk2-shared-2g#r2: "status": "ok", "wall_s": 122.333 |
| L-525 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 27 | GPU0=16576; GPU1=16576 | MB peak VRAM | hardware-dependent | stressk2-shared-2g#r2: "peak_vram_mb": {"0": 16576, "1": 16576} |
| L-526 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 27 | 0x0000000000000001, 0x0000000000000004 | nvml throttle flags | hardware-dependent | stressk2-shared-2g#r2: "throttle_reasons": ["0x0000000000000001", "0x0000000000000004"] |
| L-527 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 27 | 3005770 | n_itemsets | deterministic | stressk2-shared-2g#r2: "motifs_ok": true, "n_itemsets": 3005770 |
| L-528 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 27 | 314350393 | sum_counts | deterministic | stressk2-shared-2g#r2: "sum_counts": 314350393 |
| L-529 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 27 | a6d53e9a5e1b44a78dc01b5d23a8ebfdd0e800c0c716bef8a476f0ba0540fe2a | sha256 itemset_hash | deterministic | stressk2-shared-2g#r2: "itemset_hash": "a6d53e9a5e1b44a7…" |
| L-530 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 27 | K1 cand=35000 freq=35000; K2 cand=0 freq=1660332; K3 cand=0 freq=1310438 | per-level candidates/frequent | deterministic | stressk2-shared-2g#r2: "levels": [{"k": 1, "n_candidates": 35000, "n_frequent": 35000 … |
| L-531 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 27 | K1=47.2; K2=7198.9; K3=111923.4 | ms per level | hardware-dependent | stressk2-shared-2g#r2: per-level "ms" values (rounded to 0.1) |
| L-532 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 28 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-2g#r2", "preset": "deep_k" … "result_path": "/root/projects/ET-Miner/bench/results/campaign/deepk-shared-2g_r2.result.json" |
| L-533 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 28 | 1.361 | s wall | hardware-dependent | deepk-shared-2g#r2: "status": "ok", "wall_s": 1.361 |
| L-534 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 28 | GPU0=414; GPU1=416 | MB peak VRAM | hardware-dependent | deepk-shared-2g#r2: "peak_vram_mb": {"0": 414, "1": 416} |
| L-535 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 28 | (none) | nvml throttle flags | hardware-dependent | deepk-shared-2g#r2: "throttle_reasons": [] |
| L-536 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 28 | 8841 | n_itemsets | deterministic | deepk-shared-2g#r2: "motifs_ok": true, "n_itemsets": 8841 |
| L-537 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 28 | 266261275 | sum_counts | deterministic | deepk-shared-2g#r2: "sum_counts": 266261275 |
| L-538 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 28 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-2g#r2: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-539 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 28 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-2g#r2: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-540 | `bench/results/2026-08-31-3090x2/raw.jsonl` | 28 | K1=26.4; K2=49.2; K3=6.8; K4=8.4; K5=5.6; K6=2.9; K7=2.5; K8=2.3; K9=2.1; K10=2.2; K11=2.1 | ms per level | hardware-dependent | deepk-shared-2g#r2: per-level "ms" values (rounded to 0.1) |
| L-541 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 1 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'ET_MINER_DISABLE_NCCL': '1'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-nonccl#r0", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-nonccl_r0.result.json" |
| L-542 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 1 | 0.901 | s wall | hardware-dependent | deepk-nonccl#r0: "status": "ok", "wall_s": 0.901 |
| L-543 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 1 | GPU0=302; GPU1=310 | MB peak VRAM | hardware-dependent | deepk-nonccl#r0: "peak_vram_mb": {"0": 302, "1": 310} |
| L-544 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 1 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-nonccl#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-545 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 1 | 8841 | n_itemsets | deterministic | deepk-nonccl#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-546 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 1 | 266261275 | sum_counts | deterministic | deepk-nonccl#r0: "sum_counts": 266261275 |
| L-547 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 1 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-nonccl#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-548 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 1 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-nonccl#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-549 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 1 | K1=22.3; K2=12.5; K3=5.1; K4=3.9; K5=2.8; K6=2.1; K7=2.0; K8=1.8; K9=1.6; K10=1.8; K11=1.7 | ms per level | hardware-dependent | deepk-nonccl#r0: per-level "ms" values (rounded to 0.1) |
| L-550 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 2 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy'}; n_gpus=2; sparse_from_k=auto; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-density-auto#r0", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-density-auto_r0.result.json" |
| L-551 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 2 | 1.343 | s wall | hardware-dependent | deepk-density-auto#r0: "status": "ok", "wall_s": 1.343 |
| L-552 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 2 | GPU0=406; GPU1=408 | MB peak VRAM | hardware-dependent | deepk-density-auto#r0: "peak_vram_mb": {"0": 406, "1": 408} |
| L-553 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 2 | (none) | nvml throttle flags | hardware-dependent | deepk-density-auto#r0: "throttle_reasons": [] |
| L-554 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 2 | 8841 | n_itemsets | deterministic | deepk-density-auto#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-555 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 2 | 266261275 | sum_counts | deterministic | deepk-density-auto#r0: "sum_counts": 266261275 |
| L-556 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 2 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-density-auto#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-557 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 2 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=2251 freq=1854; K6 cand=1855 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-density-auto#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-558 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 2 | K1=25.7; K2=48.8; K3=5.6; K4=5.3; K5=48.1; K6=4.1; K7=3.5; K8=3.0; K9=2.9; K10=2.7; K11=2.7 | ms per level | hardware-dependent | deepk-density-auto#r0: per-level "ms" values (rounded to 0.1) |
| L-559 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 3 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=auto; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-density-auto-1g#r0", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-density-auto-1g_r0.result.json" |
| L-560 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 3 | 0.71 | s wall | hardware-dependent | deepk-density-auto-1g#r0: "status": "ok", "wall_s": 0.71 |
| L-561 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 3 | GPU0=680; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-density-auto-1g#r0: "peak_vram_mb": {"0": 680, "1": 4} |
| L-562 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 3 | (none) | nvml throttle flags | hardware-dependent | deepk-density-auto-1g#r0: "throttle_reasons": [] |
| L-563 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 3 | 8841 | n_itemsets | deterministic | deepk-density-auto-1g#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-564 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 3 | 266261275 | sum_counts | deterministic | deepk-density-auto-1g#r0: "sum_counts": 266261275 |
| L-565 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 3 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-density-auto-1g#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-566 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 3 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=2251 freq=1854; K6 cand=1855 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-density-auto-1g#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-567 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 3 | K1=11.4; K2=2.9; K3=6.5; K4=7.9; K5=74.1; K6=14.7; K7=13.4; K8=11.1; K9=10.8; K10=8.9; K11=2.8 | ms per level | hardware-dependent | deepk-density-auto-1g#r0: per-level "ms" values (rounded to 0.1) |
| L-568 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 4 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-prefilter-off#r0", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-prefilter-off_r0.result.json" |
| L-569 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 4 | 0.59 | s wall | hardware-dependent | deepk-prefilter-off#r0: "status": "ok", "wall_s": 0.59 |
| L-570 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 4 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-prefilter-off#r0: "peak_vram_mb": {"0": 352, "1": 4} |
| L-571 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 4 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-prefilter-off#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-572 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 4 | 8841 | n_itemsets | deterministic | deepk-prefilter-off#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-573 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 4 | 266261275 | sum_counts | deterministic | deepk-prefilter-off#r0: "sum_counts": 266261275 |
| L-574 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 4 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-prefilter-off#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-575 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 4 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-prefilter-off#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-576 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 4 | K1=11.4; K2=3.0; K3=6.0; K4=8.2; K5=10.5; K6=11.2; K7=9.6; K8=5.0; K9=2.1; K10=0.8; K11=0.4 | ms per level | hardware-dependent | deepk-prefilter-off#r0: per-level "ms" values (rounded to 0.1) |
| L-577 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 5 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'CUDA_VISIBLE_DEVICES': '0', 'ET_MINER_ENABLE_PREFILTER': '1'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-single-prefilter-on#r0", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-single-prefilter-on_r0.result.json" |
| L-578 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 5 | 0.595 | s wall | hardware-dependent | deepk-single-prefilter-on#r0: "status": "ok", "wall_s": 0.595 |
| L-579 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 5 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-single-prefilter-on#r0: "peak_vram_mb": {"0": 352, "1": 4} |
| L-580 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 5 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-single-prefilter-on#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-581 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 5 | 8841 | n_itemsets | deterministic | deepk-single-prefilter-on#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-582 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 5 | 266261275 | sum_counts | deterministic | deepk-single-prefilter-on#r0: "sum_counts": 266261275 |
| L-583 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 5 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-single-prefilter-on#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-584 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 5 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-single-prefilter-on#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-585 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 5 | K1=11.4; K2=2.9; K3=6.0; K4=7.9; K5=10.3; K6=12.4; K7=9.1; K8=4.9; K9=2.1; K10=0.8; K11=0.4 | ms per level | hardware-dependent | deepk-single-prefilter-on#r0: per-level "ms" values (rounded to 0.1) |
| L-586 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 6 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-legacy-1g#r0", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-legacy-1g_r0.result.json" |
| L-587 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 6 | 0.584 | s wall | hardware-dependent | deepk-legacy-1g#r0: "status": "ok", "wall_s": 0.584 |
| L-588 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 6 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-legacy-1g#r0: "peak_vram_mb": {"0": 352, "1": 4} |
| L-589 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 6 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-legacy-1g#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-590 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 6 | 8841 | n_itemsets | deterministic | deepk-legacy-1g#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-591 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 6 | 266261275 | sum_counts | deterministic | deepk-legacy-1g#r0: "sum_counts": 266261275 |
| L-592 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 6 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-legacy-1g#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-593 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 6 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-legacy-1g#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-594 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 6 | K1=11.4; K2=2.9; K3=6.1; K4=8.3; K5=10.4; K6=11.2; K7=10.1; K8=4.9; K9=2.1; K10=0.8; K11=0.4 | ms per level | hardware-dependent | deepk-legacy-1g#r0: per-level "ms" values (rounded to 0.1) |
| L-595 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 7 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-1g#r0", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-shared-1g_r0.result.json" |
| L-596 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 7 | 0.613 | s wall | hardware-dependent | deepk-shared-1g#r0: "status": "ok", "wall_s": 0.613 |
| L-597 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 7 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-shared-1g#r0: "peak_vram_mb": {"0": 352, "1": 4} |
| L-598 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 7 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-shared-1g#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-599 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 7 | 8841 | n_itemsets | deterministic | deepk-shared-1g#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-600 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 7 | 266261275 | sum_counts | deterministic | deepk-shared-1g#r0: "sum_counts": 266261275 |
| L-601 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 7 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-1g#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-602 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 7 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-1g#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-603 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 7 | K1=11.4; K2=6.3; K3=5.3; K4=7.7; K5=11.2; K6=14.1; K7=13.2; K8=9.2; K9=4.7; K10=3.1; K11=2.9 | ms per level | hardware-dependent | deepk-shared-1g#r0: per-level "ms" values (rounded to 0.1) |
| L-604 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 8 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'legacy'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-legacy-2g#r0", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-legacy-2g_r0.result.json" |
| L-605 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 8 | 1.338 | s wall | hardware-dependent | deepk-legacy-2g#r0: "status": "ok", "wall_s": 1.338 |
| L-606 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 8 | GPU0=414; GPU1=416 | MB peak VRAM | hardware-dependent | deepk-legacy-2g#r0: "peak_vram_mb": {"0": 414, "1": 416} |
| L-607 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 8 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-legacy-2g#r0: "throttle_reasons": ["0x0000000000000001"] |
| L-608 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 8 | 8841 | n_itemsets | deterministic | deepk-legacy-2g#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-609 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 8 | 266261275 | sum_counts | deterministic | deepk-legacy-2g#r0: "sum_counts": 266261275 |
| L-610 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 8 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-legacy-2g#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-611 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 8 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-legacy-2g#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-612 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 8 | K1=27.9; K2=46.2; K3=5.8; K4=5.4; K5=3.6; K6=2.4; K7=2.5; K8=3.2; K9=2.4; K10=2.2; K11=1.9 | ms per level | hardware-dependent | deepk-legacy-2g#r0: per-level "ms" values (rounded to 0.1) |
| L-613 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 9 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-2g#r0", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-shared-2g_r0.result.json" |
| L-614 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 9 | 1.354 | s wall | hardware-dependent | deepk-shared-2g#r0: "status": "ok", "wall_s": 1.354 |
| L-615 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 9 | GPU0=414; GPU1=416 | MB peak VRAM | hardware-dependent | deepk-shared-2g#r0: "peak_vram_mb": {"0": 414, "1": 416} |
| L-616 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 9 | (none) | nvml throttle flags | hardware-dependent | deepk-shared-2g#r0: "throttle_reasons": [] |
| L-617 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 9 | 8841 | n_itemsets | deterministic | deepk-shared-2g#r0: "motifs_ok": true, "n_itemsets": 8841 |
| L-618 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 9 | 266261275 | sum_counts | deterministic | deepk-shared-2g#r0: "sum_counts": 266261275 |
| L-619 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 9 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-2g#r0: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-620 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 9 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-2g#r0: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-621 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 9 | K1=26.4; K2=49.1; K3=7.2; K4=9.1; K5=5.4; K6=2.5; K7=2.6; K8=2.3; K9=2.6; K10=1.9; K11=1.9 | ms per level | hardware-dependent | deepk-shared-2g#r0: per-level "ms" values (rounded to 0.1) |
| L-622 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 10 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-1g#r1", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-shared-1g_r1.result.json" |
| L-623 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 10 | 0.628 | s wall | hardware-dependent | deepk-shared-1g#r1: "status": "ok", "wall_s": 0.628 |
| L-624 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 10 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-shared-1g#r1: "peak_vram_mb": {"0": 352, "1": 4} |
| L-625 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 10 | (none) | nvml throttle flags | hardware-dependent | deepk-shared-1g#r1: "throttle_reasons": [] |
| L-626 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 10 | 8841 | n_itemsets | deterministic | deepk-shared-1g#r1: "motifs_ok": true, "n_itemsets": 8841 |
| L-627 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 10 | 266261275 | sum_counts | deterministic | deepk-shared-1g#r1: "sum_counts": 266261275 |
| L-628 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 10 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-1g#r1: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-629 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 10 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-1g#r1: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-630 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 10 | K1=11.4; K2=6.3; K3=5.3; K4=7.6; K5=11.2; K6=14.1; K7=13.2; K8=9.3; K9=4.7; K10=3.1; K11=2.9 | ms per level | hardware-dependent | deepk-shared-1g#r1: per-level "ms" values (rounded to 0.1) |
| L-631 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 11 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-2g#r1", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-shared-2g_r1.result.json" |
| L-632 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 11 | 1.346 | s wall | hardware-dependent | deepk-shared-2g#r1: "status": "ok", "wall_s": 1.346 |
| L-633 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 11 | GPU0=414; GPU1=416 | MB peak VRAM | hardware-dependent | deepk-shared-2g#r1: "peak_vram_mb": {"0": 414, "1": 416} |
| L-634 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 11 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-shared-2g#r1: "throttle_reasons": ["0x0000000000000001"] |
| L-635 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 11 | 8841 | n_itemsets | deterministic | deepk-shared-2g#r1: "motifs_ok": true, "n_itemsets": 8841 |
| L-636 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 11 | 266261275 | sum_counts | deterministic | deepk-shared-2g#r1: "sum_counts": 266261275 |
| L-637 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 11 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-2g#r1: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-638 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 11 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-2g#r1: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-639 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 11 | K1=26.3; K2=48.8; K3=7.6; K4=9.5; K5=5.4; K6=2.6; K7=2.5; K8=2.2; K9=1.9; K10=2.2; K11=2.0 | ms per level | hardware-dependent | deepk-shared-2g#r1: per-level "ms" values (rounded to 0.1) |
| L-640 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 12 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared', 'CUDA_VISIBLE_DEVICES': '0'}; n_gpus=1; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-1g#r2", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-shared-1g_r2.result.json" |
| L-641 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 12 | 0.602 | s wall | hardware-dependent | deepk-shared-1g#r2: "status": "ok", "wall_s": 0.602 |
| L-642 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 12 | GPU0=352; GPU1=4 | MB peak VRAM | hardware-dependent | deepk-shared-1g#r2: "peak_vram_mb": {"0": 352, "1": 4} |
| L-643 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 12 | (none) | nvml throttle flags | hardware-dependent | deepk-shared-1g#r2: "throttle_reasons": [] |
| L-644 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 12 | 8841 | n_itemsets | deterministic | deepk-shared-1g#r2: "motifs_ok": true, "n_itemsets": 8841 |
| L-645 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 12 | 266261275 | sum_counts | deterministic | deepk-shared-1g#r2: "sum_counts": 266261275 |
| L-646 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 12 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-1g#r2: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-647 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 12 | K1 cand=112 freq=112; K2 cand=6216 freq=471; K3 cand=901 freq=901; K4 cand=1407 freq=1407; K5 cand=1854 freq=1854; K6 cand=1848 freq=1848; K7 cand=1320 freq=1320; K8 cand=660 freq=660; K9 cand=220 freq=220; K10 cand=44 freq=44; K11 cand=4 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-1g#r2: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-648 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 12 | K1=11.5; K2=6.4; K3=5.3; K4=7.7; K5=11.3; K6=14.1; K7=13.5; K8=9.2; K9=4.7; K10=3.1; K11=3.0 | ms per level | hardware-dependent | deepk-shared-1g#r2: per-level "ms" values (rounded to 0.1) |
| L-649 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 13 | preset=deep_k; env={'ET_MINER_KERNEL_VARIANT': 'shared'}; n_gpus=2; sparse_from_k=None; max_length=None; min_support=None; two_phase=False; timeout_s=1800 | params | method-parameter | "id": "deepk-shared-2g#r2", "preset": "deep_k" … "result_path": "bench/results/2026-09-01-3090x2-sparse/deepk-shared-2g_r2.result.json" |
| L-650 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 13 | 1.382 | s wall | hardware-dependent | deepk-shared-2g#r2: "status": "ok", "wall_s": 1.382 |
| L-651 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 13 | GPU0=414; GPU1=416 | MB peak VRAM | hardware-dependent | deepk-shared-2g#r2: "peak_vram_mb": {"0": 414, "1": 416} |
| L-652 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 13 | 0x0000000000000001 | nvml throttle flags | hardware-dependent | deepk-shared-2g#r2: "throttle_reasons": ["0x0000000000000001"] |
| L-653 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 13 | 8841 | n_itemsets | deterministic | deepk-shared-2g#r2: "motifs_ok": true, "n_itemsets": 8841 |
| L-654 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 13 | 266261275 | sum_counts | deterministic | deepk-shared-2g#r2: "sum_counts": 266261275 |
| L-655 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 13 | 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567 | sha256 itemset_hash | deterministic | deepk-shared-2g#r2: "itemset_hash": "4d2c8d28bcd33cd6…" |
| L-656 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 13 | K1 cand=112 freq=112; K2 cand=0 freq=471; K3 cand=0 freq=901; K4 cand=0 freq=1407; K5 cand=0 freq=1854; K6 cand=0 freq=1848; K7 cand=0 freq=1320; K8 cand=0 freq=660; K9 cand=0 freq=220; K10 cand=0 freq=44; K11 cand=0 freq=4 | per-level candidates/frequent | deterministic | deepk-shared-2g#r2: "levels": [{"k": 1, "n_candidates": 112, "n_frequent": 112 … |
| L-657 | `bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 13 | K1=25.3; K2=50.1; K3=6.8; K4=8.5; K5=5.5; K6=2.5; K7=2.5; K8=2.2; K9=2.1; K10=2.0; K11=2.4 | ms per level | hardware-dependent | deepk-shared-2g#r2: per-level "ms" values (rounded to 0.1) |
| L-658 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 2 | direct_gpu_vs_son | experiment id | method-parameter | "experiment": "direct_gpu_vs_son" |
| L-659 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 3 | 2026-02-19T05:03:26.988464+00:00 | timestamp | software | "timestamp": "2026-02-19T05:03:26.988464+00:00" |
| L-660 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 5 | 1e-05 | min_support (fraction) | method-parameter | "min_support": 1e-05 |
| L-661 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 6 | 0.001% | support_pct | method-parameter | "support_pct": "0.001%" |
| L-662 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 7 | 76890945 | n_transactions (proteins) | deterministic | "n_transactions": 76890945 |
| L-663 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 8 | 768 | min_count | method-parameter | "min_count": 768 |
| L-664 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 9 | null | max_length | method-parameter | "max_length": null |
| L-665 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 10 | true | use_gpu | method-parameter | "use_gpu": true |
| L-666 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 13 | 475865 | itemsets (direct GPU) | deterministic | "direct_gpu_result": { "itemsets": 475865 |
| L-667 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 14 | 50.72 | s (direct GPU time) | hardware-dependent | "time_seconds": 50.72 |
| L-668 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 15 | 14 | max_k (direct GPU) | deterministic | "max_k": 14 |
| L-669 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 17 | 1002 | itemsets at K=1 (direct GPU, 0.001%) | deterministic | "k_distribution": … "1": 1002 |
| L-670 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 18 | 22019 | itemsets at K=2 (direct GPU, 0.001%) | deterministic | "k_distribution": … "2": 22019 |
| L-671 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 19 | 73205 | itemsets at K=3 (direct GPU, 0.001%) | deterministic | "k_distribution": … "3": 73205 |
| L-672 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 20 | 108059 | itemsets at K=4 (direct GPU, 0.001%) | deterministic | "k_distribution": … "4": 108059 |
| L-673 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 21 | 104239 | itemsets at K=5 (direct GPU, 0.001%) | deterministic | "k_distribution": … "5": 104239 |
| L-674 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 22 | 78596 | itemsets at K=6 (direct GPU, 0.001%) | deterministic | "k_distribution": … "6": 78596 |
| L-675 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 23 | 48699 | itemsets at K=7 (direct GPU, 0.001%) | deterministic | "k_distribution": … "7": 48699 |
| L-676 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 24 | 25011 | itemsets at K=8 (direct GPU, 0.001%) | deterministic | "k_distribution": … "8": 25011 |
| L-677 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 25 | 10508 | itemsets at K=9 (direct GPU, 0.001%) | deterministic | "k_distribution": … "9": 10508 |
| L-678 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 26 | 3488 | itemsets at K=10 (direct GPU, 0.001%) | deterministic | "k_distribution": … "10": 3488 |
| L-679 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 27 | 869 | itemsets at K=11 (direct GPU, 0.001%) | deterministic | "k_distribution": … "11": 869 |
| L-680 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 28 | 152 | itemsets at K=12 (direct GPU, 0.001%) | deterministic | "k_distribution": … "12": 152 |
| L-681 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 29 | 17 | itemsets at K=13 (direct GPU, 0.001%) | deterministic | "k_distribution": … "13": 17 |
| L-682 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 30 | 1 | itemsets at K=14 (direct GPU, 0.001%) | deterministic | "k_distribution": … "14": 1 |
| L-683 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 34 | SON (Power) | method | method-parameter | "son_reference": { "method": "SON (Power)" |
| L-684 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 35 | 0.001% / 1e-05 | support (SON) | method-parameter | "support_pct": "0.001%", "min_support": 1e-05 |
| L-685 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 37 | 22846 | itemsets (SON) | deterministic | "itemsets": 22846 |
| L-686 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 38 | 1085.6 | s (SON time) | hardware-dependent | "time_seconds": 1085.6 |
| L-687 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 39 | 13 | max_k (SON) | deterministic | "max_k": 13 |
| L-688 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 42 | Direct GPU (Blitz) | method | method-parameter | "blitz_reference": { "method": "Direct GPU (Blitz)" |
| L-689 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 43 | 0.0001% / 1e-06 | support (Blitz) | method-parameter | "support_pct": "0.0001%", "min_support": 1e-06 |
| L-690 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 45 | 2841280 | itemsets (Blitz) | deterministic | "itemsets": 2841280 |
| L-691 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 46 | 119.3 | s (Blitz time) | hardware-dependent | "time_seconds": 119.3 |
| L-692 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 47 | 19 | max_k (Blitz) | deterministic | "max_k": 19 |
| L-693 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 50 | 21.4 | speedup_vs_son | hardware-dependent | "comparison": { "speedup_vs_son": 21.4 |
| L-694 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 51 | false | itemset_match | deterministic | "itemset_match": false |
| L-695 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 52 | 453019 | itemset_diff | deterministic | "itemset_diff": 453019 |
| L-696 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | (derived) | 475865 (sum of k_distribution K1..K14) | itemsets | deterministic | extractor check: sum of direct k_distribution equals itemsets=475865 |
| L-697 | `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | (derived) | 768.9 → ceil 769 | min_count implied by 1e-05 × 76890945 | deterministic | extractor check: 1e-05 × 76,890,945 = 768.909; file states min_count 768 |
| L-698 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 2 | null_model_permutation_test | experiment id | method-parameter | "experiment": "null_model_permutation_test" |
| L-699 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 3 | 2026-02-19T06:10:46.260818+00:00 | timestamp | software | "timestamp": "2026-02-19T06:10:46.260818+00:00" |
| L-700 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 5 | 1.0001177641918693e-05 | min_support (fraction) | method-parameter | "min_support": 1.0001177641918693e-05 |
| L-701 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 6 | 769 | min_count | method-parameter | "min_count": 769 |
| L-702 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 7 | 76890945 | n_transactions (proteins) | deterministic | "n_transactions": 76890945 |
| L-703 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 8 | 5 | n_permutations | method-parameter | "n_permutations": 5 |
| L-704 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 9 | 42 | seed | method-parameter | "seed": 42 |
| L-705 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 10 | false | recomputed_real | method-parameter | "recomputed_real": false |
| L-706 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 13 | 1002 | real itemsets at K=1 | deterministic | "real_distribution": … "1": 1002 |
| L-707 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 14 | 22019 | real itemsets at K=2 | deterministic | "real_distribution": … "2": 22019 |
| L-708 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 15 | 73205 | real itemsets at K=3 | deterministic | "real_distribution": … "3": 73205 |
| L-709 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 16 | 108059 | real itemsets at K=4 | deterministic | "real_distribution": … "4": 108059 |
| L-710 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 17 | 104239 | real itemsets at K=5 | deterministic | "real_distribution": … "5": 104239 |
| L-711 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 18 | 78596 | real itemsets at K=6 | deterministic | "real_distribution": … "6": 78596 |
| L-712 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 19 | 48699 | real itemsets at K=7 | deterministic | "real_distribution": … "7": 48699 |
| L-713 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 20 | 25011 | real itemsets at K=8 | deterministic | "real_distribution": … "8": 25011 |
| L-714 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 21 | 10508 | real itemsets at K=9 | deterministic | "real_distribution": … "9": 10508 |
| L-715 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 22 | 3488 | real itemsets at K=10 | deterministic | "real_distribution": … "10": 3488 |
| L-716 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 23 | 869 | real itemsets at K=11 | deterministic | "real_distribution": … "11": 869 |
| L-717 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 24 | 152 | real itemsets at K=12 | deterministic | "real_distribution": … "12": 152 |
| L-718 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 25 | 17 | real itemsets at K=13 | deterministic | "real_distribution": … "13": 17 |
| L-719 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 26 | 1 | real itemsets at K=14 | deterministic | "real_distribution": … "14": 1 |
| L-720 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 28 | 475865 | real_total | deterministic | "real_total": 475865 |
| L-721 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 32 | 171401 | total_itemsets (null run 1) | deterministic | "run": 1, "total_itemsets": 171401 |
| L-722 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 33 | 99.22 | s (null run 1 mining time) | hardware-dependent | "run": 1, "time_seconds": 99.22 |
| L-723 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 35 | K1=1002; K2=63737; K3=79166; K4=25482; K5=1994; K6=20 | k_distribution (null run 1) | deterministic | "run": 1, "k_distribution": {"1": 1002, "2": 63737, "3": 79166, "4": 25482, "5": 1994, "6": 20} |
| L-724 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 45 | 171240 | total_itemsets (null run 2) | deterministic | "run": 2, "total_itemsets": 171240 |
| L-725 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 46 | 135.62 | s (null run 2 mining time) | hardware-dependent | "run": 2, "time_seconds": 135.62 |
| L-726 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 48 | K1=1002; K2=63672; K3=79121; K4=25453; K5=1970; K6=22 | k_distribution (null run 2) | deterministic | "run": 2, "k_distribution": {"1": 1002, "2": 63672, "3": 79121, "4": 25453, "5": 1970, "6": 22} |
| L-727 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 58 | 171289 | total_itemsets (null run 3) | deterministic | "run": 3, "total_itemsets": 171289 |
| L-728 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 59 | 141.97 | s (null run 3 mining time) | hardware-dependent | "run": 3, "time_seconds": 141.97 |
| L-729 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 61 | K1=1002; K2=63750; K3=79080; K4=25438; K5=1996; K6=23 | k_distribution (null run 3) | deterministic | "run": 3, "k_distribution": {"1": 1002, "2": 63750, "3": 79080, "4": 25438, "5": 1996, "6": 23} |
| L-730 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 71 | 171395 | total_itemsets (null run 4) | deterministic | "run": 4, "total_itemsets": 171395 |
| L-731 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 72 | 143.06 | s (null run 4 mining time) | hardware-dependent | "run": 4, "time_seconds": 143.06 |
| L-732 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 74 | K1=1002; K2=63703; K3=79185; K4=25475; K5=2008; K6=22 | k_distribution (null run 4) | deterministic | "run": 4, "k_distribution": {"1": 1002, "2": 63703, "3": 79185, "4": 25475, "5": 2008, "6": 22} |
| L-733 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 84 | 171275 | total_itemsets (null run 5) | deterministic | "run": 5, "total_itemsets": 171275 |
| L-734 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 85 | 142.3 | s (null run 5 mining time) | hardware-dependent | "run": 5, "time_seconds": 142.3 |
| L-735 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 87 | K1=1002; K2=63650; K3=79120; K4=25491; K5=1990; K6=22 | k_distribution (null run 5) | deterministic | "run": 5, "k_distribution": {"1": 1002, "2": 63650, "3": 79120, "4": 25491, "5": 1990, "6": 22} |
| L-736 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 98 | real=1002; null_mean=1002.0; null_std=0.0; z_score=0.0; p_value=1.0; direction=depleted; significant=false | statistics K=1 | deterministic | "1": { "real": 1002, "null_mean": 1002.0, "null_std": 0.0, "z_score": 0.0, "p_value": 1.0 |
| L-737 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 107 | real=22019; null_mean=63702.4; null_std=42.2; z_score=-987.08; p_value=1.0; direction=depleted; significant=false | statistics K=2 | deterministic | "2": { "real": 22019, "null_mean": 63702.4, "null_std": 42.2, "z_score": -987.08, "p_value": 1.0 |
| L-738 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 116 | real=73205; null_mean=79134.4; null_std=41.5; z_score=-142.71; p_value=1.0; direction=depleted; significant=false | statistics K=3 | deterministic | "3": { "real": 73205, "null_mean": 79134.4, "null_std": 41.5, "z_score": -142.71, "p_value": 1.0 |
| L-739 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 125 | real=108059; null_mean=25467.8; null_std=21.8; z_score=3790.74; p_value=0.0; direction=enriched; significant=true | statistics K=4 | deterministic | "4": { "real": 108059, "null_mean": 25467.8, "null_std": 21.8, "z_score": 3790.74, "p_value": 0.0 |
| L-740 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 134 | real=104239; null_mean=1991.6; null_std=13.8; z_score=7402.24; p_value=0.0; direction=enriched; significant=true | statistics K=5 | deterministic | "5": { "real": 104239, "null_mean": 1991.6, "null_std": 13.8, "z_score": 7402.24, "p_value": 0.0 |
| L-741 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 143 | real=78596; null_mean=21.8; null_std=1.1; z_score=71728.1; p_value=0.0; direction=enriched; significant=true | statistics K=6 | deterministic | "6": { "real": 78596, "null_mean": 21.8, "null_std": 1.1, "z_score": 71728.1, "p_value": 0.0 |
| L-742 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 152 | real=48699; null_mean=0.0; null_std=0.0; z_score=inf; p_value=0.0; direction=enriched; significant="True" | statistics K=7 | deterministic | "7": { "real": 48699, "null_mean": 0.0, "null_std": 0.0, "z_score": inf, "p_value": 0.0 |
| L-743 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 161 | real=25011; null_mean=0.0; null_std=0.0; z_score=inf; p_value=0.0; direction=enriched; significant="True" | statistics K=8 | deterministic | "8": { "real": 25011, "null_mean": 0.0, "null_std": 0.0, "z_score": inf, "p_value": 0.0 |
| L-744 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 170 | real=10508; null_mean=0.0; null_std=0.0; z_score=inf; p_value=0.0; direction=enriched; significant="True" | statistics K=9 | deterministic | "9": { "real": 10508, "null_mean": 0.0, "null_std": 0.0, "z_score": inf, "p_value": 0.0 |
| L-745 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 179 | real=3488; null_mean=0.0; null_std=0.0; z_score=inf; p_value=0.0; direction=enriched; significant="True" | statistics K=10 | deterministic | "10": { "real": 3488, "null_mean": 0.0, "null_std": 0.0, "z_score": inf, "p_value": 0.0 |
| L-746 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 188 | real=869; null_mean=0.0; null_std=0.0; z_score=inf; p_value=0.0; direction=enriched; significant="True" | statistics K=11 | deterministic | "11": { "real": 869, "null_mean": 0.0, "null_std": 0.0, "z_score": inf, "p_value": 0.0 |
| L-747 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 197 | real=152; null_mean=0.0; null_std=0.0; z_score=inf; p_value=0.0; direction=enriched; significant="True" | statistics K=12 | deterministic | "12": { "real": 152, "null_mean": 0.0, "null_std": 0.0, "z_score": inf, "p_value": 0.0 |
| L-748 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 206 | real=17; null_mean=0.0; null_std=0.0; z_score=inf; p_value=0.0; direction=enriched; significant="True" | statistics K=13 | deterministic | "13": { "real": 17, "null_mean": 0.0, "null_std": 0.0, "z_score": inf, "p_value": 0.0 |
| L-749 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 215 | real=1; null_mean=0.0; null_std=0.0; z_score=inf; p_value=0.0; direction=enriched; significant="True" | statistics K=14 | deterministic | "14": { "real": 1, "null_mean": 0.0, "null_std": 0.0, "z_score": inf, "p_value": 0.0 |
| L-750 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 225 | 475865 | real_total (summary) | deterministic | "summary": { "real_total": 475865 |
| L-751 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 226 | 171320.0 | null_mean_total | deterministic | "null_mean_total": 171320.0 |
| L-752 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 227 | 2.78 | ratio_real_vs_null | deterministic | "ratio_real_vs_null": 2.78 |
| L-753 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 228 | 132.43 | s avg_null_mining_seconds | hardware-dependent | "avg_null_mining_seconds": 132.43 |
| L-754 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 229 | 662.17 | s total_experiment_seconds | hardware-dependent | "total_experiment_seconds": 662.17 |
| L-755 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | (derived) | 88745 | itemsets K>=7 (sum of real_distribution K7..K14) | deterministic | extractor check: 48699+25011+10508+3488+869+152+17+1 = 88,745 (notebook text says 89,566) |
| L-756 | `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | (derived) | 769.0 | min_count × 1/n (1.0001177641918693e-05 × 76890945) | deterministic | extractor check: min_support here equals 769/76,890,945 exactly |
| L-757 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | (file) | 41,665 lines; 4,749,057 bytes; 5,351 itemset blocks (K=15..19); fields per block: Support/Proteins/K, Item IDs, pLDDT features, Pfam domains, GO terms, Full decode | summary | deterministic | structured text dump; not transcribed row-by-row (see summary rows below) |
| L-758 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | (file) | 81 distinct item ids used (min 1, max 1002); pLDDT-feature line present in 4,329/5,351 blocks; Pfam line in 5,196/5,351 | summary | deterministic | extractor-computed over all 'Item IDs' lines |
| L-759 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 2 | K=15 to K=19; 214M TrEMBL | scope | method-parameter | DECODED HIGH-K ITEMSETS (K=15 to K=19) FROM 214M TrEMBL MINING |
| L-760 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 3 | 76,890,945 | proteins in dataset | deterministic | Total proteins in dataset: 76,890,945 |
| L-761 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 4 | 5,351 | itemsets K>=15 | deterministic | Total itemsets K>=15: 5,351 |
| L-762 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 8 | 1 | itemsets at K=19 | deterministic | K = 19  \|  1 itemsets |
| L-763 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 12 | support 0.000002; ~187 proteins; K 19 | K=19 itemset 1/1 | deterministic | Support: 0.000002  \|  Proteins: ~187  \|  K: 19 |
| L-764 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 13 | [1, 13, 23, 507, 508, 511, 517, 560, 594, 602, 604, 641, 677, 698, 724, 731, 794, 887, 906] | item ids (K=19 itemset) | deterministic | Item IDs: [1, 13, 23, 507, 508, 511, 517, 560, 594, 602, 604, 641, 677, 698, 724, 731, 794, 887, 906] |
| L-765 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 14 | plddt_mean_med | pLDDT feature (K=19 itemset) | deterministic | pLDDT features:  plddt_mean_med |
| L-766 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 15 | PF00271, PF00270 | Pfam (K=19 itemset) | deterministic | Pfam domains:    PF00271, PF00270 |
| L-767 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 16 | GO:0005524, GO:0046872, GO:0003677, GO:0016887, GO:1990904, GO:0005730, GO:0005654, GO:0006397, GO:0045944, GO:0003724, GO:0043138, GO:0005813, GO:0008380, … (16 GO terms) | GO terms (K=19 itemset) | deterministic | GO terms: GO:0005524 [go_term], GO:0046872 [go_term], GO:0003677 [go_term], GO:0016887 [go_term], … |
| L-768 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 20 | 19 | itemsets at K=18 | deterministic | K = 18  \|  19 itemsets |
| L-769 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 24 | support 0.000002; ~192 proteins | K=18 itemset 1/19 | deterministic | Support: 0.000002  \|  Proteins: ~192  \|  K: 18 |
| L-770 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 175 | 173 | itemsets at K=17 | deterministic | K = 17  \|  173 itemsets |
| L-771 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 179 | support 0.000008; ~611 proteins | K=17 itemset 1/173 | deterministic | Support: 0.000008  \|  Proteins: ~611  \|  K: 17 |
| L-772 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 1544 | 1003 | itemsets at K=16 | deterministic | K = 16  \|  1003 itemsets |
| L-773 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 9400 | 4155 | itemsets at K=15 | deterministic | K = 15  \|  4155 itemsets |
| L-774 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 41661 | K=19: 1 itemsets; support [0.000002, 0.000002]; proteins [187 - 187] | summary | deterministic | K=19:     1 itemsets \| support range [0.000002, 0.000002] \| proteins [187 - 187] |
| L-775 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 41662 | K=18: 19 itemsets; support [0.000002, 0.000002]; proteins [187 - 192] | summary | deterministic | K=18:    19 itemsets \| support range [0.000002, 0.000002] \| proteins [187 - 192] |
| L-776 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 41663 | K=17: 173 itemsets; support [0.000002, 0.000008]; proteins [187 - 611] | summary | deterministic | K=17:   173 itemsets \| support range [0.000002, 0.000008] \| proteins [187 - 611] |
| L-777 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 41664 | K=16: 1003 itemsets; support [0.000002, 0.000008]; proteins [187 - 612] | summary | deterministic | K=16:  1003 itemsets \| support range [0.000002, 0.000008] \| proteins [187 - 612] |
| L-778 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 41665 | K=15: 4155 itemsets; support [0.000001, 0.000009]; proteins [84 - 664] | summary | deterministic | K=15:  4155 itemsets \| support range [0.000001, 0.000009] \| proteins [84 - 664] |
| L-779 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | (derived) | 1+19+173+1003+4155 = 5351; per-K support/protein ranges recomputed from all 5,351 'Support:' lines match the SUMMARY block exactly | consistency | deterministic | extractor check |
| L-780 | `applications/alphafold/results_214m/decoded_top_k_patterns.txt` | (derived) | 84 proteins at K=15 minimum → support 84/76,890,945 = 1.09e-6 | min protein count (K=15) | deterministic | extractor check: lowest 'Proteins: ~84' in file (K=15) |
| L-781 | `applications/alphafold/results_214m/GLOSSARY.md` | 1 | 214M | proteins (title) | deterministic | # Biological Glossary - AlphaFold 214M Mining Results |
| L-782 | `applications/alphafold/results_214m/GLOSSARY.md` | 3 | 214 million | AlphaFold-predicted structures | deterministic | frequent itemset mining of 214 million AlphaFold-predicted protein structures. |
| L-783 | `applications/alphafold/results_214m/GLOSSARY.md` | 9 | K>=10; 2.84M | high-K threshold / itemsets (direct GPU) | deterministic | Domains listed here appear in high-K itemsets (K>=10) from the 2.84M direct GPU mining results. |
| L-784 | `applications/alphafold/results_214m/GLOSSARY.md` | 13 | ~34% | FDA-approved drugs targeting GPCRs | external-fact | ~34% of FDA-approved drugs target GPCRs. |
| L-785 | `applications/alphafold/results_214m/GLOSSARY.md` | 15 | ~100 aa; >100 human proteins | SH2 domain length / count | external-fact | Src Homology 2 domain (~100 aa) … Over 100 human proteins contain SH2 domains. |
| L-786 | `applications/alphafold/results_214m/GLOSSARY.md` | 16 | ~60 aa; ~300 | SH3 domain length / count in human proteome | external-fact | Src Homology 3 domain (~60 aa) … ~300 SH3 domains in the human proteome. |
| L-787 | `applications/alphafold/results_214m/GLOSSARY.md` | 18 | ~50 aa | C1_1 domain length | external-fact | Phorbol ester / diacylglycerol (DAG) binding domain (~50 aa). |
| L-788 | `applications/alphafold/results_214m/GLOSSARY.md` | 21 | 70 | human Dbl-family members | external-fact | 70 human Dbl-family members. |
| L-789 | `applications/alphafold/results_214m/GLOSSARY.md` | 140 | 2.84M frequent itemsets; K=1-19 | itemsets / K range | deterministic | *Generated from et-miner AlphaFold 214M protein mining results (2.84M frequent itemsets, K=1-19). |
| L-790 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | (file) | 29 cells (19 code, 10 markdown); 1 stored output (cell 1 stream); execution_count None on all code cells; kernel '.venv (3.10.12)' | summary | software | notebook stored without executed outputs except one print |
| L-791 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 0 (md) | 76.9M--109M | proteins | deterministic | GPU-accelerated frequent itemset mining on 76.9M--109M proteins |
| L-792 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 0 (md) | 26.8M itemsets; K=1--22 | God Mode campaign | deterministic | 1K-feature "God Mode" campaign (26.8M itemsets, K=1--22) |
| L-793 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 0 (md) | 16.8B itemsets; K=1--8 | 35K Alpha Centauri | deterministic | 35K-feature Alpha Centauri results (16.8B itemsets, K=1--8) |
| L-794 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 0 (md) | 76.9M (1K features) / 109.2M (35K features) | proteins | deterministic | UniProt TrEMBL + SwissProt, 76.9M proteins (1K features) / 109.2M proteins (35K features) |
| L-795 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 0 (md) | NVIDIA H100/H200 | GPU | hardware-dependent | **Hardware:** NVIDIA H100/H200 GPUs, CSR bitvector encoding, Apriori with popcount-based support counting |
| L-796 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 0 (md) | 7.3 minutes | God Mode mining time | hardware-dependent | 1K God Mode: 26.8M itemsets in 7.3 minutes (K=1--22) |
| L-797 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 0 (md) | 16,812,646,639; 12.07B at K=8; 67 GB | itemsets / size | deterministic | 35K Alpha Centauri: **16,812,646,639 itemsets** (K=1--8), including 12.07B K=8 itemsets (67 GB) |
| L-798 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 1 L40-41 (code) | archived/alphafold/results_214m/itemsets_214m_godmode.parquet; item_mapping_214m.parquet | input paths (NOT present in repo) | software | GODMODE_PATH = REPO_ROOT / "archived" / "alphafold" / "results_214m" / "itemsets_214m_godmode.parquet" |
| L-799 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 1 L47 (code) | 26.8M rows | God Mode itemsets | deterministic | print("Loading God Mode itemsets (26.8M rows)...") |
| L-800 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 1 L51 (code) | 1,006 | item-mapping features | deterministic | print("Loading item mapping (1,006 features)...") |
| L-801 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 1 L67 (code) | 76,890,945 | N_PROTEINS | deterministic | N_PROTEINS = son_data["parameters"]["n_transactions"]  # 76,890,945 |
| L-802 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 1 (output) | (stream) Loading God Mode itemsets (26.8M rows)... / Loading item mapping (1,006 features)... | only stored output | software | the sole output in the notebook |
| L-803 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 2 L16 (code) | K=22: 1 itemset, ~8 proteins | deepest itemset | deterministic | 22: "Deepest: 22-feature signature (1 itemset, ~8 proteins)" |
| L-804 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 2 L66 (code) | 76.9M proteins; 1,002 features (500 Pfam + 500 GO + 6 pLDDT); support >= 1e-7 | God Mode params | method-parameter | <sub>76.9M proteins, 1,002 features (500 Pfam + 500 GO + 6 pLDDT), support >= 1e-7</sub> |
| L-805 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 3 (md) | K=1--3: 527K; K=4--6: 5.8M; K=7--9: 10.1M (peak); K=10--14: 6.0M; K=15--22: 5.5K | itemsets per K range (God Mode) | deterministic | \| K=1--3 \| 527K \| … \| K=15--22 \| 5.5K \| … found in ~100--200 proteins each |
| L-806 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 3 (md) | K=9: 3.53M | peak K itemsets | deterministic | The **peak at K=9** (3.53M itemsets) |
| L-807 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 3 (md) | 0 at K>=7 | null-model itemsets | deterministic | the null model produces **zero** itemsets at K >= 7. |
| L-808 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 4 L70 (code) | 59.6% | rank-1 item support (share of proteins) | deterministic | text=f"Rank 1: support={supports[0]:.4f} (59.6% of proteins)" |
| L-809 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 5 (md) | dies at K=6; tail to K=22 | null vs real K reach | deterministic | a **fundamentally different** distribution that dies at K=6. The real data's long tail extending to K=22 |
| L-810 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 9 (md) | 16--17 of 19 features shared | K=19 itemsets overlap | deterministic | The K=19 itemsets are all minor variations of a single RNA helicase / spliceosome signature (sharing 16--17 of 19 features) |
| L-811 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 11 L15 (code) | 95.2% (453,019 of 475,865) | SON loss | deterministic | # The comparison shows SON lost 95.2% of itemsets (453,019 out of 475,865) |
| L-812 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 11 L25-26 (code) | SON 22,846 vs direct 475,865; SON max K 13, direct 14 | SON vs direct | deterministic | # SON approximation: it found 22,846 total vs 475,865 direct / # SON max K = 13, Direct max K = 14 |
| L-813 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 12 (md) | 475,865 itemsets in 50.7 s | direct GPU at 0.001% | hardware-dependent | **Direct GPU finds 475,865 itemsets** in 50.7 seconds |
| L-814 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 12 (md) | 22,846 in 1,085.6 s; 21.4x slower | SON at 0.001% | hardware-dependent | **SON finds only 22,846** in 1,085.6 seconds (21.4x slower!) |
| L-815 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 12 (md) | 95.2% | SON pattern loss | deterministic | SON **loses 95.2%** of all patterns, primarily the high-K discoveries |
| L-816 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 13 L110 (code) | Z=3.29 (p<0.001) | significance line | method-parameter | # Significance line at Z=3.29 (p < 0.001) |
| L-817 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 14 (md) | 1,002/1,002 | K=1 features real=null | deterministic | **K=1:** Identical by construction (1,002/1,002 features, sanity check) |
| L-818 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 14 (md) | Z = -987, -143 | K=2--3 z-scores | deterministic | **K=2--3:** Null **exceeds** real data (Z = -987, -143). |
| L-819 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 14 (md) | Z = 3,791 | K=4 z-score | deterministic | **K=4:** Crossover point. Real data starts exceeding null (Z = 3,791). |
| L-820 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 14 (md) | Z = 7,402 | K=5 z-score | deterministic | **K=5:** Strong enrichment (Z = 7,402). |
| L-821 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 14 (md) | Z = 71,728; null 21.8 vs real 78,596 | K=6 | deterministic | **K=6:** Extreme enrichment (Z = 71,728). Null barely reaches K=6 (21.8 itemsets vs 78,596 real). |
| L-822 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 14 (md) | 89,566 | real itemsets at K>=7 | deterministic | Every single one of the 89,566 real itemsets at K >= 7 represents genuine biological organization |
| L-823 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 15 (md) | 16,812,646,639; 109.2M proteins; 35,012 features | 35K campaign totals | deterministic | **Mining complete.** 16,812,646,639 frequent itemsets mined across K=1--8 from 109.2M proteins with 35,012 features. |
| L-824 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 15 (md) | 28,405 | 35K itemsets at K=1 (file size tiny) | deterministic | \| 1 \| 28,405 \| tiny \| |
| L-825 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 15 (md) | 5,506,372 | 35K itemsets at K=2 (file size tiny) | deterministic | \| 2 \| 5,506,372 \| tiny \| |
| L-826 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 15 (md) | 176,048,236 | 35K itemsets at K=3 (file size ~3 GB) | deterministic | \| 3 \| 176,048,236 \| ~3 GB \| |
| L-827 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 15 (md) | 1,506,508,703 | 35K itemsets at K=4 (file size ~8 GB) | deterministic | \| 4 \| 1,506,508,703 \| ~8 GB \| |
| L-828 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 15 (md) | 2,474,423,427 | 35K itemsets at K=5 (file size ~10 GB) | deterministic | \| 5 \| 2,474,423,427 \| ~10 GB \| |
| L-829 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 15 (md) | 626,781,137 | 35K itemsets at K=6 (file size ~4 GB) | deterministic | \| 6 \| 626,781,137 \| ~4 GB \| |
| L-830 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 15 (md) | 3,507,040,364 | 35K itemsets at K=7 (file size 16.6 GB) | deterministic | \| 7 \| 3,507,040,364 \| 16.6 GB \| |
| L-831 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 15 (md) | 12,072,309,005 | 35K itemsets at K=8 (file size 67 GB) | deterministic | \| 8 \| 12,072,309,005 \| 67 GB \| |
| L-832 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 17 L13-19 (code) | InterPro 12.7K; GO 5.8K; EC 1.1K; Keywords 15.2K; Taxonomy 175; Length 26; pLDDT 6 | 35K feature-group sizes | method-parameter | GROUP_LABELS = {"interpro": "InterPro (12.7K)", "go_term": "GO (5.8K)", "ec_number": "EC (1.1K)", "keyword": "Keywords (15.2K)", … |
| L-833 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 21 L94 (code) | 3_507_040_364 | K=7 known count (35K) | deterministic | "n_total": 3_507_040_364,  # known count |
| L-834 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 23 L9-16 (code) | 1: 28_405; 2: 5_506_372; 3: 176_048_236; 4: 1_506_508_703; 5: 2_474_423_427; 6: 626_781_137; 7: 3_507_040_364; 8: 12_072_309_005 | KNOWN_COUNTS (from mining logs) | deterministic | # --- Known counts (from mining logs) --- KNOWN_COUNTS = {…}  # 16,812,646,639 |
| L-835 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 25 (code) | K=2-6; support > 1e-5; min_confidence=0.5 (1K); top 10K K=4, min_confidence=0.3 (35K) | rule-mining params | method-parameter | [1K] Rule candidates … (K=2-6, support > 1e-5) … min_confidence=0.5 … Generating 35K rules (min_confidence=0.3) |
| L-836 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 28 (md) | 26.8M itemsets; 7.3 minutes; K=1--22; 76.9M proteins | God Mode scale | hardware-dependent | **Scale:** 26.8M itemsets mined in 7.3 minutes across K=1--22, from 76.9M proteins. |
| L-837 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 28 (md) | K=9: 3.53M | peak | deterministic | The K-distribution peaks at K=9 (3.53M itemsets) |
| L-838 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 28 (md) | 89,566 | patterns at K>=7 | deterministic | All 89,566 patterns at K >= 7 are **impossible** under the null hypothesis |
| L-839 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 28 (md) | 20.8x more itemsets; 21.4x less time | direct vs SON | hardware-dependent | Direct GPU mining finds 20.8x more itemsets than the SON approximation algorithm, in 21.4x less time. |
| L-840 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 28 (md) | K=19 in ~187 proteins; K=22 in ~8 proteins | deep patterns | deterministic | The K=19 sentinel … found in ~187 proteins. The K=22 apex is a single 22-feature combination found in ~8 proteins |
| L-841 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 28 (md) | 16.8B (35K) vs 26.8M (1K); K=8 12.07B, 67 GB, 71.8% | combinatorial explosion | deterministic | 35K features produced 16.8B itemsets at K=1--8, compared to 26.8M at 1K features. K=8 alone (12.07B itemsets, 67 GB) represents 71.8% |
| L-842 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 28 (md) | K=8/K=7 = 3.4x; K=5/K=4 = 1.6x; K=7/K=6 = 5.6x | growth rates | deterministic | K=8/K=7 = 3.4x growth. K=5/K=4 = 1.6x (plateau). K=7/K=6 = 5.6x (re-explosion). |
| L-843 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 28 (md) | 30--40B+ | projected K=9 itemsets | deterministic | The 3.4x growth from K=7 to K=8 suggests K=9 could yield 30--40B+ itemsets. |
| L-844 | `applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | cell 28 (md) | February 2026; Prof. Alexandre Bonvin, Utrecht University | date / reviewer | software | *Prepared for review by Prof. Alexandre Bonvin, Utrecht University* *ET-miner \| GPU-accelerated proteome mining \| February 2026* |
| L-845 | `PROGRESS.md` | 6 | 2026-09-02T00:02Z | goal timestamp | software | ## Goal (set 2026-09-02T00:02Z by Et via /goal) |
| L-846 | `PROGRESS.md` | 23 | 2026-09-02T00:04Z | run dir created | software | `runs/20260902T0000Z/` (created 2026-09-02T00:04Z; subdirs logs/, phase1/, phase2/, phase3/, phase4/) |
| L-847 | `PROGRESS.md` | 40 | 2 × NVIDIA GeForce RTX 3090, 24576 MiB, compute cap 8.6 | GPU (this box) | hardware-dependent | GPU: **2 × NVIDIA GeForce RTX 3090, 24576 MiB each, compute cap 8.6** |
| L-848 | `PROGRESS.md` | 41 | driver 595.71.05; CUDA runtime 13.2; nvcc 12.1 | software (this box) | software | Driver 595.71.05, CUDA runtime 13.2, `nvcc` 12.1. **This box is NOT an H100** |
| L-849 | `PROGRESS.md` | 44 | AMD EPYC 7402P, 24 cores / 24 threads, 1 socket, 1 NUMA node | CPU (this box) | hardware-dependent | CPU: AMD EPYC 7402P, 24 cores / 24 threads, 1 socket, 1 NUMA node. |
| L-850 | `PROGRESS.md` | 45 | 125 GiB host; cgroup limit 74,782,343,168 B (~69.6 GiB); ~1.3 GiB used; swap 8 GiB | RAM (this box) | hardware-dependent | RAM: host shows 125 GiB total; container cgroup limit `/sys/fs/cgroup/memory.max` = 74,782,343,168 B (~69.6 GiB) |
| L-851 | `PROGRESS.md` | 49 | 200 GB total, 200 GB available (939 MB used); inodes 195,351,424 total / 191,245,709 free | disk (this box) | hardware-dependent | Disk (container root, overlay): **200 GB total, 200 GB available** (939 MB used). Inodes: 195,351,424 total, 191,245,709 free. |
| L-852 | `PROGRESS.md` | 51 | 1.9 TB NVMe, 850 GB free (not mounted for data) | host disk | hardware-dependent | The host NVMe (/dev/nvme0n1, 1.9 TB, 850 GB free) is NOT mounted for data use |
| L-853 | `PROGRESS.md` | 53 | ~23 TiB; ~644M files | full AlphaFold v4 requirement | external-fact | Disk verdict vs. full AlphaFold v4 requirement (~23 TiB, ~644M files after untar) |
| L-854 | `PROGRESS.md` | 59 | gcloud SDK 583.0.0; bq 2.1.38 | software | software | (SDK 583.0.0, bq 2.1.38), authenticated as etje975@gmail.com. |
| L-855 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 4 | 214M | AlphaFold DB proteins | deterministic | de base-vocab run over de complete AlphaFold DB (214M), die Majors 1/2/3/5 uit `papers/peer_review_jul12.md` sluit |
| L-856 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 9 | 4×H200 (or 8×) | planned GPU box | hardware-dependent | **GPU-box**: 4×H200 (of 8×) op Vast.ai. |
| L-857 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 18 | base214m_20260712 | ET_UPLOAD_TAG | software | `ET_UPLOAD_TAG` staat op `base214m_20260712` |
| L-858 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 38 | 23TB | CIF corpus avoided | external-fact | De pLDDT-via-BigQuery-route (23TB CIF vermijden) matcht `af-extract build-from-metadata --plddt-csv ...` |
| L-859 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 39 | 6 | pLDDT bins | method-parameter | Beide geven de 6 pLDDT-bins bij `include_plddt=true`. |
| L-860 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 44 | 100× | row-split NCCL hit count (null@8) | method-parameter | **Row-split NCCL wordt 100× geraakt door de null@8** → `validate_row_split.py` is de verplichte gate. |
| L-861 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 70 | 300000000 | bq --max_rows | method-parameter | bq query --use_legacy_sql=false --format=csv --max_rows=300000000 |
| L-862 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 76 | 1006 defined / 1002 frequent | base vocab size | method-parameter | ### Fase 1 — Extractie (base vocab 1006 defined / 1002 frequent) |
| L-863 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 82 | --top-pfam 500 --top-go 500 | extraction params | method-parameter | --top-pfam 500 --top-go 500 \ |
| L-864 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 87 | v1: 76.9M of 205.6M | multi-feature proteins (paper v1) | deterministic | Dit zijn de nieuwe Table 1 / abstract-getallen (v1 was 76,9M van 205,6M). |
| L-865 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 93 | --subset-size 1000000 --min-count 50 --n-gpus 2 | row-split validation params | method-parameter | --subset-size 1000000 --min-count 50 --n-gpus 2 -v |
| L-866 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 95 | N_GPUS=4 (green) / N_GPUS=1 (red) | GPU count decision | method-parameter | GREEN (exit 0) → draai de suite met `N_GPUS=4`. RED (exit 1) → `N_GPUS=1` fallback |
| L-867 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 104 | 6×3 campaign; null@769 100-perm | experiment suite | method-parameter | Draait: (1) mining-campagne 6×3, (2) Direct-vs-SON, (3) null@769 100-perm, |
| L-868 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 105 | null@8 100-perm | critical experiment | method-parameter | (4) **null@8 100-perm** (de kritische Major 1+2), (5) deepest-itemset accessions. |
| L-869 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 152 | 4×H200; ~85M multi-feature | compute plan | hardware-dependent | ## Compute & timing (4×H200, ~85M multi-feature) |
| L-870 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 153 | data ~0.5-1h; extraction ~0.5-1h; campaign+Direct/SON ~1h; null@769 ~25min; null@8 ~1.5-2h; closed/maximal+accessions ~0.5h | planned durations | hardware-dependent | Data ~0,5-1u · extractie ~0,5-1u · campagne+Direct/SON ~1u · null@769 ~25min · **null@8 ~1,5-2u** · closed/maximal+accessions ~0,5u |
| L-871 | `applications/alphafold/deploy/RUNBOOK_base214m.md` | 154 | ~5-6 h wall-clock; ~$45-90 | planned total | hardware-dependent | → **~5-6u wall-clock**, ~$45-90. |
| L-872 | `applications/alphafold/deploy/run_all_experiments.sh` | 2 | 4×H200 | planned GPUs | hardware-dependent | # Phase 3: 4×H200 Full Experiment Suite |
| L-873 | `applications/alphafold/deploy/run_all_experiments.sh` | 6 | 6 thresholds × 3 runs | campaign design | method-parameter | #   1. Full mining campaign (6 thresholds × 3 runs) |
| L-874 | `applications/alphafold/deploy/run_all_experiments.sh` | 7 | 3 runs each | Direct vs SON design | method-parameter | #   2. Direct GPU vs SON controlled comparison (3 runs each) |
| L-875 | `applications/alphafold/deploy/run_all_experiments.sh` | 8 | min_count=769; 100 permutations; 4 GPUs | null model A | method-parameter | #   3. Null model at min_count=769 (100 permutations, 4 GPUs) |
| L-876 | `applications/alphafold/deploy/run_all_experiments.sh` | 9 | min_count=8; 100 permutations; 4 GPUs | null model B | method-parameter | #   4. Null model at min_count=8 (100 permutations, 4 GPUs) — THE BIG ONE |
| L-877 | `applications/alphafold/deploy/run_all_experiments.sh` | 10 | K=22 | deepest itemset analysis | deterministic | #   5. K=22 protein identification + GO hierarchy analysis |
| L-878 | `applications/alphafold/deploy/run_all_experiments.sh` | 16 | ~4 hours; ~$9/hr | expected runtime / cost (4×H200) | hardware-dependent | # Expected runtime on 4×H200 (~$9/hr): ~4 hours total |
| L-879 | `applications/alphafold/deploy/run_all_experiments.sh` | 17 | 141GB VRAM (564GB total); 4.8 TB/s; 1.4× H100 | H200 spec | external-fact | # H200 has 141GB VRAM (564GB total) and 4.8 TB/s bandwidth (1.4× H100) |
| L-880 | `applications/alphafold/deploy/run_all_experiments.sh` | 26 | /workspace/data/transactions_214m.parquet | default data path | software | DATA="${1:-/workspace/data/transactions_214m.parquet}" |
| L-881 | `applications/alphafold/deploy/run_all_experiments.sh` | 28 | 4 | N_GPUS default | method-parameter | N_GPUS="${N_GPUS:-4}" |
| L-882 | `applications/alphafold/deploy/run_all_experiments.sh` | 82 | ~25 min | estimated null@769 | hardware-dependent | echo "Estimated: ~25 min (fast at high threshold)" |
| L-883 | `applications/alphafold/deploy/run_all_experiments.sh` | 98 | ~100 min (25 batches × ~4 min/perm on H200) | estimated null@8 | hardware-dependent | echo "Estimated: ~100 min (25 batches × ~4 min/perm on H200)" |
| L-884 | `applications/alphafold/deploy/run_all_experiments.sh` | 102 | --min-count 8; --runs 100 | null@8 args | method-parameter | --min-count 8 \ --runs 100 \ --n-gpus "$N_GPUS" \ --perm-per-gpu |
| L-885 | `applications/alphafold/deploy/run_null_model_35k.sh` | 9 | 5 permutations; 0.001% support (min_count=1092) | defaults (35K) | method-parameter | # Defaults: 5 permutations, 0.001% support (min_count=1092) |
| L-886 | `applications/alphafold/deploy/run_null_model_35k.sh` | 10 | ~30-60 min per permutation on 8× H200 | expected runtime | hardware-dependent | # Expected runtime: ~30-60 min per permutation on 8× H200 |
| L-887 | `applications/alphafold/deploy/run_null_model_35k.sh` | 15 | 0.00001 | SUPPORT default | method-parameter | SUPPORT="${2:-0.00001}" |
| L-888 | `applications/alphafold/deploy/run_null_model_35k.sh` | 50 | 42 | seed | method-parameter | --seed 42 \ |
| L-889 | `applications/alphafold/deploy/deploy_project_milky_way.sh` | 128 | ~2.3 GB | transactions_35k.parquet size | deterministic | # 35K-feature transactions (the main payload, ~2.3 GB) |
| L-890 | `applications/alphafold/deploy/deploy_project_milky_way.sh` | 357 | ~478 GB | bitvec size (35K, row-split across N×H200) | deterministic | echo "  This verifies the ~478 GB bitvec fits across N×H200 with row-split." |
| L-891 | `applications/alphafold/deploy/deploy_project_milky_way.sh` | 427 | ~1TB | /dev/shm on vast.ai | hardware-dependent | # /dev/shm is pre-mounted tmpfs on vast.ai (~1TB) |
| L-892 | `applications/alphafold/deploy/deploy_project_milky_way.sh` | 456 | --runs 100 --min-count 1090 --n-gpus 8 --seed 42 | null model (35K) args | method-parameter | echo "      --runs 100 --min-count 1090 --n-gpus 8 --seed 42 \\" |
| L-893 | `applications/alphafold/deploy/deploy_project_milky_way.sh` | 472 | --support 0.00001 --max-length 8 --use-gpu --n-gpus 8 | mining (35K) args | method-parameter | echo "        --support 0.00001 --max-length 8 --use-gpu --n-gpus 8 \\" |
| L-894 | `applications/alphafold/deploy/deploy_base214m.sh` | 3 | H200 box | target | hardware-dependent | #  BASE-214M DEPLOYMENT — H200 box, base-vocab regeneration |
| L-895 | `applications/alphafold/deploy/deploy_base214m.sh` | 260 | 4 | N_GPUS hint default | method-parameter | N_GPUS_HINT="${N_GPUS:-4}" |
| L-896 | `applications/alphafold/deploy/deploy_base214m.sh` | 279 | 23TB | CIF corpus avoided | external-fact | # pLDDT from BigQuery (avoids the 23TB CIF) |
| L-897 | `applications/alphafold/deploy/deploy_base214m.sh` | 284 | 1006 defined (top-500 Pfam + top-500 GO + 6 pLDDT) | base vocab | method-parameter | # (B) Extraction — base vocab 1006 defined (top-500 Pfam + top-500 GO + 6 pLDDT): |
| L-898 | `applications/alphafold/pipeline/postprocess_tx.py` | 5 | 118K features at min_count=8 | raw 35K TSV | deterministic | Reads the raw TSV from build_tx (118K features at min_count=8), applies a higher |
| L-899 | `applications/alphafold/pipeline/postprocess_tx.py` | 183 | 3423 | --min-count default | method-parameter | parser.add_argument("--min-count", type=int, default=3423, |
| L-900 | `applications/alphafold/pipeline/pipeline_214m.py` | 290 | top_pfam=200; top_go=200; min_plddt=50.0 | run_extract defaults | method-parameter | def run_extract(data_dir: Path, top_pfam: int = 200, top_go: int = 200, … min_plddt: float = 50.0): |
| L-901 | `applications/alphafold/pipeline/extract_features.py` | 187 | pLDDT >90 → 1.0; >70 → 0.5; else 0.1 | frac_high bins | method-parameter | frac_high = 1.0 if global_plddt > 90 else (0.5 if global_plddt > 70 else 0.1) |
| L-902 | `applications/alphafold/pipeline/extract_features.py` | 194 | mean_plddt < 50 / <= 90 | plddt_mean bin edges | method-parameter | if mean_plddt < 50: … elif mean_plddt <= 90: |
| L-903 | `applications/alphafold/experiments/experiment_direct_vs_son.py` | 41 | 3 | --runs default | method-parameter | "--runs", type=int, default=3, |
| L-904 | `applications/alphafold/experiments/experiment_direct_vs_son.py` | 45 | 0.00001 | --min-support default | method-parameter | "--min-support", type=float, default=0.00001, |
| L-905 | `applications/alphafold/experiments/experiment_direct_vs_son.py` | 49 | 40,000,000 | --chunk-size default (SON) | method-parameter | "--chunk-size", type=int, default=40_000_000, |
| L-906 | `applications/alphafold/experiments/experiment_direct_vs_son.py` | 53 | 0.9 | --local-support-factor default (SON) | method-parameter | "--local-support-factor", type=float, default=0.9, |
| L-907 | `applications/alphafold/experiments/experiment_full_campaign.py` | 85 | 3 | --runs default | method-parameter | "--runs", type=int, default=3, |
| L-908 | `applications/alphafold/experiments/experiment_full_campaign.py` | 115 | min_count = ceil(min_support × n_transactions) | threshold rule | method-parameter | min_count = math.ceil(min_support * n_transactions) |
| L-909 | `applications/alphafold/pipeline/run_mining.py` | 193 | 0.01 | --support default | method-parameter | parser.add_argument("--support", type=float, default=0.01, |
| L-910 | `applications/alphafold/pipeline/run_mining.py` | 195 | 4 | --max-length default | method-parameter | parser.add_argument("--max-length", type=int, default=4, |
| L-911 | `applications/alphafold/pipeline/run_mining.py` | 203 | 5 | --baseline-runs default | method-parameter | parser.add_argument("--baseline-runs", type=int, default=5, |
| L-912 | `applications/alphafold/experiments/analyze_k22_proteins.py` | 62 | pLDDT 70-90 | plddt_mean_medium label | method-parameter | ("Struct", "plddt_mean_medium", "pLDDT 70-90"), |
| L-913 | `README.md` | 13 | 80--110x | Rust tier speedup | hardware-dependent | SIMD-vectorized CSR support counting (AVX2/AVX-512). 80--110x speedup. |
| L-914 | `README.md` | 203 | ~264 bytes across 22 levels | PCIe transfer | deterministic | GPU-resident mining: zero PCIe transfers between K-levels (~264 bytes total across 22 levels) |
| L-915 | `README.md` | 204 | n/32 | dense→sparse crossover | method-parameter | switches from dense bitvectors to sparse CSR tidsets when tidsets become the smaller representation (mean support < n/32) |
| L-916 | `README.md` | 205 | 8x H200 | max tested GPUs | hardware-dependent | Multi-GPU support with per-device work distribution (tested up to 8x H200) |
| L-917 | `README.md` | 209 | 26.8 million patterns; ~76M structures; K=22; 7.3 minutes; single H100 | AlphaFold headline | hardware-dependent | ET-Miner discovered **26.8 million co-occurrence patterns** across **~76M predicted protein structures**, reaching feature combinations of size K=22 in 7.3 minutes on a single H100 GPU. |
| L-918 | `README.md` | 211 | over 200 million | AlphaFold DB proteins | external-fact | The AlphaFold Database contains predicted protein structures for over 200 million proteins. |
| L-919 | `README.md` | 213 | ~5 GB CSR; ~26 GB bitvectors | memory footprint | deterministic | ET-Miner constructs a CSR representation directly from transactions (~5 GB), converts to GPU-resident bitvectors (~26 GB) |
| L-920 | `README.md` | 219 | 214M total; 76.9M with multiple annotations | proteins processed | deterministic | \| Proteins processed \| 214M total, 76.9M with multiple annotations \| |
| L-921 | `README.md` | 220 | 1,002 | feature vocabulary | method-parameter | \| Feature vocabulary \| 1,002 items (Pfam domains, GO terms, pLDDT bins) \| |
| L-922 | `README.md` | 221 | 26.8 million | itemsets discovered | deterministic | \| Itemsets discovered \| 26.8 million \| |
| L-923 | `README.md` | 222 | 22 | maximum K | deterministic | \| Maximum K \| 22 (mathematically proven ceiling) \| |
| L-924 | `README.md` | 223 | 7.3 minutes; single H100 | mining time (deepest tier) | hardware-dependent | \| Mining time (deepest tier) \| 7.3 minutes on single H100 \| |
| L-925 | `README.md` | 224 | 0.1% → 0.00001% | support range | method-parameter | \| Support range \| 0.1% down to 0.00001% \| |
| L-926 | `README.md` | 236 | 1,000,000,000 | transactions (billion-scale streaming) | method-parameter | \| Transactions \| 1,000,000,000 \| |
| L-927 | `README.md` | 237 | 25.9 minutes | time | hardware-dependent | \| Time \| 25.9 minutes \| |
| L-928 | `README.md` | 238 | 643,139 tx/sec | throughput | hardware-dependent | \| Throughput \| 643,139 tx/sec \| |
| L-929 | `README.md` | 239 | 14.76 GB | peak memory | hardware-dependent | \| Peak memory \| 14.76 GB \| |
| L-930 | `README.md` | 240 | 326 | itemsets found | deterministic | \| Itemsets found \| 326 \| |
| L-931 | `README.md` | 241 | Intel Core Ultra 9 275HX (24 cores), 134 GB RAM | hardware | hardware-dependent | \| Hardware \| Intel Core Ultra 9 275HX (24 cores), 134 GB RAM \| |
| L-932 | `README.md` | 243 | 819K transactions | efficient-apriori comparison dataset | method-parameter | ### vs. efficient-apriori (819K transactions) |
| L-933 | `README.md` | 245 | AMD Ryzen 5 4600G (12 cores), 30 GB RAM, CPython 3.14 free-threading, Polars 1.37, MKL sparse | system | hardware-dependent | > System: AMD Ryzen 5 4600G (12 cores), 30 GB RAM, CPython 3.14 free-threading build, Polars 1.37, MKL sparse enabled |
| L-934 | `README.md` | 249 | 0.005; 9; 1.21s / 641 MB; 0.24s / 329 MB; 5.0x | support; itemsets; efficient-apriori s/MB; et-miner s/MB; speedup (819K tx) | hardware-dependent | \| 0.005 \| 9 \| 1.21s / 641 MB \| 0.24s / 329 MB \| 5.0x \| |
| L-935 | `README.md` | 250 | 0.001; 326; 3.5s / 644 MB; 2.15s / 475 MB; 1.6x | support; itemsets; efficient-apriori s/MB; et-miner s/MB; speedup (819K tx) | hardware-dependent | \| 0.001 \| 326 \| 3.5s / 644 MB \| 2.15s / 475 MB \| 1.6x \| |
| L-936 | `README.md` | 251 | 0.0005; 1,151; 16.0s / 691 MB; 6.2s / 1113 MB; 2.6x | support; itemsets; efficient-apriori s/MB; et-miner s/MB; speedup (819K tx) | hardware-dependent | \| 0.0005 \| 1,151 \| 16.0s / 691 MB \| 6.2s / 1113 MB \| 2.6x \| |
| L-937 | `README.md` | 252 | 0.0001; 11,159; 214.2s / 2693 MB; 179.9s / 9186 MB; 1.2x | support; itemsets; efficient-apriori s/MB; et-miner s/MB; speedup (819K tx) | hardware-dependent | \| 0.0001 \| 11,159 \| 214.2s / 2693 MB \| 179.9s / 9186 MB \| 1.2x \| |
| L-938 | `README.md` | 254 | 2.5M transactions | efficient-apriori comparison dataset | method-parameter | ### vs. efficient-apriori (2.5M transactions) |
| L-939 | `README.md` | 258 | 0.005; 9; 3.9s / 1663 MB; 0.37s / 539 MB; 10.5x | support; itemsets; efficient-apriori s/MB; et-miner s/MB; speedup (2.5M tx) | hardware-dependent | \| 0.005 \| 9 \| 3.9s / 1663 MB \| 0.37s / 539 MB \| 10.5x \| |
| L-940 | `README.md` | 259 | 0.001; 336; 12.0s / 1671 MB; 3.5s / 1476 MB; 3.4x | support; itemsets; efficient-apriori s/MB; et-miner s/MB; speedup (2.5M tx) | hardware-dependent | \| 0.001 \| 336 \| 12.0s / 1671 MB \| 3.5s / 1476 MB \| 3.4x \| |
| L-941 | `README.md` | 260 | 0.0005; 1,153; 53.7s / 1704 MB; 12.1s / 2369 MB; 4.4x | support; itemsets; efficient-apriori s/MB; et-miner s/MB; speedup (2.5M tx) | hardware-dependent | \| 0.0005 \| 1,153 \| 53.7s / 1704 MB \| 12.1s / 2369 MB \| 4.4x \| |
| L-942 | `README.md` | 261 | 0.0001; 10,894; 589.4s / 3758 MB; 262.7s / 12163 MB; 2.2x | support; itemsets; efficient-apriori s/MB; et-miner s/MB; speedup (2.5M tx) | hardware-dependent | \| 0.0001 \| 10,894 \| 589.4s / 3758 MB \| 262.7s / 12163 MB \| 2.2x \| |
| L-943 | `bench/README.md` | 3 | 2× RTX 3090, 24 GB, CUDA 12 | design target box | hardware-dependent | Scripted campaign for a rented multi-GPU box (designed for 2× RTX 3090, 24 GB, CUDA 12). |
| L-944 | `bench/README.md` | 15 | ≥ 2 GB | --shm-size | method-parameter | **`--shm-size` ≥ 2 GB** (vast.ai: the "docker options"/shm setting). |
| L-945 | `bench/README.md` | 20 | Disk ≥ 40 GB; host RAM ≥ 32 GB | box requirements | method-parameter | **Disk ≥ 40 GB** (datasets + wheels + rust build), **host RAM ≥ 32 GB** |
| L-946 | `bench/README.md` | 30 | ~10 min | setup_box.sh duration | hardware-dependent | bash bench/setup_box.sh          # env + rust ext + selfcheck + datasets (~10 min) |
| L-947 | `bench/README.md` | 31 | ~30-60 min | run_smoke.sh duration | hardware-dependent | bash bench/run_smoke.sh          # ~30-60 min: tier gate → gpu tests → small matrix |
| L-948 | `bench/README.md` | 33 | ~2-4 h | run_full.sh duration | hardware-dependent | bash bench/run_full.sh           # ~2-4 h: gate → full gpu tests → full matrix |

## SECTION B — file inventory

### B.1 Surviving log-like / result-like files in the repo (all extracted above)

| path | size (B) | mtime (UTC) | type | related to base214m? | one-line description |
|---|---|---|---|---|---|
| `/root/projects/ET-Miner/bench/results/2026-08-31-3090x2/env.txt` | 2194 | 2026-09-01 22:47 | env capture | no (bench, 2×3090) | git HEAD 5a8e59f8; nvidia-smi 2×RTX 3090 driver 580.159.03 CUDA 13.0; pip freeze section empty |
| `/root/projects/ET-Miner/bench/results/2026-08-31-3090x2/report.md` | 3192 | 2026-09-01 22:47 | md report | no (bench, 2×3090) | GPU campaign report: 28 runs ok; wall-time/VRAM per config; kernel A/B; per-level tables |
| `/root/projects/ET-Miner/bench/results/2026-08-31-3090x2/FINDINGS.md` | 4634 | 2026-09-01 22:47 | md report | no (bench, 2×3090) | Findings: tier-equivalence 7/7; 114 GPU tests; shared kernel 8.8×/16.3×; prefilter drops 9,285 itemsets; density-auto 10.4 s |
| `/root/projects/ET-Miner/bench/results/2026-08-31-3090x2/raw.jsonl` | 33942 | 2026-09-01 22:47 | jsonl (28 records) | no (bench, 2×3090) | one JSON record per run: config, wall_s, levels[k,n_candidates,n_frequent,ms], peak_vram_mb, throttle, n_itemsets, sum_counts, itemset_hash |
| `/root/projects/ET-Miner/bench/results/2026-09-01-3090x2-sparse/env.txt` | 2194 | 2026-09-01 22:47 | env capture | no (bench, 2×3090) | git HEAD b8a5032b; nvidia-smi Tue Sep 1 20:09:15 2026; same driver; pip freeze empty |
| `/root/projects/ET-Miner/bench/results/2026-09-01-3090x2-sparse/report.md` | 1551 | 2026-09-01 22:47 | md report | no (bench, 2×3090) | 13 runs ok; deep_k configs only; A/B 0.99×; per-level table |
| `/root/projects/ET-Miner/bench/results/2026-09-01-3090x2-sparse/FINDINGS.md` | 4375 | 2026-09-01 22:47 | md report | no (bench, 2×3090) | Sparse-CSR follow-up: chain 9/9; signature n_itemsets=8841 sum_counts=266261275; density-auto 1.343 s |
| `/root/projects/ET-Miner/bench/results/2026-09-01-3090x2-sparse/raw.jsonl` | 18932 | 2026-09-01 22:47 | jsonl (13 records) | no (bench, 2×3090) | same schema as above, deep_k only, includes density-auto 1g/2g |
| `/root/projects/ET-Miner/applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json` | 1110 | 2026-09-01 22:48 | json result | YES (AlphaFold 1K vocab, 76.9M tx) | Direct GPU vs SON at 0.001% (min_count 768): 475,865 vs 22,846 itemsets; 50.72 s vs 1085.6 s; Blitz 2,841,280 @0.0001% |
| `/root/projects/ET-Miner/applications/alphafold/results_214m/experiment_null_model_20260219_061046.json` | 4676 | 2026-09-01 22:48 | json result | YES (AlphaFold 1K vocab, 76.9M tx) | Null-model permutation test, 5 perms, seed 42, min_count 769: real 475,865 vs null mean 171,320; per-K z-scores; K>=7 null = 0 |
| `/root/projects/ET-Miner/applications/alphafold/results_214m/decoded_top_k_patterns.txt` | 4749057 | 2026-09-01 22:48 | txt decoded itemsets | YES (AlphaFold; K=15..19 subset) | 5,351 decoded itemsets K=15..19 from '214M TrEMBL mining'; 76,890,945 proteins; per-itemset support/proteins/Pfam/GO |
| `/root/projects/ET-Miner/applications/alphafold/results_214m/GLOSSARY.md` | 17193 | 2026-09-01 22:48 | md glossary | YES (AlphaFold; narrative) | Pfam/GO glossary; states 214 million structures, 2.84M itemsets K=1-19, K>=10 subset |
| `/root/projects/ET-Miner/applications/alphafold/experiments/analysis_alpha_centauri.ipynb` | 111866 | 2026-09-01 22:48 | ipynb (29 cells, 1 output) | YES (AlphaFold 1K + 35K narratives) | Analysis notebook; hard-coded totals (26.8M K=1-22 God Mode; 16.8B K=1-8 35K); references missing archived/ parquets |
| `/root/projects/ET-Miner/applications/alphafold/deploy/RUNBOOK_base214m.md` | 7790 | 2026-09-01 22:48 | md runbook (Dutch) | YES (base214m plan) | Runbook for the base-vocab 214M run on 4×H200: vocab 1006/1002, top-500 Pfam/GO, null@769 & null@8 100-perm, v1 = 76.9M of 205.6M |
| `/root/projects/ET-Miner/applications/alphafold/deploy/run_all_experiments.sh` | 4877 | 2026-09-01 22:48 | bash | YES (base214m suite) | Experiment suite: 6×3 campaign, direct-vs-SON ×3, null@769 & null@8 (100 perms), K=22 analysis; H200 specs/estimates |
| `/root/projects/ET-Miner/applications/alphafold/deploy/deploy_base214m.sh` | 12910 | 2026-09-01 22:48 | bash | YES (base214m deploy) | Deploy script; base vocab 1006 = top-500 Pfam + top-500 GO + 6 pLDDT; N_GPUS default 4 |
| `/root/projects/ET-Miner/applications/alphafold/deploy/run_null_model_35k.sh` | 1702 | 2026-09-01 22:48 | bash | no (35K vocab) | 35K null model defaults: 5 perms, support 0.00001 (min_count=1092), seed 42 |
| `/root/projects/ET-Miner/applications/alphafold/deploy/deploy_project_milky_way.sh` | 17061 | 2026-09-01 22:48 | bash | no (35K vocab) | 35K deploy; ~2.3 GB transactions; ~478 GB bitvec; null --min-count 1090 --runs 100 --n-gpus 8 |
| `/root/projects/ET-Miner/PROGRESS.md` | 7135 | 2026-09-02 00:19 | md (audit state, created 2026-09-02) | meta (this audit) | Phase-0 hardware audit of THIS box (2×3090, driver 595.71.05, 200 GB disk) — not an original experiment log |
| `/root/projects/ET-Miner/README.md` | 12306 | 2026-09-01 22:47 | md | YES (headline numbers) | AlphaFold results table (214M/76.9M, 1,002 items, 26.8M itemsets, K=22, 7.3 min H100) + CPU benchmark tables |
| `/root/projects/ET-Miner/bench/README.md` | 3530 | 2026-09-01 22:47 | md | no | Campaign harness instructions; durations/requirements only |

Note: every repo file's mtime is the `git clone`/checkout time (2026-09-01 22:47–22:48 UTC); the meaningful provenance is the git commit: `results_214m/*`, the notebook, deploy/, pipeline/, experiments/ and paper/ were all added in ONE commit `65d9098` ("add alphafold experiment scripts", 2026-08-31 21:46:33 +0200, author Et9797); bench/results/2026-08-31-3090x2 in `6f789d8` (2026-08-31 22:39 UTC); bench/results/2026-09-01-3090x2-sparse in `b1f147e` (2026-09-01 20:11 UTC).

### B.2 Inventoried but NOT extracted (paper sources — other subagents)

| path | size (B) | mtime (UTC) | type | related to base214m? | one-line description |
|---|---|---|---|---|---|
| `/root/projects/ET-Miner/paper/et_miner_proteome.tex` | 65595 | 2026-09-01 22:48 | md/tex | yes (claims) | V1 paper LaTeX source (claims source) |
| `/root/projects/ET-Miner/paper/PAPER_V2_REVIEW.md` | 19720 | 2026-09-01 22:48 | md/tex | yes (claims) | review (14 mentions of base214m/214M) |
| `/root/projects/ET-Miner/paper/peer_review_jul12.md` | 27049 | 2026-09-01 22:48 | md/tex | yes (claims) | peer review |
| `/root/projects/ET-Miner/paper/review_b1_hostile.md` | 29488 | 2026-09-01 22:48 | md/tex | yes (claims) | review |
| `/root/projects/ET-Miner/paper/review_b2_results.md` | 16203 | 2026-09-01 22:48 | md/tex | yes (claims) | results review (10 mentions of 214M) |
| `/root/projects/ET-Miner/paper/revision_notes_b3.tex` | 32374 | 2026-09-01 22:48 | md/tex | yes (claims) | revision notes |
| `/root/projects/ET-Miner/paper/senior_review_jun01.md` | 15269 | 2026-09-01 22:48 | md/tex | yes (claims) | review |
| `/root/projects/ET-Miner/paper/senior_review_mar23.md` | 5493 | 2026-09-01 22:48 | md/tex | yes (claims) | review |
| `/root/projects/ET-Miner/CLAUDE.md` | 7282 | 2026-09-01 23:57 | md/tex | yes (claims) | correctness policy; no result numbers |

### B.3 Files found in /root (home) and elsewhere on the box — none are experiment logs

| path | size (B) | mtime (UTC) | type | related to base214m? | one-line description |
|---|---|---|---|---|---|
| `/root/.bash_history` | 1152 | 2026-09-01 23:29 | shell history | no | install of gh/gcloud, git clone of ET-Miner, checkout of branch; no data downloads or mining commands |
| `/root/.claude/history.jsonl` | 628 | 2026-09-01 23:39 | claude prompt history | no | 4 slash-command entries from this audit session |
| `/root/.claude/projects/-root-projects-ET-Miner/1bae0ea5-20f3-44f0-a8ba-07a290957c5b.jsonl` | 407052 | 2026-09-02 00:19 | Claude Code session transcript | no (this audit) | transcript of the current audit session — excluded from extraction (would be circular) |
| `/root/.claude/projects/-root-projects-ET-Miner/1bae0ea5-20f3-44f0-a8ba-07a290957c5b/subagents/` | 4096 | 2026-09-02 00:06 | dir: 5 subagent transcripts (*.jsonl) | no (this audit) | parallel Phase-1 subagent transcripts — excluded |
| `/root/.config/gcloud/logs/2026.09.01/` | 4096 | 2026-09-01 23:58 | dir: 7 gcloud debug logs | no (this audit) | gcloud components/init/auth logs from 2026-09-01 23:29–23:58; CONTAIN OAUTH TOKENS — do not copy |
| `/root/.config/gcloud/logs/2026.09.02/` | 4096 | 2026-09-02 00:07 | dir: 4 gcloud debug logs | no (this audit) | `gcloud storage ls gs://public-datasets-deepmind-alphafold-v4/{,proteomes/proteome-tax_id-9606-*,metadata/}` at 00:06 UTC (Phase-2 bucket probe) |
| `/root/projects/downloads/google-cloud-cli-linux-x86_64.tar.gz` | 86740170 | 2026-09-01 23:28 | tarball | no | gcloud SDK download (86.7 MB) |
| `/workspace/ports.log` | 6 | 2026-09-01 22:19 | log | no | contains only '26272' (vast.ai port) |
| `/workspace/onstart.sh` | 82 | 2026-09-01 22:19 | sh | no | vast.ai instance start hook (2 lines) |
| `/var/log/{alternatives,bootstrap,dpkg,jupyter,onstart,ssh_proxy}.log` | — | — | system logs | no | OS/package logs; jupyter.log and onstart.log are 0 bytes |
| `/tmp/claude-0/-root-projects-ET-Miner/1bae0ea5-20f3-44f0-a8ba-07a290957c5b/tasks/*.output` | — | — | 12 task outputs | no (this audit) | background-task outputs of the current session |
| `/root/projects/ET-Miner/runs/20260902T0000Z/RUN_DIR.txt` | 73 | 2026-09-02 00:04 | txt | meta | 'RUN_DIR=/root/projects/ET-Miner/runs/20260902T0000Z' + '2026-09-02T00:04:43Z' |

### B.4 Deleted files known only by name from git history (NOT checked out, per instructions)

| deleted path | deleted in commit | note |
|---|---|---|
| `bench/results/campaign/smoke-legacy-1g_r0.log` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/smoke-legacy-1g_r0.result.json` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/smoke-legacy-2g_r0.log` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/smoke-legacy-2g_r0.result.json` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/smoke-shared-1g_r0.log` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/smoke-shared-1g_r0.result.json` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/smoke-shared-2g_r0.log` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/smoke-shared-2g_r0.result.json` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/stressk2ml2-legacy-2g_r0.log` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/stressk2ml2-legacy-2g_r0.result.json` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/stressk2ml2-shared-2g_r0.log` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/stressk2ml2-shared-2g_r0.result.json` | 6f789d8 (2026-08-31 22:39 UTC) | smoke-matrix per-run log/result (bench, synthetic presets) |
| `bench/results/campaign/raw.jsonl` | b1f147e (2026-09-01 20:11 UTC) | earlier smoke campaign aggregate / log (bench, synthetic presets) |
| `bench/results/campaign/report.md` | b1f147e (2026-09-01 20:11 UTC) | earlier smoke campaign aggregate / log (bench, synthetic presets) |
| `bench/results/smoke_run.log` | b1f147e (2026-09-01 20:11 UTC) | earlier smoke campaign aggregate / log (bench, synthetic presets) |

No deleted file in the entire git history (`git log --all --diff-filter=D`) matches alphafold / base214 / results_214m / experiment_ / parquet / csv — i.e. the AlphaFold mining logs and parquets were never committed to this repository.

Also referenced by code but absent everywhere on disk: `archived/alphafold/results_214m/itemsets_214m_godmode.parquet`, `archived/alphafold/results_214m/item_mapping_214m.parquet` (notebook cell 1), `results_214m/parquet/frequent_k*.parquet`, `results_214m/closed_maximal.json`, `results_214m/experiment_log_*.txt`, `/workspace/data/transactions_214m*.parquet`, `/workspace/data/transactions_35k.parquet`, `/mnt/hdd/research/et-miner-data/transactions_214m.parquet`.

## SECTION C — totals

- Total claim rows: **948** (IDs L-001 … L-948)
- Rows by category: deterministic: 369; external-fact: 10; hardware-dependent: 394; method-parameter: 140; software: 35
- Rows by file:
  - `bench/results/2026-08-31-3090x2/raw.jsonl`: 253
  - `bench/results/2026-08-31-3090x2/report.md`: 146
  - `bench/results/2026-09-01-3090x2-sparse/raw.jsonl`: 118
  - `bench/results/2026-09-01-3090x2-sparse/report.md`: 62
  - `applications/alphafold/results_214m/experiment_null_model_20260219_061046.json`: 60
  - `applications/alphafold/experiments/analysis_alpha_centauri.ipynb`: 56
  - `applications/alphafold/results_214m/experiment_direct_vs_son_20260219_050326.json`: 41
  - `README.md`: 30
  - `bench/results/2026-09-01-3090x2-sparse/FINDINGS.md`: 28
  - `bench/results/2026-08-31-3090x2/FINDINGS.md`: 25
  - `applications/alphafold/results_214m/decoded_top_k_patterns.txt`: 25
  - `applications/alphafold/deploy/RUNBOOK_base214m.md`: 18
  - `applications/alphafold/deploy/run_all_experiments.sh`: 13
  - `bench/results/2026-08-31-3090x2/env.txt`: 11
  - `applications/alphafold/results_214m/GLOSSARY.md`: 10
  - `PROGRESS.md`: 10
  - `bench/results/2026-09-01-3090x2-sparse/env.txt`: 8
  - `bench/README.md`: 6
  - `applications/alphafold/deploy/deploy_project_milky_way.sh`: 5
  - `applications/alphafold/deploy/run_null_model_35k.sh`: 4
  - `applications/alphafold/deploy/deploy_base214m.sh`: 4
  - `applications/alphafold/experiments/experiment_direct_vs_son.py`: 4
  - `applications/alphafold/pipeline/run_mining.py`: 3
  - `applications/alphafold/pipeline/postprocess_tx.py`: 2
  - `applications/alphafold/pipeline/extract_features.py`: 2
  - `applications/alphafold/experiments/experiment_full_campaign.py`: 2
  - `applications/alphafold/pipeline/pipeline_214m.py`: 1
  - `applications/alphafold/experiments/analyze_k22_proteins.py`: 1
- Files inventoried: 21 extracted + 9 paper/policy files (not extracted) + 12 non-experiment locations + 15 deleted-by-name
- Surviving AlphaFold/base214m result artifacts: **2 JSON files + 1 decoded-itemset TXT** (all dated/derived from the 2026-02-19 1K-vocab run on 76,890,945 transactions), plus hard-coded numbers in the notebook, GLOSSARY, README and RUNBOOK. **Zero** AlphaFold mining logs survive.

### C.1 Cross-file observations (recorded for the merge step; no verdicts)

- `min_count` at 0.001% on 76,890,945 tx is **768** in `experiment_direct_vs_son_*.json` (L8) but **769** in `experiment_null_model_*.json` (L6, min_support 1.0001177641918693e-05 = 769/76,890,945); ceil(1e-5 × 76,890,945) = 769.
- Notebook cell 14 / cell 28 state **89,566** real itemsets at K≥7; the JSON `real_distribution` (identical in both JSON files) sums to **88,745** for K=7..14.
- Three different itemset totals refer to three different runs/thresholds: **475,865** (K≤14, 0.001%, JSONs), **2,841,280** (K≤19, 0.0001% 'Blitz', JSON L45; GLOSSARY '2.84M, K=1-19'; the decoded K=15–19 dump belongs to this run: 5,351 itemsets K≥15), and **26.8M / K≤22 / support ≥ 1e-7** ('God Mode', notebook + README; its parquet is missing).
- Notebook cell 3 says K=15–22 ≈ **5.5K** itemsets (God Mode); the decoded dump has **5,351** for K=15–19 only (Blitz run) — different runs, similar magnitude.
- Protein counts: `76,890,945` (JSONs, TXT L3) = README '76.9M with multiple annotations' of '214M total'; RUNBOOK L87 says v1 was '76,9M van 205,6M'; GLOSSARY says '214 million'; notebook cell 0 says 76.9M (1K) / 109.2M (35K).
- Hardware in the surviving AlphaFold narratives: single H100 (README L209/L223), 'H100/H200' (notebook), 4×/8× H200 (deploy scripts). All surviving *measured* logs are 2× RTX 3090 (bench). Current box (PROGRESS.md) is 2× RTX 3090, driver 595.71.05 (bench env.txt showed 580.159.03 on 2026-08-31/09-01 → different driver, possibly different box).
- Bench signature values that any reproduction on this box should hit bit-exactly (synthetic presets, deterministic): deep_k `n_itemsets=8841 sum_counts=266261275 hash 4d2c8d28bcd33cd61f763915e42cb60c0f60f16e1ffec2af32a887faabf47567`; smoke (two-phase) `632 / 1049580 / ea17ea26fd0e44f7…`; skewed_rows `10350 / 346834073 / 75262446a1c2b29b…`; stress_k2 K≤2 `1695332 / 218250884 / 8d989bfcc6e5c745…`; stress_k2 K≤3 exact `3005770 / 314350393 / a6d53e9a5e1b44a7…` vs prefilter-on legacy-1g `2996485 / 314055259 / 0b3e8434997fd3b1…` (the documented −9,285 divergence).
- `env.txt` in both bench dirs has an empty `pip freeze` section — package versions (CuPy 14.1.1 etc.) are only asserted in FINDINGS.md.
- The notebook has no executed outputs (execution_count = null everywhere; one stored print) — every number in it is hard-coded text, not a recorded result.

### C.2 Search commands used (read-only)

```
git log --all --oneline | head -80
git log --all --name-only --diff-filter=D --pretty=format: | grep -iE 'log|result|alphafold|base214|json|csv|parquet|jsonl|txt|out' | sort -u
git show --name-status --diff-filter=D 6f789d8 b1f147e ; git show --stat 65d9098
ls -laR bench/results applications/alphafold runs paper datasets .claude .github
find /root -xdev \( -path /root/projects/downloads/google-cloud-sdk -o -path '*/site-packages' -o -path '*/.cache' -o -path '*/.git' -o -path '*/node_modules' -o -path '*/.venv' \) -prune -o -type f \( -iname '*.log' -o -iname '*.jsonl' -o -iname '*.out' -o -iname 'nohup.out' -o -iname 'results*.json' -o -iname 'results*.csv' -o -iname 'results*.md' -o -iname '*alphafold*' -o -iname '*base214*' -o -iname '*et_miner*' -o -iname '*et-miner*' -o -iname '*.parquet' \) -printf '%TY-%Tm-%Td %TH:%TM %10s %p\n'
find / -xdev \( -path /proc -o -path /sys -o -path /opt/conda -o -path /root/projects/downloads/google-cloud-sdk -o -path '*/site-packages' -o -path '*/.cache' -o -path '*/.git' -o -path /root/projects/ET-Miner -o -path /usr/lib -o -path /usr/share \) -prune -o -type f \( -iname '*base214*' -o -iname '*alphafold*' -o -iname '*et_miner*' -o -iname '*et-miner*' -o -iname 'nohup.out' -o -iname '*.jsonl' -o -iname '*.parquet' \) -print
find / -xdev -type d -name archived   # → none
df -h ; ls -la /workspace /data /mnt /media /srv /tmp /var/log /root/.local /root/.jupyter /root/.ipython
cat /root/.bash_history /root/.claude/history.jsonl ; grep -nE 'gs://|base214|alphafold' /root/.boto /root/.config/gcloud/logs/*/*.log
python3 (json) summaries of bench/results/**/raw.jsonl, results_214m/*.json, decoded_top_k_patterns.txt, analysis_alpha_centauri.ipynb
```
