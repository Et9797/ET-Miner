#!/bin/bash
# Phase 3: 4×H200 Full Experiment Suite
# Co-authored-by: C. claudya <claudya@anthropic.local>
#
# Runs all experiments in sequence:
#   1. Full mining campaign (6 thresholds × 3 runs)
#   2. Direct GPU vs SON controlled comparison (3 runs each)
#   3. Null model at min_count=769 (100 permutations, 4 GPUs)
#   4. Null model at min_count=8 (100 permutations, 4 GPUs) — THE BIG ONE
#   5. K=22 protein identification + GO hierarchy analysis
#
# Usage:
#   bash run_all_experiments.sh                    # defaults
#   bash run_all_experiments.sh /path/to/data.parquet  # custom data path
#
# Expected runtime on 4×H200 (~$9/hr): ~4 hours total
# H200 has 141GB VRAM (564GB total) and 4.8 TB/s bandwidth (1.4× H100)

set -euo pipefail

# CuPy JIT-compiles its ElementwiseKernels (popcount etc.) via NVRTC, which needs the
# CUDA toolkit headers. The box ships them at /usr/local/cuda; without CUDA_PATH set,
# mining dies with "Failed to find CUDA headers". Set it for the whole suite.
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

DATA="${1:-/workspace/data/transactions_214m.parquet}"
OUTPUT_DIR="results_214m"
N_GPUS="${N_GPUS:-4}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="${OUTPUT_DIR}/experiment_log_${TIMESTAMP}.txt"

mkdir -p "$OUTPUT_DIR"

echo "=============================================" | tee "$LOG"
echo " Phase 3: 4×H200 Full Experiment Suite"       | tee -a "$LOG"
echo " $(date)"                                      | tee -a "$LOG"
echo " Data: $DATA"                                  | tee -a "$LOG"
echo " GPUs: $N_GPUS"                                | tee -a "$LOG"
echo "=============================================" | tee -a "$LOG"

# Verify GPU setup
echo "" | tee -a "$LOG"
echo "=== GPU Verification ===" | tee -a "$LOG"
python -c "
import cupy
n = cupy.cuda.runtime.getDeviceCount()
print(f'CuPy detected {n} GPU(s)')
for i in range(n):
    props = cupy.cuda.runtime.getDeviceProperties(i)
    name = props['name'].decode()
    mem_gb = props['totalGlobalMem'] / 1e9
    print(f'  GPU {i}: {name} ({mem_gb:.0f} GB)')
assert n >= int('$N_GPUS'), f'Need $N_GPUS GPUs but only {n} available'
print(f'All {n} GPUs ready ✓')
" 2>&1 | tee -a "$LOG"

# 1. Full mining campaign (triplicate)
echo "" | tee -a "$LOG"
echo "=== [1/5] Full Mining Campaign (3× each threshold) ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
python3 experiments/experiment_full_campaign.py \
    --data "$DATA" \
    --runs 3 \
    --output-dir "$OUTPUT_DIR" \
    -v 2>&1 | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"

# 2. Direct GPU vs SON controlled comparison (triplicate)
echo "" | tee -a "$LOG"
echo "=== [2/5] Direct GPU vs SON (3 runs each) ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
python3 experiments/experiment_direct_vs_son.py \
    --data "$DATA" \
    --runs 3 \
    --output-dir "$OUTPUT_DIR" \
    -v 2>&1 | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"

# 3. Null model at min_count=769 (100 permutations, multi-GPU)
echo "" | tee -a "$LOG"
echo "=== [3/5] Null Model — min_count=769 (100 perms, ${N_GPUS} GPUs) ===" | tee -a "$LOG"
echo "Estimated: ~25 min (fast at high threshold)" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
python3 experiments/experiment_null_model.py \
    --data "$DATA" \
    --min-count 769 \
    --runs 100 \
    --n-gpus "$N_GPUS" \
    --perm-per-gpu \
    --output-dir "$OUTPUT_DIR" \
    -v 2>&1 | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"

# 4. Null model at min_count=8 — THE CRITICAL EXPERIMENT
echo "" | tee -a "$LOG"
echo "=== [4/5] Null Model — min_count=8 (100 perms, ${N_GPUS} GPUs) ===" | tee -a "$LOG"
echo "This is the critical experiment: does the null reach K≥7 at Opus threshold?" | tee -a "$LOG"
echo "Estimated: ~100 min (25 batches × ~4 min/perm on H200)" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
python3 experiments/experiment_null_model.py \
    --data "$DATA" \
    --min-count 8 \
    --runs 100 \
    --n-gpus "$N_GPUS" \
    --perm-per-gpu \
    --output-dir "$OUTPUT_DIR" \
    -v 2>&1 | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"

# 5. K=22 protein identification + GO hierarchy
echo "" | tee -a "$LOG"
echo "=== [5/5] K=22 Analysis (protein IDs + GO hierarchy) ===" | tee -a "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
python3 experiments/analyze_k22_proteins.py \
    --data "$DATA" \
    --output-dir "$OUTPUT_DIR" \
    -v 2>&1 | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"

# Done
echo "" | tee -a "$LOG"
echo "=============================================" | tee -a "$LOG"
echo " ALL EXPERIMENTS COMPLETE"                     | tee -a "$LOG"
echo " $(date)"                                      | tee -a "$LOG"
echo " Log: $LOG"                                    | tee -a "$LOG"
echo "=============================================" | tee -a "$LOG"
