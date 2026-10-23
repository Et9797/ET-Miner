#!/bin/bash
# Run null model permutation test for the 35K-feature V2 dataset.
# Designed for Vast.ai deployment after deploy_project_milky_way.sh.
# Co-authored-by: C. claudya <claudya@anthropic.local>
#
# Usage (on remote GPU instance):
#   bash run_null_model_35k.sh [n_perms] [support]
#
# Defaults: 5 permutations, 0.001% support (min_count=1092)
# Expected runtime: ~30-60 min per permutation on 8× H200

set -euo pipefail

N_PERMS="${1:-5}"
SUPPORT="${2:-0.00001}"

# Input validation — consistent with run_mining.sh
[[ "$N_PERMS" =~ ^[0-9]+$ ]] || { echo "ERROR: N_PERMS must be integer, got '$N_PERMS'"; exit 1; }
[[ "$SUPPORT" =~ ^[0-9.eE+-]+$ ]] || { echo "ERROR: SUPPORT must be numeric, got '$SUPPORT'"; exit 1; }

DATA="/workspace/data/transactions_35k.parquet"
OUTPUT="/workspace/results_35k_v2/null_model"

cd /workspace/ET-miner
source .venv/bin/activate

# Auto-detect nvidia lib paths
VENV_PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
NVIDIA_BASE=".venv/lib/python${VENV_PYVER}/site-packages/nvidia"
if [ -d "$NVIDIA_BASE" ]; then
    for d in "$NVIDIA_BASE"/*/lib; do
        [ -d "$d" ] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
fi

mkdir -p "$OUTPUT"

echo "============================================"
echo "  NULL MODEL — 35K Features"
echo "  Data:    $DATA"
echo "  Support: $SUPPORT"
echo "  Perms:   $N_PERMS"
echo "  Output:  $OUTPUT"
echo "============================================"

python3 applications/alphafold/experiments/experiment_null_model.py \
    --data "$DATA" \
    --min-count "$SUPPORT" \
    --runs "$N_PERMS" \
    --seed 42 \
    --output-dir "$OUTPUT" \
    -v
