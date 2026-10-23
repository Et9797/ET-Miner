#!/bin/bash
# Runs the branch's analyze_k22_proteins.py unchanged on the CPU Opus tables.
#
# The engine venv has no pronto, so the script runs on the engine venv's
# interpreter with pronto layered on by `uv run --with` (PyPI). go.obo is the
# GPU run's committed copy, placed in the output dir so nothing is downloaded.
# Threads are pinned and no device is visible, as in run_py.sh.
#
# Usage: a1_k22.sh RUN_DIR
set -euo pipefail
R=$(cd "$1" && pwd)
REPO=$(git -C "$R" rev-parse --show-toplevel)
OUT=$R/data/mined/A1/k22
mkdir -p "$OUT"
cp "$REPO/runs/20260902T0000Z/phase3/2026_01/exp/go.obo" "$OUT/go.obo"
export CUDA_VISIBLE_DEVICES= POLARS_MAX_THREADS=6 RAYON_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
unset VIRTUAL_ENV ET_UPLOAD_GCS
cd "$OUT"
uv run --no-project --python "$REPO/../ET-Miner-engine/.venv/bin/python" --with 'pronto>=2.5' \
  python -I "$REPO/applications/alphafold/experiments/analyze_k22_proteins.py" \
  --data "$R/data/extract_2026_01/transactions_214m_base.parquet" \
  --mapping "$R/data/extract_2026_01/item_mapping_214m_base.parquet" \
  --itemset-source "$R/data/mined/C6/parquet/k22.parquet" \
  --output-dir "$OUT"
