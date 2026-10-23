#!/usr/bin/env bash
# Build the pinned ET-Miner engine checkout and run its CPU test gate.
#
# Usage: engine_gate.sh ENGINE_DIR LOG_DIR
#   ENGINE_DIR  detached checkout of ET-Miner main (own .venv is created there)
#   LOG_DIR     directory for the sync/build/test logs
# Runs with no CUDA device visible and every thread pool pinned to THREADS (default 6).
set -euo pipefail
ENGINE=$1; LOGS=$2; T=${THREADS:-6}
export CUDA_VISIBLE_DEVICES= POLARS_MAX_THREADS=$T RAYON_NUM_THREADS=$T OMP_NUM_THREADS=$T \
       MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
cd "$ENGINE"
unset VIRTUAL_ENV
env -u CONDA_PREFIX uv sync > "$LOGS/engine_uv_sync.log" 2>&1
env -u CONDA_PREFIX uv run maturin develop --release -m rust_ext/Cargo.toml > "$LOGS/engine_maturin.log" 2>&1
uv run python -c 'import et_miner,sys; print(et_miner.__file__, et_miner.has_rust_extension())' > "$LOGS/engine_import.log" 2>&1
set +e
uv run pytest -q -m "not slow" -p no:cacheprovider > "$LOGS/engine_pytest_not_slow.log" 2>&1
echo "rc=$?" >> "$LOGS/engine_pytest_not_slow.log"
uv run pytest tests/test_tier_equivalence.py -q -p no:cacheprovider > "$LOGS/engine_pytest_tier.log" 2>&1
echo "rc=$?" >> "$LOGS/engine_pytest_tier.log"
