#!/bin/bash
# ============================================================
#  PROJECT MILKY WAY — H200 DEPLOYMENT SCRIPT
#  One-shot: code + data + deps → ready for 35K-feature mining
#  Co-authored-by: C. claudya <claudya@anthropic.local>
#
#  Usage: ./deploy_project_milky_way.sh <ssh-port> <ssh-host>
#  Example: ./deploy_project_milky_way.sh 29887 212.247.220.194
#
#  10-step deploy:
#    [1/10] Test SSH + nvidia-smi
#    [2/10] rsync ET-miner source code
#    [3/10] rsync 35K-feature data (+3.5 checkpoints)
#    [4/10] Install/update Python dependencies
#    [5/10] Install rclone + gcloud CLI for results upload
#    [6/10] Rebuild Rust extension via maturin
#    [7/10] Verify imports (CuPy, et_miner, bitvec builder)
#    [8/10] CUDA kernel warm-up (zero-JIT deploy)
#    [9/10] VRAM dry run (bitvec fit check)
#   [10/10] Copy checkpoint parquets to ramdisk
#
#  Authors: Et & claudya
# ============================================================
set -euo pipefail

PORT="${1:?Usage: ./deploy_project_milky_way.sh <port> <host>}"
HOST="${2:?Usage: ./deploy_project_milky_way.sh <port> <host>}"

# Input validation — prevent command injection via CLI args
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "ERROR: PORT must be numeric, got '$PORT'"; exit 1; }
[[ "$HOST" =~ ^[a-zA-Z0-9._:-]+$ ]] || { echo "ERROR: Invalid HOST '$HOST'"; exit 1; }

SSH="ssh -p $PORT root@$HOST"
RSYNC="rsync -avz --progress -e \"ssh -p $PORT\""

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_DIR="/mnt/hdd/research/et-miner-data/project-milky-way"
REMOTE_DATA="/workspace/data"
REMOTE_CODE="/workspace/ET-miner"
REMOTE_VENV="$REMOTE_CODE/.venv"

echo "============================================"
echo "  PROJECT MILKY WAY H200 DEPLOY"
echo "  Target: root@$HOST:$PORT"
echo "  Data:   $DATA_DIR → $REMOTE_DATA"
echo "  Code:   $REPO_ROOT → $REMOTE_CODE"
echo "============================================"

# --- [1/10] Test SSH + nvidia-smi ---
echo ""
echo "[1/10] Testing SSH connection + GPU check..."
$SSH "echo 'Connection OK'; nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader" || {
    echo "ERROR: Cannot connect to $HOST:$PORT"
    exit 1
}

# --- [2/10] rsync ET-miner source ---
echo ""
echo "[2/10] Syncing ET-miner source code..."
$SSH "mkdir -p $REMOTE_CODE"
rsync -avz --progress -e "ssh -p $PORT" \
    --exclude '.git' \
    --exclude '.venv*' \
    --exclude 'target' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '*.egg-info' \
    --exclude '.claude/' \
    --exclude '.vscode/' \
    --exclude 'viz/' \
    --exclude 'dashboard/' \
    --exclude 'data-dump/' \
    --exclude 'archived/' \
    --exclude 'tools/' \
    --exclude 'papers/' \
    --exclude 'paper/' \
    --exclude 'lore/' \
    --exclude 'docs/' \
    --exclude 'tests/' \
    --exclude 'tasks/' \
    --exclude 'src/opus_pocket/' \
    --exclude 'benchmarks/' \
    --exclude 'council-of-copii/' \
    --exclude 'et/' \
    --exclude 'plan-mode-sentinel/' \
    --exclude '*.parquet' \
    --exclude '*.pkl' \
    --exclude '*.zip' \
    --exclude '*.wav' \
    --exclude '*.mp3' \
    --exclude '*.png' \
    --exclude '*.txt' \
    --exclude '*.md' \
    --exclude '*.json' \
    --exclude '.env' \
    --exclude '*.tar*' \
    --exclude 'backup-*' \
    --exclude 'claudya-voice/' \
    --exclude '.serena/' \
    --exclude '.pytest_cache/' \
    --exclude 'data/' \
    --exclude 'logs/' \
    --exclude 'results/' \
    --exclude 'cubin_cache/' \
    --exclude 'et-shannon-claudya-algorithm/' \
    --exclude '2026-*' \
    --exclude '*.log' \
    "$REPO_ROOT/" "root@$HOST:$REMOTE_CODE/"
echo "  Source synced."

# Ensure README.md exists BEFORE pip install (hatchling build backend requires it, but *.md excluded from rsync)
$SSH "touch $REMOTE_CODE/README.md"

# Sync .tmux.conf for a comfortable remote terminal
if [ -f "$HOME/.tmux.conf" ]; then
    scp -P "$PORT" "$HOME/.tmux.conf" "root@$HOST:~/.tmux.conf" 2>/dev/null
    echo "  .tmux.conf synced."
fi

# Install vim for remote editing
$SSH "apt-get update -qq && apt-get install -y -qq vim 2>/dev/null" && echo "  vim installed."

# --- [3/10] rsync data ---
echo ""
echo "[3/10] Syncing Milky Way data..."
$SSH "mkdir -p $REMOTE_DATA"

# 35K-feature transactions (the main payload, ~2.3 GB)
TX_FILE="$DATA_DIR/transactions_35k.parquet"
if [ -f "$TX_FILE" ]; then
    echo "  Syncing transactions_35k.parquet ($(du -h "$TX_FILE" | cut -f1))..."
    rsync -avz --progress -e "ssh -p $PORT" \
        "$TX_FILE" "root@$HOST:$REMOTE_DATA/"
else
    echo "  WARNING: $TX_FILE not found — skipping (need transactions_35k.parquet)"
fi

# Item mapping (small)
MAP_FILE="$DATA_DIR/item_mapping_35k.parquet"
if [ -f "$MAP_FILE" ]; then
    rsync -avz --progress -e "ssh -p $PORT" \
        "$MAP_FILE" "root@$HOST:$REMOTE_DATA/"
else
    echo "  WARNING: $MAP_FILE not found — skipping"
fi

# Also sync the original 214m data (if available — for legacy comparison)
TX_214M="/mnt/hdd/research/et-miner-data/transactions_214m.parquet"
if [ -f "$TX_214M" ]; then
    rsync -avz --progress -e "ssh -p $PORT" \
        "$TX_214M" "root@$HOST:$REMOTE_DATA/"
fi

echo "  Data synced."

# --- [3.5/10] Sync checkpoint parquets (for resume) ---
# SKIPPED: fresh V3 benchmark, no resume needed. Re-enable for incremental runs.
# CHECKPOINT_DIR="$DATA_DIR/parquet"
# if [ -d "$CHECKPOINT_DIR" ]; then ...
echo ""
echo "[3.5/10] Skipped checkpoint parquet sync (fresh run)"

# --- [4/10] Install/update Python dependencies ---
echo ""
echo "[4/10] Installing Python dependencies..."
$SSH bash -s << 'REMOTE_DEPS'
set -e

# Ensure uv is available
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"

cd /workspace/ET-miner

# Create venv if needed
if [ ! -d ".venv" ]; then
    echo "  No venv found, creating..."
    uv venv .venv --python python3
fi

source .venv/bin/activate

# Install package + gpu extras
echo "  Installing et-miner[gpu]..."
uv pip install -e ".[gpu]" 2>&1 | tail -5

# Verify critical GPU deps — explicit fallback if extras resolution missed anything
echo "  Verifying GPU dependencies..."
python3 -c "import cupy" 2>/dev/null || {
    echo "  cupy missing — installing explicitly..."
    uv pip install "cupy-cuda12x>=13.0"
}
python3 -c "import pyarrow" 2>/dev/null || {
    echo "  pyarrow missing — installing explicitly..."
    uv pip install "pyarrow>=15.0"
}
python3 -c "import polars" 2>/dev/null || {
    echo "  polars missing — installing explicitly..."
    uv pip install "polars>=1.39.0"
}

# Install maturin for Rust extension build (step 6)
if ! command -v maturin &>/dev/null; then
    echo "  Installing maturin..."
    uv pip install maturin
fi

echo "  Deps installed. Versions:"
python3 -c "
import cupy; print(f'    cupy:    {cupy.__version__}')
import pyarrow; print(f'    pyarrow: {pyarrow.__version__}')
import polars; print(f'    polars:  {polars.__version__}')
import numpy; print(f'    numpy:   {numpy.__version__}')
"
REMOTE_DEPS

# --- [5/10] Install gcloud CLI for results upload ---
echo ""
echo "[5/10] Installing gcloud CLI..."
$SSH bash -s << 'REMOTE_CLOUD'
set -e

# Install gcloud CLI if needed
if ! command -v gcloud &>/dev/null; then
    echo "  Installing gcloud CLI..."
    curl -sSfL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/opt
    ln -sf /opt/google-cloud-sdk/bin/gcloud /usr/local/bin/gcloud
    ln -sf /opt/google-cloud-sdk/bin/bq /usr/local/bin/bq
    ln -sf /opt/google-cloud-sdk/bin/gsutil /usr/local/bin/gsutil
fi
echo "  gcloud: $(gcloud version 2>/dev/null | head -1)"
REMOTE_CLOUD

# Push GCS credentials if available
if [ -f "$(dirname "$0")/../.env" ]; then
    source "$(dirname "$0")/../.env"
fi

if [ -n "${GCP_SA_KEY_B64:-}" ]; then
    echo "$GCP_SA_KEY_B64" | base64 -d | $SSH 'cat > /tmp/gcp-sa-key.json'
    [[ "${GCP_PROJECT:-et-research}" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "[ERROR] Invalid GCP_PROJECT value"; exit 1; }
    $SSH bash -s << REMOTE_GCP
gcloud auth activate-service-account --key-file=/tmp/gcp-sa-key.json --quiet
gcloud config set project '${GCP_PROJECT:-et-research}' --quiet
rm /tmp/gcp-sa-key.json
echo "  GCP service account activated."
REMOTE_GCP
else
    echo "  GCP not configured (set GCP_SA_KEY_B64 in .env)"
fi

# --- [6/10] Rebuild Rust extension ---
echo ""
echo "[6/10] Rebuilding Rust extension..."
$SSH bash -s << 'REMOTE_RUST'
set -e
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Install Rust if needed
if ! command -v cargo &>/dev/null; then
    echo "  Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

cd /workspace/ET-miner
source .venv/bin/activate

# maturin already installed in step 4, but verify
command -v maturin &>/dev/null || uv pip install maturin

cd rust_ext
echo "  Building Rust extension (release mode)..."
maturin develop --release 2>&1 | tail -3
echo "  Rust extension built."
REMOTE_RUST

# --- [7/10] Verify imports ---
echo ""
echo "[7/10] Verifying imports and GPU access..."
$SSH bash -s << 'REMOTE_VERIFY'
set -e
cd /workspace/ET-miner
source .venv/bin/activate

# Auto-detect nvidia lib paths (same as benchmarks/setup.sh CUDA patch)
VENV_PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
NVIDIA_BASE=".venv/lib/python${VENV_PYVER}/site-packages/nvidia"
if [ -d "$NVIDIA_BASE" ]; then
    for d in "$NVIDIA_BASE"/*/lib; do
        [ -d "$d" ] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
fi

python3 -c "
import cupy as cp
n_gpus = cp.cuda.runtime.getDeviceCount()
print(f'  CuPy OK | {n_gpus} GPU(s)')
for i in range(n_gpus):
    with cp.cuda.Device(i):
        free, total = cp.cuda.runtime.memGetInfo()
        print(f'    GPU {i}: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free')

from et_miner.apriori import apriori
print('  et_miner.apriori: OK')

from et_miner.cuda_csr_bitvec import build_bitvecs_from_gpu_arrays
print('  build_bitvecs_from_gpu_arrays: OK')

import et_miner_rust
print('  Rust extension: OK')

# Check data files
import os
data_dir = '/workspace/data'
for f in ['transactions_35k.parquet', 'item_mapping_35k.parquet', 'transactions_214m.parquet']:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024**3)
        print(f'  Data: {f} ({size:.2f} GB)')
    else:
        print(f'  Data: {f} — NOT FOUND')

print()
print('ALL IMPORTS VERIFIED')
"
REMOTE_VERIFY

# --- [8/10] CUDA kernel warm-up (zero-JIT deploy) ---
echo ""
echo "[8/10] CUDA kernel warm-up..."

# Warm up kernels on-device (JIT compiles in ~2-3s on H100/H200, no cubin shipping needed)
$SSH bash -s << 'REMOTE_WARMUP'
set -e
cd /workspace/ET-miner
source .venv/bin/activate 2>/dev/null || true

VENV_PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
NVIDIA_BASE=".venv/lib/python${VENV_PYVER}/site-packages/nvidia"
if [ -d "$NVIDIA_BASE" ]; then
    for d in "$NVIDIA_BASE"/*/lib; do
        [ -d "$d" ] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
fi

python3 -u scripts/warmup_kernels.py --all-gpus
REMOTE_WARMUP

# --- [9/10] VRAM dry run ---
echo ""
echo "[9/10] VRAM dry run (bitvec fit check)..."
echo "  This verifies the ~478 GB bitvec fits across N×H200 with row-split."
$SSH bash -s << 'REMOTE_DRYRUN'
set -e
cd /workspace/ET-miner
source .venv/bin/activate

VENV_PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
NVIDIA_BASE=".venv/lib/python${VENV_PYVER}/site-packages/nvidia"
if [ -d "$NVIDIA_BASE" ]; then
    for d in "$NVIDIA_BASE"/*/lib; do
        [ -d "$d" ] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
fi

# Check if 35k data exists for dry run
DATA_FILE="/workspace/data/transactions_35k.parquet"
if [ ! -f "$DATA_FILE" ]; then
    echo "  SKIP: transactions_35k.parquet not found — dry run requires data"
    echo "  Deploy data first, then re-run step [9/10] manually:"
    echo "    cd /workspace/ET-miner && source .venv/bin/activate"
    echo "    CUDA_VISIBLE_DEVICES=0 python3 -u applications/alphafold/experiments/experiment_null_model.py \\"
    echo "      --data /workspace/data/transactions_35k.parquet --runs 2 --n-gpus 1 -v"
    exit 0
fi

echo "  Checking bitvec VRAM fit across all GPUs..."
timeout 600 python3 -c "
import cupy as cp
import polars as pl
import numpy as np
import time

# Load data
print('  Loading transactions...')
df = pl.read_parquet('$DATA_FILE')
n_proteins = len(df)
# Find max item ID for bitvec width — vectorized, no .to_list() RAM explosion
max_item = df['items'].explode().max()
n_items = max_item + 1
print(f'  {n_proteins:,} proteins × {n_items:,} items')

# Estimate bitvec size (column-major: n_items columns × ceil(n_proteins/64) uint64 words)
bitvec_bytes = n_items * ((n_proteins + 63) // 64) * 8
print(f'  Estimated bitvec: {bitvec_bytes / 1e9:.1f} GB')

free_before, total = cp.cuda.runtime.memGetInfo()
print(f'  GPU 0: {total/1e9:.1f} GB total, {free_before/1e9:.1f} GB free')

# Row-split: bitvec is divided across all available GPUs
n_gpus = cp.cuda.runtime.getDeviceCount()
per_gpu_bytes = bitvec_bytes // n_gpus
print(f'  Row-split: {bitvec_bytes/1e9:.1f} GB / {n_gpus} GPUs = {per_gpu_bytes/1e9:.1f} GB/GPU')

if per_gpu_bytes > free_before:
    print(f'  ERROR: Per-GPU bitvec ({per_gpu_bytes/1e9:.1f} GB) exceeds free VRAM ({free_before/1e9:.1f} GB)')
    exit(1)

print(f'  Per-GPU fits! Headroom: {(free_before - per_gpu_bytes)/1e9:.1f} GB')
print()
print('  VRAM DRY RUN: PASSED')
" 2>&1

REMOTE_DRYRUN

# --- [10/10] Copy checkpoint parquets to /dev/shm ramdisk ---
RAMDISK="/dev/shm"
echo ""
echo "[10/10] Setting up ramdisk ($RAMDISK) for fast parquet I/O..."
$SSH bash -s << REMOTE_RAMDISK
set -e
# /dev/shm is pre-mounted tmpfs on vast.ai (~1TB)
# run_mining.py appends /parquet to output_dir, so resume expects /dev/shm/parquet/
if ls ${REMOTE_DATA}/parquet/frequent_k*.parquet 1>/dev/null 2>&1; then
    echo "  Copying checkpoint parquets to ramdisk..."
    mkdir -p ${RAMDISK}/parquet
    cp ${REMOTE_DATA}/parquet/frequent_k*.parquet ${RAMDISK}/parquet/
    echo "  Copied \$(ls ${RAMDISK}/parquet/frequent_k*.parquet | wc -l) checkpoint files to ramdisk"
    df -h ${RAMDISK}
else
    echo "  No checkpoint parquets to copy — fresh run"
fi
REMOTE_RAMDISK

echo ""
echo "============================================"
echo "  PROJECT MILKY WAY DEPLOY COMPLETE"
echo ""
echo "  SSH:  ssh -p $PORT root@$HOST"
echo "  Data: $REMOTE_DATA/"
echo "  Code: $REMOTE_CODE/"
echo "  Ramdisk: $RAMDISK (pre-mounted tmpfs)"
echo ""
echo "  Quick start:"
echo "    cd /workspace/ET-miner && source .venv/bin/activate"
echo ""
echo "  PROJECT MILKY WAY — Full null model experiment (8 GPUs, checkpointed):"
echo "    cd applications/alphafold/experiments"
echo "    python3 -u experiment_null_model.py \\"
echo "      --data /workspace/data/transactions_35k.parquet \\"
echo "      --runs 100 --min-count 1090 --n-gpus 8 --seed 42 \\"
echo "      --checkpoint --output-dir results_35k -v"
echo ""
echo "  Resume after preemption (add --resume):"
echo "    python3 -u applications/alphafold/experiments/experiment_null_model.py \\"
echo "      --data /workspace/data/transactions_35k.parquet \\"
echo "      --runs 100 --min-count 1090 --n-gpus 8 --seed 42 \\"
echo "      --checkpoint --resume --output-dir results_35k -v"
echo ""
echo "  ═══════════════════════════════════════════"
echo "  K=8 FREQUENT ITEMSET MINING (resume from K=7, RAMDISK):"
echo "    tmux new-session -d -s mining"
echo "    tmux send-keys -t mining 'cd /workspace/ET-miner/applications/alphafold/pipeline && \\"
echo "      python3 run_mining.py \\"
echo "        --input /workspace/data/transactions_35k.parquet \\"
echo "        --item-mapping /workspace/data/item_mapping_35k.parquet \\"
echo "        --support 0.00001 --max-length 8 --use-gpu --n-gpus 8 \\"
echo "        --parquet-flush --resume-from-k 7 --output-dir $RAMDISK -v' Enter"
echo "    tmux attach -t mining"
echo ""
echo "  FRESH K=8 RUN (no resume, RAMDISK):"
echo "    tmux new-session -d -s mining"
echo "    tmux send-keys -t mining 'cd /workspace/ET-miner/applications/alphafold/pipeline && \\"
echo "      python3 run_mining.py \\"
echo "        --input /workspace/data/transactions_35k.parquet \\"
echo "        --item-mapping /workspace/data/item_mapping_35k.parquet \\"
echo "        --support 0.00001 --max-length 8 --use-gpu --n-gpus 8 \\"
echo "        --parquet-flush --output-dir $RAMDISK -v' Enter"
echo "    tmux attach -t mining"
echo ""
echo "  SAVE RESULTS (ramdisk → persistent disk):"
echo "    cp $RAMDISK/frequent_k*.parquet $REMOTE_DATA/parquet/"
echo ""
echo "============================================"
