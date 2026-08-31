#!/bin/bash
# ============================================================
#  BASE-214M DEPLOYMENT — H200 box, base-vocab regeneration
#  Companion to deploy_project_milky_way.sh (which targets the 35K vocab).
#  Co-authored-by: C. claudya <claudya@anthropic.local>
#
#  Difference vs milky_way: NO 35K data rsync. The base-vocab transactions are
#  REGENERATED ON THE BOX from UniProt current_release via af-extract, then the
#  experiment suite runs against them. This script only ships code + builds +
#  verifies; data acquisition + extraction are separate on-box steps (printed at
#  the end and detailed in RUNBOOK_base214m.md).
#
#  Usage: ./deploy_base214m.sh <ssh-port> <ssh-host>
#  Example: ./deploy_base214m.sh 29887 212.247.220.194
#
#  Steps:
#    [1/8] Test SSH + nvidia-smi
#    [2/8] rsync ET-miner source (no parquet/data)
#    [3/8] Install/update Python deps (uv)
#    [4/8] Install gcloud + push GCS creds from .env
#    [5/8] Build Python Rust ext (maturin) + af-extract binary (cargo)
#    [6/8] Verify imports, Rust ext, af-extract binary, GPUs
#    [7/8] CUDA kernel warm-up (zero-JIT)
#    [8/8] Print data-acquisition + extraction + experiment quick-start
# ============================================================
set -euo pipefail

PORT="${1:?Usage: ./deploy_base214m.sh <port> <host>}"
HOST="${2:?Usage: ./deploy_base214m.sh <port> <host>}"

# Input validation — prevent command injection via CLI args
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "ERROR: PORT must be numeric, got '$PORT'"; exit 1; }
[[ "$HOST" =~ ^[a-zA-Z0-9._:-]+$ ]] || { echo "ERROR: Invalid HOST '$HOST'"; exit 1; }

SSH="ssh -p $PORT root@$HOST"
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
REMOTE_CODE="/workspace/ET-miner"
REMOTE_DATA="/workspace/data"

echo "============================================"
echo "  BASE-214M H200 DEPLOY (base-vocab regen)"
echo "  Target: root@$HOST:$PORT"
echo "  Code:   $REPO_ROOT -> $REMOTE_CODE"
echo "  Data:   regenerated on-box into $REMOTE_DATA (no rsync)"
echo "============================================"

# --- [1/8] Test SSH + nvidia-smi ---
echo ""
echo "[1/8] Testing SSH connection + GPU check..."
$SSH "echo 'Connection OK'; nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader" || {
    echo "ERROR: Cannot connect to $HOST:$PORT"
    exit 1
}

# --- [2/8] rsync ET-miner source (NO parquet/data) ---
echo ""
echo "[2/8] Syncing ET-miner source code (excluding all data)..."
$SSH "mkdir -p $REMOTE_CODE $REMOTE_DATA"
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
    --exclude 'papers/' \
    --exclude 'lore/' \
    --exclude 'docs/' \
    --exclude 'tasks/' \
    --exclude 'src/opus_pocket/' \
    --exclude 'benchmarks/' \
    --exclude 'council-of-copii/' \
    --exclude '*.parquet' \
    --exclude '*.pkl' \
    --exclude '*.zip' \
    --exclude '*.wav' \
    --exclude '*.mp3' \
    --exclude '*.png' \
    --exclude '.env' \
    --exclude '*.tar*' \
    --exclude '.pytest_cache/' \
    --exclude 'data/' \
    --exclude 'logs/' \
    --exclude 'results/' \
    --exclude '2026-*' \
    --exclude '*.log' \
    "$REPO_ROOT/" "root@$HOST:$REMOTE_CODE/"
echo "  Source synced."

# hatchling build backend requires README.md, but *.md is excluded from rsync
$SSH "touch $REMOTE_CODE/README.md"

if [ -f "$HOME/.tmux.conf" ]; then
    scp -P "$PORT" "$HOME/.tmux.conf" "root@$HOST:~/.tmux.conf" 2>/dev/null && echo "  .tmux.conf synced."
fi

# --- [3/8] Install/update Python dependencies ---
echo ""
echo "[3/8] Installing Python dependencies (uv)..."
$SSH bash -s << 'REMOTE_DEPS'
set -e
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

cd /workspace/ET-miner
[ -d ".venv" ] || uv venv .venv --python python3
source .venv/bin/activate

echo "  Installing et-miner[gpu]..."
uv pip install -e ".[gpu]" 2>&1 | tail -5

# aria2c for fast on-box data acquisition (dead-end lesson: direct on box, not via tunnel)
command -v aria2c &>/dev/null || { apt-get update -qq && apt-get install -y -qq aria2 2>/dev/null && echo "  aria2 installed."; }

python3 -c "import cupy" 2>/dev/null || uv pip install "cupy-cuda12x>=13.0"
python3 -c "import pyarrow" 2>/dev/null || uv pip install "pyarrow>=15.0"
python3 -c "import polars" 2>/dev/null || uv pip install "polars>=1.39.0"
command -v maturin &>/dev/null || uv pip install maturin

echo "  Deps installed."
REMOTE_DEPS

# --- [4/8] Install gcloud CLI + set up GCS credentials ---
# gcs.py authenticates with ADC authorized_user JSON (OAuth refresh-token,
# auto-refresh) — NOT a service-account key. Preference order here:
#   1. local ADC file (~/.config/gcloud/application_default_credentials.json) -> push it (keyless)
#   2. GCP_SA_KEY_B64 in .env -> legacy SA-key path (blocked by many org policies)
#   3. neither -> print the on-box `gcloud auth application-default login` command
echo ""
echo "[4/8] Installing gcloud CLI + setting up GCS credentials..."
$SSH bash -s << 'REMOTE_CLOUD'
set -e
if ! command -v gcloud &>/dev/null; then
    curl -sSfL https://sdk.cloud.google.com | bash -s -- --disable-prompts --install-dir=/opt
    ln -sf /opt/google-cloud-sdk/bin/gcloud /usr/local/bin/gcloud
    ln -sf /opt/google-cloud-sdk/bin/bq /usr/local/bin/bq
    ln -sf /opt/google-cloud-sdk/bin/gsutil /usr/local/bin/gsutil
fi
echo "  gcloud: $(gcloud version 2>/dev/null | head -1)"
REMOTE_CLOUD

if [ -f "$(dirname "$0")/../.env" ]; then
    source "$(dirname "$0")/../.env"
fi
LOCAL_ADC="$HOME/.config/gcloud/application_default_credentials.json"
REMOTE_ADC='/root/.config/gcloud/application_default_credentials.json'
if [ -f "$LOCAL_ADC" ]; then
    echo "  Pushing local ADC credentials (authorized_user, keyless)..."
    $SSH "mkdir -p /root/.config/gcloud"
    $SSH "cat > $REMOTE_ADC" < "$LOCAL_ADC"
    $SSH "gcloud storage ls gs://et-miner-results >/dev/null 2>&1 && echo '  GCS auth OK (ADC).' || echo '  WARNING: ADC pushed but gs://et-miner-results not readable — check the account has access.'"
elif [ -n "${GCP_SA_KEY_B64:-}" ]; then
    echo "  Using legacy service-account key from .env..."
    echo "$GCP_SA_KEY_B64" | base64 -d | $SSH 'cat > /tmp/gcp-sa-key.json'
    [[ "${GCP_PROJECT:-et-research}" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "[ERROR] Invalid GCP_PROJECT"; exit 1; }
    $SSH bash -s << REMOTE_GCP
gcloud auth activate-service-account --key-file=/tmp/gcp-sa-key.json --quiet
gcloud config set project '${GCP_PROJECT:-et-research}' --quiet
rm /tmp/gcp-sa-key.json
echo "  GCP service account activated."
REMOTE_GCP
else
    echo "  No GCS creds found (no local ADC, no GCP_SA_KEY_B64)."
    echo "  Org policies often block SA keys — use keyless ADC instead. Run ON THE BOX:"
    echo "      ssh -p $PORT root@$HOST"
    echo "      gcloud auth application-default login --no-launch-browser"
    echo "  (logs in as your Google user; writes the authorized_user JSON gcs.py expects,"
    echo "   with OAuth auto-refresh for the whole run). Then re-run this step or continue."
fi

# --- [5/8] Build Python Rust ext (maturin) + af-extract binary (cargo) ---
echo ""
echo "[5/8] Building Rust extension (maturin) + af-extract binary (cargo)..."
$SSH bash -s << 'REMOTE_RUST'
set -e
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
cd /workspace/ET-miner
source .venv/bin/activate
command -v maturin &>/dev/null || uv pip install maturin

echo "  [5a] maturin develop --release (et_miner_rust)..."
cd rust_ext && maturin develop --release 2>&1 | tail -3
cd /workspace/ET-miner

echo "  [5b] cargo build --release (af-extract)..."
cd applications/alphafold/af-extract && cargo build --release 2>&1 | tail -3
echo "  af-extract: $(ls -la target/release/af-extract | awk '{print $5, $NF}')"
REMOTE_RUST

# --- [6/8] Verify imports, Rust ext, af-extract binary, GPUs ---
echo ""
echo "[6/8] Verifying imports + af-extract binary + GPU access..."
$SSH bash -s << 'REMOTE_VERIFY'
set -e
cd /workspace/ET-miner
source .venv/bin/activate

VENV_PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"  # CuPy NVRTC needs CUDA headers
NVIDIA_BASE=".venv/lib/python${VENV_PYVER}/site-packages/nvidia"
if [ -d "$NVIDIA_BASE" ]; then
    for d in "$NVIDIA_BASE"/*/lib; do
        [ -d "$d" ] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
fi

python3 -c "
import cupy as cp
n = cp.cuda.runtime.getDeviceCount()
print(f'  CuPy OK | {n} GPU(s)')
for i in range(n):
    with cp.cuda.Device(i):
        free, total = cp.cuda.runtime.memGetInfo()
        print(f'    GPU {i}: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free')
from et_miner.apriori import apriori, _apriori_row_split_multi_gpu
print('  et_miner.apriori + row-split engine: OK')
import et_miner_rust
assert hasattr(et_miner_rust, 'prune_closed_flat_compact'), 'rust closed-filter missing'
print('  Rust extension (prune_closed_flat_compact): OK')
"
test -x applications/alphafold/af-extract/target/release/af-extract \
    && echo "  af-extract binary: OK" \
    || { echo "  af-extract binary MISSING"; exit 1; }
echo ""
echo "ALL IMPORTS + BINARIES VERIFIED"
REMOTE_VERIFY

# --- [7/8] CUDA kernel warm-up ---
echo ""
echo "[7/8] CUDA kernel warm-up (zero-JIT)..."
$SSH bash -s << 'REMOTE_WARMUP'
set -e
cd /workspace/ET-miner
source .venv/bin/activate 2>/dev/null || true
VENV_PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"  # CuPy NVRTC needs CUDA headers
NVIDIA_BASE=".venv/lib/python${VENV_PYVER}/site-packages/nvidia"
if [ -d "$NVIDIA_BASE" ]; then
    for d in "$NVIDIA_BASE"/*/lib; do
        [ -d "$d" ] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
fi
python3 -u scripts/warmup_kernels.py --all-gpus || echo "  (warmup script missing or failed — non-fatal)"
REMOTE_WARMUP

# --- [8/8] Quick-start ---
N_GPUS_HINT="${N_GPUS:-4}"
echo ""
echo "============================================"
echo "  BASE-214M DEPLOY COMPLETE"
echo "  SSH: ssh -p $PORT root@$HOST"
echo "============================================"
cat <<EOF

NEXT (on the box — see RUNBOOK_base214m.md for full detail):

  cd $REMOTE_CODE && source .venv/bin/activate
  export ET_UPLOAD_GCS=1 ET_UPLOAD_TAG=base214m_\$(date +%Y%m%d)
  AFX=applications/alphafold/af-extract/target/release/af-extract

  # (A) Data acquisition — DIRECT on box (aria2c, never via tunnel):
  aria2c -x16 -s16 -d $REMOTE_DATA \\
    'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.dat.gz'
  aria2c -x16 -s16 -d $REMOTE_DATA \\
    'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz'
  # pLDDT from BigQuery (avoids the 23TB CIF); export to $REMOTE_DATA/plddt_metadata.csv:
  bq query --use_legacy_sql=false --format=csv --max_rows=300000000 \\
    'SELECT uniprotAccession, globalMetricValue FROM \`bigquery-public-data.deepmind_alphafold.metadata\`' \\
    > $REMOTE_DATA/plddt_metadata.csv

  # (B) Extraction — base vocab 1006 defined (top-500 Pfam + top-500 GO + 6 pLDDT):
  \$AFX build-from-metadata \\
    --annotations $REMOTE_DATA/uniprot_trembl.dat.gz \\
    --plddt-csv $REMOTE_DATA/plddt_metadata.csv \\
    --top-pfam 500 --top-go 500 \\
    --output $REMOTE_DATA/transactions_214m_base.parquet \\
    --item-mapping $REMOTE_DATA/item_mapping_214m_base.parquet
  # GATE: log multi-feature protein count + item count (new Table 1 numbers).

  # (C) PRE-FLIGHT (Fase 0.3b) — validate row-split BEFORE the 100-perm null:
  python3 applications/alphafold/experiments/validate_row_split.py \\
    --data $REMOTE_DATA/transactions_214m_base.parquet \\
    --subset-size 1000000 --min-count 50 --n-gpus 2 -v
  # GREEN -> run null with --n-gpus $N_GPUS_HINT; RED -> fall back to --n-gpus 1.

  # (D) Experiment suite (all 5 artefact groups -> GCS):
  N_GPUS=$N_GPUS_HINT ET_UPLOAD_GCS=1 \\
    bash applications/alphafold/deploy/run_all_experiments.sh \\
    $REMOTE_DATA/transactions_214m_base.parquet

  # (E) Closed + maximal (Major 5) over the campaign's per-K parquets:
  python3 applications/alphafold/experiments/compute_maximal.py \\
    --results-dir results_214m/parquet \\
    --output-json results_214m/closed_maximal.json -v

EOF
echo "============================================"
