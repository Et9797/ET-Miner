#!/bin/bash
# ============================================================
#  ET-MINER — Unified Mining Runner
#  Modes:
#    v2-production  Full K=1->K=max production run (default MAX_K=50)
#    v3-bench       V2 vs V3 benchmark comparison (default MAX_K=8)
#
#  Usage: bash run_mining.sh <mode> [support] [max_k]
#  Example: bash run_mining.sh v2-production 0.00001 50
#           bash run_mining.sh v3-bench 0.00001 8
#
#  Run AFTER deploy_project_milky_way.sh has completed.
#
#  Authors: Et & claudya
#  Co-authored-by: C. claudya <claudya@anthropic.local>
# ============================================================
set -euo pipefail

MODE="${1:-}"
if [[ -z "$MODE" ]] || [[ "$MODE" != "v2-production" && "$MODE" != "v3-bench" ]]; then
    echo "Usage: bash run_mining.sh <mode> [support] [max_k]"
    echo "Modes: v2-production, v3-bench"
    exit 1
fi

SUPPORT="${2:-0.00001}"
DATA="/workspace/data/transactions_35k.parquet"

# Mode-specific defaults
case "$MODE" in
    v2-production)
        MAX_K="${3:-50}"
        OUTPUT="/workspace/results_35k_v2"
        TITLE="ET-MINER V2 PRODUCTION RUN"
        ;;
    v3-bench)
        MAX_K="${3:-8}"
        OUTPUT="/workspace/results_v3_bench"
        TITLE="ET-MINER V3 BENCHMARK"
        ;;
esac

# Input validation — prevent shell/Python injection via CLI args
[[ "$SUPPORT" =~ ^[0-9.eE+-]+$ ]] || { echo "ERROR: SUPPORT must be numeric, got '$SUPPORT'"; exit 1; }
[[ "$MAX_K" =~ ^[0-9]+$ ]] || { echo "ERROR: MAX_K must be integer, got '$MAX_K'"; exit 1; }

cd /workspace/ET-miner
if [ ! -f .venv/bin/activate ]; then
    echo "ERROR: No venv found at /workspace/ET-miner/.venv — run deploy script first"
    exit 1
fi
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

# Detect GPU count
N_GPUS=$(python3 -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())")
[[ "$N_GPUS" =~ ^[0-9]+$ ]] || { echo "ERROR: GPU detection failed, got '$N_GPUS'"; exit 1; }
echo "============================================"
echo "  $TITLE"
echo "  GPUs:    $N_GPUS"
echo "  Data:    $DATA"
echo "  Support: $SUPPORT"
echo "  Max K:   $MAX_K"
echo "  Output:  $OUTPUT"
echo "============================================"

# Print VRAM status (with GPU names)
N_GPUS="$N_GPUS" python3 -c '
import os, cupy as cp
for i in range(int(os.environ["N_GPUS"])):
    with cp.cuda.Device(i):
        free, total = cp.cuda.runtime.memGetInfo()
        name = cp.cuda.runtime.getDeviceProperties(i)["name"].decode()
        print(f"  GPU {i}: {name} — {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
'

echo ""
echo "Starting mining... ($MODE)"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ═══ MODE-SPECIFIC MINING ═══
case "$MODE" in

v2-production)
    DATA="$DATA" SUPPORT="$SUPPORT" MAX_K="$MAX_K" OUTPUT="$OUTPUT" python3 -c '
import time, sys, os
sys.path.insert(0, "src")

from et_miner.apriori import apriori
import polars as pl
import cupy as cp

data = os.environ["DATA"]
output_dir = os.environ["OUTPUT"]
support = float(os.environ["SUPPORT"])
max_k = int(os.environ["MAX_K"])

if not os.path.exists(data):
    print(f"ERROR: Data file not found: {data}")
    sys.exit(1)

print(f"Loading {data}...")
t0 = time.time()
df = pl.scan_parquet(data).select("items").collect(engine="streaming")
print(f"  Loaded {len(df):,} transactions in {time.time()-t0:.1f}s")

n_gpus = cp.cuda.runtime.getDeviceCount()
print(f"Mining with support={support}, max_k={max_k}, output_dir={output_dir}")
print(f"Using {n_gpus} GPUs (row-split multi-GPU)")

result = apriori(
    df,
    min_support=support,
    max_length=max_k,
    use_gpu=True,
    n_gpus=n_gpus,
    output_dir=output_dir,
)
print("Mining complete!")
'
    ;;

v3-bench)
    DATA="$DATA" SUPPORT="$SUPPORT" MAX_K="$MAX_K" OUTPUT="$OUTPUT" python3 -u -c '
import time, sys, os, json
sys.path.insert(0, "src")
from et_miner.apriori import apriori
import polars as pl
import cupy as cp

data = os.environ["DATA"]
support = float(os.environ["SUPPORT"])
max_k = int(os.environ["MAX_K"])
n_gpus = cp.cuda.runtime.getDeviceCount()
output_base = os.environ["OUTPUT"]

if not os.path.exists(data):
    print(f"ERROR: Data file not found: {data}")
    print("Available parquets:")
    for root, dirs, files in os.walk("/workspace"):
        for f in files:
            if f.endswith(".parquet"):
                print(f"  {os.path.join(root, f)}")
    sys.exit(1)

print(f"Loading {data}...")
t0 = time.time()
df = pl.scan_parquet(data).select("items").collect(engine="streaming")
load_time = time.time() - t0
print(f"  {len(df):,} transactions in {load_time:.1f}s")
print()

results = {}

# Test 1: V2 Baseline (no pruning, no sparse)
print("=" * 60)
print("  TEST 1: V2 Baseline (dense only, no pruning)")
print("=" * 60)
out1 = f"{output_base}/test1_baseline"
os.makedirs(out1, exist_ok=True)
t1 = time.time()
r1 = apriori(df, min_support=support, max_length=max_k, use_gpu=True,
             n_gpus=n_gpus, output_dir=out1)
t1_elapsed = time.time() - t1
results["baseline"] = {"time": t1_elapsed, "dir": out1}
print(f"  Baseline: {t1_elapsed:.1f}s")
print()

# Test 2: V3 with pruning (prune_equal_support=True)
print("=" * 60)
print("  TEST 2: V3 Pruning (closed + Apriori subset)")
print("=" * 60)
out2 = f"{output_base}/test2_pruning"
os.makedirs(out2, exist_ok=True)
t2 = time.time()
r2 = apriori(df, min_support=support, max_length=max_k, use_gpu=True,
             n_gpus=n_gpus, output_dir=out2,
             prune_equal_support=True)
t2_elapsed = time.time() - t2
results["pruning"] = {"time": t2_elapsed, "dir": out2}
print(f"  Pruning: {t2_elapsed:.1f}s ({t2_elapsed/t1_elapsed:.2f}x baseline)")
print()

# Test 3: V3 with sparse CSR (sparse_from_k=4)
print("=" * 60)
print("  TEST 3: V3 Sparse CSR (density transition at K=4)")
print("=" * 60)
out3 = f"{output_base}/test3_sparse"
os.makedirs(out3, exist_ok=True)
t3 = time.time()
r3 = apriori(df, min_support=support, max_length=max_k, use_gpu=True,
             n_gpus=n_gpus, output_dir=out3,
             sparse_from_k=4)
t3_elapsed = time.time() - t3
results["sparse"] = {"time": t3_elapsed, "dir": out3}
print(f"  Sparse: {t3_elapsed:.1f}s ({t3_elapsed/t1_elapsed:.2f}x baseline)")
print()

# Test 4: V3 Full (pruning + sparse)
print("=" * 60)
print("  TEST 4: V3 Full (pruning + sparse)")
print("=" * 60)
out4 = f"{output_base}/test4_full_v3"
os.makedirs(out4, exist_ok=True)
t4 = time.time()
r4 = apriori(df, min_support=support, max_length=max_k, use_gpu=True,
             n_gpus=n_gpus, output_dir=out4,
             prune_equal_support=True, sparse_from_k=4)
t4_elapsed = time.time() - t4
results["full_v3"] = {"time": t4_elapsed, "dir": out4}
print(f"  Full V3: {t4_elapsed:.1f}s ({t4_elapsed/t1_elapsed:.2f}x baseline)")
print()

# Summary
print("=" * 60)
print("  BENCHMARK SUMMARY")
print("=" * 60)
hdr_test = "Test"
hdr_time = "Time"
hdr_speedup = "Speedup"
sep = "-"
print(f"  {hdr_test:<25} {hdr_time:>10} {hdr_speedup:>10}")
print(f"  {sep*25} {sep*10} {sep*10}")
for name, r in results.items():
    t = r["time"]
    speedup = t1_elapsed / t if t > 0 else float("inf")
    print(f"  {name:<25} {t:>9.1f}s {speedup:>9.2f}x")

# Count itemsets per K for each test
print()
hdr_k = ["K=1", "K=2", "K=3", "K=4", "K=5+"]
print(f"  {hdr_test:<25} {hdr_k[0]:>8} {hdr_k[1]:>8} {hdr_k[2]:>8} {hdr_k[3]:>8} {hdr_k[4]:>8}")
print(f"  {sep*25} {sep*8} {sep*8} {sep*8} {sep*8} {sep*8}")
for name, r in results.items():
    counts = {}
    d = r["dir"]
    for k in range(1, max_k + 1):
        pf = f"{d}/frequent_k{k}.parquet"
        if os.path.exists(pf):
            counts[k] = len(pl.read_parquet(pf))
    k5plus = sum(v for k, v in counts.items() if k >= 5)
    print(f"  {name:<25} {counts.get(1,0):>8,} {counts.get(2,0):>8,} {counts.get(3,0):>8,} {counts.get(4,0):>8,} {k5plus:>8,}")

# Save results
with open(f"{output_base}/benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to {output_base}/benchmark_results.json")
'

    # BigQuery upload (if configured)
    if command -v bq &>/dev/null && gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q .; then
        GCP_PROJECT=$(gcloud config get project 2>/dev/null || echo "")
        if [ -n "$GCP_PROJECT" ]; then
            RUN_TAG="v3bench_$(date +%Y%m%d_%H%M)"
            DATASET="${GCP_PROJECT}:mining_results"

            echo ""
            echo "========================================"
            echo "  LOADING RESULTS INTO BIGQUERY"
            echo "  Dataset: $DATASET"
            echo "========================================"

            bq mk --dataset --location=europe-west4 "$DATASET" 2>/dev/null || true

            for pq_file in "$OUTPUT"/test*/*.parquet; do
                [ -f "$pq_file" ] || continue
                TABLE_NAME=$(basename "$pq_file" .parquet)
                echo "  Loading $TABLE_NAME..."
                bq load --source_format=PARQUET --autodetect --replace \
                    "${DATASET}.${RUN_TAG}_${TABLE_NAME}" "$pq_file" 2>&1 | tail -1
            done

            echo ""
            echo "  BigQuery tables loaded. Query with:"
            echo "  bq query 'SELECT * FROM \`${DATASET}.${RUN_TAG}_frequent_k8\` LIMIT 10'"
        fi
    else
        echo ""
        echo "  BigQuery not configured — results are local only."
        echo "  (Per-K uploads to GCS happen during mining if ET_UPLOAD_GCS=1)"
    fi
    ;;

esac

echo ""
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results in: $OUTPUT"
ls -lh "$OUTPUT"/frequent_k*.parquet 2>/dev/null || ls -lh "$OUTPUT"/test*/*.parquet 2>/dev/null || echo "  No parquet files found"
echo "  Done!"
