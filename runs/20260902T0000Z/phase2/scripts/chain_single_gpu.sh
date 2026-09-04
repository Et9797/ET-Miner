#!/usr/bin/env bash
# Re-run the six-threshold campaign and the direct-vs-SON comparison pinned to one GPU
# (CUDA_VISIBLE_DEVICES=0) while sampling both GPUs with nvidia-smi, so that the V2
# paper's "single RTX 3090" description is backed by artifacts.
#
# Usage: chain_single_gpu.sh            (detach with setsid; writes $OUT/chain.done at the end)
# Outputs: $OUT/exp/*.json (experiment results), $OUT/*.log, $OUT/nvidia_smi_*.csv (5 s samples,
#          both devices), $OUT/chain.done.
set -euo pipefail
ROOT=/root/projects/ET-Miner
R=$ROOT/runs/20260902T0000Z
OUT=$R/phase3/2026_01/single_gpu
DATA=$R/phase2/extract_2026_01/transactions_214m_base_multi.parquet
PY=$ROOT/.venv/bin/python
mkdir -p "$OUT/exp"
cd $ROOT/applications/alphafold
export CUDA_VISIBLE_DEVICES=0
export CUDA_PATH=/usr/local/cuda

sample() {  # $1 = csv path; samples both GPUs (index 0 and 1) every 5 s until killed
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 5 > "$1" 2>&1 &
  echo $!
}

echo "chain start $(date -u +%FT%TZ) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
SMI=$(sample "$OUT/nvidia_smi_campaign.csv")
$PY experiments/experiment_full_campaign.py --data "$DATA" --runs 1 --output-dir "$OUT/exp" -v > "$OUT/full_campaign_1gpu.log" 2>&1
kill $SMI || true
echo "campaign done $(date -u +%FT%TZ)"
touch "$OUT/campaign.done"

SMI=$(sample "$OUT/nvidia_smi_direct_vs_son.csv")
$PY experiments/experiment_direct_vs_son.py --data "$DATA" --min-support 0.00001 --runs 1 --output-dir "$OUT/exp" -v > "$OUT/direct_vs_son_1gpu.log" 2>&1
kill $SMI || true
echo "direct_vs_son done $(date -u +%FT%TZ)"
touch "$OUT/chain.done"
