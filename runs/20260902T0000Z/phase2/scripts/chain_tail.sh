#!/bin/bash
# Tail of Phase 3: the 100-permutation null model (2-GPU row-split branch)
# followed by the synthetic bench matrix run from the repository root.
# Usage: chain_tail.sh
set -u
RUN_DIR=/root/projects/ET-Miner/runs/20260902T0000Z
P=$RUN_DIR/phase3/2026_01; X=$RUN_DIR/phase2/extract_2026_01
PY=/root/projects/ET-Miner/.venv/bin/python
export CUDA_PATH=/usr/local/cuda ET_UPLOAD_GCS=0
echo "tail start $(date -u +%FT%TZ)"
cd /root/projects/ET-Miner/applications/alphafold
$PY experiments/experiment_null_model.py --data $X/transactions_214m_base_multi.parquet --min-count 769 --runs 100 --seed 42 --n-gpus 2 --checkpoint --resume --output-dir $P/exp_null100 -v > $P/null_model_769_100.log 2>&1
rc=$?; echo "null100 rc=$rc $(date -u +%FT%TZ)"; [ $rc -eq 0 ] && touch $P/null_model_769_100.done
cd /root/projects/ET-Miner
$PY bench/runner.py --mode full --out $P/bench --max-hours 3.5 > $P/bench_full.log 2>&1
rc=$?; echo "bench runner rc=$rc $(date -u +%FT%TZ)"
$PY bench/report.py --out $P/bench >> $P/bench_full.log 2>&1
echo "bench report rc=$? $(date -u +%FT%TZ)"; [ $rc -eq 0 ] && touch $P/bench_full.done
echo "tail end $(date -u +%FT%TZ)"
