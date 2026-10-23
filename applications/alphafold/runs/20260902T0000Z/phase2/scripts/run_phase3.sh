#!/bin/bash
# Phase 3 orchestration for the base214m reproduction: runs every mining
# experiment on the extracted ≥2-item transaction set in sequence. Each step
# writes its own log and a `.done` marker, so re-running the script resumes
# after the last completed step.
#
# Usage: run_phase3.sh [RELEASE]        (default 2026_01)
# Inputs:  phase2/extract_RELEASE/{transactions_214m_base_multi.parquet,
#          transactions_214m_base.parquet, item_mapping_214m_base.parquet, stats.json}
# Outputs: phase3/RELEASE/<step>.log, <step>.done, opus/, exp/
set -u
REL=${1:-2026_01}
RUN_DIR=/root/projects/ET-Miner/runs/20260902T0000Z
X=$RUN_DIR/phase2/extract_$REL
R=$RUN_DIR/phase3/$REL
S=$RUN_DIR/phase2/scripts
DATA=$X/transactions_214m_base_multi.parquet
FULL=$X/transactions_214m_base.parquet
MAP=$X/item_mapping_214m_base.parquet
PY=/root/projects/ET-Miner/.venv/bin/python
export CUDA_PATH=/usr/local/cuda
export ET_UPLOAD_GCS=0
mkdir -p "$R/exp" "$R/opus"
cd /root/projects/ET-Miner/applications/alphafold

step() {
    local name=$1; shift
    if [ -f "$R/$name.done" ]; then echo "skip $name (done)"; return 0; fi
    echo "start $name $(date -u +%FT%TZ)"
    local t0=$(date +%s)
    "$@" > "$R/$name.log" 2>&1
    local rc=$?
    echo "end $name rc=$rc wall_s=$(( $(date +%s) - t0 )) $(date -u +%FT%TZ)"
    if [ $rc -eq 0 ]; then touch "$R/$name.done"; fi
    return $rc
}

N=$($PY -c "import json,sys; print(json.load(open('$X/stats.json'))['n_multi_written'])")
echo "phase3 release=$REL n_multi=$N start $(date -u +%FT%TZ)"

# P3.1 row-split equivalence gate (1M-row subset, 2 GPUs)
step validate_row_split $PY experiments/validate_row_split.py --data "$DATA" --subset-size 1000000 --min-count 50 --n-gpus 2 -v

# P3.2 Opus (support 1e-7 → min_count = ceil(1e-7·n)) with per-K parquet flush
step opus_mining $PY pipeline/run_mining.py --input "$DATA" --item-mapping "$MAP" --support 0.0000001 --max-length 50 --use-gpu --n-gpus 2 --parquet-flush --output-dir "$R/opus" -v

# P3.2b Blitz (support 1e-6) with per-K parquet flush — source of the paper's highlighted patterns (K<=19)
step blitz_mining $PY pipeline/run_mining.py --input "$DATA" --item-mapping "$MAP" --support 0.000001 --max-length 50 --use-gpu --n-gpus 2 --parquet-flush --output-dir "$R/blitz" -v

# P3.7 deepest-itemset analysis on the Opus result (K = highest flushed level)
KMAX=$(ls "$R/opus/parquet" 2>/dev/null | grep -oE 'frequent_k[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)
if [ -n "${KMAX:-}" ]; then
    step k22_analysis $PY experiments/analyze_k22_proteins.py --data "$FULL" --item-mapping "$MAP" --itemset-source "$R/opus/parquet/frequent_k$KMAX.parquet" --output-dir "$R/exp" -v
fi

# P3.4 Direct vs SON at Power (1e-5), script defaults: chunk 40M, local factor 0.9
step direct_vs_son $PY experiments/experiment_direct_vs_son.py --data "$DATA" --min-support 0.00001 --runs 1 --output-dir "$R/exp" -v

# P3.6 permutation null model at min_count 769, 5 permutations, seed 42 (paper protocol)
step null_model_769_5 $PY experiments/experiment_null_model.py --data "$DATA" --min-count 769 --runs 5 --seed 42 --n-gpus 2 --perm-per-gpu --checkpoint --output-dir "$R/exp" -v

# P3.3 six-threshold campaign (exact direct GPU), one run each
step full_campaign_r1 $PY experiments/experiment_full_campaign.py --data "$DATA" --runs 1 --output-dir "$R/exp" -v

# P3.5 SON at Base (1e-3, plus rules at min_conf 0.5 as in the old pipeline log) and Super (1e-4)
step son_base $PY "$S/son_run.py" --data "$DATA" --min-support 0.001 --rules-min-confidence 0.5 --item-mapping "$MAP" --itemsets-out "$R/exp/son_base_itemsets.parquet" --output "$R/exp/son_base.json"
step son_super $PY "$S/son_run.py" --data "$DATA" --min-support 0.0001 --itemsets-out "$R/exp/son_super_itemsets.parquet" --output "$R/exp/son_super.json"

# P3.8 log-only thresholds: min_count 4 and 3 (support chosen so ceil(s·n) equals the target)
S4=$($PY -c "print((4-0.5)/$N)"); S3=$($PY -c "print((3-0.5)/$N)")
step minc4_mining $PY pipeline/run_mining.py --input "$DATA" --item-mapping "$MAP" --support "$S4" --max-length 50 --use-gpu --n-gpus 2 --parquet-flush --output-dir "$R/minc4" -v
step minc3_mining $PY pipeline/run_mining.py --input "$DATA" --item-mapping "$MAP" --support "$S3" --max-length 50 --use-gpu --n-gpus 2 --parquet-flush --output-dir "$R/minc3" -v

# P3.6b null model with 100 permutations (runbook protocol), resumable
step null_model_769_100 $PY experiments/experiment_null_model.py --data "$DATA" --min-count 769 --runs 100 --seed 42 --n-gpus 2 --perm-per-gpu --checkpoint --resume --output-dir "$R/exp_null100" -v

# P3.10 synthetic-preset bench matrix (the surviving bench/results logs are claims from the same GPU class)
step bench_full bash -c "$PY bench/runner.py --mode full --out $R/bench --max-hours 3.5 && $PY bench/report.py --out $R/bench 2>/dev/null || $PY bench/report.py $R/bench"

echo "phase3 end $(date -u +%FT%TZ)"
