#!/bin/bash
# usage: repro_nnz2.sh <checkout> <label> <mode> <max-runs>  -- stops at the first hang
S=/tmp/claude-0/-root-projects-ET-Miner/1f88fc6e-215b-4f58-b175-c6bc8f7b2cdc/scratchpad
CK=$1; LABEL=$2; MODE=$3; N=$4; EXTRA=${5:-}
cd "$CK" || exit 1
for i in $(seq 1 $N); do
  LOG=$S/trace-$LABEL-$MODE-$i.log
  env -u CONDA_PREFIX POLARS_MAX_THREADS=6 RAYON_NUM_THREADS=6 MKL_NUM_THREADS=6 OMP_NUM_THREADS=6 \
    $EXTRA ET_MINER_TILED_MIN_GROUP_PAIRS=64 ET_MINER_ROW_BALANCE=$MODE \
    NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$S/nccl-$LABEL-$MODE-$i.%p.log \
    timeout 120 uv run --no-sync python $S/repro_nnz2.py "$CK" > "$LOG" 2>&1
  rc=$?
  line=$(grep -E "^OK|hang detected|TIMED OUT|RAISED|Timeout \(" "$LOG" | head -1 | cut -c1-120)
  echo "$LABEL $MODE #$i rc=$rc :: $line"
  grep -q -E "hang detected|TIMED OUT|RAISED|Timeout \(" "$LOG" && break
done
