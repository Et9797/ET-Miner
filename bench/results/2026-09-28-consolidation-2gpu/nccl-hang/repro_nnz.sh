#!/bin/bash
# usage: repro_nnz.sh <checkout> <label> <runs-per-mode>
S=/tmp/claude-0/-root-projects-ET-Miner/1f88fc6e-215b-4f58-b175-c6bc8f7b2cdc/scratchpad
CK=$1; LABEL=$2; N=$3
cd "$CK" || exit 1
for mode in nnz rows; do
  for i in $(seq 1 $N); do
    LOG=$S/repro-$LABEL-$mode-$i.log
    env -u CONDA_PREFIX POLARS_MAX_THREADS=6 RAYON_NUM_THREADS=6 MKL_NUM_THREADS=6 OMP_NUM_THREADS=6 \
      ET_MINER_KERNEL_VARIANT=legacy ET_MINER_TILED_MIN_GROUP_PAIRS=64 ET_MINER_ROW_BALANCE=$mode \
      timeout 100 uv run --no-sync python $S/repro_nnz.py "$CK" > "$LOG" 2>&1
    rc=$?
    line=$(grep -E "^OK|^Fatal Python error|Traceback|Error" "$LOG" | head -1 | cut -c1-100)
    echo "$LABEL $mode #$i rc=$rc :: $line"
  done
done
