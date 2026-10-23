#!/bin/bash
# Splits a headed CSV into K contiguous row ranges, each with the header.
#
# Usage: split_csv_shards.sh IN.csv OUT_DIR K
#   writes OUT_DIR/s0/plddt.csv .. OUT_DIR/s{K-1}/plddt.csv (ceil(rows/K) rows each, the last fewer)
set -euo pipefail
IN=$1; OUT=$2; K=$3
N=$(( $(wc -l < "$IN") - 1 ))
PER=$(( (N + K - 1) / K ))
for s in $(seq 0 $((K - 1))); do mkdir -p "$OUT/s$s"; done
awk -v PER="$PER" -v OUT="$OUT" 'NR == 1 { hdr = $0; next }
  { s = int((NR - 2) / PER); f = OUT "/s" s "/plddt.csv"
    if (!(s in seen)) { print hdr > f; seen[s] = 1 } print > f }' "$IN"
echo "rows=$N per_shard=$PER"
for s in $(seq 0 $((K - 1))); do echo "s$s $(( $(wc -l < "$OUT/s$s/plddt.csv") - 1 ))"; done
