#!/bin/bash
# Runs unmodified af-extract build-from-metadata on one shard (CSV + its DAT).
#
# Usage: af_extract_shard.sh AFX DAT CSV OUT_DIR TOP_PFAM TOP_GO
#   writes OUT_DIR/{transactions.parquet, item_mapping.parquet, af_extract.log, time.log}
#   (the log filter and RUST_LOG are those of run_af_extract_lean.sh; time.log holds /usr/bin/time -v)
set -euo pipefail
AFX=$1; DAT=$2; CSV=$3; O=$4; TP=$5; TG=$6
mkdir -p "$O"
t0=$(date +%s)
RUST_LOG=info /usr/bin/time -v -o "$O/time.log" "$AFX" build-from-metadata \
  --annotations "$DAT" \
  --plddt-csv   "$CSV" \
  --top-pfam "$TP" --top-go "$TG" \
  --output       "$O/transactions.parquet" \
  --item-mapping "$O/item_mapping.parquet" \
  2>&1 | grep --line-buffered -vE 'Parsed [0-9]+K DAT records|Written [0-9]+M transactions|Read [0-9]+M rows' > "$O/af_extract.log"
echo "af-extract wall_seconds=$(( $(date +%s) - t0 ))" | tee -a "$O/af_extract.log"
grep -E 'Maximum resident|Exit status' "$O/time.log"
