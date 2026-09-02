#!/bin/bash
# Runs the base214m feature extraction (af-extract build-from-metadata) for
# one UniProt release and derives the ≥2-item transaction subset plus the
# dataset statistics the paper reports.
#
# Usage: run_af_extract.sh RELEASE
#   RELEASE   2026_01 or 2025_01; reads
#             phase2/data/uniprot_trembl_RELEASE.reduced.dat.gz and
#             phase2/data/plddt_metadata_storageapi.csv
# Outputs (phase2/extract_RELEASE/):
#   transactions_214m_base.parquet, item_mapping_214m_base.parquet,
#   transactions_214m_base_multi.parquet, af_extract.log, stats.json
set -euo pipefail
REL=$1
RUN_DIR=/root/projects/ET-Miner/runs/20260902T0000Z
D=$RUN_DIR/phase2/data
O=$RUN_DIR/phase2/extract_$REL
AFX=/root/projects/ET-Miner/applications/alphafold/af-extract/target/release/af-extract
mkdir -p "$O"
echo "start $(date -u +%FT%TZ) release=$REL"
ls -la "$D/uniprot_trembl_$REL.reduced.dat.gz" "$D/plddt_metadata_storageapi.csv"
t0=$(date +%s)
RUST_LOG=info "$AFX" build-from-metadata \
  --annotations "$D/uniprot_trembl_$REL.reduced.dat.gz" \
  --plddt-csv   "$D/plddt_metadata_storageapi.csv" \
  --top-pfam 500 --top-go 500 \
  --output       "$O/transactions_214m_base.parquet" \
  --item-mapping "$O/item_mapping_214m_base.parquet" \
  2>&1 | grep -vE 'Parsed [0-9]+K DAT records|Written [0-9]+M transactions|Read [0-9]+M rows' > "$O/af_extract.log"
t1=$(date +%s)
echo "af-extract wall_seconds=$((t1-t0))" | tee -a "$O/af_extract.log"
/root/projects/ET-Miner/.venv/bin/python "$RUN_DIR/phase2/scripts/extract_stats.py" "$O" 2>&1 | tee "$O/extract_stats.log"
echo "end $(date -u +%FT%TZ)"
