#!/bin/bash
# Memory-reduced base214m extraction for one UniProt release: joins the
# metadata CSV to the accessions that carry Pfam/GO records, runs af-extract
# on that subset with the canonical parameters, and reconstructs the
# full-set statistics from the full CSV. Output-equivalent to the full run
# for the mined (>=2-item) set and for every reported dataset statistic.
#
# Usage: [DAT_FILE=path] run_af_extract_lean.sh RELEASE
# Inputs:  phase2/data/uniprot_trembl_RELEASE.pfamgo.dat.gz,
#          phase2/data/dat_pfamgo_accessions_RELEASE.txt,
#          phase2/data/plddt_metadata_storageapi.csv
# Outputs: phase2/extract_RELEASE/{plddt_metadata_pfamgo.csv, transactions_214m_base.parquet,
#          item_mapping_214m_base.parquet, transactions_214m_base_multi.parquet, af_extract.log, stats.json}
set -euo pipefail
REL=$1
RUN_DIR=/root/projects/ET-Miner/runs/20260902T0000Z
D=$RUN_DIR/phase2/data
O=$RUN_DIR/phase2/extract_$REL
AFX=/root/projects/ET-Miner/applications/alphafold/af-extract/target/release/af-extract
PY=/root/projects/ET-Miner/.venv/bin/python
DAT=${DAT_FILE:-$D/uniprot_trembl_$REL.pfamgo.dat.gz}
mkdir -p "$O"
echo "start $(date -u +%FT%TZ) release=$REL dat=$DAT"
[ -s "$O/plddt_metadata_pfamgo.csv" ] && echo "csv join output exists, reusing" || $PY - "$D/plddt_metadata_storageapi.csv" "$D/dat_pfamgo_accessions_$REL.txt" "$O/plddt_metadata_pfamgo.csv" <<'PY'
import sys, time, polars as pl
csv, acc, out = sys.argv[1:4]
t0 = time.time()
accs = pl.scan_csv(acc, has_header=False, new_columns=["uniprotAccession"], schema={"uniprotAccession": pl.Utf8}).unique()
lf = pl.scan_csv(csv, schema={"uniprotAccession": pl.Utf8, "globalMetricValue": pl.Float64}).filter(pl.col("globalMetricValue") >= 50)
joined = lf.join(accs, on="uniprotAccession", how="inner")
joined.sink_csv(out)
n = pl.scan_csv(out, schema={"uniprotAccession": pl.Utf8, "globalMetricValue": pl.Float64}).select(pl.len()).collect().item()
print(f"csv join: {n} rows with pLDDT>=50 and a Pfam/GO DAT record -> {out} ({time.time()-t0:.0f}s)")
PY
t0=$(date +%s)
RUST_LOG=info "$AFX" build-from-metadata \
  --annotations "$DAT" \
  --plddt-csv   "$O/plddt_metadata_pfamgo.csv" \
  --top-pfam 500 --top-go 500 \
  --output       "$O/transactions_214m_base.parquet" \
  --item-mapping "$O/item_mapping_214m_base.parquet" \
  2>&1 | grep --line-buffered -vE 'Parsed [0-9]+K DAT records|Written [0-9]+M transactions|Read [0-9]+M rows' > "$O/af_extract.log"
t1=$(date +%s)
echo "af-extract wall_seconds=$((t1-t0))" | tee -a "$O/af_extract.log"
$PY "$RUN_DIR/phase2/scripts/extract_stats.py" "$O" --full-csv "$D/plddt_metadata_storageapi.csv" 2>&1 | tee "$O/extract_stats.log"
echo "end $(date -u +%FT%TZ)"
