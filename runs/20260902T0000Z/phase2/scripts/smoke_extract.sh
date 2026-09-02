#!/bin/bash
# Builds a small smoke dataset from the first records of a (possibly partial)
# reduced TrEMBL stream plus the matching metadata rows, then runs the same
# af-extract command and statistics script used for the full base214m
# extraction, to validate the tooling before the full run.
#
# Usage: smoke_extract.sh RELEASE N_RECORDS
set -uo pipefail
REL=$1; N=$2
RUN_DIR=/root/projects/ET-Miner/runs/20260902T0000Z
D=$RUN_DIR/phase2/data; O=$RUN_DIR/phase2/smoke_$REL; mkdir -p "$O"
AFX=/root/projects/ET-Miner/applications/alphafold/af-extract/target/release/af-extract
SRC=$(ls "$D/uniprot_trembl_$REL.reduced.dat.gz" 2>/dev/null || ls "$D/uniprot_trembl_$REL.reduced.dat.gz.partial")
echo "source: $SRC"
pigz -dc "$SRC" 2>/dev/null | awk -v n="$N" '{print} /^\/\//{c++; if (c>=n) exit}' | pigz -1 > "$O/smoke.dat.gz"
pigz -dc "$O/smoke.dat.gz" | grep -c '^//' | xargs echo "records:"
pigz -dc "$O/smoke.dat.gz" | awk '/^AC   /{split($2,a,";"); print a[1]}' | sort -u > "$O/accessions.txt"
wc -l < "$O/accessions.txt" | xargs echo "unique first accessions:"
/root/projects/ET-Miner/.venv/bin/python - "$D/plddt_metadata_storageapi.csv" "$O/accessions.txt" "$O/plddt_smoke.csv" <<'PY'
import sys, polars as pl
csv, acc, out = sys.argv[1:4]
accs = pl.read_csv(acc, has_header=False, new_columns=["uniprotAccession"])
lf = pl.scan_csv(csv, schema={"uniprotAccession": pl.Utf8, "globalMetricValue": pl.Float64})
sub = lf.join(accs.lazy(), on="uniprotAccession", how="inner").collect(engine="streaming")
sub.write_csv(out)
print("metadata rows matched:", sub.height, " >=50:", int((sub["globalMetricValue"] >= 50).sum()))
PY
RUST_LOG=info "$AFX" build-from-metadata --annotations "$O/smoke.dat.gz" --plddt-csv "$O/plddt_smoke.csv" \
  --top-pfam 500 --top-go 500 --output "$O/transactions_214m_base.parquet" --item-mapping "$O/item_mapping_214m_base.parquet" \
  2>&1 | grep -vE 'Parsed [0-9]+K DAT records|Written [0-9]+M transactions' | tail -15
/root/projects/ET-Miner/.venv/bin/python $RUN_DIR/phase2/scripts/extract_stats.py "$O" | head -30
/root/projects/ET-Miner/.venv/bin/python - "$O" <<'PY'
import sys, polars as pl
o = sys.argv[1]
t = pl.read_parquet(f"{o}/transactions_214m_base.parquet"); m = pl.read_parquet(f"{o}/item_mapping_214m_base.parquet")
print("transactions schema:", t.schema); print(t.head(3)); print("mapping schema:", m.schema); print(m.head(8)); print(m.group_by("feature_category").len())
PY
