#!/bin/bash
# Phase 1 data rebuild after the TrEMBL stream: counts, Pfam/GO filter, CSV join,
# accession filter, shard-route extraction (unmodified af-extract per shard +
# merge_shards.py), extract_stats.py and plddt_bin_edges.py.
#
# The filter scripts exit 0 on failure, so each step checks that its outputs exist.
#
# Usage: phase1_chain.sh RUN_DIR
#   Waits for RUN_DIR/data/uniprot_trembl_2026_01.reduced.dat.gz, then runs each step
#   once; a step is done when RUN_DIR/state/p1_<step>.done exists (re-running resumes).
# Options (environment): SHARDS (default 4), SAMPLE_ROWS (default 2000000)
set -euo pipefail
R=$(cd "$1" && pwd); REL=2026_01
S=$R/scripts; D=$R/data; X=$D/extract_$REL; L=$R/logs; ST=$R/state
AFX=$(cd "$R/../../applications/alphafold/af-extract" && pwd)/target/release/af-extract
K=${SHARDS:-4}; NS=${SAMPLE_ROWS:-2000000}
mkdir -p "$X" "$ST"
step() { local id=$1; shift; [ -e "$ST/p1_$id.done" ] && { echo "skip $id"; return 0; }
         echo "start $id $(date -u +%FT%TZ)"; "$@"; date -u +%FT%TZ > "$ST/p1_$id.done"; echo "done $id $(date -u +%FT%TZ)"; }

until [ -f "$D/uniprot_trembl_$REL.reduced.dat.gz" ]; do sleep 60; done
RED=$D/uniprot_trembl_$REL.reduced.dat.gz; PFG=$D/uniprot_trembl_$REL.pfamgo.dat.gz
ACC=$D/dat_pfamgo_accessions_$REL.txt; CSV=$D/plddt_metadata_storageapi.csv
JCSV=$X/plddt_metadata_pfamgo.csv; CSVACC=$D/uniprot_trembl_$REL.csvacc.dat.gz

count_dat() { "$S/count_dat.sh" "$RED" "$D/dat_${REL}_counts.txt"; cat "$D/dat_${REL}_counts.txt"; }
pfamgo() { "$S/filter_dat_pfamgo.sh" "$RED" "$PFG" "$ACC" > "$L/filter_dat_pfamgo.log" 2>&1; cat "$L/filter_dat_pfamgo.log"
           [ -s "$PFG" ] && [ -s "$ACC" ]; }
csvjoin() { "$S/run_py.sh" csv_join.py "$CSV" "$ACC" "$JCSV" > "$L/csv_join.log" 2>&1; cat "$L/csv_join.log"; }
csvacc() { "$S/filter_dat_by_accessions.sh" "$PFG" "$JCSV" "$CSVACC" > "$L/filter_dat_by_accessions.log" 2>&1
           cat "$L/filter_dat_by_accessions.log"; [ -s "$CSVACC" ]; }
shards() {  # $1 = base dir, $2 = CSV, $3 = DAT, $4 = top-pfam/top-go of the merge
  local B=$1 C=$2 T=$3 TOP=$4
  "$S/split_csv_shards.sh" "$C" "$B" "$K"
  for s in $(seq 0 $((K - 1))); do
    [ -s "$B/s$s/shard.dat.gz" ] || "$S/filter_dat_by_accessions.sh" "$T" "$B/s$s/plddt.csv" "$B/s$s/shard.dat.gz" > "$B/s$s/filter.log" 2>&1
    cat "$B/s$s/filter.log"; [ -s "$B/s$s/shard.dat.gz" ]
    [ -s "$B/s$s/transactions.parquet" ] || "$S/af_extract_shard.sh" "$AFX" "$B/s$s/shard.dat.gz" "$B/s$s/plddt.csv" "$B/s$s" 100000 100000
  done
  "$S/run_py.sh" merge_shards.py "$B/merged" $(for s in $(seq 0 $((K - 1))); do echo "$B/s$s"; done) --top-pfam "$TOP" --top-go "$TOP"
}
sample() {
  local B=$X/shardtest; mkdir -p "$B/direct"
  head -n $((NS + 1)) "$JCSV" > "$B/sample.csv"
  "$S/filter_dat_by_accessions.sh" "$CSVACC" "$B/sample.csv" "$B/sample.dat.gz" > "$B/sample_filter.log" 2>&1
  [ -s "$B/sample.dat.gz" ]
  "$S/af_extract_shard.sh" "$AFX" "$B/sample.dat.gz" "$B/sample.csv" "$B/direct" 50 50
  shards "$B" "$B/sample.csv" "$B/sample.dat.gz" 50
  "$S/run_py.sh" verify_shard_route.py "$B/direct" "$B/merged" "$B/verify_shard_route.json"
}
full() { shards "$X/shards" "$JCSV" "$CSVACC" 500
         mv "$X/shards/merged/transactions_214m_base.parquet" "$X/shards/merged/item_mapping_214m_base.parquet" "$X/"
         cp "$X/shards/merged/merge_shards.json" "$X/"; }
stats() { "$S/run_py.sh" extract_stats.py "$X" --full-csv "$CSV" > "$X/extract_stats.log" 2>&1; cat "$X/extract_stats.log"; }
edges() { "$S/run_py.sh" plddt_bin_edges.py "$X" "$CSV" > "$X/plddt_bin_edges.log" 2>&1; tail -3 "$X/plddt_bin_edges.log"; }

step count_dat count_dat &
CPID=$!
step pfamgo pfamgo
step join csvjoin
step csvacc csvacc
step sample sample
step full full
step stats stats
step edges edges
wait $CPID
echo "phase1 chain end $(date -u +%FT%TZ)"
