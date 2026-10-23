#!/bin/bash
# Third chain: filters the Pfam/GO DAT to the accessions present in the
# joined pLDDT CSV, runs the memory-reduced extraction on it, then Phase 3.
# Usage: chain3.sh [RELEASE]
set -u
REL=${1:-2026_01}
RUN_DIR=/root/projects/ET-Miner/runs/20260902T0000Z
D=$RUN_DIR/phase2/data; S=$RUN_DIR/phase2/scripts; L=$RUN_DIR/logs; X=$RUN_DIR/phase2/extract_$REL
echo "chain3 start $(date -u +%FT%TZ) release=$REL"
"$S/filter_dat_by_accessions.sh" "$D/uniprot_trembl_$REL.pfamgo.dat.gz" "$X/plddt_metadata_pfamgo.csv" "$D/uniprot_trembl_$REL.csvacc.dat.gz" > "$L/filter_dat_by_accessions.log" 2>&1
rc=$?; cat "$L/filter_dat_by_accessions.log"
{ [ $rc -eq 0 ] && [ -f "$D/uniprot_trembl_$REL.csvacc.dat.gz" ]; } || { echo "filter failed"; exit 1; }
rm -f "$X"/transactions_214m_base*.parquet "$X"/item_mapping_214m_base.parquet "$X"/stats.json "$X"/af_extract.log
DAT_FILE="$D/uniprot_trembl_$REL.csvacc.dat.gz" "$S/run_af_extract_lean.sh" "$REL" > "$L/af_extract_lean_$REL.log" 2>&1
rc=$?
echo "extraction rc=$rc $(date -u +%FT%TZ)"
if [ $rc -ne 0 ] || [ ! -f "$X/stats.json" ]; then echo "extraction failed — Phase 3 not started"; exit 2; fi
"$S/run_phase3.sh" "$REL" > "$L/phase3_$REL.log" 2>&1
echo "phase3 rc=$? $(date -u +%FT%TZ)"
