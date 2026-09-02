#!/bin/bash
# Second chain: waits for the Pfam/GO-filtered DAT of one release, runs the
# memory-reduced extraction, then the Phase 3 mining sequence.
# Usage: chain2.sh [RELEASE]
set -u
REL=${1:-2026_01}
RUN_DIR=/root/projects/ET-Miner/runs/20260902T0000Z
D=$RUN_DIR/phase2/data; S=$RUN_DIR/phase2/scripts; L=$RUN_DIR/logs
echo "chain2 start $(date -u +%FT%TZ) release=$REL"
until [ -f "$D/uniprot_trembl_$REL.pfamgo.dat.gz" ]; do
    pgrep -f "filter_dat_pfamgo[.]sh" >/dev/null || { [ -f "$D/uniprot_trembl_$REL.pfamgo.dat.gz" ] || { echo "filter not running and no output — abort"; exit 1; }; }
    sleep 20
done
echo "filter finished $(date -u +%FT%TZ)"; cat $L/filter_dat_pfamgo.log
rm -rf "$RUN_DIR/phase2/extract_$REL"
"$S/run_af_extract_lean.sh" "$REL" > "$L/af_extract_lean_$REL.log" 2>&1
rc=$?
echo "extraction rc=$rc $(date -u +%FT%TZ)"
if [ $rc -ne 0 ] || [ ! -f "$RUN_DIR/phase2/extract_$REL/stats.json" ]; then echo "extraction failed — Phase 3 not started"; exit 2; fi
"$S/run_phase3.sh" "$REL" > "$L/phase3_$REL.log" 2>&1
echo "phase3 rc=$? $(date -u +%FT%TZ)"
