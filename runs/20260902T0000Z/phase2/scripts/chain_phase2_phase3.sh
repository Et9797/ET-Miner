#!/bin/bash
# Waits for the reduced TrEMBL stream of one release to finish, then runs the
# full base214m extraction and, if it succeeds, the Phase 3 mining sequence.
#
# Usage: chain_phase2_phase3.sh [RELEASE]      (default 2026_01)
# Logs:  logs/af_extract_RELEASE.log, logs/phase3_RELEASE.log
set -u
REL=${1:-2026_01}
RUN_DIR=/root/projects/ET-Miner/runs/20260902T0000Z
D=$RUN_DIR/phase2/data; S=$RUN_DIR/phase2/scripts; L=$RUN_DIR/logs
echo "chain start $(date -u +%FT%TZ) release=$REL"
until [ -f "$D/uniprot_trembl_$REL.reduced.dat.gz" ]; do
    if ! pgrep -f "stream_trembl_release[.]sh $REL" >/dev/null && [ ! -f "$D/uniprot_trembl_$REL.reduced.dat.gz" ]; then
        echo "stream for $REL is not running and no output exists — abort $(date -u +%FT%TZ)"; exit 1
    fi
    sleep 30
done
echo "stream finished $(date -u +%FT%TZ): $(ls -la $D/uniprot_trembl_$REL.reduced.dat.gz)"
"$S/run_af_extract.sh" "$REL" > "$L/af_extract_$REL.log" 2>&1
rc=$?
echo "extraction rc=$rc $(date -u +%FT%TZ)"
if [ $rc -ne 0 ] || [ ! -f "$RUN_DIR/phase2/extract_$REL/stats.json" ]; then
    echo "extraction failed — Phase 3 not started"; exit 2
fi
"$S/run_phase3.sh" "$REL" > "$L/phase3_$REL.log" 2>&1
echo "phase3 rc=$? $(date -u +%FT%TZ)"
