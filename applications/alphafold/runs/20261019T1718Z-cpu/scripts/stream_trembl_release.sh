#!/bin/bash
# Streams uniprot_trembl.dat.gz out of a UniProt previous-release knowledgebase
# tarball and writes a reduced copy that keeps only the DAT record lines the
# af-extract parser consults (ID, AC, DE, OC, DR Pfam/GO/InterPro, //).
#
# Usage: stream_trembl_release.sh RUN_DIR RELEASE TARBALL_NAME
#   RUN_DIR       campaign run directory (data/, logs/, scripts/ below it)
#   RELEASE       e.g. 2026_01
#   TARBALL_NAME  file name under previous_releases/release-RELEASE/knowledgebase/
# Options (environment): PARALLEL (default 8), CHUNK_MIB (default 128), WINDOW (default 8)
# Output: RUN_DIR/data/uniprot_trembl_RELEASE.reduced.dat.gz (+ .partial while running)
set -o pipefail
R=$(cd "$1" && pwd); REL=$2; TB=$3
URL=https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-$REL/knowledgebase/$TB
D=$R/data
OUT=$D/uniprot_trembl_$REL.reduced.dat.gz
echo "start $(date -u +%FT%TZ) url=$URL"
python3 "$R/scripts/stream_tar_member.py" "$URL" uniprot_trembl.dat.gz \
    --chunk-mib "${CHUNK_MIB:-128}" --parallel "${PARALLEL:-8}" --window "${WINDOW:-8}" \
    --workdir "$D/stream_work_$REL" --log "$R/logs/stream_trembl_${REL}_progress.log" \
  | pigz -dc \
  | LC_ALL=C grep -E '^(ID   |AC   |DE   |OC   |//|DR   (Pfam|GO|InterPro);)' \
  | pigz -1 -p 4 > "$OUT.partial"
rc=$?
echo "pipeline rc=$rc"
if [ $rc -eq 0 ]; then mv "$OUT.partial" "$OUT"; fi
ls -la "$OUT"* 2>/dev/null
echo "end $(date -u +%FT%TZ)"
