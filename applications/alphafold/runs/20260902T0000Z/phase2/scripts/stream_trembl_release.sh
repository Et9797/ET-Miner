#!/bin/bash
# Streams uniprot_trembl.dat.gz out of a UniProt previous-release knowledgebase
# tarball and writes a reduced copy that keeps only the DAT record lines the
# af-extract parser consults (ID, AC, DE, OC, DR Pfam/GO/InterPro, //).
#
# Usage: stream_trembl_release.sh RELEASE TARBALL_NAME
#   RELEASE       e.g. 2026_01
#   TARBALL_NAME  file name under previous_releases/release-RELEASE/knowledgebase/
# Output: $D/uniprot_trembl_RELEASE.reduced.dat.gz (+ .partial while running)
set -o pipefail
REL=$1; TB=$2
URL=https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-$REL/knowledgebase/$TB
D=/root/projects/ET-Miner/runs/20260902T0000Z/phase2/data
S=/root/projects/ET-Miner/runs/20260902T0000Z/phase2/scripts
OUT=$D/uniprot_trembl_$REL.reduced.dat.gz
echo "start $(date -u +%FT%TZ) url=$URL"
python3 $S/stream_tar_member.py "$URL" uniprot_trembl.dat.gz \
    --chunk-mib 256 --parallel 16 --window 8 \
    --workdir $D/stream_work_$REL --log $D/stream_trembl_$REL.log \
  | pigz -dc \
  | LC_ALL=C grep -E '^(ID   |AC   |DE   |OC   |//|DR   (Pfam|GO|InterPro);)' \
  | pigz -1 -p 4 > "$OUT.partial"
rc=$?
echo "pipeline rc=$rc"
if [ $rc -eq 0 ]; then mv "$OUT.partial" "$OUT"; fi
ls -la "$OUT"* 2>/dev/null
echo "end $(date -u +%FT%TZ)"
