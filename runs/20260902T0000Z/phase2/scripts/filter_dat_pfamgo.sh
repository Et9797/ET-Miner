#!/bin/bash
# Reduces a UniProt DAT (gzip) to the records that carry at least one Pfam or
# GO cross-reference, keeping only the ID, AC, DR Pfam, DR GO and record
# terminator lines, and writes the list of kept primary accessions.
# This is lossless for af-extract's Pfam/GO vocabulary: records without
# Pfam/GO lines cannot contribute items, and InterPro/EC/OC lines are unused.
#
# Usage: filter_dat_pfamgo.sh IN.dat.gz OUT.dat.gz ACCESSIONS.txt
set -o pipefail
IN=$1; OUT=$2; ACC=$3
rm -f "$ACC"
echo "start $(date -u +%FT%TZ) in=$IN"
pigz -dc "$IN" | awk -v ACC="$ACC" '
  function flush() { if (acc != "" && has) { printf "%s//\n", buf; print acc >> ACC; kept++ } buf = ""; acc = ""; has = 0; total++ }
  /^\/\//        { flush(); next }
  /^AC   /       { if (acc == "") { a = $2; sub(/;.*/, "", a); acc = a } buf = buf $0 "\n"; next }
  /^DR   Pfam;/  { has = 1; buf = buf $0 "\n"; next }
  /^DR   GO;/    { has = 1; buf = buf $0 "\n"; next }
  /^ID   /       { buf = buf $0 "\n"; next }
  { next }
  END { printf "records_total=%d records_kept=%d\n", total, kept > "/dev/stderr" }
' | pigz -1 -p 4 > "$OUT.partial"
rc=$?
echo "awk/pigz rc=$rc"
[ $rc -eq 0 ] && mv "$OUT.partial" "$OUT"
ls -la "$OUT" "$ACC"; wc -l < "$ACC" | xargs echo "accessions listed:"
echo "end $(date -u +%FT%TZ)"
