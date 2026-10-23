#!/bin/bash
# Keeps only the DAT records whose primary accession appears in a CSV's
# first column (header skipped). Lossless for af-extract: annotations of
# accessions absent from the pLDDT CSV are never consulted.
#
# Usage: filter_dat_by_accessions.sh IN.dat.gz ACCESSIONS.csv OUT.dat.gz
set -o pipefail
IN=$1; CSV=$2; OUT=$3
echo "start $(date -u +%FT%TZ) in=$IN csv=$CSV"
pigz -dc "$IN" | awk -v CSV="$CSV" '
  BEGIN { while ((getline line < CSV) > 0) { if (line ~ /^uniprotAccession/) continue; split(line, parts, ","); keep[parts[1]] = 1; n++ }; printf "accessions loaded=%d\n", n > "/dev/stderr" }
  function flush() { if (acc != "" && (acc in keep)) { printf "%s//\n", buf; kept++ } buf = ""; acc = ""; total++ }
  /^\/\//   { flush(); next }
  /^AC   /  { if (acc == "") { acc1 = $2; sub(/;.*/, "", acc1); acc = acc1 } buf = buf $0 "\n"; next }
  { buf = buf $0 "\n"; next }
  END { printf "records_total=%d records_kept=%d\n", total, kept > "/dev/stderr" }
' | pigz -1 -p 4 > "$OUT.partial"
rc=$?
echo "awk/pigz rc=$rc"
[ $rc -eq 0 ] && mv "$OUT.partial" "$OUT"
ls -la "$OUT"; echo "end $(date -u +%FT%TZ)"
