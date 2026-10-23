#!/bin/bash
# Counts records, primary accessions and distinct Pfam/GO identifiers in a
# (reduced) UniProt DAT gzip, as an independent check of the extractor's
# "Loaded N annotations (P Pfam, G GO)" line.
# Usage: count_dat.sh IN.dat.gz OUT.txt
IN=$1; OUT=$2
echo "start $(date -u +%FT%TZ) in=$IN" > "$OUT"
pigz -dc "$IN" | awk '
  /^\/\//        { records++; acc = ""; next }
  /^AC   /       { if (acc == "") { a1 = $2; sub(/;.*/, "", a1); acc = a1; accs++ } next }
  /^DR   Pfam;/  { split($0, p, ";"); id = p[2]; gsub(/ /, "", id); pf[id] = 1; pfam_lines++; next }
  /^DR   GO;/    { split($0, p, ";"); id = p[2]; gsub(/ /, "", id); go[id] = 1; go_lines++; next }
  END { printf "records=%d primary_accessions=%d pfam_unique=%d go_unique=%d pfam_lines=%d go_lines=%d\n", records, accs, length(pf), length(go), pfam_lines, go_lines }
' >> "$OUT"
echo "end $(date -u +%FT%TZ)" >> "$OUT"
