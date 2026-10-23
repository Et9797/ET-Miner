#!/bin/bash
# Exports uniprotAccession,globalMetricValue from the AlphaFold BigQuery
# metadata table to CSV using the bq CLI, exactly as the base214m runbook does.
export PATH=/root/projects/downloads/google-cloud-sdk/bin:$PATH
OUT=/root/projects/ET-Miner/runs/20260902T0000Z/phase2/data/plddt_metadata.csv
echo "start $(date -u +%FT%TZ)"
bq query --use_legacy_sql=false --format=csv --max_rows=300000000 \
  'SELECT uniprotAccession, globalMetricValue FROM `bigquery-public-data.deepmind_alphafold.metadata`' \
  > "$OUT.partial" 2> "$OUT.stderr"
rc=$?
echo "bq rc=$rc lines=$(wc -l < "$OUT.partial") bytes=$(stat -c %s "$OUT.partial")"
[ $rc -eq 0 ] && mv "$OUT.partial" "$OUT"
echo "end $(date -u +%FT%TZ)"
