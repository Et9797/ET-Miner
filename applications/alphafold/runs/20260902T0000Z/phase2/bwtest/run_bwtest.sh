#!/bin/bash
# Times a single proteome-shard download from the public AlphaFold bucket.
export PATH=/root/projects/downloads/google-cloud-sdk/bin:$PATH
DEST=/root/projects/ET-Miner/runs/20260902T0000Z/phase2/bwtest
SRC=gs://public-datasets-deepmind-alphafold-v4/proteomes/proteome-tax_id-9606-12_v4.tar
echo "start $(date -u +%FT%TZ) src=$SRC"
t0=$(date +%s.%N)
gcloud storage cp "$SRC" "$DEST/" 2>&1
rc=$?
t1=$(date +%s.%N)
sz=$(stat -c %s "$DEST/$(basename $SRC)" 2>/dev/null || echo 0)
python3 -c "import sys; sz=$sz; dt=$t1-$t0; print(f'end rc=$rc bytes={sz} seconds={dt:.1f} MB/s={sz/dt/1e6:.1f} Mbit/s={sz*8/dt/1e6:.0f}')"
echo "end $(date -u +%FT%TZ)"
