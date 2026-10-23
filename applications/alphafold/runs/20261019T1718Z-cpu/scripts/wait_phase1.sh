#!/bin/bash
# Blocks until the Phase 1 chain ends, the TrEMBL stream dies without output, or a
# watched log stalls; prints why it returned.
#
# Usage: wait_phase1.sh RUN_DIR [STALL_MINUTES]
#   STALL_MINUTES  return if neither the stream progress log nor the chain log has
#                  changed for this long (default 30)
R=$1; STALL=$(( ${2:-30} * 60 ))
P=$R/logs/stream_trembl_2026_01_progress.log
while true; do
  C=$(pgrep -f "phase1_chain.sh $R" | head -1)
  [ -z "$C" ] && { echo "chain ended"; tail -3 "$R"/logs/phase1_chain_*.log; exit 0; }
  if [ ! -f "$R/data/uniprot_trembl_2026_01.reduced.dat.gz" ] && ! pgrep -f stream_tar_member.py > /dev/null; then
    echo "stream not running and no reduced DAT"; tail -3 "$P"; exit 2
  fi
  newest=$(stat -c %Y "$P" "$R"/logs/phase1_chain_*.log "$R"/state/* 2>/dev/null | sort -n | tail -1)
  [ $(( $(date +%s) - newest )) -gt $STALL ] && { echo "stalled"; tail -2 "$P"; exit 3; }
  sleep 60
done
