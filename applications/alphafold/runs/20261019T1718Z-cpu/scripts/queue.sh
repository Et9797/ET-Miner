#!/bin/bash
# Runs the queue block of PLAN.md in order and skips finished runs.
#
# The block is the fenced section opened by a line "```queue" in PLAN.md; each
# line is "ID | COMMAND", COMMAND run from the scripts directory. A run is done
# when RUN_DIR/state/ID.done exists; a failure writes ID.failed (rc) and the
# queue moves on, unless ID starts with "!" (a gate), which stops the queue.
# Lines starting with "#" are dropped runs and are not executed.
#
# Usage: queue.sh RUN_DIR
set -uo pipefail
R=$(cd "$1" && pwd)
mkdir -p "$R/state" "$R/logs"
cd "$R/scripts"
while IFS='|' read -r id cmd; do
  id=$(echo "$id" | xargs); gate=0
  case $id in !*) gate=1; id=${id#!};; esac
  [ -e "$R/state/$id.done" ] && { echo "$(date -u +%FT%TZ) skip $id (done)"; continue; }
  log=$R/logs/${id}_$(date -u +%Y%m%dT%H%MZ).log
  echo "$(date -u +%FT%TZ) start $id log=$log"
  echo "$cmd" > "$R/state/$id.cmd"
  bash -c "$cmd" > "$log" 2>&1 < /dev/null &
  echo "$!" > "$R/state/$id.pid"
  wait "$!"; rc=$?
  rm -f "$R/state/$id.pid"
  if [ $rc -eq 0 ]; then
    date -u +%FT%TZ > "$R/state/$id.done"; rm -f "$R/state/$id.failed"
    echo "$(date -u +%FT%TZ) done $id"
  else
    echo "rc=$rc $(date -u +%FT%TZ)" > "$R/state/$id.failed"
    echo "$(date -u +%FT%TZ) FAILED $id rc=$rc"
    [ $gate -eq 1 ] && { echo "gate $id failed: queue stopped"; exit 1; }
  fi
done < <(awk '/^```queue/{f=1; next} /^```/{f=0} f && NF && !/^#/' "$R/PLAN.md")
echo "$(date -u +%FT%TZ) queue end"
