#!/bin/bash
# Blocks until one of the given queue ids is finished (state/ID.done or
# state/ID.failed) or the queue process has exited, then prints the queue
# log's tail. Used as the background wait between runs. With MAX_MIN set it
# also returns after that many minutes (a progress check on a long run).
#
# Usage: [MAX_MIN=M] wait_queue.sh RUN_DIR QUEUE_PID ID [ID ...]
set -uo pipefail
R=$1; qpid=$2; shift 2
end=$(( $(date +%s) + ${MAX_MIN:-1000000} * 60 ))
while kill -0 "$qpid" 2>/dev/null; do
  if [ "$(date +%s)" -ge "$end" ]; then
    echo "max wait reached"
    exit 0
  fi
  for id in "$@"; do
    if [ -e "$R/state/$id.done" ] || [ -e "$R/state/$id.failed" ]; then
      ls -t "$R"/logs/queue_*.log | head -1 | xargs tail -n 4
      exit 0
    fi
  done
  sleep 60
done
echo "queue $qpid exited"
ls -t "$R"/logs/queue_*.log | head -1 | xargs tail -n 4
