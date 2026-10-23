#!/bin/bash
# Waits until a process exits, then runs a command.
#
# Usage: after_pid.sh PID CMD [ARGS...]
#   PID   process to wait for (polled every 30 s)
#   CMD   command started once PID is gone; it replaces this shell
while kill -0 "$1" 2>/dev/null; do sleep 30; done
shift
echo "start $(date -u +%FT%TZ): $*"
exec "$@"
