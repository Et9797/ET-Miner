#!/bin/bash
# Runs a repetition only when the first run of the same id took under 15 minutes.
#
# Reads load_s + mine_wall_s from META (the first run's meta.json). Under 900 s
# it runs COMMAND; otherwise it prints that the run stays a single, flagged
# measurement and exits 0.
#
# Usage: if_short.sh META COMMAND [ARGS...]
set -euo pipefail
meta=$1; shift
wall=$(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['load_s'] + m['mine_wall_s'])" "$meta")
if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < 900 else 1)" "$wall"; then
  exec "$@"
fi
echo "skip repetition: first run took ${wall} s >= 900 s (single run, flagged)"
