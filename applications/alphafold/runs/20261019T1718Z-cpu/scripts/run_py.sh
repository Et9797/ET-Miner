#!/bin/bash
# Runs a harness script with the pinned engine venv, no CUDA device and every
# thread pool pinned to THREADS.
#
# Usage: run_py.sh SCRIPT.py [ARGS...] | run_py.sh -m MODULE [ARGS...]
#   SCRIPT.py  a script in this directory (or a path)
# Options (environment): THREADS (default 6), ENGINE (default ../../../../ET-Miner-engine)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ENGINE=${ENGINE:-$(cd "$HERE/../../../../ET-Miner-engine" && pwd)}
export THREADS=${THREADS:-6}
export CUDA_VISIBLE_DEVICES=
export POLARS_MAX_THREADS=$THREADS RAYON_NUM_THREADS=$THREADS OMP_NUM_THREADS=$THREADS \
       MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
unset VIRTUAL_ENV ET_UPLOAD_GCS
for v in $(env | sed -n 's/^\(ET_MINER_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
S=$1; shift
case $S in -*) exec "$ENGINE/.venv/bin/python" "$S" "$@";; esac
[ -f "$S" ] || S=$HERE/$S
exec "$ENGINE/.venv/bin/python" "$S" "$@"
