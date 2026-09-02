#!/bin/bash
# Waits for the bench matrix to finish, then runs the GPU test suite once
# (the bench campaign's own pre-step) and records its result.
# Usage: after_bench.sh
RUN_DIR=/root/projects/ET-Miner/runs/20260902T0000Z; P=$RUN_DIR/phase3/2026_01
until [ -f "$P/bench_full.done" ]; do sleep 30; done
echo "bench done; gpu test suite start $(date -u +%FT%TZ)"
cd /root/projects/ET-Miner
CUDA_PATH=/usr/local/cuda .venv/bin/python -m pytest -q -m gpu -p no:cacheprovider > "$P/gpu_test_suite.log" 2>&1
echo "gpu test suite rc=$? $(date -u +%FT%TZ)"; tail -3 "$P/gpu_test_suite.log"; touch "$P/gpu_test_suite.done"
