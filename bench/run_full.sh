#!/usr/bin/env bash
# Full campaign (~2-4 h): tier gate → complete gpu suite (slow included) →
# full benchmark matrix with A/B axes. Resumable: re-running skips
# completed matrix configs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export NCCL_DEBUG=INFO
# Fresh shells don't inherit setup's environment — re-derive CUDA_PATH.
if [ -z "${CUDA_PATH:-}" ] && [ -d /usr/local/cuda ]; then export CUDA_PATH=/usr/local/cuda; fi

echo "== [1/3] Tier-equivalence gate (CLAUDE.md policy — hard gate) =="
uv run pytest tests/test_tier_equivalence.py -q

echo "== [2/3] Full GPU test suite (slow included) =="
uv run pytest -q -m gpu

echo "== [3/3] Full benchmark matrix =="
uv run python bench/runner.py --mode full --max-hours 3.5
uv run python bench/report.py

echo "== full campaign complete — copy bench/results/ back =="
