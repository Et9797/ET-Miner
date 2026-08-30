#!/usr/bin/env bash
# Smoke campaign (~30-60 min): tier-equivalence gate → gpu tests → small matrix.
# Any gate failure aborts everything — no numbers from a diverging miner.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export NCCL_DEBUG=INFO

echo "== [1/3] Tier-equivalence gate (CLAUDE.md policy — hard gate) =="
uv run pytest tests/test_tier_equivalence.py -q

echo "== [2/3] GPU test suite (not slow) =="
uv run pytest -q -m "gpu and not slow"

echo "== [3/3] Smoke benchmark matrix =="
uv run python bench/runner.py --mode smoke
uv run python bench/report.py

echo "== smoke campaign complete — copy bench/results/ back =="
