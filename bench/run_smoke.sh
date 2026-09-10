#!/usr/bin/env bash
# Smoke campaign (~30-60 min): tier-equivalence gate → gpu tests → small matrix.
# Any gate failure aborts everything — no numbers from a diverging miner.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export NCCL_DEBUG=INFO
# Fresh shells don't inherit setup's environment — re-derive CUDA_PATH.
if [ -z "${CUDA_PATH:-}" ] && [ -d /usr/local/cuda ]; then export CUDA_PATH=/usr/local/cuda; fi

echo "== [1/3] Tier-equivalence gate (CLAUDE.md policy — hard gate) =="
uv run pytest tests/test_tier_equivalence.py -q

echo "== [2/3] GPU test suite (not slow) =="
uv run pytest -q -m "gpu and not slow"

echo "== [3/3] Smoke benchmark matrix =="
# Results go in a per-revision directory. The runner resumes from whatever is
# already in --out, which is what a multi-hour campaign needs -- but with one
# shared directory a resumed run replays rows produced by OTHER revisions and
# the summary can only warn about it after the fact. Keyed by revision, a
# resume at the same commit still resumes, and a new commit starts clean, so
# the distinction is structural rather than something a reader has to notice.
# Nested under bench/results/campaign/ deliberately: that path is already
# gitignored, so per-revision dirs need no .gitignore change to stay untracked.
CAMPAIGN_OUT="bench/results/campaign/$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
uv run python bench/runner.py --mode smoke --out "$CAMPAIGN_OUT"
uv run python bench/report.py --out "$CAMPAIGN_OUT"

echo "== smoke campaign complete — copy bench/results/ back =="
