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
# The per-revision results directory is derived by the runner itself (see
# runner.py::_campaign_out) and the report reads the same one. It used to be
# computed here, in both scripts, with a bare `git rev-parse --short HEAD` --
# which drops the `-dirty.<digest>` suffix that `_git_rev()` stamps onto every
# row, so rows landed in a directory named for a revision that did not produce
# them.
# The runner's exit code is the campaign's verdict (2: refused to start, git
# could not name the revision or there is no CUDA device; 1: a correctness
# failure, an ungated matrix, a failed config, or a tree edited mid-run).
# Under `set -e` a bare call would stop the script
# there and leave no report for the rows that DID run, so the code is kept
# and the report is written before it is returned.
rc=0
uv run python bench/runner.py --mode smoke || rc=$?
uv run python bench/report.py || true
[ "$rc" -eq 0 ] || exit "$rc"

echo "== smoke campaign complete — copy bench/results/ back =="
