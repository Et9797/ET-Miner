#!/usr/bin/env bash
# Idempotent box setup: env + rust ext + on-device selfcheck + datasets.
# Run from the repo root or bench/ — it normalizes to the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== ET-Miner box setup =="
command -v curl >/dev/null || { apt-get update && apt-get install -y curl; }
command -v cc >/dev/null || { apt-get update && apt-get install -y build-essential; }

# uv
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Python env (dev group included by default) + GPU extra
uv sync --locked --extra gpu

# Rust toolchain + extension. NOTE: build from inside rust_ext — running
# maturin with -m rust_ext/Cargo.toml from the root makes it adopt the ROOT
# pyproject and clobber the editable et-miner install.
if ! command -v cargo >/dev/null && [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"
(cd rust_ext && uv run maturin develop --release)

# NVRTC needs CUDA toolkit headers at JIT time. Runtime/driver-only images
# ship none: point CUDA_PATH at a toolkit when one exists, then probe with a
# trivial RawKernel compile and self-heal via the CUDA header wheels
# (cupy's [ctk] extra — already in the gpu extra, this is belt-and-braces
# for stale lockfiles/checkouts).
if [ -z "${CUDA_PATH:-}" ] && [ -d /usr/local/cuda ]; then
  export CUDA_PATH=/usr/local/cuda
  echo "== CUDA_PATH=$CUDA_PATH (auto-detected) =="
fi

kernel_probe() {
  uv run python - <<'PYEOF'
import cupy as cp
cp.RawKernel('extern "C" __global__ void _probe(int* x) { x[0] = 1; }', "_probe").kernel
print("kernel-compile probe: OK")
PYEOF
}
if ! kernel_probe; then
  echo "== kernel compile failed — installing CUDA header wheels (cupy [ctk] extra) =="
  uv pip install "cupy-cuda12x[ctk]"
  kernel_probe
fi

echo "== /dev/shm: $(df -h /dev/shm | tail -1) =="

# On-device selfcheck: compiles every registered kernel, launches the
# critical ones, prints the device/NCCL/P2P matrix. Fails fast.
uv run python bench/selfcheck.py

# Pre-generate all benchmark datasets (seeded, deterministic).
uv run python -m et_miner.synthetic --preset all --out datasets/synth

echo "== setup complete =="
