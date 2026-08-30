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

echo "== /dev/shm: $(df -h /dev/shm | tail -1) =="

# On-device selfcheck: compiles every registered kernel, launches the
# critical ones, prints the device/NCCL/P2P matrix. Fails fast.
uv run python bench/selfcheck.py

# Pre-generate all benchmark datasets (seeded, deterministic).
uv run python -m et_miner.synthetic --preset all --out datasets/synth

echo "== setup complete =="
