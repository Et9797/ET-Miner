#!/usr/bin/env bash
# Execute every tutorial notebook in reading order, in place, from a fresh kernel.
# Stops at the first notebook that raises.
#
# Usage (from anywhere inside the repository):
#   docs/run_notebooks.sh                 # all tiers
#   docs/run_notebooks.sh tier1-polars    # one tier (any subset of the tier folders)
#
# Requirements: the project environment (`uv sync`) and, for tier 2 and 3, the Rust
# extension (`uv run maturin develop --release -m rust_ext/Cargo.toml`). Jupyter is not a
# project dependency; `uv run --with` adds nbconvert and ipykernel for this run only.
set -euo pipefail

DOCS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DOCS/.."

TIERS=("$@")
if [ ${#TIERS[@]} -eq 0 ]; then
  TIERS=(tier1-polars tier2-rust-pyo3 tier3-gpu)
fi

# The sample data is committed; regenerate it only if it is missing.
if [ ! -f "$DOCS/data/smoke.parquet" ] || [ ! -f "$DOCS/data/toy_8x6.parquet" ]; then
  uv run python docs/data/make_samples.py
fi

for tier in "${TIERS[@]}"; do
  for nb in "$DOCS/$tier"/*.ipynb; do
    [ -e "$nb" ] || continue
    echo "==> $tier/$(basename "$nb")"
    start=$(date +%s)
    uv run --with nbconvert --with ipykernel jupyter nbconvert \
      --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=900 \
      --log-level=WARN "$nb"
    echo "    ok in $(( $(date +%s) - start )) s"
  done
done
echo "All notebooks executed."
