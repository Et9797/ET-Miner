# Tier 2.1: Build and install the Rust extension

The compiled tier is a separate Python extension module, `et_miner_rust`, built from
`rust_ext/` with [maturin](https://www.maturin.rs/) and [PyO3](https://pyo3.rs/).
ET-Miner works without it; with it installed, some counting moves into compiled code and
`et_miner.apriori_from_csr` becomes available.

This page lists the toolchain, the build command, how to check the result and the errors
you are most likely to meet.

## 1. What you need

| Tool | Why | How to get it |
|---|---|---|
| Python 3.10 or newer | `requires-python = ">=3.10"` in `pyproject.toml` | system or `uv python install` |
| [uv](https://docs.astral.sh/uv/) | project environment and runner | see the uv documentation |
| Rust toolchain (`rustc`, `cargo`) | compiles `rust_ext/` | [rustup](https://rustup.rs/) |
| maturin | builds the extension into the venv | part of the `dev` dependency group, installed by `uv sync` |

The crate's dependencies are pinned in `rust_ext/Cargo.toml` and `rust_ext/Cargo.lock`:
`pyo3` and `numpy` 0.27 for the Python boundary, `rayon` for threads, `bitvec` and `wide` for bit and SIMD operations.

## 2. Build

Run every command from the repository root.

```bash
uv venv
uv sync                                                  # package + dev group, including maturin
uv run maturin develop --release -m rust_ext/Cargo.toml  # compile and install et_miner_rust
```

The build command is also stored in the source as `et_miner.backends.BUILD_COMMAND`, so
error messages print the same line.

For a build tuned to the CPU you build on (AVX-512 where present), set `RUSTFLAGS`:

```bash
RUSTFLAGS="-C target-cpu=native" uv run maturin develop --release -m rust_ext/Cargo.toml
```

A native build may not run on a different CPU. The default build is portable.

## 3. Check the build

```bash
uv run python -c "import et_miner; print(et_miner.HAS_RUST, et_miner.get_rust_version())"
uv run et-miner info
```

The first line prints `True` and the extension version (the `version` in `rust_ext/Cargo.toml`).
`et-miner info` lists the extension under `Backends` as `rust: <version>`.
Notebook [02-inside-the-rust-core](02-inside-the-rust-core.ipynb) repeats this check in its setup cell.

To run the Rust unit tests of the crate itself:

```bash
cargo test --manifest-path rust_ext/Cargo.toml
```

## 4. Common errors and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `HAS_RUST` is `False` after a `uv sync` that ran after the build | `uv sync` makes the environment match the lockfile exactly, and the extension is not in the lockfile, so it is uninstalled | Rebuild with the command in section 2, or sync with `uv sync --inexact`, which leaves extra packages in place. `uv run` does not remove it. |
| A second `.venv` appears inside `rust_ext/` | maturin or uv was run from inside `rust_ext/`, which has its own `pyproject.toml` | Delete `rust_ext/.venv` and run the build from the repository root with `-m rust_ext/Cargo.toml` |
| maturin refuses to run because both `VIRTUAL_ENV` and `CONDA_PREFIX` are set | a conda shell is active; uv exports `VIRTUAL_ENV` | Prefix the build with `env -u CONDA_PREFIX`. Setting `CONDA_PREFIX=` (empty) is not enough. |
| uv or the shell cannot find `maturin` | the `dev` group is not installed | `uv sync` (the dev group is included by default) |
| `error: linker 'cc' not found` or no `cargo` | no Rust toolchain or C linker | install rustup and your platform's C build tools |
| A warning that `et_miner_rust is stale` | the installed extension predates a function the Python code probes for (`src/et_miner/gpu/kernels/k3plus.py`) | rebuild with the command in section 2 |
| `TypeError: argument 'csr_indptr': 'ndarray' object cannot be cast as 'ndarray'` | a Rust function received an `int32` array where it expects `int64` | pass `np.asarray(x, dtype=np.int64)`; see [02-inside-the-rust-core](02-inside-the-rust-core.ipynb), section 2 |

## 5. What the extension changes

- `apriori(..., sparse=True)` and the automatic sparse choice count itemsets of length 3 and more with
  `et_miner_rust.count_itemsets_simd` (`src/et_miner/core/sparse.py:_count_support_sparse_k_gt_2_rust`).
- `et_miner.apriori_from_csr` runs the whole Apriori loop in Rust on a CSR matrix.
  Without the extension it is a stub that raises `MiningError` with the build command.
- The GPU paths use it for candidate grouping and pruning (`prune_groups_apriori`,
  `build_k3plus_groups_from_flat`); without it they fall back to slower Python code with the same results.

Results are identical with and without the extension.
Notebook [03-same-results-and-speed](03-same-results-and-speed.ipynb) checks that and measures the speed.

Next: [02-inside-the-rust-core](02-inside-the-rust-core.ipynb).
