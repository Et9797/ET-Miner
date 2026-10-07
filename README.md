# ET-Miner

**Efficient Transaction Miner** — exact frequent itemset mining (Apriori) and
association rules with a Polars frontend, an optional Rust backend and a CUDA
miner for one or more GPUs. Every route returns the same itemsets and counts;
`tests/test_tier_equivalence.py` checks each of them against efficient-apriori.

| Route | Stack | Selected by |
|-------|-------|-------------|
| CPU | Polars input; one CSR of the frequent items, counted with NumPy/SciPy (Gram matrices, bitvectors) | default |
| GPU | CuPy kernels on transaction shards, one per GPU, counts summed with NCCL | `use_gpu=True`, `n_gpus=` |
| Streaming | SON two-pass chunked mining, memory bounded by the chunk size | `streaming=True`, `chunk_size=` |

On the GPU, K≥3 candidates are enumerated inside the kernels from prefix
groups. A candidate with an infrequent (k−1)-subset is skipped on the device
(`prune_apriori`, on by default), a candidate whose count follows from a
non-free subset can be inferred instead of counted (`use_generator_pruning`),
and deep levels can switch from bitvectors to sparse tidsets (`sparse_from_k`).

## Installation

```bash
uv pip install git+https://github.com/Et9797/et-miner.git                    # CPU (Polars)
uv pip install "et-miner[gpu] @ git+https://github.com/Et9797/et-miner.git"  # + CuPy, NCCL
```

The Rust extension is built locally, from the repository root (a cwd inside
`rust_ext` makes `uv` build a second virtualenv; in a conda shell prefix the
build with `env -u CONDA_PREFIX`):

```bash
git clone https://github.com/Et9797/et-miner.git && cd et-miner
uv sync                                                   # package + dev group
uv run maturin develop --release -m rust_ext/Cargo.toml
```

It changes no result, only the time. The CPU route does not use it; SON
streaming's sparse counter and the GPU route's host steps do. In the
consolidation campaign (`bench/consolidation/REPORT.md`) the GPU route's host
steps were 2–27× slower per call without it.
Another project depends on it explicitly, pinned to the engine's revision:
`uv add "et_miner_rust @ git+https://github.com/Et9797/et-miner.git@<rev>#subdirectory=rust_ext"`.

## Quick start

```python
import polars as pl
from et_miner import apriori, generate_rules

transactions = pl.DataFrame({"items": [[1, 2, 3], [2, 3, 4], [1, 3, 5], [2, 3]]})
itemsets = apriori(transactions, min_support=0.5)
# itemset  support
# [1]      0.5
# [2]      0.75
# [3]      1.0
# [1, 3]   0.5
# [2, 3]   0.75

for rule in generate_rules(itemsets, min_confidence=0.7):
    print(f"{rule.lhs} -> {rule.rhs}: conf={rule.confidence:.2f}, lift={rule.lift:.2f}")
# [1] -> [3]: conf=1.00, lift=1.00
# [2] -> [3]: conf=1.00, lift=1.00
# [3] -> [2]: conf=0.75, lift=1.00
```

`apriori()` returns `itemset` (`List[Int64]`, items ascending) and `support`
(`Float64`). The main options (all documented in its docstring):

| Option | Effect |
|--------|--------|
| `min_support`, `max_length` | threshold (count ≥ ⌈support·N⌉) and depth |
| `use_gpu`, `n_gpus`, `bitvecs=` | GPU route; prebuilt bitvectors skip the CSR build |
| `streaming`, `chunk_size` | SON on chunks, on the CPU or the GPU |
| `prune_equal_support` | return the free-sets (generators) instead of the complete lattice |
| `prune_apriori`, `use_generator_pruning` | device-side subset test (default on), count inference |
| `sparse_from_k` | GPU dense→sparse transition (`"auto"` or a level) |
| `output_dir`, `resume_from_k` | flush each level to parquet, resume from a flushed level |

`et_miner.HAS_RUST`, `et_miner.HAS_GPU`, `et_miner.has_cupy()` and
`et_miner.get_gpu_count()` report what the environment supports; so does
`et-miner info`.

## CLI

```bash
et-miner mine -i transactions.parquet -o rules.json --min-support 0.01 --min-confidence 0.6
et-miner info
```

Configuration comes from `et-miner.toml`, `~/.config/et-miner/config.toml` and
`ET_MINER_*` variables; every `ET_*` knob is documented in `src/et_miner/_env.py`.

## Measurements

Each campaign in `bench/results/` carries its protocol, raw rows and
environment (`bench/README.md`).

From the CPU-tier campaign (`bench/cpu/PROTOCOL.md`,
`bench/results/2026-10-07-cpu-phase1/compare.md`; 4 vCPU Intel Xeon @ 2.10 GHz,
median of 3 runs, seconds of the mining call with the data loaded), the CPU
route against efficient-apriori 2.0.6 on the same machine:

| Workload | efficient-apriori | ET-Miner, 1 thread | ET-Miner, `n_jobs=4` |
|----------|-------------------|--------------------|----------------------|
| smoke (60K rows, support 0.01) | 0.56 s | 0.12 s | 0.12 s |
| deep_k (1M rows, 0.02) | 95.2 s | 2.7 s | 2.7 s |
| skewed_rows (1M rows, 0.02) | 161.5 s | 3.8 s | 2.9 s |
| wide_vocab (100K rows, 0.004) | 60.1 s | 1.3 s | 0.55 s |
| Online Retail II (36K invoices, 0.002) | 170.2 s | 7.3 s | 5.6 s |
| Online Retail II (0.0001, K≤2) | 51.6 s | 1.9 s | 0.96 s |

Both return the same itemsets and counts. From the pruning campaign
(`bench/pruning/REPORT.md`, 2× RTX A4000, median of 3 runs, stress_k2 without
the subset test one run):

| Workload | Without subset test | With subset test | With count inference |
|----------|---------------------|------------------|----------------------|
| oom_regression to K=3 (500K rows), 1 GPU | 25.8 s | 7.2 s | 7.2 s |
| stress_k2 to K=3 (2M rows), 1 GPU | 496.3 s | 52.8 s | — |
| deep_sparse_large (20M rows, K=16), 1 GPU | 24.4 s | 25.0 s | 20.2 s |

## Development

```bash
uv sync
uv run pytest -q -m "not slow"     # gpu-marked tests skip without a CUDA device
uv run ruff check src tests bench
python datasets/prepare_online_retail.py              # Online Retail II for the smoke tests
uv run python -m et_miner.synthetic --preset all --out datasets/synth
```

Python ≥ 3.10. Contributor rules (the correctness oracle, kernel constraints)
are in `CLAUDE.md`; the GPU benchmark campaign in `bench/README.md`.

## License

**PolyForm Noncommercial License 1.0.0**. See [LICENSE](LICENSE).

Free for non-commercial use (personal, academic, research, and other
noncommercial purposes). **Commercial use requires a separate license — please
contact the author** via this repository.
