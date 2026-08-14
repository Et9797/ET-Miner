# ET-Miner

**Efficient Transaction Miner** — High-performance frequent itemset mining with a Python/Polars frontend, Rust backend, and multi-GPU CUDA acceleration. Designed for billion-scale datasets with bounded memory via streaming.


## Overview

ET-Miner implements the Apriori algorithm across three performance tiers, all behind a unified API:

| Tier | Stack | Description |
|------|-------|-------------|
| **Tier 1** | Python + Polars | Vectorized boolean matrix operations. Zero dependencies beyond Polars. |
| **Tier 2** | Rust via PyO3 | SIMD-vectorized CSR support counting (AVX2/AVX-512). 80--110x speedup. |
| **Tier 3** | Multi-GPU CUDA | CSR bitvector encoding, fused popcount kernels, zero-transfer GPU-resident mining. |

The streaming engine (SON algorithm) enables bounded-memory processing of arbitrarily large datasets — limited by storage, not RAM.

## Installation

```bash
# Install from GitHub
uv pip install git+https://github.com/Et9797/et-miner.git

# Or clone and install
git clone https://github.com/Et9797/et-miner.git
cd et-miner
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Rust Backend (Tier 2)

```bash
cd rust_ext && maturin develop --release
```

When the Rust extension is installed, ET-Miner automatically uses it for k>2 support counting.

### Optional Dependencies

| Group | Packages | Purpose |
|-------|----------|---------|
| `gpu` | cupy | CUDA GPU acceleration |
| `dev` | pytest, efficient-apriori, psutil, tqdm, sparse-dot-mkl | Development and benchmarking |

```bash
uv pip install -e ".[gpu]"       # GPU support
uv pip install -e ".[dev]"       # Dev + benchmarking
```

## Quick Start

```python
import polars as pl
from et_miner import apriori, generate_rules

# Transaction data
transactions = pl.DataFrame({
    "items": [
        [1, 2, 3],
        [2, 3, 4],
        [1, 3, 5],
        [2, 3],
    ]
})

# Find frequent itemsets (minimum 50% support)
itemsets = apriori(transactions, min_support=0.5)
print(itemsets)
# shape: (5, 2)
# ┌───────────┬─────────┐
# │ itemset   ┆ support │
# │ list[i64] ┆ f64     │
# ╞═══════════╪═════════╡
# │ [3]       ┆ 1.0     │
# │ [2]       ┆ 0.75    │
# │ [1]       ┆ 0.5     │
# │ [2, 3]    ┆ 0.75    │
# │ [1, 3]    ┆ 0.5     │
# └───────────┴─────────┘

# Generate association rules (minimum 70% confidence)
rules = generate_rules(itemsets, min_confidence=0.7)
for rule in rules:
    print(f"{rule.lhs} -> {rule.rhs}: conf={rule.confidence:.2f}, lift={rule.lift:.2f}")
```

### Scaling Up

All tiers share the same API -- just add flags:

```python
# Tier 1: Pure Python + Polars
result = apriori(df, min_support=0.01)

# Tier 2: Rust backend (automatic if installed)
result = apriori(df, min_support=0.01)

# Tier 3: Single GPU
result = apriori(df, min_support=0.01, use_gpu=True)

# Tier 3: Multi-GPU streaming
result = apriori_streaming_multi_gpu(df, min_support=0.001, n_gpus=8, chunk_size=100_000_000)

# Streaming mode for datasets larger than memory
result = apriori(
    pl.scan_parquet("huge_dataset/*.parquet"),
    min_support=0.001,
    streaming=True,
    chunk_size=10_000_000,
    show_progress=True,
)
```

## Features

**Algorithm**
- Apriori with anti-monotone pruning and batched candidate generation
- Association rule generation with confidence, lift, and support metrics
- Arrow-native `List[Int64]` storage throughout (~20 bytes/itemset vs ~300 bytes for Python frozensets)

**Streaming Engine**
- SON (Savasere-Omiecinski-Navathe) algorithm for two-pass chunked mining
- Memory bounded by O(chunk_size x n_items), not O(total_transactions x n_items)
- Processes arbitrarily large datasets with constant memory

**Rust Backend**
- Zero-copy CSR matrix operations via PyO3
- SIMD vectorization (AVX2/AVX-512) for boolean intersection and popcount
- 80--110x speedup over pure Python path

**GPU Acceleration**
- Direct CSR-to-GPU bitvector conversion (bypasses dense matrix construction)
- Fused CUDA kernels: candidate generation + support counting + filtering in a single launch
- GPU-resident mining: zero PCIe transfers between K-levels (~264 bytes total across 22 levels)
- Multi-GPU support with per-device work distribution (tested up to 8x H200)

## AlphaFold Application

Applied to the AlphaFold Protein Structure Database, ET-Miner discovered **26.8 million co-occurrence patterns** across **~76M million predicted protein structures**, reaching feature combinations of size K=22 in 7.3 minutes on a single H100 GPU.

**Problem.** The AlphaFold Database contains predicted protein structures for over 200 million proteins. Which combinations of structural and functional features — Pfam domains, Gene Ontology terms, confidence scores — co-occur across the protein universe? A standard dense boolean matrix for this dataset requires 206 GB, exceeding even high-end GPU memory.

**Solution.** ET-Miner constructs a CSR representation directly from transactions (~5 GB), converts to GPU-resident bitvectors (~26 GB), and performs all Apriori iterations on-GPU with zero PCIe transfers.

### Results

| Metric | Value |
|--------|-------|
| Proteins processed | 214M total, 76.9M with multiple annotations |
| Feature vocabulary | 1,002 items (Pfam domains, GO terms, pLDDT bins) |
| Itemsets discovered | 26.8 million |
| Maximum K | 22 (mathematically proven ceiling) |
| Mining time (deepest tier) | 7.3 minutes on single H100 |
| Support range | 0.1% down to 0.00001% |

The full application pipeline (download, feature extraction, clustering, mining, validation) is described in the accompanying paper.

## Benchmarks

### Billion-Scale Streaming

| Metric | Value |
|--------|-------|
| Transactions | 1,000,000,000 |
| Time | 25.9 minutes |
| Throughput | 643,139 tx/sec |
| Peak memory | 14.76 GB |
| Itemsets found | 326 |
| Hardware | Intel Core Ultra 9 275HX (24 cores), 134 GB RAM |

No published Apriori implementation has processed 1 billion transactions on consumer hardware.

### vs. efficient-apriori (819K transactions)

> System: AMD Ryzen 5 4600G (12 cores), 30 GB RAM, Python 3.14 (free-threading), Polars 1.37, MKL sparse enabled

| Support | Itemsets | efficient-apriori | et-miner | Speedup |
|---------|----------|-------------------|----------|---------|
| 0.005 | 9 | 1.21s / 641 MB | 0.24s / 329 MB | 5.0x |
| 0.001 | 326 | 3.5s / 644 MB | 2.15s / 475 MB | 1.6x |
| 0.0005 | 1,151 | 16.0s / 691 MB | 6.2s / 1113 MB | 2.6x |
| 0.0001 | 11,159 | 214.2s / 2693 MB | 179.9s / 9186 MB | 1.2x |

### vs. efficient-apriori (2.5M transactions)

| Support | Itemsets | efficient-apriori | et-miner | Speedup |
|---------|----------|-------------------|----------|---------|
| 0.005 | 9 | 3.9s / 1663 MB | 0.37s / 539 MB | 10.5x |
| 0.001 | 336 | 12.0s / 1671 MB | 3.5s / 1476 MB | 3.4x |
| 0.0005 | 1,153 | 53.7s / 1704 MB | 12.1s / 2369 MB | 4.4x |
| 0.0001 | 10,894 | 589.4s / 3758 MB | 262.7s / 12163 MB | 2.2x |

ET-Miner is faster across all dataset sizes, with the advantage growing at scale (2-10x).

## API Reference

### `apriori()`

```python
def apriori(
    transactions: pl.DataFrame | pl.LazyFrame,
    min_support: float = 0.5,
    max_length: int | None = None,
    item_col: str = "items",
    use_gpu: bool = False,
    batch_size: int = 10_000,
    sparse: bool | None = None,
    n_jobs: int = 1,
    streaming: bool = False,
    chunk_size: int = 10_000_000,
    memory_budget_gb: float | None = None,
    show_progress: bool = False,
) -> pl.DataFrame
```

Returns a DataFrame with columns `itemset` (`List[Int64]`) and `support` (`Float64`).

### `generate_rules()`

```python
def generate_rules(
    frequent_itemsets: pl.DataFrame,
    min_confidence: float = 0.5,
) -> list[Rule]
```

Returns a list of `Rule` objects with `lhs`, `rhs`, `support`, `confidence`, and `lift`.

## Development

```bash
git clone https://github.com/Et9797/et-miner.git
cd et-miner
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,gpu]"

# Tests
pytest tests/ -v
```

## License

**PolyForm Noncommercial License 1.0.0** — Copyright 2026 Etjen Ahmic. See [LICENSE](LICENSE).

Free for non-commercial use (personal, academic, research, and other
noncommercial purposes). **Commercial use requires a separate license — please
contact the author** via this repository.
