# ET-Miner

**Efficient Transaction Miner** — High-performance frequent itemset mining with a Python/Polars frontend, Rust backend, and multi-GPU CUDA acceleration. Designed for billion-scale datasets with bounded memory via streaming.


## Overview

ET-Miner implements the Apriori algorithm across three performance tiers, all behind a unified API:

| Tier | Stack | Description |
|------|-------|-------------|
| **Tier 1** | Python + Polars | Vectorized boolean matrix operations. Zero dependencies beyond Polars. |
| **Tier 2** | Rust via PyO3 | SIMD-vectorized CSR support counting (AVX2/AVX-512). 80--110x speedup. |
| **Tier 3** | Multi-GPU CUDA | CSR bitvector encoding, fused popcount kernels, GPU-resident mining (bitvectors uploaded once, metadata-only transfers per level). |

The streaming engine (SON algorithm) enables bounded-memory processing of arbitrarily large datasets — limited by storage, not RAM.

## Package layout

```
src/et_miner/
├── core/        Apriori loop, candidate generation, matrix/sparse counting, rules
├── gpu/         CUDA kernels (sources in gpu/kernels/_src/*.cu), bitvec mining,
│                multi-GPU row-split, NCCL, dispatch heuristics
├── streaming/   SON streaming (son), multi-GPU streaming, CUDA-streams pipeline, ramdisk
├── io/          parquet flush-to-disk, Google Cloud Storage upload
├── backends.py  single source of truth for CuPy/Rust capability detection
└── _env.py      every ET_* environment knob, documented in one place
```

## Installation

### Tier 1 — pure Python (works everywhere)

```bash
uv pip install git+https://github.com/Et9797/et-miner.git
```

This installs the Polars-based implementation only. The Rust extension is
**not** built by a plain pip/uv install — see Tier 2.

### Tier 2 — Rust backend (local build)

Requires a Rust toolchain (`rustup`) and [maturin](https://maturin.rs)
(included in the dev dependency group):

```bash
git clone https://github.com/Et9797/et-miner.git
cd et-miner
uv venv && source .venv/bin/activate
uv sync                                 # installs package + dev group (incl. maturin)
cd rust_ext && maturin develop --release
```

When the extension is installed, ET-Miner automatically uses it for k>2
support counting. The default build is portable; for a machine-tuned build
(AVX-512 etc.) opt in with:

```bash
RUSTFLAGS="-C target-cpu=native" maturin develop --release
```

### Tier 3 — GPU

```bash
uv pip install "et-miner[gpu] @ git+https://github.com/Et9797/et-miner.git"
```

| Extra | Packages | Purpose |
|-------|----------|---------|
| `gpu` | cupy-cuda12x, nvidia-nccl-cu12 | CUDA acceleration + multi-GPU all-reduce |
| `gcs` | google-cloud-storage | Uploading flushed results to GCS |

The dev tooling (pytest, ruff, maturin, efficient-apriori, matplotlib,
psutil, tqdm, sparse-dot-mkl, mkl) lives in the PEP 735 `dev` dependency
group: `uv sync` installs it by default; with pip use `pip install --group dev .`.

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
from et_miner import apriori, apriori_streaming_multi_gpu

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

Capability probes tell you what the current environment actually supports:

```python
import et_miner
et_miner.HAS_RUST        # Rust extension importable
et_miner.HAS_GPU         # CuPy installed
et_miner.has_cupy()      # CuPy installed AND a CUDA device is usable
et_miner.get_gpu_count() # visible CUDA devices
```

Or from the shell: `et-miner info`.

## CLI

```bash
et-miner mine -i transactions.parquet -o rules.json --min-support 0.01 --min-confidence 0.6
et-miner info          # version, dependency, and backend status
python -m et_miner …   # same entry point
```

## Configuration

File- and environment-based configuration via `et-miner.toml` /
`~/.config/et-miner/config.toml` and `ET_MINER_*` variables
(`from et_miner import Config, load_config`). Operational knobs for the
flush/upload pipeline are environment variables, documented in
`src/et_miner/_env.py` — the important ones:

| Variable | Meaning (default) |
|----------|-------------------|
| `ET_MINER_FLUSH_PARALLEL_THRESHOLD` | rows above which parquet flush partitions in parallel (1e8) |
| `ET_FLUSH_COMPRESSION` / `ET_FLUSH_CHUNK_SIZE` / `ET_FLUSH_THREADS` | parquet flush tuning (zstd / 5e7 / 4) |
| `ET_PARQUET_TMPDIR` | staging dir for flushes (system temp dir) |
| `ET_UPLOAD_GCS` | "1" enables GCS upload of flushed results |
| `ET_MINER_GCS_BUCKET` | destination bucket (gs://et-miner-results) |
| `ET_MINER_GCS_CREDENTIALS` | path to a service-account/ADC JSON |
| `ET_MINER_LOG_DIR` | file-log directory (~/.cache/et-miner/logs) |

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
- CUDA kernel sources maintained as real `.cu` files (`src/et_miner/gpu/kernels/_src/`), compiled on first use via CuPy
- Direct CSR-to-GPU bitvector conversion (bypasses dense matrix construction)
- Fused CUDA kernels: candidate generation + support counting + filtering in a single launch
- GPU-resident mining: the bitvector matrix is uploaded once (3.1 GB of CSR arrays for the 76.9M-protein AlphaFold subset) and stays on the device across all K-levels; per level only prefix-group arrays go up and 12-byte survivor records come down (0.32 GB in total for the 26.8M-itemset run). The optional `gpu_resident=True` variant keeps candidate generation and result collection on the device as well
- Density-adaptive layout: `sparse_from_k="auto"` measures each level's mean support and switches from dense bitvectors to sparse CSR tidsets when tidsets become the smaller representation (mean support < n/32); an int pins the switch to a fixed K-level
- Multi-GPU support by candidate split or row split. Any K>=3 level with 500,000 or more candidates is dispatched to all visible GPUs automatically, so pin `CUDA_VISIBLE_DEVICES` when you want a single-GPU measurement (the 2026-09 campaign used two RTX 3090s for the null model and the per-K exports)

## AlphaFold Application

Applied to the AlphaFold Protein Structure Database, ET-Miner discovered **26.8 million co-occurrence patterns** across the **76.9 million proteins** that carry at least two annotated features, reaching feature combinations of size K=22 in 43.9 minutes on a single RTX 3090. Every value in this section comes from the full re-execution of 2026-09-02 described under [Reproduction](#reproduction).

**Problem.** The AlphaFold Database holds predicted structures for 214.7 million proteins. Which combinations of structural and functional features — Pfam domains, Gene Ontology terms, confidence scores — co-occur across the protein universe? A dense boolean matrix for the 205.6 million proteins that pass the confidence filter would need 206 GB, beyond any GPU's memory.

**Solution.** ET-Miner builds a CSR representation directly from the transactions (5.1 GB in coordinate form for the mined subset), copies its arrays to the GPU once (3.1 GB), expands them on-device into a 9.6 GB bitvector matrix that stays resident across all K-levels, and moves only prefix-group arrays up and 12-byte survivor records down at each level.

### Results (re-execution of 2026-09-02, one RTX 3090)

| Metric | Value |
|--------|-------|
| Proteins | 214,683,829 AlphaFold DB entries; 205,620,298 with mean pLDDT >= 50; 76,890,945 with at least two annotated features (the mined set) |
| Annotations | UniProt TrEMBL release 2026_01 (Pfam and GO cross-references) |
| Feature vocabulary | 1,006 defined items (500 Pfam domains, 500 GO terms, 6 pLDDT bins), 1,002 populated |
| Itemsets discovered | 26,849,505 at a minimum support of 8 proteins (0.00001%) |
| Maximum K | 22 (one itemset shared by 8 proteins; an empirical ceiling, not a proven one) |
| Mining time (deepest threshold) | 43.9 min on one RTX 3090 (24 GB) |
| Support range | 0.1% down to 0.00001% (six thresholds) |
| Null model | 100 permutations preserving per-protein feature counts: no null run reaches K >= 7 |
| Streaming (SON) vs direct | same itemsets (identity verified at 0.1% and 0.01%; same count and K-distribution at 0.001%), 99x slower at 0.001% (6,151 s vs 62 s) |

Six-threshold campaign, all exhaustive on the direct CSR-to-GPU path with `CUDA_VISIBLE_DEVICES=0`:

| Threshold | Min support | Min count | Itemsets | Max K | Time |
|-----------|-------------|-----------|----------|-------|------|
| Base | 0.1% | 76,891 | 5,305 | 9 | 31.1 s |
| Super | 0.01% | 7,690 | 113,405 | 14 | 52.6 s |
| Power | 0.001% | 769 | 475,865 | 14 | 74.7 s |
| Blitz | 0.0001% | 77 | 2,841,280 | 19 | 4.9 min |
| Ultra | 0.00002% | 16 | 14,558,875 | 20 | 33.3 min |
| Opus | 0.00001% | 8 | 26,849,505 | 22 | 43.9 min |

### Reproduction

The original result data was lost, so the whole pipeline was re-executed on 2026-09-02 on a vast.ai box with two RTX 3090s: AlphaFold metadata download, feature extraction from UniProt TrEMBL 2026_01, the six-threshold campaign, the 100-permutation null model and the streaming comparison. Every script, log, per-K itemset table and experiment JSON is under `runs/20260902T0000Z/` (start at its `README.md`). `RESULTS.md` lists each fresh value with its artifact, and `COMPARISON_REPORT.md` checks the 3,521 claims of the original preprint, its reviews and its logs against the fresh values: 1,307 confirmed, 119 hallucinated, 555 expected hardware deviations, 1,540 inconclusive. The preprint revised to report only this run is `paper/et_miner_proteome.tex` (branch `v2`; `CHANGELOG_V1_V2.md` maps every value from V1 to V2, and `runs/20260902T0000Z/phase4/citation_audit/` records the citation audit). The superseded V1 preprint is archived at Zenodo, DOI [10.5281/zenodo.18674353](https://doi.org/10.5281/zenodo.18674353).

## Benchmarks

> Historical measurements from the author's machines (environments noted per
> table); the harnesses used are not part of this repository, so treat the
> numbers as indicative rather than reproducible.

### Billion-Scale Streaming

| Metric | Value |
|--------|-------|
| Transactions | 1,000,000,000 |
| Time | 25.9 minutes |
| Throughput | 643,139 tx/sec |
| Peak memory | 14.76 GB |
| Itemsets found | 326 |
| Hardware | Intel Core Ultra 9 275HX (24 cores), 134 GB RAM |

### vs. efficient-apriori (819K transactions)

> System: AMD Ryzen 5 4600G (12 cores), 30 GB RAM, CPython 3.14 free-threading build, Polars 1.37, MKL sparse enabled

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
uv sync                                    # package + dev group
cd rust_ext && maturin develop --release && cd ..   # optional: Tier 2

# Tests (Python >= 3.10; CI-tested on 3.10-3.12)
pytest tests/
```

GPU-dependent tests carry the `gpu` marker and skip automatically when no
CUDA device is present; long-running cases carry `slow`
(deselect with `-m "not slow"`).

### Test dataset

The correctness smoke tests use the UCI Online Retail II dataset, which is
generated locally (parquet files are gitignored):

```bash
python datasets/prepare_online_retail.py
```

This downloads the source zip from the UCI archive on first run and writes
`datasets/online_retail_ii/transactions.parquet`.

## License

**PolyForm Noncommercial License 1.0.0** — Copyright 2026 Etjen Ahmic. See [LICENSE](LICENSE).

Free for non-commercial use (personal, academic, research, and other
noncommercial purposes). **Commercial use requires a separate license — please
contact the author** via this repository.
