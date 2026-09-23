# ET-Miner tutorials

These tutorials teach ET-Miner in three tiers that build on each other: the Polars implementation, the compiled
Rust core, and the CUDA path on one or more GPUs. Every concept has a worked example that calls the library and
shows real output. This page is the map: reading order, how to run everything, what each file covers, the
conventions for adding to it, and a glossary.

## 1. Reading order and prerequisites

Read the tiers in order: tier 1, then tier 2, then tier 3. Within a tier, read the notebooks in numeric order. [concepts.md](concepts.md) is the
reference that every notebook links to.

| Tier | You need | Hardware |
|---|---|---|
| 1, Polars | Python and basic Polars; the project environment (section 3) | any CPU |
| 2, Rust + PyO3 | tier 1; a Rust toolchain to build the extension (no Rust knowledge needed to follow the notebooks) | any CPU |
| 3, GPU | tiers 1 and 2; for the GPU cells, an NVIDIA GPU with a CUDA 12 driver and the `gpu` extra | one GPU; two or more for the multi-GPU cells |

Tier 3 notebooks run without a GPU too: they check for a device first, run the CPU parts (decision rules, planning,
simulations, SON), and print `skipped: no CUDA device` in each GPU cell. Each ends with a checklist for a GPU run.

## 2. What is in this folder, and the other documentation

Runtimes were measured with `run_notebooks.sh` on x86_64, 4 logical CPUs, no GPU, on 2026-09-23.
Tier 3 times are for the CPU parts only; on a GPU machine the GPU cells add to them.

| File | Covers | Runtime | Hardware |
|---|---|---|---|
| [README.md](README.md) | this map | | |
| [concepts.md](concepts.md) | definitions, support semantics, routes, parameter reference | | |
| [run_notebooks.sh](run_notebooks.sh) | executes every notebook in order, stops at the first error | sum of the rows below | |
| [data/make_samples.py](data/make_samples.py) | writes the bundled samples below | | CPU |
| `data/toy_8x6.parquet` | the 8-row, 6-item table of tier 1 | | |
| `data/smoke.parquet`, `data/smoke.json` | the shared sample: preset `smoke` of `src/et_miner/synthetic.py`, 60,000 rows, seed 42, and its parameters | | |
| [tier1-polars/01-the-problem-and-the-data.ipynb](tier1-polars/01-the-problem-and-the-data.ipynb) | transactions, itemsets, support, input format, the boolean matrix | 2 s | CPU |
| [tier1-polars/02-apriori-step-by-step.ipynb](tier1-polars/02-apriori-step-by-step.ipynb) | the level loop by hand: join, prune, count, frontier; reading the result | 3 s | CPU |
| [tier1-polars/03-mining-a-realistic-sample.ipynb](tier1-polars/03-mining-a-realistic-sample.ipynb) | parameters and their effect; cost per level; two data sizes; limits of Polars | 22 s | CPU |
| [tier2-rust-pyo3/01-build-and-install.md](tier2-rust-pyo3/01-build-and-install.md) | toolchain, build command, checks, common errors | build: 32 s | CPU |
| [tier2-rust-pyo3/02-inside-the-rust-core.ipynb](tier2-rust-pyo3/02-inside-the-rust-core.ipynb) | the PyO3 boundary, bit vectors, counting, candidate generation, threads | 3 s | CPU |
| [tier2-rust-pyo3/03-same-results-and-speed.ipynb](tier2-rust-pyo3/03-same-results-and-speed.ipynb) | tier 1 = tier 2 = efficient-apriori; speed of each route at three sizes; limits | 35 s | CPU |
| [tier3-gpu/01-gpu-setup-and-first-run.ipynb](tier3-gpu/01-gpu-setup-and-first-run.ipynb) | requirements, device detection, kernel self-check, first GPU run | 3 s | CPU; GPU cells need 1 GPU |
| [tier3-gpu/02-gpu-resident-mining.ipynb](tier3-gpu/02-gpu-resident-mining.ipynb) | what stays on the GPU, footprint estimate, device memory | 2 s | CPU; GPU cells need 1 GPU |
| [tier3-gpu/03-shared-memory-and-tiled-routes.ipynb](tier3-gpu/03-shared-memory-and-tiled-routes.ipynb) | prefix groups, the tiled kernel, kernel variants, chunk planning | 2 s | CPU; GPU cells need 1 GPU, one cell 2 |
| [tier3-gpu/04-sparse-dense-crossover.ipynb](tier3-gpu/04-sparse-dense-crossover.ipynb) | bit vectors versus tidsets, the crossover rule, observing the switch | 3 s | CPU; GPU cells need 1 GPU |
| [tier3-gpu/05-son-streaming.ipynb](tier3-gpu/05-son-streaming.ipynb) | SON: both passes, exactness proof, chunk size, memory | 7 s | CPU; one GPU cell |
| [tier3-gpu/06-multi-gpu.ipynb](tier3-gpu/06-multi-gpu.ipynb) | row split, partial counts, global pruning, running on N GPUs, fallbacks | 3 s | CPU; multi-GPU cells need 2+ GPUs |

Other documentation in the repository:

| File | Covers |
|---|---|
| [../README.md](../README.md) | project overview, installation per tier, quick start, CLI, API summary |
| [../CHANGELOG.md](../CHANGELOG.md) | changes per version, including changes to mined output |
| [../bench/README.md](../bench/README.md) | running the GPU benchmark and validation scripts on a GPU machine |
| `src/et_miner/_env.py` (module docstring) | every `ET_*` environment variable |

## 3. Environment setup

All commands run from the repository root. Each was run to check it on the machine named in section 2;
`uv sync --inexact --extra gpu` was checked with `--dry-run` only, because that machine has no GPU.

```bash
uv venv
uv sync                                                  # package + dev group (includes efficient-apriori)
uv run maturin develop --release -m rust_ext/Cargo.toml  # Rust extension, needed from tier 2 on
```

For the GPU cells of tier 3, on a machine with an NVIDIA GPU:

```bash
uv sync --inexact --extra gpu                            # CuPy with CUDA headers, NCCL
```

A plain `uv sync` after building removes the Rust extension; `--inexact` keeps it.
See [tier 2, notebook 1](tier2-rust-pyo3/01-build-and-install.md#4-common-errors-and-fixes).

Jupyter is not a project dependency, and the notebooks need nothing beyond the project environment.
`uv run --with` adds Jupyter for one command without changing `pyproject.toml`:

```bash
uv run --with jupyterlab jupyter lab docs/                # read and run interactively
docs/run_notebooks.sh                                     # execute all notebooks in place, in order
docs/run_notebooks.sh tier1-polars tier2-rust-pyo3        # only some tiers
```

`run_notebooks.sh` uses `uv run --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute --inplace`
for each notebook and stops at the first error. It regenerates the samples with `uv run python docs/data/make_samples.py`
if they are missing.

## 4. Conventions for adding documentation

**Place and name.** Tutorials live in `docs/tier<N>-<topic>/` as `NN-short-title.ipynb` (or `.md` when nothing
needs executing), numbered in reading order. Add every new file to the table in section 2. The repository's `.gitignore` lists
`docs/`, so stage new files with `git add -f`.

**Notebook structure.** In this order:

1. title (`# Tier N.M: ...`), one paragraph that states the main point, **What you will learn**, **Prerequisites** with links;
2. `## 0. Setup` with the imports; tier 3 notebooks start with the device check that sets `SKIP_GPU` and `SKIP_MULTI`;
3. numbered sections (`## 1. ...`), each as: a short markdown cell saying what the next code cell does and what to
   look for, the code cell, then a one or two sentence observation;
4. `## Summary and next step`, with a link to the next notebook; tier 3 adds a checklist for GPU runs.

**Execution.** A notebook runs top to bottom from a fresh kernel through `run_notebooks.sh`, within 2 minutes on a
CPU for tiers 1 and 2 and 5 minutes on one GPU for tier 3. Seeds are fixed. Notebooks share nothing except files in
`docs/data/`. Commit notebooks with the outputs of a real run; never edit an output by hand.

**Numbers.** Every runtime, count or size in the text comes from a cell in the same notebook. Cells that time
something print the hardware, data size and date with the result. GPU cells without a device print a skip line
instead of a result.

**Data.** Put small samples in `docs/data/` and the code that makes them in `docs/data/make_samples.py`.
Do not download external datasets in a notebook.

**Source references.** Write them as `path/file.py:function`, relative to the repository root. Quote at most 15
lines; the tutorials print quotes from the files at run time (`show()`), so line numbers stay current.

**Style.** Lead with the main point. One idea per sentence. State the observation, then what it means. Simple,
precise words; no marketing adjectives, no em dashes, no emoji, no exclamation marks. Numbered headings. American
spelling. Diagrams only where they help: Mermaid in `.md` files, figures generated in a cell in notebooks.

## 5. Glossary

| Term | Meaning | Explained in |
|---|---|---|
| anchor items | items that restrict the reported itemsets to those containing one of them (row-split only) | [concepts 4](concepts.md#4-options-that-change-the-result) |
| Apriori | level-wise mining: generate candidates of length K from frequent (K-1)-itemsets, count, keep the frequent ones | [T1.2](tier1-polars/02-apriori-step-by-step.ipynb) |
| bit vector | one bit per row for one item or itemset, 64 rows per `uint64` word | [T2.2 §3](tier2-rust-pyo3/02-inside-the-rust-core.ipynb), [concepts 1](concepts.md#1-transactions-items-and-the-internal-representation) |
| boolean matrix | tier 1's representation: one Polars boolean column per frequent item | [T1.1 §4](tier1-polars/01-the-problem-and-the-data.ipynb) |
| candidate | an itemset proposed for counting at the current level | [T1.2 §2](tier1-polars/02-apriori-step-by-step.ipynb) |
| chunk (SON) | a slice of rows mined on its own in SON pass 1 | [T3.5](tier3-gpu/05-son-streaming.ipynb) |
| chunk (GPU) | a range of candidates counted in one kernel launch sequence, sized to free GPU memory | [T3.3 §5](tier3-gpu/03-shared-memory-and-tiled-routes.ipynb) |
| count | number of rows that contain an itemset | [concepts 3](concepts.md#3-support-and-the-minimum-count) |
| crossover | the mean count `N / 32` at which tidsets become smaller than bit vectors | [T3.4](tier3-gpu/04-sparse-dense-crossover.ipynb) |
| CSR | compressed sparse row: `indptr` (row starts) and `indices` (item columns) | [T2.2 §1](tier2-rust-pyo3/02-inside-the-rust-core.ipynb) |
| CSC | the column-major twin of CSR, built inside the Rust core | [T2.2 §3](tier2-rust-pyo3/02-inside-the-rust-core.ipynb) |
| downward closure | support can only fall as an itemset grows, so a candidate with an infrequent subset cannot be frequent | [T1.2 §2](tier1-polars/02-apriori-step-by-step.ipynb) |
| efficient-apriori | an independent Python Apriori used as the correctness reference | [T2.3 §1](tier2-rust-pyo3/03-same-results-and-speed.ipynb) |
| free-set (generator) | itemset with no proper subset of the same support; the output of `prune_equal_support=True` | [T1.3 §4](tier1-polars/03-mining-a-realistic-sample.ipynb) |
| frequent | `count >= min_count` | [concepts 3](concepts.md#3-support-and-the-minimum-count) |
| frontier | the frequent itemsets of the current level | [T1.2 §3](tier1-polars/02-apriori-step-by-step.ipynb) |
| fused kernel | a kernel that generates candidates from their index, counts and filters in one launch | [T3.4 §4](tier3-gpu/04-sparse-dense-crossover.ipynb) |
| GPU-resident | mode in which every level stays in GPU memory until the end | [T3.2](tier3-gpu/02-gpu-resident-mining.ipynb) |
| itemset (pattern) | a set of items | [T1.1 §1](tier1-polars/01-the-problem-and-the-data.ipynb) |
| K, level | itemset length, and the step that mines itemsets of that length | [concepts 2](concepts.md#2-itemsets-k-and-the-frontier) |
| kernel variant | `legacy` or `shared` dense counting kernels, set by `ET_MINER_KERNEL_VARIANT` | [T3.3 §4](tier3-gpu/03-shared-memory-and-tiled-routes.ipynb) |
| local support factor | SON's pass-1 threshold as a fraction of `min_support` (default 0.9) | [T3.5 §3](tier3-gpu/05-son-streaming.ipynb) |
| min_count | `ceil(min_support * N)`, the integer threshold | [concepts 3](concepts.md#3-support-and-the-minimum-count) |
| min_support | the support threshold, a fraction in `[0, 1]` | [concepts 3](concepts.md#3-support-and-the-minimum-count) |
| NCCL | NVIDIA's collective library, used to sum partial counts onto GPU 0 | [T3.6 §4](tier3-gpu/06-multi-gpu.ipynb) |
| NVRTC | CUDA's runtime compiler; kernels are compiled with it at first use | [T3.1 §1](tier3-gpu/01-gpu-setup-and-first-run.ipynb) |
| partial count | one GPU's count of a candidate over its own rows | [T3.6 §2](tier3-gpu/06-multi-gpu.ipynb) |
| Pascal rule | inferring a candidate's count from its subsets (`use_generator_pruning`) | [T1.3 §5](tier1-polars/03-mining-a-realistic-sample.ipynb) |
| popcount | counting the set bits of a word | [T2.2 §4](tier2-rust-pyo3/02-inside-the-rust-core.ipynb) |
| prefix group | candidates that share their first K-1 items | [T3.3 §1](tier3-gpu/03-shared-memory-and-tiled-routes.ipynb) |
| prefix join | building a K-candidate from two (K-1)-itemsets with the same first K-2 items | [T1.2 §2](tier1-polars/02-apriori-step-by-step.ipynb) |
| PyO3 | the Rust library that exposes the Rust core to Python | [T2.2 §2](tier2-rust-pyo3/02-inside-the-rust-core.ipynb) |
| rayon | the Rust thread pool that spreads candidates over cores | [T2.2 §7](tier2-rust-pyo3/02-inside-the-rust-core.ipynb) |
| route | the code path that serves one call | [concepts 6](concepts.md#6-routes-and-tiers) |
| row, transaction | one input record: a set of items | [T1.1 §1](tier1-polars/01-the-problem-and-the-data.ipynb) |
| row split | multi-GPU mode in which each GPU holds a contiguous range of rows | [T3.6](tier3-gpu/06-multi-gpu.ipynb) |
| shared memory | fast on-chip GPU memory shared by one thread block | [T3.3 §2](tier3-gpu/03-shared-memory-and-tiled-routes.ipynb) |
| SON | Savasere, Omiecinski and Navathe's two-pass partition algorithm | [T3.5](tier3-gpu/05-son-streaming.ipynb) |
| sparse (CPU) | `sparse=` parameter: Polars versus SciPy + Rust counting | [concepts 6](concepts.md#6-routes-and-tiers), [T2.3 §4](tier2-rust-pyo3/03-same-results-and-speed.ipynb) |
| sparse_from_k (GPU) | level at which the GPU switches from bit vectors to tidsets | [T3.4 §3](tier3-gpu/04-sparse-dense-crossover.ipynb) |
| subset prune | dropping a candidate whose (K-1)-subset is not frequent | [T1.2 §2](tier1-polars/02-apriori-step-by-step.ipynb) |
| support | `count / N` | [concepts 3](concepts.md#3-support-and-the-minimum-count) |
| tidset | the sorted row ids (transaction ids) of an itemset | [T3.4 §1](tier3-gpu/04-sparse-dense-crossover.ipynb) |
| tier | one of the three implementations: Polars, Rust, GPU | [concepts 6](concepts.md#6-routes-and-tiers) |
| tile, tile pair | 32 suffixes of a prefix group, and a pair of tiles served by one GPU block | [T3.3 §2](tier3-gpu/03-shared-memory-and-tiled-routes.ipynb) |
