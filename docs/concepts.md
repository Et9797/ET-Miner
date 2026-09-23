# Concepts and parameter reference

This page defines the terms the tutorials use and lists the public parameters of the mining entry points.
Every definition points to the code that implements it, as `path/file.py:function`, at the version in this
repository. Paths are relative to the repository root.

## 1. Transactions, items and the internal representation

- A **transaction** (or **row**) is one set of **items**, such as the products in one basket.
- The input is a Polars `DataFrame` or `LazyFrame` with one list column, `items` by default (`item_col=` selects another).
  Items are integers or strings. Every row counts toward the row total `N`, including rows with an empty list.
- The library never counts on the lists directly. Each route first builds its own representation:

| Representation | Built by | Used by |
|---|---|---|
| Boolean matrix: one Polars boolean column per frequent item, one row per transaction | `src/et_miner/core/matrix.py:build_boolean_matrix` | tier 1 (Polars counting) |
| SciPy CSR matrix built from that boolean matrix | `src/et_miner/core/matrix.py:_polars_to_sparse_csr` | tier 2 per-level route (`sparse=True`) |
| CSR arrays (`indptr`, `indices`, int64) supplied by the caller | the caller | tier 2 whole-loop route (`apriori_from_csr`) |
| CSR matrix of frequent items built straight from the lists | `src/et_miner/core/matrix.py:_build_csr_from_transactions` | GPU routes, before upload |
| **Bit vectors**: one bit per row, 64 rows per `uint64` word, one vector per item | `rust_ext/src/core/bitvec.rs:build_column_bitvecs_u64_raw`, `src/et_miner/gpu/bitvec.py:_build_gpu_bitvec_matrix` | Rust counting, GPU dense counting |
| **CSR tidsets**: sorted int32 row ids per frequent itemset | `src/et_miner/gpu/sparse_csr.py:convert_shards_to_csr` | GPU sparse levels (section 6) |

Items that are not frequent on their own are dropped before any of these is built, except in `apriori_from_csr`,
which keeps a bit vector for every column it is given.

Worked example: [tier 1, notebook 1](tier1-polars/01-the-problem-and-the-data.ipynb) and [tier 2, notebook 2](tier2-rust-pyo3/02-inside-the-rust-core.ipynb).

## 2. Itemsets, K and the frontier

- An **itemset** (also called a pattern) is a set of items. Its length is **K**.
- **Level K** is the step of the algorithm that finds the frequent itemsets of length K.
- The **frontier** is the list of frequent itemsets of the current level. The next level is generated from it.
- **Candidate generation**: two frequent (K-1)-itemsets that share their first K-2 items are joined into a K-candidate
  (the **prefix join**), and a candidate with an infrequent (K-1)-subset is dropped before counting (the **subset
  prune**, safe because support can only fall as an itemset grows). CPU: `src/et_miner/core/candidates.py:_generate_candidates`;
  Rust: `rust_ext/src/core/candidates.rs:generate_candidates_kplus1`.
- A **prefix group** is the set of candidates of one level that share their first K-1 items. The GPU kernels
  enumerate candidates group by group from their index (`src/et_miner/gpu/kernels/k3plus.py:build_k3plus_groups`).
- The GPU group routes prune at group granularity, so their per-level candidate count, as reported through
  `level_callback`, can be larger than the CPU route's for the same level. The emitted itemsets are the same.

Worked example: [tier 1, notebook 2](tier1-polars/02-apriori-step-by-step.ipynb).

## 3. Support and the minimum count

- The **count** of an itemset is the number of rows that contain all its items.
- Its **support** is `count / N`, where `N` is the number of rows in the input.
- `min_support` is a fraction in `[0, 1]` (`src/et_miner/core/apriori.py:_validate_parameters` rejects anything else).
- The library turns it into an integer **minimum count** once, for all levels:
  `min_count = ceil(min_support * N)`, computed exactly by `src/et_miner/core/result.py:_min_count`.
  The float is read as the shortest decimal that prints as it, so `0.07` means exactly 7/100.
  The same rule is implemented in Rust (`rust_ext/src/core/apriori.rs:exact_min_count`) and in
  `src/et_miner/synthetic.py:SynthSpec.min_count`.
- An itemset is **frequent** when `count >= min_count`. The threshold does not depend on K.
- The result reports `support` as a float. `support * N`, rounded, recovers the integer count.

Pitfalls:

- **Small inputs make the threshold coarse.** On 8 rows every `min_support` up to 0.125 means a count of 1, so every
  item and every subset of every row is frequent. [Tier 1, notebook 1](tier1-polars/01-the-problem-and-the-data.ipynb), section 3, prints the steps.
- **`min_support=0` means a minimum count of 0.** Candidates that occur in no row then pass the test and are reported
  with support 0. Use a positive threshold.
- **Comparing with another miner** needs care at the boundary: pass it a threshold just below `min_count / N`.
  The tutorials call efficient-apriori with `min_support = (min_count - 0.5) / N` and an explicit `max_length`
  ([tier 2, notebook 3](tier2-rust-pyo3/03-same-results-and-speed.ipynb), section 1).

## 4. Options that change the result

| Option | Effect on the result | Where it works |
|---|---|---|
| `max_length=K` | stops after level K; supports of the kept itemsets do not change | all routes |
| `prune_equal_support=True` | returns only **free-sets** (generators): itemsets with no proper subset of the same support. Every omitted frequent itemset has the support of one of its subsets. This is not closed-itemset mining; there is no closed-itemset option. | CPU; on the GPU only the row-split miner implements it, and `apriori` routes there automatically. Refused with `streaming=True`. |
| `anchor_items={...}` | reports only itemsets that contain at least one anchor item | GPU row-split miner only (`use_gpu=True`, no streaming) |

Options that change only the work, never the result:

| Option | Effect | Where it works |
|---|---|---|
| `use_generator_pruning=True` | infers the count of a candidate with a non-free subset instead of counting it (Pascal rule) | CPU route only; refused with `streaming=True` |
| `prune_apriori` (default `True`) | the subset prune on the row-split miner; `False` skips it there | `False` is refused on every other route |
| `sparse`, `n_jobs`, `batch_size`, `enable_length_filter` | how the CPU route counts | CPU routes |
| `sparse_from_k` | when the GPU switches from bit vectors to tidsets | GPU routes |

`src/et_miner/core/apriori.py:_validate_route_support` rejects every combination a route cannot honor instead of
ignoring a parameter. Worked examples: [tier 1, notebook 3](tier1-polars/03-mining-a-realistic-sample.ipynb), sections 3 to 5.

## 5. The result frame

`apriori` returns a Polars `DataFrame` with two columns:

- `itemset`: a list of items, always in ascending item order, on every route;
- `support`: `count / N` as a float.

The order of the rows is not part of the contract. With `profile=True` the call returns `(DataFrame, ProfilingSession)`.
`apriori_from_csr` returns `(itemsets, counts)` instead: lists of column indices and integer counts.

## 6. Routes and tiers

A **route** is the code path that serves one call. `src/et_miner/core/apriori.py:apriori` picks it from the
arguments, in this order: `bitvecs=` first, then `streaming=True`, then `use_gpu=True`, else the CPU route.

| Route | How to select it | Loop | Counting | Tier |
|---|---|---|---|---|
| CPU, Polars | `apriori(df, sparse=False)` | Python | Polars boolean columns | 1 |
| CPU, SciPy + Rust | `apriori(df, sparse=True)`, or `sparse=None` when the automatic rule picks it | Python | SciPy for K=2, Rust `count_itemsets_simd` for K>=3 (NumPy without the extension) | 2 |
| Rust whole loop | `apriori_from_csr(indptr, indices, n_rows, n_cols, min_support, max_length)` | Rust | Rust bit vectors, rayon threads | 2 |
| GPU, single device | `apriori(df, use_gpu=True)` | Python, per level on the host | CUDA kernels on bit vectors; optional tidsets | 3 |
| GPU-resident | `apriori(df, use_gpu=True, gpu_resident=True)` | levels stay on the GPU | CUDA kernels | 3 |
| GPU row-split | `use_gpu=True` with `n_gpus>1`, `prune_equal_support` or `anchor_items` | Python | CUDA kernels per GPU, counts reduced on GPU 0 | 3 |
| SON streaming | `apriori(df, streaming=True, chunk_size=...)` | two passes over chunks | CPU, or GPU with `use_gpu`/`gpu_resident` | 1 to 3 |
| SON, multi-GPU | `apriori(df, streaming=True, n_gpus>1)` | two passes, chunks in waves over GPUs | GPU | 3 |

**Two meanings of "sparse".** `sparse=` (CPU) chooses between Polars and SciPy counting. With `sparse=None`,
`src/et_miner/core/sparse.py:_choose_counting_strategy` picks SciPy when there are more than 100,000 candidate pairs,
or more than 500 items at under 10% density, or when the dense matrix would exceed 1 GB.
`sparse_from_k=` (GPU) chooses the level at which the GPU switches from bit vectors to CSR tidsets:
`None` never, an int K from level K (never before 3), `"auto"` at the first level whose previous level has a mean
count below `N / 32` (`src/et_miner/gpu/density.py:should_transition_to_sparse`). The two are unrelated.

**Kernel variants.** On the GPU, `ET_MINER_KERNEL_VARIANT` selects the dense counting kernels: `legacy`, `shared`
(the tiled kernel that stages words in shared memory) or `auto`, which means `shared`
(`src/et_miner/gpu/dispatch.py:resolved_kernel_variant`). The GPU-resident route does not read it.

**Forcing a route.** Use the arguments in the table. `use_gpu=True` never falls back to the CPU: without CuPy it
raises. `n_gpus` is capped at the number of visible devices.

Worked examples: [tier 2, notebook 3](tier2-rust-pyo3/03-same-results-and-speed.ipynb) and [tier 3, notebooks 1 to 6](tier3-gpu/).

## 7. Parameter reference

### 7.1 `et_miner.apriori`

Defined in `src/et_miner/core/apriori.py:apriori`. "CPU" is the CPU route (tiers 1 and 2), "GPU" the single-GPU
routes, "row-split" the GPU row-split miner, "SON" the streaming routes.

| Parameter | Default | Meaning | Honored by |
|---|---|---|---|
| `transactions` | `None` | `DataFrame` or `LazyFrame` with a list column; required unless `bitvecs` is given | all |
| `min_support` | `0.5` | minimum support fraction in `[0, 1]`; see section 3 | all |
| `max_length` | `None` | largest itemset length; `None` means no limit | all |
| `item_col` | `"items"` | name of the list column | all |
| `use_gpu` | `False` | mine on the GPU | GPU, row-split, SON (per-chunk counting) |
| `batch_size` | `10_000` | candidates per Polars counting batch; `None` means one batch | CPU (Polars counting), SON; not used by the GPU miners |
| `profile` | `False` | also return a `ProfilingSession` with per-phase time and memory | CPU, single GPU, single-GPU SON; refused on row-split and multi-GPU SON |
| `show_progress` | `False` | progress bars (needs `tqdm`) | CPU, SON |
| `warn_complexity` | `True` | warn when the level-2 candidate count is large | CPU |
| `prune_equal_support` | `False` | return free-sets instead of the full lattice (section 4) | CPU, row-split; refused with streaming |
| `use_generator_pruning` | `False` | infer some counts instead of counting them; same result | CPU; refused with streaming |
| `prune_apriori` | `True` | subset prune on the row-split miner; `False` only allowed there | row-split |
| `sparse` | `None` | CPU counting: `False` Polars, `True` SciPy + Rust, `None` automatic | CPU, SON |
| `n_jobs` | `1` | worker threads for SciPy/Rust counting; `-1` means all cores | CPU with SciPy/Rust counting, SON |
| `enable_length_filter` | `True` | skip rows shorter than K when counting level K | CPU |
| `streaming` | `False` | use SON over row chunks | SON |
| `chunk_size` | `10_000_000` | rows per SON chunk | SON |
| `n_gpus` | `1` | GPUs to use; >1 selects the row-split miner (or multi-GPU SON with `streaming=True`) | row-split, multi-GPU SON, single-GPU fan-out with `bitvecs=` |
| `memory_budget_gb` | `None` | derive `chunk_size` from a memory budget; only with `streaming=True` | SON |
| `progress_callback` | `None` | `(phase, chunk_idx, n_chunks, metrics)` per SON chunk and pass | SON |
| `level_callback` | `None` | `(k, n_candidates, n_frequent, duration_ms)` per level; `n_candidates` is route-dependent | CPU, GPU, row-split |
| `gpu_resident` | `False` | keep all levels in GPU memory | single GPU, single-GPU SON; refused elsewhere |
| `bitvecs` | `None` | pre-built GPU bit vectors `(cupy_array, col_to_item, n_rows)` instead of `transactions` | GPU |
| `output_dir` | `None` | write each level to Parquet as it finishes | row-split only |
| `resume_from_k` | `None` | resume from a level written by `output_dir` | row-split only |
| `max_ram_gb` | `800.0` | host-memory guard between levels | single-GPU bit-vector miner |
| `max_vram_gb` | `70.0` | GPU-memory guard between levels | single-GPU bit-vector miner |
| `sparse_from_k` | `None` | GPU switch to CSR tidsets: `None`, an int K (>= 3), or `"auto"` (section 6) | GPU, row-split |
| `anchor_items` | `None` | report only itemsets containing an anchor item | row-split only |

### 7.2 Other entry points

| Function | Signature (defaults) | Notes |
|---|---|---|
| `et_miner.apriori_from_csr` | `(csr_indptr, csr_indices, n_rows, n_cols, min_support, max_length)` | Rust extension only. int64 arrays, strictly increasing column indices within each row, `max_length=0` means no limit. Returns column indices and counts. |
| `et_miner.apriori_streaming` | `(transactions, min_support=0.5, max_length=None, item_col="items", chunk_size=40_000_000, memory_budget_gb=None, local_support_factor=0.9, use_gpu=False, gpu_resident=False, batch_size=10_000, profile=False, show_progress=True, sparse=None, n_jobs=1, progress_callback=None)` | SON on one device. `local_support_factor` scales the pass-1 threshold; any value `<= 1` keeps the result exact. |
| `et_miner.apriori_streaming_multi_gpu` | `(transactions, min_support=0.5, max_length=None, item_col="items", n_gpus=8, chunk_size=10_000_000, memory_budget_gb=None, local_support_factor=0.9, batch_size=10_000, show_progress=True, sparse=None, n_jobs=1, progress_callback=None)` | SON with chunks in waves over GPUs; raises `MiningError` without a CUDA device. |
| `et_miner.generate_rules` | `(frequent_itemsets, min_confidence=0.5)` | association rules from an `apriori` result, as `Rule(lhs, rhs, support, confidence, lift)` |

### 7.3 Environment variables used in the tutorials

All `ET_*` variables are documented in `src/et_miner/_env.py`. The GPU tutorials use these:

| Variable | Values | Effect |
|---|---|---|
| `ET_MINER_KERNEL_VARIANT` | `auto` (default) / `legacy` / `shared` | dense counting kernel family; `auto` means `shared` |
| `ET_MINER_TILED_MIN_GROUP_PAIRS` | int, default 64 | on the row-split route, prefix groups with fewer candidate pairs use the legacy kernel |
| `ET_MINER_ROW_BALANCE` | `rows` (default) / `nnz` | how the row-split miner cuts rows across GPUs |
| `ET_MINER_DISABLE_NCCL` | `1` | use the staged device-to-device reduce instead of NCCL |
| `ET_MINER_MAX_CHUNK_CANDS` | int | cap on candidates per dense chunk on the row-split route |
| `LOGURU_LEVEL` | `DEBUG`, `INFO`, `WARNING`, ... | level of the default log handler (a loguru setting, not an ET-Miner one); the notebooks set `WARNING` |

Log lines that show a route decision: `DENSITY TRANSITION at K=...` (GPU switch to tidsets, INFO),
`[k>2 RUST SIMD] ...` (a Rust counting call, DEBUG), `[auto-detect] ... -> sparse` (the CPU automatic choice, DEBUG),
`Generator pruning: ...` (Pascal inference, DEBUG).
