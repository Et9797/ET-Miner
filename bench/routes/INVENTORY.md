# Route inventory (round 2), `main` at `895f3ab`

Every row was checked against the code on `main` at `895f3ab`. Paths are
relative to `src/et_miner/` unless they start with `tests/`, `bench/` or
`rust_ext/`. "Exercised by" names the regime of `PROTOCOL.md` that runs the
variant at its defaults (from the kernels each regime ran in
`bench/results/2026-10-18-moonshots-phase3/final.jsonl`, the last campaign on
this code), or the forcing config the harness adds. "Pinned by" names the
test and the tier leg (`tests/test_tier_equivalence.py`) that pin it.

## 1. Entry points

| id | Route | Entry and selection | Counting |
|---|---|---|---|
| R-CPU | array miner | `core/cpu_miner.py::mine_cpu`; `apriori()` with neither `use_gpu`, `bitvecs=` nor `streaming` (`core/apriori.py:819-833`) | one CSR of the frequent items, the arms of §3 |
| R-GPU | row-split miner | `gpu/row_split.py::_apriori_row_split_multi_gpu` (`:225`); `bitvecs=` first, whatever `use_gpu`/`streaming` say (`core/apriori.py:600-666`), then `use_gpu` without `streaming` (`:726-816`). Also entered by SON's GPU pass 1 through the internal `bitvecs_list=` (`streaming/son.py:534-550`) | per level, the arms of §2 |
| R-SON | SON, one device | `streaming/son.py::apriori_streaming` (`:134`); `streaming=True` with `n_gpus<=1` (`core/apriori.py:703-719`). Two passes only when `n_total > effective chunk size` (`son.py:263`); otherwise `apriori()` again with the same arguments minus streaming (`:272-282`) | **CPU passes** (`use_gpu=False`, `son.py:306`, `_son_cpu` `:752`): pass 1 `build_transaction_csr` + `_mine_levels` per chunk (`_local_levels` `:695`), union `_CandidateUnion` (int32 rows, summed local counts, `sum_rows`), `_bound` (`:668`, applied `:820`), pass 2 `_map_rows` + `cpu_miner.count_itemsets` (`_count_chunk` `:736`). **GPU passes** (`son.py:323-512`): per chunk Polars `build_boolean_matrix` (`:358`) → `_polars_to_sparse_csr` → `_build_gpu_bitvec_matrix` → R-GPU on column ids (`_mine_chunk_gpu` `:515`), so no device layer, no row-wise K=2/K=3 (no rows), no lex order; Python `set` union (`:331`); no `_bound`; pass 2 Polars boolean matrix over the candidate items (`_build_matrix_for_items` `:869`) + `count_itemsets_cuda` (`_count_candidates_gpu` `:557`) |
| R-SON-MG | SON, several GPUs | `streaming/multi_gpu.py::apriori_streaming_multi_gpu` (`:116`); `streaming=True, n_gpus>1` (`core/apriori.py:684`), **whatever `use_gpu` says** (not forwarded, `:687-700`). Needs CuPy and a GPU before anything else (`multi_gpu.py:197-205`); one chunk → `apriori(use_gpu=True, n_gpus=min(n_gpus, devices))` (`:246-265`) | waves of one chunk per device (`:344-361`, `:469-486`), R-SON's GPU helpers per chunk (`:315-326`, `:444-454`), `set` union under a lock (`:369-371`), no `_bound`, no profile |
| R-RUST | all-Rust miner | `et_miner.apriori_from_csr` (`__init__.py:71-86`, `rust_ext/src/lib.rs:237-259`, `rust_ext/src/core/apriori.rs:118`) | no route calls it; the tier chain's "Tier 2" leg (`tests/test_tier_equivalence.py:184`) |
| API | public counting | `count_support_batched` (`core/matrix.py:277`; not in `core/sparse.py`): GPU → `gpu/bitvec.py::count_support_gpu_bitvec` (`_build_gpu_bitvec_matrix` + `count_itemsets_cuda`); `sparse` (or auto via `_choose_counting_strategy`) → `core/sparse.py::count_support_sparse` (`:452`; pairs `_sparse_matmul`, MKL float64 or scipy, `:212-245`; the rest `cpu_miner.count_itemsets`, `:248-300`); else Polars `count_support_vectorized` (`core/matrix.py:186`) in `batch_size` batches (`:410-445`). `build_boolean_matrix` (`core/matrix.py:89`) | no `apriori()` route calls `count_support_*`; R-SON's and R-SON-MG's GPU passes call `build_boolean_matrix` |

`mine_two_phase` (`gpu/row_split.py:1516-1670`) is a wrapper: two
`apriori(use_gpu=True, prune_equal_support=True, output_dir=...)` calls, the
second with `anchor_items` read from the first's parquet. It has no mining
logic of its own.

## 2. Arms inside R-GPU

| Step | Variant | Dispatch rule (facts known before the level) | Pin | Exercised by | Pinned by |
|---|---|---|---|---|---|
| input | device layer (`gpu/input_layer.py::build_device_input`, `:461`) | `_env.input_layer() != "host"`, item dtype Int8–Int64 / UInt8–UInt32 (`core/apriori.py:742`, `core/result.py:30`), List dtype, < 2³¹ rows (`input_layer.py:475`), span `max − min + 1 ≤ 2²²` (`:478`, `cpu_miner.INT_SPAN_LIMIT`, imported by name) | `ET_MINER_INPUT_LAYER=host` (only `host` changes anything) | every integer workload | `tests/test_input_layer.py`; legs `input layer device` 1/2 GPUs |
| input | host CSR (`core/matrix.py::_build_csr_from_transactions`, `:521-618`: explode + join + COO) → `gpu/csr_bitvec.py::build_bitvecs_row_split` → `csr_to_bitvec` kernel (`row_split.py:388`) | every column the device layer does not serve (`NOT_SERVED` → `core/apriori.py:779-781`); UInt64/Int128/UInt128 mine column ids when an id lies beyond Int64 (`:789-792`, `:816`) | `ET_MINER_INPUT_LAYER=host` | string / wide variants (`*-str`, `*-wide`); `host` arm of G1 | `tests/test_input_layer.py`, `tests/test_item_dtypes.py`; legs `input layer host` 1/2 GPUs |
| input | prebuilt `bitvecs=` (`shard_prebuilt_bitvecs`, `row_split.py:186-222`) | the caller's array; no rows (`:394`) | — | `*-bv` variants (harness) | `tests/test_gpu_device_affinity.py`, `tests/test_row_split_e2e.py`; no tier leg |
| row order | lex (two radix argsorts of the top-128 rank bits, `csr_to_bitvec.cu:76-101`) + 2,048-row tile-occupancy skip in the K≥3 per-candidate and group kernels (`row_split.py:405-419`, `:695`) | device layer only, unconditionally (`core/apriori.py:752`: `_env.row_order() or "lex"`); host CSR and `bitvecs=` keep their order (`row_split.py:405`) | `ET_MINER_ROW_ORDER=keep\|lex` | every device-layer regime with K≥3 | `tests/test_row_order.py`; legs `row order keep/lex` 1/2 GPUs (plain, forced chunks, count inference) |
| K=2 | row-wise, shared-memory histogram (`count_pairs_k2_rows_shared`) for a chunk of ≤ `K2_ROWS_SHARED_PAIRS = 11264` pairs, global atomics (`count_pairs_k2_rows`) above (`gpu/kernels/k2.py:73`, `:121-122`) | rows present (transactions input), distinct devices, pin ≠ dense (`row_split.py:958`), r = Σ_rows C(len,2) / (pairs × words) < `K2_ROWS_MAX_R = 3.95e-3` (`row_split_chunks.py:129-141`), rows fit (`row_split.py:715-718`); chunks `plan_candidate_chunks` | `ET_MINER_K2_KERNEL=dense\|rows`; shared vs global: none (tests patch `k2.K2_ROWS_SHARED_PAIRS = 0`) | oom2, oom2ml3, sk2ml2, sk2ml3 (global atomics); shared only on small item sets with r below the crossover — none of the regimes at their defaults; forced by `ET_MINER_K2_KERNEL=rows` on dsl | `tests/test_k2_rows.py`; legs `row-wise K=2` (shared/global × plain/forced chunks; by dispatch) 1 GPU, (plain/forced) 2 GPUs |
| K=2 | dense per-candidate (`count_pairs_k2_dense`, `count_pairs_k2_per_candidate`) | n_pairs < `TILED_MIN_GROUP_PAIRS[2] = 120` (≤ 15 frequent items) (`row_split.py:991-994`), or ≥ 2 GPUs with n_pairs > the chunk budget (per-candidate sub-chunks, `row_split_chunks.py:314-318`) | `ET_MINER_TILED_MIN_GROUP_PAIRS` (also moves K≥3) | no regime at its defaults; sk2ml2-bv and oom2-bv on 2 GPUs (sub-chunks) | `tests/test_k2_counting.py`; legs `per-candidate pinned`, `2 GPUs forced chunks` |
| K=2 | dense tiled (`count_pairs_k2_shared` → `count_shared_tiled_dense`) | n_pairs ≥ 120 and the pair space fits one chunk (`row_split.py:991`, `:1003`) | `ET_MINER_TILED_MIN_GROUP_PAIRS` | smoke, deepk, skew, dsl, or003, or002 | `tests/test_shared_kernels.py`; legs `tiled pinned` |
| K=2 | fused tiled (`count_tiled_fused` with `k2_groups`) | one device and n_pairs > the chunk budget (`row_split.py:980-987`), whatever the pins | none | sk2ml2-bv, oom2-bv on 1 GPU (12 GB) | `tests/test_k2_counting.py`, `tests/test_shared_kernels.py`; leg `1 GPU forced chunks` |
| K=3 | row-wise (`count_k3_rows`, frequent-pair bitmap and slots) | `ET_MINER_K3_KERNEL ≠ dense`, `prune_apriori`, max_length ≥ 3 (`row_split.py:428`); rows kept past K=2 (`:968-973`, `:1126`); dense level; unpinned: slots ≤ chunk budget and r3 < `K3_ROWS_MAX_R = 2.5e-4` (`row_split_chunks.py:148-158`); rows fit (`row_split.py:763-766`); every row ≤ 64 active items, pinned or not (`:774-776`) | `ET_MINER_K3_KERNEL=dense\|rows` | oom2ml3, sk2ml3 | `tests/test_k3_rows.py`; legs `row-wise K=3` 1/2 GPUs |
| K≥3 | per-candidate (`count_k3plus_dense`) | per prefix group (`group_kernels`, `row_split_chunks.py:96-120`): pairs < `GROUP_MIN_PAIRS` (20 at K=3, 9 from K=4); also every oversize group of a group-aligned set on ≥ 2 GPUs (`:314-318`) | `ET_MINER_SMALL_GROUP_KERNEL=percand\|group`, `ET_MINER_TILED_MIN_GROUP_PAIRS` | every K≥3 regime | `tests/test_chunked_dense.py`; legs `per-candidate pinned` |
| K≥3 | group (`count_group_pairs`, ≤ 64 suffixes, `gpu/kernels/group_pairs.py:21`, `:62-63`) | pairs ≥ `GROUP_MIN_PAIRS` and < `GROUP_TILED_MIN_PAIRS` (2080 at K=3–6, 877 at 7, 370 from 8) and ≤ 64 suffixes | `ET_MINER_SMALL_GROUP_KERNEL` | every K≥3 regime | `tests/test_shared_kernels.py`; legs `group kernel pinned` 1/2 GPUs |
| K≥3 | tiled (`count_shared_tiled_dense`) | pairs ≥ `GROUP_TILED_MIN_PAIRS[k]` or > 64 suffixes (under `percand`: pairs ≥ `TILED_MIN_GROUP_PAIRS[k]` = 120/91/66/45/32/23, no suffix check, `:112-114`) | `ET_MINER_TILED_MIN_GROUP_PAIRS` | K=3–6 of smoke, deepk, skew, or003, or002, dsl | `tests/test_shared_kernels.py`; legs `tiled pinned` |
| K≥3 | fused tiled (`count_tiled_fused`) | one device, a tiled group with pairs > the chunk budget (`row_split.py:1082-1100`) | none | sk2ml3-bv, oom2ml3-bv on 1 GPU | `tests/test_shared_kernels.py`, `tests/test_subset_prune.py`; leg `1 GPU forced chunks` |
| layout | ESCO sparse tidsets (`gpu/density.py`, `gpu/sparse_csr.py`, `gpu/kernels/csr_warp.py`; kernels `csr_count_range`, `csr_count_gather`, `csr_write_gather` (`_src/csr_warp.cu`), `bitvec_extract_tids`; Rust `build_k3plus_groups_from_flat(with_src_rows=True)`) | `sparse_from_k=` int (from max(3, k)) or `"auto"` (previous level's mean count < n/32, `density.py:43-101`); a misfit stays dense under auto, raises for an int (`row_split.py:905-911`); one-way; multi-GPU with a dense reduce (`sparse_csr.py:291`) | `sparse_from_k=` (opt-in, default None) | `*-esco-auto`, `*-esco3` (harness) | `tests/test_density_transition.py`, `tests/test_density_auto_gpu.py`, `tests/test_csr_warp.py`, `tests/test_group_src_rows.py`, `tests/test_esco_benchmark.py`; legs `ESCO` 1/2 GPUs |
| candidates | subset test (`kernels/subset_index.py`, `_src/_subset_index.cu`) on every K≥3 dense and ESCO kernel | `prune_apriori` and K≥3 (`row_split.py:629`); the index of the free level in a free-set run (`:631-634`); not uploaded when it needs more than ¼ of the smallest free VRAM, and that level counts every candidate (`:645-652`) | — | every K≥3 regime | `tests/test_subset_prune.py`; every leg asserts the index mode |
| candidates | no index (`prune_apriori=False`; also the fallback above) | public option, row-split only (`core/apriori.py:319-324`) | — | `*-noprune` (harness) | `tests/test_routing_contracts.py`; legs `no subset test` |
| candidates | count inference (`SUBSET_INFER`), with deduction bounds (`SUBSET_BOUNDS`, depth ≤ 3, K ≤ 24) | `use_generator_pruning` (needs `prune_apriori`), complete-lattice runs (`row_split.py:636-637`); bounds wherever levels k−1…k−3 are kept and fit (`:550`, `:638-662`) | `ET_MINER_DEDUCTION=0` (only `0` changes anything) | dsl-infer | `tests/test_deduction.py`, `tests/test_free_set_semantics.py`; legs `count inference` (dispatch/tiled/per-candidate/forced × bounds on/off) |
| reduce (≥ 2 GPUs) | compacted (written entries only) or dense (whole chunk arrays); NCCL reduce-to-root under `NCCL_P2P_LEVEL=NVL` (`gpu/nccl.py:83-142`, `:299-305`) or staged through host memory (`:259-281`) | compacted with an index on ≥ 2 GPUs (`row_split.py:1063`), never at K=2, row-wise K=3 or ESCO; staged when NCCL is off or fails | `ET_MINER_REDUCE=dense`, `ET_MINER_DISABLE_NCCL=1`, `ET_MINER_DIRECT_D2D=1` | every 2-GPU K≥3 regime | `tests/test_compact_reduce.py`, `tests/test_cross_device_copy.py`, `tests/test_nccl_teardown.py`; legs `compacted / dense reduce` |
| host step | R2 prefix-group build: Rust `build_k3plus_groups_from_flat` (`gpu/kernels/k3plus.py:277-280`) / numpy `_build_k3plus_groups_numpy` (`:312`), warns once when missing or stale | every dense K≥3 level (`row_split.py:1040`) and ESCO level (`:930`); not K=2 (`k2_groups`) or row-wise K=3 (`k3_layout`) | `ET_MINER_DISABLE_RUST=1` (read at import of `et_miner.backends`) | every K≥3 regime | `tests/test_group_src_rows.py`, `tests/test_disable_rust.py`, `tests/test_rust_ffi_boundary.py` |
| host step | R4 free-set prune: Rust `prune_non_free_flat` (`gpu/mining.py:101-110`, with the pre-0.3.0 name `prune_closed_flat` accepted) / Python fallback (`mining.py:48`, `:119-121`, **silent**) | free-set runs from K=2 (`row_split.py:1166-1175`) | `ET_MINER_DISABLE_RUST=1` | dsl-free | `tests/test_prune_non_free_sorted.py`, `tests/test_free_set_semantics.py` |
| host step | R5 bitvector build: Rust `build_column_bitvecs_u64` (`gpu/bitvec.py:70-82`) | **not an R-GPU step**: the fallback of `_build_gpu_bitvec_matrix` when the `csr_to_bitvec` build raises, reached only from R-SON's GPU passes (`son.py:531-539`, `:566-571`) and `count_support_batched(use_gpu=True)`; R-GPU's host CSR builds through `build_bitvecs_row_split` with no Rust fallback (`csr_bitvec.py:496-530`) | `ET_MINER_DISABLE_RUST=1` | no regime (the CUDA build succeeds); forced only by a failing CUDA build | `tests/test_rust_ffi_boundary.py` |

Sixteen kernels are registered in `gpu/kernels/loader.py::_KERNEL_FILES`
(`:171-188`; preludes `:194-199`), plus the unregistered `popcount_u64`
ElementwiseKernel (`:240-260`, K=1, ESCO transition, K=3 pair bitmap):

| Kernel (`_src/*.cu`) | Wrapper | Arm |
|---|---|---|
| `count_pairs_k2_dense` (`pairs_k2_dense.cu`) | `count_pairs_k2_per_candidate` | K=2 per-candidate |
| `count_pairs_k2_rows`, `count_pairs_k2_rows_shared` (`pairs_k2_rows.cu`) | `count_pairs_k2_rows` | K=2 row-wise (global / shared) |
| `count_k3_rows` (`k3_rows.cu`) | `count_k3_rows` | K=3 row-wise |
| `count_k3plus_dense` (`k3plus_dense.cu`) | `count_k3plus_per_candidate` | K≥3 per-candidate, oversize sub-chunks |
| `count_group_pairs` (`group_pairs.cu`) | `count_group_pairs` | K≥3 group |
| `count_shared_tiled_dense` (`shared_tiled.cu`) | `count_shared_tiled_allcounts`, `count_pairs_k2_shared` | K≥3 tiled, K=2 tiled |
| `count_shared_tiled_fused` (`shared_tiled.cu`) | `count_tiled_fused` | one-GPU fused, K=2 and K≥3 |
| `csr_count_range`, `csr_count_gather`, `csr_write_gather` (`csr_warp.cu`) | `count_csr_range`, `count_csr_gather`, `write_csr_gather` | ESCO count, materialize |
| `bitvec_extract_tids` | `convert_shards_to_csr` | ESCO transition |
| `count_itemsets_batch` (`itemset_count.cu`) | `count_itemsets_cuda` | not R-GPU: SON pass 2, `count_support_batched(use_gpu=True)` |
| `csr_to_bitvec` (`csr_to_bitvec.cu`) | `build_bitvecs_gpu` | host CSR input |
| `ids_to_bitvec`, `row_order_keys` (`csr_to_bitvec.cu`) | `_scatter_bitvecs`, `_row_order` | device input, lex |

## 3. Arms inside R-CPU and R-SON's CPU passes

No pins. The constants are module globals read at call time, so a bench child
selects an arm by setting the module attribute before the call. Exceptions:
`INT_SPAN_LIMIT` is imported by name into `gpu/input_layer.py:60`;
`son.LOCAL_SUPPORT_FACTOR` is captured as a default (`son.py:141`,
`multi_gpu.py:45`, `:124`); `RowSpace` fixes its bitvector decision and
`proj_min` when built (`cpu_miner.py:645-647`).

| Step | Variant | Dispatch rule | Module constant | Exercised by | Pinned by |
|---|---|---|---|---|---|
| input | `_int_ids` (bincount + lookup table) | List of Int8–Int64 / UInt8–UInt32, span ≤ 2²² (`cpu_miner.py:157-165`) | `INT_SPAN_LIMIT` | every integer workload | `tests/test_cpu_miner.py` |
| input | Polars `group_by` / `replace_strict` | everything else (`:184-205`) | — | `*-str`, `*-wide` | `tests/test_cpu_miner.py:697`, `tests/test_item_dtypes.py` |
| K=2 | Gram `M.T @ M` (scipy int32, column blocks under `GRAM_BUDGET_BYTES`; `count_pairs`, `:331-378`) | bitvectors do not fit `BITVEC_BUDGET_BYTES` (512 MB), or pairs × words > `GRAM_BITVEC_RATIO` (3.0) × Σ_rows C(len,2) (`_k2_on_bitvecs`, `:847-854`) | `GRAM_BITVEC_RATIO`, `BITVEC_BUDGET_BYTES` | wide, or003, or002, or0001k2 | `tests/test_cpu_miner.py` |
| K=2 | column bitvectors (`count_pairs_bitvec`, `:380`) | otherwise | same | smoke, deepk, skew, dsl | `tests/test_cpu_miner.py` |
| K≥3 | pair popcounts on the prefix AND's non-zero words (`_count_group_bitvec`) | per prefix group with fewer than `proj_min_suffixes(words)` suffixes: 40 at ≤ 2,048 words, 80 above (`PROJ_MIN_SUFFIXES`, `:83`, `:616-621`, `:798-826`) | `PROJ_MIN_SUFFIXES` | every K≥3 regime | `tests/test_cpu_miner.py` |
| K≥3 | Gram projection over the prefix's rows (`_count_group_proj`); also every group when the space has no bitvectors (CSC tidset) | otherwise | same | wide, or002, dsl | `tests/test_cpu_miner.py`, `tests/test_sparse.py` |
| SON pass 2, K≥3 (`count_itemsets`, `:873-931`) | per candidate (AND of k bitvectors) | the level's space has bitvectors and ≤ `PER_CANDIDATE_MAX_WORDS` (512) words, or fewer than `PER_CANDIDATE_MEAN_GROUP` (2.5) candidates per prefix group | `PER_CANDIDATE_MAX_WORDS`, `PER_CANDIDATE_MEAN_GROUP` | dsl-son, deepk-son | `tests/test_cpu_miner.py`, `tests/test_sparse.py` |
| SON pass 2, K≥3 | by prefix group (`count_candidates`) | otherwise, or no bitvectors | same | dsl-son | same |

`mine_cpu` never counts per candidate. `count_itemsets` also serves
`count_support_sparse`'s non-pairs.

## 4. Capability matrix

S supported, R raises `ValueError`, I accepted and silently ignored.

| Capability | R-CPU | R-GPU (transactions) | R-GPU (`bitvecs=`) | R-SON CPU passes | R-SON GPU passes | R-SON-MG | R-RUST |
|---|---|---|---|---|---|---|---|
| `bitvecs=` | — (selects R-GPU) | — | S | I (`bitvecs=` wins, `streaming` dropped) | I | I | — |
| string / categorical ids | S | **broken**: host CSR, then `col_to_item` is copied into an int64 array (`row_split.py:432-434`): non-numeric strings raise after the bitvector build, numeric strings come back as Int64 in string order, floats are truncated (1.2 and 1.7 both become 1). No test | int64 copy, same | S | S (column ids, `son.py:521-555`) | S | column ids only |
| UInt64 / Int128 / UInt128, span > 2²² | S (Polars input) | S (host CSR; column ids beyond Int64; R with output_dir/resume/anchors beyond Int64, `core/apriori.py:734-740`) | ids into int64 | S | S | S | — |
| `output_dir` | R | S (per-K parquet; returns an empty frame, `row_split.py:1266-1270`) | S | R | R | R | — |
| `resume_from_k` | R | S with `output_dir` and ≥ 2; **I without `output_dir`** (`row_split.py:546`) | same | R | R | R | — |
| free sets | S | S | S | R | R | R | — |
| count inference | S | S (needs `prune_apriori`) | S | R | R | R | — |
| `anchor_items` | R | S | R | R | R | R | — |
| `sparse_from_k` | I (validated, not passed, documented) | S | S | R | R | R | — |
| `prune_apriori=False` | R | S | S | R | R | R | — |
| `profile` | S | S | S | S | S | R | — |
| `max_ram_gb` / `max_vram_gb` | I | S | S | I | I (pass 1 passes none) | I | — |
| multi-GPU | `n_gpus` I | S | S | — | one device | S | — |
| out-of-core | — | — | caller-owned | S | S | S | — |
| `level_callback` | S | S | S | I | I | I | — |
| `progress_callback` | I | I | I | S (more than one chunk) | S | S | — |
| `max_length` | S | S | S | S | S | S | S |

## 5. Knobs, parameters and exports that select a path

**`ET_*` knobs** (all read through `_env.py`; documented in its docstring
except `ET_MINER_ESCO_MATERIALIZE`, a removed knob that raises at `_env.py:241`):

| Knob | Selects | Read at |
|---|---|---|
| `ET_MINER_INPUT_LAYER` | host CSR (`host`) | `core/apriori.py:742` |
| `ET_MINER_ROW_ORDER` | keep / lex | `core/apriori.py:752` |
| `ET_MINER_K2_KERNEL` | dense / rows at K=2 (also the device layer's `keep_rows`) | `row_split.py:958`, `row_split_chunks.py:137`, `core/apriori.py:751` |
| `ET_MINER_K3_KERNEL` | dense / rows at K=3 | `row_split.py:428`, `row_split_chunks.py:154`, `core/apriori.py:751` |
| `ET_MINER_SMALL_GROUP_KERNEL` | percand / group below the tiled crossover | `row_split_chunks.py:110` |
| `ET_MINER_TILED_MIN_GROUP_PAIRS` | tiled crossover, every level | `row_split_chunks.py:83`, `:115` |
| `ET_MINER_MAX_CHUNK_CANDS` | caps the chunk budget (forces chunks, fused, sub-chunks) | `row_split_chunks.py:250` |
| `ET_MINER_DEDUCTION` | bounds off (`0`) | `row_split.py:550`, `:641` |
| `ET_MINER_REDUCE` | dense reduce (`dense`) | `row_split.py:1063` |
| `ET_MINER_DISABLE_NCCL` | staged reduce | `gpu/nccl.py` |
| `ET_MINER_DIRECT_D2D` | direct peer copies | `gpu/nccl.py:59-80` |
| `ET_MINER_DISABLE_RUST` | every Rust role's fallback (read at import) | `backends.py:49-61` |
| `ET_MINER_FILTER_IMPL`, `ET_MINER_ROW_BALANCE=nnz`, `ET_MINER_KERNEL_VARIANT`, `ET_MINER_ESCO_MATERIALIZE` | removed, raise | `_env.py:221-245` |

`config.py:276` reads `ET_MINER_APRIORI_*`, `ET_MINER_STREAMING_*`,
`ET_MINER_CACHE_*` and `ET_MINER_LOGGING_*` outside `_env.py`; none selects a
route (`apriori.sparse` and `apriori.n_jobs` are read nowhere outside
`config.py`). The CLI selects R-CPU or, with `--streaming`, R-SON's CPU passes
(`cli.py:177-193`); no CLI flag selects a GPU path, `sparse` or `batch_size`.

**Public parameters that select or tune a path:** `apriori(use_gpu,
streaming, n_gpus, chunk_size, memory_budget_gb, bitvecs, sparse_from_k,
prune_apriori, use_generator_pruning, prune_equal_support, sparse, batch_size,
n_jobs)`; `apriori_streaming(use_gpu, chunk_size, local_support_factor,
sparse, batch_size, n_jobs)`; `apriori_streaming_multi_gpu(n_gpus, chunk_size,
local_support_factor, sparse, batch_size, n_jobs)`; `count_support_batched(
use_gpu, sparse, batch_size, n_jobs)`.

Silent today (verified):
- `batch_size` is validated by `apriori()` (`core/apriori.py:89-94`) and read
  only by `count_support_batched` (`core/matrix.py:412-433`), which no route
  calls; it is forwarded unread at `core/apriori.py:695`, `:714`, `son.py:278`,
  `multi_gpu.py:261`.
- `sparse=` warns `DeprecationWarning` on the CPU route and SON's CPU passes
  (`core/apriori.py:672-679`, `son.py:225-232`, the latter only when
  `apriori_streaming` is called directly) and is read only by
  `count_support_batched` (`core/matrix.py:398-404`). It is ignored without a
  warning on R-GPU, `bitvecs=`, R-SON's GPU passes and R-SON-MG. `config.py:57`
  defaults it to `True`.
- `streaming=True, n_gpus>1` runs R-SON-MG on the GPU whatever `use_gpu` says
  (`core/apriori.py:684`).
- Also found: `resume_from_k` without `output_dir` (`row_split.py:546`),
  `level_callback` on every SON route, `n_gpus` on R-CPU, `max_*_gb` on R-CPU
  and SON, `streaming`/`chunk_size`/`memory_budget_gb`/`progress_callback`
  with `bitvecs=`.

**Public exports that select a path** (`et_miner.__all__`, `__init__.py:89-128`;
`et_miner.core.__all__`, `core/__init__.py:18-29`): `apriori`,
`apriori_streaming`, `apriori_streaming_multi_gpu`, `apriori_from_csr`,
`count_support_batched`, `count_support_sparse`, `count_support_vectorized`,
`build_boolean_matrix`. Not exported at the top level: `mine_two_phase`
(`gpu.row_split`), `count_itemsets_cuda` (`gpu.kernels`).

## 6. Code no mining route uses

Method: a static import graph of `src/et_miner` (every `import`, including the
ones inside functions; the only `importlib` use is `importlib.resources` in
`gpu/kernels/loader.py:12`) from `et_miner`, `et_miner.cli` and
`et_miner.__main__`, then a function-level pass from the public `__all__`s,
`cli.main` and the mining entries; plus `sys.modules` after
`import et_miner, et_miner.cli`, which holds none of the modules below.
47 of 52 modules are reachable; the other five are the three below,
`streaming/ramdisk.py` (out of scope) and `synthetic.py` (the test and bench
fixture module).

| Code | Proof | Action (Phase 1) |
|---|---|---|
| `core/candidates.py` (7 functions) | no importer in `src/` (last route imports removed in `7669fb7`, `2833775`); referenced by a comment (`core/matrix.py:145`), a docstring (`core/cpu_miner.py:489`), the reference oracle of `tests/test_cpu_miner.py:277-283` (27 cases), `bench/cpu/stakes.py:273` (`pyref` variant), `bench/cpu/son_stakes.py:139` (the `current` arm, which already refuses to run on this tree) | delete; the test oracle becomes an inline brute-force join and subset test with the same exact-equality assertions; `pyref` leaves `stakes.py`; `son_stakes.py` refuses before the import |
| `gpu/type_safety.py` | no importer anywhere (src, tests, bench) | delete |
| `gpu/utils.py` | no importer anywhere; `streaming/multi_gpu.py:72`, `:93` hold their own copies | delete |
| `core/apriori.py::_prune_equal_support` | no caller anywhere (prose only: `core/rules.py:503`, `tests/test_self_sufficiency.py:34`) | delete, fix the prose |
| `core/apriori.py::_infer_count_from_subsets` | only `tests/test_free_set_semantics.py:145-153` | move under `tests/` with its test |
| `gpu/csr_bitvec.py`: `PinnedBufferPool`, `get_default_pinned_pool`, the `buffer_pool` parameters and branch, `build_bitvecs_from_gpu_arrays` | no caller passes a pool; `build_bitvecs_from_gpu_arrays` has no caller | delete with their `__all__` entries and docs |
| `gpu/kernels/loader.py::_get_device_lock` (+ `_cuda_device_locks`, `_cuda_locks_init`) | no caller; the docs that describe per-device locks (`loader.py:5-6`, `:15-17`, `csr_bitvec.py:450-451`) are stale | delete, fix the docs |
| Rust `get_num_threads` (`rust_ext/src/lib.rs:214`) | no caller in src, tests or bench | delete |
| stale comments | `gpu/row_split.py:1028` ("Pair cache disabled"; no pair cache exists), `gpu/csr_bitvec.py:323-324` (a "null model" that does not exist), `core/matrix.py:143-147` (ordering from `core.candidates`), "Phase N" labels in `gpu/csr_bitvec.py` and `gpu/bitvec.py:59` | delete or rewrite |

Reachable only through public exports that no route calls (not dead; C4
decides them): `count_support_batched` with `count_support_vectorized`,
`count_support_gpu_bitvec` and all of `core/sparse.py`; `core/progress.py`.
Left alone: `_logging.py`'s unused helpers (`disable_logging`, `LogContext`,
`log_mining_*`; no mining path), `config.get_config`/`set_config` (`config.py`
is out of scope beyond options that select a removed path) and
`streaming/ramdisk.py` (out of scope).
