# GPU-layer inventory — `main` @ 7221037

Verified against the code before any consolidation change: static tracing of
every route, kernel wrapper, Rust call site and `ET_*` knob, plus the baseline
test runs below. `file:line` references are to `7221037`.

## Baseline

Box: 2× RTX 3060 12 GB (sm_86, driver 550.144.03, CUDA 12.4, CuPy 14.1.1, no
P2P, NCCL init OK with the reduce binding), Ryzen 5 5600X (6C/12T, `nproc` 12),
31 GB RAM. The campaign brief assumed 2× RTX 3090 24 GB; every number in this
directory is from the 3060 box.

| Command | Result |
|---|---|
| `bench/selfcheck.py` | READY — 19 kernels compiled on both devices |
| `pytest tests/test_tier_equivalence.py -q` | 10 passed |
| `pytest -q -m "not slow"` | 792 passed, 4 skipped (Online Retail dataset absent), 1 xfailed (strict, `test_routing_contracts.py:463`, pre-existing) |
| `pytest -q -m "gpu and not slow"` | 200 passed, 2 skipped (Online Retail dataset absent) |

With `datasets/online_retail_ii/transactions.parquet` present, the two
`test_smoke_correctness.py::*::test_support_001` tests (not marked slow) do not
terminate: at `min_support=0.001` the lattice already holds 11.46M itemsets at
K≤4 (GPU, 73 s) and keeps growing, and the CPU half reached 18 GB RSS after
13 minutes. They were skipped on every recorded baseline because the dataset
is gitignored. The consolidation runs deselect those two tests.

## 1. Routes

`apriori()` (`core/apriori.py:444`) dispatches in this order: `bitvecs=`
branch (`:622`), `streaming` (`:719`), `use_gpu` (`:764`), CPU loop (`:831`).
Parameter/route mismatches are refused above the routing by
`_validate_route_support` (`:218`).

| id | Route | Entry | Reached by | K=2 | K≥3 | Sparse CSR |
|---|---|---|---|---|---|---|
| A | single-GPU bitvec miner | `gpu/mining.py:469` `_apriori_from_bitvecs` | `use_gpu`, `n_gpus<=1`, no anchors/pruning/`gpu_resident`; `bitvecs=` without pruning or `gpu_resident`; the D (`use_gpu=True`) and E single-chunk fall-throughs | `dispatch_k2` (`dispatch.py:73`): fan-out `count_pairs_fused_k2_multi_gpu` when `min(n_gpus, devices)>1` and pairs ≥ 15M, else `count_pairs_k2_shared_fused` (variant shared/auto) or `count_pairs_fused_k2` (legacy) | `dispatch_k3plus_fused` (`dispatch.py:136`): fan-out `count_k3plus_fully_fused_multi_gpu` at ≥ 500K candidates, else `count_k3plus_shared_fused` or `count_k3plus_fully_fused`. Prefix groups built in pure Python (`build_k3plus_groups`), no Apriori subset prune, no tiny/mega routing | `sparse_from_k` int (floored to 3) or `"auto"`: `convert_shards_to_csr` → `run_sparse_level` → `materialize_survivors` on one shard; groups via Rust R2 |
| B | GPU-resident miner | `gpu/mining.py:919` | `gpu_resident=True` with `use_gpu`, `n_gpus<=1`, no anchors/pruning; `bitvecs=` + `gpu_resident` without pruning; D pass 1 per chunk | `dispatch_k2_gpu_resident` → `count_pairs_fused_k2_gpu_resident` (legacy `count_pairs_fused_k2` kernel, whatever the variant) or its `_multi_gpu` fan-out | `dispatch_k3plus_gpu_resident` → GPU prefix groups (`build_prefix_groups_gpu`) → `count_k3plus_gpu_resident` + `decode_candidates_gpu` + CuPy lexsort, or the `_multi_gpu` fan-out | none; `sparse_from_k` is silently ignored |
| C | row-split miner | `gpu/row_split.py:93` | `use_gpu` and any of `n_gpus>1`, `anchor_items`, `prune_equal_support`; `bitvecs=` + `prune_equal_support` (one shard, `prune_non_free` forced, `n_gpus` only logged); `mine_two_phase` | one synthetic group → `plan_group_chunks` → `count_pairs_k2_allcounts`: shared tiled only when one chunk covers the whole pair space and the space has ≥ 64 pairs, else legacy `count_pairs_k2_dense` | Rust R2 groups → Rust R3 prune (`prune_apriori`) → `plan_group_chunks` → `count_k3plus_allcounts`, shared or legacy per chunk: mega-groups (> chunk budget) and tiny groups (< `ET_MINER_TILED_MIN_GROUP_PAIRS`, default 64) go legacy; the tiny routing is dropped per level when it would multiply a > 64-chunk plan by > 4 | fixed K or `"auto"`: per-shard CSR, `plan_candidate_chunks`, csr_warp kernels, same reduce + compact filter |
| D | SON, single GPU or CPU | `streaming/son.py:79` | `streaming`, `n_gpus<=1`, rows > effective chunk size (`chunk_size`, default 10M from `apriori()`; or derived from `memory_budget_gb`) | pass 1: B per chunk if `gpu_resident`; otherwise (and as B's fallback) `_mine_chunk_frequent` → `count_support_batched(use_gpu)` → `count_support_gpu_bitvec` (CSR + bitvec rebuild per level, Python loop per itemset) or Polars/scipy | same as K=2 | none |
| E | SON, multi-GPU | `streaming/multi_gpu.py:114` | `streaming`, `n_gpus>1`, rows > effective chunk size; always GPU (`use_gpu` ignored) | waves of one chunk per GPU; both passes `count_support_batched(use_gpu=True)` → `count_support_gpu_bitvec` | same | none |
| F | CPU | `core/apriori.py:831` | not `use_gpu`, not streaming; D fall-through with `use_gpu=False` | `count_support_batched` → Polars `count_support_vectorized` or scipy `count_support_sparse` (K=2 `_sparse_matmul`, MKL when `sparse_dot_mkl` imports; chunk-parallel only when `n_jobs!=1`, pairs > 100 and rows > 10K) | Rust `count_itemsets_simd` when built, else scipy per itemset (sequential or ThreadPool) | n/a |

Every GPU route's K=1 is the `popcount_u64` ElementwiseKernel. D pass 2 is
`count_itemsets_cuda` (`count_itemsets_batch`) with `gpu_resident`, otherwise
the pass-1 counter with `enable_length_filter=False`.

### D and E error handling (the SON completeness hazard)

- D pass 1, matrix build: `except Exception` → WARNING `Chunk {} failed` →
  `continue` (`son.py:251-259`). The chunk contributes no candidates.
- D pass 1, B per chunk: any exception (including the 10M-ceiling
  `RuntimeError`) → WARNING → re-mine with the caller's `use_gpu`
  (`son.py:269-294`, `:534-536`). Complete but slow.
- D pass 2, `gpu_resident` counter: exception → exact fallback (`son.py:406-416`).
- E pass 1: matrix-build failure drops the chunk (`multi_gpu.py:307-309`); any
  other exception is logged at ERROR and dropped (`:386-387`).
- E pass 2: any exception is logged and the chunk's counts are dropped while
  support is still divided by the full row count (`:543-555`).
- `count_support_gpu_bitvec` raises without the Rust extension even though its
  build and count are CUDA (`gpu/bitvec.py:190-193`), so E without Rust drops
  every chunk and returns an empty frame.

### Single-chunk fall-through (n_rows ≤ chunk size)

D calls `apriori()` again with `min_support`, `max_length`, `item_col`,
`use_gpu`, `batch_size`, `profile`, `show_progress`, `sparse`, `n_jobs`: A with
`use_gpu`, else F. `gpu_resident` and `progress_callback` are dropped;
`level_callback`, `sparse_from_k` and the memory guards reset to defaults. E
calls `apriori(use_gpu=True)` → A on one GPU, dropping `n_gpus` and
`progress_callback`.

## 2. Capability matrix

`yes`/`no`/`partial`; "refused" = `_validate_route_support` raises.

| Capability | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| `bitvecs=` input | yes | yes | partial: with pruning, one shard | no (bitvecs branch runs first) | no | no |
| `output_dir` | refused, except the known hole `bitvecs`+`use_gpu`+`n_gpus>1` (dropped; strict xfail) | refused, same hole | yes (returns an empty frame) | refused | refused | refused |
| `resume_from_k` | refused (same hole) | refused (same hole) | partial: needs `output_dir`, silently mines from K=1 without it | refused | refused | refused |
| `prune_equal_support` | routed to C | routed to C | yes | refused | refused | yes |
| `prune_apriori=False` honoured | refused | refused | yes | refused | refused | refused (test always runs) |
| `anchor_items` | refused | refused | yes (transactions entry only) | refused | refused | refused |
| `sparse_from_k` | yes | silently ignored | yes | silently ignored | silently ignored | silently ignored |
| `profile` | yes (over-refused for `bitvecs`+`use_gpu`+`n_gpus>1`) | yes (same over-refusal) | refused | yes | refused | yes |
| `max_ram_gb`/`max_vram_gb` | yes (MemoryError between levels) | silently ignored | silently ignored (VRAM-budgeted chunks) | ignored | ignored | ignored |
| 10M result ceiling | legacy fused levels; the fan-out per device under any variant | every K≥2 level, per device | none | via B (swallowed into a re-mine) | none | none |
| multi-GPU | fan-out via `bitvecs=` only, full bitvec copy per device through host RAM | fan-out via `bitvecs=` only | row shards + NCCL reduce / staged D2D | no | chunk waves | no |
| out-of-core | no | no | partial: per-K flush; input CSR in host RAM | yes | yes | no |
| `use_generator_pruning` | silently ignored | ignored | ignored | refused | refused | yes |
| `level_callback` | yes (dense K≥3 reports n_frequent as n_candidates) | yes (K≥3 n_candidates = 0) | yes | not a parameter; dropped | dropped | yes |
| `progress_callback` | ignored | ignored | ignored | yes (per chunk, both passes) | yes | ignored |

## 3. Kernels

Registry: `gpu/kernels/loader.py:275` (`_KERNEL_FILES`, 19 entry points in 15
`.cu` files) plus the `popcount_u64` ElementwiseKernel (`loader.py:347`). No
other RawKernel/ElementwiseKernel exists in `src/`.

| Family | Entry point (.cu) | Wrappers | Routes | Selected by | Directly pinned by |
|---|---|---|---|---|---|
| legacy fused, per candidate | `count_pairs_fused_k2` (pairs_k2.cu) | `k2.py:10` `count_pairs_fused_k2`, `:89` `_multi_gpu`; `gpu_resident.py:305/414` | A K=2 (legacy), A fan-out (any variant), B K=2 (always) | `ET_MINER_KERNEL_VARIANT=legacy`; fan-out threshold; `gpu_resident` | `test_fused_k2_kernel.py`, `test_shared_kernels.py::TestFusedEquivalence::test_k2_fused_set_equal`, `test_gpu_device_affinity.py` (k2 tests), `test_kernel_input_guards.py` |
| legacy fused, per candidate | `count_k3plus_from_groups` (k3plus_fullyfused.cu) | `k3plus.py:237` `count_k3plus_fully_fused`, `:349` `_multi_gpu` | A K≥3 (legacy), A fan-out | variant, fan-out threshold | `test_shared_kernels.py::TestFusedEquivalence::test_k3_set_equal_with_boundary_counts`, `test_gpu_device_affinity.py::test_k3plus_fully_fused` |
| explicit-candidate fused | `count_itemsets_fused_k3plus` (k3plus_fused.cu) | `k3plus.py:14`, `:95` | **none** (`dispatch_k3plus` has no caller) | — | `test_fused_k3plus_kernel.py`, `test_gpu_correctness.py::TestKernelKCap::test_an_oversized_candidate_raises_instead_of_miscounting`, `test_gpu_device_affinity.py::test_k3plus_candidate_list` |
| GPU-resident | `count_k3plus_gpu_resident`, `decode_candidates_gpu` | `gpu_resident.py:159`, `:594` | B K≥3 | `gpu_resident` | `test_kernel_input_guards.py`, `test_gpu_device_affinity.py` (gpu_resident tests), `test_gpu_resident_e2e.py` |
| legacy dense, per candidate | `count_pairs_k2_dense`, `count_k3plus_dense` | `k2.py:233`, `k3plus.py:694` | C (legacy variant; mega, tiny, multi-chunk K=2, pair space < 64) | variant, `plan_group_chunks`, `ET_MINER_MAX_CHUNK_CANDS`, `ET_MINER_TILED_MIN_GROUP_PAIRS` | `test_shared_kernels.py::TestDenseEquivalence` (`variant="legacy"`), `::TestVariantWiring`; forced chunks in `test_chunked_dense.py::TestChunkedEquivalenceGPU`, `test_free_set_semantics.py` (cap 37), `test_density_auto_gpu.py` (cap 700), `test_row_split_e2e.py::TestOOMRegression` (slow) |
| shared/tiled | `count_shared_tiled_dense`, `count_shared_tiled_fused` (shared_tiled.cu) | `shared_tiled.py:68` (dense, also K=2 via `count_pairs_k2_shared`), `:141` `_run_fused` (via `:204`, `:220`) | C dense (default); A single-GPU fused (default) | variant `auto`→`shared` | `test_shared_kernels.py` (dense and fused equivalence, overflow retry with `max_results=3`), `test_tier_equivalence.py::test_multi_gpu_shared_matches_oracle` |
| sparse CSR | `csr_count_range`, `csr_count_gather`, `csr_write_gather` (csr_warp.cu); `bitvec_extract_tids` | `kernels/csr_warp.py`, `gpu/sparse_csr.py` | A and C after the density transition | `sparse_from_k` | `test_csr_warp.py`, sparse tier-chain legs, `test_density_auto_gpu.py`, `test_gpu_correctness.py::test_sparse_csr_branch`, `test_row_split_e2e.py::TestClosedPruning/TestTwoPhaseSparse` |
| generic batch | `count_itemset_fused`, `count_itemsets_batch` (itemset_count.cu) | `batch.py:10` `count_itemsets_cuda` | D pass 2 with `gpu_resident` | `gpu_resident` | reference oracle only in `test_fused_k2_kernel.py`, `test_fused_k3plus_kernel.py` |
| infrastructure | `compact_threshold` | `kernels/filter.py:127` | C dense, A/C sparse | `ET_MINER_FILTER_IMPL=compact` (int32 counts) | `test_compact_threshold.py`, `test_chunked_dense.py::test_forced_chunks_all_filter_impls_agree` |
| infrastructure | `csr_to_bitvec` | `gpu/csr_bitvec.py:194` | bitvec build for every GPU route | — | `test_cuda_csr_bitvec.py`, `test_balance_split.py::TestBalanceSplitGPU` |
| infrastructure | `popcount_u64` | `loader.py:347` | K=1 everywhere; CSR conversion; D/E counting | — | `bench/selfcheck.py` only |
| benchmark generators | `fill_row_ids`, `bootstrap_copy` | `gpu/csr_build.py:122`, `:466` | **none** | — | `test_cuda_csr_build.py` |

What the mandated tier chain actually runs: no leg sets
`ET_MINER_KERNEL_VARIANT`, so the "single-GPU legacy" leg runs the shared fused
kernels (route A) and the "multi-GPU legacy" leg runs the same kernels as
"shared multi-GPU" (route C: shared, plus legacy dense for tiny/mega groups).
The legacy fused kernels are not in the chain at all.

Result ceiling (`loader._warn_result_truncation`, raises on overflow): `k2.py:40`
and `:163` (hard-coded 10M), `k3plus.py` `max_results=10_000_000` defaults
(`:14`, `:95`, `:237`, `:349`), `gpu_resident.py:229`, `:365`, `:505`, `:701`.
The shared fused path re-runs at the exact reported capacity instead
(`shared_tiled.py:156-194`); `compact_threshold` counts first and scatters at
exact capacity (`filter.py:139-180`).

K cap (`loader.py:86-107`, `MAX_SUPPORTED_K = 62`): enforced by `k3plus.py`
(`:32`, `:112`, `:261`, `:369`, `:730`), `shared_tiled.py:42-52`,
`gpu_resident.py:210`, `:639`. The value 62 exists only because
`count_itemsets_fused_k3plus` is correct to K=62; the group kernels are correct
to K=64 and `shared_tiled.cu` caches `s_pref[62]` (prefix ≤ 62, K ≤ 64).

## 4. Rust roles

Detection: `backends.py:43-51` sets `_rust_ext` and `RUST_INSTALLED`, two
sources of truth. Import-time copies: `src/et_miner/__init__.py:66-70`
(`HAS_RUST` and a direct `from et_miner_rust import apriori_from_csr`),
`core/sparse.py:28`, `gpu/bitvec.py:17`. Lazy readers: `k3plus.py:805`,
`mining.py:98`, `:190`, `:340`.

| Role | Call site | Rust fn | Fallback | Loud? | Routes | Pool |
|---|---|---|---|---|---|---|
| R1 CPU K>2 counting | `core/sparse.py:531` | `count_itemsets_simd` (`count_itemsets_sparse` on a stale wheel) | scipy per itemset, sequential or ThreadPool (`sparse.py:405-443`) | no (DEBUG) | F sparse, D CPU | scoped, `n_jobs` budget (`-1` = global) |
| R2 prefix-group build | `k3plus.py:782` `build_k3plus_groups_from_flat` | same name | `_build_k3plus_groups_numpy` | WARNING once (flag shared with R3) | C dense + sparse, A sparse | global |
| R3 Apriori group prune | `mining.py:238` `_prune_groups_apriori` | `prune_groups_apriori` | Python double loop | WARNING once, masked by R2's flag in practice | C with `prune_apriori` | global |
| R4 free-set prune | `mining.py:178` `_prune_non_free_mask` (`row_split.py:744`, then a numpy fancy-index at `:745-748`) | `prune_non_free_flat` | `_prune_non_free_mask_python` | no (DEBUG) | C with `prune_equal_support` (from K=2) | global |
| R5 bitvec build | `gpu/bitvec.py:80` | `build_column_bitvecs_u64` | none: CUDA `csr_to_bitvec` is primary; Rust runs only if CUDA fails | WARNING when CUDA fails | A, B, D; not C | global |
| R6 public API | `src/et_miner/__init__.py:70` `apriori_from_csr` | same | none (a stub raises `MiningError`) | — | none internal | global |

`count_support_gpu_bitvec` (`gpu/bitvec.py:162`) additionally hard-requires
`RUST_INSTALLED` although it uses no Rust unless the CUDA build fails.

`sparse=None` (F): per level, after the length filter,
`_choose_counting_strategy` (`sparse.py:899`) picks sparse when C(n,2) > 100K
(n ≥ 448 frequent items) or when n·rows/8 > 1 GiB; its density rule can never
fire first. MKL: `_setup_mkl_library_path` runs when `core.sparse` is imported
(at package import); only `_sparse_matmul` uses MKL (K=2).

## 5. Knobs that select a kernel or route

| Knob | Values | Acts on | Read at |
|---|---|---|---|
| `ET_MINER_KERNEL_VARIANT` | `auto` (→ shared) / `legacy` / `shared` | A single-GPU fused kernels, C dense kernels; not B, not the A fan-out | `_env.py:126` ← `dispatch.py:48` |
| `ET_MINER_TILED_MIN_GROUP_PAIRS` | int, default 64, 0 disables | C tiny-group routing to legacy dense | `row_split_chunks.py:208` |
| `ET_MINER_MAX_CHUNK_CANDS` | int cap | C dense and sparse chunk budget; forces multi-chunk K=2 (legacy) and mega-groups | `row_split_chunks.py:161` |
| `ET_MINER_FILTER_IMPL` | `compact` / `cupy` / `cpu` | survivor filter (C, sparse levels) | `filter.py:204` |
| `ET_MINER_DISABLE_NCCL` | `1` | C reduce path (staged D2D) and chunk budget | `nccl.py:32` |
| `ET_MINER_ROW_BALANCE` | `rows` / `nnz` | C shard cuts | `csr_bitvec.py:478` |

Route selectors that are not env knobs: `use_gpu`, `n_gpus`, `gpu_resident`,
`bitvecs=`, `prune_equal_support`, `anchor_items`, `streaming`, `chunk_size`,
`memory_budget_gb`, `sparse_from_k`, `sparse`, `n_jobs`, `prune_apriori`, and
the dispatch thresholds `PAIR_COUNT_THRESHOLD` (15M) and
`CANDIDATE_COUNT_THRESHOLD_K3` (500K).

## 6. Code no mining route reaches

Established by grep plus an AST reachability pass seeded from
`et_miner.__all__`, `cli.main`, `mine_two_phase` and module-level code.

| Item | Evidence | Dependants |
|---|---|---|
| `dispatch.dispatch_k3plus`, `count_itemsets_fused_k3plus(_multi_gpu)`, `k3plus_fused.cu` | `dispatch_k3plus` has no caller; the only other caller is the dead `async_pipeline.py` | `bench/repro/d23_24_silent_result_truncation.py` (reproducible through `count_k3plus_fully_fused(..., max_results=)`), `loader.py` K-cap rationale, `test_gpu_correctness.py:193-205`, `test_fused_k3plus_kernel.py`, `test_gpu_device_affinity.py:112-127` |
| `streaming/async_pipeline.py` | imported nowhere (`streaming/__init__.py` imports `son`, `multi_gpu`) | `test_stream_sync.py` tests 1–4 are CuPy probes for its design; test 5 skips on ImportError |
| `gpu/multi_gpu.py` | only `test_apriori.py:405`, `:438` import it | distinct from route E (`streaming/multi_gpu.py`) |
| `gpu/csr_build.py`, `fill_row_ids.cu`, `bootstrap_copy.cu` | imported by the two dead modules and tests only | `test_cuda_csr_build.py`; Rust `generate_random_csr` |
| `dispatch.should_use_multi_gpu`, `should_use_multi_gpu_k3` | tests only / no reference | `test_gpu_dispatch.py::TestShouldUseMultiGpu` (the only k=2 threshold boundary test — re-point at `_resolve_gpus`) |
| `batch._count_batch_prebuilt`, `decode.decode_k3plus_candidates` | no reference / re-export only | — |
| `mining._prune_non_free_flat`, Rust `prune_non_free_flat_compact` | `tests/test_prune_non_free_sorted.py` only | that file also pins `_rows_sorted` and the mask fn's unsorted-input guard: trim, don't delete |
| Rust `bitvec_to_tidsets`, `unique_columns_from_flat`, `generate_random_csr` | no Python caller in any route | Rust unit tests in `groups.rs`, `matrix.rs` |
| `gpu/type_safety.py`, `gpu/utils.py`, `gpu/memory_budget.py` | imported nowhere | — |
| `csr_bitvec.build_bitvecs_from_gpu_arrays`, `PinnedBufferPool` and the `buffer_pool` branch | no route passes `buffer_pool` | — |

## 7. Corrections to the brief's maps

1. A is reached with `n_gpus<=1` (not `==1`) and from the D/E single-chunk
   fall-throughs.
2. B never uses `count_k3plus_from_groups`; its K≥3 is the GPU-resident kernel.
3. D pass 1's B fallback is logged at WARNING and re-mines with the caller's
   `use_gpu` (on the GPU when `use_gpu=True`), not silently on the CPU.
4. F's K=2 chunk-parallel path needs `n_jobs!=1`, pairs > 100 and rows > 10K.
5. `fill_row_ids`/`bootstrap_copy` are reached only by `gpu/csr_build.py`'s own
   generators, not by `async_pipeline.py` or `gpu/multi_gpu.py`.
6. C's K=2 also goes legacy when the pair space is below 64 pairs; sparse
   levels plan with `plan_candidate_chunks`, not `plan_group_chunks`.
7. R1's fallback is silent and R1 has a thread budget; R4's fallback is silent.
8. CLAUDE.md: `bench/selfcheck.py` compiles every registered kernel but
   smoke-launches six.

## 8. Pre-existing defects noticed (not fixed in Phase 0)

- The two SON completeness hazards in §1 (fixed in Phase 1).
- `count_support_gpu_bitvec`'s Rust requirement (fixed in Phase 1).
- Validator: `streaming=True` + `prune_equal_support` is refused on the
  `bitvecs=` branch although C would serve it; `output_dir`/`resume_from_k`
  with `bitvecs`+`use_gpu`+`n_gpus>1` pass validation and are dropped (strict
  xfail); `profile` is over-refused for `bitvecs`+`use_gpu`+`n_gpus>1`.
- Silently ignored parameters: `sparse_from_k` on B/D/E/F, memory guards off A,
  `level_callback` on D/E, `progress_callback` off D/E, `use_generator_pruning`
  on A/B/C, `n_gpus` on F, `use_gpu` on E, `memory_budget_gb` with
  `bitvecs`+`streaming`.
- `test_smoke_correctness.py::*::test_support_001` cannot terminate (above).
