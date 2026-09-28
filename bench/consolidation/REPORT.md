# GPU-layer consolidation — report

Branch `refactor/kernel-consolidation` (local only, not pushed), bundle
`~/et-miner-consolidation.bundle` (`main..HEAD`). Evidence:
`bench/results/2026-09-27-consolidation/` (Phase 2 campaign, `FINDINGS.md`),
`bench/results/2026-09-28-consolidation-verify/` (Phase 4 re-run). Box: 2× RTX
3060 12 GB (sm_86), one of which fell off the bus mid-campaign; every number
below is from device 0.

## Decisions

Phase 2 cells are median [min, max] seconds over 3 reps (1 where marked);
"now" is the Phase 4 re-run of the consolidated tree (59 config-reps at `840f3ef`, every regime's signature identical to Phase 2).

| DP | Decision | Deciding numbers (Phase 2) | Now |
|---|---|---|---|
| DP1 in-core miner | Row-split (C) is the one in-core GPU miner; the single-GPU bitvec (A) and GPU-resident (B) miners are gone. A's one win (pair counts beyond one dense chunk on one GPU) is kept as a dispatch inside C: that level is counted fused on the tiled kernel. | C won oom2 8.22 vs A 10.34, dsl 24.19 vs B 26.98, or002 0.35 vs B 2.66; A won sk2ml2 56.21 vs C 453.57; B won nothing | C at its default dispatch: sk2ml2 44.21 [43.27, 44.68]; sk2ml3 711.57 (1 rep; A had 734.66); oom2 7.76; dsl 23.65; or002 0.18 |
| DP2 K=2 | Tiled; per-candidate only for pair spaces chunked across GPUs | oom2 K=2 7.64 vs 82.49 (C1), sk2ml2 47.21 vs 453.16 (A1) | K=2 level: sk2ml2 41.27 (fused tiled, one GPU), oom2 7.17 (tiled dense) |
| DP3 K≥3 | Per prefix group at the measured crossover (rule 4): tiled from 120 pairs at K=3 down to 23 at K≥8, per-candidate below; separate candidate spaces | dsl per-candidate 14.63 vs tiled 76.39 (C1); or002 tiled 1.26 vs 2.61 (A1); supplementary oom2 to K=3 tiled 93.08 vs 492.21 (C1); kernel sweep: ratio independent of rows, parity 120/91/66/45/23 pairs at K=3/4/5/6/8 | default vs pinned per-candidate / tiled: dsl 23.65 vs 24.36 / 115.35; or002 0.18 vs 0.63 / 0.19; oom2 to K=3 36.83 vs — / 36.99 (K=3 level 28.96, was 93.08) |
| DP4 fallback | Per-candidate kernel kept by role (small groups; oversize groups across GPUs); small-group routing kept; fragmentation guard gone | C1-shared won oom2 8.22 vs 83.09; C1-legacy won dsl 24.19 vs 86.00; tiling small groups too won nothing | sk2ml3 K=3: 11 groups (6.37B candidates) fused, the rest tiled dense in 16 chunks: 666.08 for the level |
| DP5 layout | Dense only; `sparse_from_k` raises | dense won dsl (every sparse layout out of memory); sparse's best gap 0.26 s (or002) | — |
| DP6 multi-GPU | **Open**: no data (device lost). A/B split fan-outs left with A and B; `bitvecs=` on several GPUs uses C's row split | C2 one rep on 5 regimes; splits none | — |
| DP7 out-of-core | SON stays; pass 1 on C per chunk, pass 2 on the batched itemset kernel, on both SON paths | D1-resident deepk 3.69 vs D1-gpu 22.64 / D1-cpu 12.21; dsl 194.75 (1 rep) vs >600 | SON on C + batched pass 2: deepk 3.49 [3.37, 4.01]; dsl 191.50 (1 rep) |
| DP8 CPU | Unchanged: Polars and sparse+Rust both kept, `sparse=None` already dispatches to the winner | Polars deepk 2.64 vs 5.46, skew 3.52 vs 6.71; sparse+Rust or003 39.81 vs 44.77 | control, path unchanged: F-auto deepk 2.66, or003 39.80 |
| DP9 host roles | R2, R4 kept; the Apriori group prune and R3 gone (`prune_apriori` raises) | per call Rust 2–27× the fallback; prune vs no prune a tie (≤ 0.31 s); supplementary oom2 to K=3 101.25 with vs 37.62 without | — |
| DP10 A/B arms | **Open**: one rep each, baseline missing. Nothing removed | sk2ml2 C2 filter compact 28.45 / cupy 22.94 / cpu 24.94 (1 rep) | — |

## Lines changed per area (`git diff --numstat main..HEAD`)

| Area | Added | Removed | Net |
|---|---|---|---|
| CUDA sources (`gpu/kernels/_src/*.cu`) | 1 | 643 | −642 |
| `gpu/` Python | 560 | 5,465 | −4,905 |
| `streaming/` | 129 | 1,109 | −980 |
| `core/` | 87 | 248 | −161 |
| tests | 882 | 3,918 | −3,036 |
| `rust_ext/` | 0 | 568 | −568 |
| bench harness and docs | 1,712 | 393 | +1,319 |
| bench results (campaign data) | 1,736 | 0 | +1,736 |

Kernels: 19 registered entry points in 15 `.cu` files on `main`, 7 in 6 now
(`count_pairs_k2_dense`, `count_k3plus_dense`, `count_shared_tiled_dense`,
`count_shared_tiled_fused`, `count_itemsets_batch`, `compact_threshold`,
`csr_to_bitvec`), plus the popcount ElementwiseKernel. In-core GPU miners:
three, now one.

## Public API and knob removals

Every one raises `ValueError` naming its replacement (listed with measured
speedups under `### Removed` in `CHANGELOG.md`):

- `apriori(gpu_resident=True)`, `apriori_streaming(gpu_resident=True)`.
- `apriori(sparse_from_k=...)`, `mine_two_phase(sparse_from_k=...)` (any value
  but `None`; `mine_two_phase`'s default changed from `"auto"` to `None`).
- `apriori(prune_apriori=...)` (any value; the default is now `None`).
- `ET_MINER_KERNEL_VARIANT` (any value).
- `ET_MINER_TILED_MIN_GROUP_PAIRS` changed meaning: unset now means the
  measured crossover per K instead of a flat 64; a value pins every level.

Exports removed from `et_miner.gpu.kernels`: `count_pairs_fused_k2(_multi_gpu)`,
`count_k3plus_fully_fused(_multi_gpu)`, `count_pairs_fused_k2_gpu_resident(_multi_gpu)`,
`count_k3plus_gpu_resident(_multi_gpu)`, `build_prefix_groups_gpu`,
`build_k3plus_groups`, `count_k3plus_shared_fused`, `count_pairs_k2_shared_fused`,
`CANDS_PER_BLOCK`, `count_csr_range`, `count_csr_gather`, `write_csr_gather`;
renamed: `count_pairs_k2_allcounts` → `count_pairs_k2_per_candidate`,
`count_k3plus_allcounts` → `count_k3plus_per_candidate` (no `variant=`);
added: `count_tiled_fused`, `k2_groups`, `select_k3plus_groups`.
`count_itemsets_cuda` lost `use_batch`. `K3PlusGroups` lost `suffix_src_rows`.

## Rust, per role

| Role | Verdict | Evidence |
|---|---|---|
| R1 CPU K>2 counting (`count_itemsets_simd`) | stays | sparse CPU K≥3 levels 4–36× slower without it (F-sparse vs F-sparse-norust); DP8 keeps the sparse route |
| R2 prefix-group build (`build_k3plus_groups_from_flat`) | stays | per call 2.1–13× faster than the numpy fallback |
| R3 Apriori group prune (`prune_groups_apriori`) | call sites removed; the function stays in `rust_ext` | the role disappeared with the prune (DP9) |
| R4 free-set prune (`prune_non_free_flat`) | stays | per call 6–27× faster than the Python fallback |

R2 is not the only reason Rust survives (R1 and R4 are measured wins too), so
no `refactor/drop-rust` branch was prepared.

## Phase 4 verification

| Check | Result |
|---|---|
| `uv run ruff check src tests` | clean |
| `uv run pytest -q -m "not slow"` | 668 passed, 18 skipped (2-GPU tests, Online Retail absent where marked), `test_support_001` ×2 deselected (see findings) |
| tier gate `tests/test_tier_equivalence.py` | 10 passed (5 CPU, 5 one-GPU legs, each pin proven by call spies), 3 skipped (2-GPU legs) |
| `uv run pytest -q -m gpu` incl. slow | 186 passed, 18 skipped (2-GPU tests), the four infeasible `test_support_001`/`0001` deselected (finding 10) |
| `bench/selfcheck.py` | READY: 7 kernels compiled and each launched with a known answer on the one device |
| reduced matrix re-run | 59 config-reps, all ok, every regime's signature identical to Phase 2 (table below) |
| grep for removed symbols | only the removal checks themselves, the Phase 2 campaign record (matrix, report renderer), CHANGELOG, and CLAUDE.md's intrinsic list (finding 6) |
| independent review | adversarial verifier, CPU emulator of the kernels (676/676 tests on it); findings below |

Re-run (`bench/results/2026-09-28-consolidation-verify/`) against Phase 2, wall
s, median [min, max]:

| Config | Now | Phase 2 twin | Old A/B best |
|---|---|---|---|
| smoke C1 | 0.04 [0.04, 0.04] | 0.04 (C1-tiny0) | 0.03 (A1-shared) |
| deepk C1 | 0.46 [0.45, 0.46] | 0.43 (C1-legacy) | 0.48 (B1) |
| skew C1 | 0.61 [0.60, 0.64] | 0.59 (C1-legacy) | 0.66 (A1-shared) |
| oom2 C1 | 7.76 [7.71, 7.82] | 8.22 (C1-shared) | 10.34 (A1-shared) |
| sk2ml2 C1 | 44.21 [43.27, 44.68] | 453.57 (C1-shared) | 56.21 (A1-shared) |
| sk2ml3 C1 (1 rep) | 711.57 | not run (gate) | 734.66 (A1-shared) |
| oom2ml3 C1 | 36.83 [36.71, 37.00] | 101.25 (C1-shared, supplementary) | 41.28 (A1-shared) |
| dsl C1 | 23.65 [23.63, 24.47] | 24.19 (C1-legacy) | 26.98 (B1) |
| dsl C1 per-candidate pinned | 24.36 [24.16, 24.61] | 24.19 (C1-legacy) | |
| dsl C1 tiled pinned | 115.35 [115.30, 115.70] | 115.33 (C1-tiny0) | |
| wide C1 | 0.16 [0.15, 0.17] | 0.16 (C1-shared) | 0.16 (A1-shared) |
| or005 C1 | 0.07 [0.07, 0.08] | 0.07 (C1-tiny0) | 0.13 (A1-shared) |
| or003 C1 | 0.10 [0.10, 0.10] | 0.11 (C1-tiny0) | 0.47 (A1-shared) |
| or002 C1 | 0.18 [0.17, 0.18] | 0.35 (C1-tiny0) | 2.66 (B1) |
| or002 C1 per-candidate pinned | 0.63 [0.62, 0.66] | 0.82 (C1-legacy) | |
| or002 C1 tiled pinned | 0.19 [0.19, 0.20] | 0.35 (C1-tiny0) | |
| deepk D1 (SON, 4 chunks) | 3.49 [3.37, 4.01] | 3.69 (D1-resident) | |
| dsl D1 (SON, 4 chunks, 1 rep) | 191.50 | 194.75 (D1-resident) | |
| deepk F-auto (CPU control) | 2.66 [2.61, 2.96] | 2.61 | |
| or003 F-auto (CPU control) | 39.80 [38.97, 42.97] | 39.43 | |

Nothing is slower than its Phase 2 twin beyond noise (the largest gap is
deepk C1, +0.03 s); the CPU controls' slow reps overlapped the review's own
CPU-bound test runs.

After the review fixes (`2cc5d73`), rep 0 of the same matrix without
`stress_k2` to K=3 (`bench/results/2026-09-28-consolidation-verify-postfix/`,
20 configs) gave the same signatures in all 11 regimes and the same times
within noise: sk2ml2 43.22, dsl 23.58, oom2 to K=3 36.72, SON dsl 184.57.

## Open risks

- **Multi-GPU is unverified.** GPU 1 was lost at 20:38 UTC on the first day;
  DP6 and DP10 have no answer, and every 2-GPU test skips here: the three
  2-GPU tier legs, `bitvecs=` sharding across devices, the device-affinity
  tests, multi-GPU SON. The multi-shard reduce itself is tested on one device
  (two shards on device 0, with and without forced chunks).
- **sm_90 is untested**, and so are the RTX 3090s the brief assumed: every
  number is from an RTX 3060 12 GB (sm_86).
- **Regimes not covered:** AlphaFold scale (77M rows, K up to 22), cards
  larger than 12 GB, anything multi-GPU at scale; `stress_k2` to K=3 on the
  consolidated tree ran once.
- **Kept under rule 4:** the per-candidate kernel for small prefix groups (and
  oversize groups across GPUs), the fused tiled kernel for levels beyond one
  dense chunk on one GPU, and the CPU's `sparse=None` dispatch. The crossover
  is measured at K=3, 4, 5, 6 and 8; K=7 is interpolated and K>8 holds the K=8
  value.
- The one-GPU fused path keeps a level's survivors on the device (as the
  removed single-GPU miner did); a level with a very high pass rate that does
  not fit one dense chunk can run out of device memory.
- The memory guards (`max_ram_gb` 800, `max_vram_gb` 70 by default) now apply
  to multi-GPU row-split runs, which ignored them before; a device pool above
  70 GB (possible on an H200) now raises between levels unless the limit is
  raised.

## Review findings

From the independent adversarial review (a fresh agent that built none of
this, kernels emulated on the CPU while the GPU ran the re-run) and from this
work. "Fixed" items carry a test unless marked.

1. (must-fix, fixed) GPU SON crashed on string items, item ids beyond int32
   and empty baskets, which the old per-level counter mined (introduced by the
   DP7 port; the in-core route's int32 assert, by DP1). Chunks are now mined
   on column indices and mapped back, and the in-core item table widens to
   int64 when an id needs it.
2. (must-fix, fixed, pre-existing) `bitvecs=` accepted a strided view such as
   `bv[::2]`; the K≥2 kernels read it as other memory (right K=1 supports,
   wrong ones above). It now raises, at `apriori()` and in every kernel
   wrapper's guard.
3. (should, fixed) The K=1 fix for the whole-matrix popcount (commit
   `2c54a35`, out of memory on stress_k2 at 12 GB) lived only in the two
   removed miners; the row-split miner now uses `column_popcounts` too, its
   temporary capped at a quarter of the measured headroom (256 MiB at most),
   which the pool-limit OOM regression test needed.
4. (should, fixed, untested here) `bitvecs=` with `n_gpus=1` copied an array
   living on device 1 to device 0; a single shard now stays on the array's
   device.
5. (should, fixed) `count_support_batched(use_gpu=True)` launched an empty grid
   when no row survived its length filter; `count_itemsets_cuda` now returns
   zeros for zero words and rejects empty itemsets.
6. (should, open) `CLAUDE.md`'s intrinsic list names the deleted
   `_src/csr_warp.cu` and `_src/bitvec_extract_tids.cu` (`__ffsll` is no longer
   used). Left alone: the brief allowed only tier-chain edits to CLAUDE.md.
7. (should, fixed) The bench harness dropped removed settings silently
   (`child_run` stopped forwarding `sparse_from_k`, `consolidation_run` ignored
   `gpu_resident`). A config setting a removed key now fails; so does the old
   `bench/baseline/perf_baseline.py` config `deepk-gpu2-sparse`, which would
   otherwise measure the dense route under a sparse id. `bench/README.md`
   states that the Phase 2 matrix runs only at `e4bb3ae`.
8. (should, open, pre-existing) Empty baskets that reach min_count crash the
   in-core GPU route (`_build_csr_from_transactions` makes a `None` item).
9. (should, open, pre-existing) `anchor_items` with no frequent anchor returns
   the full unanchored lattice.
10. (should, open, pre-existing) `test_smoke_correctness.py::*::test_support_001`
    (not slow) and `::test_support_0001` (slow) cannot finish with the Online
    Retail dataset present; `test_support_0001` exhausted the 31 GB host and
    was killed. Deselected in every run here.
11. (nice, fixed) Docs still describing deleted code: README's "zero PCIe
    transfers", the route validator's docstring, comments in `gpu/mining.py`,
    `synthetic.py`, `csr_bitvec.py`, `row_split.py`, two test docstrings, and
    the CHANGELOG's `suffix_src_rows` line.
12. (nice, fixed) `mine_two_phase(sparse_from_k=...)` created its phase-1
    directory before raising.
13. (nice, fixed) `ET_MINER_KERNEL_VARIANT` was rejected only inside the miner
    (after the CSR build); it now raises at `apriori()` entry. A negative
    `ET_MINER_TILED_MIN_GROUP_PAIRS` raises. The one-GPU oversize path uses the
    fused tiled kernel whatever the pin; `_env.py` says so.
14. (nice, fixed, no test) `n_gpus<=0` with `use_gpu=True` crashed in the
    thread pool; GPU routes clamp it to 1, as the removed single-GPU miner did.
15. (nice, fixed) The memory guard fired after the last level (discarding a
    complete lattice) and its VRAM message recommended `output_dir`.
16. (nice, fixed, no test) A split level kept the whole level's host group
    arrays next to the two copies; the whole level is released once split.
17. (nice, open) Removed or renamed parameters of the exported kernel wrappers
    (`use_batch`, `with_src_rows`, `variant`, the `*_allcounts` names) fail
    with TypeError/ImportError, not a ValueError naming the replacement; they
    are internal wrappers, listed in the CHANGELOG.
18. (nice, open, pre-existing) Parameters still accepted and ignored:
    `level_callback` on streaming, `n_gpus` on the CPU route, `use_gpu=False`
    and `batch_size`/`sparse`/`n_jobs` on multi-GPU SON, `memory_budget_gb`
    with `bitvecs=`+`streaming`, `streaming`/`chunk_size` with `bitvecs=`,
    `resume_from_k` without `output_dir`.
19. (nice, open, pre-existing) `bench/repro` scripts such as
    `d06_10_routing_params_dropped.py` exit 0 while the defect is live, the
    inverse of `bench/repro/README.md`.
20. (nice, open, pre-existing) Modules nothing imports: `gpu/type_safety.py`,
    `gpu/utils.py`, `gpu/memory_budget.py`; `csr_bitvec`'s `buffer_pool` path
    and `build_bitvecs_from_gpu_arrays`. `cargo clippy -- -D warnings` fails
    with 6 errors present on `main`. `bench/runner.py` re-runs non-ok configs
    when resuming.

Claims the reviewer tried and failed to break: exact output on every kernel
setting and output mode (complete, free-sets, anchors, both) on 1 and 2
emulated devices, with and without Rust; `select_k3plus_groups` and
`plan_group_chunks` under randomised trials; `shard_prebuilt_bitvecs` for 1 to
1,000 rows, 1 to 3 devices; the level-end sort with several candidate spaces;
output_dir/resume with `bitvecs=`; profile and level_callback; every removed
parameter raising; an unweakened tier chain with every surviving kernel pinned.
