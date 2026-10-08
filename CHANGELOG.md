# Changelog

All notable changes to ET-Miner are recorded here. Versions follow
[Semantic Versioning](https://semver.org/), with the caveat below.

> **Pre-1.0 caveat.** At 0.x a minor bump may change mined output. Every such
> change is listed under **Behaviour changes** with a before/after and a
> migration note, because a frequent-itemset miner's output *is* its API:
> a number that moves silently invalidates whatever was published from it.

---

## [Unreleased] — 0.2.0

### CPU route: one array miner

- **The CPU route of `apriori()` mines every level from one CSR of the
  frequent items** (`core/cpu_miner.py`). It replaces the Polars boolean-matrix
  counter and the per-candidate sparse counter on that route. Polars stays the
  input layer and reads the transactions once: the row count, the K=1 counts,
  and a CSR of each row's frequent items, built in 500K-entry chunks. Then:
  - **K=2** from one Gram matrix `M.T @ M` (scipy, int32), split into column
    blocks under 256 MB, or from column bitvectors (popcount of the AND) when
    candidate pairs × words ≤ 3 × pair occurrences. Measured on the campaign
    workloads, bitvectors take 0.61–0.69× of the Gram's time at work ratios
    1.7–2.7 and 6.3–38.6× at 7.1–175. Phase 0 counted K=2 one candidate at a
    time (22–211 s).
  - **K≥3 generation** on lexsorted int32 arrays: prefix runs joined in chunks
    of 2M candidates, then the subset test through a pair mask (K=3) or packed
    keys and a binary search. Phase 0's generation was quadratic (or002 K=6:
    354 s).
  - **K≥3 counting per prefix group**: the prefix AND once, then either pair
    popcounts on its non-zero words or a Gram matrix of the suffix columns over
    the prefix's rows. Projection takes a group from 40 suffixes (≤ 2,048
    bitvector words) or 80 (more words), measured on real prefix groups by
    `bench/cpu/l3_crossover.py`. Rows with fewer than k items are dropped once
    the remaining rows are at most half of the current ones.
  - **Free-set test and count inference** (`prune_equal_support`,
    `use_generator_pruning`) on the level arrays; one Polars gather emits the
    result.
  - **`n_jobs > 1`** runs the K=2 Gram blocks (from 5M pair occurrences), the
    bitvector build and the heavy prefix groups (from 2M pairs × words) on a
    thread pool; numpy and scipy release the GIL there. `n_jobs=1` stays on the
    calling thread. In Phase 0, `n_jobs=4` made wide, or005, or003 and or0001k2
    1.4–1.8× slower.
- Pre-registered campaign (`bench/cpu/PROTOCOL.md`, amendment 2;
  `bench/results/2026-10-07-cpu-phase1/compare.md`). Hardware: 4 vCPU Intel Xeon
  @ 2.10 GHz, 15 GB. Figures are wall seconds of the mining call in a fresh
  process (data already loaded; efficient-apriori's conversion to tuples not
  counted), median of 3 runs (¹ Phase 0 one run). Default arm (`sparse=None`),
  1 and 4 threads; efficient-apriori 2.0.6 (`itemsets_from_transactions`) from
  the same campaign:

  | Workload | efficient-apriori | Phase 0, 1 / 4 threads | Phase 1, 1 / 4 threads |
  |---|---|---|---|
  | smoke (60K rows, s=0.01) | 0.56 | 0.59 / 0.33 | 0.12 / 0.12 |
  | deep_k (1M, s=0.02) | 95.22 | 13.39 / 4.38 | 2.74 / 2.69 |
  | skewed_rows (1M, s=0.02) | 161.48 | 16.90 / 5.93 | 3.84 / 2.91 |
  | wide_vocab (100K, s=0.004) | 60.05 | 99.60¹ / 142.18¹ | 1.27 / 0.55 |
  | Online Retail II, s=0.005 | 19.19 | 34.02 / 54.93 | 0.72 / 0.40 |
  | Online Retail II, s=0.003 | 51.31 | 83.73¹ / 123.16¹ | 2.07 / 1.24 |
  | Online Retail II, s=0.002 | 170.24 | 585.66¹ / 594.76¹ | 7.27 / 5.58 |
  | Online Retail II, s=0.0001, K≤2 | 51.63 | 281.45¹ / 515.43¹ | 1.87 / 0.96 |

  Every `sparse` arm at both thread counts beats efficient-apriori on every
  workload under the protocol's rule (median ≥ 10 % lower, ranges apart, and
  the gap at least 1 s, or 10 % of efficient-apriori's median below 10 s).
- **Peak RSS** (`ru_maxrss`) falls to 0.18–0.36× of Phase 0 on wide_vocab and
  Online Retail II (or0001k2: 4,985 → 1,041 MB at 1 thread) and to 0.73–0.85× on
  smoke. On deep_k and skewed_rows at 1 thread it grows 1.17–1.18× (299 → 350,
  331 → 389 MB), within the 1.25× bound the protocol allows.
- **No mined output changes**: each workload keeps one itemset signature across
  every arm of both campaigns, and the tier-equivalence chain passes.
- **`sparse=` is deprecated on the CPU route and in SON's CPU passes.** A
  non-`None` value there warns (`DeprecationWarning`) and is ignored, because
  every level is counted by the array miner. The Rust extension is no longer
  used by either; `count_support_batched(sparse=True)` and the GPU route's
  host steps still use it.
- **SON's CPU passes (`streaming=True` without `use_gpu`) run on the array
  miner** (`streaming/son.py`). Pass 1 builds each chunk's CSR at the local
  threshold and mines it with the CPU route's levels; the union of the local
  results is one int32 array per length, deduplicated chunk by chunk. Pass 2
  maps each chunk onto the candidate items and counts the candidates with
  `cpu_miner.count_itemsets`: K=1 by a bincount, K=2 on bitvectors or from the
  Gram (the in-core rule), K≥3 per candidate (AND of the k bitvectors) when
  the row space has at most 512 words or the level averages fewer than 2.5
  candidates per prefix group, by prefix group otherwise
  (`bench/cpu/PROTOCOL.md` Amendments 3–4). The Polars boolean matrix,
  `_generate_candidates` and `count_support_batched` are gone from SON's CPU
  passes; the GPU passes keep theirs. `batch_size` is no longer read by any
  route. Output is unchanged. Pass 1's `progress_callback` candidate count is
  now an upper bound until the end of the pass (duplicates across chunks are
  removed once the pending rows outgrow `UNION_PENDING_BYTES` and the union);
  the profile's `n_candidates` is exact.
  Measured with 4 chunks (`bench/results/2026-10-08-son-s1/`, median of 3,
  12-thread Ryzen 5 4600G; the old SON's single S0 value): deep_k 31.2 →
  4.7 s at 1 thread and 15.1 → 4.2 s at 4; skewed_rows 157.0 → 31.2 s and
  79.0 → 15.1 s; wide_vocab 267.9 → 1.2 s and 235.9 → 0.6 s; Online Retail II
  at 0.005 and at 0.0001 with max_length 2 from over 600 s (the cap) to
  7.9–13.9 s. Peak RSS falls to 0.13–0.87× (wide_vocab at 4 threads: 1,732 →
  226 MB).
- **SON's CPU pass 2 counts only what pass 1 cannot settle**
  (`streaming/son.py` `_bound`). Pass 1 mines each chunk completely at its
  local min_count, so a chunk that did not emit an itemset holds it in at most
  min_count − 1 rows. The union now sums each candidate's local counts and the
  slack of the chunks that emitted it. A candidate whose upper bound misses the
  global min_count is dropped; one with no slack left keeps pass 1's exact
  count; pass 2 counts the rest and reads no chunk when none is left. The
  union's merges sort packed keys (`cpu_miner.sum_rows`, replacing
  `unique_rows`, whose `np.unique` hashed int64 keys). Output is unchanged.
  The profile's pass 1 reports `n_bounded` and `n_exact`, pass 2's `n_counted`
  is the number it counted, and pass 2 reports no `progress_callback` chunk
  when it reads none.
  Measured with 4 chunks against main at `e892d8a` on the same box
  (`bench/results/2026-10-08-son-bound/`, median of 3, `bench/cpu/PROTOCOL.md`
  Amendment 5): deep_k 4.7 → 2.6 s at 1 thread and 4.2 → 2.3 s at 4;
  skewed_rows 31.2 → 22.3 s and 15.0 → 11.0 s; Online Retail II at 0.005
  13.9 → 8.1 s and 13.4 → 7.8 s, at 0.0001 with max_length 2 7.9 → 2.8 s and
  8.0 → 2.9 s (peak RSS 768 → 579 and 854 → 672 MB); smoke and wide_vocab
  within 0.02 s.
- **The CSR build counts and maps integer item ids without hashing**
  (`core/cpu_miner.py` `_int_ids`). For a list column of integer ids
  (Int8–Int64, UInt8–UInt32) spanning at most 2²² values, the K=1 counts are a
  bincount over the id range and each row chunk's exploded values reach their
  column through a lookup table, in place of Polars' `group_by` and
  `replace_strict`. Null lists and null items are dropped as before. Other
  columns (strings, categoricals, floats, UInt64, wider spans) still go
  through Polars. In both paths a row's start among the kept entries comes
  from a running count instead of per-entry row ids. The in-core route and
  both SON passes use it. Output is unchanged, and pyarrow is not loaded.
  Measured against main at `c973ed3` on the same box
  (`bench/results/2026-10-08-input-layer/`, median of 3, `bench/cpu/PROTOCOL.md`
  Amendments 6–7): deep_sparse_large (20M rows, 247M list entries, s=0.015,
  max_length 2) in core 17.5 → 11.6 s at 1 thread and 12.5 → 8.7 s at 4, with
  SON (4 chunks) 16.5 → 11.7 s and 11.1 → 8.5 s; deep_k and skewed_rows
  0.13–0.64 s faster (in core at 1 thread 2.05 → 1.76 s and 2.63 → 2.33 s);
  wide_vocab, Online Retail II and smoke within 0.10 s. Peak RSS 0.92–1.03×.
- **The tier-equivalence chain's Tier 2 leg runs the all-Rust miner**
  (`apriori_from_csr`) on the smoke CSR, the oracle's own input. It ran
  `apriori(sparse=True)`, which now reaches the array miner, so the Rust miner
  had no leg. It skips without the Rust extension.
- `ProfilingSession.record_phase(name, duration_ms, **extra)` records a phase
  timed elsewhere: the CPU route interleaves candidate generation and counting
  per chunk. `apriori(profile=True)` keeps its phase names.
- `bench/cpu/`: the CPU-tier campaign (protocol, matrix, timed child, per-level
  split, stakes, crossover measurements, report and compare scripts).

### Polars 2.0

- **Runtime floor raised to `polars>=2.0.0`** (was `>=1.39.0`). Polars 2.0
  makes the streaming engine the default for `LazyFrame.collect()`. The four
  calls that relied on the old in-memory default now name it
  (`engine="in-memory"`): the `list.contains` matrix builds in
  `build_boolean_matrix` and SON pass 2, the K=2 and prefix-group cross joins
  in `core/candidates.py`, and the remote resume read in `gpu/row_split.py`.
  Every other `collect` already named its engine. No mined output changes.
- The old reason for keeping `list.contains` off the streaming engine (a ~2 %
  undercount) does not reproduce: `bench/cpu/list_contains_check.py` found
  exact column sums on both engines, on 1.43.2 and 2.0.0, for 600 items on
  smoke, deep_k, skewed_rows and stress_k2 (up to 2M rows). In-memory stays
  because it is faster there (skewed_rows 12.7 s vs 18.9 s streaming on 2.0.0).
- `explode()` of an empty list now yields no row instead of a null row. On 1.x
  an empty transaction added a null "item" to the K=1 counts, whose
  `list.contains(None)` column summed to 0 and was dropped at K=1; on 2.0 it is
  never counted. Same output either way.

### Candidate pruning on the GPU (supersedes the DP9 removal)

- **`prune_apriori` is back, default `True`, as a device-side subset test.**
  The K≥3 counting kernels (per-candidate, tiled dense and fused, sparse CSR)
  get an index of the previous level and leave a candidate with a missing
  (k−1)-subset uncounted (per candidate, or per 32×32 tile-pair in the tiled
  kernels); candidate indices never change, so chunking, the multi-GPU reduce
  and the decode are untouched. `False` counts every generated candidate and is
  accepted only by the row-split miner. This supersedes the consolidation's DP9
  removal of the host-side, suffix-granular prune: that prune reached 4–18 % of
  the prunable candidates and cost 63.6 s on oom_regression to K=3, where the
  device-side test (pre-registered campaign, `bench/pruning/REPORT.md`, 2× RTX
  A4000) takes oom_regression to K=3 from 25.76 to 7.18 s on one GPU (17.21 →
  8.16 s on two) and stress_k2 to K=3 from 496.27 to 52.77 s; every other
  regime ties.
- **`use_generator_pruning` now works on the row-split miner** (it needs
  `prune_apriori`): a candidate with a non-free (k−1)-subset gets the minimum of
  its subset counts instead of a count, written by one GPU so the reduce stays
  exact. deep_sparse_large: 24.99 → 20.16 s on one GPU, 20.85 → 18.74 s on two.
  Default unchanged (`False`).
- **Free-set runs** test against the free level itself: a candidate with a
  subset outside it is not free, survivors with one are dropped on the host
  before the free-set test, and the complete generated level is no longer
  kept. A run resumed from a free-set level is now exact (it could under-prune
  before; still so with `prune_apriori=False`).
- No mined output changes: every regime of the campaign has one signature
  across all arms, equal to its earlier signature. Its first run caught one
  (fixed before the deciding run): free-sets on the tiled kernels emitted 442
  non-free itemsets of 204,972 on deep_sparse_large.
- Tier chain: new legs without the subset test, with count inference (each
  kernel pinned, forced chunks, ESCO, two GPUs); free-set runs checked against
  free-sets derived from efficient-apriori's lattice.

### Candidate-waste measurement (GPU candidate pruning, phase 0)

- Added `bench/runner.py --mode waste`: the row-split miner on one GPU,
  complete lattice and free-sets, with each level's time split into phases
  (`bench/level_split.py`) and the lattice dumped for
  `bench/candidate_waste.py`, an offline classifier that replays the route's
  candidate generation and labels every candidate prunable (an infrequent
  subset), inferable (a non-free subset) or countable. No miner code changed.
- Findings in `bench/results/2026-10-05-candidate-waste/FINDINGS.md`: 34–99 %
  of the K≥3 candidates the GPU counts have an infrequent subset; on the K=3
  explosions (oom_regression and stress_k2 to K=3) about 97 % of the tiled
  kernel's tile-pairs hold nothing else, while the removed suffix-granular
  prune reached 4–18 % of them.
- The runner's environment capture also records `uv pip freeze`; under uv,
  `python -m pip freeze` printed nothing.

### ESCO restored (after kernel consolidation)

- Restored the opt-in GPU dense→sparse CSR crossover via `sparse_from_k="auto"`
  or an integer K, keeping dense mining as the default. This supersedes the
  sparse-layout removal recorded under GPU-layer consolidation below.
- Restored the warp intersection kernels, source-row mapping, ownership checks,
  density tests and ESCO oracle-equivalence legs on one and two GPUs. The
  current dense kernels, filtering and reduce paths serve both layouts.
- Added `esco` / `esco-retail` benchmark modes comparing ESCO with the dense
  dispatcher and both pinned dense kernels. Retail runs use supports 0.0001
  and 0.00005 at explicit max_length 2/3/4; CPU and efficient-apriori K=2
  controls can run without CUDA using `--cpu-only`. See `bench/ESCO.md`.

Remediation of the 62 defects recorded in `BUGS_FOUND.md` (an adversarial
four-reviewer review of `22cb1dc`), plus 19 further defects found while
planning it. Landing across PRs 0–11; this section is filled in as they merge.

### Behaviour changes

Six queued fixes change mined output. Before any of them land,
`bench/baseline/behaviour-change-impact.md` records which already-published
figures move — measured, per artifact, not assumed.

| Defect | Change | Who is affected |
|---|---|---|
| **#11** | `_min_count` becomes `ceil(Fraction(str(s)) * N)` instead of `ceil(fl64(s) * N)`, at all three sites (`core/result.py`, `rust_ext`, `synthetic.py`) | any run whose `(min_support, n_rows)` pair shifts — check with `bench/baseline/min_count_impact.py` |
| **#12** | `compute_self_sufficiency` aggregates `min`, not `max`, over the (K-1)-subsets, and the output column is renamed `max_k_minus1_support` → `min_k_minus1_support` | **every row's ratio changes**; any published self-sufficiency value is invalidated, not merely shifted. Readers of the output frame must rename the column. |
| **#13** | `count_support_batched`'s length filter uses the batch **minimum**, not `itemsets[0]` | callers passing mixed-length batches; `apriori()` always passes a uniform level and is unaffected |
| **#22** | `anchor_items` becomes an output selector; the generating level is no longer filtered | `mine_two_phase` and `apriori(anchor_items=...)` return the itemsets they always advertised — up to 99.3% more — and Phase 2 gets slower |
| **#24** | any GPU result-buffer overflow raises instead of silently truncating | runs that were completing with a non-deterministic ≤5% loss now fail loudly and say which knob to raise |
| **#51** | `--min-lift` filters at `>=` its value, including the default `1.0` | CLI users relying on the accidental no-op at 1.0 |

### Migration

- **Re-check any published figure** against
  `uv run python bench/baseline/min_count_impact.py --exhaustive --n-rows <N>`.
  The base214m run (N = 76,890,945) is cleared: 0 of 189,999 thresholds shift.
  That is a property of *that row count* — the same sweep finds 2,457 shifting
  thresholds at N = 1,000,000.
- **Self-sufficiency values must be recomputed.** The old `max`-based ratio and
  the new `min`-based one agree on no row in practice.
- **Phase 2 of `mine_two_phase` gets slower** and its default `phase2_support`
  is raised. That is the fix working: the old speed was purchased by discarding
  up to 99.3% of the correct answer. `bench/baseline/perf-baseline.json` holds
  the before.

### Fixed

- **Two-GPU runs on a box whose PCIe P2P drops device-to-device writes.**
  Such a box (a Ryzen AM4 host with two RTX A4000s behind the CPU's host
  bridge, `bench/results/2026-09-28-consolidation-2gpu/nccl-hang/`) reports
  peer access and then loses the copies: `cudaMemcpy`, `cudaMemcpyPeer` and
  CuPy assignment from GPU 1 to GPU 0 return success with the destination
  untouched (small copies land, copies of 64 KiB and more drop), so the
  staged D2D reduce (the NCCL-absent fallback) summed only GPU 0's shard —
  `smoke` on two GPUs returned 342 itemsets with roughly half their counts
  instead of 694 — `bitvecs=` sharded across two GPUs mined a shard of
  stale memory, and NCCL's P2P transport hung about one run in three at
  the first collective. No probe can certify such a pair, so
  `et_miner.gpu.nccl.copy_between_devices` now stages every cross-device
  copy through host memory (the staged reduce and the `bitvecs=` shards go
  through it; `ET_MINER_DIRECT_D2D=1` opts back into direct copies for
  NVLink or a known-good PCIe switch), and NCCL communicators are created
  under `NCCL_P2P_LEVEL=NVL` (P2P over NVLink only) unless the caller set
  `NCCL_P2P_LEVEL` or `NCCL_P2P_DISABLE`; the variable is put back
  afterwards, and it has no effect when another library initialised NCCL
  first, since NCCL reads it once per process. Tests:
  `tests/test_cross_device_copy.py` (real transfers of 8M int32 and more,
  no monkeypatching; the default path never issues a direct copy).

*PR 1 — canonical order and the exact threshold*

- **#1, #2, #3, #20** — itemsets are emitted as **ascending tuples of item ids
  on every route**, and that contract is now written into `apriori()`'s Returns
  block. `build_boolean_matrix`'s column names are zero-padded, which makes
  lexicographic name order equal item-id order and so fixes the CPU route at
  the root rather than re-sorting at emission. Measured: 271/461 emitted
  itemsets were non-ascending, now 0; `generate_rules` was dropping 408 of
  2,256 rules (18.1%) and reporting `lift = 0.0` on 408 more, now 0 and 0.
  `_build_support_lookup` is additionally keyed on the sorted tuple, so it no
  longer depends on its producer's ordering.
- **#11, #14** — `_min_count` is the **exact decimal ceiling**,
  `ceil(Fraction(str(s)) * N)`, at all three sites (`core/result.py`,
  `synthetic.py`, and `exact_min_count` in the Rust extension). The old
  `ceil(fl64(s) * N)` returned 701 where the exact ceiling is 700 at
  `s = 0.07, N = 10000`, dropping the boundary itemset and the entire cone
  above it — 3 itemsets mined against the oracle's 7. All three sites are
  checked against one shared table, `tests/fixtures/min_count_cases.json`,
  read by both the Python tests and a Rust `#[test]`: three implementations
  agreeing with each other is worth nothing when they share a bug, which is
  exactly how this defect survived. The tier gate gains an independent boundary
  case, since it previously derived its oracle threshold from the expression it
  was meant to check.

*PR 2 — sparse/MKL numerics and the Rust boundary*

- **#4** — `_sparse_matmul` widens integer input to **float64**, not float32.
  float32's 24-bit significand made a support count stick at
  2²⁴ = 16,777,216: measured 20,000,000 → 16,777,216, a 16.1% **under**-count
  that silently dropped frequent pairs and cascaded into every higher K. Costs
  2× the value-array bytes inside the matmul.
- **#5** — `panic = "abort"` removed from the release profile, so an FFI panic
  raises a catchable `PanicException` instead of an uncatchable SIGABRT that
  loses every unflushed level of a campaign. All FFI entry points now share one
  contract: non-contiguous numpy input is copied rather than panicking (half of
  them already did this and half did not), and CSR shape problems raise a
  `ValueError` naming the offending value instead of an index-out-of-bounds
  panic from inside a kernel.
- **#16** — importing `core.sparse` no longer overrides the process-global MKL
  thread count. The module is imported lazily from inside
  `count_support_batched`, so this fired mid-run and silently oversubscribed a
  host application's own MKL configuration (measured: a host setting of 2
  became 24). `_restore_mkl_threads` now restores the value it displaced
  rather than re-deriving one from the environment.
- **#17** — MKL discovery globs `libmkl_rt.so*` across candidate directories
  and logs the outcome either way. It previously tested one filename at one
  location and returned silently on a miss, so nothing distinguished "the path
  was already fine" from "found nothing" — and the numerics then ran against
  whichever unpinned system MKL loaded, which is what #4's accuracy depends on.
- **#18** — `n_jobs` is honoured on the Rust path. rayon took every core
  regardless, so `n_jobs=1` — documented as "sequential execution" — silently
  oversubscribed a shared box. The counting kernels now accept a thread budget
  and run on a scoped pool (verified: `n_threads=1` is 5.9× slower than
  unbounded, with identical counts). The dead `_RUST_MIN_ITEMSETS` fossil is
  gone.

*PR 3 — routing parameters*

- **#6, #7, #8, #9, #10, N1** — every parameter/route mismatch now raises an
  explicit `ValueError` naming the parameter and the route, checked **above**
  the routing rather than after it. Previously the caller passed the parameter,
  the route dropped it, and nothing in the return value or the logs said so.
  Measured on a 400-row fixture: `streaming=True` + `prune_equal_support`
  returned the complete 214-itemset lattice where the gated answer is 176;
  `anchor_items` on the CPU route returned 214 rows identical to unanchored;
  `output_dir` on the CPU route wrote no files; `profile=True` on multi-GPU
  streaming returned a bare DataFrame that **unpacks into two Series**, so
  `result, session = apriori(...)` succeeded and handed back a column of
  itemsets and a column of floats. **N1** (not in the reviewed 62) is the same
  shape: `gpu_resident=True` was silently ignored whenever
  `prune_equal_support` was set.

  Where a route *can* honour a parameter it is forwarded instead of rejected:
  `output_dir` / `resume_from_k` on the `bitvecs=` + pruning route, and
  `memory_budget_gb` on multi-GPU streaming (which gained the parameter,
  mirroring `apriori_streaming`).

  `level_callback`'s `n_candidates` is documented as route-dependent — the GPU
  group path over-approximates the subset test where the CPU path does not, so
  the two report different counts for the same input at the same level.

*PR 4 — host counting and rule metrics*

- **#12** — `compute_self_sufficiency` aggregates **`min`** over the
  (K-1)-subsets, not `max`, and the output column is renamed
  `max_k_minus1_support` → **`min_k_minus1_support`**. Under `max`, a maximally
  redundant itemset ({1,2,3} at 0.30 with subsets 0.30/0.90/0.95, item 3 fully
  implied by {1,2}) scored **0.32** and read as "genuine combinatorial signal" —
  exactly backwards, so anyone filtering on the ratio kept the redundancy and
  discarded the signal. It now scores 1.0. Recalibrating a cutoff was not
  available: under `max` the ratio is not monotone in the property described, so
  two equally redundant itemsets scored 0.9375 and 0.3158.
- **#13** — `count_support_batched`'s length filter uses the batch **minimum**
  instead of `len(itemsets[0])`. Whichever itemset happened to be first set the
  row filter for the whole call, so a mixed-length batch counted differently
  when permuted and a caller building the list from a set got a different
  answer per run (measured: `("i_0",)` counted 1 against a true 5, a 75%
  undercount). No in-tree caller is affected — `apriori()` always passes a
  uniform level and both SON pass-2 callers already pass
  `enable_length_filter=False`.
- **#21** — `compute_self_sufficiency` returns its documented empty frame
  instead of raising `TypeError` from inside a log statement. The chunk append
  is now guarded exactly as the sibling `generate_rules_drop1` guards its own.
- **#15**, **#19** — documentation corrected where it described neither
  implementation: the k=2 "O(1) memory" claim (the caller drains the generator
  into a list — 1,242 MB at 6,000 items), and `_prune_groups_apriori`'s
  "kept only if ALL its (k-1)-subsets are in prev_frequent_set", which carried a
  verification badge and was wrong about both the Python and the Rust
  implementation. Both keep a suffix that participates in **at least one** valid
  pair, which over-approximates one-sidedly.

*PR 5 — GPU correctness*

- **#22** — `anchor_items` is an **output selector**, applied only where a level
  is emitted. It used to filter `current_flat`, and that one array then became
  both the subset oracle and the generation base, which is unsound twice over:
  the apriori oracle must test the (k-1)-subsets that *drop* the anchor (they
  are unanchored by construction), and the prefix-join needs the surviving
  family closed under its two prefix-parents. Anchoredness is not
  anti-monotone — that single property is why the same code shape is sound for
  the free-set prune and catastrophic here. K=1 is masked too; it used to be
  exempt. The unsound `_apply_anchor_filter` helper is deleted rather than left
  for reuse.
- **`mine_two_phase`** is re-documented accordingly and its `phase2_support`
  default raised **0.00001 → 0.0005**. Phase 2 mines the full lattice at that
  threshold and post-filters; it does not and cannot prune, so the old default
  was chosen on a false premise. A pruning-preserving variant was sought and
  measured not to exist (hoist+remap loses 129 of 321). Per-anchor conditional
  databases are the one sound shape if candidate reduction is genuinely needed.
- **#23, #24** — any GPU result-buffer overflow **raises**. Four multi-GPU
  kernels clamped with a bare `min(n, gpu_max)` while their single-GPU siblings
  called the helper, and the helper itself tolerated 5% loss with a warning —
  so neither fix works alone. The dropped set is non-deterministic (`atomicAdd`
  append), so those routes disagreed with the row-split path, with the CPU
  tiers, and with themselves run twice. The error names the level, the knob to
  raise and the K to resume from.
- **#25** — a single `MAX_SUPPORTED_K = 62` is enforced host-side from all
  seven K≥3 wrappers; previously only the shared/tiled one checked anything, and
  K=63 silently returned 640 where the true count is 0. **Zero `.cu` edits** —
  widening the device guards buys unreachable capacity and leaves K≥65 corrupt.
- **#28** — a tripped memory guard raises `MemoryError` instead of breaking out
  of the level loop and returning a truncated lattice as if complete (measured:
  249 itemsets against 31,160, a 99.2% loss, with no exception). The VRAM half
  could never fire at all, because the probe's exception was swallowed to 0.0.
- **#32** — `n_gpus` is honoured: every dispatch entry point takes it as a cap,
  so `n_gpus=1` on a two-GPU box stays on one device instead of landing on the
  fan-out path. `gpu/dispatch.py` had no logging at all; it now records the
  resolved device count with the caller's request, which is what makes a
  truncation diagnosable after the fact.

*Council review — defects found in the fixes above*

A four-lens adversarial review of PRs 1–5 blocked unanimously and found eight
defects in the remediation itself, all measured. Fixed here:

- **`validate_csr` had five holes**, one of them a logic bug in the guard added
  by PR 2. `n_rows + 1` wrapped at `usize::MAX`, skipping the guard entirely;
  `indptr` monotonicity was never checked, so the kernels' per-row slice
  panicked on all four guarded entry points; a negative interior `indptr` entry
  wrapped; and the column-range test read `max_idx < 0` over the array's
  **maximum**, so a negative index beside a positive one passed clean.
  **The worst was pre-existing and silent**: `count_itemsets_sparse_raw`
  binary-searches each row and nothing validated that rows are sorted — a row
  stored descending returned **2 against a truth of 3**, a 33% undercount, while
  the SIMD path returned 3. Which number a caller got was decided by
  `hasattr(rust, "count_itemsets_simd")`, so **Tier 2 of the mandated chain
  disagreed with itself depending on how the wheel was built.** All five now
  close in one O(n_rows + nnz) pass.
- **`anchor_items` was accepted and dropped on the `bitvecs=` route** — measured
  210 itemsets returned with 136 unanchored, defect #8's exact failure mode in
  the guard written to close it. The guard now tests whether the call *resolves*
  to row-split rather than which parameters were passed. It refuses rather than
  forwards, so everything that route flushes stays a complete level structurally.
- **`resume_from_k` + `anchor_items` silently lost 93% of the lattice** —
  838 → 57 itemsets. An anchored per-K parquet is a report, not a resumable
  mining state, but resume reloads it as both generation base and subset oracle.
  The invariant is now written down: *a persisted K-level is a valid resume
  artifact iff it is the complete frequent level at that K.* The **read** is
  refused, not the write, so `mine_two_phase` keeps working.
- **`memory_budget_gb` was still dropped below 10M rows** — the override sat
  after the single-chunk shortcut that consumes `chunk_size`. Now resolved
  first, in `son.py`'s order.
- **Two error messages advised remedies this same release forbids** —
  `resume_from_k` and `output_dir` are both refused by the route validator on
  exactly the routes that raise, and `max_results` is not a public parameter.
  Reworded to name only reachable remedies.
- **#12's aggregate change had no test at all.** Reverting `.min()` to `.max()`
  left the whole suite green. Now pinned by `tests/test_self_sufficiency.py`.
- Smaller: the `u32` cast in `exact_min_count` now saturates rather than
  wrapping; a `%s` in a loguru call printed literally; a stale "take max"
  comment; a dead log line; an `int | None` annotation.

*PR 6 — GPU hygiene, device affinity, docs (one schema change: #26)*

- **#26** — the row-split miner returned **three different `itemset` dtypes**.
  Both empty-result paths and the list fallback gave `List(Int64)`; the PyArrow
  fast path — the one that normally runs — gave `List(Int32)`, and the sibling
  `_apriori_from_bitvecs` route built its lookup as int64, so the two GPU routes
  disagreed with each other. Every in-memory return is now `List(Int64)`. The
  flushed parquet deliberately stays `large_list<int32>`: widening it would
  double the itemset bytes of every existing artifact, no reader is
  dtype-sensitive, and widening `core/rules.py`'s join keys would cost ~84 GB on
  a K=7 K-1 frame (an estimate — rows × 6 × 4 B on a ~3.5e9-row K=6 frame — not
  a measurement; the row count behind it is not recorded). A test enumerates every `itemset` reader so the asymmetry
  cannot spread unnoticed.
- **#30, N10, N20 — three device/ownership assumptions, all silent.**
  `_apriori_from_bitvecs` and the row-split density transition both logged
  *"Freed bitvec VRAM"* while freeing nothing, and on the
  `apriori(bitvecs=..., prune_equal_support=True)` route the arrays belong to the
  **caller**, so no reordering could make the claim true. The row-split path also
  mutated a caller-supplied list. Separately, five multi-GPU fan-out wrappers
  hardcoded `if device_id == 0` for the alias device (**N10**), and three more
  sites hardcoded device 0 in the *output* half — two merge/decode tails and
  `build_prefix_groups_gpu`, which allocated on the ambient device rather than
  its input's (**N20**). A caller whose bitvecs live on device 1 hit
  *"the device where the array resides (0) is different from the current device
  (1)"*. Latent on every in-tree route today; PR 10 makes it reachable.
  `count_k3plus_gpu_resident_multi_gpu` now states a co-residency contract and
  raises on a mixed-device call rather than hiding it behind a transfer.
- **#27** — one bare `except Exception: pass` wrapped two unrelated releases, so
  a failure freeing the group arrays silently skipped the CSR shards. Split, and
  logged instead of swallowed.
- **#31** — the dense K≥3 group arrays (tens of GB) had no `finally`, so any
  raise inside the chunked level — the truncation `RuntimeError` most obviously —
  left them resident on every device for the rest of the run.
- **#29** — both row-split callers materialised a `set()` of the previous level
  and handed it to a Rust path that never reads it. Measured cost and the regime
  it depends on are in `gpu/mining.py::_prune_groups_apriori`; an earlier
  revision of this entry quoted "~35 s per level", which was never measured and
  is ~4× high. Built now only where it is read, with the
  array-wins-over-set precedence written down and made build-independent.

*PR 8 — the council rounds on PR 6: the campaign gate, the input guards, the memory identity*

Six adversarial reviews of PR 6 and of each remediation in turn. The blocking
items of the last three rounds had one shape — a sentence that quantified over
a different set than the one it named — and each fix below replaces such a
sentence with a check wherever one is possible.

- **The smoke gate's tick.** `bash bench/run_smoke.sh` printed "equivalence
  groups consistent ✓" across all seven PR 6 commits while running nothing:
  every row in `bench/results/campaign/` predated them and the check was
  recomputed from a replay. Fixed in stages, each stage's review finding the
  next defect: rows are stamped with the revision that produced them
  (`d013333`); an all-crashed matrix no longer ticks and `--only <no match>` no
  longer ticks over an empty selection (`ee23ebd`, `3d1e431`); the runner
  derives one per-revision results directory itself instead of each shell
  script deriving a different one (`3d1e431`); the ok-count no longer subtracts
  overlapping failure lists — it printed "0 of 2" for 1 and went negative — and
  `-dirty` carries a 48-bit digest of `git diff HEAD`, so two edits at one
  commit are two revisions (`f5163d1`); and a git failure — which stamped every
  row "unknown", found each equal to an "unknown" `here`, and ticked — now
  refuses to start, while a tree edited during the run refuses the tick even
  when the edit lands after the last config (`8fcca70`); the two shell scripts
  keep that exit code and still write the report, which `set -e` used to skip
  (`64edf9d`).
  `tests/test_campaign_gate.py` drives the NOT-GATED branch, which a green
  campaign never reaches, and `main` itself with the child stubbed.
- **Input guards on the gpu-resident entry points.** `_assert_home` checked
  only which device the inputs were on. A wrong-dtype `prev_freq_gpu` is read
  through an `int*` cast and returned plausible garbage silently — measured,
  `[[0, 0, 0]]` at the correct count, so a count-only check passes it.
  `_assert_rank` and `_assert_dtype` now run at every guarded entry point,
  ordered so the readable error is the one that fires (`8e38366`, `c00786d`),
  and the K=2 single-GPU launch pins to its inputs' device like its K>=3
  sibling (`4344be8`). Which entry points are guarded is carried by two tuples
  in `gpu_resident.py`, not by a sentence — successive rewordings were each
  false about `build_prefix_groups_gpu`, which is exported and calls no guard
  (`7d2b982`). The tests holding the tuples to the source sat under a device
  mark that hid all seven of them on a box with no device; they are in
  `tests/test_kernel_guard_claims.py` now, gate-free, reading call sites off
  the AST (`fed0a8e`), keyed by line and column so two calls on one line are
  two; the sentence in `gpu_resident.py` that sends a reader to those tests
  went on naming the gated file after they had left it, and is now held to
  the two files' definitions by a test (`9f736fd`). Stated, not fixed:
  `dispatch_k3plus_gpu_resident` reaches `build_prefix_groups_gpu` before any
  guard.
- **Host-RAM peak of the deferred-frame build**, the `output_dir=None` route.
  #26's `.astype(np.int64, copy=False)` can never satisfy `copy=False` and
  doubled the peak, 8N -> 16N, undeclared; the `widths` list and its
  concatenation were both live beside it. Both temporaries are gone — measured
  VmHWM at 200M items, 3.00 -> 2.56 GiB, output byte-identical (`0a21f35`).
  What the peak IS was then wrong three times in the comment describing it:
  `16N` was the fixture's value, not an identity (`old = 12N + max(4N, 2R)`,
  crossover at kbar = 4, re-measured on a second instrument); `offsets` does
  not cancel under differencing; "kbar = 2.365, measured" had no artifact
  (`512c773`, `e70225e`, `543c29d`, `b758632`). The test pinning the identity
  replaced a ratio band that rejected correct code at the real `smoke` lattice
  with an absolute slack. That slack was 3.5x the smoke identity, so a build
  that doubled the peak at the production shape passed it; it is 16,384 B now,
  under the smoke identity, derived from that doubling and held by a test that
  measures one (`0ace189`, `de0b6db`, `c1e00a2`). The three known regressions are
  inside 16,384 B at the smoke shape as they were inside 65,536 B; what rejects
  them there is the differencing, not the slack. The real build's smoke excess
  is held within half a page of its 40M excess — a difference the constant
  cancels in and an extra offsets-sized array does not — and the two R-scale
  forms are held that far above the real build at the same shape, where the
  constant does not cancel and the margin is the array minus the constant. The
  astype form is separated by no peak bound at that shape; a test asserts the
  relation, in place of figures a fresh-process measurement had given with the
  wrong sign (`c1e00a2`).
- **The pinned MKL was never the one loaded.** `core/sparse.py` extended
  `LD_LIBRARY_PATH` from inside the interpreter so `sparse_dot_mkl` could find
  the venv's `libmkl_rt`; the dynamic loader reads that variable once, at
  process start, so the edit reached child processes only. Measured: on the
  dev box the copy that loaded was conda's `/opt/conda/lib/libmkl_rt.so.2`
  whether or not the variable was preset, and on the CI runner, which has no
  system MKL, `sparse_dot_mkl` failed to import and CI had been red on every
  push of this PR. `MKL_RT` is now set to the venv's highest `libmkl_rt.so.N`
  (kept if preset), which `sparse_dot_mkl` dlopens by absolute path; two tests
  read `/proc/self/maps` to hold that the mapped copy is the one under
  `sys.prefix`, in-process and in a fresh interpreter with no loader path.
- **Narration.** Every figure and citation in PR 6 that was asserted rather
  than measured was re-measured or removed: "~35 s per level" was ~4x high
  (8.9 s measured); four of five line-number citations in one comment block
  were wrong, two went stale again within the same session, and the block now
  names symbols; PR 6 was labelled "(no output change)" over a schema change;
  `perf.md` said the baseline had never been re-recorded when git said
  otherwise (`68e9a97`, `777287c`, `f25fee9`).

*PR 9 — downward closure decoupled from the free-set gate*

- **`prune_apriori`** is a parameter of `apriori()` in its own right, default
  `True`. The row-split miner's K>=3 Apriori subset test was wired to the
  free-set gate at the dispatch site (`prune_apriori=prune_equal_support`), so
  a complete-lattice run on that route — `n_gpus>1` or `anchor_items` with
  `prune_equal_support=False` — counted every suffix extension of every
  frequent group with no downward closure at all. The test is exact (it only
  removes candidates with an infrequent (k-1)-subset), so **no mined output
  changes**; what moves is the K>=3 candidate count. Measured on 2× RTX 3090,
  50,000 rows, 80 items, `min_support=0.004`, `max_length=6`: identical 5,135
  itemsets either way; the test removed 0 of 82,160 candidates at K=3, 15,804
  of 46,338 at K=4 (34%), 10,804 of 13,526 at K=5 (80%) and 2,628 of 2,635 at
  K=6 (99.7%). Wall time at that size went the other way (1.41 s with the test,
  0.60 s without) because the measuring box has no Rust extension and
  `_prune_groups_apriori` ran its Python fallback; the campaign-scale figure is
  not measured here. Every existing `prune_equal_support=True` caller already
  resolved the flag to `True`, so those runs are byte-identical.
  `prune_apriori=False` off the row-split miner is refused in
  `_validate_route_support` rather than silently ignored: the CPU route applies
  the subset test unconditionally and the single-GPU bitvec miner has no such
  step. Tests: `tests/test_prune_apriori_decoupled.py` (exactness on both
  settings, the subset test observed engaging or not, the refusals);
  reproduction: `bench/repro/d64_prune_apriori_welded_to_free_set_gate.py`.
  (Superseded: the consolidation below removed the prune and the parameter,
  with those two files.)

*GPU-layer consolidation*

- **`bitvecs=` must be a C-contiguous uint64 array.** A strided view such as
  `bv[::2]` passed validation, and the K≥2 kernels, which index
  `col * n_u64s + word`, read it as other memory: right supports at K=1,
  wrong ones above. It now raises, and every kernel wrapper checks it too.
- **The memory guards no longer run after the last level.** `max_ram_gb` /
  `max_vram_gb` tripping there could only throw away a complete lattice.

### Documentation

- `docs/specs/et_miner_fix_spec.md` amended in place. All three of its items are
  fixed, and it pointed at four paths that do not exist. It also recorded two
  things wrongly: its "Done when" for Defect A measured **closed** itemsets,
  while `prune_equal_support` returns **free-sets** — a notion the engine does
  compute — so that criterion could not have reached zero even on correct
  output. Defect A now has the standing gate it never had, at
  `bench/repro/d63_spec_defect_a_row_split_drop.py`.
- `CLAUDE.md`'s CUDA intrinsic list stated a closed enumeration that was missing
  two intrinsics the tree uses (`__ffsll`, and the 32-bit `__popc`). It now
  states the *rule* — sm_60+, NVRTC-compilable with no arch flags — with the
  current contents as an example rather than a boundary.

### Added

- `bench/baseline/` — behaviour-change impact assessment, the min-count sweep
  tool, and a performance baseline covering every code path a slow fix touches.
- `bench/repro/` — one reproduction per defect, exiting non-zero while the
  defect is live and zero once fixed, so the same file is evidence and gate.
- `tests/fixtures/min_count_cases.json` — shared ground truth for the min-count
  rule, read by the Python and Rust test suites alike.

### Removed

GPU-layer consolidation, each removal decided by the measurements in
`bench/results/2026-09-27-consolidation/FINDINGS.md` (one RTX 3060 12 GB unless
noted). Every removed parameter or value raises `ValueError` naming its
replacement.

- **SON's GPU-resident mode.** `apriori_streaming(gpu_resident=True)` (and
  `apriori(streaming=True, gpu_resident=True)`) raises. With `use_gpu=True`,
  single- and multi-GPU SON mine every chunk on the row-split miner and count
  pass 2 with the batched itemset kernel; the per-level bitvec rebuild and the
  per-itemset counting loop are gone, and `count_support_batched(use_gpu=True)`
  counts with the batched kernel too. `count_itemsets_cuda` always launches
  the batched kernel (its `use_batch` flag and the one-itemset-per-launch
  kernel `count_itemset_fused` are gone). The batched pass 2 took 1.70 s on
  `deep_k` where the per-level counter took 6.84 s, and a four-chunk SON run on
  `deep_sparse_large` finished in 194.75 s with it and hit the 600 s cap
  without.
- **The single-GPU bitvec miner and the GPU-resident miner.** The row-split
  miner now serves every in-core GPU call: `use_gpu=True` on one GPU or many,
  and `bitvecs=` (sharded by 64-row words across `n_gpus` devices). It gained
  `profile=True` and the `max_ram_gb` / `max_vram_gb` guards, and `output_dir`
  / `resume_from_k` now work with `bitvecs=` too. `apriori(gpu_resident=True)`
  raises. Gone with the two miners: their pair/candidate fan-out across GPUs
  (which copied the whole bitvec matrix to every device through host RAM), the
  per-candidate fused kernels (`pairs_k2.cu`, `k3plus_fullyfused.cu`) with
  their 10M-result ceiling, the GPU-resident kernels
  (`k3plus_gpu_resident.cu`, `decode_candidates.cu`), and the exports
  `count_pairs_fused_k2(_multi_gpu)`, `count_k3plus_fully_fused(_multi_gpu)`,
  `count_*_gpu_resident(_multi_gpu)`, `build_prefix_groups_gpu`,
  `build_k3plus_groups`, `count_k3plus_shared_fused` and
  `count_pairs_k2_shared_fused` (the tiled fused kernel is
  `count_tiled_fused`). Measured wall time, row-split vs the faster of the
  two: `oom_regression` (K≤2) 8.22 s vs 10.34 s, `deep_sparse_large` 24.19 s
  vs 26.98 s, Online Retail at 0.002 0.35 s vs 2.66 s. Where the old
  single-GPU miner won (`stress_k2` K≤2 on 12 GB, 56.21 s vs 453.57 s: the
  pair counts do not fit one dense chunk), the row-split miner now counts the
  way it did, with the fused tiled kernel.
- **The sparse CSR layout on the GPU (`sparse_from_k`).** Any value other
  than `None` raises, on `apriori()` and `mine_two_phase` (whose default was
  `"auto"`). The GPU miner keeps dense bitvectors at every level; gone are
  the dense→sparse transition, the CSR shard kernels (`csr_warp.cu`,
  `bitvec_extract_tids.cu`), `gpu/sparse_csr.py`, `gpu/density.py` and the
  Python groups' `suffix_src_rows` (the Rust builder still computes them;
  nothing reads them). The layout won no regime in the row-split miner
  (Online Retail at 0.002: 0.56 s with `sparse_from_k=3` vs 0.82 s dense, a
  sub-second gap) and ran out of memory on `deep_sparse_large` (20M rows,
  12 GB) where the dense layout mined it in 24.19 s: its shards hold four
  bytes per supporting transaction of every itemset in the level.
- **The host-side Apriori group prune (`prune_apriori`).** Any value raises.
  The GPU miner counts every candidate its prefix groups generate instead of
  subset-testing them on the host first; the results are the same. The prune
  won no measured regime against `prune_apriori=False` (largest gap 0.31 s,
  on `deep_sparse_large`), and on wide levels it cost more than the counting
  it saved: `oom_regression` to K=3 mined in 37.62 s without it and 101.25 s
  with it, and on `stress_k2` the Rust prune alone took 487 s at K=3. The
  Rust function stays in `rust_ext`; nothing calls it.
- **`ET_MINER_KERNEL_VARIANT`.** Setting it raises. The row-split miner picks
  the kernel per prefix group at the measured crossover: tiled for groups of
  at least 120 candidate pairs at K=3, falling to 23 at K≥8
  (`TILED_MIN_GROUP_PAIRS`), per-candidate below, each set counted as its own
  candidate space so the two never fragment each other's chunks.
  `ET_MINER_TILED_MIN_GROUP_PAIRS` pins that choice (0 = tiled everywhere).
  The wrappers lost `variant=`: `count_pairs_k2_allcounts` and
  `count_k3plus_allcounts` are now `count_pairs_k2_per_candidate` and
  `count_k3plus_per_candidate`. Measured before the change: the tiled K=2
  kernel was 10× faster (`oom_regression` 7.64 s vs 82.49 s); at K≥3 the old
  default tiled every group once the small-group routing fragmented its plan
  and was 5× slower than per-candidate on `deep_sparse_large` (76.39 s vs
  14.63 s), while per-candidate was 5× slower than tiled on `oom_regression`
  to K=3 (492.21 s vs 93.08 s).
- **`ET_MINER_ROW_BALANCE=nnz` and `balance="nnz"`.** Either raises; the
  multi-GPU row split is by equal row counts (`rows`, the old default, stays
  accepted as a no-op). Measured on two GPUs (2× RTX A4000 16 GB, NCCL with
  P2P disabled, `bench/results/2026-09-28-consolidation-2gpu/`), the
  nnz-balanced cut tied the rows split in both regimes built for it:
  `skewed_rows` 1.07 s vs 1.07 s and `deep_sparse_large` 18.14 s vs 17.91 s
  (medians of 3), so rule 5 keeps the smaller code. Gone with it: the
  searchsorted cut and its "largest shard must fit the smallest device"
  feasibility check.
- **`ET_MINER_FILTER_IMPL`, the `compact_threshold` kernel and the
  whole-array CPU filter.** Setting the knob raises. The dense survivor
  filter is the sliced `cp.nonzero` path,
  `et_miner.gpu.kernels.filter.threshold_filter` (`compact_threshold_filter`
  and its `impl=` are gone), with a per-slice host fallback when a slice does
  not fit the device. On two GPUs (2× RTX A4000 16 GB, NCCL with P2P
  disabled, `bench/results/2026-09-28-consolidation-2gpu/`) the three
  implementations tied in both regimes built for them: `stress_k2` K≤2
  20.00 s (compact) vs 19.05 s (cupy) vs 19.99 s (cpu), `deep_sparse_large`
  17.91 vs 18.49 vs 18.83 s (medians of 3), so rule 5 keeps the smallest: no
  kernel, no host sort (48 B/survivor), no host-RAM probe. Six registered
  kernels remain. Per 64M-element slice on an A4000 the filter takes 10 ms
  at a 1% pass rate (500 ms when every element survives) and 1.1 B/element
  of extra VRAM at 1% (12 B/element, 732 MiB, at 100%); at the
  10B-candidate levels of the unmeasured AlphaFold regime that is about
  1.6 s per level where the kernel took two passes over the array — the one
  place it could have won.
- **`et_miner.gpu.memory_budget`** (the `safe_threshold_filter` shim, imported
  by nothing, which silently ignored `max_gpu_elements`): gone.

---

## [0.1.0]

Initial development release.
