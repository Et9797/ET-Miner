# Route optimizations after candidate pruning — spec

Eight optimizations proposed at the end of the candidate-pruning work
(`bench/pruning/REPORT.md`, PR #19), for a next session to pre-register,
build and measure. With the K≥3 explosion handled by the device-side subset
test, the remaining time sits elsewhere: the K=2 level, the multi-GPU reduce,
ESCO's materialization. Every figure below cites its source; "estimate" marks
what is not measured.

## Ground rules

- **Base.** Branch from `perf/gpu-candidate-pruning` (PR #19), or from `main`
  once #19 is merged; O2, O3, O4 and O7 build on its subset index.
- **Correctness policy (`CLAUDE.md`) is binding.** Every new path gets tier-chain
  legs in `tests/test_tier_equivalence.py` and the chain in `CLAUDE.md` (the
  only `CLAUDE.md` edit allowed); exact itemsets and absolute counts against
  efficient-apriori with the `(min_count - 0.5) / N` convention and an explicit
  `max_length`. Free-set paths are checked against free-sets derived from
  efficient-apriori's lattice (`tests/test_free_set_semantics.py`).
- **Kernels** follow `CLAUDE.md`: plain C, NVRTC without arch flags (sm_86 and
  sm_90), static shared memory ≤ 48 KB, blockDim 256 where counting, int32
  dense counts; register in `loader._KERNEL_FILES`; `bench/selfcheck.py`
  launches each with a known answer.
- **Measurement.** Write `bench/optimizations/PROTOCOL.md` before the first
  timed run, with the consolidation protocol's rules 1, 2, 4 and 5. One box per
  comparison (runner records `nvidia-smi -L`, driver, CUDA, commit, packages),
  fresh process per config, 3 reps rep-major, medians with [min, max], nothing
  else on the GPUs during a run, never compare across boxes. Freeze tracked
  files during a campaign (the runner stamps rows with the tree's digest).
  Budget to agree with the owner up front.
- **Process.** No commit, push or PR without the owner's word; chat in Dutch,
  files in English; short module docstrings (what + Usage + Options), details in
  function docstrings, no development history in code; findings numbered and
  tagged (must-fix)/(should)/(nice); the core stays domain-agnostic.

## Priority

| Phase | Items | Why first |
|---|---|---|
| A | O1, O2, O3 | measured stakes of seconds to tens of seconds per run |
| B | O4, O5 | ESCO: memory at depth, and a transition rule that wins |
| C | O6, O7 | smaller, partly measured |
| research | O8 | no evidence yet; measure offline before building |

## O1 — K=2 counted from the rows

**Evidence.** After the subset test K=2 is the largest level where the
explosion was (`bench/results/2026-10-05-pruning-final/`, one GPU, prune arm):
stress_k2 to K=3 K=2 31.07 of 52.76 s; oom_regression to K=3 K=2 5.51 of
7.18 s. The tiled kernel ANDs every frequent-item pair over every word. Work
per level, computed on the datasets (frequent items only):

| Dataset (support) | pairs | words | Σ_rows C(len, 2) | pairs × words |
|---|---|---|---|---|
| stress_k2 (0.000015) | 612,482,500 | 31,250 | 562,968,354 | 1.91e13 |
| oom_regression (0.00003) | 449,985,000 | 7,813 | 98,735,918 | 3.52e12 |
| deep_sparse_large (0.015) | 3,570 | 312,500 | 1,546,715,052 | 1.12e9 |
| online_retail (0.002) | 3,415,191 | 570 | 40,906,330 | 1.95e9 |

**Proposal.** A row-wise pair kernel: each row adds 1 to every pair of its
frequent columns in the same dense int32 pair array the current K=2 kernels
fill (pair (i < j) of the frequent-column list at `j·(j−1)/2 + i`), so the
reduce, threshold filter and `decode_k2_pairs_flat` stay as they are.
Atomics with warp aggregation or shared-memory privatization for hot pairs
(Zipf data). Input: each shard's rows as CSR of frequent-column positions (the
transactions route builds a CSR before the bitvecs; `bitvecs=` input keeps the
dense kernel). A pair array beyond one chunk is processed by ranges of j.
The CPU tier already counts co-occurrences in Rust
(`rust_ext/src/core/cooccurrence.rs`).

**Dispatch (rule 4).** From facts known before K=2: Σ_rows C(len, 2) against
pairs × words. The table predicts rows on stress_k2 and oom_regression, dense on
deep_sparse_large; the crossover is measured, not assumed. Estimate (not
measured): stress_k2 K=2 from ≈ 31 s to a few seconds.

**Acceptance.** Identical signatures; tier legs (row-wise K=2 pinned, 1 and 2
GPUs, forced chunks); a K=2 unit test against NumPy pair counts; the protocol
decides the dispatch.

## O2 — multi-GPU reduce that skips the skipped candidates

**Evidence.** On two GPUs the K=3 level is slower than on one, and the reduce
dominates it (final run, prune arm, level split): stress_k2 to K=3 K=3 26.79 s
of which reduce 16.51 s (one GPU: 18.87 s, no reduce); oom_regression to K=3
K=3 3.63 s of which reduce 2.92 s. The NCCL sum covers whole int32 chunk arrays
that are zero wherever the subset test skipped a candidate (≈ 97 % of the
tile-pairs at K=3 on these two, Phase 0). This box runs NCCL without P2P.

**Proposal.** The skip decision is a pure function of the shared index and
candidate space, so every GPU skips the same entries. Have the kernels record
which tile-pairs (tiled) and candidates (per-candidate) they counted, compact
the counted ranges into one buffer per GPU in the same order, reduce the
compact buffers, and threshold-filter them with their original indices.
Alternative to weigh: exchange only nonzero partials (variable sizes, needs a
size exchange first).

**Acceptance.** Identical signatures; the 2-GPU tier legs (plain, forced
chunks, count inference) green; a test that a compacted reduce equals the
dense one on randomized skip masks; measured on the 2-GPU rows of the pruning
matrix (dsl, oom2ml3, sk2ml3).

## O3 — ESCO materialization without recounting

**Evidence.** `gpu/sparse_csr.py::materialize_survivors` intersects every
survivor twice more after the count pass (`count_csr_gather` for the new
offsets, then `write_csr_gather`). The ESCO levels' unattributed time ("other"
in the level split, which includes the transition and the materialization) is
0.10 s at deep_k K=3 and 0.08 s at or002 K=3 (final run, `C1-esco-prune`);
ESCO lost both measured regimes to dense (deep_k 0.83 vs 0.50 s, or002 0.39 vs
0.27 s).

**Proposal.** Add `transition` and `materialize` phases to `bench/level_split.py`
first. On one shard the shard counts are the global counts, so the new offsets
are the exclusive scan of the survivor counts and the gather-count pass goes.
On several shards keep the survivors' per-shard partial counts from the count
pass (gathered before the in-place reduce, or recovered as global minus the
other shards) instead of recounting.

**Acceptance.** Identical signatures; the existing row-length checks (per-shard
lengths sum to the counts) stay; ESCO tier legs on 1 and 2 GPUs.

## O4 — inferred survivors share their tidset

**Evidence.** With count inference, a candidate X with a non-free subset has
rows(X) = rows(Y) for any (k−1)-subset Y with count(Y) = count(X), and Y is a
row of the previous shard. Such candidates are 61–100 % of deep_sparse_large's
levels from K=9 and 82–100 % of deep_k's from K=5 (Phase 0,
`bench/results/2026-10-05-candidate-waste/FINDINGS.md`). ESCO ran out of memory
on deep_sparse_large on 12 GB (consolidation DP5).

**Proposal.** The CSR kernel writes, for an inferred candidate, the row of an
equal-count subset; materialization copies that row instead of intersecting.
Sharing without a copy needs a (start, length) shard layout instead of prefix
offsets — weigh that against the copy on memory.

**Acceptance.** Identical signatures; ESCO legs with count inference; peak VRAM
recorded on deep_sparse_large with `sparse_from_k="auto"`.

## O5 — ESCO transition by cost, not bytes

**Evidence.** `"auto"` switches when the previous level's mean count falls below
N/32, a byte crossover (`bench/ESCO.md` says it is not a speed crossover). ESCO
from K=3 lost both measured regimes (O3).

**Proposal.** A per-level cost model from facts known before the level: dense
≈ candidates × words (per kernel), sparse ≈ Σ over candidates of
min(|A|, |B|) from the parent counts plus the one-time conversion; constants
from a sweep in the style of `bench/kernel_crossover.py`; rule 4 sets the
crossover. A fixed `sparse_from_k` stays.

**Acceptance.** The protocol decides whether `"auto"` changes; signatures
identical whichever layout runs.

## O6 — small prefix groups share their prefix AND

**Evidence.** The per-candidate kernel reads every prefix row for every
candidate; the tiled kernel's 32×32 tiles waste most of a tile on groups of 2–4
suffixes (15–45× slower there, consolidation DP3). On deep_sparse_large the
per-candidate kernel takes 5.30 s of the K≥3 levels with the subset test and
1.51 s with count inference (final run).

**Proposal.** A small-tile variant of the tiled body (8 or 16 suffixes per
tile), or a group-per-block kernel that stages the prefix AND once per word and
serves all pairs of a small group; then re-measure the per-K crossover
(`TILED_MIN_GROUP_PAIRS`).

**Acceptance.** Identical signatures; pinned tier legs for the new kernel; the
crossover table regenerated.

## O7 — three small items from the pruning report

- **(a) Free-set host filter.** `_all_subsets_in` runs on every survivor of a
  free-set level and costs ≈ 0.24 s per run on deep_sparse_large and or002
  ("other" 0.27 vs 0.02 s). Only the tiled and fused kernels (and a level
  without the index) let such a survivor through: filter just those, or move
  the filter to Rust. On deep_sparse_large free-sets the subset test missed
  rule 2's 10 % by 0.02 s.
- **(b) Subset test where levels cost milliseconds.** or002 0.22 → 0.27 s,
  or002 ESCO 0.25 → 0.39 s, or002 free-sets 0.32 → 0.50 s (test, per-level sort,
  item a). A rule-4 dispatch on words per row and candidates.
- **(c) K=3 frequent-pair bitmap** (n_items² bits: 153 MB at 35,000 items)
  instead of the binary search; estimate ≈ 1–2 s of stress_k2's 18.9 s K=3
  level.

## O8 — research: row order and early tile skips

`shared_tiled.cu` stages a word tile's suffix rows before it checks whether the
prefix AND over those 32 words is zero. On randomly ordered rows such tiles are
rare (deep_sparse_large's deep itemsets cover ≈ 2 % of rows, so a 2,048-row
tile is almost never empty). Reordering rows so that transactions sharing items
are adjacent (counts are permutation-invariant) could make them common. Measure
first, offline: the fraction of all-zero prefix tiles before and after a
candidate reordering, per level. Build only if that fraction is large.

## Out of scope

Other condensed representations (closed, maximal, non-derivable), depth-first
or hybrid search, item reordering for generation order.
