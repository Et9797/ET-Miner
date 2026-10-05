# Candidate pruning on the GPU — report

Phase 1 of `perf/gpu-candidate-pruning`. Phase 0 (what the route counted
needlessly): `bench/results/2026-10-05-candidate-waste/FINDINGS.md`. Protocol,
fixed before the first timed run: `PROTOCOL.md`. Evidence:
`bench/results/2026-10-05-pruning-final/` (the deciding run) and
`bench/results/2026-10-05-pruning/` (the first run, which found the must-fix
below; kept as evidence, not used for any decision).

## What changed

- **Kernels.** The three K≥3 counting kernels (per-candidate
  `count_k3plus_dense`, tiled `count_shared_tiled_dense` / `_fused`, sparse
  `csr_count_range`) take an index of the previous level: sorted int32 rows,
  int32 counts, free flags (`_src/_subset_index.cu`, prepended by the loader;
  `kernels/subset_index.py`). A candidate with a (k−1)-subset missing from the
  index is not counted; a tile-pair of the tiled kernels skips its word loop
  when none of its pairs needs counting. With count inference a candidate
  with a non-free subset gets the minimum of its subset counts. Plain C, one
  binary search per subset, static shared memory, blockDim 256; registered
  kernels unchanged, `selfcheck.py` launches the new path on every device.
- **Candidate order never changes.** A skipped candidate keeps its index and
  count slot (zero), so chunk plans, the NCCL sum, decode and ESCO's
  `suffix_src_rows` are untouched; every GPU gets the same index, and only the
  first device writes inferred counts, so the reduce stays exact.
- **Levels.** Every level is sorted at its end (the index is binary-searched).
  The index is uploaded before the chunk budget is measured, so the budget
  accounts for it; an index above a quarter of a device's free VRAM is not
  uploaded and that level counts every candidate (logged).
- **Free-sets.** The index is the free level (Phase 0, hypothesis 6): a
  candidate with a subset outside it is infrequent or not free. Survivors with
  such a subset are dropped on the host before the free-set test (see the
  must-fix), which then resolves against the free level alone; the complete
  generated level (`prev_full_flat`) is no longer kept, and a resumed free-set
  run is exact (tested; without the subset test it still under-prunes, as
  documented).
- **API.** `prune_apriori` is revived, default `True`: the device-side subset
  test; `False` counts every candidate and is accepted only where the
  row-split miner runs (the CPU route and SON always test). The row-split miner
  now honours `use_generator_pruning` (it needs `prune_apriori`); its default
  stays `False`.

## Decisions (protocol rule, final run)

Wall s, median [min, max] over 3 reps (sk2ml3 `off` and the 2-GPU sk2ml3 row:
1 rep). Σ K≥3 is the sum of the K≥3 level times. Table and rule-2 outcomes:
`uv run python bench/pruning/decide.py bench/results/2026-10-05-pruning-final/raw.jsonl`.

| regime | off | prune | infer | Σ K≥3 off | Σ K≥3 prune | Σ K≥3 infer | DP-P | DP-I |
|---|---|---|---|---|---|---|---|---|
| sk2ml3, 1 GPU | 496.27 | 52.77 [52.76, 53.17] | — | 462.67 | 18.87 [18.82, 19.01] | — | prune wins | — |
| sk2ml3, 2 GPUs | — | 45.19 | — | — | 26.79 | — | not judged | — |
| oom2ml3, 1 GPU | 25.76 [25.73, 25.82] | 7.18 [7.17, 7.19] | 7.24 [7.20, 7.28] | 19.87 | 1.05 | 1.08 | prune wins | tie |
| oom2ml3, 2 GPUs | 17.21 [17.21, 17.21] | 8.16 [8.15, 8.22] | — | 12.78 | 3.63 | — | prune wins | — |
| dsl, 1 GPU | 24.38 [24.34, 24.47] | 24.99 [24.97, 25.28] | 20.16 [19.81, 20.25] | 10.23 | 9.74 | 4.59 | tie | infer wins |
| dsl, 2 GPUs | 19.51 [19.45, 19.58] | 20.85 [20.32, 20.90] | 18.74 [18.47, 18.86] | 5.18 | 4.98 | 2.47 | tie | infer wins |
| dsl free-sets, 1 GPU | 19.79 [19.69, 19.86] | 17.83 [17.62, 17.98] | — | 4.21 | 3.55 | — | tie (−9.9 %) | — |
| or002, 1 GPU | 0.22 [0.21, 0.22] | 0.27 [0.26, 0.29] | 0.34 [0.32, 0.34] | 0.10 | 0.15 | 0.21 | tie | tie |
| or002 ESCO, 1 GPU | 0.25 [0.25, 0.26] | 0.39 [0.37, 0.39] | 0.43 [0.42, 0.43] | 0.14 | 0.24 | 0.29 | tie | tie |
| or002 free-sets | 0.32 [0.29, 0.33] | 0.50 [0.49, 0.51] | — | 0.19 | 0.38 | — | tie | — |
| or003 / or003 free-sets | 0.13 / 0.14 | 0.14 / 0.18 | 0.15 / — | | | | tie | tie |
| deepk / ESCO / free-sets | 0.49 / 0.87 / 0.49 | 0.50 / 0.83 / 0.49 | 0.50 / 0.81 / — | | | | tie | tie |
| skew / free-sets | 0.71 / 0.70 | 0.73 / 0.70 | 0.73 / — | | | | tie | tie |
| smoke / free-sets | 0.09 / 0.10 | 0.09 / 0.10 | 0.10 / — | | | | tie | tie |

- **DP-P, the subset test: `prune` wins three regimes and loses none →
  `prune_apriori` defaults to `True`.** oom2ml3 1 GPU −72 % (25.76 → 7.18),
  2 GPUs −53 %, sk2ml3 −89 % (496.27 → 52.77; K=3 462.67 → 18.87 s). Every
  other regime ties: where the K≥3 levels cost milliseconds (Online Retail,
  smoke, deepk, skew) the subset test and the per-level sort cost up to 0.18 s
  (or002 free-sets), below rule 2's 1 s; dsl free-sets misses the 10 % bar by
  0.02 s.
- **DP-I, count inference: `infer` wins dsl on one GPU (−19 %, 24.99 →
  20.16) and two (−10.1 %, 20.85 → 18.74) and loses none → the GPU route
  honours `use_generator_pruning`.** Its default stays `False` (the owner's
  call, outside the protocol). On dsl the per-candidate kernel's K≥3 time goes
  5.30 → 1.51 s and the tiled kernel's 4.17 → 2.75 s.
- Rule 4 is not reached (no regime is lost), so no dispatch rule is added;
  rule 5 is not reached either.

On dsl the `prune` arm's wall is 0.6 s (1 GPU) and 1.3 s (2 GPUs) above `off`
while its K≥3 levels are 0.5 s and 0.2 s faster. The difference sits entirely
before K=1 (CSR and bitvec build, which `prune_apriori` does not reach): 14.07
vs 15.17 s on 1 GPU, while on the free-set rows the same phase went the other
way (15.50 vs 14.19 s), and four alternating development runs measured
14.05/15.17 s without and 16.34/13.73 s with the test. It is process-to-process
variation of the data preparation, below rule 2 either way.

## Correctness

- **Rule 1 fired on the first run and the fix was re-measured.** In the first
  run (`../results/2026-10-05-pruning/`), dsl free-sets with the subset test
  emitted 205,414 itemsets against 204,972 (442 extra, K=5–8): a counted
  tile-pair counts all of its pairs, so a frequent, non-free candidate with a
  subset outside the free level survived next to a pair that needed counting,
  and the free-set test, resolving against the free level only, could not see
  its witnesses. Fixed by dropping such survivors on the host before the test
  (`gpu/mining.py::_all_subsets_in`); the regression test
  (`test_free_sets_with_hidden_witnesses`, six paths × one and two GPUs) fails
  on the tiled path without the fix. The final run repeated the whole matrix at
  the fixed revision.
- **Final run:** 137 config-reps ok, no fallback, one signature per
  equivalence group (14 groups, every arm, dense and ESCO, one and two GPUs);
  every signature equals the one Phase 0 recorded for its regime.
- **Gate on the final tree** (see the end of this file for the numbers):
  ruff, `pytest -m "not slow"`, the tier chain with the new legs (no subset
  test; count inference with each kernel pinned and forced chunks; ESCO with
  and without inference; the same on two GPUs), `-m "gpu and slow"`,
  `-m "gpu and multigpu"`, `selfcheck.py`. Free-set runs are checked against
  the free-sets derived from efficient-apriori's lattice
  (`test_free_sets_match_the_efficient_apriori_lattice`) and against brute
  force with every subset-test setting, layout and chunking.

## Findings

1. (must-fix, fixed) Free-set runs with the subset test emitted non-free
   itemsets when a counted tile-pair also counted a candidate with a subset
   outside the free level (dsl: 442 of 204,972). Found by the campaign's
   signature check, not by the tests, whose fixtures were too small to put
   such a candidate into a counted tile-pair; fixed and covered (above).
2. (should) The host-side drop of those survivors costs ≈ 0.24 s per free-set
   run on dsl and on or002 ("other" in the level split: 0.27 vs 0.02 s on dsl).
   Only the tiled and fused kernels can let such a survivor through; limiting
   the filter to their survivors, or moving it to Rust, would remove most of
   it (on dsl free-sets that alone would clear rule 2's 10 % bar). Not done
   after the campaign, so not measured.
3. (should) Where a level costs milliseconds the subset test does not pay:
   or002 0.22 → 0.27 s, or002 ESCO 0.25 → 0.39 s, or002 free-sets 0.32 →
   0.50 s (the test, the per-level sort and finding 2). A rule-4 dispatch on
   words per row and candidates could skip it there; the protocol does not
   call for one (no regime lost by rule 2).
4. (nice) The K=3 test is a binary search over the frequent pairs; on sk2ml3
   it costs ≈ 1–2 s of the 18.9 s K=3 level (the counted tile-pairs account
   for ≈ 16.6 s). A frequent-pair bitmap (n_items² bits, 153 MB at 35,000
   items) would cut that; not implemented.
5. (nice) `use_generator_pruning` defaults to `False` on every route; on the
   GPU it wins dsl and ties elsewhere, so flipping the default is the owner's
   decision.
6. (nice) `level_callback`'s `n_candidates` still counts every generated
   candidate on the GPU route, skipped ones included (docstring updated).
7. (nice, out of scope, from Phase 0) The tiled kernel recomputes the prefix
   AND per tile-pair; row/word compaction at deep K; item reordering; other
   condensed representations and depth-first search.

## Open risks

- Measured on one box (2× RTX A4000, sm_86) only; sm_90, NVLink and cards
  above 16 GB are untested.
- An index larger than a quarter of free VRAM turns the test off for that
  level (exact, tested by forcing it); at AlphaFold scale a K≥7 level can be
  that large, and chunking the index is not implemented.
- Count inference on ESCO's sparse path saves the count only: inferred
  survivors' tidsets are still intersected by `materialize_survivors`.

## Gate on the final tree

`NCCL_P2P_DISABLE=1`, both A4000s; the four `test_smoke_correctness.py`
`test_support_001` / `test_support_0001` ids deselected, as in the
consolidation (they cannot finish with Online Retail present).

| Check | Result |
|---|---|
| `uv run ruff check src tests bench` | clean |
| `uv run pytest -q -m "not slow"` | 894 passed, 12 deselected |
| `uv run pytest tests/test_tier_equivalence.py` | 32 passed (incl. the 2-GPU legs) |
| `uv run pytest -q -m "gpu and slow"` | 5 passed |
| `uv run pytest -q -m "gpu and multigpu"` | 34 passed |
| `uv run python bench/selfcheck.py` | READY, the subset path launched on both devices |
| runner signature check, final run | 137 config-reps, 14 groups, consistent ✓ |
