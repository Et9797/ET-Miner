# Candidate waste on the GPU row-split route (Phase 0)

What the row-split miner counts per K≥3 level that it need not count, measured
without changing the miner: every generated candidate is labelled offline, and
each level's time is split into phases on the GPU. The go/no-go per lever is at
the end; Phase 1 gets its own pre-registered protocol.

Labels, per generated candidate X of level k:

- **P (prunable)**: at least one (k−1)-subset is infrequent. X is infrequent.
- **I (inferable)**: every (k−1)-subset is frequent and at least one is not
  free. X is then frequent and count(X) = min of its (k−1)-subset counts
  (Pascal, Bastide et al. 2000): a non-free subset Y has some y with
  rows(Y∖y) = rows(Y), so rows(X∖y) = rows(X).
- **C (must count)**: everything else.

## Conditions

- **Box.** 2× RTX A4000 16 GB (sm_86), driver 550.163.01 (CUDA 12.4), CUDA
  runtime 12.9, CuPy 14.1.1, NCCL 2.31.2, Ryzen 5 5600X 6C/12T, 46 GB RAM,
  Python 3.10.13, numpy 2.2.6, polars 1.43.2, `et_miner_rust` 0.3.0
  (`env.txt`). Every number here is from this box; none is compared with a
  number from another box.
- **Revision.** `f53c1d0` with the uncommitted bench-only diff of this branch
  (rows stamped `f53c1d0-dirty.52a7e82cddca`; the oom2ml3 rows, run after one
  more matrix line was added, `f53c1d0-dirty.3bfbf82c4188`). `src/` is
  unchanged.
- **Runs.** `bench/runner.py --mode waste` (`build_waste_matrix`): route C on
  one GPU (`CUDA_VISIBLE_DEVICES=0`, `ET_MINER_DISABLE_NCCL=1`), default
  dispatch, thread pools pinned to 6, fresh process per config with the usual
  warm-up, 3 reps rep-major, `NCCL_P2P_DISABLE=1` exported, nothing else on the
  GPUs. Complete lattice and free-sets (`prune_equal_support=True`) are separate
  regimes. `sk2ml3` ran once (526 s). `oom2ml3` (oom_regression to K=3, the
  DP9 supplementary regime) is beyond the requested set; it was added after the
  matrix and ran 3 reps into `oom2ml3/`. Cells are median [min, max].
- **Level split.** `bench/level_split.py` wraps the miner's inner calls with
  timers (group build, group upload, budget probe, per-candidate / tiled /
  fused counting, reduce, filter, decode, sort, free-set prune; the rest is
  "other"). The kernel wrappers synchronize, so wall time is device time. The
  wrappers add a timer pair per call; what they cannot attribute ("other") is
  0.045 s of dsl's 10.4 s of K≥3 levels.
- **Correctness.** All 40 config-reps ok; one signature per regime, equal to
  the earlier campaigns' signature in every regime that has one (11 of 13, plus
  oom2ml3); skew and or003 free-sets have no earlier row. Runner tick:
  "equivalence groups consistent ✓ (37 configs in 13 groups)".
- **Budget.** 0.209 GPU-hours for the matrix, 0.024 for oom2ml3, < 0.01 for
  development runs and the preset prototype: ≈ 0.24 of the 2 allowed.

## The classifier and its validation

`bench/candidate_waste.py classify` reads a lattice dump of the complete-lattice
run (`dump_lattice`), replays the route's generation (the K=2 pair space over
the generation base, then `build_k3plus_groups_from_flat`, the builder the
route uses, at every K≥3) and labels every generated candidate with exact
lookups: a level-t itemset's key is `row(prefix in level t−1) · n_items + last
item`, so one binary search per item resolves any itemset. Validation, all
passing:

- Every replayed level's generated count and emitted count equal the GPU run's
  `level_callback` (`n_candidates`, `n_frequent`), every level of every regime.
- P + I + C = classified = generated at every level (every candidate is
  enumerated, except sk2ml3's K=3, below).
- No P candidate is frequent; every I candidate is frequent and its inferred
  count equals the lattice count (all regimes, 0 exceptions).
- Brute force on the CPU (bitsets from the dataset's rows) over every generated
  candidate of smoke (15,257 complete, 15,222 free-sets) and or003 (4,000,622
  and 4,000,615):
  exact counts equal the lattice for every frequent candidate, frequency agrees
  for every candidate, every P is infrequent, every I's inferred count equals
  its true count. 0 mismatches.
- The free-set replay (generate from the emitted free level, test equality
  against the frequent generated level, as the route does) equals both the true
  free sets of the complete lattice and the GPU's free-set dump, every level.
- The candidates with a (k−1)-subset missing from the free level are exactly
  P ∪ I at every level, in both regimes.

**sk2ml3 is sampled, as instructed.** K=3 (11,796,796,856 candidates in 784
prefix groups) is classified on a systematic sample of prefix groups: groups
ordered by size, every tenth from a random start (seed 0), so each group's
inclusion probability is 0.1. The sample holds 78 groups and 1,046,625,693
candidates. K=2 is counted analytically (a pair's subsets are frequent items,
so K=2 has no P). The sample's P share, 98.65 % (SE 0.62), agrees with the
exact population share, 98.77 %, computed independently as the number of
frequent-pair triangles (the method equals full enumeration on smoke, dsl and
or002). The sample does not represent survivors: 4.3·10⁻⁵ frequent in the
sample against 1.1·10⁻⁴ in the level, which sits in the few giant groups of the
most popular items. Tile and suffix shares for sk2ml3 are sample shares.

## Results

Σ over K≥3. "Counting on P" is the model below applied per level with the
median phase times: the share of each kernel's time spent on candidates a given
prune granularity removes (per-candidate kernel ∝ candidates, tiled kernel ∝
tile-pairs).

| workload | regime | wall s | Σ K≥3 levels s | Σ K≥3 counting s | K≥3 generated | P % | I % | counting on P, per candidate | per tile-pair | per suffix | on P ∪ I, per candidate | per tile-pair |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sk2ml3 (1 rep, K=3 sampled) | full | 526.25 | 493.44 | 492.93 | 11,796,796,856 | 98.7 | 0.0 | 486.29 | 475.28 | 18.95 | 486.29 | 475.28 |
| oom2ml3 | full | 27.01 [25.97, 28.22] | 20.90 [20.08, 21.89] | 20.80 | 2,081,967,305 | 99.2 | 0.0 | 20.65 | 20.17 | 3.80 | 20.65 | 20.17 |
| dsl | full | 27.73 [25.35, 30.51] | 10.40 [10.20, 10.44] | 10.25 | 895,084 | 34.7 | 41.2 | 3.36 | 1.07 | 0.20 | 8.30 | 6.40 |
| dsl | free | 19.93 [19.78, 21.89] | 4.31 [4.24, 4.33] | 4.13 | 434,743 | 33.8 | 16.7 | 1.25 | 0.43 | 0.08 | 2.09 | 1.06 |
| or002 | full | 0.217 [0.211, 0.218] | 0.092 [0.091, 0.094] | 0.027 | 10,004,205 | 65.1 | 0.1 | 0.016 | 0.005 | 0.003 | 0.016 | 0.005 |
| or002 | free | 0.307 [0.295, 0.330] | 0.183 [0.174, 0.203] | 0.028 | 9,991,240 | 65.1 | 0.1 | 0.016 | 0.005 | 0.004 | 0.017 | 0.005 |
| or003 | full | 0.142 [0.136, 0.213] | 0.026 [0.022, 0.028] | 0.006 | 1,701,182 | 65.5 | 0.0 | 0.004 | 0.001 | 0.001 | 0.004 | 0.001 |
| or003 | free | 0.149 [0.147, 0.152] | 0.034 [0.034, 0.037] | 0.007 | 1,701,175 | 65.5 | 0.0 | 0.004 | 0.001 | 0.001 | 0.004 | 0.001 |
| skew | full | 0.750 [0.725, 0.755] | 0.053 [0.052, 0.116] | 0.032 | 98,399 | 88.2 | 1.0 | 0.029 | 0.017 | 0.023 | 0.030 | 0.017 |
| skew | free | 0.749 [0.726, 0.773] | 0.114 [0.061, 0.115] | 0.032 | 97,709 | 88.8 | 0.3 | 0.029 | 0.017 | 0.023 | 0.029 | 0.017 |
| deepk | full | 0.515 [0.514, 1.735] | 0.034 [0.032, 0.040] | 0.017 | 19,187 | 54.8 | 37.2 | 0.011 | 0.003 | 0.005 | 0.016 | 0.007 |
| deepk | free | 0.499 [0.497, 0.527] | 0.028 [0.027, 0.028] | 0.013 | 12,531 | 83.9 | 3.9 | 0.012 | 0.003 | 0.008 | 0.012 | 0.003 |
| smoke | full | 0.097 [0.064, 0.099] | 0.008 [0.007, 0.008] | 0.002 | 8,354 | 91.9 | 0.8 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| smoke | free | 0.095 [0.071, 0.100] | 0.007 [0.007, 0.007] | 0.001 | 8,319 | 92.3 | 0.3 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |

What the table says:

1. **P is a third to nearly all of what the route counts at K≥3**: 35–45 %
   per level on dsl's big levels (K=3–7), 60–91 % on Online Retail, 84–94 % on
   skew, 92 % on smoke, 99.2 % on oom2ml3, 98.8 % on sk2ml3.
2. **Counting is the whole K≥3 level wherever the level costs anything**: dsl
   98.5 % (per-candidate kernel 5.98 s, tiled 4.27 s; every other phase
   together 0.15 s), oom2ml3 99.5 %, sk2ml3 99.9 %.
   In smoke, deepk, skew, or003 and or002 all K≥3 counting together is at most
   0.032 s, below what rule 2 can resolve (≥ 1 s absolute).
3. **The granularity decides what is recoverable.** In the K=3 explosions
   almost every 32×32 tile-pair is entirely P: oom2ml3 96.95 % of tile-pairs
   (97.8 % of P sits in them), sk2ml3 96.4 % (97.8 %). On dsl no tiled group has
   a single fully-P tile-pair (≤ 85 frequent items, so a group spans ≤ 3 tiles,
   and P and C pairs share every tile); on Online Retail 0–2.6 % of P. There the
   per-candidate kernel's groups (58 % of dsl's counting) are the only part a
   prune can reach without changing the tiled kernel's work unit.
4. **The suffix granularity of the removed prune reaches little of P** where P
   costs time: 2.5–6.3 % at dsl's K=5–8, 18.4 % on oom2ml3, 3.9 % on sk2ml3
   (sample).
5. **I is zero at K=3 on every synthetic preset** (no pair is non-free: the
   free and complete K=2 levels are equal) and 4 / 8 pairs on Online Retail
   (or003 / or002). It dominates the deep levels of the planted-motif presets:
   dsl 61–100 % from K=9 (41 % of all K≥3 candidates), deepk 82–100 % from K=5.
   A deep subset of a planted motif occurs (almost) only in the planted rows, so
   it implies the rest of the motif: set-level implications, not item-level
   ones.

### Why DP9 tied

DP9 measured the removed host-side prune at a tie (largest gap 0.31 s on dsl)
and a 63.6 s loss on oom2 to K=3. Both follow from the numbers above, and P
is not rare in either:

- **dsl**: P is 35–45 % of the big levels and the per-candidate share of their
  counting is worth 3.36 s, but the suffix granularity reaches 2.5–6.3 % of P
  there; the model gives 0.20 s for what the suffix prune could save, against
  DP9's measured 0.31 s gap. The granularity discarded the gain.
- **oom2ml3**: P is 99.2 % of 2.08 B candidates; the suffix prune reaches
  18.4 % of it (≈ 3.8 s of the 20.8 s level here), while its host-side
  `HashSet<Vec<i32>>` pass cost 63.6 s on the 3060. Cost far above a gain the
  granularity had already cut to a fifth.

## The free-level argument (hypothesis 6)

Claim. In a free-set run, let F_{k−1} be the emitted level k−1. A generated
candidate X with a (k−1)-subset Y ∉ F_{k−1} is infrequent or not free, so it is
neither emitted at level k nor part of any later generation base.

Proof. F_{k−1} is exactly the frequent free (k−1)-itemsets (below). If Y is
infrequent, X ⊇ Y is infrequent. If Y is frequent but not free, there is y ∈ Y
with rows(Y∖y) = rows(Y); then rows(X∖y) = rows(Y∖y) ∩ rows(X∖Y) = rows(X), so X
is not free (freeness is anti-monotone, Boulicaut et al. 2003). ∎

For every counted candidate all (k−1)-subsets are then in F_{k−1}, and X is not
free iff count(X) equals the count of one of them (support is monotone along
any chain Z ⊂ W ⊂ X, so an equal-support proper subset implies an
equal-support (k−1)-subset). The equality test needs F_{k−1}'s counts only;
`prev_full_flat` becomes unnecessary.

Why the route's F_{k−1} is exactly the frequent free sets, with the equality
test it runs today: induction on k. F₁ is the frequent items below n. A free
frequent X has free frequent prefix-parents (anti-monotone), so X is generated.
If X is generated and frequent but not free, with witness Y = X∖x, then Y is in
the level the route tests against: if x is one of X's last two items, Y is a
prefix-parent of X (in F_{k−1}); otherwise Y's own prefix-parents are
X∖{x, x_k} ⊂ X∖x_k and X∖{x, x_{k−1}} ⊂ X∖x_{k−1}, subsets of X's free
prefix-parents, hence free, frequent and in F_{k−2}, so Y was generated and is
frequent. The replay confirmed the equality on every level of the six measured
free-set regimes and of the preset prototype.

Reconciling `row_split.py:154-168`. That warning is about the current order of
operations: every generated candidate is counted, then each (k−1)-subset is
looked up and a subset that is **not found counts as "no equal subset", i.e.
keep**. Resolving that lookup against F_{k−1} instead of the frequent generated
level would keep a candidate X whose only equal-count subset is itself non-free
(Y ∉ F_{k−1}) — the under-prune the docstring warns about. With the subset test
first, a subset that is not found means **drop** (by the claim, X is P or I),
so every candidate that reaches the equality test has all its subsets in
F_{k−1} and the lookup can no longer miss. The two rules differ only in what
"not found" means. One consequence: a resumed free-set run, whose parquet holds
only F (`row_split.py:396-408` warns it "may under-prune slightly"), becomes
exact.

The docstring's description of `prev_full_flat` as "the complete frequent
level" is not what the code holds in a free-set run: it is the frequent
candidates generated from free sets. The results are exact anyway (above); see
finding 1.

## Levers: go / no-go

The test a lever needs per candidate is (k−2) lookups of a (k−1)-itemset (the
two prefix-parents are frequent by construction): dependent random reads,
⌈log₂|L_{k−1}|⌉ per lookup with binary search, one or two with hashing, one bit
at K=3 with a frequent-pair bitmap. Counting a candidate reads N/64 words per
row it ANDs, coalesced; the tiled kernel stages 2·32 + (k−2) rows per word per
tile-pair. Σ K≥3 binary-search steps over counting word reads: dsl 6·10⁻⁵,
deepk 6–12·10⁻⁴, skew 3·10⁻³, smoke 3·10⁻², sk2ml3 1.2·10⁻² (5.6·10⁻⁴ as one
bitmap read per candidate), oom2ml3 3.8·10⁻² (2.0·10⁻³ with the bitmap),
Online Retail 0.14–0.16. A random read costs several coalesced ones, so the test
is noise wherever counting matters, and can cost as much as it saves on Online
Retail's 570-word bitvectors, where everything is milliseconds.

### Lever 1 — exact GPU subset prune (P): GO

- **Payoff (model).** oom2ml3: 20.17 s of the 20.80 s K=3 counting at
  tile-pair granularity (wall 27.0 s → ≈ 7 s). sk2ml3: 475 s of 493 s (wall
  526 s → ≈ 51 s). dsl: 1.07 s (full) and 0.43 s (free) of 27.7 s / 19.9 s
  wall, under rule 2's 10 %; 3.36 s would need candidate granularity inside
  tiled groups, which the tiled kernel does not have. Elsewhere ≤ 0.03 s.
- **Granularity.** Per candidate in the per-candidate kernel, per 32×32
  tile-pair in the tiled and fused kernels. Group or suffix granularity loses
  most of P (finding 4 above). Compaction of surviving candidates buys nothing
  over a skip at these two granularities in the measured regimes and changes
  the candidate indices.
- **Shape this suggests for Phase 1** (to weigh against the brief's separate
  pre-pass): the same test as a block-level early exit inside the two counting
  kernels, or as a per-tile-pair / per-candidate skip mask computed by a
  pre-pass. Either way a skipped candidate keeps its index and gets count 0,
  which the threshold filter drops (P is infrequent; with min_count = 0 there
  is no P). Candidate order, chunk plans, the NCCL sum, decode and ESCO's
  `suffix_src_rows` are untouched, and every GPU skips the same entries.
  Index: a frequent-pair bitmap at K=3 (n_items² bits: 153 MB at 35,000
  items, 0.9 MB at 2,614), sorted keys or open addressing at K≥4, counted in
  `compute_chunk_budget`.
- **Expected verdict under the protocol**: a rule-2 win on oom2ml3 and sk2ml3,
  a tie elsewhere. Where N is small the test can match the counting it saves,
  so Phase 1 must either show it never loses or carry a rule-4 dispatch from
  facts known before the level (words per row, candidates, |L_{k−1}|).

### Lever 2 — GPU Pascal inference (I), complete lattice: GO, after lever 1

- **Payoff (model).** dsl: P ∪ I at tile-pair granularity 6.40 s of 10.25 s
  counting against 1.07 s for P alone, i.e. +5.3 s from I (−19 % of wall; the
  K≥3 levels 10.4 s → ≈ 4 s). At candidate granularity 8.30 s. deepk's I is
  worth 0.005 s; the synthetic presets have no I at K=3, sk2ml3 and oom2ml3 none
  at all.
- **What it needs.** The previous complete level's counts and free flags on
  the device: the free test then runs on every level, not only in free-set runs
  (dsl's free-set prune costs 0.028 s for all K≥3). The inferred count goes into
  the dense array through the same index the P test uses (the min over the
  found subsets), written by one GPU while the others write 0, so the reduce
  stays exact and survivors leave the filter in candidate order: the level-end
  merge and lexsort are unchanged.
- **Limits.** On ESCO's sparse path an inferred survivor still needs its tidset
  (`materialize_survivors` intersects its parents'), which is the counting work
  itself; the gain there is the count only. The payoff is concentrated in deep
  levels of data with set-level implications, which the measured set has only
  through planted motifs.

### Lever 3 — free-level prune (free-sets): GO as part of lever 1

- **Payoff (model).** dsl free: P ∪ I (the candidates missing from F) is 1.06 s
  at tile-pair granularity, 2.09 s per candidate, of 19.9 s wall: no rule-2
  win on its own in the measured set. On the K=3 explosions it equals lever 1.
- **Why still go.** It is lever 1's mechanism with F_{k−1} as the index instead
  of the complete level; it needs no inference. It makes `prev_full_flat`
  unnecessary (the equality test needs F only: argument above) and makes a
  resumed free-set run exact.

## Proposed preset (not merged): item-level implications

Confirmed: none of smoke, deep_k, skewed_rows, oom_regression, stress_k2 or
deep_sparse_large has a non-free item or pair (0 of 290 / 471 / 717 / 422,486 /
1,660,332 / 1,058 frequent pairs), so no candidate is I before K=4, while
real hierarchical data (an ontology term implies its ancestors) has
implications from K=2. Proposal:

```python
# SynthSpec, new fields (0 disables both):
#: Item hierarchy: items i >= implication_fanout have the parent
#: i // implication_fanout (items 0..fanout-1 are roots), and every drawn item
#: brings its ancestors up to implication_depth steps, so a child always
#: implies its parent.
implication_fanout: int = 0
implication_depth: int = 0

"implications": SynthSpec(
    name="implications", n_rows=200_000, vocab_size=2_000, zipf_a=1.05,
    row_len_mean=9, row_len_max=40, motif_count=3, motif_size=5,
    motif_penetration=0.02, implication_fanout=8, implication_depth=2,
    min_support=0.01, seed=43,
),
```

`generate_csr` adds each (row, item)'s ancestors before its global dedupe (the
prototype does it after, `implication/prototype.py::ancestors_closure`).
Self-check in `check_preset_purpose`, in the style of the existing ones: no
root's expected count may reach n_rows (else K=1 degenerates into non-free
items); at least 50 expected-frequent items must have an expected-frequent
parent (the implications the preset exists for); the planted-motif checks as
today. The oracle convention applies unchanged with an explicit max_length;
efficient-apriori is too slow for the full depth (below), so a tier-chain leg
would use a 20,000-row slice and max_length 4.

Prototype (smoke's spec at 200,000 rows, fanout 8, depth 2, one GPU, single
run; `implication/`): nnz 1.66 M → 3.02 M; the complete lattice has 790,823
itemsets to K=17, the free sets 29,651 to K=8. I is 1.7 % of K=3, 4.5 % of
K=4, 15.6 % of K=5, 47.9 % of K=6, 88.7 % of K=7 and 99.5–100 % from K=8:
62.5 % of all K≥3 candidates (P 34.3 %, C 3.2 %). Every classifier check
passed on it.

## Findings

No (must-fix) item: nothing found changes mined output.

1. (should) `row_split.py:154-168` (and `_prune_equal_support`'s docstring in
   `core/apriori.py`, same claim for the CPU path) call the level the free-set
   test resolves against "the complete frequent level". In a free-set run it is
   the frequent candidates generated from free sets, a strict subset whenever
   some frequent set has a non-free prefix-parent (dsl K=7: 68,838 vs 114,509).
   Output is exact (proof above; replay equal on every level of seven regimes,
   prototype included),
   but the stated invariant is not the one that holds, and Phase 1 replaces the
   mechanism. Fix the wording with Phase 1.
2. (should) Lever 1's test is not free where N is small: Σ binary-search steps
   are 14–16 % of the counting word reads on Online Retail. Phase 1 must show
   it never loses, use hashing / the K=3 bitmap, or dispatch on facts known
   before the level (rule 4).
3. (should, fixed) The runner's environment capture recorded no package
   versions under uv (`python -m pip freeze` prints nothing without pip).
   `capture_environment` now also records `uv pip freeze`; this run's versions
   were appended to `env.txt` by hand after the runs, from the same venv.
4. (nice) The tiled kernel's work unit decides lever 1 on dsl-like data: P and
   C share every tile-pair of dsl's ≤ 3-tile groups, so 2.3 s of P counting in
   tiled groups (3.36 − 1.07) is out of reach without a smaller work unit.
5. (nice) sk2ml3's uniform group sample estimates P and tile shares but not
   survivors (giant groups of popular items hold the frequent triples). A
   survivor estimate would need size-proportional sampling or a census of the
   largest groups.
6. (nice) `prune_apriori`'s ValueError text ("as fast or faster in every
   measured regime") holds for the removed suffix prune only; to be reworded
   with these numbers when Phase 1 revives or replaces the parameter.
7. (nice, out of scope) Per-candidate cost: `shared_tiled.cu` recomputes the
   prefix AND in every tile-pair block for every word (each of a group's
   tile-pairs re-reads its k−2 prefix rows); row or word compaction at deep K;
   item reordering.
8. (nice, out of scope) Other condensed representations (closed, maximal,
   non-derivable itemsets) and depth-first or hybrid search.

## Files

| File | What |
|---|---|
| `raw.jsonl`, `oom2ml3/raw.jsonl` | runner rows: per-level times, `timings.level_split`, signatures |
| `waste.jsonl` | classifier rows, one per (workload, regime, K) |
| `env.txt`, `oom2ml3/env.txt` | `nvidia-smi -L`, `nvidia-smi`, commit, package versions |
| `implication/` | the preset prototype, its classifier rows and summary |

Lattice dumps (`*.lattice.parquet`, gitignored), per-config logs and
`*.result.json` stay beside them, uncommitted; the runner regenerates them.

Reproduce:

```bash
uv run python -m et_miner.synthetic --preset all --out datasets/synth
uv run python datasets/prepare_online_retail.py
OUT=bench/results/2026-10-05-candidate-waste
NCCL_P2P_DISABLE=1 uv run python bench/runner.py --mode waste --out $OUT --max-gpu-hours 1.5
# per workload W and regime (add --free and the -free result for free-sets; --brute on smoke/or003;
# --sample-groups 0.1 --seed 0 for sk2ml3):
uv run python bench/candidate_waste.py classify $OUT/W-C1-split.lattice.parquet --workload W \
    --out $OUT/waste.jsonl --gpu-result $OUT/W-C1-split_r0.result.json \
    --free-dump $OUT/W-C1-split-free.lattice.parquet
uv run python bench/candidate_waste.py report --raw $OUT/raw.jsonl $OUT/oom2ml3/raw.jsonl --waste $OUT/waste.jsonl
```

## Per-level tables

Generated by `candidate_waste.py report`. "P in fully-P tiles" is over the P
candidates of tiled groups; "P per suffix" is what the removed suffix prune
removes, over all P; "rest" is the level's time outside counting.

### deepk, free

| K | generated | frequent | emitted | P | I | C | P % | I % | P in fully-P tiles % (tiled groups) | P per suffix % | level s | counting s | rest s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 6,216 | 471 | 471 | 0 | 0 | 6,216 | 0.0 | 0.0 | — | — | 0.006 [0.006, 0.006] | 0.005 [0.004, 0.005] | 0.002 [0.002, 0.002] |
| 3 | 9,174 | 901 | 509 | 7,912 | 0 | 1,262 | 86.2 | 0.0 | 27.4 | 34.6 | 0.009 [0.009, 0.009] | 0.005 [0.005, 0.005] | 0.005 [0.004, 0.005] |
| 4 | 2,896 | 644 | 159 | 2,203 | 433 | 260 | 76.1 | 15.0 | 19.3 | 87.5 | 0.009 [0.009, 0.009] | 0.004 [0.004, 0.004] | 0.004 [0.004, 0.004] |
| 5 | 454 | 57 | 6 | 393 | 51 | 10 | 86.6 | 11.2 | 0.0 | 96.9 | 0.008 [0.008, 0.008] | 0.004 [0.004, 0.004] | 0.004 [0.004, 0.004] |
| 6 | 7 | 0 | 0 | 7 | 0 | 0 | 100.0 | 0.0 | — | 100.0 | 0.002 [0.002, 0.002] | 0.000 [0.000, 0.000] | 0.002 [0.002, 0.002] |

Σ K≥3 time split, s, median [min, max] over reps:

| group_build | group_upload | budget | count_percand | count_tiled | count_fused | reduce | filter | decode | sort | free_prune | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 [0.001, 0.001] | 0.004 [0.004, 0.004] | 0.000 [0.000, 0.000] | 0.002 [0.002, 0.002] | 0.012 [0.012, 0.012] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.003 [0.003, 0.003] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.001, 0.001] | 0.005 [0.005, 0.005] |

### deepk, full

| K | generated | frequent | emitted | P | I | C | P % | I % | P in fully-P tiles % (tiled groups) | P per suffix % | level s | counting s | rest s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 6,216 | 471 | 471 | 0 | 0 | 6,216 | 0.0 | 0.0 | — | — | 0.006 [0.006, 0.006] | 0.005 [0.005, 0.005] | 0.001 [0.001, 0.001] |
| 3 | 9,174 | 901 | 901 | 7,912 | 0 | 1,262 | 86.2 | 0.0 | 27.4 | 34.6 | 0.008 [0.007, 0.008] | 0.004 [0.004, 0.005] | 0.003 [0.002, 0.003] |
| 4 | 3,659 | 1,407 | 1,407 | 2,203 | 1,196 | 260 | 60.2 | 32.7 | 19.3 | 87.5 | 0.007 [0.007, 0.008] | 0.005 [0.004, 0.005] | 0.003 [0.002, 0.003] |
| 5 | 2,251 | 1,854 | 1,854 | 393 | 1,848 | 10 | 17.5 | 82.1 | 0.0 | 96.9 | 0.008 [0.007, 0.008] | 0.005 [0.005, 0.005] | 0.003 [0.002, 0.004] |
| 6 | 1,855 | 1,848 | 1,848 | 7 | 1,848 | 0 | 0.4 | 99.6 | — | 100.0 | 0.003 [0.002, 0.004] | 0.001 [0.001, 0.001] | 0.002 [0.001, 0.003] |
| 7 | 1,320 | 1,320 | 1,320 | 0 | 1,320 | 0 | 0.0 | 100.0 | — | — | 0.002 [0.002, 0.003] | 0.001 [0.001, 0.001] | 0.001 [0.001, 0.002] |
| 8 | 660 | 660 | 660 | 0 | 660 | 0 | 0.0 | 100.0 | — | — | 0.002 [0.002, 0.003] | 0.001 [0.001, 0.001] | 0.001 [0.001, 0.002] |
| 9 | 220 | 220 | 220 | 0 | 220 | 0 | 0.0 | 100.0 | — | — | 0.002 [0.001, 0.002] | 0.000 [0.000, 0.000] | 0.001 [0.001, 0.002] |
| 10 | 44 | 44 | 44 | 0 | 44 | 0 | 0.0 | 100.0 | — | — | 0.002 [0.001, 0.002] | 0.000 [0.000, 0.000] | 0.001 [0.001, 0.002] |
| 11 | 4 | 4 | 4 | 0 | 4 | 0 | 0.0 | 100.0 | — | — | 0.002 [0.002, 0.002] | 0.000 [0.000, 0.000] | 0.001 [0.001, 0.002] |

Σ K≥3 time split, s, median [min, max] over reps:

| group_build | group_upload | budget | count_percand | count_tiled | count_fused | reduce | filter | decode | sort | free_prune | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 [0.001, 0.001] | 0.005 [0.004, 0.006] | 0.001 [0.000, 0.001] | 0.005 [0.005, 0.006] | 0.012 [0.012, 0.012] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.003 [0.003, 0.004] | 0.001 [0.001, 0.001] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.006 [0.005, 0.009] |

### dsl, free

| K | generated | frequent | emitted | P | I | C | P % | I % | P in fully-P tiles % (tiled groups) | P per suffix % | level s | counting s | rest s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 3,570 | 1,058 | 1,058 | 0 | 0 | 3,570 | 0.0 | 0.0 | — | — | 0.071 [0.071, 0.071] | 0.067 [0.067, 0.067] | 0.004 [0.004, 0.004] |
| 3 | 15,673 | 6,043 | 6,043 | 5,542 | 0 | 10,131 | 35.4 | 0.0 | 0.0 | 19.5 | 0.090 [0.090, 0.114] | 0.085 [0.085, 0.086] | 0.004 [0.004, 0.028] |
| 4 | 43,877 | 21,430 | 19,327 | 18,775 | 0 | 25,102 | 42.8 | 0.0 | 0.0 | 6.7 | 0.207 [0.206, 0.221] | 0.198 [0.197, 0.199] | 0.008 [0.008, 0.023] |
| 5 | 85,191 | 46,966 | 40,364 | 38,075 | 5,533 | 41,583 | 44.7 | 6.5 | 0.0 | 2.5 | 0.572 [0.560, 0.588] | 0.547 [0.541, 0.551] | 0.021 [0.019, 0.041] |
| 6 | 116,734 | 71,154 | 52,788 | 45,580 | 18,072 | 53,082 | 39.0 | 15.5 | 0.0 | 4.0 | 1.073 [1.061, 1.076] | 1.041 [1.031, 1.043] | 0.031 [0.030, 0.033] |
| 7 | 98,724 | 68,838 | 44,635 | 29,886 | 24,203 | 44,635 | 30.3 | 24.5 | 0.0 | 6.3 | 1.164 [1.143, 1.172] | 1.128 [1.115, 1.134] | 0.036 [0.027, 0.038] |
| 8 | 51,062 | 42,668 | 26,372 | 8,394 | 16,296 | 26,372 | 16.4 | 31.9 | 0.0 | 13.1 | 0.752 [0.737, 0.759] | 0.718 [0.714, 0.727] | 0.025 [0.023, 0.041] |
| 9 | 18,114 | 17,514 | 11,800 | 600 | 5,714 | 11,800 | 3.3 | 31.5 | 0.0 | 64.2 | 0.334 [0.333, 0.337] | 0.320 [0.320, 0.325] | 0.013 [0.012, 0.014] |
| 10 | 4,967 | 4,967 | 2,500 | 0 | 2,467 | 2,500 | 0.0 | 49.7 | — | — | 0.080 [0.076, 0.102] | 0.072 [0.071, 0.076] | 0.004 [0.004, 0.031] |
| 11 | 401 | 401 | 0 | 0 | 401 | 0 | 0.0 | 100.0 | — | — | 0.012 [0.009, 0.013] | 0.010 [0.007, 0.011] | 0.002 [0.002, 0.002] |

Σ K≥3 time split, s, median [min, max] over reps:

| group_build | group_upload | budget | count_percand | count_tiled | count_fused | reduce | filter | decode | sort | free_prune | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.008 [0.007, 0.008] | 0.011 [0.010, 0.011] | 0.001 [0.001, 0.001] | 2.327 [2.303, 2.343] | 1.797 [1.781, 1.803] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.008 [0.008, 0.009] | 0.016 [0.016, 0.016] | 0.049 [0.041, 0.049] | 0.028 [0.022, 0.030] | 0.061 [0.050, 0.062] |

### dsl, full

| K | generated | frequent | emitted | P | I | C | P % | I % | P in fully-P tiles % (tiled groups) | P per suffix % | level s | counting s | rest s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 3,570 | 1,058 | 1,058 | 0 | 0 | 3,570 | 0.0 | 0.0 | — | — | 0.071 [0.071, 0.071] | 0.067 [0.066, 0.067] | 0.004 [0.004, 0.005] |
| 3 | 15,673 | 6,043 | 6,043 | 5,542 | 0 | 10,131 | 35.4 | 0.0 | 0.0 | 19.5 | 0.090 [0.090, 0.104] | 0.086 [0.085, 0.086] | 0.004 [0.004, 0.019] |
| 4 | 43,877 | 21,430 | 21,430 | 18,775 | 0 | 25,102 | 42.8 | 0.0 | 0.0 | 6.7 | 0.206 [0.202, 0.231] | 0.197 [0.197, 0.199] | 0.006 [0.005, 0.034] |
| 5 | 92,935 | 51,243 | 51,243 | 41,542 | 9,810 | 41,583 | 44.7 | 10.6 | 0.0 | 2.5 | 0.593 [0.578, 0.595] | 0.583 [0.569, 0.586] | 0.010 [0.009, 0.010] |
| 6 | 148,891 | 88,374 | 88,374 | 60,517 | 35,292 | 53,082 | 40.6 | 23.7 | 0.0 | 3.6 | 1.296 [1.263, 1.296] | 1.276 [1.245, 1.283] | 0.018 [0.014, 0.020] |
| 7 | 181,999 | 114,509 | 114,509 | 67,490 | 69,874 | 44,635 | 37.1 | 38.4 | 0.0 | 5.0 | 2.021 [1.971, 2.025] | 1.992 [1.944, 2.006] | 0.027 [0.019, 0.029] |
| 8 | 171,479 | 114,174 | 114,174 | 57,305 | 87,802 | 26,372 | 33.4 | 51.2 | 0.0 | 6.3 | 2.251 [2.189, 2.256] | 2.225 [2.160, 2.233] | 0.027 [0.022, 0.028] |
| 9 | 125,178 | 88,698 | 88,698 | 36,480 | 76,898 | 11,800 | 29.1 | 61.4 | 0.0 | 7.6 | 1.814 [1.773, 1.828] | 1.796 [1.749, 1.808] | 0.020 [0.018, 0.024] |
| 10 | 70,690 | 53,910 | 53,910 | 16,780 | 51,410 | 2,500 | 23.7 | 72.7 | 0.0 | 9.4 | 1.157 [1.115, 1.164] | 1.141 [1.094, 1.149] | 0.016 [0.015, 0.021] |
| 11 | 30,802 | 25,554 | 25,554 | 5,248 | 25,554 | 0 | 17.0 | 83.0 | 0.0 | 11.5 | 0.562 [0.546, 0.566] | 0.552 [0.538, 0.559] | 0.009 [0.007, 0.010] |
| 12 | 10,336 | 9,336 | 9,336 | 1,000 | 9,336 | 0 | 9.7 | 90.3 | 0.0 | 13.7 | 0.256 [0.250, 0.257] | 0.252 [0.245, 0.252] | 0.005 [0.005, 0.005] |
| 13 | 2,652 | 2,564 | 2,564 | 88 | 2,564 | 0 | 3.3 | 96.7 | 0.0 | 15.9 | 0.132 [0.130, 0.147] | 0.127 [0.118, 0.129] | 0.003 [0.003, 0.029] |
| 14 | 504 | 504 | 504 | 0 | 504 | 0 | 0.0 | 100.0 | — | — | 0.013 [0.013, 0.015] | 0.011 [0.011, 0.013] | 0.002 [0.002, 0.002] |
| 15 | 64 | 64 | 64 | 0 | 64 | 0 | 0.0 | 100.0 | — | — | 0.006 [0.006, 0.006] | 0.004 [0.004, 0.004] | 0.002 [0.001, 0.002] |
| 16 | 4 | 4 | 4 | 0 | 4 | 0 | 0.0 | 100.0 | — | — | 0.007 [0.007, 0.007] | 0.006 [0.006, 0.006] | 0.001 [0.001, 0.001] |

Σ K≥3 time split, s, median [min, max] over reps:

| group_build | group_upload | budget | count_percand | count_tiled | count_fused | reduce | filter | decode | sort | free_prune | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.037 [0.033, 0.040] | 0.016 [0.015, 0.017] | 0.002 [0.002, 0.029] | 5.983 [5.808, 6.017] | 4.268 [4.158, 4.289] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.015 [0.014, 0.022] | 0.036 [0.030, 0.038] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.045 [0.037, 0.084] |

### oom2ml3, full

| K | generated | frequent | emitted | P | I | C | P % | I % | P in fully-P tiles % (tiled groups) | P per suffix % | level s | counting s | rest s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 449,985,000 | 422,486 | 422,486 | 0 | 0 | 449,985,000 | 0.0 | 0.0 | — | — | 5.474 [5.265, 5.687] | 5.377 [5.138, 5.583] | 0.105 [0.097, 0.127] |
| 3 | 2,081,967,305 | 97,198 | 97,198 | 2,066,149,526 | 0 | 15,817,779 | 99.2 | 0.0 | 97.8 | 18.4 | 20.898 [20.078, 21.892] | 20.803 [19.985, 21.793] | 0.096 [0.094, 0.099] |

Σ K≥3 time split, s, median [min, max] over reps:

| group_build | group_upload | budget | count_percand | count_tiled | count_fused | reduce | filter | decode | sort | free_prune | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.005 [0.005, 0.006] | 0.005 [0.005, 0.005] | 0.001 [0.001, 0.001] | 0.000 [0.000, 0.000] | 20.802 [19.984, 21.792] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.072 [0.071, 0.076] | 0.004 [0.004, 0.004] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.007 [0.007, 0.008] |

### or002, free

| K | generated | frequent | emitted | P | I | C | P % | I % | P in fully-P tiles % (tiled groups) | P per suffix % | level s | counting s | rest s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 3,415,191 | 60,406 | 60,398 | 0 | 0 | 3,415,191 | 0.0 | 0.0 | — | — | 0.014 [0.013, 0.016] | 0.004 [0.004, 0.004] | 0.010 [0.009, 0.012] |
| 3 | 6,615,235 | 136,146 | 135,867 | 3,995,796 | 6 | 2,619,433 | 60.4 | 0.0 | 0.1 | 0.3 | 0.050 [0.049, 0.057] | 0.009 [0.009, 0.009] | 0.041 [0.040, 0.048] |
| 4 | 2,592,919 | 140,880 | 134,080 | 1,935,579 | 722 | 656,618 | 74.6 | 0.0 | 0.1 | 9.5 | 0.064 [0.058, 0.068] | 0.008 [0.008, 0.009] | 0.055 [0.050, 0.060] |
| 5 | 646,170 | 70,623 | 61,108 | 469,704 | 4,460 | 172,006 | 72.7 | 0.7 | 0.0 | 20.3 | 0.038 [0.037, 0.045] | 0.006 [0.006, 0.006] | 0.032 [0.031, 0.038] |
| 6 | 117,570 | 16,796 | 14,339 | 88,013 | 2,013 | 27,544 | 74.9 | 1.7 | 0.2 | 47.8 | 0.016 [0.014, 0.016] | 0.002 [0.002, 0.002] | 0.014 [0.012, 0.014] |
| 7 | 17,225 | 2,340 | 2,067 | 13,749 | 227 | 3,249 | 79.8 | 1.3 | 2.6 | 72.4 | 0.007 [0.006, 0.007] | 0.001 [0.001, 0.001] | 0.006 [0.005, 0.007] |
| 8 | 2,023 | 192 | 162 | 1,766 | 26 | 231 | 87.3 | 1.3 | 0.0 | 90.5 | 0.005 [0.005, 0.005] | 0.001 [0.001, 0.001] | 0.004 [0.004, 0.004] |
| 9 | 98 | 2 | 0 | 94 | 2 | 2 | 95.9 | 2.0 | 0.0 | 100.0 | 0.004 [0.004, 0.004] | 0.001 [0.001, 0.001] | 0.004 [0.003, 0.004] |

Σ K≥3 time split, s, median [min, max] over reps:

| group_build | group_upload | budget | count_percand | count_tiled | count_fused | reduce | filter | decode | sort | free_prune | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.009 [0.009, 0.013] | 0.009 [0.009, 0.011] | 0.000 [0.000, 0.001] | 0.011 [0.011, 0.011] | 0.017 [0.017, 0.017] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.011 [0.010, 0.012] | 0.025 [0.023, 0.026] | 0.048 [0.046, 0.054] | 0.029 [0.024, 0.030] | 0.024 [0.023, 0.028] |

### or002, full

| K | generated | frequent | emitted | P | I | C | P % | I % | P in fully-P tiles % (tiled groups) | P per suffix % | level s | counting s | rest s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 3,415,191 | 60,406 | 60,406 | 0 | 0 | 3,415,191 | 0.0 | 0.0 | — | — | 0.013 [0.012, 0.014] | 0.005 [0.004, 0.005] | 0.009 [0.007, 0.009] |
| 3 | 6,615,929 | 136,149 | 136,149 | 3,996,487 | 9 | 2,619,433 | 60.4 | 0.0 | 0.1 | 0.3 | 0.026 [0.026, 0.027] | 0.009 [0.009, 0.009] | 0.017 [0.017, 0.018] |
| 4 | 2,596,191 | 140,950 | 140,950 | 1,938,781 | 792 | 656,618 | 74.7 | 0.0 | 0.1 | 9.5 | 0.027 [0.026, 0.028] | 0.008 [0.008, 0.008] | 0.019 [0.017, 0.020] |
| 5 | 650,462 | 73,597 | 73,597 | 471,022 | 7,434 | 172,006 | 72.4 | 1.1 | 0.0 | 19.9 | 0.018 [0.017, 0.018] | 0.006 [0.006, 0.006] | 0.012 [0.012, 0.012] |
| 6 | 120,590 | 19,049 | 19,049 | 88,780 | 4,266 | 27,544 | 73.6 | 3.5 | 0.2 | 45.3 | 0.009 [0.008, 0.009] | 0.002 [0.002, 0.002] | 0.007 [0.006, 0.007] |
| 7 | 18,359 | 2,882 | 2,882 | 14,341 | 769 | 3,249 | 78.1 | 4.2 | 2.5 | 71.4 | 0.004 [0.004, 0.005] | 0.001 [0.001, 0.001] | 0.004 [0.004, 0.004] |
| 8 | 2,473 | 291 | 291 | 2,117 | 125 | 231 | 85.6 | 5.1 | 0.0 | 89.5 | 0.004 [0.003, 0.004] | 0.000 [0.000, 0.001] | 0.003 [0.003, 0.003] |
| 9 | 199 | 17 | 17 | 180 | 17 | 2 | 90.5 | 8.5 | 0.0 | 99.4 | 0.003 [0.003, 0.003] | 0.000 [0.000, 0.000] | 0.003 [0.003, 0.003] |
| 10 | 2 | 1 | 1 | 1 | 1 | 0 | 50.0 | 50.0 | — | 100.0 | 0.002 [0.001, 0.002] | 0.000 [0.000, 0.000] | 0.001 [0.001, 0.001] |

Σ K≥3 time split, s, median [min, max] over reps:

| group_build | group_upload | budget | count_percand | count_tiled | count_fused | reduce | filter | decode | sort | free_prune | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.014 [0.014, 0.015] | 0.009 [0.008, 0.009] | 0.001 [0.000, 0.001] | 0.011 [0.011, 0.011] | 0.016 [0.016, 0.017] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.010 [0.010, 0.010] | 0.020 [0.016, 0.020] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.013 [0.013, 0.013] |

### or003, free

| K | generated | frequent | emitted | P | I | C | P % | I % | P in fully-P tiles % (tiled groups) | P per suffix % | level s | counting s | rest s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 2,299,440 | 22,732 | 22,728 | 0 | 0 | 2,299,440 | 0.0 | 0.0 | — | — | 0.009 [0.008, 0.009] | 0.003 [0.003, 0.003] | 0.006 [0.005, 0.006] |
| 3 | 1,465,045 | 23,062 | 23,035 | 938,106 | 0 | 526,939 | 64.0 | 0.0 | 0.0 | 0.7 | 0.012 [0.011, 0.012] | 0.003 [0.003, 0.003] | 0.009 [0.008, 0.009] |
| 4 | 205,597 | 10,752 | 10,588 | 152,262 | 11 | 53,324 | 74.1 | 0.0 | 0.7 | 20.8 | 0.008 [0.007, 0.008] | 0.002 [0.002, 0.002] | 0.006 [0.006, 0.006] |
| 5 | 27,021 | 2,384 | 2,375 | 20,771 | 5 | 6,245 | 76.9 | 0.0 | 0.0 | 45.5 | 0.006 [0.005, 0.007] | 0.001 [0.001, 0.001] | 0.005 [0.004, 0.006] |
| 6 | 3,267 | 402 | 401 | 2,481 | 1 | 785 | 75.9 | 0.0 | 0.0 | 77.8 | 0.004 [0.004, 0.005] | 0.001 [0.001, 0.001] | 0.003 [0.003, 0.004] |
| 7 | 238 | 33 | 32 | 190 | 0 | 48 | 79.8 | 0.0 | 0.0 | 93.2 | 0.004 [0.004, 0.004] | 0.001 [0.000, 0.001] | 0.003 [0.003, 0.004] |
| 8 | 7 | 0 | 0 | 6 | 0 | 1 | 85.7 | 0.0 | — | 100.0 | 0.002 [0.001, 0.002] | 0.000 [0.000, 0.000] | 0.002 [0.001, 0.002] |

Σ K≥3 time split, s, median [min, max] over reps:

| group_build | group_upload | budget | count_percand | count_tiled | count_fused | reduce | filter | decode | sort | free_prune | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.002 [0.002, 0.002] | 0.006 [0.005, 0.007] | 0.000 [0.000, 0.000] | 0.003 [0.002, 0.003] | 0.004 [0.004, 0.004] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.004 [0.003, 0.004] | 0.002 [0.002, 0.002] | 0.004 [0.004, 0.004] | 0.003 [0.002, 0.003] | 0.008 [0.007, 0.009] |

### or003, full

| K | generated | frequent | emitted | P | I | C | P % | I % | P in fully-P tiles % (tiled groups) | P per suffix % | level s | counting s | rest s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 2,299,440 | 22,732 | 22,732 | 0 | 0 | 2,299,440 | 0.0 | 0.0 | — | — | 0.009 [0.008, 0.089] | 0.003 [0.003, 0.003] | 0.005 [0.005, 0.085] |
| 3 | 1,465,045 | 23,062 | 23,062 | 938,106 | 0 | 526,939 | 64.0 | 0.0 | 0.0 | 0.7 | 0.009 [0.006, 0.010] | 0.003 [0.003, 0.003] | 0.006 [0.003, 0.007] |
| 4 | 205,600 | 10,755 | 10,755 | 152,262 | 14 | 53,324 | 74.1 | 0.0 | 0.7 | 20.8 | 0.006 [0.005, 0.006] | 0.002 [0.002, 0.002] | 0.004 [0.003, 0.005] |
| 5 | 27,025 | 2,388 | 2,388 | 20,771 | 9 | 6,245 | 76.9 | 0.0 | 0.0 | 45.5 | 0.004 [0.004, 0.004] | 0.001 [0.001, 0.001] | 0.003 [0.003, 0.004] |
| 6 | 3,267 | 402 | 402 | 2,481 | 1 | 785 | 75.9 | 0.0 | 0.0 | 77.8 | 0.003 [0.003, 0.004] | 0.000 [0.000, 0.001] | 0.003 [0.002, 0.003] |
| 7 | 238 | 33 | 33 | 190 | 0 | 48 | 79.8 | 0.0 | 0.0 | 93.2 | 0.003 [0.002, 0.003] | 0.000 [0.000, 0.000] | 0.002 [0.002, 0.003] |
| 8 | 7 | 0 | 0 | 6 | 0 | 1 | 85.7 | 0.0 | — | 100.0 | 0.001 [0.001, 0.001] | 0.000 [0.000, 0.000] | 0.001 [0.001, 0.001] |

Σ K≥3 time split, s, median [min, max] over reps:

| group_build | group_upload | budget | count_percand | count_tiled | count_fused | reduce | filter | decode | sort | free_prune | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.003 [0.002, 0.003] | 0.005 [0.005, 0.005] | 0.000 [0.000, 0.000] | 0.002 [0.002, 0.002] | 0.004 [0.004, 0.004] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.004 [0.002, 0.004] | 0.002 [0.001, 0.002] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.006 [0.005, 0.006] |

### sk2ml3, full (K=3 sampled)

| K | generated | frequent | emitted | P | I | C | P % | I % | P in fully-P tiles % (tiled groups) | P per suffix % | level s | counting s | rest s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 612,482,500 | 1,660,332 | 1,660,332 | 0 | 0 | 612,482,500 | 0.0 | 0.0 | — | — | 29.895 | 29.565 | 0.330 |
| 3 | 11,796,796,856 | 1,310,438 | 1,310,438 | 1,032,517,384 | 0 | 14,108,309 | 98.7 | 0.0 | 97.8 | 3.9 | 493.437 | 492.930 | 0.507 |

Σ K≥3 time split, s, median [min, max] over reps:

| group_build | group_upload | budget | count_percand | count_tiled | count_fused | reduce | filter | decode | sort | free_prune | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.020 | 0.012 | 0.001 | 0.001 | 492.929 | 0.000 | 0.000 | 0.381 | 0.058 | 0.000 | 0.000 | 0.035 |

### skew, free

| K | generated | frequent | emitted | P | I | C | P % | I % | P in fully-P tiles % (tiled groups) | P per suffix % | level s | counting s | rest s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 6,903 | 717 | 717 | 0 | 0 | 6,903 | 0.0 | 0.0 | — | — | 0.006 [0.006, 0.006] | 0.005 [0.005, 0.005] | 0.001 [0.001, 0.002] |
| 3 | 20,123 | 1,928 | 1,905 | 16,905 | 0 | 3,218 | 84.0 | 0.0 | 33.8 | 25.5 | 0.008 [0.008, 0.040] | 0.004 [0.004, 0.004] | 0.004 [0.004, 0.035] |
| 4 | 27,454 | 2,893 | 2,740 | 23,962 | 59 | 3,433 | 87.3 | 0.2 | 30.9 | 69.9 | 0.037 [0.010, 0.070] | 0.006 [0.005, 0.006] | 0.032 [0.005, 0.065] |
| 5 | 26,807 | 2,478 | 2,291 | 24,126 | 187 | 2,494 | 90.0 | 0.7 | 18.9 | 81.5 | 0.010 [0.010, 0.012] | 0.007 [0.007, 0.007] | 0.004 [0.003, 0.005] |
| 6 | 16,351 | 1,178 | 1,178 | 15,127 | 0 | 1,224 | 92.5 | 0.0 | 20.2 | 88.7 | 0.009 [0.008, 0.010] | 0.005 [0.005, 0.005] | 0.003 [0.003, 0.004] |
| 7 | 5,881 | 315 | 315 | 5,561 | 0 | 320 | 94.6 | 0.0 | 18.5 | 93.3 | 0.007 [0.007, 0.008] | 0.004 [0.004, 0.004] | 0.003 [0.003, 0.004] |
| 8 | 1,035 | 33 | 33 | 1,002 | 0 | 33 | 96.8 | 0.0 | 32.8 | 97.6 | 0.006 [0.006, 0.007] | 0.003 [0.003, 0.003] | 0.003 [0.002, 0.004] |
| 9 | 58 | 0 | 0 | 58 | 0 | 0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.006 [0.005, 0.006] | 0.003 [0.003, 0.003] | 0.003 [0.002, 0.003] |

Σ K≥3 time split, s, median [min, max] over reps:

| group_build | group_upload | budget | count_percand | count_tiled | count_fused | reduce | filter | decode | sort | free_prune | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.001 [0.001, 0.002] | 0.008 [0.006, 0.034] | 0.028 [0.001, 0.033] | 0.004 [0.004, 0.004] | 0.028 [0.028, 0.028] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.004 [0.003, 0.004] | 0.001 [0.001, 0.001] | 0.001 [0.001, 0.001] | 0.001 [0.001, 0.001] | 0.010 [0.008, 0.040] |

### skew, full

| K | generated | frequent | emitted | P | I | C | P % | I % | P in fully-P tiles % (tiled groups) | P per suffix % | level s | counting s | rest s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 6,903 | 717 | 717 | 0 | 0 | 6,903 | 0.0 | 0.0 | — | — | 0.006 [0.006, 0.028] | 0.005 [0.005, 0.005] | 0.001 [0.001, 0.023] |
| 3 | 20,123 | 1,928 | 1,928 | 16,905 | 0 | 3,218 | 84.0 | 0.0 | 33.8 | 25.5 | 0.007 [0.007, 0.008] | 0.004 [0.004, 0.004] | 0.003 [0.003, 0.003] |
| 4 | 27,493 | 2,932 | 2,932 | 23,962 | 98 | 3,433 | 87.2 | 0.4 | 30.9 | 69.9 | 0.008 [0.008, 0.039] | 0.005 [0.005, 0.005] | 0.003 [0.003, 0.034] |
| 5 | 27,012 | 2,683 | 2,683 | 24,126 | 392 | 2,494 | 89.3 | 1.5 | 18.9 | 81.5 | 0.010 [0.010, 0.038] | 0.007 [0.007, 0.007] | 0.003 [0.003, 0.031] |
| 6 | 16,631 | 1,458 | 1,458 | 15,127 | 280 | 1,224 | 91.0 | 1.7 | 20.2 | 88.7 | 0.009 [0.008, 0.010] | 0.006 [0.005, 0.006] | 0.003 [0.003, 0.004] |
| 7 | 6,009 | 443 | 443 | 5,561 | 128 | 320 | 92.5 | 2.1 | 18.5 | 93.3 | 0.007 [0.007, 0.008] | 0.004 [0.004, 0.004] | 0.003 [0.003, 0.004] |
| 8 | 1,069 | 67 | 67 | 1,002 | 34 | 33 | 93.7 | 3.2 | 32.8 | 97.6 | 0.006 [0.006, 0.007] | 0.003 [0.003, 0.003] | 0.003 [0.002, 0.004] |
| 9 | 62 | 4 | 4 | 58 | 4 | 0 | 93.5 | 6.5 | 100.0 | 100.0 | 0.006 [0.005, 0.006] | 0.003 [0.003, 0.003] | 0.003 [0.002, 0.003] |

Σ K≥3 time split, s, median [min, max] over reps:

| group_build | group_upload | budget | count_percand | count_tiled | count_fused | reduce | filter | decode | sort | free_prune | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.002 [0.002, 0.002] | 0.006 [0.006, 0.008] | 0.000 [0.000, 0.029] | 0.005 [0.004, 0.005] | 0.028 [0.028, 0.028] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.004 [0.004, 0.005] | 0.001 [0.001, 0.001] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.007 [0.007, 0.040] |

### smoke, free

| K | generated | frequent | emitted | P | I | C | P % | I % | P in fully-P tiles % (tiled groups) | P per suffix % | level s | counting s | rest s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 6,903 | 290 | 290 | 0 | 0 | 6,903 | 0.0 | 0.0 | — | — | 0.023 [0.002, 0.056] | 0.001 [0.001, 0.029] | 0.022 [0.001, 0.027] |
| 3 | 7,045 | 202 | 175 | 6,485 | 0 | 560 | 92.1 | 0.0 | 21.8 | 50.8 | 0.003 [0.003, 0.003] | 0.001 [0.001, 0.001] | 0.002 [0.002, 0.002] |
| 4 | 1,226 | 49 | 18 | 1,145 | 28 | 53 | 93.4 | 2.3 | 6.7 | 92.3 | 0.003 [0.003, 0.003] | 0.001 [0.001, 0.001] | 0.002 [0.002, 0.002] |
| 5 | 48 | 0 | 0 | 48 | 0 | 0 | 100.0 | 0.0 | — | 100.0 | 0.001 [0.001, 0.001] | 0.000 [0.000, 0.000] | 0.001 [0.001, 0.001] |

Σ K≥3 time split, s, median [min, max] over reps:

| group_build | group_upload | budget | count_percand | count_tiled | count_fused | reduce | filter | decode | sort | free_prune | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.000 [0.000, 0.000] | 0.001 [0.001, 0.001] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] | 0.001 [0.001, 0.001] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.001, 0.001] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.002 [0.002, 0.002] |

### smoke, full

| K | generated | frequent | emitted | P | I | C | P % | I % | P in fully-P tiles % (tiled groups) | P per suffix % | level s | counting s | rest s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 6,903 | 290 | 290 | 0 | 0 | 6,903 | 0.0 | 0.0 | — | — | 0.025 [0.024, 0.054] | 0.001 [0.001, 0.029] | 0.023 [0.023, 0.026] |
| 3 | 7,045 | 202 | 202 | 6,485 | 0 | 560 | 92.1 | 0.0 | 21.8 | 50.8 | 0.003 [0.002, 0.003] | 0.001 [0.001, 0.001] | 0.002 [0.002, 0.002] |
| 4 | 1,240 | 63 | 63 | 1,145 | 42 | 53 | 92.3 | 3.4 | 6.7 | 92.3 | 0.003 [0.003, 0.003] | 0.001 [0.001, 0.001] | 0.002 [0.002, 0.002] |
| 5 | 66 | 18 | 18 | 48 | 18 | 0 | 72.7 | 27.3 | — | 100.0 | 0.001 [0.001, 0.001] | 0.000 [0.000, 0.000] | 0.001 [0.001, 0.001] |
| 6 | 3 | 3 | 3 | 0 | 3 | 0 | 0.0 | 100.0 | — | — | 0.001 [0.001, 0.001] | 0.000 [0.000, 0.000] | 0.001 [0.001, 0.001] |

Σ K≥3 time split, s, median [min, max] over reps:

| group_build | group_upload | budget | count_percand | count_tiled | count_fused | reduce | filter | decode | sort | free_prune | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.000 [0.000, 0.000] | 0.002 [0.001, 0.002] | 0.000 [0.000, 0.000] | 0.001 [0.001, 0.001] | 0.001 [0.001, 0.001] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.002 [0.001, 0.002] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.003 [0.002, 0.003] |

