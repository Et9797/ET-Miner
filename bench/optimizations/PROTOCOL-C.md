# Route-optimizations measurement protocol (phase C: O6)

Fixed before the first timed run of phase C of `bench/optimizations/SPEC.md`.
Phase C is O6, the group kernel (SPEC, O6 amendment); O7 is dropped (SPEC, O7
amendment). Three steps: a kernel sweep sets the dispatch (rule 4), a campaign
decides, and a final check confirms the decided tree. The sweep is
`bench/group_crossover.py` (`kernel_crossover.py`'s synthetic groups, with the
group kernel and this grid); the configs come from
`bench/consolidation_matrix.py::build_o6_matrix` and `build_o6_final` and run
with `bench/runner.py --mode o6` and `--mode o6-final` (built after this
protocol is approved).

## What is compared

The row-split miner (route C, transactions input) at the defaults of main
`63a59e2`: the K=2 r dispatch, the compacted reduce, `prune_apriori=True`,
`use_generator_pruning=False`, and the ESCO fit check. O6 changes only which
kernel counts a prefix group of a dense K≥3 level:

| arm | `ET_MINER_SMALL_GROUP_KERNEL` | groups below the tiled crossover | tiled from |
|---|---|---|---|
| `base` | `percand` | per-candidate kernel | today's `TILED_MIN_GROUP_PAIRS` |
| `group` | unset | the sweep's dispatch: per-candidate below `GROUP_MIN_PAIRS[k]` (if the sweep finds such a floor), the group kernel from there | the sweep's crossover, and every group of more than 64 suffixes |

- **The group kernel** (`count_group_pairs` in `_src/group_pairs.cu`): one block
  of 256 threads per prefix group of at most 64 suffixes. It classifies the
  group's pairs with the subset test (`_subset_index.cu`), emits the inferred
  ones, and keeps the pairs to count in a list in shared memory. Then, one tile
  of 32 words at a time, it stages the prefix AND and the suffix rows in shared
  memory and counts only the listed pairs. Static shared memory is about 30 KB.
- **The entries it writes** are the ones the per-candidate kernel writes:
  skipped pairs are left unwritten; counted and inferred pairs are written on
  every GPU, with the inferred count on the GPU that writes them and 0 on the
  others. So the compacted reduce and the chunk loop apply unchanged. Its chunks
  are group-aligned, as for the tiled kernel.
- **`ET_MINER_SMALL_GROUP_KERNEL=group`** pins the group kernel for every group
  below the tiled crossover (no per-candidate floor). With
  `ET_MINER_TILED_MIN_GROUP_PAIRS` set high, every group of at most 64 suffixes
  goes to the group kernel. The tier legs use this.
- **Unchanged:** the K=2 kernels, ESCO levels, and the one-GPU fused path for
  groups beyond one chunk.

Both arms mine the same itemsets with the same counts.

## Box and budget

2× RTX A4000 16 GB (sm_86, `NCCL_P2P_DISABLE=1`), Ryzen 5 5600X 6C/12T, 46 GB
RAM: phase A's and B's box. Budget ≤ 1.5 GPU-hours (process wall × devices the
config uses) for the sweep, the campaign and the final check together. The
campaign runs with `--max-gpu-hours` set to what is left, and the owner is asked
before more is spent. Estimate ≈ 0.8:

- sweep ≈ 0.22 (at 312,500 words: about 45 s per K for the tiled kernel, at
  most as much for the group kernel, and 18–48 s for the per-candidate kernel,
  4 launches each; a tenth of that at 31,250 words);
- campaign ≈ 0.47 (about 560 s of process wall per rep from phase A's rows);
- final check ≈ 0.08.

Every config pins `POLARS_MAX_THREADS`, `RAYON_NUM_THREADS`, `MKL_NUM_THREADS`
and `OMP_NUM_THREADS` to 6, restricts one-GPU configs to device 0 with
`ET_MINER_DISABLE_NCCL=1`, and records each level's time split
(`bench/level_split.py`, with a new phase `count_group` for the group kernel's
launches). Nothing else runs on the box during the sweep, the campaign or the
final check.

## Step 1: kernel sweep (the rule-4 input; counted in the budget)

- **Points.** Synthetic prefix groups as in the consolidation sweep:
  correlated random bitvecs, `m` suffixes per group. Every pair is counted (no
  subset index). Each point has 1,000 groups, so every kernel launches at least
  1,000 blocks. The consolidation sweep's fixed 200,000 candidates gave 99
  blocks at m = 64, under one wave on 48 SMs.
- **Grid.** K ∈ {3, 4, 5, 6, 8}; words ∈ {31,250, 312,500};
  m ∈ {2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64}, the consolidation
  grid up to the group kernel's cap.
- **Per-candidate kernel.** It is timed up to m = 24, and beyond that only
  while it beats the group kernel. The tiled kernel already beats it from
  m ≈ 16 at K=3 and m ≈ 7 at K=8 (`TILED_MIN_GROUP_PAIRS`), and at m = 64 it
  would take about 29 s per launch at K=8.
- **Timing.** One process per (K, words), with a warm-up. Then 3 interleaved
  reps of each kernel per point. A point's time is each kernel's median.
- **The table** (`row_split_chunks.py`), per K at 312,500 words (dsl's row count):
  - `TILED_MIN_GROUP_PAIRS[k]` is where t_group / t_tiled crosses 1: the
    geometric mean of the two adjacent pair counts C(m, 2). If the group kernel
    is faster up to m = 64, the cap decides alone.
  - `GROUP_MIN_PAIRS[k]` is where t_percand / t_group crosses 1, found the same
    way. If the group kernel is faster at every m, it is 0.
  - K=7 takes the geometric mean of its neighbours; K ≥ 9 takes the K=8 value,
    as today.
- **Readings.**
  1. At 31,250 words every crossover must lie within one grid step of the
     312,500-word value. Otherwise the row count is a dispatch fact the table
     does not have: (must-fix), and the campaign waits for the owner.
  2. A line that crosses 1 more than once: the table uses the first crossing,
     and the report records the others (should).
  3. If the group kernel is slower than both other kernels at every point of
     every K, O6 ends after the sweep: no campaign, and the report records the
     sweep.
- **Freeze.** The table goes into the tree, and the campaign's tree is frozen
  from its first timed run to its last (the runner stamps every row with the
  tree's digest). The sweep's rows carry their own digest.

The sweep counts every pair, but the real levels skip some (35–45 % of dsl's
big levels). That moves the group kernel against both neighbours in opposite
directions. The campaign checks the dispatch on the real levels, as phase A's
campaign checked r*.

## Step 2: campaign

| id | dataset | min_support | max_length | options | 1 GPU | 2 GPUs |
|---|---|---|---|---|---|---|
| smoke, deepk, skew, or003, or002 | as in the consolidation protocol | | | | base, group | — |
| dsl | deep_sparse_large (20M) | 0.015 | — | | base, group | base, group |
| dsl-infer | as dsl | | | `use_generator_pruning=True` | base, group | base, group |
| dsl-free | as dsl | | | `prune_equal_support=True` | base, group | — |
| oom2ml3 | oom_regression (500K) | 0.00003 | 3 | | base, group | base, group |
| sk2ml3 | stress_k2 (2M) | 0.000015 | 3 | | base, group | base, group |

- **Stakes.** dsl, dsl-infer and dsl-free carry O6's stake. The rest check that
  the dispatch loses nothing: short levels, or K=3 explosions where almost
  every group stays tiled.
- **Order.** Fresh process per config with the runner's warm-up; 3 reps,
  rep-major. Within a rep: the short configs, then dsl, dsl-infer, dsl-free,
  oom2ml3, sk2ml3. Cells are median [min, max].

### Metric

`wall_s` of the timed call. The K≥3 time split (`count_percand`, `count_group`,
`count_tiled`) is recorded as evidence, never as the deciding metric.

### Decision rule (the consolidation protocol's rules 1, 2, 4, 5, applied mechanically)

1. **Signature mismatch or logged fallback** disqualifies the arm in that
   regime: (must-fix), and nothing is decided on that point until it is fixed.
2. **Winning a regime**: an arm's median is ≥ 10 % below the other arm's, the
   two [min, max] ranges do not overlap, and it saves ≥ 1 s absolute.
4. **A partial winner** stays only if a dispatch rule selects it from facts
   known before the step it changes runs.
5. **Ties everywhere**: keep whatever serves the most routes with the least
   code; break remaining ties on peak VRAM.

A regime is one row of the table at one GPU count. A config that errors or
times out cannot win.

**DP-O6: `group` against `base`, in every regime.**

- `group` wins a regime and loses none: the sweep's dispatch stays the default,
  and `ET_MINER_SMALL_GROUP_KERNEL=percand` stays as the pin.
- `base` wins a regime: the dispatch misclassifies groups there. That is
  (must-fix), and the decision waits for it. The owner chooses between a fix
  with a rerun of that regime and rule 4 on the facts the dispatch has (k and
  the pairs per group).
- Ties everywhere: rule 5 removes the group kernel and its knob, and today's
  dispatch stays.

## Final check (after the decision is implemented; counted in the budget; not judged)

One rep of every regime with the knobs unset (the decided defaults). The
signatures must match the campaign's; a mismatch is (must-fix).

## Correctness gate (before the sweep, and on the final tree)

- `uv run ruff check src tests bench`.
- `uv run pytest -q -m "not slow"` and `-m "gpu and slow"`, both with the four
  known `test_smoke_correctness` `test_support_001`/`0001` ids deselected.
- `uv run pytest tests/test_tier_equivalence.py`, with new legs, each checking
  with a spy that the group kernel ran:
  - row-split 1 GPU, group kernel pinned: plain, forced chunks, count
    inference, without the subset test;
  - row-split 2 GPUs, group kernel pinned: plain, forced chunks, count
    inference;
  - row-split 1 GPU at the default dispatch.

  The chain in `CLAUDE.md` gets the same legs (the only `CLAUDE.md` edit).
- `tests/test_free_set_semantics.py` with the group kernel pinned.
- `uv run pytest -q -m "gpu and multigpu"`.
- `uv run python bench/selfcheck.py`: the kernel is registered in
  `loader._KERNEL_FILES` and launched with a known answer.
