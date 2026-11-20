# Route consolidation, round 2: final report

Branch `refactor/route-consolidation`, reviewed from `f6c3d36` to `2328b38`, with the review's fixes in
`be8a09e`. Inputs are Phase 4 (`raw.jsonl` here, 2x RTX 3060 12 GB, at `af85dad`), its comparison with Phase 2
(`COMPARE.md`), and an adversarial review of the branch diff.

## In short

The consolidation does what it set out to do. Every mining path that lost its measurement is gone, and the
miner still finds exactly the same itemsets with exactly the same counts:
- all 64 configs that ran in both phases mine the same lattice, i.e. the same set of frequent itemsets and
  counts (`COMPARE.md`);
- the tier-equivalence chain holds;
- the review's differential runs found no wrong count anywhere: 20 item dtypes on 7 routes, SON against
  efficient-apriori in 300 runs, the GPU kernels against brute force on 320 shapes.

Nothing got slower beyond noise.

The review did find three bugs on the branch, and they are fixed in `be8a09e`, each with a test that fails
without the fix:
- The lex row order's room check let a near-full card through, and the sort then crashed instead of falling back.
- SON on the GPU listed the items of an Enum column in string order instead of category order.
- `apriori(bitvecs=..., sparse=...)` did not raise.

One cost remains for the owner to weigh: the CPU route peaks about 1.7 GB higher in host memory on the
20M-row dsl workload. The cause is that K=2 (counting every pair of items) now always runs on the Gram matrix,
which holds a second, transposed copy of the data while it counts.

## Findings

Tags: (must-fix) wrong or crashing behaviour; (should) a gap worth closing before the merge, or a decision
for the owner; (nice) cosmetic or historical.

1. **(must-fix, fixed in `be8a09e`) G2: the lex sort could run out of device memory after its room check
   passed.**
   - What it means: lex sorts each GPU's rows by their most frequent items before the bitvectors are built
     (it makes later counting faster). Before sorting, the code checks for `LEX_BYTES_PER_ROW` bytes of free
     VRAM per row and keeps the original row order otherwise. The check assumed 56 B per row, but the sort
     peaks at 64.3 B per row. That figure was measured with a CuPy memory hook at 1M, 4M and 20M rows, so it
     does not depend on size.
   - Where the 56 came from: the G2 measurement (`../2026-10-27-routes-gpu/FINDINGS.md`, the G2 keep row)
     took nvidia-smi figures, which are MiB, and divided them by 10^6. 1,090 MiB over 20M rows is 57.1 B per
     row, already above 56.
   - Repro: 2M rows, 100 items, a 170 MiB pool limit. Lex fails with `MemoryError: std::bad_alloc` from
     thrust's sort, while the kept order mines 5,050 itemsets. On a 20M-row shard the window is about
     160 MB of free memory.
   - Fix (`src/et_miner/gpu/input_layer.py`):
     - `LEX_BYTES_PER_ROW = 65`.
     - A `MemoryError` from the sort now falls back to the kept order. This covers both thrust's plain
       `MemoryError` and CuPy's `OutOfMemoryError`.
   - Tests (`tests/test_row_order.py`):
     - the fallback on either error;
     - the hook-measured peak stays within the constant;
     - the 170 MiB repro mines the CPU route's counts.
   - Follow-up: dslxl (60M rows) now needs 3.9 GB free for lex instead of 3.4 GB. Its sort did fit in
     Phase 4, so free memory was at least 3.86 GB. Rerun `dslxl-C1-default` on the Phase 4 box to confirm it
     still takes lex, i.e. no "does not fit" line in its log.
2. **(should, fixed in `be8a09e`) S1/S2: SON on the GPU emitted Enum items in string order.**
   - An Enum column with categories `["z", "a"]` mined `["a", "z"]` on SON GPU and `["z", "a"]` everywhere
     else (CPU, in-core GPU, SON CPU, and SON GPU at `f6c3d36`). Sets and counts were equal.
   - Cause: `_local_levels_gpu` sorted the chunk's items after the route had emitted them as String.
   - Fix: it casts them back to the column's own dtype first (`src/et_miner/streaming/son.py`).
   - Test: `tests/test_son_gpu.py` gains an Enum leg whose categories run against the strings' order.
   - The 20-dtype differential now matches the CPU route's item order on every route.
3. **(should, fixed in `be8a09e`) C4: `apriori(bitvecs=..., sparse=...)` did not raise.**
   - The CHANGELOG promises a raise for any `sparse=` but `None`. The check ran after the bitvecs route had
     already returned.
   - It now runs before the routing (`src/et_miner/core/apriori.py`).
   - Test: `tests/test_cpu_miner.py::test_sparse_raises_before_the_bitvecs_route`.
4. **(should, owner decision) C1 raised the CPU route's host-memory peak by about 1.7 GB on dsl.**
   - Measured in Phase 4: dsl-free-F1 went from 5,138-5,330 to 6,935-6,973 MB RSS, and dsl-infer-F1 from
     5,226-5,491 to 7,020-7,068 MB. RSS is the memory the process actually holds in RAM.
   - Attribution. C1 made the scipy Gram (`count_pairs`) the CPU route's only K=2 counter. With a thread
     pool it holds an int32 data array (4 B per CSR entry) and a transposed copy of the CSR (8 B per entry)
     for the whole level. Phase 2's rule (`_k2_on_bitvecs`) chose the bitvector counter on dsl's shape, and
     that one needs only the column bitvectors (85 columns x rows/8).
   - Measured with `count_pairs` against the removed `count_pairs_bitvec`, on dsl's shape at 1/5 and 2/5 of
     its rows, 17 workers. dsl itself could not be generated on the review box (30 GiB):

     | rows | CSR entries | Gram K=2 extra RSS | bitvector K=2 extra RSS |
     |---|---|---|---|
     | 4M | 44.8M | 536-557 MiB (12.5 B per entry) | 116-141 MiB |
     | 8M | 89.6M | 1,028 MiB (12.0 B per entry) | 156 MiB |

   - Scaled to dsl's 20M rows (about 224M entries), that is about 2.6 GB at K=2 against about 0.3 GB.
     That K=2 step is now the run's peak; in Phase 2 a later step was.
   - Side effect on time: the K=3 level went from 4.1 to 12.1 s. In Phase 2 the bitvectors built for K=2 were
     reused at K=3; now K=3 builds them. K=2 got faster (10.5 to 8.3 s, dsl-free-F1 rep 0), and the run's
     wall time is unchanged (260-281 s against 267-273 s).
   - Options:
     - (a) accept it: the CPU route is not dsl's route, since the GPU mines it in 4 s;
     - (b) cut the transposed copy, e.g. Gram blocks from column slices of one CSC;
     - (c) restore the bitvector K=2 for shapes where its bitvectors are much smaller than the CSR.
5. **(should) `applications/alphafold/deploy/run_mining.sh:204,219` pass `sparse_from_k=4`**, which raises
   since G6. Out of the branch's scope; not edited.
6. **(should, owner decision) Removed environment knobs are silently ignored.**
   - The knobs are `ET_MINER_ROW_ORDER`, `ET_MINER_SON_GPU`, `ET_MINER_LEAVE_OUT` and `ET_APRIORI_SPARSE`.
   - `_env.reject_removed_knobs` says "Raise for a knob that no longer exists rather than ignore it", and it
     does raise for `ET_MINER_INPUT_LAYER=coded/rebuilt`, `ET_MINER_ESCO_MATERIALIZE` and others.
   - Leaving out the guards followed the G2 precedent on purpose, but `ET_MINER_ROW_ORDER` was user-facing.
   - A rerun of the M5b keep/lex arms in `bench/consolidation_matrix.py:635` now mines lex twice, without
     an error.
7. **(should, known) 2-GPU `bitvecs=` on sk2ml3 raises `MemoryError` upfront.** The input is 8.75 GB on 12 GB
   cards, and device 0 must hold the caller's array beside its own shard. This clear early raise is the
   intended behaviour of `2aa6b72`; Phase 2 failed with an `OutOfMemoryError` mid-run.
8. **(should) `apriori_from_csr` at min_count 0 emits every candidate with count 0.** The CPU route emits
   only itemsets that occur.
9. **(should) The CPU route takes more than 900 s on dsl.** For comparison: the Rust miner takes 225 s, and
   the CPU route takes 263-270 s with free sets or count inference. Rust on deepk20 is slow but not hung
   (max_length 7/9/11: 16/197/878 s).
10. **(nice) G1: a `pl.Array(pl.Int64, 2)` item column on `use_gpu=True` now fails with a raw Polars error.**
    The error is `InvalidOperationError: list.eval operation not supported for dtype array[i64, 2]`, raised
    from the coded layer at `core/apriori.py:672`. At `f6c3d36` it mined 3 itemsets. The CPU route refuses
    Array too, so the routes now agree, but the change is undocumented and the error is unclear.
11. **(nice) SON on the GPU keeps 0.0 where the CPU route keeps -0.0, when both occur in the data.** The
    merged count is the same (319 in the review's probe), and the result depends on chunk order.
12. **(nice) The coded layer codes the whole column before its 2\*\*22 distinct-items check**
    (`core/apriori.py:672-673`). Above the limit, an Int32 copy is built and thrown away before the host CSR
    takes over. The result is correct.
13. **(nice) Bench scripts that now fail, and stale historical tools.**
    - Now raise: `bench/routes/count_api.py:198-199` (`sparse=`), and the `prune_apriori=False` and
      `sparse_from_k` arms in `bench/consolidation_matrix.py`.
    - Stale: `bench/cpu/l3_crossover.py`, `son_stakes.py`, `sparse_non_pairs.py`, `bench/repro/d04*`.
14. **(nice) SON on one GPU peaks lower in Phase 4 than in Phase 2** (dsl 725 against 1,237 MB, dslxl 1,889
    against 3,435 MB): the VRAM rise S1 brought, which the owner accepted, is smaller now.
15. **(nice) `bench/routes_matrix.py:259` keys `SUBSECOND` on the workload name, not on a measured time.**
    Separately, `cargo clippy` (2 errors) and `cargo fmt` (28 diffs) drift, as on `f6c3d36`.

## Verification

At `be8a09e`, on the review box (2x RTX 3060 12 GB, 11.5 CPUs, 30 GiB):

- `uv run ruff check src tests bench`: clean.
- `uv run python bench/selfcheck.py`: READY (11 kernels, 2 devices, Rust extension present).
- `uv run pytest tests/test_tier_equivalence.py -q`: 75 passed.
- `uv run pytest -q -m "not slow"`: 1,499 passed, 2 skipped (1,490 before the fixes; the 9 new tests pass).
- not-slow with CuPy unimportable: 836 passed, 516 skipped.
- `cargo test --release`: 28 passed. `cargo clippy -- -D warnings` fails with the same 2 errors as on
  `f6c3d36` (`div_ceil`, a complex type) and `cargo fmt --check` shows 28 diffs (32 on `f6c3d36`):
  pre-existing drift, no Rust changed in `be8a09e`.

The adversarial review ran at `2328b38`, in its own venv with the `gpu` extra and the Rust extension:
- tier 75 passed; not-slow 1,490 passed, 2 skipped; slow 19 passed, 4 skipped (no `online_retail_ii`);
  ruff clean; `cargo test` 28 passed;
- `test_large_presets_match_the_host_path[deep_sparse_large-*]` could not run on the 30 GiB box: killed
  with exit 137, twice;
- the differentials listed above, plus edge cases (0 rows, all-null, empty lists, 1 row, 1 frequent item,
  `n_gpus` 8/0/-1 on 2 GPUs), `bitvecs=` on 36 shapes with the caller's array bit-identical afterwards,
  the removed parameters on every entry point, and the host CSR above 2\*\*22 distinct items.

## Not done

- `/council review`: the skill cannot be started by the model. The owner runs it.
- `deep_sparse_large` cannot be generated on the review box: the generator is killed at the 30 GiB limit.
  That is why finding 4 is measured on dsl's shape at 4M and 8M rows.
