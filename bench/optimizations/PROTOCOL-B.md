# Route-optimizations measurement protocol (phase B: O5)

Fixed before the first timed run of phase B of `bench/optimizations/SPEC.md`.
Phase B is O5; O4 is deferred (SPEC, O4 amendment). Step 1, the calibration,
asks whether ESCO wins where the model says it should. Step 2 (cost model,
sweep, campaign) is added here as an amendment after step 1, before its first
timed run, and only if step 1 allows it. The configs come from
`bench/consolidation_matrix.py::build_o5_calibration_matrix` and run with
`bench/runner.py --mode o5-calibration` (built after this protocol is approved).

## What is compared

The row-split miner (route C, transactions input) at the defaults of main
`a58dc1d`: the K=2 r dispatch, the compacted reduce, `prune_apriori=True`,
`use_generator_pruning=False`, and the ESCO fit check. Two arms:

| arm | `sparse_from_k` | K=3 |
|---|---|---|
| `dense` | `None` (the default) | today's dense kernels on the bitvecs |
| `esco` | `3` | the frequent pairs' tidsets (`convert_shards_to_csr`), then the CSR kernel |

K=1 and K=2 are the same in both arms. On these workloads `"auto"` also
converts at K=3 (the mean K=2 count is far below N/32), so it is not a separate
arm. Both arms mine the same itemsets with the same counts.

## Box and budget

2× RTX A4000 16 GB (sm_86, `NCCL_P2P_DISABLE=1`), Ryzen 5 5600X 6C/12T, 46 GB
RAM: phase A's box. Step 1 budget ≤ 0.2 GPU-hours (process wall × devices the
config uses), enforced with `--max-gpu-hours 0.2`. Worst case with every ESCO
config at its cap: ≈ 0.15 (dense ≈ 0.025 from phase A's final check).

Every config pins `POLARS_MAX_THREADS`, `RAYON_NUM_THREADS`, `MKL_NUM_THREADS`
and `OMP_NUM_THREADS` to 6, restricts one-GPU configs to device 0 with
`ET_MINER_DISABLE_NCCL=1`, and records each level's time split
(`bench/level_split.py`, with a new phase `count_csr` for the CSR kernel
launches; `transition` already times the conversion). Nothing else runs on the
box during the calibration. Tracked files are frozen from the first run to the
last (the runner stamps every row with the tree's digest).

## Step 1: calibration (counted in the budget; not judged)

| id | dataset | min_support | max_length | 1 GPU | 2 GPUs | ESCO cap |
|---|---|---|---|---|---|---|
| oom2ml3 | oom_regression (500K) | 0.00003 | 3 | dense, esco | dense, esco | 30 s |
| sk2ml3 | stress_k2 (2M) | 0.000015 | 3 | dense, esco | dense, esco | 120 s |

- One rep, fresh process per config with the runner's warm-up, in this order:
  oom2ml3 before sk2ml3; within each, 1 GPU before 2; dense before esco.
- Dense configs are capped at 600 s. An ESCO config that hits its cap loses,
  and its time is recorded as the cap. The caps are 4–5× (sk2ml3) and 6–8×
  (oom2ml3) the dense process wall in phase A's final check.
- Every `esco` config sets `expect_transition`: a run that does not convert at
  K=3 fails. The fit check's bound is 1.37 GB (sk2ml3) and 0.13 GB (oom2ml3),
  so both fit.

### Metric and reading

`wall_s` of the timed call. The K=3 level time and its split (`transition`,
`count_csr`, `reduce`, `filter`, the rest) are recorded as evidence.

1. **Signatures** must match between the arms per workload and GPU count. A
   mismatch or a logged fallback is (must-fix), and nothing goes on until it
   is fixed.
2. **ESCO shows a win** in a regime when its `wall_s` is ≥ 10 % below dense
   and ≥ 1 s faster: rule 2's thresholds on one rep, without the range test.
   oom2ml3 cannot meet the 1 s (its dense K=3 level is ≈ 1 s); it is evidence
   for the cost model only.
3. **Step 2 goes ahead** only if ESCO shows a win on sk2ml3 at either GPU
   count. Otherwise O5 ends: `"auto"` stays as it is, and the report records
   the calibration and where ESCO's K=3 time went (conversion, CSR kernel).
4. **Before O5 ends on a loss**, the report sets `count_csr` against the
   intersection model (sparse merge bytes, split evenly over the GPUs, at the
   device-to-device copy bandwidth that the calibration mode measures on each
   GPU before its first config: 1.13e12 B on sk2ml3, 3.74e10 B on oom2ml3,
   `bench/results/2026-10-06-phase-b-stake/k3_cost.txt`). If the kernel takes
   more than twice what the model allows, the loss is the enumeration of the
   generated candidates (11.8 B on sk2ml3, of which 145 M survive the subset
   test), not the intersections: a (should) finding that names a skip of
   prunable candidates in the CSR kernel as its own item, and O5 ends only on
   the owner's word. The calibration adds no other workload: the sparsest data
   in the suite is already in it, a very sparse level with few candidates
   holds milliseconds of dense work, and one level deeper adds mostly
   enumeration (`k4_cost.txt` there).

## Step 2 (outline; fixed by amendment before its first timed run)

A per-level cost model from facts known before the level: dense ≈ candidates
× words per kernel, sparse ≈ Σ over candidates of the parents' tidset lengths
plus the one-time conversion. Constants come from a sweep in the style of
`bench/k2_crossover.py`, and rule 4 sets the crossover. A campaign follows over
sk2ml3, oom2ml3, deepk, or002 and dsl, on 1 and 2 GPUs, 3 reps rep-major, with
the consolidation protocol's rules 1, 2, 4 and 5 as in phase A. Its budget is
agreed with the owner before the amendment. A fixed `sparse_from_k` stays.

## Correctness gate (before the calibration)

`uv run ruff check src tests bench`; `uv run pytest -q -m "not slow"` and
`-m "gpu and slow"`, both with the four known `test_smoke_correctness`
`test_support_001`/`0001` ids deselected; `uv run pytest
tests/test_tier_equivalence.py` (the ESCO legs already exist: `"auto"` and K=3, with
and without count inference, plain and forced chunks on 1 GPU, forced chunks on
2 GPUs); `uv run pytest -q -m "gpu and multigpu"`; `uv run python
bench/selfcheck.py`.
