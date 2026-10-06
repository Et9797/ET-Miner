# Route optimizations, phase A (O1–O3) — report

Phase A of `SPEC.md` on `perf/route-optimizations`. Protocol, fixed before the
first timed run (amended twice before it, both times on the owner's decision):
`PROTOCOL.md`. Evidence: `bench/results/2026-10-05-optimizations/`:

- `raw.jsonl` and `env.txt`: the calibration and the campaign, at `201f1c2`.
- `k2_crossover.jsonl`: the K=2 sweep, at `201f1c2`.
- `final.jsonl` and `final-env.txt`: the final check, at the decided tree.

Box: 2× RTX A4000 16 GB (sm_86, `NCCL_P2P_DISABLE=1`), Ryzen 5 5600X, 46 GB
RAM, idle throughout.

## What changed

- **O1, K=2 kernel: dispatch on r.** With transactions input and one shard per
  device, the row-split miner counts K=2 from the rows when
  `r = Σ_rows C(len, 2) / (pairs × words)` over the frequent columns is below
  `K2_ROWS_MAX_R = 3.95e-3` (`gpu/row_split_chunks.py::k2_counts_rows`), with the
  dense pair kernels at and above it. r is read from the CSR's row pointers
  after K=1 (`row_split._k2_row_pairs`); the shards of frequent positions are
  built only when the rows count. `ET_MINER_K2_KERNEL` pins either kernel;
  `bitvecs=` input counts dense.
- **O2, multi-GPU reduce: compacted by default.** A dense K≥3 level with a
  subset index on more than one GPU reduces only the entries its kernels wrote;
  `ET_MINER_REDUCE=dense` pins the whole-array reduce. K=2, levels without an
  index, ESCO levels and one GPU reduce dense, as before.
- **O3, ESCO materialization from the count pass: removed.** The source changes
  of `9e63919` are reverted: the knob, the peer partials of the chunk loop and
  the ESCO level, `csr_write_gather_checked` and its tests and tier legs.
  `ET_MINER_ESCO_MATERIALIZE` is now refused at entry
  (`_env.reject_removed_knobs`). The level-split phases `transition` and
  `materialize` stay, since they time `recount` too.
- **Tier chain** (`CLAUDE.md`, `tests/test_tier_equivalence.py`):
  - the row-wise K=2 leg selected by the r dispatch (its crossover raised above
    smoke's r);
  - the 2-GPU reduce legs run the compacted default and the dense pin;
  - the ESCO reuse legs are gone.
- **Bench**:
  - `runner.py --mode optimizations-calibration | optimizations | optimizations-final`;
  - `k2_crossover.py`;
  - `decide.py`;
  - the level-split phase `k2_dispatch`.

## Calibration

dsl-esco on `base` ran out of memory on one and on two GPUs, at the transition
to ESCO at K=6 (`Out of memory allocating 3,290,000,384 bytes`, after 12 s). Per
the protocol it is dropped from the campaign at both GPU counts.

## K=2 crossover sweep

Ratio = rows / dense K=2 level time, median of 3 interleaved reps, one GPU.
Full table: `uv run python bench/k2_crossover.py --analyze --out
bench/results/2026-10-05-optimizations/k2_crossover.jsonl`.

| point (N = 1M) | r | dense ms | rows ms | ratio |
|---|---|---|---|---|
| uniform F=30000 L=5 | 1.4e-6 | 9507.1 | 43.4 | 0.005 |
| uniform F=10000 L=40 | 1.0e-3 | 1649.6 | 468.0 | 0.284 |
| uniform F=1000 L=5 | 1.3e-3 | 20.8 | 18.2 | 0.874 |
| uniform F=3000 L=20 | 2.7e-3 | 154.0 | 100.8 | 0.655 |
| uniform F=1000 L=10 | 5.8e-3 | 20.6 | 30.3 | 1.472 |
| uniform F=3000 L=40 | 1.1e-2 | 153.5 | 284.6 | 1.854 |
| uniform F=100 L=40 | 10.1 | 4.7 | 135.9 | 28.929 |
| Zipf F=3000 L=20 | 2.7e-3 | 146.6 | 60.4 | 0.412 |
| Zipf F=1000 L=20 | 2.4e-2 | 21.0 | 59.9 | 2.853 |

- **Uniform lines.** Sorted by r, the 24 uniform points cross 1 exactly once,
  between r = 2.7e-3 and 5.8e-3, so `r* = 3.95e-3` (the geometric mean).
- **Zipf line.** It crosses between 2.7e-3 and 2.4e-2, which is higher, so r*
  stays the uniform value.
- **N = 4M.** The two points re-run at N = 4M keep their side of 1: ratio 0.681
  against 0.655 at 1M, and 1.434 against 1.472. So r, not N, predicts.
- **Rows floor.** Row-wise K=2 costs about 18 ms at 1M rows whatever F is: the
  host CSR and its upload. That is why the dense kernels win at small F.

On the campaign workloads, `r*` puts oom2ml3 and sk2ml3 (r ≈ 3e-5) on the rows
and every other workload (r ≥ 0.02) on the dense kernels.

## Decisions (protocol rule, campaign at `201f1c2`)

Wall s, median [min, max] over 3 reps, rep-major. All 105 configs returned ok
with no fallback, and every equivalence group has one signature. Tables:
`uv run python bench/optimizations/decide.py
bench/results/2026-10-05-optimizations/raw.jsonl
bench/results/2026-10-05-optimizations/final.jsonl`.

### DP-O1: the K=2 kernel

| regime | base | rows | K=2 level, base | K=2 level, rows | rule 2 |
|---|---|---|---|---|---|
| sk2ml3, 1 GPU | 50.58 [50.38, 50.58] | 20.91 [20.90, 20.98] | 29.16 | 0.65 | rows wins (−29.67 s) |
| sk2ml3, 2 GPUs | 45.19 [45.19, 45.21] | 31.15 [30.68, 31.19] | 15.46 | 1.37 | rows wins (−14.04 s) |
| oom2ml3, 1 GPU | 6.89 [6.86, 6.92] | 1.82 [1.82, 1.83] | 5.19 | 0.18 | rows wins (−5.07 s) |
| oom2ml3, 2 GPUs | 7.77 [7.77, 7.78] | 5.26 [5.25, 5.27] | 3.31 | 0.83 | rows wins (−2.51 s) |
| dsl, 1 GPU | 20.20 [20.09, 20.30] | 22.04 [21.95, 22.09] | 0.07 | 1.83 | tie (+1.84 s) |
| dsl, 2 GPUs | 15.75 [15.66, 15.78] | 16.65 [16.63, 16.78] | 0.07 | 1.16 | tie (+0.90 s) |
| smoke, deepk, skew, or003, or002 | 0.04–0.71 | 0.04–0.77 | ≤ 0.01 | ≤ 0.11 | tie (≤ +0.06 s) |

**Outcome.** r* lies inside the grid and `rows` wins regimes, so the dispatch is
implemented (the amended DP-O1).

**The check against the campaign.** The dispatch picks `rows` exactly where
`rows` wins, and dense everywhere else, where the two tie. No regime is
misclassified. On dsl, `rows` is 1.84 s slower: the loss the amended DP-O1 was
written for, which the dispatch avoids.

### DP-O2: the multi-GPU reduce

| regime | base | compact | K≥3 reduce + compact, base | …, compact | rule 2 |
|---|---|---|---|---|---|
| sk2ml3, 2 GPUs | 45.19 [45.19, 45.21] | 30.11 [30.10, 30.12] | 16.53 | 1.08 | compact wins (−15.08 s) |
| oom2ml3, 2 GPUs | 7.77 [7.77, 7.78] | 5.14 [5.14, 5.14] | 2.91 | 0.19 | compact wins (−2.63 s) |
| dsl, 2 GPUs | 15.75 [15.66, 15.78] | 15.29 [15.28, 15.38] | 0.01 | 0.02 | tie (−0.46 s) |
| dsl infer, 2 GPUs | 13.16 [13.16, 13.19] | 12.92 [12.87, 13.08] | 0.01 | 0.03 | tie (−0.24 s) |

**Outcome.** `compact` wins two regimes and loses none, so it is the default and
`ET_MINER_REDUCE=dense` stays as the pin.

### DP-O3: ESCO materialization

| regime | base | reuse | materialize, base | materialize, reuse | rule 2 |
|---|---|---|---|---|---|
| deepk ESCO, 1 GPU | 0.87 [0.87, 0.88] | 0.74 [0.74, 0.74] | 0.22 | 0.10 | tie (−0.13 s) |
| deepk ESCO, 2 GPUs | 0.98 [0.97, 0.99] | 0.99 [0.98, 1.00] | 0.14 | 0.06 | tie (+0.01 s) |
| or002 ESCO, 1 GPU | 0.35 [0.32, 0.37] | 0.38 [0.36, 0.39] | 0.02 | 0.02 | tie (+0.04 s) |
| or002 ESCO, 2 GPUs | 0.66 [0.66, 0.67] | 0.67 [0.66, 0.67] | 0.02 | 0.03 | tie (+0.00 s) |

**Outcome: tie everywhere,** as the protocol expected without dsl-esco. Rule 5
decides on code:

- `recount` needs no code of its own, since `reuse` still gather-counts the
  survivors that are 0 on every further shard.
- `reuse` needed about 180 lines in `src/` (Python and CUDA, without blank and
  comment lines).

So `reuse` is removed.

## Final check (decided tree, knobs unset, one rep)

| regime | wall | K=2 kernel | K≥3 reduce | signature |
|---|---|---|---|---|
| smoke / deepk / skew / or003 / or002, 1 GPU | 0.10 / 0.54 / 0.73 / 0.14 / 0.28 | dense | — | matches |
| deepk ESCO, 1 / 2 GPUs | 0.86 / 0.99 | dense | — / dense | matches |
| or002 ESCO, 1 / 2 GPUs | 0.32 / 0.61 | dense | — / dense | matches |
| dsl, 1 / 2 GPUs | 20.17 / 15.42 | dense | — / compact | matches |
| dsl infer, 2 GPUs | 12.88 | dense | compact | matches |
| oom2ml3, 1 / 2 GPUs | 1.86 / 2.70 | rows | — / compact | matches |
| sk2ml3, 1 / 2 GPUs | 20.68 / 16.11 | rows | — / compact | matches |

Every signature matches the campaign's. Where both changes apply, they add up:
- oom2ml3 on 2 GPUs: 7.77 s on `base`, 2.70 s with both.
- sk2ml3 on 2 GPUs: 45.19 s on `base`, 16.11 s with both.

## Budget

0.83 of 3 GPU-hours: calibration 0.013, sweep 0.164, campaign 0.590, final
check 0.06.

## Findings

1. (nice) **The dispatch's r costs about 0.1 s on dsl.** It appears as phase
   `k2_dispatch` and raises the dense K=2 level from 0.07 to 0.17 s; the wall is
   20.17 s against `base`'s 20.20. It is O(rows) when every column of the CSR
   is frequent, which is how the miner's CSR is built. Otherwise it is O(nnz):
   0.8 s at dsl's 220M non-zeros, measured on the CPU.
2. (nice) **ESCO's conversion seems to size by the global count.** The
   calibration allocates the same 3.29 GB on one and on two GPUs at the ESCO
   transition. That suggests the conversion sizes each GPU's tidsets by the
   global count rather than by its shard. Not examined; outside phase A.
3. (nice) **Thermal slowdown on a few sk2ml3 runs.** Hardware and software
   thermal slowdown was flagged on some sk2ml3 runs, as in the pruning
   campaign. The ranges are within 0.5 s and the margins are 14–30 s, so no
   outcome depends on it.
