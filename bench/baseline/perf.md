# Performance baseline (PR 0b)

Captured at `22cb1dc` + PR 0, before any remediation fix. **Re-run this matrix
after every PR and publish the delta in that PR's description:**

```
uv run python bench/baseline/perf_baseline.py --compare bench/baseline/perf-baseline.json
```

## Why it exists

Five queued fixes are explicitly slower — **#4** (int32 → float64 doubles the
MKL value array), **#13** (a lower length filter filters fewer transactions),
**#18** (honouring `n_jobs=1` gives up rayon's every-core default), **#22** (the
anchor filter becomes emit-only, so Phase 2 mines the full lattice), **#24**
(any overflow raises instead of completing).

The remediation plan warns that *"anyone who reverts #22 when Phase 2 slows down
has restored a 99.3% silent data loss to fix a performance problem."* Without a
measured before, that warning is an appeal to memory, and memory loses to a
profiler. This is the before.

Each config runs in a fresh process via `bench/child_run.py`: fresh CUDA
context, context-free `nvidia-smi` VRAM peak, `ru_maxrss` RSS peak, per-K
timings from `level_callback`, and an order-independent `itemset_hash` so a
performance rerun doubles as a correctness check.

**Every config runs three times and the minimum wall time is kept.** A single
sample is not usable as a gate at this scale: `smoke-gpu2` measured 1.11 s,
1.58 s and 1.57 s on *identical* code — a 42% spread, because CUDA/NCCL context
setup dominates a ~1 s run. The first comparison run reported `smoke-gpu2
+47.4%` for a change that provably cannot affect it, which is exactly the way a
noisy gate gets ignored. The minimum is the right statistic for a timing (the
sample least contaminated by unrelated work on the box), each config records
its own `wall_spread_pct`, and `--compare` only counts a regression as real
when it exceeds that measured spread.

`itemset_hash` must be identical across the repeats; a config that varies is
reported as NON-DETERMINISTIC, which is a correctness failure, not a timing one.

## Baseline

| config | route | wall | spread | itemsets | K | peak RSS | peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| `smoke-gpu1` | GPU x1 | 0.20s | 12.2% ⚠ | 694 | 6 | 467 MB | 3 MB |
| `smoke-gpu2` | GPU x2 | 1.07s | 48.7% ⚠ | 694 | 6 | 1,070 MB | 419 MB |
| `deepk-gpu2` | GPU x2 | 2.03s | 2.1% | 8,841 | 11 | 1,338 MB | 425 MB |
| `deepk-gpu2-sparse` | GPU x2 (CSR) | 2.04s | 1.7% | 8,841 | 11 | 1,330 MB | 483 MB |
| `deepk-gpu2-low` | GPU x2 | 2.16s | 0.3% | 111,985 | 14 | 1,313 MB | 435 MB |
| `twophase-deepk` | two-phase | 3.26s | 1.0% | 10,752 | 10 | 1,831 MB | 501 MB |
| `smoke-cpu-polars` | CPU polars | 0.33s | 0.9% | 694 | 6 | 372 MB | 3 MB |
| `smoke-cpu-sparse-j1` | CPU sparse | 0.57s | 1.7% | 694 | 6 | 341 MB | 3 MB |
| `smoke-cpu-sparse-j8` | CPU sparse | 0.59s | 13.4% ⚠ | 694 | 6 | 380 MB | 3 MB |
| `deepk-cpu-sparse-j8` | CPU sparse | 4.39s | 5.5% | 2,891 | 4 | 1,733 MB | 3 MB |
| `stressk2-cpu-sparse` | CPU sparse | 75.03s | 1.0% | 2,826 | 2 | 4,238 MB | 3 MB |

⚠ marks a config whose run-to-run spread exceeds 10%: sub-second runs where
CUDA/NCCL context setup dominates. **Do not read a regression into those from a
single comparison.** The configs that guard the slow fixes are stable —
`stressk2-cpu-sparse` (#4) 1.0%, `twophase-deepk` (#22) 1.0%,
`deepk-gpu2-low` 0.3%.

*Captured after PR 1. PR 1 changed itemset element order and the min-count
expression; every config's `itemset_hash` was identical before and after, and
every timing delta fell inside the measured spread. The original single-sample
capture is kept as `perf-baseline-pr0-singlesample.json` — it is what
demonstrated the need for repeats.*


## What each config guards

| config | the fix it measures |
|---|---|
| `smoke-gpu1`, `smoke-gpu2` | GPU row-split, 1 vs 2 devices — #24's raise path |
| `deepk-gpu2`, `deepk-gpu2-low` | deeper lattices, where a per-level regression compounds |
| `deepk-gpu2-sparse` | the dense→sparse CSR transition |
| `twophase-deepk` | **#22.** Distinct phase supports (0.02 → 0.005) — the only configuration in which the anchor filter does anything. Equal supports make the mask all-True, which is the vacuous shape `tests/test_row_split_e2e.py:102` also has. |
| `smoke-cpu-polars` | the CPU tier, dense path |
| `smoke-cpu-sparse-j1` vs `-j8` | **#18.** `n_jobs` is dead today; after the fix these must diverge. |
| `deepk-cpu-sparse-j8` | CPU sparse at K=4 |
| `stressk2-cpu-sparse` | **#4.** 2M rows × 35k items, k=2 matmul-bound. `min_support` is raised to 0.002 from the preset's 1.5e-5 (~612M candidate pairs, hours) so the config stays matmul-bound but finishes in ~90 s — a matrix re-run after every PR cannot afford an hours-long config. |

## Reading the delta

`--compare` prints a per-config before/after with the percentage change **and
whether `itemset_hash` moved**. A changed hash on a config the PR was not
supposed to affect is a correctness regression, not a performance one — that is
the more important half of the output.
