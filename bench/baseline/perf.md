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

## Baseline

| config | route | wall | itemsets | K | peak RSS | peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| `smoke-gpu1` | GPU x1 | 0.22s | 694 | 6 | 467 MB | 3 MB |
| `smoke-gpu2` | GPU x2 | 1.08s | 694 | 6 | 1,069 MB | 391 MB |
| `deepk-gpu2` | GPU x2 | 2.08s | 8,841 | 11 | 1,345 MB | 425 MB |
| `deepk-gpu2-sparse` | GPU x2 (CSR) | 2.02s | 8,841 | 11 | 1,377 MB | 483 MB |
| `deepk-gpu2-low` | GPU x2 | 2.14s | 111,985 | 14 | 1,381 MB | 437 MB |
| `twophase-deepk` | two-phase | 3.15s | 10,752 | 10 | 1,901 MB | 433 MB |
| `smoke-cpu-polars` | CPU polars | 0.30s | 694 | 6 | 372 MB | 3 MB |
| `smoke-cpu-sparse-j1` | CPU sparse | 0.52s | 694 | 6 | 340 MB | 3 MB |
| `smoke-cpu-sparse-j8` | CPU sparse | 0.59s | 694 | 6 | 395 MB | 3 MB |
| `deepk-cpu-sparse-j8` | CPU sparse | 4.98s | 2,891 | 4 | 2,040 MB | 3 MB |
| `stressk2-cpu-sparse` | CPU sparse | 87.62s | 2,826 | 2 | 5,487 MB | 3 MB |

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
