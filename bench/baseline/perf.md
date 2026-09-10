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
| `smoke-gpu1` | GPU x1 | 0.20s | 7.5% | 694 | 6 | 444 MB | 3 MB |
| `smoke-gpu2` | GPU x2 | 1.07s | 48.9% ⚠ | 694 | 6 | 1,045 MB | 391 MB |
| `deepk-gpu2` | GPU x2 | 2.04s | 1.4% | 8,841 | 11 | 1,312 MB | 425 MB |
| `deepk-gpu2-sparse` | GPU x2 (CSR) | 2.02s | 2.4% | 8,841 | 11 | 1,333 MB | 551 MB |
| `deepk-gpu2-low` | GPU x2 | 2.15s | 0.4% | 111,985 | 14 | 1,370 MB | 437 MB |
| `twophase-deepk` | two-phase | 3.23s | 1.3% | 10,752 | 10 | 1,791 MB | 433 MB |
| `smoke-cpu-polars` | CPU polars | 0.31s | 8.8% | 694 | 6 | 348 MB | 3 MB |
| `deepk-cpu-sparse-j1` | CPU sparse | 6.22s | 0.8% | 2,891 | 4 | 915 MB | 3 MB |
| `smoke-cpu-sparse` | CPU sparse | 0.81s | 12.9% ⚠ | 694 | 6 | 393 MB | 3 MB |
| `deepk-cpu-sparse-j8` | CPU sparse | 5.28s | 3.2% | 2,891 | 4 | 1,791 MB | 3 MB |
| `stressk2-cpu-sparse` | CPU sparse | 75.86s | 2.3% | 2,826 | 2 | 4,793 MB | 3 MB |

⚠ marks a config whose run-to-run spread exceeds 10%: sub-second runs where
CUDA/NCCL context setup dominates. **Do not read a regression into those from a
single comparison.** The configs that guard the slow fixes are stable —
`stressk2-cpu-sparse` (#4) 2.3%, `twophase-deepk` (#22) 1.3%,
`deepk-cpu-sparse-j1` (#18) 0.8%, `deepk-gpu2-low` 0.4%.

*Captured after PR 2, with the same `itemset_hash` on every config as PR 1 and
PR 0. PR 2's measured deltas: every GPU config unchanged;
`stressk2-cpu-sparse` **+1.8%** for #4's float64 widening, which is the
trustworthy number because it is the matmul-bound config with a 2.3% spread;
and `deepk-cpu-sparse-j1` slower than `-j8` (6.22 s vs 5.28 s) which is #18
**working** — `n_jobs=1` now genuinely means one thread. The single-sample PR-0
capture is kept as `perf-baseline-pr0-singlesample.json`.*


## What each config guards

| config | the fix it measures |
|---|---|
| `smoke-gpu1`, `smoke-gpu2` | GPU row-split, 1 vs 2 devices — #24's raise path |
| `deepk-gpu2`, `deepk-gpu2-low` | deeper lattices, where a per-level regression compounds |
| `deepk-gpu2-sparse` | the dense→sparse CSR transition |
| `twophase-deepk` | **#22.** Distinct phase supports (0.02 → 0.005) — the only configuration in which the anchor filter does anything. Equal supports make the mask all-True, which is the vacuous shape `tests/test_row_split_e2e.py`'s equal-support case also has. |
| `smoke-cpu-polars` | the CPU tier, dense path |
| `deepk-cpu-sparse-j1` vs `-j8` | **#18.** `n_jobs` was dead; after the fix these must diverge. Run on `deep_k` rather than `smoke` — the smoke sparse configs finish in ~0.6 s with a 14-22% spread, too noisy to gate a 20% change on. |
| `deepk-cpu-sparse-j8` | CPU sparse at K=4 |
| `stressk2-cpu-sparse` | **#4.** 2M rows × 35k items, k=2 matmul-bound. `min_support` is raised to 0.002 from the preset's 1.5e-5 (~612M candidate pairs, hours) so the config stays matmul-bound but finishes in ~90 s — a matrix re-run after every PR cannot afford an hours-long config. |

## Reading the delta

`--compare` prints a per-config before/after with the percentage change **and
whether `itemset_hash` moved**. A changed hash on a config the PR was not
supposed to affect is a correctness regression, not a performance one — that is
the more important half of the output.

**Treat a large percentage as unexplained until you have found its cause, and
read the hash column regardless.** `perf-baseline.json` was last recorded at
`865286c` (`git log --follow` gives `5a1fac0 -> 7b06872 -> 865286c`, which
rewrote it 199+/212- with a changed config set) — not at `22cb1dc` + PR 0 as
an earlier revision of this file claimed, though it does still predate the
remediation, which is what makes the hash column meaningful.

Three separate mechanisms have produced large deltas here on runs whose hashes
were all `same`, and they are worth distinguishing rather than collapsing into
"the hardware":

- **Run-to-run noise on sub-second configs.** `smoke-gpu2` measured a 48.9%
  spread on identical code; `smoke-cpu-sparse` measured 20.5-26.8%. `compare()`
  takes `max(old_spread, new_spread)`, so a delta inside that band is not
  counted as a regression at all.
- **Concurrent load.** One reviewer's +70.2% was measured while a 200M-element
  host-RAM benchmark ran on the same box.
- **Different hardware.** The baseline carries no machine identifier, so this
  one is not checkable from the artifact — infer it only after excluding the
  other two.

### Known standing delta: `twophase-deepk`, since `2eb81b0`

`--compare` has reported `CHANGED 10752->10570` on `twophase-deepk` on **every
run since `2eb81b0`**, and it is correct to do so: that is #22 landing, not a
regression. The baseline JSON is pinned at `22cb1dc` + PR 0 and has deliberately
never been re-recorded, so this row will keep firing until it is.

`2eb81b0`'s own message states the arithmetic and it closes exactly:

> K=1 is masked too. `_anchor_keep_mask` returned None for k<2, so a two-phase
> run emitted every frequent item at K=1 while filtering every deeper level.
> [...] 294 frequent items at phase2_support minus 112 anchors = 182 =
> 10,752 - 10,570.

Bisected on 2026-09-10 to rule out PR 6: `04770ca` (PR 6's branch point) already
gives 10,570, as do all seven PR 6 commits; `2090cdc` and everything before it
gives 10,752.

**This is the one row on which a changed hash is expected.** Every other config
is still a correctness signal, and this one is only benign until the baseline is
re-recorded — at which point the exemption must be deleted along with it.
