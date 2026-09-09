# Changelog

All notable changes to ET-Miner are recorded here. Versions follow
[Semantic Versioning](https://semver.org/), with the caveat below.

> **Pre-1.0 caveat.** At 0.x a minor bump may change mined output. Every such
> change is listed under **Behaviour changes** with a before/after and a
> migration note, because a frequent-itemset miner's output *is* its API:
> a number that moves silently invalidates whatever was published from it.

---

## [Unreleased] — 0.2.0

Remediation of the 62 defects recorded in `BUGS_FOUND.md` (an adversarial
four-reviewer review of `22cb1dc`), plus 19 further defects found while
planning it. Landing across PRs 0–11; this section is filled in as they merge.

### Behaviour changes

Six queued fixes change mined output. Before any of them land,
`bench/baseline/behaviour-change-impact.md` records which already-published
figures move — measured, per artifact, not assumed.

| Defect | Change | Who is affected |
|---|---|---|
| **#11** | `_min_count` becomes `ceil(Fraction(str(s)) * N)` instead of `ceil(fl64(s) * N)`, at all three sites (`core/result.py`, `rust_ext`, `synthetic.py`) | any run whose `(min_support, n_rows)` pair shifts — check with `bench/baseline/min_count_impact.py` |
| **#12** | `compute_self_sufficiency` aggregates `min`, not `max`, over the (K-1)-subsets | **every row's ratio changes**; any published self-sufficiency value is invalidated, not merely shifted |
| **#13** | `count_support_batched`'s length filter uses the batch **minimum**, not `itemsets[0]` | callers passing mixed-length batches; `apriori()` always passes a uniform level and is unaffected |
| **#22** | `anchor_items` becomes an output selector; the generating level is no longer filtered | `mine_two_phase` and `apriori(anchor_items=...)` return the itemsets they always advertised — up to 99.3% more — and Phase 2 gets slower |
| **#24** | any GPU result-buffer overflow raises instead of silently truncating | runs that were completing with a non-deterministic ≤5% loss now fail loudly and say which knob to raise |
| **#51** | `--min-lift` filters at `>=` its value, including the default `1.0` | CLI users relying on the accidental no-op at 1.0 |

### Migration

- **Re-check any published figure** against
  `uv run python bench/baseline/min_count_impact.py --exhaustive --n-rows <N>`.
  The base214m run (N = 76,890,945) is cleared: 0 of 189,999 thresholds shift.
  That is a property of *that row count* — the same sweep finds 2,457 shifting
  thresholds at N = 1,000,000.
- **Self-sufficiency values must be recomputed.** The old `max`-based ratio and
  the new `min`-based one agree on no row in practice.
- **Phase 2 of `mine_two_phase` gets slower** and its default `phase2_support`
  is raised. That is the fix working: the old speed was purchased by discarding
  up to 99.3% of the correct answer. `bench/baseline/perf-baseline.json` holds
  the before.

### Fixed

*PR 1 — canonical order and the exact threshold*

- **#1, #2, #3, #20** — itemsets are emitted as **ascending tuples of item ids
  on every route**, and that contract is now written into `apriori()`'s Returns
  block. `build_boolean_matrix`'s column names are zero-padded, which makes
  lexicographic name order equal item-id order and so fixes the CPU route at
  the root rather than re-sorting at emission. Measured: 271/461 emitted
  itemsets were non-ascending, now 0; `generate_rules` was dropping 408 of
  2,256 rules (18.1%) and reporting `lift = 0.0` on 408 more, now 0 and 0.
  `_build_support_lookup` is additionally keyed on the sorted tuple, so it no
  longer depends on its producer's ordering.
- **#11, #14** — `_min_count` is the **exact decimal ceiling**,
  `ceil(Fraction(str(s)) * N)`, at all three sites (`core/result.py`,
  `synthetic.py`, and `exact_min_count` in the Rust extension). The old
  `ceil(fl64(s) * N)` returned 701 where the exact ceiling is 700 at
  `s = 0.07, N = 10000`, dropping the boundary itemset and the entire cone
  above it — 3 itemsets mined against the oracle's 7. All three sites are
  checked against one shared table, `tests/fixtures/min_count_cases.json`,
  read by both the Python tests and a Rust `#[test]`: three implementations
  agreeing with each other is worth nothing when they share a bug, which is
  exactly how this defect survived. The tier gate gains an independent boundary
  case, since it previously derived its oracle threshold from the expression it
  was meant to check.

*PR 2 — sparse/MKL numerics and the Rust boundary*

- **#4** — `_sparse_matmul` widens integer input to **float64**, not float32.
  float32's 24-bit significand made a support count stick at
  2²⁴ = 16,777,216: measured 20,000,000 → 16,777,216, a 16.1% **under**-count
  that silently dropped frequent pairs and cascaded into every higher K. Costs
  2× the value-array bytes inside the matmul.
- **#5** — `panic = "abort"` removed from the release profile, so an FFI panic
  raises a catchable `PanicException` instead of an uncatchable SIGABRT that
  loses every unflushed level of a campaign. All FFI entry points now share one
  contract: non-contiguous numpy input is copied rather than panicking (half of
  them already did this and half did not), and CSR shape problems raise a
  `ValueError` naming the offending value instead of an index-out-of-bounds
  panic from inside a kernel.
- **#16** — importing `core.sparse` no longer overrides the process-global MKL
  thread count. The module is imported lazily from inside
  `count_support_batched`, so this fired mid-run and silently oversubscribed a
  host application's own MKL configuration (measured: a host setting of 2
  became 24). `_restore_mkl_threads` now restores the value it displaced
  rather than re-deriving one from the environment.
- **#17** — MKL discovery globs `libmkl_rt.so*` across candidate directories
  and logs the outcome either way. It previously tested one filename at one
  location and returned silently on a miss, so nothing distinguished "the path
  was already fine" from "found nothing" — and the numerics then ran against
  whichever unpinned system MKL loaded, which is what #4's accuracy depends on.
- **#18** — `n_jobs` is honoured on the Rust path. rayon took every core
  regardless, so `n_jobs=1` — documented as "sequential execution" — silently
  oversubscribed a shared box. The counting kernels now accept a thread budget
  and run on a scoped pool (verified: `n_threads=1` is 5.9× slower than
  unbounded, with identical counts). The dead `_RUST_MIN_ITEMSETS` fossil is
  gone.

### Added

- `bench/baseline/` — behaviour-change impact assessment, the min-count sweep
  tool, and a performance baseline covering every code path a slow fix touches.
- `bench/repro/` — one reproduction per defect, exiting non-zero while the
  defect is live and zero once fixed, so the same file is evidence and gate.
- `tests/fixtures/min_count_cases.json` — shared ground truth for the min-count
  rule, read by the Python and Rust test suites alike.

---

## [0.1.0]

Initial development release.
