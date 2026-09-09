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

### Added

- `bench/baseline/` — behaviour-change impact assessment, the min-count sweep
  tool, and a performance baseline covering every code path a slow fix touches.
- `bench/repro/` — one reproduction per defect, exiting non-zero while the
  defect is live and zero once fixed, so the same file is evidence and gate.

---

## [0.1.0]

Initial development release.
