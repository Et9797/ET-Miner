# Session Handoff: ET-Miner, 2026-10-28 14:08

**Branch:** `refactor/route-consolidation` @ `fd343d0` (pushed, in sync)
**Tree:** clean (only this handoff)
**Focus:** Route consolidation round 2: Phase 3 (one commit per decision point), then Phase 4
**Reason:** long session; state checkpoint mid-Phase 3

## Verify on Resume
`git status --short --branch`, `git rev-parse --short HEAD` (expect `fd343d0`). Read
`bench/results/2026-10-27-routes-gpu/FINDINGS.md` (decisions + owner decisions) and
`bench/routes/PROTOCOL.md` (rule; "Owner decisions after the measurements" at the end).

## Current State
- Phases 2a, 2b, C4, H: done, committed (bench/results/2026-10-27-routes-{cpu,gpu}/).
- Phase 3 done (each: tests, tier gate 89, not-slow, no-CuPy, ruff, pushed): G3 `a75b22b`, C1 `11f5657`,
  C2 `4e2ed6e`, C4 `60adeb5`, H `2b38fed`, S1/S2 `fd343d0`. CHANGELOG has a "Route consolidation, round 2"
  section at the top of Unreleased; add one bullet per DP.

## Next Steps (Phase 3, remaining)
1. **G1**: host CSR stays as named fallback by rule 7 (only layer for > INT_SPAN_LIMIT distinct items, and for
   output_dir/resume_from_k/anchor_items on columns the device layer does not serve; also a tier leg pins it).
   Make `coded` the default for unserved columns (measured faster than host: dsl-str 18.3 vs 30.6 s, dsl-wide
   10.7 vs 29.1 s); drop `rebuilt` (`core/apriori.py::_rebuilt_csr`) and the bench-only pin values
   rebuilt/coded. Tell the owner host stays by role.
2. **G2**: lex stays default; `keep` becomes the named fallback chosen when the lex order does not fit
   (needs an upfront VRAM check in the device input layer, `gpu/input_layer.py`); drop the ET_MINER_ROW_ORDER pin
   unless a tier leg needs it ("row order keep / lex pinned" legs exist in the chain: keep the pin for them,
   or drop the legs with the owner's word).
3. **G7**: remove `prune_apriori` (public param; owner approved) + its tier legs in CLAUDE.md + tests; CHANGELOG.
4. **G6**: remove ESCO (`sparse_from_k`, `gpu/sparse_csr.py`, `gpu/density.py`, kernels) using the pushed
   branch `refactor/drop-esco` as a basis (check its base first); tier legs + CLAUDE.md; CHANGELOG.
5. **Pins cleanup**: ET_MINER_LEAVE_OUT (bench only; no tier leg needs it) goes; routes_matrix keeps
   PHASE2_LEAVE_OUT as the record.
6. **(must-fix)** 2-GPU `bitvecs=` OOM: `gpu/row_split.py::shard_prebuilt_bitvecs` makes a contiguous copy per
   shard on the input device (8.75 GB + 4.375 GB on 12 GB). Copy each row range straight to its device; raise an
   upfront MemoryError naming sizes where it cannot fit. Regression test.
7. Phase 4: `bench/routes/final.py::final_configs(n_dev)` for `--mode routes-final` (every surviving default
   config on the final tree), run it, adversarial-verifier review, `/council review`, REPORT.md.

## Owner decisions (all recorded in PROTOCOL.md / FINDINGS.md)
- Rule 7 counts a memory rise only at >= 1 GB and > 1.25x. S1 boolean goes (VRAM accepted). G2 keep stays as
  fallback. C3: `apriori_from_csr` stays. H got +0.5 GPU-h (used 0.075). G6 + G7 API removal approved with
  CHANGELOG. Earlier: SON stays; use_generator_pruning default False; n_gpus exact; row split stays.
- Approved exception used in C4: sparse= raises unless None. I extended it to count_support_batched's n_jobs
  (only the sparse counter read it): flagged for the owner in the commit and to be listed in the report.

## Findings to report (not fixed)
- apriori_from_csr at min_count 0 emits every candidate with count 0; the CPU route emits only occurring
  itemsets (Tier 2 vs Tier 1 at min_count 0, untested edge).
- CPU route > 900 s on dsl (Rust 227 s; infer/free 280 s). Rust deepk20 slow, not hung (ml 7/9/11: 16/197/878 s).
- SUBSECOND keyed on workload not regime (no rep shed this time).
- cargo clippy -D warnings: 2 errors on main (div_ceil, complex type); cargo fmt differs on main.
- Historical bench tools now stale: bench/cpu/l3_crossover.py (already stale), bench/cpu/son_stakes.py,
  bench/cpu/sparse_non_pairs.py, bench/repro/d04 (import removed code).

## Dead Ends / cautions
- Never run uv/maturin inside rust_ext/ (cd back to the repo root after cargo).
- `.claude/handoffs/` is gitignored; LATEST.md is tracked: commit it with `git commit -- .claude/handoffs/LATEST.md`.
- Gate recipe: `uv run pytest tests/test_tier_equivalence.py -q`; not-slow `-m "not slow" -k "not test_support_00"`;
  no-CuPy via `<scratchpad>/gates/nocupy.py` (sets sys.modules["cupy"]=None); `uv run python bench/selfcheck.py`
  after kernel changes; rebuild Rust with `env -u CONDA_PREFIX uv run maturin develop --release -m rust_ext/Cargo.toml`.

## Scratchpads
- This session: `/tmp/claude-0/-root-projects-ET-Miner/f1b33288-d153-44c1-9cfe-35bc8dad319e/scratchpad`
  (gates/, c4/ dumps, h/ dumps, c2check/, rustk20/).
- Previous: `/tmp/claude-0/-root-projects-ET-Miner/0b85df72-8420-4cb0-b6da-84d256972ef0/scratchpad` (PLAN.md).
