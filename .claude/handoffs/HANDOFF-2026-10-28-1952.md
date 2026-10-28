# Session Handoff: ET-Miner, 2026-10-28 19:52

**Branch:** `refactor/route-consolidation` @ `af85dad` (in sync with upstream)
**Tree:** clean except untracked `bench/results/2026-10-28-routes-final/` (the running Phase 4 campaign), 0 stashes
**Focus:** Route consolidation round 2: Phase 4 (routes-final) running; then compare, reviews, REPORT.md
**Reason:** owner asked for a handoff mid-run (task continues)

## Verify on Resume
Run `git status --short --branch` and `git rev-parse --short HEAD`. HEAD should be `af85dad` (or a later commit of the Phase 4 results / this handoff) and the tree should match "Tree" above. If not, read the diff before trusting "Current State".
Check whether the run is still going: `ps aux | grep "runner.py --mode routes-final" | grep -v grep` (runner python pid was 3156068).

## Current State
- Phase 3 is complete: G6 (ESCO) and the pins cleanup are committed and pushed.
- Phase 4 is running, detached (setsid nohup), started 19:22:
  `uv run python bench/runner.py --mode routes-final --out bench/results/2026-10-28-routes-final --max-hours 3`
  Log: `/tmp/claude-0/-root-projects-ET-Miner/809f9290-2a5a-4164-8add-95272e4c0bbd/scratchpad/routes-final.log`. 195 rows (65 configs x 3), est. 1.5-2 h; 57 rows done at 19:52.
- The runner is resumable: rerunning the same command skips ids already in `raw.jsonl`.
- **No tracked edits and no commits until the runner exits**: `bench/runner.py:270-282` stamps each row with `git rev-parse HEAD` + a digest of the full `git diff HEAD`. That includes `.claude/handoffs/LATEST.md`, so this handoff is NOT copied to LATEST.md yet.
- One expected failure so far: `sk2ml3-bv-C2-default#r0`: `MemoryError: bitvecs= on 2 GPUs: device 0 needs 4.32 GB for its row shard(s) beside the caller's bitvectors and has 3.28 GB free. Mine with n_gpus=1, or pass the transactions instead of bitvecs= ...`. It OOM'd in Phase 2 too (`OutOfMemoryError ... 4,375,000,064 bytes`); the clear upfront raise is the intended behaviour of 2aa6b72.

## Next Steps
1. When the runner exits:
   - `cp .claude/handoffs/HANDOFF-2026-10-28-1952.md .claude/handoffs/LATEST.md`;
   - commit and push LATEST.md (`git commit -- .claude/handoffs/LATEST.md`);
   - commit and push `bench/results/2026-10-28-routes-final/`.
2. Compare final vs Phase 2 by `base_id`:
   `uv run python /tmp/claude-0/-root-projects-ET-Miner/809f9290-2a5a-4164-8add-95272e4c0bbd/scratchpad/g6/compare_final.py .`
   - It prints a markdown table: P2/P4 median wall, max VRAM/RSS, signature same/DIFF, failures.
   - Phase 2 rows are in `bench/results/2026-10-27-routes-{cpu,gpu}/raw.jsonl`.
   - Signatures must match (rule 1: `bench/routes/decide.py:77`, `_sig` = n_itemsets, sum_counts, itemset_hash).
   - If the script is lost: join on `config.base_id` and use `decide._sig` / `decide._peak_vram`.
   - Put the table in `bench/results/2026-10-28-routes-final/` (e.g. COMPARE.md), then commit and push.
3. adversarial-verifier review of the branch diff vs `f6c3d36` (only after the run: it runs tests on CPU/GPU), then `/council review`.
4. `bench/results/2026-10-28-routes-final/REPORT.md`: a numbered list tagged (must-fix)/(should)/(nice), sorted, plus a Dutch chat summary. Commit and push. No PR/merge without the owner.

## Session Instructions
- Owner: "Commit en push frequent vanaf nu": commit and push after every verified step without asking (memory `commit-push-often.md`). This does not apply during a timed run.
- Chat in Dutch, concise; files, code and commits in English. Reports: numbered lists tagged (must-fix)/(should)/(nice), sorted.
- The earlier owner decisions still hold (see the previous handoff `HANDOFF-2026-10-28-1451.md`, `bench/routes/PROTOCOL.md` end, `bench/results/2026-10-27-routes-gpu/FINDINGS.md`).

## Decisions & Rationale
- **G6 API**: `sparse_from_k` stays in `apriori()` and `mine_two_phase` and raises unless None (`core/apriori.py:95-99`), the G7/C4 precedent.
  - Bench drivers pass `cfg.get("sparse_from_k")` (None) and keep working.
  - Rejected: deleting the keyword. It would break 6 bench driver lines (`child_run.py:196,224`, `consolidation_run.py:122,130`, `routes_run.py:146,150`).
- **`ET_MINER_ESCO_MATERIALIZE` guard kept**, with the message reworded (`_env.py:257`); its test is in `tests/test_routing_contracts.py::TestSparseFromK`.
- **`with_src_rows` kept on the group builders** (`bench/candidate_waste.py:279`, `bench/moonshots/replay_stakes.py:89` read it). It is removed only from `upload_k3plus_groups`, along with the dead `"tc"` key.
- **`bench/level_split.py`** imports `sparse_csr` optionally and patches the CSR phases only when it exists, so it keeps working on older src/ trees.
- **Rescued non-ESCO tests**:
  - the resume count round trip moved to `tests/test_flush_branches.py::TestResumeCountRoundTrip`;
  - the retail oracle boundary moved to `tests/test_retail_oracle_boundary.py` (renamed from test_esco_benchmark.py);
  - the caller-array ownership checks stay in `tests/test_gpu_ownership.py`.
- **Pins**: `ET_MINER_LEAVE_OUT` was removed without a removed-knob guard (the G2 `ET_MINER_ROW_ORDER` precedent). `bench/routes_matrix.py::PHASE2_LEAVE_OUT` stays as the record.
- **`bench/ESCO.md`** is kept as a record, with a "Removed again (G6)" note. `bench/README.md` points to it.

## Dead Ends
- Do not watch the `$!` of `setsid nohup ... &`: that was a wrapper pid which exited at once and looked like the runner dying. Watch the runner python pid (`ps aux | grep runner.py`).
- `uv run ty check` fails: `ty` is not installed in this project. Not run.

## Key Findings (for the report)
- (should) `applications/alphafold/deploy/run_mining.sh:204,219` pass `sparse_from_k=4` and now raise. Out of scope, not edited.
- (nice) Phase 4 SON on one GPU peaks lower than Phase 2 (dsl 725 vs 1237 MB, dslxl 1889 vs 3435 MB): the S1 VRAM rise the owner accepted is smaller now.
- (should, known) 2-GPU `bitvecs=` on sk2ml3 (8.75 GB input, 12 GB cards) raises `MemoryError` upfront: device 0 holds the caller's array plus its own shard copy.
- Earlier findings still to report (previous handoff): `apriori_from_csr` at min_count 0 emits zero-count candidates; the CPU route takes > 900 s on dsl; clippy and fmt drift on main; stale historical bench tools.

## Test Status
- At `0ea3664` (G6):
  - ruff clean;
  - `uv run python bench/selfcheck.py` READY (11 kernels, 2 devices);
  - tier `uv run pytest tests/test_tier_equivalence.py -q` 75 passed (87 minus the 12 ESCO legs);
  - not-slow 1498 selected, exit 0;
  - no-CuPy 833 passed / 516 skipped.
- At `af85dad` (pins):
  - not-slow `uv run pytest -q -m "not slow" -k "not test_support_00"` 1490 passed;
  - no-CuPy 833 passed / 508 skipped;
  - tier + routes + chunked 147 passed.

## Commits This Session
- `af85dad` Pins: remove the bench-only ET_MINER_LEAVE_OUT
- `0ea3664` G6: remove ESCO, the GPU's dense-to-sparse CSR transition
- `e0fd369` handoff: 2026-10-28 14:51

## Key Files
| File | Role |
|------|------|
| `bench/routes/final.py` | Phase 4 matrix (65 configs, base_ids of the Phase 2 survivors) |
| `bench/runner.py:270-282` | row revision stamp (sha + digest of `git diff HEAD`) |
| `bench/routes/decide.py:77` | `_sig`, the signature rule |
| `CHANGELOG.md` | "Route consolidation, round 2": G6 and LEAVE_OUT bullets added |
| `CLAUDE.md` | tier chain without the ESCO legs; intrinsics list re-pointed (`_src/group_pairs.cu`, `_src/k3_rows.cu`) |

## Extra Context
- Gate recipe:
  - no-CuPy: `uv run python /tmp/claude-0/-root-projects-ET-Miner/f1b33288-d153-44c1-9cfe-35bc8dad319e/scratchpad/gates/nocupy.py` (it sets `sys.modules["cupy"]=None` and runs the not-slow suite);
  - selfcheck: `uv run python bench/selfcheck.py`.
- `.claude/handoffs/` is gitignored; LATEST.md is tracked.
