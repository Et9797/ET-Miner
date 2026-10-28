# Session Handoff: ET-Miner, 2026-10-28 14:51

**Branch:** `refactor/route-consolidation` @ `7b6d640` (in sync with upstream)
**Tree:** clean, 0 stashes
**Focus:** Route consolidation round 2: Phase 3 removals (G6 ESCO + pins cleanup left), then Phase 4
**Reason:** owner asked for a handoff (task continues)

## Verify on Resume
Run `git status --short --branch` and `git rev-parse --short HEAD`. HEAD should be `7b6d640` and the tree clean. If not, read the diff before trusting "Current State".

## Current State
- Phase 3 done for G3, C1, C2, C4, H, S1/S2, G1, G2, G7 and the must-fix shard bug; each commit passed ruff, tier gate, not-slow, no-CuPy (selfcheck after kernel changes).
- Waiting on an explorer-opus agent's ESCO removal map (read-only trace of `sparse_from_k`, `gpu/sparse_csr.py`, `gpu/density.py`, csr_warp/bitvec_extract_tids kernels, ESCO branches in `gpu/row_split.py`, 13 test files, CLAUDE.md). If its report is lost with the session, re-trace: `grep -rln "sparse_from_k\|sparse_csr\|ESCO\|esco" src tests`.
- `refactor/drop-esco` / `refactor/drop-rust` do NOT exist on the remote (`git ls-remote --heads origin`: only main + this branch). Build G6 from scratch.
- Last gates (at `2aa6b72` + `7b6d640`): tier 87 passed, not-slow 1598 passed + the fixed shard test, no-CuPy 865 passed / 564 skipped, routes harness 15 passed.

## Next Steps
1. **G6 ESCO removal** (owner approved, with CHANGELOG): `sparse_from_k` keeps the "removed param raises" pattern (raise ValueError unless None); delete `gpu/sparse_csr.py`, `gpu/density.py`, ESCO-only kernels (check each kernel has no non-ESCO caller first; `bench/selfcheck.py` + `gpu/kernels/loader.py::_KERNEL_FILES`), ESCO branches in `gpu/row_split.py`; tier legs "ESCO 1 GPU"/"ESCO 2 GPUs" in CLAUDE.md + `tests/test_tier_equivalence.py`; CLAUDE.md's intrinsics list names `_src/csr_warp.cu`/`_src/bitvec_extract_tids.cu`: update if those go. One CHANGELOG bullet in "Route consolidation, round 2".
2. **Pins cleanup**: `ET_MINER_LEAVE_OUT` (bench only, no tier leg uses it; `_env.py::LEAVE_OUT`, `gpu/row_split.py` k2/K>=3 leave-out branches, `tests/test_leave_out.py`). `bench/routes_matrix.py::PHASE2_LEAVE_OUT` stays as the record.
3. **Phase 4**: `uv run python bench/runner.py --mode routes-final --out bench/results/2026-10-28-routes-final` (matrix `bench/routes/final.py`, 65 configs x 3, CPU + GPU; est. 1.5-2 h; ask the owner for a budget if unsure). No tracked edits/commits while it runs. Compare final vs Phase 2 rows by `id` (same base_ids, no pins); signatures must match.
4. adversarial-verifier review of the branch diff vs `f6c3d36`, then `/council review`, then `bench/results/.../REPORT.md` (numbered list, tagged (must-fix)/(should)/(nice), sorted) and a chat summary. No PR/merge without the owner.

## Session Instructions
- Chat in Dutch, concise; files/code/commits in English. Reports: numbered lists tagged (must-fix)/(should)/(nice), sorted.
- Owner decisions (recorded at the end of `bench/routes/PROTOCOL.md` and in `bench/results/2026-10-27-routes-gpu/FINDINGS.md`): rule 7 needs a memory rise >= 1 GB and > 1.25x (`bench/routes/decide.py::RISE_FLOOR_MB`); S1 boolean SON goes despite +1.7 GB VRAM; G2 keep = lex's named fallback; C3 keep `apriori_from_csr`; H +0.5 GPU-h (used 0.075); G6 + G7 API removal approved with CHANGELOG.
- From the brief (earlier sessions): no PR/merge without the owner; SON stays; `use_generator_pruning` default False; `n_gpus` exact (no fallback); row split stays; out of scope: `core/rules.py`, `io/`, GCS, `streaming/ramdisk.py`, config/cli beyond removed-path options, `bench/results/*` and finished campaigns' docs.

## Decisions & Rationale
- **G1 host CSR kept by role** (rule 7 role clause): only layer for > INT_SPAN_LIMIT distinct items and for output_dir/resume/anchors on unserved columns; `coded` became the default for unserved columns (`core/apriori.py` ~650-720). Tell the owner: their "host goes" was overridden by the role clause.
- **C4 n_jobs**: `count_support_batched(n_jobs!=1)` now raises like `sparse=` (only the removed counter read it). Not pre-approved: flag in the report.
- **C1 scope**: only the route's K=2 bitvec path went; `count_itemsets`' `_k2_on_bitvecs` rule (SON pass 2) was not measured and stays.
- **C2**: verified SON CPU before/after (same lattices, skew 18.4 -> 10.8 s) since `count_candidates` is shared.
- **G2 fallback**: `gpu/input_layer.py::LEX_BYTES_PER_ROW = 56` (measured 54-55 B/row); tier keep legs force it by monkeypatching that constant to `1 << 50`.

## Dead Ends
- Do not run uv/maturin inside `rust_ext/` (cd back to the root after cargo).
- Do not assert pool `used_bytes() == 0` in GPU tests: other tests leave blocks; compare before/after.
- Do not run all of `tests/test_input_layer.py` including slow tests in one process: the large presets together get OOM-killed (pre-existing, unrelated to G1).

## Key Findings (for the report)
- (must-fix, fixed `2aa6b72`) `gpu/row_split.py::shard_prebuilt_bitvecs`: 2-GPU `bitvecs=` OOM (contiguous temp per shard on the input device). Now direct + 256 MB blocks + upfront MemoryError; 6 GB input splits, 8.75 GB raises clearly.
- (should) `apriori_from_csr` at min_count 0 emits every candidate with count 0 (CPU route emits only occurring itemsets).
- (should) CPU route > 900 s on dsl (Rust 227 s, infer/free ~280 s); Rust deepk20 slow, not hung (ml 7/9/11: 16/197/878 s).
- (nice) SUBSECOND keyed on workload; clippy 2 errors + cargo fmt drift on main; stale historical bench tools (`bench/cpu/l3_crossover.py`, `son_stakes.py`, `sparse_non_pairs.py`, `bench/repro/d04*`).

## Test Status
`uv run pytest -q -m "not slow" -k "not test_support_00"` at 2aa6b72: 1598 passed + shard test fixed after (63 passed in its modules); tier 87 passed; no-CuPy 865/564; `tests/test_routes_bench.py` 15 passed at 7b6d640.

## Commits This Session (newest first, max 10)
- `7b6d640` routes: the Phase 4 matrix (routes-final)
- `2aa6b72` Fix: bitvecs= on several GPUs without the shard's temporary copy
- `324afad` G7: remove prune_apriori=False
- `57ac99e` G2: lex row order, with the kept order as its no-room fallback
- `74cd518` G1: every item column on the device input layer, the host CSR by role
- `fd343d0` S1/S2: SON on the GPU runs the array skeleton only
- `2b38fed` H: drop the Rust bitvector builder (R5)
- `60adeb5` C4: remove the scipy sparse counter
- `4e2ed6e` C2: K>=3 prefix groups on the bitvectors whenever they exist
- `11f5657` C1: the CPU route counts K=2 on the Gram only
(earlier this session: a75b22b G3, 2d8f08e H measurements, 536d4ae/1f50821 owner decisions, 6d1375d findings, 3552435 Phase 2b, a7e44ca decide fix, 0eeb5d3/eaead85 C4, 795670a Phase 2a, ddb9c65 harness)

## Key Files
| File | Role |
|------|------|
| `bench/results/2026-10-27-routes-gpu/FINDINGS.md` | decisions per DP, rule-7 numbers, findings, owner decisions |
| `bench/routes/PROTOCOL.md` | rule + "Owner decisions after the measurements" |
| `bench/routes/decide.py` | mechanical rule (head-to-head per GPU count, rule 7 floor) |
| `bench/routes/final.py` | Phase 4 matrix |
| `CHANGELOG.md` | "Route consolidation, round 2" + "Fixed (route consolidation, round 2)" at the top of Unreleased |
| `CLAUDE.md` | tier chain (updated per DP) |

## Extra Context
- Gate recipe: tier `uv run pytest tests/test_tier_equivalence.py -q`; no-CuPy `uv run python /tmp/claude-0/-root-projects-ET-Miner/f1b33288-d153-44c1-9cfe-35bc8dad319e/scratchpad/gates/nocupy.py`; `uv run python bench/selfcheck.py`; Rust rebuild `env -u CONDA_PREFIX uv run maturin develop --release -m rust_ext/Cargo.toml`.
- `.claude/handoffs/` is gitignored; LATEST.md is tracked: `git commit -- .claude/handoffs/LATEST.md`.
- Memory: `/root/.claude/projects/-root-projects-ET-Miner/memory/rust-miner-removal-approved.md` (C3 outcome).
