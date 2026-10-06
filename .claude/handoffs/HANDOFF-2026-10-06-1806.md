# Session Handoff: ET-Miner, 2026-10-06 18:06

**Branch:** `main` @ the commit that adds this handoff (on top of `061d35d`, the merge of PR #24, phase C)
**Tree:** clean apart from the 28 untracked lattice dumps in `bench/results/2026-10-05-candidate-waste/` (deliberately uncommitted, regenerable with `--mode waste`)
**Focus:** the remaining items of `bench/optimizations/SPEC.md` after phase C
**Reason:** end of session (the owner asked for a handoff for the remaining SPEC items, committed on main)

## Verify on Resume
Run `git status --short --branch` and `git rev-parse --short HEAD`. HEAD should be the latest handoff commit on `main` (on top of `061d35d`) and the tree should match "Tree" above. If not, the state moved after this handoff: read the diff before trusting "Current State".

## Current State
- Phase C (O6) is done, reported (`bench/optimizations/REPORT-C.md`) and merged into main as PR #24 (`061d35d`). Nothing runs, nothing is mid-edit.
- O7 a/b/c dropped by SPEC amendment (`SPEC.md`, O7 section). O5 ended in phase B, O4 deferred.
- The owner asked "are there open questions in the spec?"; the five open items are the Next Steps below.

## Next Steps
1. Ask the owner which item goes first; branch it from main.
2. **O6 follow-up** (REPORT-C finding 1, should): on dsl the per-candidate kernel still takes 3.23 s on 1 GPU, in groups below `GROUP_MIN_PAIRS` (9 pairs from K=4, i.e. ≤ 4 suffixes; 20 at K=3). The sweep counted every pair; dsl skips 35–45 %. Cheap check: dsl with `ET_MINER_SMALL_GROUP_KERNEL=group` (≈17 s per GPU count) against the campaign's `group` arm (`bench/results/2026-10-06-group-kernel/raw.jsonl`). Agree the protocol (3 reps rep-major, alternate arm order) and budget first.
3. **O8** (`SPEC.md`, O8; research, offline first): fraction of all-zero 32-word prefix tiles per level, before and after a candidate row reordering. Both the tiled kernel (`_src/shared_tiled.cu`, `s_skip`) and the group kernel (`_src/group_pairs.cu`, per-warp ballot on the staged prefix AND) skip such tiles, so O8 now reaches both. Data: regenerate the synthetic presets (`et_miner.synthetic`); groups from the lattice dumps. Exact counts if cheap, else a stated sample.
4. **"ESCO on K=3 explosions"**: the owner decides whether it becomes a SPEC item. Content from `REPORT-B.md` (findings 1–2, "Open"): a skip of prunable candidates in the CSR kernel, the conversion from the row-wise K=2 CSR instead of the bitvecs, and a compacted sparse reduce on 2 GPUs. Phase C adds: generating K=3 candidates as triangles of the frequent-pair graph (145 M of 11.8 B on sk2ml3) would also let the dense tiled kernel launch only the 376,436 counted tile-pairs of 11,571,649 (`bench/results/2026-10-06-phase-c-stake/k3_tiles.txt`).
5. **O4**: stays deferred; it returns only if item 4 finds ESCO levels that win on time but do not fit.
6. **`use_generator_pruning` default** (not in SPEC; pruning REPORT finding 5): the owner's decision. Phase C data: dsl 1 GPU 13.16 s with count inference vs 16.76 s without (both on the group kernel, `REPORT-C.md`).

## Session Instructions
- Chat in Dutch, concise; files, code, commits and PRs in English. Findings numbered and tagged (must-fix)/(should)/(nice).
- New this session, now in `CLAUDE.md` "Git": commit and push often after every verified step; never during a timed run (rows are stamped per finish, `bench/runner.py:350`); PR or merge only on the owner's word. The owner explicitly asked this handoff to be committed on main.
- The owner gates steps: build, report, then the next step. Ask before spending GPU budget; nothing else on the GPUs during a measurement; correctness gate before timed runs.
- "Meten is weten": exact offline stakes before proposing a build; replace samples with exact counts when cheap.
- When asked "wat stel jij voor?", give one recommendation, not a menu.
- `ruff format`: only fix your own lines. Pre-existing drift in `row_split.py`, `sparse_csr.py`, `runner.py`, `level_split.py`, `consolidation_matrix.py`, `selfcheck.py`, `tests/test_optimizations_bench.py`, `test_tier_equivalence.py`, `test_free_set_semantics.py`, `test_subset_prune.py`.

## Decisions & Rationale
- **O7 dropped**: (a) 0.22–0.24 s, (b) ≤ 0.18 s, (c) ≤ 1.37 s of 20.68 s; rule 2 needs ≥ 1 s and ≥ 10 %, and a tie goes to rule 5.
- **O6 as a group kernel, not small tiles**: dsl's per-candidate groups hold 3.0 counted pairs each, so even 8×8 tiles stay 95 % empty; only work that follows the counted pairs reaches them.
- **Separate `bench/group_crossover.py`**: keeps `kernel_crossover.py` reproducing the 09-27 crossover. 1,000 groups per point, because the old 200,000 candidates gave 99 blocks at m = 64 (under one wave on 48 SMs).
- **Dispatch facts stay k and the pairs per group**; K=8 uses the first crossing (370 pairs), per the protocol.
- **`--mode waste` pins `ET_MINER_SMALL_GROUP_KERNEL=percand`**: `candidate_waste.py` models the two-kernel split.

## Dead Ends
- Do not commit or edit tracked files during a timed run: the runner stamps each row with `_git_rev()` as it finishes.
- Do not use `gh pr edit` for bodies: it fails with "GraphQL: Projects (classic) is being deprecated in favor of the new Projects experience". Use `gh api -X PATCH repos/Et9797/ET-Miner/pulls/N -F body=@file`.
- Do not run `uv run ty check`: "Failed to spawn: ty" (not in the dev dependencies).
- Do not retry ESCO from K=3 without the three fixes in item 4; do not retry O4 for dsl memory (49.5 GB at K=5 with ideal sharing).
- Do not triangle-enumerate K=3 in a Python loop; use scipy submatrices per prefix (`bench/results/2026-10-06-phase-c-stake/k3_tiles.py`, 7 s).
- Do not `pkill -f <script>` in the command that starts it, or `git stash` while a gate runs.

## Key Findings
- Tiled kernel on dsl: 1.97–2.19 ns per counted 32×32 tile-pair per word at K=5–10, whatever the slot use (3.4 %). Per-candidate kernel: 1.38–1.50 TB/s of k-row reads per counted candidate, ≈ 3× DRAM (`phase-c-stake/o6_stake.txt`).
- The group kernel took dsl's K≥3 counting from 9.45 to 6.03 s on 1 GPU; the offline model said 3.2 s. Fewer redundant reads mean fewer cache hits.
- HW slowdown/thermal (0x48) hit all three `group` reps of sk2ml3 on 1 GPU and no `base` rep; `base` always ran first within a rep. That likely explains its +0.21 s; alternate the arm order in future protocols.
- Remaining dsl budget after O6: per-candidate 3.23 s, group 2.74 s, tiled 0.06 s (1 GPU, K≥3).

## Blockers & Pending Decisions
- Whether "ESCO on K=3 explosions" becomes a SPEC item, and the order of items 2–6 (owner).

## Test Status
Gate on `238e95a` (`NCCL_P2P_DISABLE=1`, 2026-10-06 ~17:30): `ruff check src tests bench` clean; `pytest -q -m "not slow"` with the 4 known `test_smoke_correctness` ids deselected: 1064 passed, 12 deselected; `tests/test_tier_equivalence.py`: 54 passed; `-m "gpu and slow"`: 5 passed; `-m "gpu and multigpu"`: 53 passed; `bench/selfcheck.py`: READY (13 kernels).

## What Was Done
- Merged PR #23 (phase B, `63a59e2`); the owner merged PR #24 (`061d35d`). Phase C protocol, kernel, dispatch, tests, tooling, sweep (0.155 GPU-h), campaign (0.384), final check (0.062): 0.601 of 1.5.
- `src/et_miner/gpu/kernels/_src/group_pairs.cu`, `kernels/group_pairs.py`, `row_split_chunks.py` (`group_kernels`, `GROUP_MIN_PAIRS`, `GROUP_TILED_MIN_PAIRS`), `_env.small_group_kernel`.

## Commits This Session
- `823ddb1` bench: phase C report (O6: the group kernel wins dsl, loses nothing)
- `0767b38` bench: phase C campaign and final check; CLAUDE.md: commit and push often
- `238e95a` gpu: measured three-way K>=3 dispatch table (phase C sweep)
- `910ec52` bench: phase C tooling (K>=3 kernel sweep, o6 matrices, count_group phase)
- `0222d71` gpu: group kernel for small prefix groups, three-way K>=3 dispatch
- `0668355` bench: phase C protocol (O6 group kernel, O7 dropped) and offline stakes

## Key Files
| File | Role |
|------|------|
| `bench/optimizations/SPEC.md` | O1–O8 with amendments; O8 is the only untouched item |
| `bench/optimizations/PROTOCOL-C.md`, `REPORT-C.md` | phase C protocol and outcome (template for the next protocol) |
| `bench/optimizations/REPORT-B.md` | ESCO K=3 loss and the three fixes (item 4) |
| `bench/results/2026-10-06-phase-c-stake/` | offline stakes (o6_stake.py, k3_tiles.py) |
| `bench/results/2026-10-06-group-kernel/` | sweep, campaign, final rows |
| `src/et_miner/gpu/row_split_chunks.py` | dispatch tables and `group_kernels` |
| `bench/group_crossover.py`, `bench/optimizations/decide_c.py` | sweep and DP-O6 decision tooling |

## Extra Context
- Remaining SPEC items: O8, "ESCO on K=3 explosions", O4, the O6 follow-up, the `use_generator_pruning` default. This handoff is committed on main at the owner's request (`git add -f`, since handoffs are excluded locally).
