# Session Handoff: ET-Miner, 2026-09-11 19:44

**Branch:** `council-session-blockers` @ `0286a96` (in sync with origin; pushed this session)
**Tree:** 6 modified, 5 untracked, 0 stashes — **ALL of it is ANOTHER SESSION's live work, none of this session's**
**Focus:** Remediated the 5th council's nine blockers (4 commits, pushed, PR #8 updated), ran the gate set, then ran the 6th council on those 4 commits.
**Reason:** End of session. 6th council closed 4 BLOCK / 0 APPROVE with ONE converged fix list; the user has not yet chosen among options (1) implement all + re-gate, (2) blocking items only, (3) leave as pushed.

## Verify on Resume
Run `git status --short --branch` and `git rev-parse --short HEAD`. HEAD should be `0286a96`. The tree will NOT be clean and that is expected — it is the other session's work (see Blockers). If HEAD moved, read the diff before trusting "Current State".

## Current State
- **Nothing in progress. No file is being edited. Nothing uncommitted from this session.**
- 4 commits pushed (`fed0a8e`, `8fcca70`, `de0b6db`, `0286a96`); PR #8 body updated via REST (`gh pr edit` fails on a projectCards GraphQL error — use `gh api --method PATCH repos/Et9797/ET-Miner/pulls/8 -F body=@file`). PR shows 43 commits.
- Full gate set green at `0286a96` (see Test Status). Logs: `runs/20260911T1830Z_council5_gate/`.
- 6th council (`/council review this session's work`, 7 files): **4 BLOCK / 0 APPROVE, 3 rounds**, reviewers idle (in-process teammates — they do NOT survive `/resume`; do not message them, start a new council).
- **Waiting on the user's option.** If (1) or (2): implement the list below on this branch, re-run the gate set, push, then the user types `/council` again.

## Next Steps (the converged fix list — every item reproduced by ≥2 reviewers)
1. **A** `src/et_miner/gpu/kernels/gpu_resident.py:11-13` still names `tests/test_kernel_input_guards.py` as holder of the tuple assertions (they are `tests/test_kernel_guard_claims.py::test_every_exported_entry_point_is_classified_exactly_once`, `..._really_call_all_three_guards`, `..._call_no_guard_and_say_why`); `:69-71` WAS re-pointed. Re-point, then add an AST-based citation check in `test_kernel_guard_claims.py`: `test_[a-z0-9_]+` tokens from `gpu_resident.__doc__` and each `_EXEMPT_ENTRY_POINTS` reason, resolved against top-level `ast.FunctionDef` names of BOTH guard test files (each token defined in exactly one; that file's basename in the same prose block). **Never resolve by importing the test module** (couples the gate-free file to the gated one). No `"test_kernel_input_guards.py" not in doc` negative assert.
2. **B** `tests/test_kernel_guard_claims.py::_call_sites`: key is `(path, lineno, owner)` — two same-line calls collapse (reproduced → 1). Add `child.col_offset` to the tuple; update return annotation and unpacking sites; leave `len(sites) == 3`. Counts unchanged today (4/3/4/4).
3. **C1** `test_the_modules_under_test_import_without_cupy`: subprocess must also import `et_miner.gpu.kernels` (tests import it at `:93`; `et_miner` at `:179`); `gpu.dispatch` is only `ast.parse`d — keep it but say the premise is the modules the tests IMPORT. Delete the "1 failed, 7 passed" number from both files' tallies (keep the claim); **do NOT add a `--collect-only` count pin (C2 dropped, unopposed)**.
4. **D** `tests/test_row_split_memory.py::_excess`: ONE line — call `fn` once on the tiny input before `_peak_of` (per-`fn` warm). `:200`/`:237` (direct `_warm()` callers) untouched. Then re-state — do NOT re-quote as literals — the regression paragraph: in-test figures are smoke astype ~1,364 / cumsum 6,307 / widths 12,972 B, 2M astype ~1,207 B; **astype is BELOW the real build** (my item-4 "correction" was inverted; the handoff-5 "1,175, below" was right). Differences over the reference: cumsum 3,929, widths 10,594.
5. **E** `:430` → `assert astype <= real` — **no constant, no `ASTYPE_BLIND_B`**; `SMOKE_DIFF_B` keeps its one meaning. Replace measured literals with validity intervals stated only in relations verified invariant in N and chunk count: `SLACK_B` rejects a doubling at smoke for ANY C ≥ 0 (18,872 > 16,384, no measurement needed); `SMOKE_DIFF_B` valid while C < R − 2,048 = 3,512 B structural (C = 2,378 today). Delete the "~100 B jitter" (spread in a fixed context is ZERO) and the false universal at `:51-54` (4,096 would separate; headroom is the honest reason for 16,384). Quote NO threshold derived from δ (see Dead Ends).
6. **F** New `test_the_slack_rejects_a_doubling_at_the_smoke_shape`, smoke shape only, two assertions, zero literals: `assert SLACK_B < _identity(arrs)` (pins the constant) then `assert _excess(_regress_double, arrs, supports) > SLACK_B` (pins the apparatus; `_regress_double` = `_fill`, in-place cumsum, `.copy()` both, arrow on the copies, keep originals live; ~20,092 B measured). Independent of D; `SLACK_B = 65_536` currently leaves all 16 tests green.
7. **G** `:395` — difference the two R-scale regressions against the real build AT THE SMOKE SHAPE (28 B from the 40M figure), dropping the second 40M build (0.45 s, 624 MiB, unmarked); keep the one at `:266`. **Land with E**: `SMOKE_DIFF_B`'s docstring must be re-scoped in the same edit, citing the N-invariance at `:247`.
8. **H** `CHANGELOG.md` PR 8 memory bullet: separate "3.5x the smoke identity" from "It is 16,384 B now" (16,384 also admits all three at smoke — only the differencing closes it); fix "so the constant cancels" (true of the real-build row only).
9. Shell (non-blocking, verified by auditor): `bench/run_full.sh:25-26` and `run_smoke.sh` — `rc=0; uv run python bench/runner.py ... || rc=$?; uv run python bench/report.py || true; exit "$rc"` so drift's `return 1` under `set -e` no longer destroys the report (two-line form loses exit 2).
10. Re-run the gate set (serial; do NOT edit tracked files while `run_smoke.sh` runs), push, then a 7th `/council` is the user's call.

## Session Instructions
- User chose (AskUserQuestion): `SLACK_B = 16_384` + smoke differenced against 40M; push to PR #8 after green. Both done.
- Never `git checkout <file>` here; back up to scratchpad and `cp` back. Commit by staging only your own hunks (`git apply --cached` on a filtered patch for CHANGELOG.md).
- `/council` is `disable-model-invocation`; the user types it.

## Decisions & Rationale
- **Unit B named the enclosing function (`dispatch_k3plus_gpu_resident`) instead of pinning `gpu/dispatch.py:224`** as the handoff said: user's recorded symbols-not-line-numbers rule; a pinned line goes red on every edit above it. Council endorsed.
- **New file `tests/test_kernel_guard_claims.py`** rather than per-test markers in the old file: council's `alternatives` proposed merging back, then withdrew — `pyproject.toml:107-112` is scoped to `multigpu`, and twelve device-only files carry the module `importorskip` + file-wide `pytestmark` shape.
- **Runner refuses to start at rev `"unknown"` (exit 2)** and **refuses the tick on drift (exit 1)**: a campaign that cannot name its revision establishes nothing; auditor's mutations confirm each mechanism is separately tested.
- **CHANGELOG entry written from the post-PR-6 commit messages only** (`0a21f35..0286a96`), staged as its own hunk.

## Dead Ends
- Do not measure tracemalloc regression figures one fresh process per cell and quote them: the form under test pays its own ~1.4 KB first-traced-region cost there; inside the test file an earlier test has paid it (figures even differ between `pytest` and `pytest -k <subset>`). Warm the exact function measured (item D) and derive numbers in the test.
- Do not quote δ (the astype form's additive term) or any threshold from it: only two statements have two-harness support — δ(1 array) = 220 in every context, and saturation at 348 from 16 arrays through 128. Every intermediate cell (4/6/8 arrays) was retracted by BOTH reviewers as process- or position-dependent (236/252/268/300 all observed at the same shape); four mechanism hypotheses were falsified (warm order, chunk count alone, first-call step, a "settled row"). The reviewers reproduced the defect under review while measuring it — the reason the rule is unconditional.
- Do not use `gh pr edit 8` — fails on `repository.pullRequest.projectCards` deprecation; use `gh api --method PATCH`.
- Do not trust `bash bench/run_smoke.sh` exit 0 as evidence for runner changes (all-fresh runs never reach NOT-GATED); `tests/test_campaign_gate.py` covers it, incl. four `main()`-level tests with stubs.
- Do not re-propose: cardinal "four" in `loader.py` prose; keying the campaign dir by mode; scoping `check_equivalence`; merging the two guard test files.
- Do not run background Python without `-u`.

## Key Findings
- **Council 6's key finding is against my own measurement**: `de0b6db` declared "astype 1,332 B, still below" false and wrote "2,708–2,724 B, ABOVE"; in the context the test runs in astype is ~1,364 B vs real 2,350 (BELOW). Same shape as the 19 prior items, one level down.
- `gpu_resident.py:11-13` stale citation — missed in the file I was editing; `:69-71` was updated.
- `_call_sites` key without `col_offset` contradicts its own docstring's "two calls on one line are two".
- Runner correctness core (`8fcca70`) survived all five of auditor's mutations; `"unknown"` false green closed on every path reached (`--out` explicit, missing `rev`, mid-run git failure).
- `_git_rev`'s `diff.returncode != 0` branch has no test and is the one branch whose PRE-fix behaviour was a false clean-tree claim (bare sha); fixture: corrupt `.git/index` → `git diff HEAD` rc 128 while `rev-parse` succeeds.
- `_drive_main`'s `next(it, last)` silently absorbs an ADDED `_git_rev()` call; positive control uses `revs=["A"]`, blind to call count. Fix: assert the iterator is exhausted.
- perf compare: `deepk-cpu-sparse-j1` +4.7% at 3 repeats; at 7 repeats +8.4% against a 20.1% spread → +0.0% beyond noise; no mining code changed.

## Blockers & Pending Decisions
- **USER DECISION pending**: option (1) full converged list + re-gate, (2) blocking items only (A, B, C1, D, E, F), (3) leave PR #8 as pushed.
- Another Claude session has live uncommitted work: `core/prune_observer.py`, `tests/test_prune_observer.py` (new), `core/apriori.py` (+46), `core/rules.py` (+52), `tests/test_rules.py` (+106), `CHANGELOG.md` (+42 at ~:367, in `### Added`), `.gitignore` (+2), `gpu/row_split.py` (+18 at `:106` and `:746-762`). Leave it alone; `git diff` must show exactly this.
- Still owed, out of scope: `output_dir=""` validation in `core/apriori.py`; in-tree artifacts for the `skewed_rows`/`deep_k` rows of `row_split.py`'s kbar table; `tests/test_gpu_device_affinity.py:440` says the device file "covers the other three" rank checks (it checks one) — pre-existing.

## Test Status
All green at `0286a96` (`runs/20260911T1830Z_council5_gate/`, 18:30–18:48 UTC):
- `uv run pytest -q`: 720 passed, 6 skipped (was 710; +10 this session)
- `tests/test_tier_equivalence.py`: 10 passed; `ruff check src tests bench`: clean; `bench/selfcheck.py`: READY, 19 kernels, 2 devices; `bench/repro/run_all.py`: 13 live, 11 fixed, 0 broken
- `bash bench/run_smoke.sh`: exit 0, all 6 configs fresh, ✓ at `0286a96-dirty.ae18a2dc43e4`; in-script GPU suite 199 passed, 2 skipped
- perf `--compare`: hashes `same` 10/11 (11th the documented `twophase-deepk` exemption)
- Mutations at tip: memory file 7/16 red on revert-cumsum or reintroduce-astype, 3 red on `SLACK_B`×1000; seed-cast mutation → guard_claims 1 failed/7 passed, input_guards 8 passed; gate: subtraction form 4 red, digest drop 1 red.

## What Was Done
- `tests/test_kernel_guard_claims.py` (new): 7 moved device-free tests + `test_the_modules_under_test_import_without_cupy`; `_call_sites` via `ast.walk`; exemption prose names `gpu/dispatch.py::dispatch_k3plus_gpu_resident`.
- `tests/test_kernel_input_guards.py`: 8 device tests only; module-level device skip removed (conftest skips gpu-marked items).
- `bench/runner.py`: `_git_rev` prints the failure to stderr, checks both returncodes; `_coverage` `fresh()` requires `rev == here and rev != "unknown"`; `main` refuses at `"unknown"` (exit 2), reads `now = _git_rev()` post-loop, `covered` requires `now == here`, drift → exit 1; `_campaign_out(rev=None)`; docstring reworded (invocation as subject).
- `tests/test_campaign_gate.py`: `shutil.which` guard, hex sha assert, fixed commit date (sha `97bb2c0`), +7 tests (2 `_coverage`, 1 `_git_rev` stderr, 4 `main()` with stubs).
- `tests/test_row_split_memory.py`: `SLACK_B = 16_384`, `SMOKE_DIFF_B = 2_048`, smoke differenced in `test_the_excess_does_not_scale_with_n`, new `test_the_smoke_difference_rejects_the_r_scale_regressions`, `..._at_k2` → `..._at_low_kbar` (+ smoke shape), docstring rewritten (partly wrong — see Findings).
- `CHANGELOG.md`: `*PR 8*` block under `### Fixed`.
- PR #8 body: fifth-round section, corrected notes, new gate table.
- Memory updated: `plan-review-standard.md` (+derived bounds; +measure in the test's context).

## Commits This Session
- `0286a96` docs: record the PR 6 council rounds in the CHANGELOG
- `de0b6db` fix: the memory slack admitted every regression at the production shape, and four sentences about it were false
- `8fcca70` fix: a git failure gated green, and a tree edited mid-run could still take the tick
- `fed0a8e` fix: the guard claims sat behind a device gate, and counted call sites in prose

## Key Files
| File | Role |
|------|------|
| `tests/test_row_split_memory.py` | items D, E, F, G — `_excess` ~`:188`, `SMOKE_DIFF_B` `:124-130`, `:430` sentinel, `:395` second 40M build, `:51-54` false universal |
| `src/et_miner/gpu/kernels/gpu_resident.py` | item A at `:11-13` (stale) vs `:69-71` (correct) |
| `tests/test_kernel_guard_claims.py` | items A (checker), B (`:193` key), C1 (`:72` subprocess) |
| `tests/test_kernel_input_guards.py` | tally prose `:19-27` |
| `bench/runner.py` | correctness core; `:171` untested `diff.returncode` branch |
| `tests/test_campaign_gate.py` | `_drive_main` `:271` iterator flag |
| `bench/run_full.sh`, `bench/run_smoke.sh` | item 9 rc-preserving change |
| `CHANGELOG.md` | item H; other session's hunk at ~`:367` |
| `runs/20260911T1830Z_council5_gate/` | gate logs (untracked) |

## Extra Context
- Twenty-eight-for-twenty-eight now: every blocking item across six councils is a sentence quantifying over a different set than the one it names — including the ones written while fixing the previous ones, and including two cells of the reviewers' own measurement table this round.
- The variant with nothing to quote (`assert astype <= real`) is the one that cannot be quoted wrongly; when a figure must appear, state the relation verified invariant across the dimensions the file actually varies, and let the failure message print the number.
