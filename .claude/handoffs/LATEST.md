# Session Handoff: ET-Miner, 2026-10-08 23:39

**Branch:** `perf/input-layer` @ `bf87644` (in sync with origin)
**Tree:** clean; 1 stash (`stash@{0}`: README local edits, made on main)
**Focus:** handoff item 2 (input layer) is done; PR #30 is open and waits for the owner's merge
**Reason:** end of session
**Previous handoff:** `HANDOFF-2026-10-08-2011.md` (its item 2 is done; the other items carry over below)

## Verify on Resume
- Run `git status --short --branch` and `git rev-parse --short HEAD`. Expect HEAD `bf87644` on `perf/input-layer` and a clean tree.
- If PR #30 has been merged: `git switch main && git pull --ff-only`, then branch for the next item.

## Current State
- PR #30 (https://github.com/Et9797/ET-Miner/pull/30): CI green on `bf87644`; the body includes the council review. Not merged: the owner decides.
- `/council review` of PR #30: unanimous APPROVE in round 1. The flags are fixed in `bf87644`.
- No run is active. The council teammates (architect-3, math-3, alternatives-3, auditor-3) were not shut down; ignore their idle messages.

## Next Steps
1. Owner: merge PR #30 (or give feedback).
2. **(should) GPU verification** (needs a GPU box): the GPU legs of `tests/test_tier_equivalence.py` (SON 1 and 2 GPUs, forced chunks) and `bench/selfcheck.py`. PRs #25, #28, #29 and #30 add to it. The GPU route's CSR build (`core/matrix.py` `_build_csr_from_transactions`) is unchanged.
3. **(should, owner decides) SON pass 1 is 61–100 % of SON's wall time.** Levers:
   - `local_support_factor`;
   - count a candidate only in the chunks that did not emit it: −25 % pairs on skew, −40 % on or005. Without a uint64 bitmask, which caps at 64 chunks.
4. **(should, small, GPU)** Apply the SON bound to SON's GPU passes.
5. **(nice) Left open by the council on PR #30:**
   - every `_map_rows` caller must pass `ids` (`son.py:720`);
   - `_INT_DTYPES` (`cpu_miner.py:99`) and `_emit`'s `is_integer()` are two idioms for integer dtypes;
   - a lookup table sized to the items' range instead of the column's would skip SON pass 2's per-chunk `list.min`/`list.max` and admit columns whose span exceeds 2²²;
   - a tiny input with a 2²² span costs a fixed ~45 MB.
6. **(nice)** Carried from PRs #26 and #28: de-duplicate the K=1 seeding; two predicates are stated twice; thread `count_per_candidate`; the Gram peak is 1.8–1.9× its budget; the `test_support_001` oracle convention.
7. Remind the owner about `stash@{0}`: popping it on main may conflict in README.md.

## Session Instructions
- Chat in Dutch, concise. Files, code, comments, commits and PRs in English.
- Rule 2's 1 s floor stays; the owner chose to add a workload at scale (`dslk2`) instead of lowering it.
- Measure first; pre-register a `bench/cpu/PROTOCOL.md` amendment before any timed run; correct pre-registered text only by erratum.
- No commits or tracked-file edits during a timed run. A PR or merge only on the owner's word: "Open een PR" covered PR #30 only.

## Decisions & Rationale
- **Integer path** (`cpu_miner.py:146-219`):
  - applies to Int8–Int64 and UInt8–UInt32 with span ≤ `INT_SPAN_LIMIT` = 2²² (`:74`);
  - K=1 is a bincount over the id range; mapping is a lookup table on Polars' exploded values;
  - null lists and null items map to −1.
  - Rejected UInt64: it cannot be cast safely to intp for bincount.
- **Running-count row starts** in `_map_rows` (`:222`), shared by both paths, replace per-entry row ids plus bincount. A chunk whose explode does not match its lengths raises `RuntimeError` (`:249`).
- **A/B/A design with harness overlay:** the base tree `c973ed3` lacks `dslk2`, so the base runs used the input commit's `son_stakes.py`; their rows carry `c973ed3+dirty`. The repeated base is the drift control, because both arms run the changed code.
- **Bitvector build untouched:** on this box the pooled build beats the sequential one (0.17 → 0.10 s on deepk).

## Dead Ends
- Do not read Arrow buffers (`Series.to_arrow`) on the CPU route: it imports pyarrow, which adds 27 MB RSS (80 → 107 MB) and failed the memory rule in Phase I1.
- Do not use `deep_sparse_large` at full depth for the input layer: K≤5 alone took over 130 s against ~9 s of CSR build.
- Do not `np.compress(..., out=out[slice])` into the CSR output: it was slower (4.83 against 4.35 s on dsl).
- `git switch main` with a modified `LATEST.md` is refused, and a following `git merge --ff-only origin/main` then fast-forwards the current branch. Commit or stash first.
- Carried: `uv run ty` fails, use `uvx ty check <file>`; `gh pr edit --body-file` fails, use `gh api -X PATCH repos/Et9797/ET-Miner/pulls/<n> -F body=@file`; `gh pr checks --watch` does not exist.

## Key Findings
- I2 (`bench/results/2026-10-08-input-layer/FINDINGS.md`), medians of 3, against `c973ed3`:
  - dslk2 in core 17.46 → 11.61 s (T1) and 12.47 → 8.69 s (T4); SON 16.54 → 11.73 s and 11.07 → 8.53 s;
  - deepk and skew 0.13–0.64 s faster; others within 0.10 s;
  - RSS 0.92–1.03×; drift −2.6 % to +1.9 %.
- CSR build at T1: deepk 0.55 → 0.24 s, skew 0.60 → 0.28 s, dsl 9.2 → 5.0 s (the Arrow version was 4.3 s).
- Polars 2.0 `Series.explode()` drops empty lists and emits one null per null list; the signature shows `empty_as_null=True` all the same. `_map_rows` passes `empty_as_null=False, keep_nulls=True` explicitly.
- `deep_sparse_large`: generating it takes 4.5 min with a 24 GB peak; 382 MB parquet, SHA-256 `f6828374…48b1`, gitignored, now present locally.

## Blockers & Pending Decisions
- Merge of PR #30: owner.
- GPU legs: need a GPU box.
- `local_support_factor` default; `batch_size` deprecate or remove: owner.

## Test Status
At `bf87644` (23:20):
- `uv run pytest -q -m "not slow"` (four `test_support_00*` deselected): 829 passed, 276 skipped.
- Tier/SON/streaming/free-set suites: 59 passed, 138 skipped.
- ruff clean; ty 7 diagnostics on `cpu_miner.py` (equal to main); CI green.

## What Was Done
- `src/et_miner/core/cpu_miner.py`: `_IntIds`/`_int_ids` (`:146`, `:158`), integer `_count_items` (`:169`), `_column_ids` (`:199`), `_map_rows` running count plus length check (`:222`).
- `src/et_miner/streaming/son.py:720`: pass 2 passes `_int_ids(column)`.
- `tests/test_cpu_miner.py`: int path vs Polars path across dtypes, a chunked and sliced column, `_int_ids` cases, items outside the range, and explode/lengths raising.
- `bench/cpu/input_stages.py` (stage diagnostic); `bench/cpu/son_stakes.py:108` (`INPUT_WORKLOADS`) and `:314` (S0 arm keeps the Polars mapping).
- `bench/cpu/PROTOCOL.md` Amendments 6 (`:408`) and 7 (`:497`) plus erratum (`:537`); `bench/results/2026-10-08-input-layer/`; CHANGELOG entry.

## Commits This Session
- `bf87644` cpu: council flags for PR #30
- `6ad386b` bench: input-layer Phase I2 rows and findings; CHANGELOG numbers
- `b328cc1` cpu: integer CSR path from Polars' explode instead of the Arrow buffers
- `ff5e92c` bench: input-layer Phase I1 rows and findings; protocol Amendment 7
- `7fddd05` cpu: read integer item ids from the Arrow buffers in the CSR build
- `df8ecac` bench: input-layer stage diagnostic, dslk2 workload and protocol Amendment 6 (also carries the previous LATEST.md)

## Key Files
| File | Role |
|------|------|
| `src/et_miner/core/cpu_miner.py` | CSR build: `_int_ids`, `_count_items`, `_column_ids`, `_map_rows`, `build_transaction_csr` (`:275`) |
| `src/et_miner/streaming/son.py` | SON; pass 2 mapping at `:720` |
| `bench/cpu/son_stakes.py` | timing harness (`--matrix --arms built,incore`, `dslk2`) |
| `bench/cpu/input_stages.py` | stage diagnostic (instruments the `c973ed3` build) |
| `bench/cpu/PROTOCOL.md` | Amendments 6–7 and erratum |
| `bench/results/2026-10-08-input-layer/` | stages, I1 (base/raw/base2), I2 (base3/raw2), FINDINGS |

## Extra Context
- Box: Ryzen 5 4600G, 12 threads, 30 GB, no GPU. One A/B campaign run with `dslk2` takes about 8–9 min.
- Gate: `uv run ruff check src tests bench`; the not-slow suite with the four `test_support_00*` deselected; the tier/SON/streaming/free-set suites.
