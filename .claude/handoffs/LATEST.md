# Session Handoff: ET-Miner, 2026-10-08 15:58

**Branch:** `council-cpu-miner-fixes` @ `1a6fa21` (in sync with origin; branched from `main` at `c09b1ff`, the merge of PR #25)
**Tree:** clean apart from the handoff files; 1 stash (`stash@{0}`: README local edits, made on main)
**Focus:** PR #25 review fixes (council option a), now open as PR #26; the CPU tier's open (should) items
**Reason:** end of session
**Previous handoff:** `.claude/handoffs/HANDOFF-2026-10-07-2034.md` (tracked). Everything still valid from it is carried over below. Read it for the Phase 1 tables and the campaign history.

## Verify on Resume
Run `git status --short --branch` and `git rev-parse --short HEAD`. HEAD should be `1a6fa21` and the tree clean, except `.claude/handoffs/LATEST.md` (tracked, updated by this handoff) unless it was committed since.

Check PR #26 (https://github.com/Et9797/ET-Miner/pull/26):
- merged: branch the next work from `main`;
- still open: ask the owner whether to stack on `council-cpu-miner-fixes`.

If HEAD moved, read the diff before trusting "Current State".

## Current State
- PR #26 is open (base `main`, head `council-cpu-miner-fixes`) and waits on the owner. Do not merge it yourself.
- No code work is in progress. The last action was `gh pr create`.
- PR #25 (the CPU array miner, `perf/cpu-tier`) is merged into main at `c09b1ff`. Its GPU legs never ran: no GPU in either session.
- Council teammates are all shut down. `verifier` got a shutdown_request after its PASS.

## Next Steps
1. **PR #26:** handle the owner's review on the same branch.
2. **GPU verification (still open since PR #25):**
   - on a GPU box, run the GPU legs of `tests/test_tier_equivalence.py` and `bench/selfcheck.py`;
   - PR #25 touched `gpu/row_split.py` (`collect(engine="in-memory")` for Polars 2.0), and none of the GPU side has been executed.
3. **Tier 2 leg (should).**
   - The problem: `tests/test_tier_equivalence.py:168` (`test_tier2_rust_matches_oracle`) calls `apriori(sparse=True)`. That now runs the array miner, the same engine as Tier 1, and emits a DeprecationWarning. The Rust miner is uncovered.
   - The change: point the leg at `et_miner.apriori_from_csr` (export at `src/et_miner/__init__.py:70`).
     - Its signature (`rust_ext/src/lib.rs:311`): `(indptr: i64[], indices: i64[], n_rows, n_cols, min_support, max_length) -> (itemsets as column-index lists, counts)`.
     - `build_transaction_csr` returns int32 arrays: cast them, and map the columns back through `TransactionCSR.items`.
   - Constraints:
     - skip cleanly without the Rust extension;
     - keep exact itemsets AND counts against efficient-apriori, with the oracle convention;
     - rename the leg in `CLAUDE.md`, and show the owner that diff before committing (it is the correctness policy).
4. **SON onto the array miner (should).**
   - What SON (`streaming=True`) still uses:
     - the quadratic `_generate_candidates` (`src/et_miner/streaming/son.py:545`, `:578`);
     - the Polars or sparse counters behind `count_support_batched` (`son.py:401`, `:583`, `core/matrix.py`);
     - `_choose_counting_strategy` and its 1,000/100 thresholds, which now serve SON only.
   - The port:
     - Pass 1 (local mining per chunk): `mine_cpu`-style on the chunk's CSR.
     - Pass 2 (global count of the union of local candidates): `count_candidates` on a `RowSpace` per chunk.
   - Must stay green: `tests/test_son_completeness.py`, `tests/test_streaming.py`, and the SON legs of the tier chain.
   - Afterwards `sparse=` has no role left at all. Removing it is the owner's call.
   - Process: measure the time per SON phase first, and pre-register a protocol before building.
5. **Input layer and threaded bitvector build (should).**
   - Evidence (`bench/results/2026-10-07-cpu-phase1/compare.md`):
     - on deep_k and skewed_rows, the Polars CSR build takes 0.8–1.2 s, about 30 % of wall time;
     - at 4 threads deep_k gains nothing (2.74 → 2.69 s);
     - the threaded bitvector build is slower than the sequential one (0.40 → 0.62 s).
   - Measure at T1 and T4:
     - `build_transaction_csr` stages: `_count_items` (`cpu_miner.py:130`), and `_map_rows` with `replace_strict` per 500K-entry chunk;
     - `build_bitvecs` with and without the pool.
   - Candidate levers:
     - a sorted-key `search_sorted` mapping;
     - larger budgeted mapping chunks;
     - a work threshold that keeps small bitvector builds on one thread.
   - Memory: deep_k and skewed_rows at T1 are already at 1.17–1.18× of Phase 0, and the limit is 1.25×.
6. **Campaign:** an amendment to `bench/cpu/PROTOCOL.md`, 3 reps, same machine class. CHANGELOG and README numbers come only from that campaign.
7. Remind the owner about `stash@{0}`. Popping it on main may conflict, because README.md changed upstream in #25.

## Session Instructions
(carried over from the previous handoff and still binding, plus this session's)
- Chat in Dutch, concise. Files, code, comments, commits and PRs in English.
- Findings are numbered and tagged (must-fix)/(should)/(nice), ordered by severity.
- When asked "wat stel jij voor?", give one recommendation, not a menu.
- Docstrings: a short module docstring (what it does + Usage + Options), implementation steps in function docstrings, no development history in code.
- Git:
  - commit and push after every verified step;
  - never during a timed run;
  - not on `main`, no force-push;
  - a PR or merge only on the owner's word. This session's "Open een PR" covered PR #26 only.
- "Meten is weten": exact stakes before a build, and dispatch thresholds only from measured crossovers.
- Ask the owner before any run longer than 2 h in total, or before adding a runtime dependency or raising a floor.
- Tier 1 must stay installable and correct without Rust.
- Baselines:
  - Phase 0: `bench/results/2026-10-07-cpu-baseline/raw.jsonl`;
  - Phase 1: `bench/results/2026-10-07-cpu-phase1/raw.jsonl`;
  - never the README tables.
- Phase 1 invariants on every CPU change:
  - itemsets are emitted as ascending tuples;
  - `_min_count` stays the exact decimal ceiling, in sync with Rust and synthetic via `min_count_cases.json`;
  - `max_length` and the length cap keep their behaviour;
  - free-set runs test against the complete previous level and drop K=1 items present in every row;
  - Pascal inference stays exact;
  - `n_jobs=1` means sequential;
  - peak RSS may grow at most 25 %;
  - every new dense structure has a budget and a tested fallback.

## Decisions & Rationale
- **One array engine on the CPU route**, with Polars only reading the input (the owner's decision). `sparse=` warns and is ignored there, and selects a counter only in multi-chunk SON.
- **Floors:** NumPy stays `>=1.20` (popcount uses `np.bitwise_count` or a 16-bit lookup table); `polars>=2.0.0`.
- **Measured dispatch constants** (top of `cpu_miner.py`):
  - `PROJ_MIN_SUFFIXES = ((2048, 40), (inf, 80))`, from `bench/cpu/l3_crossover.py`;
  - `GRAM_BITVEC_RATIO = 3.0`;
  - `HEAVY_GROUP_WORK = 2M`;
  - `PARALLEL_GRAM_WORK = 5M`.
- **The 25 % memory rule** was met with:
  - an int32 indptr;
  - CSR mapping in 500K-entry chunks;
  - bitvectors built from CSR rows (no CSC);
  - chunked K=1 counting;
  - projection gathers without a data copy.
- **This session: `gram_pairs` replaces `RowSpace.gram`** (`cpu_miner.py:561`).
  - It counts only the requested (ia, ib) pairs, in row blocks within `GRAM_BUDGET_BYTES` (12 B per dense entry, the same as K=2).
  - With a pool the budget is split as `// (n_workers + 1)` (`cpu_miner.py:664`).
  - Rejected: a dispatch change. It needs a measured crossover.
- **This session: `_count_group_proj` keeps its positional signature, with an optional budget**, because `bench/cpu/l3_crossover.py:86` calls it.
- **This session: the bench F arms in `bench/cpu/matrix.py` were left alone.** They belong to the protocol campaign, so they are the owner's call.
- **This session: the uint8 tests use explicit SON chunk sizes** (`tests/test_sparse.py:27`, `:736`).
  - SON counts each chunk separately and sums in Python (`son.py:401`), so only a chunk with 256+ rows of the itemset can wrap a uint8 counter.
  - A single chunk falls back to the CPU route, which ignores `sparse=`.

## Dead Ends
- Do not run `test_support_001` or `test_support_0001` (`tests/test_smoke_correctness.py`) in the gate.
  - At s=0.001, Online Retail has 71M frequent itemsets at K=6 alone; the run was stopped at 11.6 GB, and no engine finishes it.
  - This session it hung for more than 20 min on main and on this branch too.
  - Deselect the 4 ids listed in Extra Context.
- Do not put every K≥3 prefix group on the thread pool: GIL hand-offs on small groups made or002 2.5× slower.
- Do not build bitvectors through a CSC: it breaks the memory rule on deep_k and skewed_rows.
- Do not commit or edit tracked files during a timed run. If one must change, use `git update-index --assume-unchanged`, and undo it afterwards.
- Do not import the module as `et_miner.core.apriori`: that name resolves to the function, because `core/__init__` shadows it. Use `importlib.import_module("et_miner.core.apriori")`.
- Local machine:
  - no `py-spy` (ptrace is denied);
  - no rustc, so `uv run maturin develop` fails and `tests/test_disable_rust.py` skips.
- Mutation checks of the sparse counter: patch `et_miner.core.sparse.count_support_sparse` (it returns a dict), not `matrix.count_support_batched`, because son.py imports the name directly.
- Plain `uvx ty check` lists 669 diagnostics for the whole repo. Compare per file against base instead.
- `pgrep -f <name>` inside `bash -c` also matches the shell itself.

## Key Findings
- **This session, PR #26:**
  - projection Gram: one group with s=15,912 went from 3.5 GB / 19.5 s to 0.2 s;
  - `oom_regression` to K=3 under a 12 GB cgroup cap: main took 289 s at a 10.2 GB peak, this branch 156 s at ≤ 0.9 GB; both gave the same 549,684 itemsets;
  - the UInt32 wrap of `pl.len()` on polars-runtime-32 was real, and is fixed at `cpu_miner.py:130`;
  - `min_support=0`: the Gram and bitvector K=2 paths disagreed, now fixed at `cpu_miner.py:833`;
  - the int64 indptr switch at 2**31 already existed. It is now named (`INDPTR32_LIMIT`, `cpu_miner.py:71`).
- **Verifier flags (not blocking):**
  - the Gram peak is about 1.8–1.9× the budget, because the sparse product is not counted;
  - `sel` in `gram_pairs` scans all of `ia` per block, which is slow only at budget=1 (searchsorted would avoid it);
  - no test asserts the per-worker budget split.
- **Phase 1 vs EA** (F-auto T1/T4 vs EA, seconds):
  - deep_k 2.74/2.69 vs 95.2;
  - skewed_rows 3.84/2.91 vs 161.5;
  - or002 7.27/5.58 vs 170.2.
  - The full table is in the previous handoff.
- **EA's profile:** 88–99 % of its time goes to `set.intersection`.

## Blockers & Pending Decisions
- PR #26 merge: owner.
- GPU verification of PR #25 and PR #26: needs a GPU box.
- Tier 2 leg and its `CLAUDE.md` wording: owner (policy).
- (nice) `test_support_001`: an explicit `max_length`, the oracle convention (−0.5) and a set comparison, or a `slow` mark. Today its oracle call has no −0.5, uses `max_length=100` and compares counts only. Owner.
- (nice) The GPU route adopting `build_transaction_csr`, the leaner CSR build: only after a GPU test.

## Test Status
This session, on the tree of `1a6fa21`, no GPU:
- `uv run ruff check src tests bench`: clean.
- `test_cpu_miner`, `test_sparse`, `test_tier_equivalence`, `test_disable_rust`: 180 passed, 50 skipped.
- Full `-m "not slow"`, run in two parts covering every file, with `test_support_001` deselected: 596 + 114 passed, 0 failed.

## What Was Done
- `src/et_miner/core/cpu_miner.py`:
  - `gram_pairs` (`:561`);
  - the budget is threaded through `count_candidates` (`:640`) and `_count_groups` (`:676`);
  - Int64 counts, the `min_count` clamp, `INDPTR32_LIMIT`, and the module docstring.
- `src/et_miner/core/apriori.py`, `src/et_miner/core/result.py`: docstrings now match the routes.
- `tests/test_cpu_miner.py`: a test per fix, plus a `_RecordingPool` check that each pool kernel ran.
- `tests/test_sparse.py`, `tests/test_disable_rust.py`: the `sparse=` tests go through multi-chunk SON. A mod-256 counter fails 5 of the 11.

## Commits This Session
- `1a6fa21` test: sparse= tests run through multi-chunk SON; docstrings name the routes
- `79d369a` cpu: block the K>=3 projection Gram within GRAM_BUDGET_BYTES

## Key Files
| File | Role |
|------|------|
| `src/et_miner/core/cpu_miner.py` | CPU route; measured constants at the top (`mine_cpu` :780, `build_transaction_csr` :194) |
| `src/et_miner/streaming/son.py`, `core/matrix.py`, `core/candidates.py`, `core/sparse.py` | SON's current path (Next Step 4) |
| `tests/test_tier_equivalence.py`, `src/et_miner/__init__.py`, `rust_ext/src/lib.rs` | Tier 2 leg and the Rust miner (Next Step 3) |
| `tests/test_smoke_correctness.py` | Online Retail vs EA; `test_support_001` :51, oracle call :32 |
| `bench/cpu/PROTOCOL.md`, `bench/cpu/*.py` | Pre-registered protocol (amendments 1–2) and campaign tooling |
| `bench/results/2026-10-07-cpu-{baseline,phase1}/` | Phase 0/1 rows, FINDINGS, `compare.md` |

## Extra Context
- Gate:
  ```
  uv run ruff check src tests bench
  uv run pytest -q -m "not slow" \
    --deselect tests/test_smoke_correctness.py::TestCPUvsEfficientApriori::test_support_001 \
    --deselect tests/test_smoke_correctness.py::TestCPUvsEfficientApriori::test_support_0001 \
    --deselect tests/test_smoke_correctness.py::TestGPUvsCPU::test_support_001 \
    --deselect tests/test_smoke_correctness.py::TestGPUvsCPU::test_support_0001
  uv run pytest -q tests/test_tier_equivalence.py tests/test_son_completeness.py tests/test_streaming.py tests/test_free_set_semantics.py
  ```
- Campaign:
  - run: `uv run --inexact python bench/runner.py --mode cpu-baseline --out <dir> --max-hours 1.5`;
  - compare: `uv run python bench/cpu/compare.py <baseline dir> <new dir>`.
- Phase 0/1 ran on a 4 vCPU Xeon @ 2.10 GHz with 15 GB. This session's machine has 12 cores and 30 GB. Never compare timings across machines; a campaign elsewhere needs its own EA and baseline arms.
- Datasets: `python datasets/prepare_online_retail.py`, and `uv run python -m et_miner.synthetic --preset all --out datasets/synth`.
