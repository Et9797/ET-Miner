# Session Handoff: ET-Miner, 2026-10-08 20:11

**Branch:** `perf/son-partition-bound` @ `a1f4c2c` (in sync with origin; merged into `main` as `c973ed3`)
**Tree:** clean apart from this handoff (`LATEST.md` is tracked); 1 stash (`stash@{0}`: README local edits, made on main)
**Focus:** work through the remaining open items (should/nice) after PR #29
**Reason:** end of session
**Previous handoff:** `HANDOFF-2026-10-08-1900.md`. Its items 1 (PR #28) and 4 (partition bound) are done; the rest is carried over below.

## Verify on Resume
- Run `git status --short --branch` and `git rev-parse --short HEAD`.
- Expect HEAD `a1f4c2c` on `perf/son-partition-bound`, and `origin/main` at `c973ed3` (the merge of PR #29). `M .claude/handoffs/LATEST.md` is this handoff.
- Next work branches from `main`: `git switch main && git merge --ff-only origin/main`, then a new branch. The modified `LATEST.md` comes along; commit it there (`git add -f` is not needed, it is tracked).

## Current State
- PR #29 (SON partition upper bound) is merged (2026-10-08 18:09 UTC). CI was green on `a1f4c2c`.
- `/council review` of PR #29: unanimous APPROVE in round 1. Its flags are fixed in `a1f4c2c`.
- No work is in progress and no run is active.
- Council teammates (architect-2, math-2, alternatives-2, auditor-2) were not sent shutdown requests; ignore their idle messages.

## Next Steps (no must-fix open)
1. **(should) GPU verification**, open since PR #25. PR #28 and PR #29 add to it. Needs a GPU box.
   - Run the GPU legs of `tests/test_tier_equivalence.py` (SON 1 GPU / 2 GPUs, forced chunks) and `bench/selfcheck.py`.
   - The GPU SON passes are unchanged by #29.
2. **(should) Input layer and threaded bitvector build.**
   - The CSR build is about 30 % of wall time on deep_k/skew.
   - The threaded `build_bitvecs` is slower than the sequential one.
   - Measure `build_transaction_csr` stages and `build_bitvecs` at T1/T4 first, then pre-register an amendment.
   - Memory headroom: deep_k/skew T1 are at 1.17-1.18x of Phase 0 (limit 1.25x).
3. **(should, owner decides) SON pass 1 is now 61-100 % of SON's wall time.**
   - or005 mines 1.13M local itemsets for 10,488 frequent ones.
   - Levers:
     - `local_support_factor` 0.9 -> 1.0 or another value (the owner's call). With the bound, 1.0 is worse on pass 2 (more slack, fewer exact counts). Measure pass 1 + pass 2 together.
     - Count a candidate only in the chunks that did not emit it: -25 % (candidate, chunk) pairs on skew, -40 % on or005 (`stakes.jsonl` `pairs_to_count`). Each union row must then carry its emitting chunks. Do not use a uint64 bitmask: it caps at 64 chunks.
4. **(should, small) Apply the bound to SON's GPU passes.** Both mine each chunk completely, so `_bound` carries over. Out of Amendment 5's scope; needs a GPU to verify.
5. **(nice) Carried from PR #28's council flags:**
   - De-duplicate mine_cpu's K=1 seeding (`cpu_miner.py` `mine_cpu`, `son.py` `_local_levels`, `bench/cpu/son_stakes.py` `run_array`).
   - Two predicates stated twice: the K=2 bitvec/Gram rule and the row-space compaction rule (`count_itemsets` vs `_mine_levels`).
   - Thread the per-candidate counter (`count_per_candidate`), which is single-threaded.
   - SON GPU pass 1 via `_build_csr_from_transactions`.
   - The pair-mask K=2 union idea is moot: `sum_rows` took or0001k2's pass 1 from 6.2 to 2.8 s.
6. **(nice) Carried from PR #26:**
   - The Gram peak is about 1.8-1.9x the budget.
   - `sel` in `gram_pairs` scans all of `ia` per block.
   - There is no test for the per-worker budget split.
   - `test_support_001` oracle convention (owner).
   - The GPU route adopting `build_transaction_csr`.
7. **(nice) Profile schema:** only the CPU SON pass 1 reports `n_bounded`/`n_exact` (`son.py` `_son_cpu`). Align the GPU branch if item 4 lands.
8. Remind the owner about `stash@{0}`: popping it on main may conflict in README.md.

## Session Instructions
- Chat in Dutch, concise. Files, code, comments, commits and PRs in English.
- Tag findings (must-fix)/(should)/(nice). For "wat stel jij voor?", give one recommendation.
- Measure first:
  - Write the stakes as an untimed count where possible.
  - Pre-register a `bench/cpu/PROTOCOL.md` amendment before any timed run.
  - Do not rewrite pre-registered text; correct it with an erratum, as Amendment 5 does.
- A/B timing: run base (main, `git checkout --detach <sha>`) and the change back to back with `son_stakes.py --matrix`. `incore` is the drift control.
- No commits or tracked-file edits during a timed run.
- PR or merge only on the owner's word. "Open een PR" covered PR #29 only.
- Ask before runs > 2 h, new runtime dependencies, or raising a floor.

## Decisions & Rationale
- **The bound** (`son.py` `_bound`):
  - `count(X) <= known + S - slack`, where slack per chunk is `_local_min_count - 1` and S sums the slack of every chunk, including chunks with no frequent item.
  - Exact when `S - slack == 0`.
  - It needs pass 1 to mine every chunk completely; `_local_levels` raises if a chunk's row count differs from `chunk_sizes`.
- **Union sums** are int32 below 2^31 transactions, int64 otherwise. Each sum is <= n_total because only emitting chunks add to a row.
- **`sum_rows` replaces `unique_rows`** (argsort / lexsort + `np.add.reduceat(dtype=w.dtype)`). Without `dtype`, reduceat promotes int32 to int64.
- **B1 baseline**: a fresh run at `e892d8a`, not S1's rows. S1 reproduced within 1.2 %.
- **Retargeted test:** the CPU SON completeness test runs at min_support 0.035, because at 0.05 pass 2 reads no chunk. It asserts `n_counted > 0` so it cannot go vacuous again.
- **Not changed:** `local_support_factor` stays 0.9.

## Dead Ends
- Do not use `np.unique` on int64 keys in hot paths: NumPy 2.5.2 hashes them. On 12M keys it took 6.37 s against 0.23 s for sort + mask; `np.unique(return_inverse=True)` is faster (1.8 s) but still slower than argsort.
- Do not prune the union during pass 1: the bound only rises as chunks arrive (a newly emitting chunk adds `count - slack >= 1`).
- `gh pr checks --watch` and `gh pr view --json headRefOid` do not exist in this gh version. Poll `gh pr checks <n>` in a loop and read the run's sha via `gh api repos/Et9797/ET-Miner/actions/runs/<id> --jq .head_sha`.
- `git add` of a tracked file under the ignored `.claude/handoffs` prints a hint and exits 1, but still stages it. This put `LATEST.md` into `5b10e79`.
- Carried: `gh pr edit --body-file` fails; use `gh api -X PATCH repos/Et9797/ET-Miner/pulls/<n> -F body=@file`. `pkill -f`/`pgrep -f` kill the tool shell. `uv run ty` fails; use `uvx ty check <file>` against main.

## Key Findings
- B1 (`bench/results/2026-10-08-son-bound/FINDINGS.md`), base vs bound, medians of 3:
  - deepk 4.74 -> 2.59 s (T1) and 4.25 -> 2.26 s (T4);
  - skew 31.2 -> 22.3 s and 15.0 -> 11.0 s;
  - or005 13.9 -> 8.1 s and 13.4 -> 7.8 s;
  - or0001k2 7.9 -> 2.8 s and 8.0 -> 2.9 s; RSS 768 -> 579 and 854 -> 672 MB;
  - smoke and wide within 0.02 s; RSS 0.75-1.02x; incore drift -2.0 to +1.2 %.
- Stakes (`stakes.jsonl`, factor 0.9): or005 1,125,261 -> 36,978 within the bound (4,138 exact); deepk and or0001k2 are fully exact.
- or0001k2's local min_count is 1 at factors 0.9 and 1.0, so it has no slack.
- `0.02 * 0.9 = 0.018000000000000002`, so deepk's local min_count is 4,501.
- SON is still 1.3-28x slower than in-core (or005 8.1 s vs 0.43 s).

## Blockers & Pending Decisions
- GPU legs and selfcheck: need a GPU box.
- `local_support_factor` default (owner).
- `batch_size`: deprecate or remove (owner).

## Test Status
At `a1f4c2c`:
- `uv run pytest -q -m "not slow"` (4 `test_support_00*` deselected): 793 passed, 276 skipped.
- Tier/SON/streaming/free-set suites: 59 passed, 138 skipped.
- ruff clean; ty on `son.py` equals main (7).
- CI green.

## What Was Done
- `src/et_miner/streaming/son.py`: `_CandidateUnion` (known/slack), `_bound`, `_local_min_count`, the row guard in `_local_levels`, the `_son_cpu` wiring, pass 2 skipped when empty.
- `src/et_miner/core/cpu_miner.py`: `sum_rows`.
- Tests:
  - `tests/test_streaming.py` (bound, exact, empty bound, row guard, factors, int64);
  - `tests/test_cpu_miner.py` (`sum_rows`);
  - `tests/test_son_completeness.py` (0.035).
- Bench:
  - scripts `bench/cpu/son_bound_stakes.py` and `bench/cpu/son_phases.py`;
  - `bench/cpu/PROTOCOL.md` Amendment 5 and its erratum;
  - `bench/results/2026-10-08-son-bound/`;
  - the CHANGELOG entry.

## Commits This Session
- `a1f4c2c` son: council flags for PR #29
- `d94d2b7` bench: SON partition-bound Phase B1 rows and findings; CHANGELOG numbers
- `740404f` son: bound pass-2 candidates by pass 1's local counts
- `5b10e79` bench: SON partition-bound stakes and protocol Amendment 5 (also carries the previous LATEST.md)

## Key Files
| File | Role |
|------|------|
| `src/et_miner/streaming/son.py` | SON; `_CandidateUnion`, `_bound`, `_local_min_count`, `_local_levels`, `_son_cpu` |
| `src/et_miner/core/cpu_miner.py` | `sum_rows`, `count_itemsets`, `build_transaction_csr` |
| `bench/cpu/son_stakes.py` | SON timing harness (`--matrix --arms built,incore`) |
| `bench/cpu/son_bound_stakes.py` | untimed bound stakes with oracle checks |
| `bench/cpu/son_phases.py` | diagnostic pass-1/pass-2 split |
| `bench/cpu/PROTOCOL.md` | Amendments 3-5 |
| `bench/results/2026-10-08-son-bound/` | B1 rows, stakes, FINDINGS |

## Extra Context
- PR #29 gemerged; open punten afwerken. Start with item 1 if a GPU box is available. Otherwise item 2 or 3: measure first, amendment before timing.
- Box: AMD Ryzen 5 4600G, 12 threads, 30 GB, no GPU. A B1-style A/B campaign takes about 7 min per run.
- Rust extension build: `env -u CONDA_PREFIX PATH="$HOME/.cargo/bin:$PATH" uv run maturin develop --release -m rust_ext/Cargo.toml`.
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
