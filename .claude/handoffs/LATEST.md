# Session Handoff: ET-Miner, 2026-10-08 19:00

**Branch:** `perf/son-array-miner` @ `150dc1a` (in sync with origin)
**Tree:** clean (handoffs ignored; `.claude/handoffs/LATEST.md` is tracked); 1 stash (`stash@{0}`: README local edits, made on main)
**Focus:** afwerken van de open must-fix/should/nice items (finishing the open items)
**Reason:** end of session
**Previous handoff:** `.claude/handoffs/HANDOFF-2026-10-08-1558.md`. Its items 3 (Tier 2 leg) and 4 (SON port) are done; the rest are carried over below.

## Verify on Resume
Run `git status --short --branch` and `git rev-parse --short HEAD`. HEAD should be `150dc1a` and the tree clean. If not, read the diff first.
- PR #28 (https://github.com/Et9797/ET-Miner/pull/28, `perf/son-array-miner` -> `main`):
  - merged: branch the next work from `main`;
  - open: check its CI on `150dc1a` with `gh pr checks 28`. The last green run was on `0c1d5cb`, before the flag fixes.

## Current State
- PR #28 (SON CPU passes on the array miner) is open and waits on the owner. Do not merge it yourself.
- `/council review` of PR #28: unanimous APPROVE in round 1.
  - Council flags 1-4 are fixed in `150dc1a`: the UInt32 sum, the figures, the harness `current` arm, and the `son.py` docs.
  - Flags 5-7 are open; see Next Steps.
- No work is in progress. The last action was a PATCH of the PR #28 body (deep_k 4.8 -> 4.7 s).
- Council teammates (architect, math, alternatives, auditor) were not sent shutdown requests.

## Next Steps (no must-fix open)
1. **PR #28:** handle the owner's review on the same branch. Re-check CI on `150dc1a`.
2. **GPU verification (open since PR #25; PR #28 adds to it):**
   - on a GPU box, run the GPU legs of `tests/test_tier_equivalence.py` (SON 1 GPU / 2 GPUs, forced chunks) and `bench/selfcheck.py`;
   - PR #28 removed the CPU `else` branches from the GPU path of `apriori_streaming` (`src/et_miner/streaming/son.py`).
3. **(should) Input layer and threaded bitvector build** (previous handoff, item 5):
   - the CSR build is about 30 % of wall time on deep_k/skew;
   - the threaded `build_bitvecs` is slower than the sequential one;
   - measure `build_transaction_csr` stages and `build_bitvecs` at T1/T4 first;
   - pre-register the protocol before building;
   - memory headroom: deep_k/skew T1 are at 1.17-1.18x of Phase 0 (limit 1.25x).
4. **(should, biggest SON lever, needs stakes first) Partition upper-bound pruning before pass 2.**
   - Pass-1 local counts are thrown away at `son.py` `_local_levels` (`return tc.items, [sets for sets, _ in emitted]`).
   - Per union row keep `known` (sum of local counts) and `seen_slack` (sum of `local_min_count_i - 1` over the chunks where it was frequent).
   - `bound = known + total_slack - seen_slack`; drop rows with `bound < global min_count`.
   - Aggregate both through `unique_rows` with `return_inverse` + `bincount` weights.
   - Expected to remove most of or005's 1.13M candidates against 10,488 frequent (unmeasured).
5. **(nice) Council flags 5-6 on PR #28:**
   - De-duplicate mine_cpu's K=1 seeding (`max(1, _min_count(...))`). It lives in `cpu_miner.py` `mine_cpu`, in `son.py` `_local_levels` and in `bench/cpu/son_stakes.py` `run_array`. A shared `cpu_miner` helper returning the emitted levels would fix it.
   - Two predicates for the rules now stated twice: the K=2 bitvec/Gram rule and the row-space compaction rule (`count_itemsets` vs `_mine_levels`).
   - A pair-mask K=2 union when `n_items**2 <= PAIR_MASK_BYTES`. The union costs 4.9 of or0001k2's 7.9 s.
   - Thread the per-candidate counter: it is single-threaded, which is why skew T4 separates.
   - SON GPU pass 1 via `_build_csr_from_transactions` instead of `build_boolean_matrix` (GPU only).
6. **Carried (nice):**
   - PR #26 verifier flags:
     - the Gram peak is about 1.8-1.9x the budget;
     - `sel` in `gram_pairs` scans all of `ia` per block;
     - no test for the per-worker budget split.
   - `test_support_001` oracle convention (owner).
   - The GPU route adopting `build_transaction_csr` (after a GPU test).
7. **Campaign:** a protocol amendment and 3 reps once items 3/4 are built. CHANGELOG/README numbers come only from campaigns.
8. Remind the owner about `stash@{0}`: popping it on main may conflict in README.md.

## Session Instructions
- Chat in Dutch, concise. Files, code, comments, commits and PRs in English.
- Findings are tagged (must-fix)/(should)/(nice), ordered by severity. For "wat stel jij voor?", give one recommendation.
- Measure first: pre-register the protocol (`bench/cpu/PROTOCOL.md` amendment) before any timed run. Dispatch thresholds come only from measured crossovers; changing a registered rule after seeing data needs the owner's approval (as Amendment 4 got).
- No commits or tracked-file edits during a timed run. Stamped rows carry `rev` (`+dirty` if tracked files differ).
- PR or merge only on the owner's word. "Open een PR" covered PR #27 and PR #28 only.
- Ask before runs > 2 h, new runtime dependencies, or raising a floor. The owner approved installing rustup locally.

## Decisions & Rationale
- **SON pass-2 K>=3 dispatch (Amendment 4, owner):**
  - per-candidate when the row space has <= 512 words or the mean candidates per prefix group is < 2.5; else prefix groups;
  - constants `cpu_miner.py` `PER_CANDIDATE_MAX_WORDS`, `PER_CANDIDATE_MEAN_GROUP`;
  - rejected: registered rule 3's geomean pick (per-candidate everywhere), which loses 11.4/19.9 s on skew T1/T4.
- **`sparse=` deprecated on SON's CPU passes too (owner)**, not removed. The warning is raised in `apriori()` before routing (`src/et_miner/core/apriori.py`, `if sparse is not None and not use_gpu and not (streaming and n_gpus > 1)`) so it points at the caller. `apriori_streaming` warns only on a direct call.
- **Workloads:** or003/or002 are excluded from the SON matrix (Amendment 3).
- **Union design (`son.py` `_CandidateUnion`):**
  - ids by first sight, renumbered in `finish()`;
  - pending rows are merged only past `max(UNION_PENDING_BYTES=256MB, held)` per length;
  - pass-1 progress `candidates` is therefore an upper bound; the profile `n_candidates` is exact.
- **sparse-counter tests** (`tests/test_sparse.py` `_counter_counts`) and the Rust-switch probe (`tests/test_disable_rust.py`) count through `count_support_batched(sparse=True)` directly, because SON no longer reaches `core/sparse.py`.
- **The CPU estimate in `_estimate_chunk_size_from_memory`** stays on the boolean-matrix model (conservative for the CSR). Changing it would change `memory_budget_gb` behaviour without a measurement.

## Dead Ends
- Do not run SON with 4 chunks on or003/or002: the local lattice explodes (or003 chunk 1: 11,153,889 itemsets to K=16, 112 s; chunks 2-3 > 120 s).
- Do not deduplicate the union every chunk: it re-sorts the growing union (or0001k2 18.1 s, of which 13 s in `unique_rows`).
- Do not find the K=2 Gram columns with `np.unique(c)` over the pairs: it took 0.77 vs 0.38 s per chunk. Use the used-column mask.
- `gh pr edit --body-file` fails: `GraphQL: Projects (classic) is being deprecated ...`. Use `gh api -X PATCH repos/Et9797/ET-Miner/pulls/<n> -F body=@file`.
- `pkill -f <pattern>` / `pgrep -f` inside a Bash or Monitor command matches its own shell (it killed the tool shell, exit 144).
- `uv run ty` fails ("Failed to spawn: ty"); use `uvx ty check <file>` and compare per file against `main` (stash, check, pop).
- Ignore stray idle messages from the old `verifier` (PR #26 era).

## Key Findings
- S0 (`bench/results/2026-10-08-son-s0/`): old SON 0.96-268 s (or005/or0001k2 > 600 s cap); array port 0.14-43 s; per-level counter winners in FINDINGS.
- S1 (`bench/results/2026-10-08-son-s1/`, 3 reps at `58f1bb6`):
  - built SON 3.6-409x faster where the old SON finished, 7.9-13.9 s where it capped;
  - peak RSS 0.13-0.87x;
  - one signature per workload across S0/S1;
  - smoke T4 misses the 1 s floor (0.82 s saved);
  - the first S1 run is kept as `superseded-2833775.jsonl`.
- SON stays 1.4-48x slower than in-core: or005 has 1.13M local candidates against 10,488 frequent.
- Polars `UInt32` sum wraps (`2x3e9 -> 1705032704`); fixed in `son.py` `_count_chunk` with `.cast(pl.Int64)`.
- `local_support_factor=0.9` is not needed for completeness (exact ceil `_min_count`); 1.0 is complete. Changing it is the owner's call.

## Blockers & Pending Decisions
- PR #28 merge: owner.
- GPU legs of SON and the selfcheck: need a GPU box.
- `batch_size` is read by no route now: deprecate or remove? (owner)
- `local_support_factor` default 0.9 -> 1.0? (owner; changes runtime, not results)

## Test Status
`uv run pytest -q -m "not slow"` (the 4 `test_support_00*` deselected) at `150dc1a`: 784 passed, 276 skipped, 0 failed. Tier/SON/streaming/free-set suites: 51 passed, 138 skipped. `ruff check src tests bench` clean. Rust extension built locally (`et_miner_rust` 0.3.0).

## What Was Done
- PR #27 (merged): Tier 2 leg on `apriori_from_csr` (`tests/test_tier_equivalence.py` `test_tier2_rust_matches_oracle`); CLAUDE.md leg renamed.
- PR #28 (open):
  - `src/et_miner/streaming/son.py`: `_son_cpu`, `_CandidateUnion`, `_local_levels`, `_count_chunk`;
  - `src/et_miner/core/cpu_miner.py`: `count_itemsets`, `count_per_candidate`, `unique_rows`, module-level `gram_pairs`;
  - the `apriori.py` warning;
  - tests;
  - `bench/cpu/son_stakes.py`;
  - PROTOCOL Amendments 3-4, S0/S1 results, CHANGELOG/README.

## Commits This Session
- `150dc1a` son: council flags for PR #28
- `0c1d5cb` bench: SON Phase S1 rows and findings; CHANGELOG numbers
- `58f1bb6` son: deduplicate the pass-1 union once it outgrows a budget
- `2833775` son: CPU passes on the array miner
- `508cdb0` bench: protocol Amendment 4, the pass-2 K>=3 counter dispatch
- `51215a9` bench: SON Phase S0 stakes rows and findings
- `6dabc9e` bench: SON stakes harness and protocol Amendment 3
- `1e913e7` docs: handoff for PR #26 and the CPU tier's follow-ups
- `68ef180` test: Tier 2 leg runs the all-Rust miner (apriori_from_csr)

## Key Files
| File | Role |
|------|------|
| `src/et_miner/streaming/son.py` | SON; CPU passes `_son_cpu` and helpers; GPU passes unchanged; `UNION_PENDING_BYTES` |
| `src/et_miner/core/cpu_miner.py` | array miner; `count_itemsets` and dispatch constants |
| `src/et_miner/core/apriori.py` | routing; `sparse=` deprecation warning ahead of routing |
| `bench/cpu/son_stakes.py` | SON harness (`--matrix`, arms built/incore/array/array-pc/current) |
| `bench/cpu/PROTOCOL.md` | Amendments 3-4 (SON protocol and dispatch) |
| `bench/results/2026-10-08-son-s{0,1}/` | SON rows and FINDINGS |
| `tests/test_streaming.py` (`TestArrayPasses`), `tests/test_cpu_miner.py` (count_itemsets tests) | coverage of the port |

## Extra Context
- Focus for the next session: work off the open items above in order. There is no must-fix; start with item 1 (PR #28 review/CI), then GPU verification if a box is available, otherwise item 3 or 4 (measure first, amendment before timing).
- Box: AMD Ryzen 5 4600G, 12 threads, 30 GB, no GPU. S0/S1 ran here. Never compare with the Phase 0/1 rows (4-vCPU container).
- Datasets generated locally (gitignored): `datasets/synth/{smoke,deep_k,skewed_rows,wide_vocab}.parquet`, `datasets/online_retail_ii/transactions.parquet`.
- rustup is in `~/.cargo`, not on PATH. Build the extension with `env -u CONDA_PREFIX PATH="$HOME/.cargo/bin:$PATH" uv run maturin develop --release -m rust_ext/Cargo.toml`.
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
