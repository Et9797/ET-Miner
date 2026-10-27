# Session Handoff: ET-Miner, 2026-10-27 17:13

**Branch:** `refactor/route-consolidation` @ `f6c3d36` (in sync with origin)
**Tree:** 6 modified, 7 untracked (Phase 1 harness work, uncommitted), 0 stashes
**Focus:** Route consolidation round 2 (owner's brief, pasted in chat): measure every mining path, delete what loses
**Reason:** owner asked for a handoff mid-session (task continues)

## Verify on Resume
Run `git status --short --branch` and `git rev-parse --short HEAD`. HEAD should be `f6c3d36` and the tree should match "Tree" above. If not, read the diff before trusting "Current State".

## Current State
- Phase 0 done (inventory committed), Phase 1 nearly done: dead code, a must-fix bug, bench-only arms, harness, protocol draft.
- Two background jobs were running at handoff time:
  - `deep_sparse_xl` generation (`datasets/synth/deep_sparse_xl.parquet`, 60M rows; peak ~21.5 GB RSS; log `/tmp/claude-0/-root-projects-ET-Miner/0b85df72-8420-4cb0-b6da-84d256972ef0/scratchpad/gen_xl.log`).
  - Census: `bench/runner.py --mode routes-census --skip dslxl --out <scratchpad>/census` then `bench/routes/census.py <scratchpad>/census/raw.jsonl` (kernels per level per regime). Harness check only, not a result.
- Uncommitted: `bench/routes/{PROTOCOL,census,count_api,decide}.py|md`, `bench/routes_matrix.py`, `bench/routes_run.py`, `tests/test_routes_bench.py`, edits to `bench/{README.md,child_run.py,microbench_rust.py,runner.py}`, `src/et_miner/synthetic.py` (block_rows + `deep_sparse_xl` preset), `tests/test_synth.py`. All pass ruff; `tests/test_routes_bench.py` 9 passed, `tests/test_synth.py` 35 passed.

## Next Steps
1. When generation ends: census on dslxl (`--mode routes-census --only dslxl`), read census, adjust `routes_matrix.py` (G3/G5 arm placement, SUBSECOND) and PROTOCOL.md to match what each regime runs.
2. Run the not-slow suite + no-CuPy run, then commit `routes: harness and protocol` (PROTOCOL must be committed BEFORE the first timed run; untracked files are not in the runner's rev digest).
3. Phase 2a: `uv run python bench/runner.py --mode routes-cpu --out bench/results/2026-10-27-routes-cpu --max-hours 8`; plus C4 microbench (`bench/routes/count_api.py dump or002|wide|dsl DIR`, `dump or005 DIR --son-chunk 9106`, then `time`). Write env.txt extras (lscpu, cgroup, free). No tracked-file edits/commits while it runs.
4. Phase 2b: GPU gates on main were already green (see Test Status), then `--mode routes --out bench/results/2026-10-27-routes-gpu --max-gpu-hours 6`; H dumps `RAYON_NUM_THREADS=17 bench/microbench_rust.py dump deep_sparse_large DIR`, `dump deep_sparse_xl`, `dump stress_k2 --max-length 2 --then-k3`, `dump deep_sparse_large --free-sets`, `dump deep_sparse_large --son-chunk 5000000`, `dump deep_k --son-chunk 250000`, then `time DIR`.
5. `bench/routes/decide.py` on both raw.jsonl → FINDINGS.md + report.md per results dir; commit `routes: measurements`.
6. Phase 3 (one commit per DP), Phase 4 (verify, `routes-final` mode needs `bench/routes/final.py` with `final_configs(n_dev)`, adversarial-verifier review, `/council review`), final REPORT.md.

## Session Instructions
- The owner's brief (pasted at session start) governs: ground rules 1–11, "stop and ask" list, decision rule 1–11 (copied into `bench/routes/PROTOCOL.md`). Key points:
  - Chat in Dutch, concise; files/code/commits English. Reports: numbered lists tagged (must-fix)/(should)/(nice), sorted.
  - No PR/merge without owner's word; stacked branches `refactor/drop-esco`, `refactor/drop-rust` pushed, not merged.
  - Approved exception: if C4 removes their paths, `batch_size` and `sparse=` may default to None and raise otherwise.
  - Owner decisions: SON stays; `use_generator_pruning` default stays False; `n_gpus` exact; row split stays.
  - Out of scope: `core/rules.py`, `io/`, GCS, `streaming/ramdisk.py`, config/cli beyond removed-path options; `bench/results/*` and finished campaigns' docs untouched.
  - Use explorer-opus for tracing, adversarial-verifier for review; `/council review` at the end.

## Decisions & Rationale
- **THREADS = 17**: cgroup `cpu.max 1792000 100000` (17.92 CPUs); 28 physical cores would be CFS-throttled. Recorded in PROTOCOL.
- **Scale regime = new preset `deep_sparse_xl`** (dsl shape, 60M rows, 20M-row blocks, seed 4343): `ET_BENCH_ALPHAFOLD` unset, no base214m on box. Block generation needed: one-shot 60M would exceed the 34.8 GiB memory limit.
- **Bench-only pins** (Phase 3 removes unless a tier leg needs them): `ET_MINER_LEAVE_OUT` (k2-shared,k2-percand,k2-tiled,k2-fused,percand,group,tiled,fused), `ET_MINER_INPUT_LAYER=rebuilt|coded`, `ET_MINER_SON_GPU=boolean|ported`; CPU arms by module overrides in `routes_run._apply_overrides`.
- **`build_bitvecs_from_gpu_arrays` kept**: `applications/alphafold/experiments/experiment_null_model.py:200` calls it (agent missed applications/).
- **`core/rules.py:503` docstring path updated** (only touch in an out-of-scope file): it named the deleted `_prune_equal_support`; now `core/cpu_miner.py::_non_free`. Flag in the report.
- **Default config per regime is shared** across DPs (full dispatch / today's arm), to save budget.

## Dead Ends
- Do not run `uv`/maturin from inside `rust_ext/` (second venv); always repo root.
- Do not use `git stash` on the whole tree to compare against main (worked, but risky); use `git stash push -- src`.

## Key Findings
- (must-fix, fixed `a1244a3`) GPU route with String/Float/Categorical items: host CSR then `gpu/row_split.py:432-434` copies col_to_item into np.int64 → strings raised or came back Int64, floats truncated. Now column ids + `_as_item_dtype`; `core/result.py::_emitted_items` shared with `cpu_miner._emit`; output_dir/resume/anchors raise for non-integer items; `bitvecs=` rejects non-integer col_to_item.
- R5 (`build_column_bitvecs_u64`) is not an R-GPU host step: only `_build_gpu_bitvec_matrix` fallback (`gpu/bitvec.py:70-82`), reached from SON GPU passes and `count_support_batched(use_gpu)`.
- Census from moonshots phase3: K=2 dense tiled on small-item regimes, row-wise K=2/K=3 on oom2ml3/sk2ml3; K=2 per-candidate (<120 pairs) and fused run at no default regime → forcing regimes `deepk15` (deep_k @0.15, 10 items) and `*-bv` inputs.
- deep_k SON pass 2 counts nothing (bound settles all) → C4 SON batch uses or005 (`--son-chunk 9106`).
- `cargo clippy -D warnings`: 2 pre-existing errors on main (div_ceil, complex type) — (nice) finding, not fixed.
- Silent today (verified): `batch_size`, `sparse=`, `streaming=True,n_gpus>1` ignoring `use_gpu` (`core/apriori.py:684`); also `resume_from_k` without output_dir (`row_split.py:546`).

## Blockers & Pending Decisions
- None yet. Stop-and-ask triggers are in the brief (budget exceeded, capability port > ~150 lines, new public param, rule gives no answer, non-obvious signature mismatch).

## Test Status
- main @895f3ab (Phase 0): ruff clean; not-slow `-k "not test_support_00"` 1627 passed; tier 91 passed; no-CuPy 896 passed/563 skipped; `gpu and not slow` 729 passed; selfcheck READY (16 kernels, 2 devices).
- After `f6c3d36` work: not-slow 1636 passed (at `a1244a3`), no-CuPy 899 passed; tier + SON suites 296 passed at `f6c3d36`; leave-out/input-arm/son-ported tests 22 passed.
- No-CuPy recipe: `<scratchpad>/gates/nocupy.py` (sets `sys.modules["cupy"]=None`).

## What Was Done
- `bench/routes/INVENTORY.md`: verified route map (commit b0c1b82).
- Dead code removed (4da7ef6): `core/candidates.py`, `gpu/type_safety.py`, `gpu/utils.py`, list references in `core/apriori.py`, csr_bitvec pinned pool, loader device locks, Rust `get_num_threads` + `core/utils.rs`.
- Bench arms: `gpu/row_split.py` K=2 (~:980-1000) and K≥3 (~:1075-1095) leave-out, `row_split_chunks.group_kernels`, `gpu/kernels/k2.py:121`, `core/apriori.py` `_coded_items`/`_rebuilt_csr`, `streaming/son.py` `_son_arrays`/`_local_levels_gpu`/`_count_chunk_gpu`, `streaming/multi_gpu.py` ported branch, `gpu/kernels/batch.py::count_itemsets_flat`.

## Commits This Session
- `f6c3d36` routes: bench-only SON on the array skeleton for GPU passes
- `53909ab` routes: bench-only arms for the leave-one-out and input-layer decisions
- `a1244a3` GPU route: mine string, float and categorical items as column ids
- `4da7ef6` routes: dead code
- `b0c1b82` routes: inventory

## Key Files
| File | Role |
|------|------|
| `bench/routes/PROTOCOL.md` | boxes, budgets, arms, regimes per DP, rule verbatim (draft, uncommitted) |
| `bench/routes_matrix.py` | config builders: routes-cpu (226), routes (546), routes-census (23), routes-final |
| `bench/routes_run.py` | child for mode "routes": routes C, C-bitvecs, D, E, F, RUST; input variants str/wide; overrides |
| `bench/routes/decide.py` | mechanical rules 1–7, 11; LEAVE_ONE_OUT / HEAD_TO_HEAD / g7 tables |
| `bench/routes/count_api.py` | C4 microbench (dump/time) |
| `bench/microbench_rust.py` | H microbench R2/R4/R5 (dump via apriori(use_gpu), time) |
| `<scratchpad>/PLAN.md`, `<scratchpad>/inv/agent{1,2,3}-key.md` | working plan and the three trace reports (condensed) |

## Extra Context
- Scratchpad: `/tmp/claude-0/-root-projects-ET-Miner/0b85df72-8420-4cb0-b6da-84d256972ef0/scratchpad`.
- Box: 2× RTX 3060 12 GB (no P2P), 2× Xeon E5-2680 v4, 34.8 GiB cgroup memory, Rust via `~/.cargo/bin` (export PATH). Datasets in `datasets/synth/` + `datasets/online_retail_ii/` (not committed).
