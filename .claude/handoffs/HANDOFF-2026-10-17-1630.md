# Session Handoff: ET-Miner, 2026-10-17 16:30

**Branch:** `perf/moonshot-baseline` @ `1204082` (in sync with origin)
**Tree:** clean before this handoff (`.claude/handoffs/LATEST.md` is tracked, so copying this file dirties it; the handoff is committed separately)
**Focus:** GPU-route moonshots campaign (owner's brief, ideas M1–M12 + X0): Phases 0–1 done; Phase 2 starts with M1
**Reason:** end of session (owner asked for a handoff, committed and pushed)

## Verify on Resume
- Run `git status --short --branch` and `git rev-parse --short HEAD`.
- HEAD should be the handoff commit on top of `1204082`, and the tree clean.
- If not, the state moved after this handoff: read the diff before trusting "Current State".

## Current State
- Phase 0 (baseline) and Phase 1 (stakes) are committed and pushed. There is no PR.
- The campaign stopped at the brief's mandatory Phase 1 stop. The owner then replied: **"akkoord met jouw volgorde voorstel"**, approving the order. The owner set no GPU budgets.
- Approved order:
  1. M1 (GPU-native input layer)
  2. M2 (row-wise K=3), with M10 (item order) as the cheap fallback arm in the same protocol
  3. the X0 table (count inference with and without, the whole matrix, ~0.1 GPU-h; flip the default only on the owner's word)
  4. M7 (deduction rules depth ≤ 3, on top of inference)
  5. M5b (row reordering + occupancy tile skip)
  6. then M6 and M9, on a 2-GPU box
  7. M3 only if the tiled kernels still matter after M2, and only after the owner approves the CLAUDE.md rule change
- No code for any Phase 2 idea exists yet.

## Next Steps
1. Ask the owner (Dutch, short) for:
   - the GPU budgets per idea;
   - the base for idea branches. The brief says `perf/moonshot-<id>` off `main`, but the harness and scripts live only on `perf/moonshot-baseline` (unmerged). Propose branching off it, or a PR for it.
2. M1 Phase 2 per the brief:
   - `bench/moonshots/PROTOCOL-M1.md`: regimes, arms, pin `ET_MINER_INPUT_LAYER=host|device`, the rules 1/2/4/5 decision rule, budget.
   - Build behind the pin, as designed in `bench/moonshots/SPEC.md` § M1: the `_int_ids` span ≤ 2²² rule, chunked Polars explode, pinned async H2D, device occurrence bincount, LUT, bitvec scatter, the row-wise K=2 CSR from the device entries.
   - Byte-identical bitvecs test on every preset, plus null lists/items, duplicates, empty rows, span limit, negative ids, UInt32 > 2³¹, n_rows not a multiple of 64.
   - Tier legs on 1 and 2 GPUs; the 2-GPU legs skip on this box. Register any new kernel in `loader._KERNEL_FILES` and give it a `selfcheck.py` launch.
   - Campaign, decide mechanically, final check, `REPORT-M1.md`, CHANGELOG under Unreleased. Commit and push after each verified step, never during a timed run.
3. Then M2 (+M10), and so on in the order above. Each idea gets its own protocol, branch, tests and report.
4. Phase 3 after the last idea:
   - ruff, `pytest -m "not slow"`, the tier gate, `-m gpu`, selfcheck;
   - a reduced matrix against Phase 0;
   - an adversarial-verifier review;
   - the final `bench/moonshots/REPORT.md`.

## Session Instructions (from the owner's brief, which is not in any file)
- Exactness is the entry ticket: efficient-apriori stays the oracle. Every new path gets pinned tier legs on 1 and 2 GPUs, forced chunks, and count-inference and free-set variants where served. The only CLAUDE.md edits allowed: the tier-chain list and (with approval) the M3 intrinsic rule.
- Before code, each idea gets an exactness argument in its SPEC section, plus a brute-force property test: ties at min_count, empty rows, duplicate items, n_rows not a multiple of 64/256, all-ones words, 1 and 2 GPUs.
- Unchanged: itemsets, counts, dtypes, emission order, output_dir files, resume_from_k, max_length, anchor_items, bitvecs=, memory guards, profile, level_callback.
- Knobs are `ET_MINER_*` pins only, documented in `_env.py`. No new public parameters and no default change without the owner's word. An unserved option falls back by dispatch.
- Measurement:
  - a `PROTOCOL-<id>.md` before an idea's first timed run;
  - one box per comparison, recorded; a fresh process per config; warm-up excluded; ≥ 3 reps rep-major;
  - medians with [min, max], throttle reasons recorded;
  - no tracked-file edits during a run; never compare across boxes.
- No new runtime dependency without the owner's word (cuDF/KvikIO: no).
- Git: one branch per idea `perf/moonshot-<id>`. Commit and push after verified steps. No PR or merge without the owner's word.
- Docs: short module docstrings (what + Usage/Options); details in function docstrings; no history in code.
- Chat in Dutch, concise. Files, code and commits in English.
- Reports are numbered lists tagged (must-fix)/(should)/(nice), sorted in that order.
- Out of scope: CPU route, SON CPU passes, `core/rules.py`, `io/`, GCS, `config.py`, new public features.
- Stop and ask only when:
  - main fails the gates;
  - an M3 kernel is about to enter `_src`;
  - a new dependency or a public API/default change is needed;
  - an exactness argument fails for an option and needs > ~150 lines;
  - a signature mismatch appears that is not an obvious new-code bug.

## Decisions & Rationale
- **Polars compat runtime in the venv**: the Pentium G5420 has no AVX, and the locked `polars-runtime-32` dies silently. `uv pip install polars-runtime-compat==2.0.0`; Polars prefers it, and the lock and pyproject stay unchanged.
- **Threads pinned to the box's 4 logical CPUs** in `build_moonshots_baseline` (`bench/consolidation_matrix.py`), not 6 as in the earlier campaigns.
- **M9 per-shard counts computed exactly on the CPU** with the miner's `_row_split_cuts`, instead of a 2-GPU dump.
- **Amendments:**
  - dropped: M5a (2.34 < 2.52 s), M7 on today's walls (1.65 s; ≈ 34 % after M1), M4 for the current matrix, M12 (≤ 0.32 s);
  - not built: M11 (Online Retail ratio/planes 0.20).

## Dead Ends
- Do not wait with `until ! pgrep -f "<pattern>"` inside a backgrounded Bash: the waiter's own command line matches. Use a marker file (`grep -q "runner exit=" log`) or the task notification.
- Do not `import et_miner.core.apriori as m`: `et_miner.core` re-exports the function under that name. Use `importlib.import_module("et_miner.core.apriori")` (`bench/level_split.py` install()).
- Do not `rm -f $VAR/*.err`: the safety check blocks it. Use a literal path or `"${VAR:?}"`.
- Do not `cd rust_ext`: it moves the session cwd. Use `-m rust_ext/Cargo.toml` from the root.
- Foreground `sleep N` is blocked; use Monitor or a backgrounded until-loop.

## Key Findings
- **Phase 0, 1 GPU, medians:**
  - dsl 25.21 s, of which input before K=1 is 18.57 (COO→CSR 8.25, explode+join 6.53, host K=1 2.28, H2D 1.02);
  - dsl-infer 21.54, dsl-free 21.09;
  - sk2ml3 23.22 (tiled K=3 16.62), oom2ml3 2.48; the rest < 1.1 s.
- **Rates:** b1 `mma.sync.m16n8k256 .and.popc` is 16.3× `__popcll` (9.51e12 against 5.82e11 slot-words/s), bit-exact on 36 cases. The tiled kernel is ≥ 54 % staging on sk2ml3.
- **Stakes (upper bounds):**
  - M1: dsl 16.8 s, sk2ml3 4.8;
  - M2: sk2ml3 16.2 (exact increments 375,181,499, not ~1e9; reproduces K=3 exactly);
  - M10: sk2ml3 7.8 (generated 11.8 B → 145 M under ascending support);
  - M3: sk2ml3 7.3;
  - M5b: dsl 5.6 (lex row order: 87–97 % of 2,048-row prefix tiles are zero, against 0 %; item-occupancy AND finds nearly all of them);
  - M8: dsl 4.0;
  - X0: −3.66 s measured;
  - M9: 1.19 % of written entries flagged.
- **M7:** depth 3 settles 100 % of dsl's counted candidates from K=8. Depth 2 settles almost none, so the k−3 index is required.
- **M4:** every preset has one big component (dsl 52–80 items). The proposed `deep_k20` was checked on the CPU: 2,097,793 itemsets to K=20, two 20-item components from K0=6.
- **Unverified, by code reading:** string items cannot pass the GPU route (`src/et_miner/gpu/row_split.py` builds `col_to_item_arr` as int64). Out of scope; follow-up.

## Blockers & Pending Decisions
- The owner must give:
  - GPU budgets;
  - the branch base for the idea branches;
  - whether `deep_k20` joins `PRESETS`;
  - whether to keep the committed lattice dumps (14 MB, committed because `!bench/results/**` overrides `*.parquet`).
- M6, M9 and every 2-GPU leg need a 2-GPU box. The commands are in `bench/results/2026-10-17-moonshots-baseline/README.md`.
- M3 needs the owner's CLAUDE.md rule change (b1 MMA is sm_80+, not "sm_60+").

## Test Status
At `ec3de12`'s `src/` (= `main`), 1 GPU:
- `uv run pytest tests/test_tier_equivalence.py -q`: 32 passed, 22 skipped (2-GPU).
- `uv run pytest -q -m "gpu and not slow"`: 390 passed, 87 skipped.
- `uv run pytest -q -m "not slow" -k "not test_support_00"`: 1264 passed, 86 skipped.
- `uv run python bench/selfcheck.py`: READY.
- `uv run pytest tests/test_campaign_gate.py tests/test_optimizations_bench.py`: 48 passed at `1204082`.
- `ruff check bench src tests`: clean.

## What Was Done
- `bench/level_split.py`: the span phases before K=1 and after the last level (`start()` and `finish()`, `SPAN_PHASES`).
- `bench/consolidation_run.py`: `timings["read_s"]` and `timings["span"]`.
- `bench/consolidation_matrix.py`: `build_moonshots_baseline`. `bench/runner.py`: `--mode moonshots-baseline`.
- `bench/moonshots/`:
  - `microbench.py`, `split_table.py`;
  - the stake scripts (`k3_rows_stake`, `replay_stakes`, `k3_census`, `component_stakes`, `tile_stakes`, `dedup_stakes`, `input_floor`, `relabel_lattice`, `preset_check`, `rows`, `stakes`);
  - `SPEC.md`, `STAKES.md`, `README.md`.
- `bench/results/2026-10-17-moonshots-baseline/` (raw, microbench, split, lattices, README) and `…-stakes/` (all outputs, README).

## Commits This Session
- `1204082` moonshots: stakes
- `3e00fc5` moonshots: baseline README, the lattice dumps are committed
- `c088cb0` moonshots: baseline
- `ec3de12` moonshots: baseline harness and stake scripts

## Key Files
| File | Role |
|------|------|
| `bench/moonshots/SPEC.md` | per-idea evidence, proposal, exactness, dispatch, acceptance; amendments |
| `bench/moonshots/STAKES.md` | ranked table, interactions, suggested order, brief facts checked |
| `bench/moonshots/stakes.py` | recomputes every bound from the baseline and stake outputs |
| `bench/results/2026-10-17-moonshots-baseline/README.md` | box, gates, wall split, rates, 2-GPU box commands |
| `bench/results/2026-10-17-moonshots-stakes/README.md` | every stake output and its command |
| `src/et_miner/core/matrix.py:521` | `_build_csr_from_transactions`, M1's target |
| `src/et_miner/gpu/csr_bitvec.py:194` | `build_bitvecs_gpu` (int64 indices, pageable H2D) |
| `src/et_miner/core/cpu_miner.py:157` | `_int_ids`, the span ≤ 2²² rule M1 reuses |
| `src/et_miner/gpu/row_split.py` | the row-split miner; K=2 dispatch ~:778, K≥3 ~:858 |
| `src/et_miner/gpu/kernels/_src/pairs_k2_rows.cu` | row-wise K=2, M2's template |

## Extra Context
- The owner's reply: "akkoord met jouw volgorde voorstel hierboven; Commit en push de handoff".
- Box: 1× RTX A4000 16 GB, driver 535.146.02, CuPy 14.1.1 / NVRTC 12.9, Pentium G5420, 31 GB, slow network (~1.6 MB/s).
- The Rust extension is built: `env -u CONDA_PREFIX uv run maturin develop --release -m rust_ext/Cargo.toml`, with cargo in `~/.cargo/bin`.
- The datasets are generated in `datasets/synth` and `datasets/online_retail_ii`.
- Memory notes: `gpu-box-no-avx`, `moonshots-campaign`.
