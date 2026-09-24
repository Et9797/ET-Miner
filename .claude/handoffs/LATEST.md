# Session Handoff: ET-Miner, 2026-09-24 15:57 UTC

**Branch:** `docs/tiered-tutorials` @ `8df7b5f` (ahead 0 / behind 0 of origin/docs/tiered-tutorials)
**Tree:** clean, 0 stashes (`.claude/handoffs/` is excluded via `.git/info/exclude`)
**Focus:** tier 3 GPU documentation: get the GPU cells of `docs/tier3-gpu/*.ipynb` executed on real hardware
**Reason:** end of session / task switch; PR #15 open

## Verify on Resume
Run `git status --short --branch` and `git rev-parse --short HEAD`. HEAD should be `8df7b5f` and the tree clean. If not, read the diff before trusting "Current State". Also check PR https://github.com/Et9797/ET-Miner/pull/15 for review comments.

## Current State
- All docs are written, executed on CPU, committed, pushed; PR #15 (`docs/tiered-tutorials` -> `main`) opened.
- Last command: `docs/run_notebooks.sh` → all 11 notebooks ok (~1.5 min, x86_64 4 CPUs, no GPU); `verify.py` → "no problems".
- Tier 3 GPU cells print `skipped: no CUDA device`; none has run on a GPU. The tier 3 equality asserts (GPU == tier 1) are unverified.
- The user was asked whether to watch PR #15 (subscribe_pr_activity); no answer yet.

## Next Steps
1. On a CUDA machine: `uv sync --inexact --extra gpu`, `uv run maturin develop --release -m rust_ext/Cargo.toml`, `uv run python bench/selfcheck.py`, then `docs/run_notebooks.sh tier3-gpu`.
2. Work the checklist at the end of each tier 3 notebook; fix any GPU cell that raises (docs only, never library code).
3. Update `docs/README.md` section 2 runtimes/hardware for tier 3 with the GPU run's numbers; commit outputs from the real run.
4. Ask the user about PR watching if not answered.

## Session Instructions
- Diff limited to `docs/`; never modify library code. If a tutorial needs a code change, stop and ask.
- No fabricated numbers: every number in prose must come from an executed cell, labelled with hardware/data size/date. Never hand-write outputs.
- No references to private components, internal tooling, AI/agent workflows, project history, preprints, private individuals (CLAUDE.md is not linked from docs).
- Style: lead with main point, one idea per sentence, no em dashes, no emoji, no exclamation marks, numbered headings, American spelling (chosen), Mermaid only where it helps (removed from README).
- Don't add jupyter/nbconvert to `pyproject.toml`; use `uv run --with`.
- Commit docs with `git add -f` (docs/ is gitignored at `.gitignore:6`), user chose this over editing `.gitignore`.
- Branch `docs/tiered-tutorials`, push after each green tier (user chose this).
- Tier 3 without GPU: GPU cells print a one-line skip message as real output (user chose this).
- Chat with user in Dutch; docs in English.

## Decisions & Rationale
- **Shared sample = `smoke` preset (60k rows, seed 42)** in `docs/data/smoke.parquet`: same data as `tests/test_tier_equivalence.py`. Rejected 5k-row shrink: smoke runs fast enough.
- **T2.3 renamed `03-same-results-and-speed`** (user choice): `apriori(sparse=True)` measured slower than Polars; `apriori_from_csr` 14-38x faster. Both shown honestly.
- **Notebooks authored as jupytext percent sources** (`.claude/handoffs/nbsrc/*.py`), built by `build.sh` (expands `@@GPU_SETUP@@` from `gpu_setup.txt`, jupytext → ipynb, `fixmeta.py` sets python3 kernelspec, nbconvert executes). Sources are NOT committed; originals were in the ephemeral scratchpad.
- **GPU cells use integer items**: `_build_results_from_gpu` (`src/et_miner/gpu/mining.py:878`) maps via int64 array; string toy is remapped a..f→0..5 in T3.1.
- **Source quotes via `show()`** reading files at run time, so line numbers stay current.

## Dead Ends
- Do not build CSR with Polars join/list.unique in T2.3 `to_csr`: 48-50 s at 960k rows. Use the NumPy stable-sort key version (current).
- Do not call `apriori_streaming_multi_gpu` on smoke without `chunk_size`: 60k < 10M default takes the single-chunk shortcut (`src/et_miner/streaming/multi_gpu.py:231-246`), not multi-GPU SON. Now passes `chunk_size=15_000`.
- Do not use `uv sync` (exact) after building the extension: it uninstalls `et-miner-rust` (not in uv.lock). Use `uv sync --inexact`.

## Key Findings
- `apriori(sparse=True)` slower than Polars (0.3-0.46x) due to per-level `_polars_to_sparse_csr` + per-call CSC/bitvec rebuild in `rust_ext/src/core/counting.rs:count_itemsets_simd_raw`.
- Silently unused params: `sparse_from_k` on CPU/resident, `use_generator_pruning` on GPU, `level_callback` with streaming, `gpu_resident` in single-chunk SON (`src/et_miner/streaming/son.py:183-194`).
- `min_support=0` returns zero-support itemsets.
- SON skips failed chunk: `logger.warning("Chunk {} failed: {}", ...)` at `src/et_miner/streaming/son.py:258`.
- Resident route raises above `max_results=10_000_000` per level (`src/et_miner/gpu/kernels/loader.py:31 _warn_result_truncation`).
- Default build has no explicit SIMD/popcnt; `rust_ext/.cargo/config.toml` leaves target-cpu unset.

## Blockers & Pending Decisions
- No GPU in any available cloud environment: tier 3 execution needs the user or a GPU host.
- Pending: whether to watch PR #15.

## Test Status
- `docs/run_notebooks.sh`: 11/11 ok (2026-09-24).
- `uv run pytest tests/test_tier_equivalence.py -q`: 5 passed, 5 skipped (GPU).
- `cargo test --manifest-path rust_ext/Cargo.toml`: 53 passed.

## What Was Done
- `docs/README.md`, `docs/concepts.md`, `docs/run_notebooks.sh`, `docs/data/make_samples.py` (+ toy_8x6.parquet, smoke.parquet, smoke.json).
- Tier 1: 3 notebooks; Tier 2: `01-build-and-install.md` + 2 notebooks; Tier 3: 6 notebooks.
- Adversarial review: 12 claims corrected in commit `8df7b5f`.
- PR #15 opened.

## Commits This Session
- `8df7b5f` docs: correct claims found in review
- `f5deb77` docs: regenerate all notebooks from a fresh run and fix spelling
- `d4b6784` docs: add the docs index and the concepts and parameter reference
- `48ca9a3` docs: add tier 3 GPU and multi-GPU tutorials
- `0e524bf` docs: add tier 2 Rust/PyO3 tutorials
- `28b3f14` docs: declare the python3 kernel in tier 1 notebooks
- `1721409` docs: add tier 1 Polars tutorials, sample data and notebook runner

## Key Files
| File | Role |
|------|------|
| `docs/tier3-gpu/0[1-6]-*.ipynb` | GPU tutorials; GPU cells gated on `SKIP_GPU`/`SKIP_MULTI` |
| `.claude/handoffs/nbsrc/t3_0[1-6].py`, `gpu_setup.txt` | editable sources for the tier 3 notebooks |
| `.claude/handoffs/nbsrc/build.sh` | rebuild one notebook: `build.sh <src.py> <dest.ipynb>` (fix its hard-coded scratchpad path `S=` first) |
| `.claude/handoffs/nbsrc/verify.py` | static checks: links, anchors, source refs, forbidden words, style |
| `.claude/handoffs/nbsrc/outs.py` | print text outputs of a notebook: `outs.py <nb> [start_cell]` |
| `docs/run_notebooks.sh` | official runner, `docs/run_notebooks.sh tier3-gpu` |
| `docs/README.md` section 2 | runtime table to update after a GPU run |

## Extra Context
- Focus given: "tier 3 GPU documentatie".
- `build.sh` and `verify.py` reference the old scratchpad path `/tmp/claude-0/.../e480e379-.../scratchpad`; point `S=` at `.claude/handoffs/nbsrc` before reuse.
