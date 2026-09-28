# GPU-layer consolidation — handoff

State on 2026-09-28, branch `refactor/kernel-consolidation`. Phases 0–4 of the
consolidation brief are done on one GPU. Two decision points, DP6 and DP10, are
open: they need the second GPU, which failed mid-campaign.

## Where things are

| What | Where |
|---|---|
| Decisions, numbers, API removals, risks, review findings | `bench/consolidation/REPORT.md` |
| One row per decision point, with evidence | `bench/results/2026-09-27-consolidation/FINDINGS.md` |
| Protocol and decision rule (fixed before the campaign) | `bench/consolidation/PROTOCOL.md` |
| Routes, kernels and knobs as they were on `main` | `bench/consolidation/INVENTORY.md` |
| Phase 2 campaign data (at `e4bb3ae`) | `bench/results/2026-09-27-consolidation/` (`raw.jsonl`, `report.md`, `crossover.jsonl`, `microbench.jsonl`, `supplement/`) |
| Phase 4 re-runs on the consolidated tree | `bench/results/2026-09-28-consolidation-verify/`, `…-verify-postfix/` |
| User-facing changes | `CHANGELOG.md`, `### Removed` and the end of `### Fixed` |

## What the branch changed

- One in-core GPU miner: `gpu/row_split.py::_apriori_row_split_multi_gpu`
  serves `use_gpu=True` on one or more GPUs and `bitvecs=` (sharded by
  `shard_prebuilt_bitvecs`). The single-GPU bitvec miner and the GPU-resident
  miner are gone.
- Kernel choice per prefix group at the measured crossover:
  `gpu/row_split_chunks.py::TILED_MIN_GROUP_PAIRS` (tiled from 120 pairs at K=3
  down to 23 at K≥8, per-candidate below). On one GPU, a level whose dense
  counts exceed one chunk is counted by `count_tiled_fused`.
- SON: pass 1 runs the row-split miner per chunk, pass 2 the batched itemset
  kernel, on both SON paths.
- Removed, each raising `ValueError` with its replacement: `gpu_resident=True`,
  `sparse_from_k`, `prune_apriori`, `ET_MINER_KERNEL_VARIANT`.
  `ET_MINER_TILED_MIN_GROUP_PAIRS` now pins the dispatch (unset = measured).
- Seven registered kernels remain; `bench/selfcheck.py` launches each one.
- Tier chain renamed to the surviving routes, every kernel pinned by a leg's
  own environment and proven with call spies (`tests/test_tier_equivalence.py`,
  `CLAUDE.md`).

## Open work, in order

1. **Restore the second GPU.** GPU 1 (`0000:2B:00.0`) fell off the bus on
   2026-09-27 at 20:38 UTC; `nvidia-smi -L` reports "Unable to determine the
   device handle". It needs a host restart, which the container cannot do.
   While it is missing, NCCL init fails (the one-GPU paths skip NCCL) and
   every 2-GPU test skips.
2. **Verify the branch on two GPUs:**
   ```bash
   uv run pytest tests/test_tier_equivalence.py -q      # the three 2-GPU legs
   uv run pytest -q -m "gpu and multigpu"
   uv run python bench/selfcheck.py
   uv run python bench/runner.py --mode verify --out bench/results/<date>-consolidation-verify-2gpu
   ```
   `--mode verify` adds the C2 and E2 rows when two devices are present.
3. **Measure DP6 and DP10.** The missing rows are the 2-GPU configs of the
   Phase 2 matrix, which runs only at its own revision (the routes it compared
   are gone from this branch). Use a separate clone, not a worktree:
   ```bash
   git clone <repo> etm-campaign && cd etm-campaign && git checkout e4bb3ae
   uv sync --extra gpu && uv run maturin develop --release -m rust_ext/Cargo.toml
   uv run python -m et_miner.synthetic --preset all --out datasets/synth
   uv run python datasets/prepare_online_retail.py
   OUT=<this checkout>/bench/results/2026-09-27-consolidation
   uv run python bench/runner.py --mode consolidation --out $OUT --only C2
   uv run python bench/runner.py --mode consolidation --out $OUT --only split2
   uv run python bench/runner.py --mode consolidation --out $OUT --only E2
   uv run python bench/consolidation_report.py --out $OUT
   ```
   Then apply the decision rule from `PROTOCOL.md` to the DP6 and DP10
   tables and record the result in `FINDINGS.md` and `REPORT.md`. A losing
   knob value (`ET_MINER_FILTER_IMPL`, `ET_MINER_ROW_BALANCE`) must raise
   `ValueError` naming its replacement. The staged D2D reduce stays either way:
   it is the NCCL-absent fallback, not an A/B arm.
4. **Open review findings** (`REPORT.md`, "Review findings"): the intrinsic
   list in `CLAUDE.md` names deleted `.cu` files (the owner's edit), empty
   baskets that reach min_count crash the in-core GPU route, `anchor_items`
   with no frequent anchor returns the unanchored lattice, and the listed
   pre-existing silent drops.
5. **Untested hardware and regimes:** sm_90 (H100/H200), cards above 12 GB,
   AlphaFold scale (77M rows, K up to 22). The crossover is measured at K=3, 4,
   5, 6 and 8 only; `bench/kernel_crossover.py --ks 7,9,10` would close the
   gap.

## Working on this box

- Two RTX 3060 12 GB (sm_86), Ryzen 5 5600X, 31 GB RAM, about 5 GB of disk
  free; `/dev/shm` is RAM-backed and large.
- `uv sync` with its default cache can run out of disk; use
  `UV_CACHE_DIR=/dev/shm/uvcache` and `uv cache clean` afterwards. A sync
  removes `et_miner_rust`: rebuild with
  `env -u CONDA_PREFIX uv run maturin develop --release -m rust_ext/Cargo.toml`.
- With `datasets/online_retail_ii/` present, deselect
  `tests/test_smoke_correctness.py::*::test_support_001` and `::test_support_0001`:
  they never finish, and 0.0001 exhausts host RAM.
- `bench/runner.py` resumes past ok rows but re-runs failed ones. A config
  that sets a removed key (`sparse_from_k`, `gpu_resident`, `prune_apriori`)
  now fails instead of measuring another route.
- A wait loop keyed on `pgrep -f <script>` matches its own shell's command
  line and never ends; wait on a PID or on a file instead.
- `ty` is not installed; `cargo clippy -- -D warnings` fails with six errors
  that are also on `main`.

## Budget used

Phase 2: 4.68 GPU-hours in the campaign, about 0.8 for calibration and the
microbench dumps, 1.04 supplementary (kernel sweep and `oom_regression` to
K=3), about 6.5 of the 8 allowed. Phase 4 re-runs: 0.57 and 0.16.
