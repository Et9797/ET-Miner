# GPU-layer consolidation — handoff

State on 2026-09-28 (evening), branch `refactor/kernel-consolidation`. Every
decision point of the brief is closed: DP1–DP5 and DP7–DP9 on the first box
(2× RTX 3060 12 GB), DP6 and DP10 on a second box (2× RTX A4000 16 GB) after
the first one lost a GPU. The work after `93224ef` is committed on the
owner's word (one commit per decision point, `COMMITS.md` has the messages),
the branch is pushed and a pull request against `main` is open.

## Where things are

| What | Where |
|---|---|
| Decisions, numbers, API removals, risks, review findings | `bench/consolidation/REPORT.md` |
| One row per decision point, with evidence | `bench/results/2026-09-27-consolidation/FINDINGS.md` |
| Protocol and decision rule (fixed before the campaign) | `bench/consolidation/PROTOCOL.md` |
| Routes, kernels and knobs as they were on `main` | `bench/consolidation/INVENTORY.md` |
| Phase 2 campaign data (at `e4bb3ae`, 3060 box) | `bench/results/2026-09-27-consolidation/` |
| Phase 4 re-run on the consolidated tree (3060 box) | `bench/results/2026-09-28-consolidation-verify/`, `…-verify-postfix/` |
| The 2-GPU rows of the Phase 2 matrix (DP6, DP10; A4000 box, at `e4bb3ae`) | `bench/results/2026-09-28-consolidation-2gpu/` (`report.md`, `raw.jsonl`, `nccl-hang/`) |
| Verify matrix on the final tree (A4000 box, two GPUs) | `bench/results/2026-09-28-consolidation-verify-2gpu/` |
| User-facing changes | `CHANGELOG.md`, `### Removed` and the top of `### Fixed` |

## What the branch changed

- One in-core GPU miner: `gpu/row_split.py::_apriori_row_split_multi_gpu`
  serves `use_gpu=True` on one or more GPUs and `bitvecs=` (sharded by
  `shard_prebuilt_bitvecs`). The single-GPU bitvec miner and the GPU-resident
  miner are gone.
- Kernel choice per prefix group at the measured crossover
  (`gpu/row_split_chunks.py::TILED_MIN_GROUP_PAIRS`). On one GPU, a level
  whose dense counts exceed one chunk is counted by `count_tiled_fused`.
- SON: pass 1 runs the row-split miner per chunk, pass 2 the batched itemset
  kernel, on both SON paths.
- Multi-GPU (DP6): C's row split with the NCCL reduce; the A/B
  pair/candidate splits won nothing on two GPUs.
- DP10: the sliced CuPy survivor filter is the one implementation
  (`gpu/kernels/filter.py::threshold_filter`); the `compact_threshold` kernel,
  the whole-array CPU path and `ET_MINER_FILTER_IMPL` are gone. The nnz row
  balance is gone (`ET_MINER_ROW_BALANCE=nnz` and `balance="nnz"` raise).
- Removed, each raising `ValueError` with its replacement: `gpu_resident=True`,
  `sparse_from_k`, `prune_apriori`, `ET_MINER_KERNEL_VARIANT`,
  `ET_MINER_FILTER_IMPL`, `ET_MINER_ROW_BALANCE=nnz`.
  `ET_MINER_TILED_MIN_GROUP_PAIRS` pins the dispatch (unset = measured).
- Six registered kernels remain; `bench/selfcheck.py` launches each one.
- Peer-copy probe (`gpu/nccl.py::peer_copy_works`): on a box whose PCIe P2P
  drops device-to-device writes, the staged reduce and the `bitvecs=` shards
  go through host memory and NCCL starts with `NCCL_P2P_DISABLE=1`.
- Tier chain: every surviving kernel pinned by a leg's own environment and
  proven with call spies (`tests/test_tier_equivalence.py`, `CLAUDE.md`); the
  three 2-GPU legs pass on the A4000 box.

## Open work, in order

1. **Review the two rule-5 outcomes** before merging (they follow the
   protocol mechanically, and the owner may weigh the unmeasured regime):
   the `compact_threshold` kernel went on a tie (it was never faster in a
   measured regime; at the unmeasured 10B-candidate levels the sliced filter
   costs ≈ 15 s per level where the kernel took two passes), and the nnz row
   balance went on a tie.
2. **Open review findings** (`REPORT.md`, "Review findings"): `CLAUDE.md`'s
   intrinsic list names three deleted `.cu` files (owner's edit); a hung NCCL
   collective never times out (finding 22); empty baskets that reach
   min_count crash the in-core GPU route; `anchor_items` with no frequent
   anchor returns the unanchored lattice; the listed pre-existing silent
   drops.
3. **Untested hardware and regimes:** sm_90 (H100/H200), cards above 16 GB,
   AlphaFold scale (77M rows, K up to 22). The crossover is measured at K=3,
   4, 5, 6 and 8 only; `bench/kernel_crossover.py --ks 7,9,10` would close
   the gap.

## Working on the A4000 box

- 2× RTX A4000 16 GB (sm_86) behind the CPU's host bridge (`nvidia-smi topo
  -m`: PHB), Ryzen 5 5600X, 46 GB RAM; CuPy 14.1.1, NCCL 2.31.
- **PCIe P2P drops writes from GPU 1 to GPU 0** (the other direction lands).
  Export `NCCL_P2P_DISABLE=1` for every two-GPU job (the miner now sets it
  itself when its probe fails, but NCCL reads it once per process, so set it
  before anything else initialises NCCL). Never run P2P traffic (the probe
  tests, `nccl-hang/repro_nnz2.py`, cudaMemcpyPeer experiments) next to a
  measurement on the same GPU: a dropped P2P write landed in another
  process's memory once (`REPORT.md`, "Incident during this run").
- The campaign clone at `e4bb3ae` is `/root/projects/etm-campaign` (venv with
  CuPy and `et_miner_rust`; `datasets/` symlinked to this checkout). Run it
  with `env -u CONDA_PREFIX uv run --no-sync …` so uv never re-syncs the venv.
- `py-spy` cannot ptrace in this container; for a thread dump use
  `faulthandler.dump_traceback_later` or `PYTHONFAULTHANDLER=1` and SIGABRT.
- With `datasets/online_retail_ii/` present, deselect
  `tests/test_smoke_correctness.py::*::test_support_001` and `::test_support_0001`
  (they never finish; 0.0001 exhausts host RAM).
- `bench/runner.py` resumes past ok rows but re-runs failed ones, and stamps
  rows with a dirty hash of the tree: edit nothing under `src/` while a matrix
  runs, or its coverage summary reports every later row as stale.
- `ty` is not installed; `cargo clippy -- -D warnings` fails with six errors
  that are also on `main`.

## Budget used

Phase 2: 4.68 GPU-hours in the campaign, about 0.8 for calibration and the
microbench dumps, 1.04 supplementary (kernel sweep and `oom_regression` to
K=3), Phase 4 re-runs 0.57 and 0.16 — about 7.2 of the 8 allowed on the
first box. Second box: 3.44 GPU-hours for the 2-GPU rows (82 config-reps),
the verify matrix on two GPUs 0.76, and about 1.5 for the
hang and corruption diagnostics (30 workaround repro runs, three
`stress_k2` K=3 dumps).
