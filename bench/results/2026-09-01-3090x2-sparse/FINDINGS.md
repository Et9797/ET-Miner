# Sparse-CSR follow-up — 2×RTX 3090, 2026-09-01

Branch under test: `claude/sparse-csr-gpu-resident` (GPU-resident, row-split
sparse-CSR levels). Same box as the 2026-08-31 campaign (2× RTX 3090 24 GB,
sm_86, driver 580.159.03, CuPy 14.1.1, NCCL over SHM). Raw data:
`raw.jsonl`; per-config medians: `report.md`; environment: `env.txt`.
Matrix: the campaign's deep_k configs plus the new `deepk-density-auto-1g`.

## Correctness first

- Tier-equivalence chain 9/9 — now extended with single-GPU and multi-GPU
  sparse CSR legs (`sparse_from_k=3` on the smoke preset; CLAUDE.md).
- All 13 deep_k configs (dense legacy/shared × 1/2 GPUs, NCCL fallback,
  prefilter on/off, density-auto × 1/2 GPUs) produce the campaign's exact
  signature: `n_itemsets=8841`, `sum_counts=266261275`,
  `itemset_hash=4d2c8d28bcd33cd6…` — bit-identical to dense mining.
- Two exact alignment checks run inside every sparse run (per-shard row
  lengths vs dense counts at the transition, per-shard survivor lengths vs
  survivor counts at every materialization); both would raise, neither did.
- The closed-prune regression specs that failed on `main`
  (`tests/test_row_split_e2e.py::TestClosedPruning`, `TestTwoPhaseSparse`)
  pass: sparse+prune == dense+prune == CPU+prune.

## Finding 3, revisited — density-auto now costs the same as dense

The campaign measured `deepk-density-auto` at 10.4 s vs 1.3 s dense and
attributed it to the sparse path's host pair-building loop. The log
timestamps refuted that: pair building was ~30 ms per level, while ~2.5 s
per level (84%) went to rebuilding survivors' tidsets on the host with one
`np.intersect1d` per itemset, plus a Python decode loop.

| Config | 2026-08-31 | 2026-09-01 | vs dense (same run) |
|---|---|---|---|
| `deepk-density-auto` (2 GPUs) | 10.431 s | **1.343 s** | legacy-2g 1.338 s, shared-2g 1.35 s → **1.00×** |
| `deepk-density-auto-1g` (new) | — | **0.710 s** | legacy-1g 0.584 s → 1.22× |

Per-level, 2 GPUs (ms; the sparse path runs from K=5):

| K | frequent | density-auto 08-31 | density-auto 09-01 | dense legacy-2g |
|---|---|---|---|---|
| 5 | 1,854 | 3,145 | 48 (transition + first-use NVRTC compile of the CSR kernels) | 3.6 |
| 6 | 1,848 | 2,553 | 4.1 | 2.4 |
| 7 | 1,320 | 1,817 | 3.5 | 2.5 |
| 8 | 660 | 902 | 3.0 | 3.2 |
| 9 | 220 | 374 | 2.9 | 2.4 |
| 10 | 44 | 65 | 2.7 | 2.2 |
| 11 | 4 | 14 | 2.7 | 1.9 |

Sparse levels are now single-digit milliseconds — ~700× faster per level
than the host rebuild — and equal to dense within noise; the whole run is
the dense wall time. Peak VRAM 406/408 MB (dense 414/416 MB): the shards
replace the bitvecs on each GPU instead of a replicated host CSR being
re-uploaded to every GPU per level, so the memory model is the dense
path's again. Sparse levels' candidate counts now appear in `raw.jsonl`
(`n_candidates`, previously 0).

What changed (see the PR): each GPU converts its own bitvec shard to a
CSR of shard-local tids on-device; candidates are enumerated in-kernel
from the resident group arrays (new `csr_warp.cu`, one warp per
candidate, bounded binary-search intersection); per-GPU int32 partial
counts go through the same NCCL/staged reduce and compact filter as the
dense chunk loop; survivors' tidsets are materialized on-device in
survivor order. The n/32 crossover is unchanged; at deep_k scale it saves
almost nothing (156 MB tidsets vs 168 MB bitvecs at the transition) but
no longer costs anything either.

**Decision: defaults still unchanged** (`apriori(sparse_from_k=None)`).
"auto" is now safe to enable when memory is the constraint; its
throughput at large scale (many millions of sparse candidates per level,
rows no longer L2-resident) has not been measured on this box — the
kernel's documented follow-up is a merge-path intersection if profiling
ever shows the CSR kernels dominating a level.

## Other readings

- The single-GPU sparse levels cost ~11–15 ms each vs ~2–11 ms dense on
  one 3090 (K=6–9): the one-shard case pays the same per-level fixed
  costs (group upload, count + gather + write launches, D2H of survivor
  counts) with half the parallelism of the 2-GPU run. Not worth tuning at
  a 0.7 s wall.
- `deepk-nonccl` (0.90 s) is faster than NCCL (1.34 s) on this tiny data,
  as in the campaign: NCCL init dominates. Dense throttle flags (⚠) are
  the idle-clock reason bit, not thermal.
