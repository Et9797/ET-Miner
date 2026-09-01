"""Candidate-range chunk planning and the shared chunked dense-counting loop.

The row-split dense paths (K=2 and K>=3) count candidates into chunk-sized
int32 arrays sized from *measured* headroom, reduce the per-GPU partials,
and compact survivors per chunk. This module owns:

- the byte model (``chunk_budget_from_bytes`` — pure math, unit-tested at
  AlphaFold-scale numbers without a GPU),
- the VRAM measurement that honors per-device CuPy memory-pool limits
  (``compute_chunk_budget``),
- chunk planning (plain candidate ranges for K=2; group-aligned ranges for
  K>=3, ready for kernels that require whole prefix groups per chunk),
- the chunk loop itself (``run_chunked_dense_level``), shared by both the
  K=2 and K>=3 branches of ``row_split``.

Byte model per chunk candidate: 4 B for the int32 dense counts on every
GPU, plus reduce workspace — zero for the in-place NCCL path, another
4 B/candidate on GPU 0 for today's peer-copy fallback (a fixed staging
buffer replaces that term when the staged reduce lands). Survivor
compaction (12 B/survivor on GPU 0, ~48 B/survivor of host sort workspace)
is deliberately *not* budgeted per candidate: the filter checks its own
feasibility at call time and falls back to the bounded sliced-D2H valve,
so a degenerate ~100%-survivor chunk degrades to a slower path instead of
sizing every normal chunk for the worst case.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from loguru import logger

from et_miner import _env

#: int32 dense counts — one per candidate per GPU.
CHUNK_BYTES_PER_CANDIDATE = 4

#: Floor/fraction for the safety margin: max(1 GiB, 4% of device VRAM).
#: Replaces the old hardcoded 6 GiB, which was 25% of an RTX 3090.
MARGIN_FLOOR_BYTES = 1 << 30
MARGIN_VRAM_FRACTION = 0.04


class ChunkPlan(NamedTuple):
    """One contiguous candidate range [start, start + size)."""

    start: int
    size: int
    #: True when this range must run on the legacy per-candidate kernel
    #: (a single prefix group too large for group-aligned chunking).
    use_legacy: bool = False


def chunk_budget_from_bytes(
    avail_bytes: int,
    total_vram_bytes: int,
    group_data_bytes: int = 0,
    use_nccl: bool = True,
    staging_bytes: int = 0,
    env_cap: int | None = None,
) -> int:
    """Max candidates per dense chunk — the pure byte model.

    Args:
        avail_bytes: Measured available VRAM (min across GPUs, pool-limit
            aware — see ``compute_chunk_budget``).
        total_vram_bytes: Device VRAM (smallest participating GPU), for the
            fractional safety margin.
        group_data_bytes: Resident K>=3 group arrays (0 for K=2).
        use_nccl: In-place NCCL reduce (no extra per-candidate workspace)
            vs the fallback, which needs peer-copy/staging room on GPU 0.
        staging_bytes: Fixed staging buffer reserved by the non-NCCL
            reduce (0 while the fallback still full-copies peers).
        env_cap: ``ET_MINER_MAX_CHUNK_CANDS`` — caps (never raises) the
            computed budget so tests can force multi-chunk runs.

    Returns:
        Maximum candidates per chunk (>= 1).
    """
    # Safety margin: max(1 GiB, 4% of VRAM), but never more than a quarter
    # of what is actually available — a small pool limit must shrink the
    # chunks, not zero out the budget (1-candidate chunks are a de-facto
    # hang at scale).
    margin = max(MARGIN_FLOOR_BYTES, int(total_vram_bytes * MARGIN_VRAM_FRACTION))
    margin = min(margin, max(0, avail_bytes) // 4)
    usable = avail_bytes - group_data_bytes - margin - (0 if use_nccl else staging_bytes)
    if use_nccl or staging_bytes > 0:
        # counts + slack for allocator fragmentation
        per_candidate = CHUNK_BYTES_PER_CANDIDATE + 2
    else:
        # today's fallback materializes a full peer copy on GPU 0
        per_candidate = 2 * CHUNK_BYTES_PER_CANDIDATE + 2
    max_cands = max(1, int(usable // per_candidate))
    if env_cap is not None:
        max_cands = max(1, min(max_cands, env_cap))
    return max_cands


def _device_available_bytes(device_id: int) -> tuple[int, int]:
    """(available, total) VRAM for one device, honoring its pool limit.

    CuPy memory pools are per-device: each pool is read under its own
    device context, and a set pool limit caps availability at
    ``limit - used`` even when the physical device has more free.
    """
    import cupy as cp

    with cp.cuda.Device(device_id):
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()  # flush cached blocks for an accurate reading
        free, total = cp.cuda.Device().mem_info
        limit = pool.get_limit()
        if limit and limit > 0:
            free = min(free, max(0, limit - pool.used_bytes()))
            total = min(total, limit)
    return int(free), int(total)


def compute_chunk_budget(
    device_ids,
    group_data_bytes: int = 0,
    use_nccl: bool = True,
    staging_bytes: int | None = None,
) -> int:
    """Measure per-device headroom and apply the byte model.

    Availability is the minimum across participating GPUs (shards can be
    unequal), each measured under its own device context with its own
    pool limit honored. When NCCL is off, the fallback reduce's fixed
    staging buffer is reserved automatically.
    """
    if staging_bytes is None:
        from et_miner.gpu.nccl import STAGING_BYTES

        staging_bytes = 0 if use_nccl else STAGING_BYTES
    per_device = [_device_available_bytes(d) for d in device_ids]
    avail = min(free for free, _ in per_device)
    total = min(t for _, t in per_device)
    return chunk_budget_from_bytes(
        avail,
        total,
        group_data_bytes=group_data_bytes,
        use_nccl=use_nccl,
        staging_bytes=staging_bytes,
        env_cap=_env.max_chunk_candidates(),
    )


def plan_candidate_chunks(total_candidates: int, max_cands: int) -> list[ChunkPlan]:
    """Plain contiguous ranges (K=2 and legacy K>=3 kernels)."""
    if total_candidates <= 0:
        return []
    max_cands = max(1, max_cands)
    return [
        ChunkPlan(start, min(max_cands, total_candidates - start))
        for start in range(0, total_candidates, max_cands)
    ]


def plan_group_chunks(
    cumulative_pairs, max_cands: int, tiled_min_group_pairs: int | None = None
) -> list[ChunkPlan]:
    """Group-aligned contiguous ranges over the K>=3 candidate space.

    Every chunk boundary lands on a prefix-group boundary, which kernels
    that stage per-group state (the shared/tiled variant) require. Two
    kinds of group are routed to the legacy per-candidate kernel via
    ``use_legacy=True``:

    - **mega-groups** (pairs > ``max_cands``): cannot be group-aligned —
      emitted as plain candidate-range sub-chunks;
    - **tiny groups** (pairs < ``tiled_min_group_pairs``, default from
      ``ET_MINER_TILED_MIN_GROUP_PAIRS``): a 256-thread tile-pair block
      would idle on a handful of pairs, so contiguous runs of them go to
      the legacy kernel wholesale. ``tiled_min_group_pairs=0`` disables
      the routing.

    The plan is a deterministic function of its inputs, so every GPU in a
    row-split run derives the identical plan — a requirement for the
    collective reduce. Chunks are emitted in ascending candidate order and
    cover the space exactly.
    """
    if tiled_min_group_pairs is None:
        tiled_min_group_pairs = _env.tiled_min_group_pairs()
    cp_arr = np.asarray(cumulative_pairs, dtype=np.int64)
    n_groups = len(cp_arr) - 1
    total = int(cp_arr[-1]) if n_groups >= 0 and len(cp_arr) else 0
    if n_groups <= 0 or total <= 0:
        return []
    max_cands = max(1, max_cands)

    sizes = np.diff(cp_arr)
    # class 2 = mega (legacy sub-chunks), 1 = tiny (legacy), 0 = tiled
    klass = np.where(sizes > max_cands, 2, np.where(sizes < tiled_min_group_pairs, 1, 0))
    change = np.nonzero(np.diff(klass))[0] + 1
    if len(change) > 100_000:
        # Pathological tiny/tiled alternation would fragment the plan;
        # fall back to alignment-only classification (mega vs tiled).
        klass = np.where(sizes > max_cands, 2, 0)
        change = np.nonzero(np.diff(klass))[0] + 1
    run_bounds = np.concatenate([[0], change, [n_groups]])

    plans: list[ChunkPlan] = []
    for r in range(len(run_bounds) - 1):
        g_lo, g_hi = int(run_bounds[r]), int(run_bounds[r + 1])
        k = int(klass[g_lo])
        if k == 2:
            # Each mega-group individually sub-chunked by candidate range.
            for g in range(g_lo, g_hi):
                g_start, g_end = int(cp_arr[g]), int(cp_arr[g + 1])
                plans.extend(
                    ChunkPlan(start, min(max_cands, g_end - start), use_legacy=True)
                    for start in range(g_start, g_end, max_cands)
                )
            continue
        # Greedy whole-group chunks within the run, budget-bounded.
        use_legacy = k == 1
        g = g_lo
        while g < g_hi:
            start = int(cp_arr[g])
            j = int(np.searchsorted(cp_arr, start + max_cands, side="right")) - 1
            j = min(max(j, g + 1), g_hi)
            size = int(cp_arr[j]) - start
            if size > 0:
                plans.append(ChunkPlan(start, size, use_legacy=use_legacy))
            g = j
    return plans


def run_chunked_dense_level(
    bitvecs_list,
    chunks: list[ChunkPlan],
    launch_chunk,
    min_count: int,
    nccl_comms,
    use_nccl: bool,
    level_label: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Count → reduce → compact each chunk; return global survivors.

    For every chunk, each GPU counts the same candidate range against its
    own transaction shard (``launch_chunk(bitvec_gpu, device_id, chunk)``
    → device-local int32 counts), the partials are summed across GPUs, and
    GPU 0 compacts survivors. Per-chunk survivor indices are ascending and
    chunks are processed in ascending order, so the concatenated result
    keeps the global ascending-index contract.

    Returns:
        ``(indices, counts)`` — int64 NumPy arrays over the full candidate
        space (indices already offset by each chunk's start).
    """
    from concurrent.futures import ThreadPoolExecutor

    import cupy as cp

    from et_miner.gpu.kernels.filter import compact_threshold_filter
    from et_miner.gpu.nccl import reduce_sum_to_gpu0

    device_ids = [did for _, did, _ in bitvecs_list]
    all_indices: list[np.ndarray] = []
    all_counts: list[np.ndarray] = []

    with ThreadPoolExecutor(max_workers=len(bitvecs_list)) as pool:
        for chunk_idx, chunk in enumerate(chunks):
            if len(chunks) > 1:
                logger.debug(
                    f"    {level_label} chunk {chunk_idx + 1}/{len(chunks)}: "
                    f"candidates [{chunk.start:,}, {chunk.start + chunk.size:,})"
                    + (" (legacy sub-chunk)" if chunk.use_legacy else "")
                )

            futures = [pool.submit(launch_chunk, bv, did, chunk) for bv, did, _ in bitvecs_list]
            gpu_results = [f.result() for f in futures]

            # Sum partials onto GPU 0: ncclReduce to root, or the bounded
            # staged D2D fallback (never a full peer copy).
            reduce_sum_to_gpu0(gpu_results, device_ids, comms=nccl_comms if use_nccl else None)

            with cp.cuda.Device(device_ids[0]):
                global_counts = gpu_results[0]
                freq_idx, freq_cnt = compact_threshold_filter(global_counts, min_count)
                n_freq_chunk = len(freq_idx)
                pass_rate = 100 * n_freq_chunk / chunk.size if chunk.size else 0.0
                logger.info(
                    f"  {level_label} chunk {chunk_idx + 1}/{len(chunks)} filtering: "
                    f"{chunk.size:,} candidates → {n_freq_chunk:,} frequent "
                    f"({pass_rate:.1f}% pass rate, min_count={min_count:,})"
                )
                if n_freq_chunk > 0:
                    all_indices.append(freq_idx + chunk.start)
                    all_counts.append(freq_cnt)

                del global_counts
                for i in range(len(gpu_results)):
                    gpu_results[i] = None
                del gpu_results
                for _, did, _ in bitvecs_list:
                    with cp.cuda.Device(did):
                        cp.get_default_memory_pool().free_all_blocks()

    if not all_indices:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(all_indices), np.concatenate(all_counts)
