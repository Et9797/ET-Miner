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
    margin = max(MARGIN_FLOOR_BYTES, int(total_vram_bytes * MARGIN_VRAM_FRACTION))
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


def plan_group_chunks(cumulative_pairs, max_cands: int) -> list[ChunkPlan]:
    """Group-aligned contiguous ranges over the K>=3 candidate space.

    Every chunk boundary lands on a prefix-group boundary, which kernels
    that stage per-group state (the shared/tiled variant) require. A single
    group whose pair count exceeds ``max_cands`` cannot be group-aligned;
    it is emitted as plain candidate-range sub-chunks flagged
    ``use_legacy=True`` so the per-candidate legacy kernel handles it.

    The plan is a deterministic function of (cumulative_pairs, max_cands),
    so every GPU in a row-split run derives the identical plan — a
    requirement for the collective reduce.

    Args:
        cumulative_pairs: int64 array, len n_groups + 1; prefix sums of
            per-group candidate counts (``K3PlusGroups.cumulative_pairs``).
        max_cands: Chunk budget from ``compute_chunk_budget``.
    """
    cp_arr = np.asarray(cumulative_pairs, dtype=np.int64)
    n_groups = len(cp_arr) - 1
    total = int(cp_arr[-1]) if n_groups >= 0 and len(cp_arr) else 0
    if n_groups <= 0 or total <= 0:
        return []
    max_cands = max(1, max_cands)

    plans: list[ChunkPlan] = []
    g = 0
    while g < n_groups:
        g_start = int(cp_arr[g])
        g_size = int(cp_arr[g + 1]) - g_start
        if g_size > max_cands:
            # Mega-group: sub-chunk by candidate range on the legacy kernel.
            plans.extend(
                ChunkPlan(start, min(max_cands, g_start + g_size - start), use_legacy=True)
                for start in range(g_start, g_start + g_size, max_cands)
            )
            g += 1
            continue
        # Greedily take whole groups while the range stays within budget.
        # searchsorted(right) - 1 = last boundary <= g_start + max_cands.
        j = int(np.searchsorted(cp_arr, g_start + max_cands, side="right")) - 1
        j = max(j, g + 1)
        plans.append(ChunkPlan(g_start, int(cp_arr[j]) - g_start))
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
