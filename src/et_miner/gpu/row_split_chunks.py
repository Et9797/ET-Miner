"""Candidate-range chunk planning and the shared chunked dense-counting loop.

The row-split dense paths (K=2 and K>=3) count candidates into chunk-sized
int32 arrays sized from *measured* headroom, reduce the per-GPU partials,
and filter survivors per chunk. This module owns:

- the byte model (``chunk_budget_from_bytes`` — pure math, unit-tested at
  AlphaFold-scale numbers without a GPU),
- the VRAM measurement that honors per-device CuPy memory-pool limits
  (``compute_chunk_budget``),
- which kernel counts a prefix group (``tiled_min_group_pairs``, the
  measured crossover),
- chunk planning (plain candidate ranges for the per-candidate kernel;
  group-aligned ranges for the tiled kernel, which needs whole prefix groups
  per chunk),
- the chunk loop itself (``run_chunked_dense_level``), shared by both the
  K=2 and K>=3 branches of ``row_split``.

Byte model per chunk candidate: 4 B for the int32 dense counts on every
GPU (plus slack), 1 bit on GPU 0 for the compacted reduce's mask of written
entries, and for the non-NCCL reduce a fixed staging buffer on
GPU 0 (``gpu.nccl.STAGING_BYTES``) instead of any per-candidate term. Survivor
filtering is deliberately *not* budgeted per candidate: the filter works
in 64M-element slices whose worst case (13 B/element, every element
surviving) fits the safety margin, and a slice that does not fit is
filtered on the host, so a degenerate ~100%-survivor chunk degrades to a
slower path instead of sizing every normal chunk for the worst case.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from loguru import logger

from et_miner import _env

#: int32 dense counts — one per candidate per GPU.
CHUNK_BYTES_PER_CANDIDATE = 4

#: The compacted reduce's packed mask of written entries, on the reducing GPU.
COMPACT_MASK_BITS_PER_CANDIDATE = 1

#: Pairs per prefix group at which the tiled kernel becomes faster than the
#: per-candidate kernel, per K. Measured by bench/kernel_crossover.py
#: (bench/results/2026-09-27-consolidation/crossover.jsonl): the ratio does
#: not depend on the row count, since both kernels scale with the words, and
#: it falls with K, since the per-candidate kernel reads every prefix word for
#: every candidate. K=7 sits between its measured neighbours; beyond K=8 the
#: K=8 value holds. K=2 is one group with an empty prefix, counted as K=3.
TILED_MIN_GROUP_PAIRS = {2: 120, 3: 120, 4: 91, 5: 66, 6: 45, 7: 32, 8: 23}


def tiled_min_group_pairs(k: int) -> int:
    """Pairs a prefix group needs to be counted by the tiled kernel at level ``k``.

    ``ET_MINER_TILED_MIN_GROUP_PAIRS`` pins it for every level (0 = tiled for
    every group); unset, it is the measured crossover.
    """
    pinned = _env.tiled_min_group_pairs()
    if pinned is not None:
        return pinned
    return TILED_MIN_GROUP_PAIRS.get(k, TILED_MIN_GROUP_PAIRS[max(TILED_MIN_GROUP_PAIRS)])


#: Floor/fraction for the safety margin: max(1 GiB, 4% of device VRAM).
#: Replaces the old hardcoded 6 GiB, which was 25% of an RTX 3090.
MARGIN_FLOOR_BYTES = 1 << 30
MARGIN_VRAM_FRACTION = 0.04


class ChunkPlan(NamedTuple):
    """One contiguous candidate range [start, start + size)."""

    start: int
    size: int
    #: True when this range runs on the per-candidate kernel rather than the
    #: tiled one (small groups, or a group too large for group-aligned chunks).
    per_candidate: bool = False


def chunk_budget_from_bytes(
    avail_bytes: int,
    total_vram_bytes: int,
    group_data_bytes: int = 0,
    use_nccl: bool = True,
    staging_bytes: int | None = None,
    env_cap: int | None = None,
    compact_reduce: bool = False,
) -> int:
    """Max candidates per dense chunk — the pure byte model.

    Args:
        avail_bytes: Measured available VRAM (min across GPUs, pool-limit
            aware — see ``compute_chunk_budget``).
        total_vram_bytes: Device VRAM (smallest participating GPU), for the
            fractional safety margin.
        group_data_bytes: Resident K>=3 group arrays (0 for K=2).
        use_nccl: In-place NCCL reduce (no reduce workspace) vs the
            staged fallback, which reserves its staging buffer on GPU 0.
        staging_bytes: The non-NCCL reduce's fixed staging buffer; None
            means ``gpu.nccl.STAGING_BYTES`` (0 with NCCL).
        env_cap: ``ET_MINER_MAX_CHUNK_CANDS`` — caps (never raises) the
            computed budget so tests can force multi-chunk runs.
        compact_reduce: The chunks are reduced compacted, which keeps a
            1-bit mask per candidate on GPU 0.

    Returns:
        Maximum candidates per chunk (>= 1).
    """
    # Safety margin: max(1 GiB, 4% of VRAM), but never more than a quarter
    # of what is actually available — a small pool limit must shrink the
    # chunks, not zero out the budget (1-candidate chunks are a de-facto
    # hang at scale).
    if staging_bytes is None:
        from et_miner.gpu.nccl import STAGING_BYTES

        staging_bytes = STAGING_BYTES
    margin = max(MARGIN_FLOOR_BYTES, int(total_vram_bytes * MARGIN_VRAM_FRACTION))
    margin = min(margin, max(0, avail_bytes) // 4)
    usable = avail_bytes - group_data_bytes - margin - (0 if use_nccl else staging_bytes)
    per_candidate = CHUNK_BYTES_PER_CANDIDATE + 2  # counts + slack for allocator fragmentation
    per_candidate_bits = 8 * per_candidate + (COMPACT_MASK_BITS_PER_CANDIDATE if compact_reduce else 0)
    max_cands = max(1, int(max(0, usable) * 8 // per_candidate_bits))
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
    compact_reduce: bool = False,
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
        compact_reduce=compact_reduce,
    )


def plan_candidate_chunks(total_candidates: int, max_cands: int) -> list[ChunkPlan]:
    """Plain contiguous ranges for the per-candidate kernel."""
    if total_candidates <= 0:
        return []
    max_cands = max(1, max_cands)
    return [
        ChunkPlan(start, min(max_cands, total_candidates - start), per_candidate=True)
        for start in range(0, total_candidates, max_cands)
    ]


def plan_group_chunks(cumulative_pairs, max_cands: int) -> list[ChunkPlan]:
    """Group-aligned contiguous ranges over a candidate space, for the tiled kernel.

    Every chunk boundary lands on a prefix-group boundary, which the tiled
    kernel requires. A group larger than ``max_cands`` cannot be group-aligned
    and is split into plain candidate ranges on the per-candidate kernel.

    The plan is a deterministic function of its inputs, so every GPU in a
    row-split run derives the identical plan — a requirement for the
    collective reduce. Chunks are emitted in ascending candidate order and
    cover the space exactly.
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
        start, end = int(cp_arr[g]), int(cp_arr[g + 1])
        if end - start > max_cands:
            plans.extend(
                ChunkPlan(s, min(max_cands, end - s), per_candidate=True) for s in range(start, end, max_cands)
            )
            g += 1
            continue
        # Greedy whole-group chunk: every following group that still fits.
        j = int(np.searchsorted(cp_arr, start + max_cands, side="right")) - 1
        j = min(max(j, g + 1), n_groups)
        if int(cp_arr[j]) - start > 0:
            plans.append(ChunkPlan(start, int(cp_arr[j]) - start))
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
    compact: bool = False,
    peer_counts: bool = False,
) -> tuple:
    """Count → reduce → compact each chunk; return global survivors.

    For every chunk, each GPU counts the same candidate range against its
    own transaction shard (``launch_chunk(bitvec_gpu, device_id, chunk)``
    → device-local int32 counts), the partials are summed across GPUs, and
    GPU 0 filters survivors. Per-chunk survivor indices are ascending and
    chunks are processed in ascending order, so the concatenated result
    keeps the global ascending-index contract.

    With ``compact`` the launches leave the entries they do not write at
    ``filter.UNTOUCHED`` and write the same entries on every GPU (the subset
    test is a function of the shared index). Each GPU moves its written
    entries to the front of its array, and only that prefix is reduced; GPU 0
    maps the survivors back through its mask of the written entries. GPUs that
    wrote different entries raise RuntimeError before the collective.

    With ``peer_counts`` (dense reduce only) the result gains the survivors'
    partial counts on every GPU but the first: int64 ``(len(bitvecs_list) - 1,
    n)``, read from those GPUs' arrays after the reduce, or None when the
    reduce in use overwrites them (``nccl.keeps_peer_arrays``).

    Returns:
        ``(indices, counts)`` — int64 NumPy arrays over the full candidate
        space (indices already offset by each chunk's start) — and the peer
        counts as a third element with ``peer_counts``.
    """
    from concurrent.futures import ThreadPoolExecutor

    import cupy as cp

    from et_miner.gpu.kernels.filter import compact_written, threshold_filter, threshold_filter_compacted
    from et_miner.gpu.nccl import keeps_peer_arrays, reduce_sum_to_gpu0

    if peer_counts and compact:
        raise ValueError("peer_counts needs the dense reduce: the compacted one moves the peers' entries")
    device_ids = [did for _, did, _ in bitvecs_list]
    n_peers = len(bitvecs_list) - 1
    keep_peers = peer_counts and keeps_peer_arrays(nccl_comms if use_nccl else None)
    if peer_counts and not keep_peers:
        logger.warning(f"  {level_label}: the allReduce fallback overwrites the per-GPU partial counts; none kept")
    all_indices: list[np.ndarray] = []
    all_counts: list[np.ndarray] = []
    all_peers: list[np.ndarray] = []

    def _gather(counts, device_id, idx):
        with cp.cuda.Device(device_id):
            return counts[cp.asarray(idx)].get().astype(np.int64)

    with ThreadPoolExecutor(max_workers=len(bitvecs_list)) as pool:
        for chunk_idx, chunk in enumerate(chunks):
            if len(chunks) > 1:
                logger.debug(
                    f"    {level_label} chunk {chunk_idx + 1}/{len(chunks)}: "
                    f"candidates [{chunk.start:,}, {chunk.start + chunk.size:,})"
                    + (" (per-candidate)" if chunk.per_candidate else "")
                )

            futures = [pool.submit(launch_chunk, bv, did, chunk) for bv, did, _ in bitvecs_list]
            gpu_results = [f.result() for f in futures]

            # Sum partials onto GPU 0: ncclReduce to root, or the bounded
            # staged D2D fallback (never a full peer copy).
            kept = None
            if compact:

                def _compact(i, counts):
                    with cp.cuda.Device(device_ids[i]):
                        return compact_written(counts, keep_mask=i == 0)

                packed = list(pool.map(_compact, range(len(gpu_results)), gpu_results))
                kept = packed[0]
                if any(p.n != kept.n or not np.array_equal(p.slice_counts, kept.slice_counts) for p in packed[1:]):
                    raise RuntimeError(
                        f"{level_label} chunk {chunk_idx + 1}/{len(chunks)}: the GPUs wrote different entries "
                        f"({[p.n for p in packed]}); the compacted reduce needs the same entries on every GPU"
                    )
                del packed
                if kept.n:
                    reduce_sum_to_gpu0(
                        [r[: kept.n] for r in gpu_results], device_ids, comms=nccl_comms if use_nccl else None
                    )
            else:
                reduce_sum_to_gpu0(gpu_results, device_ids, comms=nccl_comms if use_nccl else None)

            with cp.cuda.Device(device_ids[0]):
                global_counts = gpu_results[0]
                if kept is None:
                    freq_idx, freq_cnt = threshold_filter(global_counts, min_count)
                else:
                    freq_idx, freq_cnt = threshold_filter_compacted(global_counts, kept, min_count)
                n_freq_chunk = len(freq_idx)
                pass_rate = 100 * n_freq_chunk / chunk.size if chunk.size else 0.0
                logger.info(
                    f"  {level_label} chunk {chunk_idx + 1}/{len(chunks)} filtering: "
                    f"{chunk.size:,} candidates → {n_freq_chunk:,} frequent "
                    f"({pass_rate:.1f}% pass rate, min_count={min_count:,})"
                    + ("" if kept is None else f", {kept.n:,} reduced compacted")
                )
                if n_freq_chunk > 0:
                    all_indices.append(freq_idx + chunk.start)
                    all_counts.append(freq_cnt)
                    if keep_peers:
                        all_peers.append(
                            np.stack([_gather(gpu_results[i], device_ids[i], freq_idx) for i in range(1, n_peers + 1)])
                            if n_peers else np.empty((0, n_freq_chunk), dtype=np.int64)
                        )

                del global_counts, kept
                for i in range(len(gpu_results)):
                    gpu_results[i] = None
                del gpu_results
                for _, did, _ in bitvecs_list:
                    with cp.cuda.Device(did):
                        cp.get_default_memory_pool().free_all_blocks()

    indices = np.concatenate(all_indices) if all_indices else np.empty(0, dtype=np.int64)
    counts = np.concatenate(all_counts) if all_counts else np.empty(0, dtype=np.int64)
    if not peer_counts:
        return indices, counts
    if not keep_peers:
        return indices, counts, None
    peers = np.concatenate(all_peers, axis=1) if all_peers else np.empty((n_peers, 0), dtype=np.int64)
    return indices, counts, peers
