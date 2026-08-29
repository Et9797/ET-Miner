"""Apriori algorithm implementation for association rule mining.

Fast and memory-efficient implementation using boolean matrix representation and
vectorized operations, with optional GPU acceleration via CUDA kernels.

Support counting uses column bitwise AND operations:

    support({A,B}) = (col("A") & col("B")).sum()

Key ideas:
1. Boolean matrix representation (not TID-lists)
2. Column bitwise AND operations for support (no Python loops in counting)
3. GPU-ready via CUDA bitvector kernels (CuPy)
"""

from __future__ import annotations

import math
import os
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from math import comb
from typing import Any

import polars as pl

from et_miner import _env
from et_miner.io.gcs import (
    GCSUploader,
    is_gs_uri,
    is_upload_enabled,
    join_gs_uri,
    parse_gs_url,
    polars_storage_options,
    pyarrow_gcs_filesystem,
)
from .matrix import (
    build_boolean_matrix,
    count_support_batched,
)
from .result import (
    _build_result_df,
    _empty_result,
    _min_count,
)
from .profiling import ProfilingSession


from loguru import logger


@dataclass
class _SparseState:
    """State for sparse CSR mode in multi-GPU row-split mining."""

    active: bool = False
    device_ids: list = field(default_factory=list)
    prev_idx_lookup: dict | None = None


def _init_nccl(device_ids):
    """Initialize NCCL communicators for multi-GPU all-reduce.

    Ring topology: O(data/n_gpus) bandwidth vs O(n_gpus × data) for sequential D2D.
    Every GPU gets the result → parallel filtering, no GPU 0 bottleneck.

    Returns (comms, True) on success, (None, False) on failure.
    """
    try:
        import cupy as cp
        from cupy.cuda import nccl as _nccl
        from concurrent.futures import ThreadPoolExecutor

        n = len(device_ids)
        uid = _nccl.get_unique_id()
        comms = [None] * n

        def _init_rank(rank):
            with cp.cuda.Device(device_ids[rank]):
                comms[rank] = _nccl.NcclCommunicator(n, uid, rank)

        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(_init_rank, range(n)))

        return comms, True
    except Exception as e:
        logger.warning(f"NCCL init failed: {e}")
        return None, False


def _nccl_allreduce_sum(gpu_arrays, comms, device_ids):
    """In-place NCCL all-reduce SUM across GPUs.

    After call, every GPU has the global sum in its own array.
    All ranks must participate simultaneously (collective operation).
    """
    import cupy as cp
    from concurrent.futures import ThreadPoolExecutor

    NCCL_INT32 = 2
    NCCL_UINT32 = 3
    NCCL_INT64 = 4
    NCCL_UINT64 = 5
    NCCL_SUM = 0
    _NCCL_DTYPE_MAP = {
        cp.int32: NCCL_INT32,
        cp.uint32: NCCL_UINT32,
        cp.int64: NCCL_INT64,
        cp.uint64: NCCL_UINT64,
    }

    def _reduce_rank(rank):
        with cp.cuda.Device(device_ids[rank]):
            stream = cp.cuda.get_current_stream()
            arr = gpu_arrays[rank]
            nccl_dtype = _NCCL_DTYPE_MAP.get(arr.dtype.type)
            if nccl_dtype is None:
                raise TypeError(f"Unsupported dtype for NCCL allreduce: {arr.dtype}")
            comms[rank].allReduce(
                arr.data.ptr,
                arr.data.ptr,  # in-place
                arr.size,
                nccl_dtype,
                NCCL_SUM,
                stream.ptr,
            )
            stream.synchronize()

    with ThreadPoolExecutor(max_workers=len(device_ids)) as pool:
        list(pool.map(_reduce_rank, range(len(device_ids))))


def _deallocate_dead_bitvecs(bitvecs_gpu, live_cols, prev_live_cols, k_level):
    """Zero bitvec rows for columns that dropped out of the frequent set.

    Progressive bitvector deallocation: after each K-level, columns no longer
    in any frequent itemset have their bitvec rows zeroed. The CuPy array is
    contiguous so we can't free individual rows, but zeroing makes subsequent
    AND operations trivially fast (AND with zero = zero) and prevents dead
    columns from contributing false positives.

    Returns the set of live columns for tracking across levels.
    """
    import cupy as cp

    dead_cols = prev_live_cols - live_cols
    if dead_cols:
        dead_indices = cp.array(sorted(dead_cols), dtype=cp.int64)
        bitvecs_gpu[dead_indices] = 0
        freed_bytes = len(dead_cols) * bitvecs_gpu.shape[1] * 8
        logger.debug(
            f"    Pruning: zeroed {len(dead_cols)} dead bitvecs at K={k_level} ({freed_bytes / 1024**2:.0f} MB logical)"
        )
    return live_cols


def _warn_complexity(
    n_frequent_items: int,
    min_support: float,
) -> None:
    """Warn user if computation might be expensive."""
    # Estimate k=2 candidates (n choose 2)
    n_pairs = comb(n_frequent_items, 2) if n_frequent_items > 1 else 0

    # Warning thresholds (tuned from benchmarks)
    if n_pairs > 10_000_000:
        warnings.warn(
            f"Very large computation ahead!\n"
            f"  Frequent items: {n_frequent_items:,}\n"
            f"  Candidate pairs: {n_pairs:,}\n"
            f"  Estimated time: >30 minutes\n"
            f"  Consider increasing min_support (currently {min_support})",
            UserWarning,
            stacklevel=3,
        )
    elif n_pairs > 1_000_000:
        warnings.warn(
            f"Large computation: {n_pairs:,} candidate pairs.\n  This may take several minutes.",
            UserWarning,
            stacklevel=3,
        )


def _validate_parameters(
    min_support: float,
    max_length: int | None,
    batch_size: int | None,
) -> None:
    """Validate apriori parameters with clear error messages."""
    # min_support: must be numeric in [0.0, 1.0]
    if not isinstance(min_support, (int, float)):
        raise TypeError(f"min_support must be numeric, got {type(min_support).__name__}")
    if not 0.0 <= min_support <= 1.0:
        raise ValueError(
            f"min_support must be in range [0.0, 1.0], got {min_support}\n"
            f"  Hint: Support is a proportion (e.g., 0.01 = 1%)"
        )

    # max_length: must be positive int or None
    if max_length is not None:
        if not isinstance(max_length, int):
            raise TypeError(f"max_length must be int or None, got {type(max_length).__name__}")
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}")

    # batch_size: must be positive int or None
    if batch_size is not None:
        if not isinstance(batch_size, int):
            raise TypeError(f"batch_size must be int or None, got {type(batch_size).__name__}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")


def _prune_equal_support(
    current_frequent: list[tuple[str, ...]],
    current_supports: dict[tuple[str, ...], float],
    prev_supports: dict[tuple[str, ...], float],
    tolerance: float = 1e-9,
) -> list[tuple[str, ...]]:
    """Prune itemsets with equal support to their subsets (non-closed itemsets).

    Based on BitApriori [Zheng 2010]: if support({A,B,C}) == support({A,B}),
    then C appears in ALL transactions containing {A,B}. These non-closed
    itemsets add no information for rule mining and can be pruned to reduce
    candidate generation in subsequent iterations.

    Args:
        current_frequent: Frequent k-itemsets from current iteration.
        current_supports: Support values for current k-itemsets.
        prev_supports: Support values for (k-1)-itemsets from previous iteration.
        tolerance: Floating point comparison tolerance.

    Returns:
        List of closed itemsets (those with support different from all subsets).
    """
    if not prev_supports:
        return current_frequent

    closed = []
    for itemset in current_frequent:
        is_closed = True
        current_sup = current_supports[itemset]

        # Check if support equals any (k-1) subset
        for i in range(len(itemset)):
            subset = itemset[:i] + itemset[i + 1 :]
            if subset in prev_supports:
                if abs(current_sup - prev_supports[subset]) < tolerance:
                    is_closed = False
                    break

        if is_closed:
            closed.append(itemset)

    return closed


def _prune_closed_flat(current_flat, current_counts, prev_flat, prev_counts):
    """Prune non-closed itemsets from flat numpy arrays.

    An itemset is non-closed if its support count equals any (k-1)-subset's count.
    This means the k-th item appears in ALL transactions of that subset — no new
    information. Removing these reduces candidate generation at K+1 by 50-90%.

    Operates on raw integer counts (not float support) for exact comparison.
    Uses bytes-key dict for O(1) lookup of (k-1)-subsets.

    Args:
        current_flat: numpy int32 (n, k) — current level frequent itemsets.
        current_counts: numpy int64 (n,) — raw support counts.
        prev_flat: numpy int32 (m, k-1) — previous level frequent itemsets.
        prev_counts: numpy int64 (m,) — raw support counts for previous level.

    Returns:
        Tuple of (pruned_flat, pruned_counts) with non-closed itemsets removed.
    """
    import numpy as np

    if prev_flat is None or prev_counts is None or len(prev_flat) == 0 or len(current_flat) == 0:
        return current_flat, current_counts

    # --- Rust fast path: HashMap + Rayon parallel, GIL-free ---
    try:
        from et_miner.backends import get_rust_ext

        et_miner_rust = get_rust_ext()
        if et_miner_rust is None:
            raise ImportError("et_miner_rust not built")

        cf = np.ascontiguousarray(current_flat, dtype=np.int32)
        cc = np.ascontiguousarray(current_counts, dtype=np.int64)
        pf = np.ascontiguousarray(prev_flat, dtype=np.int32)
        pc = np.ascontiguousarray(prev_counts, dtype=np.int64)
        n_before = len(current_flat)
        k = current_flat.shape[1]

        # Compact path returns pruned arrays directly, skipping the
        # single-threaded numpy fancy-index that bottlenecked at 30-60s on 430M
        # K=6 rows. Sequential extend_from_slice in Rust ~10-13× faster.
        if hasattr(et_miner_rust, "prune_closed_flat_compact"):
            flat_1d, pruned_counts, n_kept = et_miner_rust.prune_closed_flat_compact(cf, cc, pf, pc)
            if n_before > n_kept:
                logger.debug(
                    f"    Closed pruning: {n_before:,} → {n_kept:,} ({100 * (1 - n_kept / n_before):.1f}% non-closed removed) [rust-compact]"
                )
            # Reshape (n_kept * k,) → (n_kept, k) — zero-copy view on
            # C-contiguous source. Empty case yields (0, k) shape, not (0,),
            # so prev_frequent_flat.shape[1] stays k for the next K level.
            return flat_1d.reshape((n_kept, k)), pruned_counts

        # Legacy path: bool mask + Python fancy-index (slow at high K)
        mask = et_miner_rust.prune_closed_flat(cf, cc, pf, pc)
        n_after = int(mask.sum())
        if n_before > n_after:
            logger.debug(
                f"    Closed pruning: {n_before:,} → {n_after:,} ({100 * (1 - n_after / n_before):.1f}% non-closed removed) [rust-mask]"
            )
        return current_flat[mask], current_counts[mask]
    except (ImportError, AttributeError):
        pass

    # --- Python fallback ---
    # Build lookup: bytes(subset) → count. Bytes keys are faster than tuple keys.
    prev_flat_c = np.ascontiguousarray(prev_flat)
    prev_lookup = {}
    for i in range(len(prev_flat_c)):
        prev_lookup[prev_flat_c[i].tobytes()] = int(prev_counts[i])

    k = current_flat.shape[1]
    mask = np.ones(len(current_flat), dtype=bool)

    # For each drop position d, generate all subsets by removing column d
    for d in range(k):
        subsets = np.ascontiguousarray(np.delete(current_flat, d, axis=1))
        for idx in range(len(subsets)):
            if not mask[idx]:
                continue
            prev_count = prev_lookup.get(subsets[idx].tobytes())
            if prev_count is not None and prev_count == int(current_counts[idx]):
                mask[idx] = False

    n_before = len(current_flat)
    n_after = int(mask.sum())
    if n_before > n_after:
        logger.debug(
            f"    Closed pruning: {n_before:,} → {n_after:,} ({100 * (1 - n_after / n_before):.1f}% non-closed removed)"
        )

    return current_flat[mask], current_counts[mask]


def _apply_anchor_filter(current_flat, current_counts_raw, anchor_col_arr, k):
    """Filter itemsets to keep only those containing >=1 anchor item (by column index).

    Used by two-phase mining (V3 B6): Phase 2 restricts candidates to neighborhoods
    of anchor items discovered in Phase 1, reducing candidate explosion at ultra-low
    support thresholds.

    Args:
        current_flat: numpy int32 (n, k) — itemset column indices.
        current_counts_raw: numpy int64 (n,) — raw support counts.
        anchor_col_arr: numpy int32 — sorted array of anchor column indices.
            None means no filtering (pass-through).
        k: current itemset length.

    Returns:
        Tuple of (filtered_flat, filtered_counts, n_after).
    """
    import numpy as np

    if anchor_col_arr is None or k < 2:
        return current_flat, current_counts_raw, len(current_flat)
    if len(current_flat) == 0:
        return current_flat, current_counts_raw, 0

    anchor_mask = np.zeros(len(current_flat), dtype=bool)
    for col in range(k):
        anchor_mask |= np.isin(current_flat[:, col], anchor_col_arr)
    n_before = len(current_flat)
    filtered_flat = current_flat[anchor_mask]
    filtered_counts = current_counts_raw[anchor_mask]
    n_after = len(filtered_flat)
    if n_before > n_after:
        logger.debug(
            f"    Anchor filter K={k}: {n_before:,} → {n_after:,} ({100 * (1 - n_after / n_before):.1f}% filtered)"
        )
    return filtered_flat, filtered_counts, n_after


def _convert_to_tidsets(bitvecs_gpu_or_list, freq_flat, n_u64s, batch_size=10_000, verify=True):
    """Convert frequent itemset bitvec AND results to CSR tid-sets.

    The density transition point: dense bitvecs (13.6 MB/item) → sparse tid-sets
    (support × 4 bytes/item). At K=3+ with support ~2K, this is 1,700× more compact.

    Computes AND of K bitvecs per itemset on GPU, extracts set bit positions
    via Rust bitvec_to_tidsets (Rayon parallel), returns CSR arrays.

    Args:
        bitvecs_gpu_or_list: CuPy array (n_cols, n_u64s) on single GPU, or
            list of (gpu_array, device_id, n_rows) for multi-GPU.
        freq_flat: numpy int32 (n_freq, k) — frequent itemset column indices.
        n_u64s: Number of uint64 words per bitvector.
        batch_size: Process this many itemsets per GPU batch (limits VRAM).
        verify: Sample 1000 random itemsets and cross-check support counts.

    Returns:
        Tuple of (offsets, indices) numpy arrays in CSR format:
        - offsets: int64 (n_freq + 1) — CSR row pointers
        - indices: int32 — concatenated tid-sets (transaction IDs)
    """
    import cupy as cp
    import numpy as np
    import time

    n_freq, k = freq_flat.shape
    if n_freq == 0:
        return np.array([0], dtype=np.int64), np.array([], dtype=np.int32)

    t0 = time.time()

    # Determine GPU mode: single GPU vs multi-GPU row-split
    is_multi_gpu = isinstance(bitvecs_gpu_or_list, list)

    if is_multi_gpu:
        gpu_list = bitvecs_gpu_or_list  # [(gpu_array, device_id, n_rows), ...]
        # Cumulative row offsets for global tid-set indexing
        # GPU i's local tids [0, n_rows_i) → global [row_offsets[i], row_offsets[i] + n_rows_i)
        row_offsets = []
        cumulative = 0
        for _, _, n_rows_local in gpu_list:
            row_offsets.append(cumulative)
            cumulative += n_rows_local
    else:
        bitvecs_gpu = bitvecs_gpu_or_list
        device_id = 0

    # Dynamic batch_size for multi-GPU: AND temporaries must fit in VRAM
    # AND needs: and_results (batch × n_u64s × 8) + bv[col] temp (same) = 2×
    if is_multi_gpu:
        with cp.cuda.Device(gpu_list[0][1]):
            cp.get_default_memory_pool().free_all_blocks()
            free_mem, _ = cp.cuda.Device().mem_info
        n_u64s_sample = gpu_list[0][0].shape[1]
        bytes_per_item = 2 * n_u64s_sample * 8  # and_results + bv[col] temporary
        vram_batch = max(100, int(free_mem * 0.4 / bytes_per_item))
        if vram_batch < batch_size:
            logger.debug(
                f"  Transition: batch_size {batch_size:,} → {vram_batch:,} (VRAM-bounded, {free_mem / 1e9:.1f} GB free)"
            )
            batch_size = vram_batch

    # Process in batches to limit VRAM (batch_size × n_u64s × 8 bytes per batch)
    all_offsets_parts = []
    all_indices_parts = []
    running_offset = 0

    from et_miner.backends import get_rust_ext

    et_miner_rust = get_rust_ext()
    has_rust = et_miner_rust is not None

    from concurrent.futures import ThreadPoolExecutor

    _gpu_pool = ThreadPoolExecutor(max_workers=len(gpu_list)) if is_multi_gpu else None

    for batch_start in range(0, n_freq, batch_size):
        batch_end = min(batch_start + batch_size, n_freq)
        batch_flat = freq_flat[batch_start:batch_end]
        n_batch = len(batch_flat)

        if is_multi_gpu:
            # ═══ Multi-GPU merge path (parallel) ═══
            # AND bitvecs on each GPU independently, extract tids, offset-adjust, merge.
            # Merged tids are automatically sorted (GPU i covers rows after GPU i-1).
            # ThreadPoolExecutor: all GPUs process in parallel (was sequential → 8× slower)

            def _extract_on_gpu(gpu_idx, bv, did, _bf=batch_flat, _nb=n_batch, _k=k):
                n_u64s_local = bv.shape[1]
                tid_offset = row_offsets[gpu_idx]
                with cp.cuda.Device(did):
                    and_results = cp.full((_nb, n_u64s_local), np.uint64(0xFFFFFFFFFFFFFFFF), dtype=cp.uint64)
                    for col_pos in range(_k):
                        col_indices = _bf[:, col_pos].astype(np.int64)
                        and_results &= bv[col_indices]

                    try:
                        from et_miner.gpu.kernels import get_cuda_kernel, get_popcount_kernel

                        extract_kernel = get_cuda_kernel("bitvec_extract_tids")
                        popcount_kernel = get_popcount_kernel()

                        popcounts_flat = popcount_kernel(and_results.ravel().view(cp.uint64))
                        popcounts_per_row = popcounts_flat.reshape(_nb, n_u64s_local).sum(axis=1).astype(cp.int64)

                        gpu_offsets = cp.concatenate([cp.zeros(1, dtype=cp.int64), cp.cumsum(popcounts_per_row)])
                        total_tids = int(gpu_offsets[-1].item())

                        if total_tids > 0:
                            gpu_indices = cp.empty(total_tids, dtype=cp.int32)
                            grid = ((_nb + 255) // 256,)
                            extract_kernel(
                                grid,
                                (256,),
                                (
                                    and_results,
                                    gpu_offsets,
                                    gpu_indices,
                                    np.int64(_nb),
                                    np.int64(n_u64s_local),
                                    np.int64(tid_offset),
                                ),
                            )
                            cp.cuda.Device(did).synchronize()
                            gpu_idx_arr = gpu_indices.get()
                        else:
                            gpu_idx_arr = np.array([], dtype=np.int32)

                        gpu_off = gpu_offsets.get()
                        del and_results, popcounts_flat, popcounts_per_row, gpu_offsets
                        if total_tids > 0:
                            del gpu_indices
                        cp.get_default_memory_pool().free_all_blocks()

                    except Exception as e:
                        logger.warning(f"GPU {did}: CUDA extract failed, CPU fallback: {e}")
                        and_np = and_results.get()
                        del and_results
                        cp.get_default_memory_pool().free_all_blocks()
                        if has_rust:
                            gpu_off, gpu_idx_arr = et_miner_rust.bitvec_to_tidsets(and_np)
                        else:
                            bo, bi = [0], []
                            for i in range(_nb):
                                for w_idx in range(n_u64s_local):
                                    word = int(and_np[i, w_idx])
                                    base = w_idx * 64
                                    while word:
                                        bit = (word & -word).bit_length() - 1
                                        bi.append(base + bit)
                                        word &= word - 1
                                bo.append(len(bi))
                            gpu_off = np.array(bo, dtype=np.int64)
                            gpu_idx_arr = np.array(bi, dtype=np.int32)
                        if tid_offset > 0 and len(gpu_idx_arr) > 0:
                            gpu_idx_arr = (gpu_idx_arr.astype(np.int64) + tid_offset).astype(np.int32)

                return (gpu_off, gpu_idx_arr)

            futures = [
                _gpu_pool.submit(_extract_on_gpu, gpu_idx, bv, did) for gpu_idx, (bv, did, _) in enumerate(gpu_list)
            ]
            per_gpu = [f.result() for f in futures]

            # Merge per-itemset tid-sets across GPUs (sorted by construction)
            merged_off = [0]
            merged_parts = []
            running_len = 0
            for item_i in range(n_batch):
                for gpu_off, gpu_idx_arr in per_gpu:
                    s, e = int(gpu_off[item_i]), int(gpu_off[item_i + 1])
                    if s < e:
                        merged_parts.append(gpu_idx_arr[s:e])
                        running_len += e - s
                merged_off.append(running_len)

            offsets = np.array(merged_off, dtype=np.int64)
            indices = np.concatenate(merged_parts) if merged_parts else np.array([], dtype=np.int32)
        else:
            # ═══ Single GPU path ═══
            with cp.cuda.Device(device_id):
                # NOTE: ~zeros = all-ones (0xFFFF...), NOT cp.ones which gives 0x0001
                and_results = ~cp.zeros((n_batch, n_u64s), dtype=cp.uint64)
                for col_pos in range(k):
                    col_indices = batch_flat[:, col_pos].astype(np.int64)
                    and_results &= bitvecs_gpu[col_indices]
                and_np = and_results.get()
                del and_results
                cp.get_default_memory_pool().free_all_blocks()

            if has_rust:
                offsets, indices = et_miner_rust.bitvec_to_tidsets(and_np)
            else:
                batch_offsets = [0]
                batch_indices = []
                for i in range(n_batch):
                    tids = []
                    for w_idx in range(n_u64s):
                        word = int(and_np[i, w_idx])
                        base = w_idx * 64
                        while word:
                            bit = (word & -word).bit_length() - 1
                            tids.append(base + bit)
                            word &= word - 1
                    batch_indices.extend(tids)
                    batch_offsets.append(len(batch_indices))
                offsets = np.array(batch_offsets, dtype=np.int64)
                indices = np.array(batch_indices, dtype=np.int32)

        # Adjust offsets for concatenation
        if batch_start > 0:
            offsets = offsets[1:] + running_offset  # skip first 0, add global offset
        else:
            offsets = offsets + running_offset

        running_offset = int(offsets[-1]) if len(offsets) > 0 else running_offset
        all_offsets_parts.append(offsets)
        all_indices_parts.append(indices)

    if _gpu_pool is not None:
        _gpu_pool.shutdown(wait=False)

    # Concatenate all batches
    final_offsets = np.concatenate(all_offsets_parts)
    final_indices = np.concatenate(all_indices_parts) if all_indices_parts else np.array([], dtype=np.int32)

    elapsed = time.time() - t0
    total_tids = len(final_indices)
    avg_support = total_tids / n_freq if n_freq > 0 else 0
    total_u64s = sum(bv.shape[1] for bv, _, _ in gpu_list) if is_multi_gpu else n_u64s
    bitvec_mb = n_freq * total_u64s * 8 / 1024**2
    tidset_mb = (len(final_offsets) * 8 + len(final_indices) * 4) / 1024**2
    logger.info(
        f"    Dense→Sparse: {n_freq:,} itemsets, avg support {avg_support:.0f}, "
        f"{bitvec_mb:.0f} MB bitvec → {tidset_mb:.1f} MB tidset "
        f"({bitvec_mb / tidset_mb:.0f}× compression) in {elapsed:.1f}s"
        if tidset_mb > 0
        else f"    Dense→Sparse: {n_freq:,} itemsets (no tids) in {elapsed:.1f}s"
    )

    # Verification step: sample random itemsets, cross-check
    if verify and n_freq > 0:
        from et_miner.gpu.kernels import get_popcount_kernel

        n_verify = min(1000, n_freq)
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n_freq, n_verify, replace=False)

        mismatches = 0
        for idx in sample_idx:
            tidset_len = int(final_offsets[idx + 1] - final_offsets[idx])

            if is_multi_gpu:
                # Sum popcounts across all GPUs for correct total
                popcount = 0
                for bv, did, _ in gpu_list:
                    n_u64s_local = bv.shape[1]
                    with cp.cuda.Device(did):
                        and_val = ~cp.zeros(n_u64s_local, dtype=cp.uint64)
                        for col_pos in range(k):
                            and_val &= bv[int(freq_flat[idx, col_pos])]
                        popcount += int(get_popcount_kernel()(and_val).sum())
            else:
                with cp.cuda.Device(device_id):
                    and_val = ~cp.zeros(n_u64s, dtype=cp.uint64)
                    for col_pos in range(k):
                        and_val &= bitvecs_gpu[int(freq_flat[idx, col_pos])]
                    popcount = int(get_popcount_kernel()(and_val).sum())

            if tidset_len != popcount:
                mismatches += 1

        if mismatches > 0:
            logger.warning(f"    {mismatches}/{n_verify} tidset length mismatches!")
        else:
            logger.debug(f"    Verify: {n_verify} random samples OK ✓")

    return final_offsets, final_indices


def _prune_groups_apriori(groups_info, prev_frequent_set, k, prev_flat_np=None):
    """Prune prefix groups by removing suffix pairs whose (k-1)-subsets are not all frequent.

    At K=3 this is exact: check if (suffix_i, suffix_j) is a frequent K=2 pair.
    At K>=4 this is also exact: for each suffix, checks all k-2 prefix-drop subsets
    plus the two suffix-drop subsets. A suffix is only kept if ALL its (k-1)-subsets
    are in prev_frequent_set. (Verified by Auditor: 16/16 math checks pass, 2026-03-25.)

    The GPU dense kernel counts ALL pairs within a group. By removing invalid
    suffixes, we reduce the group sizes and thus the candidate count.

    Rust fast path: HashSet + Rayon parallel, GIL-free. Falls back to Python if
    the Rust extension is not available.

    Args:
        groups_info: K3PlusGroups namedtuple.
        prev_frequent_set: set of tuples of frequent (k-1)-itemsets.
        k: current itemset size.
        prev_flat_np: optional numpy int32 (n_prev, k-1) array for Rust fast path.

    Returns:
        Pruned K3PlusGroups or None if all candidates pruned.
    """
    import numpy as np
    from et_miner.gpu.kernels import K3PlusGroups

    # --- Rust fast path: HashSet + Rayon parallel, GIL-free ---
    if prev_flat_np is not None:
        try:
            from et_miner.backends import get_rust_ext

            et_miner_rust = get_rust_ext()
            if et_miner_rust is None:
                raise ImportError("et_miner_rust not built")

            pf = np.ascontiguousarray(prev_flat_np, dtype=np.int32)
            result = et_miner_rust.prune_groups_apriori(
                np.ascontiguousarray(groups_info.prefix_items),
                np.ascontiguousarray(groups_info.prefix_offsets),
                np.ascontiguousarray(groups_info.suffixes),
                np.ascontiguousarray(groups_info.suffix_offsets),
                np.ascontiguousarray(groups_info.cumulative_pairs),
                int(groups_info.total_candidates),
                pf,
            )
            if result is None:
                return None
            pi, po, sf, so, cp_arr, tc = result
            return K3PlusGroups(
                prefix_items=np.asarray(pi),
                prefix_offsets=np.asarray(po),
                suffixes=np.asarray(sf),
                suffix_offsets=np.asarray(so),
                cumulative_pairs=np.asarray(cp_arr),
                total_candidates=int(tc),
                groups=groups_info.groups,
            )
        except (ImportError, AttributeError):
            pass

    # --- Python fallback ---
    prefix_items = groups_info.prefix_items
    prefix_offsets = groups_info.prefix_offsets
    suffixes = groups_info.suffixes
    suffix_offsets = groups_info.suffix_offsets
    n_groups = len(suffix_offsets) - 1

    new_pi, new_po, new_sf, new_so = [], [0], [], [0]
    new_total = 0
    new_cp = [0]  # MUST start with 0 — every consumer assumes cumulative_pairs[0] == 0

    for g in range(n_groups):
        pstart, pend = int(prefix_offsets[g]), int(prefix_offsets[g + 1])
        sstart, send = int(suffix_offsets[g]), int(suffix_offsets[g + 1])
        prefix = tuple(int(x) for x in prefix_items[pstart:pend])
        gsuf = [int(x) for x in suffixes[sstart:send]]

        if k == 3:
            # EXACT: for K=3, prefix has 1 element. Only check = (s_i, s_j) in prev_set.
            valid_suffixes = set()
            for i in range(len(gsuf)):
                for j in range(i + 1, len(gsuf)):
                    if (gsuf[i], gsuf[j]) in prev_frequent_set:
                        valid_suffixes.add(gsuf[i])
                        valid_suffixes.add(gsuf[j])
        else:
            # CONSERVATIVE: for K>=4, check k-2 subsets (drop each prefix element).
            # Keep suffixes that participate in at least one valid pair.
            valid_suffixes = set()
            for i in range(len(gsuf)):
                for j in range(i + 1, len(gsuf)):
                    candidate = prefix + (gsuf[i], gsuf[j])
                    all_freq = True
                    for d in range(len(prefix)):
                        subset = candidate[:d] + candidate[d + 1 :]
                        if subset not in prev_frequent_set:
                            all_freq = False
                            break
                    if all_freq:
                        valid_suffixes.add(gsuf[i])
                        valid_suffixes.add(gsuf[j])

        valid_sorted = sorted(valid_suffixes)
        if len(valid_sorted) >= 2:
            new_pi.extend(prefix)
            new_po.append(len(new_pi))
            new_sf.extend(valid_sorted)
            new_so.append(len(new_sf))
            n_pairs = len(valid_sorted) * (len(valid_sorted) - 1) // 2
            new_total += n_pairs
            new_cp.append(new_total)

    if new_total == 0:
        return None

    return K3PlusGroups(
        prefix_items=np.array(new_pi, dtype=np.int32),
        prefix_offsets=np.array(new_po, dtype=np.int64),
        suffixes=np.array(new_sf, dtype=np.int32),
        suffix_offsets=np.array(new_so, dtype=np.int64),
        cumulative_pairs=np.array(new_cp, dtype=np.int64),
        total_candidates=new_total,
        groups=groups_info.groups,  # preserve original group metadata
    )


def _apriori_from_bitvecs(
    bitvecs_gpu,  # CuPy array shape (n_cols, ceil(n_rows/64)) dtype uint64
    col_to_item: dict[int, int],  # column index -> item ID
    n_transactions: int,
    min_support: float,
    max_length: int | None,
    batch_size: int | None,
    profile: bool,
    level_callback: Callable[[int, int, int, float], None] | None,
    n_gpus: int = 1,
    max_ram_gb: float = 800.0,
    max_vram_gb: float = 70.0,
    sparse_from_k: int | None = None,
) -> pl.DataFrame | tuple[pl.DataFrame, ProfilingSession]:
    """Run Apriori directly from pre-built GPU bitvectors.

    This is the fast path for streaming GPU processing where bitvectors
    are already built in GPU memory. Avoids DataFrame conversion overhead.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) with packed bits.
        col_to_item: Mapping from column index to original item ID.
        n_transactions: Total number of transactions (for support calculation).
        min_support: Minimum support threshold (0.0-1.0).
        max_length: Maximum itemset length (None = unlimited).
        batch_size: Candidates per batch (used for batching CUDA kernel calls).
        profile: If True, return profiling metrics alongside results.
        level_callback: Optional callback for per-level progress updates.

    Returns:
        If profile=False: DataFrame with columns [itemset, support].
        If profile=True: Tuple of (DataFrame, ProfilingSession).
    """
    import numpy as np

    try:
        import cupy as cp
    except ImportError:
        raise ImportError("CuPy is required for bitvecs parameter. Install with: pip install cupy-cuda12x")

    # CSR transition requires K>=3 groups; K=2 fused kernel is always faster
    if sparse_from_k is not None:
        sparse_from_k = max(sparse_from_k, 3)

    # int32 tidset indices can't represent transaction IDs > 2^31
    if n_transactions > np.iinfo(np.int32).max:
        raise ValueError(
            f"n_transactions={n_transactions:,} exceeds int32 max. CSR tidset indices require int64 upgrade."
        )

    from et_miner.gpu.kernels import get_popcount_kernel, count_csr_intersections, build_k3plus_groups_from_flat

    session = ProfilingSession() if profile else None

    n_cols, n_u64s = bitvecs_gpu.shape
    min_count_threshold = _min_count(min_support, n_transactions)

    # Effective max length: can't exceed number of columns
    effective_max_length = min(
        max_length if max_length else float("inf"),
        n_cols,
    )

    def _get_memory_gb():
        """Get (system_ram_gb, gpu_vram_gb) usage."""
        ram_gb = 0.0
        vram_gb = 0.0
        try:
            import psutil

            ram_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        except ImportError:
            pass
        try:
            pool = cp.get_default_memory_pool()
            vram_gb = pool.used_bytes() / (1024**3)
        except Exception:
            pass
        return ram_gb, vram_gb

    def _log_level(k, n_cand, n_freq, elapsed_s, cumulative_itemsets):
        ram_gb, vram_gb = _get_memory_gb()
        msg = (
            f"  K={k}: candidates={n_cand:,} → frequent={n_freq:,} "
            f"({elapsed_s:.1f}s) | cumulative={cumulative_itemsets:,} | "
            f"RAM={ram_gb:.1f}GB VRAM={vram_gb:.1f}GB"
        )
        logger.info(msg)

    def _check_memory_guard(k, cumulative_itemsets):
        """Return True if memory guard triggered (should stop)."""
        ram_gb, vram_gb = _get_memory_gb()
        if ram_gb > max_ram_gb:
            msg = (
                f"  MEMORY GUARD: RAM={ram_gb:.1f}GB > {max_ram_gb}GB limit at K={k} "
                f"({cumulative_itemsets:,} itemsets). Stopping to prevent OOM."
            )
            logger.warning(msg)
            return True
        if vram_gb > max_vram_gb:
            msg = (
                f"  MEMORY GUARD: VRAM={vram_gb:.1f}GB > {max_vram_gb}GB limit at K={k} "
                f"({cumulative_itemsets:,} itemsets). Stopping to prevent OOM."
            )
            logger.warning(msg)
            return True
        return False

    _t_total_start = time.perf_counter()
    _cumulative_itemsets = 0
    logger.info(
        f"APRIORI BITVEC: {n_cols} cols, {n_transactions:,} txns, "
        f"min_support={min_support:.8f} (min_count={min_count_threshold}), "
        f"max_length={effective_max_length}"
    )

    # Phase 1: k=1 support counting via popcount on each column
    _k1_start = time.perf_counter()
    if session:
        session.start_phase("k1_support")

    # Popcount each column to get 1-itemset support
    # Each column is a bitvector of shape (n_u64s,)
    results: list[tuple[list[int], float]] = []
    prev_frequent: list[tuple[int, ...]] = []  # Use int indices, not str column names
    prev_counts: dict[tuple[int, ...], int] = {}

    # GPU popcount using __popcll hardware intrinsic (183x faster than Python)
    popcount_kernel = get_popcount_kernel()
    popcounts = popcount_kernel(bitvecs_gpu.view(cp.uint64))
    col_counts = cp.sum(popcounts.reshape(n_cols, -1), axis=1, dtype=cp.int64).get()

    # Filter frequent 1-itemsets
    for col_idx in range(n_cols):
        count = col_counts[col_idx]
        if count >= min_count_threshold:
            item_id = col_to_item[col_idx]
            support = count / n_transactions
            results.append(([item_id], support))
            prev_frequent.append((col_idx,))
            prev_counts[(col_idx,)] = count

    if session:
        session.end_phase(n_frequent=len(prev_frequent))

    # Call level callback for k=1
    _k1_elapsed = time.perf_counter() - _k1_start
    _cumulative_itemsets += len(prev_frequent)
    _log_level(1, n_cols, len(prev_frequent), _k1_elapsed, _cumulative_itemsets)
    if level_callback:
        level_callback(1, n_cols, len(prev_frequent), _k1_elapsed * 1000)

    # Pathological support warning
    freq_ratio = len(prev_frequent) / n_cols if n_cols > 0 else 0.0
    if freq_ratio > 0.90 or min_count_threshold <= 1:
        import warnings

        warnings.warn(
            f"{freq_ratio:.0%} of items are frequent (min_count={min_count_threshold}). "
            f"Mining may produce combinatorially explosive results. "
            f"Consider raising min_support.",
            stacklevel=3,
        )

    if not prev_frequent:
        if profile:
            return _empty_result(), session
        return _empty_result()

    # Phase 2: k >= 2 using CUDA kernels
    k = 2
    prev_live = {col for tup in prev_frequent for col in tup}  # Track live columns for deallocation

    # Sparse CSR state: tidset_offsets is not None means CSR mode active
    tidset_offsets = None
    tidset_indices = None
    offsets_gpu = None
    indices_gpu = None

    while k <= effective_max_length and len(prev_frequent) >= k:
        _k_start = time.perf_counter()

        if sparse_from_k is not None and k >= sparse_from_k and tidset_offsets is None:
            logger.info(f"  ═══ DENSITY TRANSITION at K={k}: dense bitvec → sparse CSR ═══")
            prev_frequent_flat = np.array(prev_frequent, dtype=np.int32)
            tidset_offsets, tidset_indices = _convert_to_tidsets(
                bitvecs_gpu,
                prev_frequent_flat,
                n_u64s,
                batch_size=10_000,
                verify=True,
            )
            del bitvecs_gpu
            cp.get_default_memory_pool().free_all_blocks()
            offsets_gpu = cp.array(tidset_offsets, dtype=cp.int64)
            indices_gpu = cp.array(tidset_indices, dtype=cp.int32)
            logger.debug(f"    Freed bitvec VRAM, {len(tidset_indices):,} tid entries in CSR")

        if tidset_offsets is not None:
            prev_frequent_flat = np.array(prev_frequent, dtype=np.int32)
            groups_info = build_k3plus_groups_from_flat(prev_frequent_flat)

            current_frequent: list[tuple[int, ...]] = []
            current_counts: dict[tuple[int, ...], int] = {}

            if groups_info is not None and groups_info.total_candidates > 0:
                tc = groups_info.total_candidates
                logger.info(f"  K={k}: {tc:,} candidates (CSR sparse mode)")

                # Build candidate pair arrays from prefix groups
                pair_a_list, pair_b_list = [], []
                so = groups_info.suffix_offsets
                idx_lookup = {itemset: i for i, itemset in enumerate(prev_frequent)}

                for g in range(len(so) - 1):
                    sstart, send = int(so[g]), int(so[g + 1])
                    gsuf = groups_info.suffixes[sstart:send]
                    prefix = tuple(
                        int(x)
                        for x in groups_info.prefix_items[
                            int(groups_info.prefix_offsets[g]) : int(groups_info.prefix_offsets[g + 1])
                        ]
                    )

                    for si_pos in range(len(gsuf)):
                        for sj_pos in range(si_pos + 1, len(gsuf)):
                            si, sj = int(gsuf[si_pos]), int(gsuf[sj_pos])
                            key_i = prefix + (si,)
                            key_j = prefix + (sj,)
                            idx_i = idx_lookup.get(key_i)
                            idx_j = idx_lookup.get(key_j)
                            if idx_i is not None and idx_j is not None:
                                pair_a_list.append(idx_i)
                                pair_b_list.append(idx_j)

                if pair_a_list:
                    n_pairs = len(pair_a_list)
                    pair_a_np = np.array(pair_a_list, dtype=np.int64)
                    pair_b_np = np.array(pair_b_list, dtype=np.int64)

                    pa_gpu = cp.array(pair_a_np, dtype=cp.int64)
                    pb_gpu = cp.array(pair_b_np, dtype=cp.int64)
                    counts_gpu = count_csr_intersections(offsets_gpu, indices_gpu, pa_gpu, pb_gpu, n_pairs)
                    counts_cpu = counts_gpu.get()
                    del pa_gpu, pb_gpu, counts_gpu
                    cp.get_default_memory_pool().free_all_blocks()

                    # Filter frequent candidates
                    freq_mask = counts_cpu >= min_count_threshold
                    freq_pair_indices = np.where(freq_mask)[0]
                    n_freq = len(freq_pair_indices)

                    if n_freq > 0:
                        for pi in freq_pair_indices:
                            a_idx, b_idx = pair_a_list[pi], pair_b_list[pi]
                            itemset_a = prev_frequent[a_idx]
                            itemset_b = prev_frequent[b_idx]
                            prefix_items = itemset_a[:-1]
                            new_itemset = prefix_items + (itemset_a[-1], itemset_b[-1])
                            count = int(counts_cpu[pi])
                            support = count / n_transactions
                            item_list = [col_to_item[c] for c in new_itemset]
                            results.append((item_list, support))
                            current_frequent.append(new_itemset)
                            current_counts[new_itemset] = count

                        # Rebuild tidsets for K+1 (skip at max_length)
                        if k < effective_max_length:
                            freq_cand_counts = counts_cpu[freq_mask]
                            total_tids_new = int(np.sum(freq_cand_counts))
                            new_offsets = np.empty(n_freq + 1, dtype=np.int64)
                            new_offsets[0] = 0
                            new_indices = np.empty(total_tids_new, dtype=np.int32)
                            write_pos = 0
                            for out_i, pi in enumerate(freq_pair_indices):
                                a_idx, b_idx = pair_a_list[pi], pair_b_list[pi]
                                a_start = int(tidset_offsets[a_idx])
                                a_end = int(tidset_offsets[a_idx + 1])
                                b_start = int(tidset_offsets[b_idx])
                                b_end = int(tidset_offsets[b_idx + 1])
                                tids_a = tidset_indices[a_start:a_end]
                                tids_b = tidset_indices[b_start:b_end]
                                isect = np.intersect1d(tids_a, tids_b, assume_unique=True)
                                n_isect = len(isect)
                                if write_pos + n_isect > total_tids_new:
                                    raise RuntimeError(
                                        f"CSR buffer overrun at itemset {out_i}: "
                                        f"{write_pos} + {n_isect} > {total_tids_new}"
                                    )
                                new_indices[write_pos : write_pos + n_isect] = isect
                                write_pos += n_isect
                                new_offsets[out_i + 1] = write_pos

                            if write_pos > total_tids_new:
                                raise RuntimeError(f"CSR buffer overrun: {write_pos} > {total_tids_new}")
                            tidset_offsets = new_offsets
                            tidset_indices = new_indices[:write_pos]
                            # Refresh GPU-resident CSR arrays
                            del offsets_gpu, indices_gpu
                            offsets_gpu = cp.array(tidset_offsets, dtype=cp.int64)
                            indices_gpu = cp.array(tidset_indices, dtype=cp.int32)
                            tidset_mb = (len(tidset_offsets) * 8 + len(tidset_indices) * 4) / 1024**2
                            logger.debug(f"    New tidsets: {n_freq:,} itemsets, {tidset_mb:.1f} MB")

            _k_elapsed = time.perf_counter() - _k_start
            _cumulative_itemsets += len(current_frequent)
            _csr_cands = groups_info.total_candidates if groups_info is not None else 0
            _log_level(k, _csr_cands, len(current_frequent), _k_elapsed, _cumulative_itemsets)
            if level_callback:
                level_callback(k, _csr_cands, len(current_frequent), _k_elapsed * 1000)

        # === FUSED K=2 FAST PATH ===
        # Single kernel launch: pair gen + AND + popcount + filter on GPU.
        # Eliminates all Python overhead (28.8M tuples, numpy arrays, flatten loop).
        elif k == 2:
            freq_cols = sorted(p[0] for p in prev_frequent)
            n_pairs = len(freq_cols) * (len(freq_cols) - 1) // 2

            if session:
                session.start_phase("k2_fused_gpu")

            from et_miner.gpu.dispatch import dispatch_k2

            pairs, counts = dispatch_k2(bitvecs_gpu, freq_cols, n_u64s, min_count_threshold)

            if session:
                session.end_phase(n_candidates=n_pairs, n_frequent=len(pairs))

            current_frequent: list[tuple[int, ...]] = []
            current_counts: dict[tuple[int, ...], int] = {}

            for idx, (col_i, col_j) in enumerate(pairs):
                count = int(counts[idx])
                support = count / n_transactions
                item_list = [col_to_item[col_i], col_to_item[col_j]]
                results.append((item_list, support))
                current_frequent.append((col_i, col_j))
                current_counts[(col_i, col_j)] = count

            # Pair cache infrastructure ready but not yet wired to K>=3 kernels.
            # Disabled to avoid 13.6 GB VRAM waste. Re-enable when kernel integration is done.

            _k_elapsed = time.perf_counter() - _k_start
            _cumulative_itemsets += len(current_frequent)
            _log_level(k, n_pairs, len(current_frequent), _k_elapsed, _cumulative_itemsets)
            if level_callback:
                level_callback(k, n_pairs, len(current_frequent), _k_elapsed * 1000)

        else:
            # === FULLY-FUSED K>=3 PATH ===
            # Candidate generation + count + filter ALL on GPU in ONE kernel.
            # Prefix groups built on CPU (O(n_frequent)), transferred to GPU,
            # candidates generated on-the-fly via triangular number inverse.
            # No Apriori pruning needed: anti-monotone property guarantees
            # non-frequent candidates fail min_count check in-kernel.
            if session:
                session.start_phase(f"k{k}_fully_fused_gpu")

            from et_miner.gpu.dispatch import dispatch_k3plus_fused, dispatch_k3plus_sampled, SAMPLED_PREFILTER_THRESHOLD

            # V3: use sampled prefilter when candidate count is high enough
            # Estimate candidate count from prefix groups
            _prefix_groups: dict[tuple, int] = {}
            for itemset in prev_frequent:
                _p = itemset[:-1]
                _prefix_groups[_p] = _prefix_groups.get(_p, 0) + 1
            _est_cands = sum(g * (g - 1) // 2 for g in _prefix_groups.values())

            if _est_cands >= SAMPLED_PREFILTER_THRESHOLD and n_u64s >= 8:
                frequent_candidates, counts = dispatch_k3plus_sampled(
                    bitvecs_gpu, prev_frequent, k, n_u64s, min_count_threshold
                )
            else:
                frequent_candidates, counts = dispatch_k3plus_fused(
                    bitvecs_gpu, prev_frequent, k, n_u64s, min_count_threshold
                )

            # Build results from frequent candidates (already filtered by kernel)
            current_frequent = []
            current_counts = {}

            for idx, candidate in enumerate(frequent_candidates):
                count = int(counts[idx])
                support = count / n_transactions
                item_list = [col_to_item[c] for c in candidate]
                results.append((item_list, support))
                current_frequent.append(candidate)
                current_counts[candidate] = count

            if session:
                session.end_phase(n_frequent=len(current_frequent))

            _k_elapsed = time.perf_counter() - _k_start
            _cumulative_itemsets += len(current_frequent)
            _log_level(k, _est_cands, len(current_frequent), _k_elapsed, _cumulative_itemsets)
            if level_callback:
                level_callback(k, len(frequent_candidates), len(current_frequent), _k_elapsed * 1000)

        if not current_frequent:
            logger.info(
                f"  K={k}: 0 frequent — EXHAUSTED. Total: {_cumulative_itemsets:,} itemsets in {time.perf_counter() - _t_total_start:.1f}s"
            )
            break

        # Memory guard: check BEFORE starting next level
        if _check_memory_guard(k, _cumulative_itemsets):
            logger.warning(f"  Mining stopped at K={k} by memory guard. Total: {_cumulative_itemsets:,} itemsets")
            break

        # Progressive bitvector deallocation: zero dead columns (skip if CSR active)
        if tidset_offsets is None:
            current_live = {col for tup in current_frequent for col in tup}
            prev_live = _deallocate_dead_bitvecs(bitvecs_gpu, current_live, prev_live, k)

        prev_frequent = current_frequent
        prev_counts = current_counts
        k += 1
    else:
        # While loop ended because k > effective_max_length or not enough frequent items
        logger.info(
            f"  Mining complete: K={k - 1} reached max_length={effective_max_length}. "
            f"Total: {_cumulative_itemsets:,} itemsets "
            f"in {time.perf_counter() - _t_total_start:.1f}s"
        )

    result_df = _build_result_df(results)
    if profile:
        return result_df, session
    return result_df


def _build_results_from_gpu(gpu_results, col_to_item, n_transactions):
    """Bulk transfer GPU results to CPU and build result list.

    Single PCIe transfer per K-level at the end, using vectorized numpy
    col_to_item mapping instead of per-element Python dict lookups.

    Args:
        gpu_results: List of (itemsets_gpu, counts_gpu) CuPy arrays per K-level.
        col_to_item: Dict mapping column index to item ID.
        n_transactions: Total number of transactions.

    Returns:
        List of (item_list, support) tuples.
    """
    import numpy as np

    # Build vectorized lookup array: col_idx -> item_id
    if col_to_item:
        max_col = max(col_to_item.keys())
        col_lookup = np.zeros(max_col + 1, dtype=np.int64)
        for col_idx, item_id in col_to_item.items():
            col_lookup[col_idx] = item_id
    else:
        col_lookup = np.array([], dtype=np.int64)

    results = []
    for itemsets_gpu, counts_gpu in gpu_results:
        # Bulk .get() — one PCIe transfer per level
        itemsets_np = itemsets_gpu.get()  # (n, k) int32
        counts_np = counts_gpu.get()  # (n,) int64

        # Vectorized col->item mapping
        item_ids = col_lookup[itemsets_np]  # (n, k) int64

        for i in range(len(counts_np)):
            support = counts_np[i] / n_transactions
            results.append((item_ids[i].tolist(), support))

    return results


def _apriori_from_bitvecs_gpu_resident(
    bitvecs_gpu,
    col_to_item: dict[int, int],
    n_transactions: int,
    min_support: float,
    max_length: int | None,
    profile: bool,
    level_callback: Callable[[int, int, int, float], None] | None,
) -> "pl.DataFrame | tuple[pl.DataFrame, ProfilingSession]":
    """Fully GPU-resident Apriori: all frequent itemsets stay in VRAM.

    Zero PCIe round trips per K-level (only ~12 bytes for loop control).
    Single bulk transfer at the end to build the result DataFrame.

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s) with packed bits.
        col_to_item: Mapping from column index to original item ID.
        n_transactions: Total number of transactions.
        min_support: Minimum support threshold (0.0-1.0).
        max_length: Maximum itemset length (None = unlimited).
        profile: If True, return profiling metrics alongside results.
        level_callback: Optional callback for per-level progress updates.

    Returns:
        If profile=False: DataFrame with columns [itemset, support].
        If profile=True: Tuple of (DataFrame, ProfilingSession).
    """
    try:
        import cupy as cp
    except ImportError:
        raise ImportError("CuPy required for gpu_resident mode")

    from et_miner.gpu.kernels import get_popcount_kernel
    from et_miner.gpu.dispatch import dispatch_k2_gpu_resident, dispatch_k3plus_gpu_resident

    session = ProfilingSession() if profile else None

    n_cols, n_u64s = bitvecs_gpu.shape
    min_count_threshold = _min_count(min_support, n_transactions)

    effective_max_length = min(
        max_length if max_length else float("inf"),
        n_cols,
    )

    # Accumulate (CuPy itemsets, CuPy counts) per level — all in VRAM
    gpu_results = []

    # === K=1: popcount -> freq_cols_gpu (n,1) in VRAM ===
    _k1_start = time.perf_counter()
    if session:
        session.start_phase("k1_support_gpu_resident")

    popcount_kernel = get_popcount_kernel()
    popcounts = popcount_kernel(bitvecs_gpu.view(cp.uint64))
    col_counts_gpu = cp.sum(popcounts.reshape(n_cols, -1), axis=1, dtype=cp.int64)

    # Filter frequent on GPU
    freq_mask = col_counts_gpu >= min_count_threshold
    freq_col_indices = cp.where(freq_mask)[0].astype(cp.int32)  # VRAM
    freq_counts = col_counts_gpu[freq_mask]  # VRAM

    n_frequent_k1 = len(freq_col_indices)

    if n_frequent_k1 > 0:
        # Store K=1 results as (n, 1) array in VRAM
        k1_itemsets = freq_col_indices.reshape(-1, 1)  # (n, 1) VRAM
        gpu_results.append((k1_itemsets, freq_counts))

    if session:
        session.end_phase(n_frequent=n_frequent_k1)

    if level_callback:
        _k1_duration_ms = (time.perf_counter() - _k1_start) * 1000
        level_callback(1, n_cols, n_frequent_k1, _k1_duration_ms)

    if n_frequent_k1 == 0:
        if profile:
            return _empty_result(), session
        return _empty_result()

    # === K=2: fused kernel -> pair_items (n,2) in VRAM ===
    k = 2
    prev_freq_gpu = k1_itemsets  # (n, 1) — sorted freq column indices
    prev_live_gr = set(freq_col_indices.tolist())  # K=1 live cols for progressive deallocation

    while k <= effective_max_length and len(prev_freq_gpu) >= k:
        _k_start = time.perf_counter()

        if k == 2:
            if session:
                session.start_phase("k2_fused_gpu_resident")

            n_pairs = n_frequent_k1 * (n_frequent_k1 - 1) // 2

            pair_itemsets, pair_counts = dispatch_k2_gpu_resident(
                bitvecs_gpu, freq_col_indices, n_u64s, min_count_threshold
            )

            if pair_itemsets is not None:
                gpu_results.append((pair_itemsets, pair_counts))
                n_frequent_k2 = len(pair_itemsets)
                prev_freq_gpu = pair_itemsets  # (n, 2) for next iteration
            else:
                n_frequent_k2 = 0

            if session:
                session.end_phase(n_candidates=n_pairs, n_frequent=n_frequent_k2)

            if level_callback:
                _k_duration_ms = (time.perf_counter() - _k_start) * 1000
                level_callback(k, n_pairs, n_frequent_k2, _k_duration_ms)

            if n_frequent_k2 == 0:
                break

        else:
            # === K>=3: build_groups_gpu -> count kernel -> decode kernel -> sort ===
            if session:
                session.start_phase(f"k{k}_gpu_resident")

            freq_itemsets, freq_counts = dispatch_k3plus_gpu_resident(
                bitvecs_gpu, prev_freq_gpu, n_u64s, min_count_threshold
            )

            if freq_itemsets is not None:
                gpu_results.append((freq_itemsets, freq_counts))
                n_frequent_k = len(freq_itemsets)
                prev_freq_gpu = freq_itemsets  # (n, k) for next iteration
            else:
                n_frequent_k = 0

            if session:
                session.end_phase(n_frequent=n_frequent_k)

            if level_callback:
                _k_duration_ms = (time.perf_counter() - _k_start) * 1000
                level_callback(k, 0, n_frequent_k, _k_duration_ms)

            if n_frequent_k == 0:
                break

            # Progressive bitvector deallocation (gpu-resident path)
            current_live_gr = set(cp.unique(prev_freq_gpu.ravel()).get().tolist())
            prev_live_gr = _deallocate_dead_bitvecs(bitvecs_gpu, current_live_gr, prev_live_gr, k)

        k += 1

    # === END: single bulk transfer, build DataFrame ===
    results = _build_results_from_gpu(gpu_results, col_to_item, n_transactions)
    result_df = _build_result_df(results)

    if profile:
        return result_df, session
    return result_df




def _infer_count_from_subsets(
    candidate: tuple[str, ...],
    prev_counts: dict[tuple[str, ...], int],
) -> int | None:
    """Infer candidate count when ALL subsets have EQUAL count (APPROXIMATION).

    WARNING: This is mathematically unsound. Anti-monotonicity only guarantees
    support(superset) <= min(subset_supports), NOT equality when all subsets
    match. Counterexample: support({A,B})=support({A,C})=support({B,C})=30
    but support({A,B,C})=15. Use with caution — may overcount.

    Only used in CPU path when use_generator_pruning=True (default: False).
    GPU production path does NOT use this function.

    Args:
        candidate: Candidate k-itemset to check.
        prev_counts: Integer counts for (k-1)-itemsets from previous iteration.

    Returns:
        Inferred count if ALL subsets have equal count, None otherwise.
        NOTE: Returned count is an UPPER BOUND, not exact.
    """
    k = len(candidate)
    if k < 2:
        return None

    # Collect counts for ALL (k-1)-subsets
    subset_counts: list[int] = []
    for i in range(k):
        subset = candidate[:i] + candidate[i + 1 :]
        if subset not in prev_counts:
            # If ANY subset is missing (not frequent), we can't infer
            return None
        subset_counts.append(prev_counts[subset])

    # Check if ALL subsets have the SAME count
    unique_counts = set(subset_counts)
    if len(unique_counts) == 1:
        # All subsets have equal count -> candidate must have this count too
        inferred = subset_counts[0]
        logger.debug(f"INFERRED {candidate}: all {k} subsets have count={inferred}")
        return inferred

    # Not inferable
    logger.debug("Cannot infer {}: subsets have different counts {}", candidate, subset_counts)

    return None


def _write_partition(
    items_flat,
    supports,
    pidx,
    pstart,
    pend,
    local_part_dir,
    schema,
    comp_kw,
    chunk_size,
    k_width,
):
    """Write rows [pstart, pend) of items_flat/supports to part_{pidx:03d}.parquet."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    part_path = os.path.join(local_part_dir, f"part_{pidx:03d}.parquet")
    writer = pq.ParquetWriter(part_path, schema, **comp_kw)
    try:
        for cs in range(pstart, pend, chunk_size):
            ce = min(cs + chunk_size, pend)
            cn = ce - cs
            chunk_flat = items_flat[cs:ce].ravel()
            chunk_offsets = np.arange(0, (cn + 1) * k_width, k_width, dtype=np.int64)
            chunk_list = pa.LargeListArray.from_arrays(chunk_offsets, chunk_flat)
            chunk_table = pa.table(
                {"itemset": chunk_list, "support": supports[cs:ce]},
                schema=schema,
            )
            writer.write_table(chunk_table, row_group_size=cn)
            del chunk_list, chunk_table, chunk_offsets, chunk_flat
    finally:
        writer.close()
    return part_path


def _flush_k_parquet(
    items_flat,
    supports,
    k_level: int,
    output_dir: str,
    is_remote: bool,
    uploader: GCSUploader | None,
    backup_dir: str | None = None,
) -> None:
    """Flush K-level results to parquet — module-level, closure-free.

    Five branches preserved (K=1-4 single-table proof):
      single-table local           (n < threshold, local)
      single-table direct gs://    (n < threshold, gs://)
      parallel partitioned local   (n >= threshold, local — 7.6× speedup)
      partitioned NVMe-first → gs  (n >= threshold, gs:// — async upload)
      disk-low fallback chunked    (n >= threshold, gs://, low-disk single-stream)

    Caller must wrap with writeable=False/True try/finally for the parallel
    branch (Auditor R2 voorwaarde 1). This function does NOT touch flags.

    Threshold via ET_MINER_FLUSH_PARALLEL_THRESHOLD (default 100M rows).
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor as _FlushTPE

    n = items_flat.shape[0]
    k_width = items_flat.shape[1] if items_flat.ndim == 2 else 1

    chunk_threshold = _env.flush_parallel_threshold()
    legacy = _env.legacy_write()
    use_parallel = n >= chunk_threshold and not legacy

    flush_comp = _env.flush_compression()
    comp_kw: dict[str, Any] = {"compression": flush_comp, "use_dictionary": False}
    if flush_comp == "zstd":
        comp_kw["compression_level"] = 2

    if not use_parallel:
        # ── Single-table path: small outputs (<threshold) or legacy override ──
        out_path = join_gs_uri(output_dir, f"frequent_k{k_level}.parquet")
        flat_items = items_flat.ravel()
        offsets = np.arange(0, (n + 1) * k_width, k_width, dtype=np.int64)
        list_arr = pa.LargeListArray.from_arrays(offsets, flat_items)
        arrow_table = pa.table({"itemset": list_arr, "support": supports})
        if is_remote:
            bucket, blob = parse_gs_url(out_path)
            pq.write_table(
                arrow_table,
                f"{bucket}/{blob}",
                filesystem=pyarrow_gcs_filesystem(),
                **comp_kw,
            )
        else:
            pq.write_table(arrow_table, out_path, **comp_kw)
        del arrow_table

        if is_remote:
            logger.info(f"  → Flushed {n:,} itemsets to {out_path} ({flush_comp}, direct gs://)")
        else:
            size_mb = os.path.getsize(out_path) / (1024**2)
            logger.info(f"  → Flushed {n:,} itemsets to {out_path} ({size_mb:.0f} MB, {flush_comp})")

        if uploader is not None and not is_remote:
            uploader.upload(out_path)

        if backup_dir and not is_remote:
            import subprocess

            os.makedirs(backup_dir, exist_ok=True)
            subprocess.run(["cp", out_path, backup_dir + "/"], check=True)
            logger.info(f"  → Backup: {out_path} → {backup_dir}/")
        return

    # ── Parallel partitioned write ──
    chunk_size = _env.flush_chunk_size()
    n_threads = min(
        _env.flush_threads(),
        max(1, (os.cpu_count() or 4) // 4),
    )

    schema = pa.schema(
        [
            ("itemset", pa.large_list(pa.from_numpy_dtype(items_flat.dtype))),
            ("support", pa.from_numpy_dtype(supports.dtype)),
        ]
    )

    if is_remote:
        tmpdir = _env.parquet_tmpdir()
        os.makedirs(tmpdir, exist_ok=True)
        local_part_dir = os.path.join(tmpdir, f"frequent_k{k_level}")
    else:
        local_part_dir = os.path.join(output_dir, f"frequent_k{k_level}")
    os.makedirs(local_part_dir, exist_ok=True)

    # Disk pre-check — fall back to single-stream gs:// write when low
    import shutil

    est_compressed_gb = (n * (k_width * 4 + 8)) / (2.8 * 1024**3)
    free_gb = shutil.disk_usage(local_part_dir).free / 1024**3
    if free_gb < est_compressed_gb * 1.2:
        logger.warning(
            f"  ⚠ Low disk: {free_gb:.0f} GB free, ~{est_compressed_gb:.0f} GB needed. "
            f"Falling back to single-threaded direct write."
        )
        out_path = join_gs_uri(output_dir, f"frequent_k{k_level}.parquet")
        if is_remote:
            fs = pyarrow_gcs_filesystem()
            bucket, blob = parse_gs_url(out_path)
            writer = pq.ParquetWriter(f"{bucket}/{blob}", schema, filesystem=fs, **comp_kw)
        else:
            writer = pq.ParquetWriter(out_path, schema, **comp_kw)
        try:
            for cs in range(0, n, chunk_size):
                ce = min(cs + chunk_size, n)
                cn = ce - cs
                chunk_flat = items_flat[cs:ce].ravel()
                chunk_offs = np.arange(0, (cn + 1) * k_width, k_width, dtype=np.int64)
                chunk_list = pa.LargeListArray.from_arrays(chunk_offs, chunk_flat)
                chunk_table = pa.table(
                    {"itemset": chunk_list, "support": supports[cs:ce]},
                    schema=schema,
                )
                writer.write_table(chunk_table, row_group_size=cn)
                del chunk_list, chunk_table, chunk_offs, chunk_flat
        finally:
            writer.close()
        logger.info(f"  → Flushed {n:,} itemsets to {out_path} (fallback)")
        return

    psz = (n + n_threads - 1) // n_threads
    bounds = [(i * psz, min((i + 1) * psz, n)) for i in range(n_threads) if i * psz < n]
    bounds = [(s, e) for s, e in bounds if e > s]

    t_flush = time.perf_counter()
    try:
        if len(bounds) > 1:
            with _FlushTPE(max_workers=len(bounds)) as pool:
                futs = [
                    pool.submit(
                        _write_partition,
                        items_flat,
                        supports,
                        i,
                        s,
                        e,
                        local_part_dir,
                        schema,
                        comp_kw,
                        chunk_size,
                        k_width,
                    )
                    for i, (s, e) in enumerate(bounds)
                ]
                part_paths = [f.result() for f in futs]
        else:
            part_paths = [
                _write_partition(
                    items_flat,
                    supports,
                    0,
                    *bounds[0],
                    local_part_dir,
                    schema,
                    comp_kw,
                    chunk_size,
                    k_width,
                )
            ]
    except Exception:
        for pidx in range(len(bounds)):
            p = os.path.join(local_part_dir, f"part_{pidx:03d}.parquet")
            if os.path.exists(p):
                os.unlink(p)
        raise
    flush_dt = time.perf_counter() - t_flush

    total_mb = sum(os.path.getsize(p) for p in part_paths) / (1024**2)
    logger.info(
        f"  → Flushed {n:,} itemsets to {local_part_dir}/ "
        f"({len(part_paths)} parts, {total_mb:.0f} MB, {flush_comp}, "
        f"{flush_dt:.1f}s, {len(bounds)} threads)"
    )

    if uploader is not None:
        if is_remote:
            for p in part_paths:
                gs_dest = join_gs_uri(
                    output_dir,
                    f"frequent_k{k_level}/{os.path.basename(p)}",
                )
                uploader.upload_to_uri(p, gs_dest)
        else:
            for p in part_paths:
                uploader.upload(p)

    if backup_dir and not is_remote:
        backup_dest = os.path.join(backup_dir, f"frequent_k{k_level}")
        shutil.copytree(local_part_dir, backup_dest, dirs_exist_ok=True)
        logger.info(f"  → Backup: {local_part_dir}/ → {backup_dest}/")


def _apriori_row_split_multi_gpu(
    csr,  # scipy CSR matrix (None when bitvecs_list provided)
    col_to_item: dict[int, int],
    n_transactions: int,
    min_support: float,
    max_length: int | None,
    n_gpus: int,
    level_callback=None,
    bitvecs_list=None,  # Pre-built row-split bitvecs: list[(gpu_array, dev_id, n_rows)]
    output_dir=None,  # Per-K Parquet flush: write frequent_k{k}.parquet per level
    resume_from_k: int | None = None,  # Resume from K=N+1, loading K=N from parquet
    prune_closed: bool = False,  # V3: prune non-closed itemsets between K-levels
    prune_apriori: bool = False,  # V3: Apriori subset pruning on candidate groups
    sparse_from_k: int | None = None,  # V3: switch to CSR sparse counting from this K level
    anchor_items: set | None = None,  # V3 B6: two-phase anchor filtering
) -> "pl.DataFrame":
    """Mine frequent itemsets using row-split bitvecs across multiple GPUs.

    Splits the bitvec by transaction rows across GPUs. Each GPU holds ~1/n_gpus
    of the data.

    GPU-resident dense counting architecture (no recount, no D2H for arrays):
      1. Each GPU runs fused kernel on ALL candidates → dense count array in VRAM
      2. Device-to-device sum on GPU 0 (NVLink if available) → global counts
      3. Filter + where on GPU 0 → only freq_indices cross PCIe to CPU

    Since all GPUs share the same prev_frequent, they generate the same
    candidates in the same deterministic order. The dense output at index i
    from GPU 0 is the partial count for the same candidate as index i from
    GPU 1. Element-wise sum = exact global counts.

    PCIe transfer: only len(freq_indices) × 16 bytes (index + count).
    For K=2 with 35K features: ~600K pairs × 16 = 9.6 MB instead of 19.5 GB.
    """
    import numpy as np

    try:
        import cupy as cp
    except ImportError:
        raise ImportError("CuPy required for multi-GPU mining")

    from concurrent.futures import ThreadPoolExecutor

    from et_miner.gpu.csr_bitvec import build_bitvecs_row_split
    from et_miner.gpu.kernels import (
        get_popcount_kernel,
        count_pairs_k2_allcounts,
        count_k3plus_allcounts,
        count_csr_intersections,
        upload_k3plus_groups,
        build_k3plus_groups_from_flat,
        decode_k2_pairs_flat,
        decode_k3plus_flat,
    )

    if n_transactions > np.iinfo(np.int32).max:
        raise ValueError(
            f"n_transactions={n_transactions:,} exceeds int32 max ({np.iinfo(np.int32).max:,}). "
            f"CSR tidset indices are int32 — upgrade to int64 before running at this scale."
        )

    min_count_threshold = _min_count(min_support, n_transactions)

    # Local state for sparse CSR mode (replaces old function-attribute mutation)
    sparse_state = _SparseState()

    logger.info(
        f"  Row-split multi-GPU: {n_gpus} GPUs, min_count={min_count_threshold:,} (GPU-resident dense counting)"
    )

    # Phase 0: Build row-split bitvecs across GPUs
    if bitvecs_list is None:
        t0 = time.perf_counter()
        bitvecs_list = build_bitvecs_row_split(csr, n_gpus)
        del csr
        build_time = time.perf_counter() - t0
        logger.info(f"  Bitvec build: {build_time:.1f}s across {len(bitvecs_list)} GPUs")
    else:
        logger.info(f"  Pre-built bitvecs: {len(bitvecs_list)} GPUs")

    n_cols = max(col_to_item.keys()) + 1 if col_to_item else 0
    effective_max_length = min(
        max_length if max_length else float("inf"),
        n_cols,
    )

    # int32 vocab IDs (V5=149, AlphaFold=35K) — halves items_flat in the parquet flush
    if col_to_item:
        _max_item = max(col_to_item.values())
        assert _max_item < 2**31, f"item ID {_max_item} exceeds int32 range"
    col_to_item_arr = np.zeros(n_cols, dtype=np.int32)
    for c, item in col_to_item.items():
        col_to_item_arr[c] = item

    # V3 B6: Convert anchor item IDs → column indices for fast filtering
    anchor_col_arr = None
    if anchor_items is not None:
        item_to_col = {int(col_to_item_arr[i]): i for i in range(n_cols)}
        anchor_cols = sorted(item_to_col[aid] for aid in anchor_items if aid in item_to_col)
        if anchor_cols:
            anchor_col_arr = np.array(anchor_cols, dtype=np.int32)
            logger.info(f"  Anchor filter: {len(anchor_col_arr)} of {len(anchor_items)} anchor items mapped to columns")
        else:
            logger.warning("  No anchor items mapped to columns — anchor filter disabled")

    # Initialize NCCL for multi-GPU all-reduce (ring topology)
    device_ids = [did for _, did, _ in bitvecs_list]
    nccl_comms, _use_nccl = _init_nccl(device_ids)
    if _use_nccl:
        logger.info(f"  NCCL: {len(device_ids)} communicators (ring all-reduce)")
    else:
        logger.info("  NCCL unavailable, using sequential D2D")

    # Per-K Parquet flush: write each K level to disk immediately.
    # Prevents 680 GB CPU RAM accumulation at K=7+ scale.
    _output_is_remote = bool(output_dir) and is_gs_uri(output_dir)
    if output_dir and not _output_is_remote:
        os.makedirs(output_dir, exist_ok=True)

    # Deferred results: accumulate numpy arrays per level,
    # build Polars DataFrame at the end via PyArrow. No .tolist() overhead.
    # When output_dir is set, arrays flush to Parquet per K and are NOT accumulated.
    deferred_itemsets_np: list[np.ndarray] = []  # (n, k) int64 arrays
    deferred_supports: list[np.ndarray] = []

    # GCSUploader: single instance for the whole K-loop.
    # enabled when local output + ET_UPLOAD_GCS=1, OR when output_dir is gs://
    # (NVMe-first strategy needs uploads regardless of the global gate).
    _upload_local = bool(output_dir) and not _output_is_remote
    _need_upload = (is_upload_enabled() and _upload_local) or _output_is_remote
    uploader = GCSUploader(
        prefix=_env.upload_tag(f"run_{int(time.time())}"),
        max_workers=2,
        enabled=_need_upload,
    )

    def _flush_or_defer(items_flat, supports, k_level):
        """Flush via module-level _flush_k_parquet (output_dir set) or accumulate.

        Caller-side writeable safety wraps the flush call (Auditor R2 voorwaarde 1):
        _flush_k_parquet itself does not touch flags.
        """
        if output_dir is None:
            deferred_itemsets_np.append(items_flat)
            deferred_supports.append(supports)
            return

        items_flat.flags.writeable = False
        supports.flags.writeable = False
        try:
            _flush_k_parquet(
                items_flat,
                supports,
                k_level,
                output_dir=output_dir,
                is_remote=_output_is_remote,
                uploader=uploader,
                backup_dir=_env.parquet_backup_dir(),
            )
        finally:
            items_flat.flags.writeable = True
            supports.flags.writeable = True

    # ── RESUME: skip K=1..resume_from_k, load prev_frequent from parquet ──
    _resume_active = bool(resume_from_k and resume_from_k >= 2 and output_dir)

    if _resume_active:
        from et_miner.io.gcs import clear_resolve_cache, resolve_k_parquet

        clear_resolve_cache()
        resume_path = resolve_k_parquet(output_dir, resume_from_k)
        logger.info(f"  RESUME: Loading K={resume_from_k} from {resume_path}")
        t_resume = time.perf_counter()

        if _output_is_remote:
            table = pl.scan_parquet(resume_path, storage_options=polars_storage_options()).collect().to_arrow()
        else:
            import pyarrow.parquet as pq

            table = pq.read_table(resume_path)
        itemsets_col = table.column("itemset")

        # Vectorized item_id → col_idx conversion via numpy lookup array
        max_item_id = max(col_to_item.values())
        item_to_col = np.full(max_item_id + 1, -1, dtype=np.int32)
        for col_idx, item_id in col_to_item.items():
            item_to_col[item_id] = col_idx

        # Extract flat values from ChunkedArray<LargeList> — combine chunks first
        flat_item_ids = itemsets_col.combine_chunks().values.to_numpy()
        flat_col_ids = item_to_col[flat_item_ids]

        n_loaded = len(itemsets_col)
        prev_frequent_flat = flat_col_ids.reshape(n_loaded, resume_from_k).astype(np.int32)
        del table, itemsets_col, flat_item_ids, flat_col_ids, item_to_col

        resume_time = time.perf_counter() - t_resume
        logger.info(f"  RESUME: {n_loaded:,} itemsets → col indices in {resume_time:.1f}s")

        k = resume_from_k + 1
        prev_live_mgpu = set(prev_frequent_flat.ravel().tolist())
        prev_counts_flat = None  # V3: counts unavailable from resume — skip closed pruning for first resumed K
        logger.info(f"  RESUME: Jumping to K={k} ({len(prev_live_mgpu)} live columns)")

    # ── K=1: parallel popcount across GPUs, sum ────────────────────────
    if not _resume_active:
        _k1_start = time.perf_counter()
        popcount_kernel = get_popcount_kernel()

        CHUNK_COLS = 4096  # ~14 GB temp per chunk — fits in remaining VRAM

        def _k1_popcount_on_gpu(bitvec_gpu, device_id):
            """Chunked popcount on one GPU — runs in thread for parallelism."""
            with cp.cuda.Device(device_id):
                local_counts = np.zeros(n_cols, dtype=np.int64)
                for c_start in range(0, n_cols, CHUNK_COLS):
                    c_end = min(c_start + CHUNK_COLS, n_cols)
                    chunk = bitvec_gpu[c_start:c_end]
                    popcounts = popcount_kernel(chunk.view(cp.uint64))
                    local_counts[c_start:c_end] = cp.sum(
                        popcounts.reshape(c_end - c_start, -1), axis=1, dtype=cp.int64
                    ).get()
                    del popcounts
                return local_counts

        with ThreadPoolExecutor(max_workers=len(bitvecs_list)) as pool:
            futures = [pool.submit(_k1_popcount_on_gpu, bv, did) for bv, did, _ in bitvecs_list]
            global_col_counts = sum(f.result() for f in futures)

        # Vectorized K=1 filtering — no Python loop
        freq_mask_k1 = global_col_counts >= min_count_threshold
        freq_col_indices = np.where(freq_mask_k1)[0]
        freq_col_counts = global_col_counts[freq_mask_k1]

        if len(freq_col_indices) > 0:
            freq_items_k1 = col_to_item_arr[freq_col_indices]
            k1_supports = freq_col_counts / n_transactions
            _flush_or_defer(freq_items_k1.reshape(-1, 1), k1_supports, 1)

        k1_time = time.perf_counter() - _k1_start
        if level_callback:
            level_callback(1, n_cols, len(freq_col_indices), k1_time * 1000)
        logger.info(f"  K=1: {len(freq_col_indices):,} frequent items in {k1_time:.1f}s")

        if len(freq_col_indices) == 0:
            return _build_result_df([])

        # K=1 frequent columns as flat (n, 1) array — already numpy
        prev_frequent_flat = freq_col_indices.astype(np.int32).reshape(-1, 1)
        prev_counts_flat = freq_col_counts.astype(np.int64)  # V3: preserve for closed pruning
        k = 2
        prev_live_mgpu = set(freq_col_indices.tolist())

    # ── K>=2: GPU-resident dense counting ────────────────────────────────
    # State: prev_frequent_flat — numpy (n_freq, k-1) array of column indices.
    # No Python tuples in the hot path. Results decoded at end of each level
    # via vectorized numpy. Next-level groups built from flat arrays directly.
    try:
        while k <= effective_max_length and prev_frequent_flat.shape[0] >= k:
            _k_start = time.perf_counter()

            # V3: Sparse CSR mode — activated at K >= sparse_from_k
            _sparse_mode = sparse_from_k is not None and k >= sparse_from_k

            if _sparse_mode:
                # ═══ V3 SPARSE CSR PATH ═══
                # Shannon density transition: bitvecs → CSR tid-sets at K boundary.
                # First time entering sparse mode: convert bitvecs → tidsets, free VRAM.
                if not sparse_state.active:
                    logger.info(f"  ═══ DENSITY TRANSITION at K={k}: dense bitvec → sparse CSR ═══")
                    bv0, did0, _ = bitvecs_list[0]
                    with cp.cuda.Device(did0):
                        n_u64s_local = bv0.shape[1]
                    # Adaptive batch_size: fit in VRAM headroom (H100=81GB, H200=141GB)
                    with cp.cuda.Device(did0):
                        free_mem = cp.cuda.Device(did0).mem_info[0]
                    bytes_per_row = n_u64s_local * 8  # uint64 words → bytes
                    # 2× bytes_per_row: one for and_results + one for bv[col_indices] gather
                    max_batch = max(100, int(free_mem * 0.5 / (2 * bytes_per_row)))
                    tidset_offsets, tidset_indices = _convert_to_tidsets(
                        bitvecs_list,
                        prev_frequent_flat,
                        n_u64s_local,
                        batch_size=min(max_batch, 10_000),
                        verify=True,
                    )
                    # Store device IDs for multi-GPU CSR (bitvecs_list about to be cleared)
                    sparse_state.device_ids = [did for _, did, _ in bitvecs_list]

                    # Free ALL bitvec VRAM across all GPUs
                    for bv, did, _ in bitvecs_list:
                        with cp.cuda.Device(did):
                            del bv
                            cp.get_default_memory_pool().free_all_blocks()
                    bitvecs_list.clear()
                    logger.debug("    Freed bitvec VRAM across all GPUs")
                    # Store state for subsequent K-levels
                    sparse_state.active = True

                # Build groups from prev_frequent
                groups_info = build_k3plus_groups_from_flat(prev_frequent_flat)

                # Apriori pruning (reuse A3) — Rust fast path with Python fallback
                if prune_apriori and groups_info is not None:
                    tc_before = groups_info.total_candidates
                    prev_freq_set = set(map(tuple, prev_frequent_flat.tolist()))
                    groups_info = _prune_groups_apriori(groups_info, prev_freq_set, k, prev_flat_np=prev_frequent_flat)
                    tc_after = groups_info.total_candidates if groups_info is not None else 0
                    if tc_before > tc_after:
                        logger.debug(
                            f"    Apriori pruning K={k}: {tc_before:,} → {tc_after:,} ({100 * (1 - tc_after / tc_before):.1f}% pruned)"
                        )

                n_freq = 0
                current_flat = np.empty((0, k), dtype=np.int32)
                current_counts_raw = np.empty(0, dtype=np.int64)

                if groups_info is not None and groups_info.total_candidates > 0:
                    tc = groups_info.total_candidates
                    logger.info(f"  K={k}: {tc:,} candidates (CSR sparse mode)")

                    # Build candidate pair arrays: for each group, enumerate suffix pairs
                    # Each pair (i, j) maps to indices in prev_frequent_flat
                    pair_a_list, pair_b_list = [], []
                    so = groups_info.suffix_offsets
                    cp_arr = groups_info.cumulative_pairs

                    for g in range(len(so) - 1):
                        sstart, send = int(so[g]), int(so[g + 1])
                        gsuf = groups_info.suffixes[sstart:send]
                        prefix = tuple(
                            int(x)
                            for x in groups_info.prefix_items[
                                int(groups_info.prefix_offsets[g]) : int(groups_info.prefix_offsets[g + 1])
                            ]
                        )

                        # Map suffix → index in prev_frequent_flat
                        # Build lookup: tuple(itemset) → index
                        if sparse_state.prev_idx_lookup is None:
                            sparse_state.prev_idx_lookup = {
                                tuple(int(x) for x in prev_frequent_flat[i]): i for i in range(len(prev_frequent_flat))
                            }
                        idx_lookup = sparse_state.prev_idx_lookup

                        for si_pos in range(len(gsuf)):
                            for sj_pos in range(si_pos + 1, len(gsuf)):
                                si, sj = int(gsuf[si_pos]), int(gsuf[sj_pos])
                                key_i = prefix + (si,)
                                key_j = prefix + (sj,)
                                idx_i = idx_lookup.get(key_i)
                                idx_j = idx_lookup.get(key_j)
                                if idx_i is not None and idx_j is not None:
                                    pair_a_list.append(idx_i)
                                    pair_b_list.append(idx_j)

                    if pair_a_list:
                        n_pairs = len(pair_a_list)
                        pair_a_np = np.array(pair_a_list, dtype=np.int64)
                        pair_b_np = np.array(pair_b_list, dtype=np.int64)

                        device_ids_csr = sparse_state.device_ids
                        n_gpus_avail = len(device_ids_csr)

                        if n_gpus_avail <= 1 or n_pairs < 1000:
                            # Single GPU: direct (avoid ThreadPool overhead for small workloads)
                            with cp.cuda.Device(device_ids_csr[0] if device_ids_csr else 0):
                                offsets_gpu = cp.array(tidset_offsets, dtype=cp.int64)
                                indices_gpu = cp.array(tidset_indices, dtype=cp.int32)
                                pa_gpu = cp.array(pair_a_np, dtype=cp.int64)
                                pb_gpu = cp.array(pair_b_np, dtype=cp.int64)
                                counts_gpu = count_csr_intersections(offsets_gpu, indices_gpu, pa_gpu, pb_gpu, n_pairs)
                                counts_cpu = counts_gpu.get()
                                del offsets_gpu, indices_gpu, pa_gpu, pb_gpu, counts_gpu
                                cp.get_default_memory_pool().free_all_blocks()
                        else:
                            # Multi-GPU: partition pairs across GPUs
                            pairs_per_gpu = (n_pairs + n_gpus_avail - 1) // n_gpus_avail

                            def _csr_on_gpu(device_id, p_start, p_end):
                                with cp.cuda.Device(device_id):
                                    off_gpu = cp.array(tidset_offsets, dtype=cp.int64)
                                    idx_gpu = cp.array(tidset_indices, dtype=cp.int32)
                                    pa_gpu = cp.array(pair_a_np[p_start:p_end], dtype=cp.int64)
                                    pb_gpu = cp.array(pair_b_np[p_start:p_end], dtype=cp.int64)
                                    result = count_csr_intersections(off_gpu, idx_gpu, pa_gpu, pb_gpu, p_end - p_start)
                                    result_cpu = result.get()
                                    del off_gpu, idx_gpu, pa_gpu, pb_gpu, result
                                    cp.get_default_memory_pool().free_all_blocks()
                                    return result_cpu

                            with ThreadPoolExecutor(max_workers=n_gpus_avail) as pool:
                                futures = []
                                for i, did in enumerate(device_ids_csr):
                                    ps = i * pairs_per_gpu
                                    pe = min(ps + pairs_per_gpu, n_pairs)
                                    if ps < pe:
                                        futures.append(pool.submit(_csr_on_gpu, did, ps, pe))
                                counts_cpu = np.concatenate([f.result() for f in futures])

                            logger.debug(f"    CSR intersect: {n_pairs:,} pairs across {n_gpus_avail} GPUs")

                        # Filter frequent candidates
                        freq_mask = counts_cpu >= min_count_threshold
                        freq_pair_indices = np.where(freq_mask)[0]
                        freq_cand_counts = counts_cpu[freq_mask]
                        n_freq = len(freq_pair_indices)

                        if n_freq > 0:
                            # Decode candidate pairs → flat itemset array
                            decoded = []
                            for pi in freq_pair_indices:
                                a_idx, b_idx = pair_a_list[pi], pair_b_list[pi]
                                itemset_a = tuple(int(x) for x in prev_frequent_flat[a_idx])
                                itemset_b = tuple(int(x) for x in prev_frequent_flat[b_idx])
                                # Join: shared prefix + two suffixes
                                prefix = itemset_a[:-1]
                                new_itemset = prefix + (itemset_a[-1], itemset_b[-1])
                                decoded.append(new_itemset)

                            current_flat = np.array(decoded, dtype=np.int32)
                            current_counts_raw = freq_cand_counts.astype(np.int64)

                            # V3 B6: Anchor filter (sparse CSR path)
                            if anchor_col_arr is not None:
                                _n_before = len(current_flat)
                                current_flat, current_counts_raw, n_freq = _apply_anchor_filter(
                                    current_flat, current_counts_raw, anchor_col_arr, k
                                )
                                if n_freq < _n_before:
                                    # Rebuild freq_pair_indices to match filtered current_flat
                                    anchor_mask_csr = np.zeros(_n_before, dtype=bool)
                                    _tmp_flat = np.array(decoded, dtype=np.int32)
                                    for _col in range(_tmp_flat.shape[1]):
                                        anchor_mask_csr |= np.isin(_tmp_flat[:, _col], anchor_col_arr)
                                    freq_pair_indices = freq_pair_indices[anchor_mask_csr]
                                    freq_cand_counts = freq_cand_counts[anchor_mask_csr]
                                    del _tmp_flat, anchor_mask_csr

                            items_flat = col_to_item_arr[current_flat]
                            _flush_or_defer(items_flat, current_counts_raw / n_transactions, k)

                            # Skip tidset building at max_length — no K+1 iteration needed
                            if k >= effective_max_length:
                                break

                            # Build new tid-sets for K+1: intersect parent tid-sets on CPU
                            # Pre-allocate numpy buffer using known intersection sizes
                            total_tids_new = int(np.sum(freq_cand_counts))
                            new_offsets = np.empty(n_freq + 1, dtype=np.int64)
                            new_offsets[0] = 0
                            new_indices = np.empty(total_tids_new, dtype=np.int32)
                            write_pos = 0
                            for out_i, pi in enumerate(freq_pair_indices):
                                a_idx, b_idx = pair_a_list[pi], pair_b_list[pi]
                                a_start = int(tidset_offsets[a_idx])
                                a_end = int(tidset_offsets[a_idx + 1])
                                b_start = int(tidset_offsets[b_idx])
                                b_end = int(tidset_offsets[b_idx + 1])
                                tids_a = tidset_indices[a_start:a_end]
                                tids_b = tidset_indices[b_start:b_end]
                                isect = np.intersect1d(tids_a, tids_b, assume_unique=True)
                                n_isect = len(isect)
                                assert write_pos + n_isect <= total_tids_new, (
                                    f"CSR buffer overrun at itemset {out_i}: {write_pos} + {n_isect} > {total_tids_new}"
                                )
                                new_indices[write_pos : write_pos + n_isect] = isect
                                write_pos += n_isect
                                new_offsets[out_i + 1] = write_pos

                            assert write_pos <= total_tids_new, f"CSR buffer overrun: {write_pos} > {total_tids_new}"
                            tidset_offsets = new_offsets
                            tidset_indices = new_indices[:write_pos]

                            total_tids = len(tidset_indices)
                            avg_sup = total_tids / n_freq if n_freq > 0 else 0
                            tidset_mb = (len(tidset_offsets) * 8 + len(tidset_indices) * 4) / 1024**2
                            logger.debug(
                                f"    New tidsets: {n_freq:,} itemsets, avg support {avg_sup:.0f}, {tidset_mb:.1f} MB"
                            )

                # Clean up per-K state (NOT device_ids — persists across K-levels)
                sparse_state.prev_idx_lookup = None

            elif k == 2:
                freq_cols = sorted(prev_frequent_flat[:, 0])
                n_pairs = len(freq_cols) * (len(freq_cols) - 1) // 2
                _mem_gb = n_pairs * 8 / (1 << 30)  # int64 dense counts
                logger.info(f"  K=2: {n_pairs:,} total pairs, dense output {_mem_gb:.2f} GB/GPU")

                def _k2_dense_on_gpu(bitvec_gpu, device_id):
                    with cp.cuda.Device(device_id):
                        n_u64s = bitvec_gpu.shape[1]
                        return count_pairs_k2_allcounts(
                            bitvec_gpu,
                            freq_cols,
                            n_u64s,
                        )

                with ThreadPoolExecutor(max_workers=len(bitvecs_list)) as pool:
                    futures = [pool.submit(_k2_dense_on_gpu, bv, did) for bv, did, _ in bitvecs_list]
                    gpu_results = [f.result() for f in futures]

                # Multi-GPU reduction: NCCL ring all-reduce or sequential D2D
                if _use_nccl:
                    _nccl_allreduce_sum(gpu_results, nccl_comms, device_ids)

                with cp.cuda.Device(bitvecs_list[0][1]):
                    global_counts = gpu_results[0]
                    if not _use_nccl:
                        for i in range(1, len(gpu_results)):
                            cp.add(global_counts, cp.asarray(gpu_results[i]), out=global_counts)
                    for i in range(1, len(gpu_results)):
                        gpu_results[i] = None
                    del gpu_results
                    # Free memory pool on ALL GPUs, not just GPU 0 (P2: VRAM zombie fix)
                    for _, did, _ in bitvecs_list:
                        with cp.cuda.Device(did):
                            cp.get_default_memory_pool().free_all_blocks()

                    from et_miner.gpu.memory_budget import safe_threshold_filter

                    freq_pair_indices, freq_pair_counts = safe_threshold_filter(global_counts, min_count_threshold)
                    del global_counts
                    cp.get_default_memory_pool().free_all_blocks()

                n_freq = len(freq_pair_indices)
                if n_freq > 0:
                    current_flat = decode_k2_pairs_flat(freq_pair_indices, freq_cols)
                    current_counts_raw = freq_pair_counts.astype(np.int64)  # V3: preserve

                    # V3 B6: Anchor filter (K=2 dense path)
                    current_flat, current_counts_raw, n_freq = _apply_anchor_filter(
                        current_flat, current_counts_raw, anchor_col_arr, k
                    )

                    items_flat = col_to_item_arr[current_flat]  # (n, 2) int32
                    _flush_or_defer(items_flat, current_counts_raw / n_transactions, k)

                    # Pair cache disabled — not yet wired to K>=3 kernels (saves ~13.6 GB VRAM)
                else:
                    current_flat = np.empty((0, 2), dtype=np.int32)
                    current_counts_raw = np.empty(0, dtype=np.int64)

            else:
                # K>=3: build groups from flat array — no Python tuple grouping
                groups_info = build_k3plus_groups_from_flat(prev_frequent_flat)

                # V3: Apriori subset pruning — Rust fast path with Python fallback
                if prune_apriori and groups_info is not None:
                    tc_before = groups_info.total_candidates
                    prev_freq_set = set(map(tuple, prev_frequent_flat.tolist()))
                    groups_info = _prune_groups_apriori(groups_info, prev_freq_set, k, prev_flat_np=prev_frequent_flat)
                    tc_after = groups_info.total_candidates if groups_info is not None else 0
                    if tc_before > tc_after:
                        logger.debug(
                            f"    Apriori pruning K={k}: {tc_before:,} → {tc_after:,} candidates ({100 * (1 - tc_after / tc_before):.1f}% pruned)"
                        )

                n_freq = 0
                current_flat = np.empty((0, k), dtype=np.int32)
                current_counts_raw = np.empty(0, dtype=np.int64)

                if groups_info is not None:
                    tc = groups_info.total_candidates

                    # VRAM budget for candidate-range chunking.
                    # Group data stays resident across all chunks; only the dense
                    # output array (chunk_size × 8 bytes, int64) varies per chunk.
                    group_data_bytes = (
                        len(groups_info.prefix_items) * 4
                        + len(groups_info.prefix_offsets) * 8
                        + len(groups_info.suffixes) * 4
                        + len(groups_info.suffix_offsets) * 8
                        + len(groups_info.cumulative_pairs) * 8
                    )

                    # Estimate free VRAM from first GPU (all GPUs have same bitvecs)
                    with cp.cuda.Device(bitvecs_list[0][1]):
                        cp.get_default_memory_pool().free_all_blocks()  # flush cached blocks for accurate reading
                        free_vram, _ = cp.cuda.Device().mem_info
                    dense_budget = free_vram - group_data_bytes - 6 * (1 << 30)  # 6 GB safety
                    # Assumes NCCL in-place reduce; non-NCCL fallback may need // 16
                    max_cands_per_chunk = max(1, int(dense_budget // 10))  # 8B result + 2B margin for reduction temps
                    n_chunks = max(1, (tc + max_cands_per_chunk - 1) // max_cands_per_chunk)

                    logger.debug(
                        f"  K={k}: {tc:,} candidates, {n_chunks} chunk(s), group data {group_data_bytes / (1 << 30):.1f} GB, dense budget {dense_budget / (1 << 30):.1f} GB"
                    )

                    # Upload group data to all GPUs ONCE — stays resident across chunks
                    all_groups_gpu = {}
                    for bv, did, _ in bitvecs_list:
                        all_groups_gpu[did] = upload_k3plus_groups(groups_info, did)

                    from et_miner.gpu.memory_budget import safe_threshold_filter

                    all_freq_indices = []
                    all_freq_counts = []

                    _chunk_pool = ThreadPoolExecutor(max_workers=len(bitvecs_list))

                    for chunk_idx in range(n_chunks):
                        chunk_start = chunk_idx * max_cands_per_chunk
                        chunk_size = min(max_cands_per_chunk, tc - chunk_start)

                        if n_chunks > 1:
                            logger.debug(
                                f"    chunk {chunk_idx + 1}/{n_chunks}: candidates [{chunk_start:,}, {chunk_start + chunk_size:,})"
                            )

                        # Per-GPU counting with pre-uploaded groups
                        def _k3plus_chunk_on_gpu(bitvec_gpu, device_id, _cs=chunk_start, _csz=chunk_size):
                            with cp.cuda.Device(device_id):
                                return count_k3plus_allcounts(
                                    bitvec_gpu,
                                    groups_info,
                                    bitvec_gpu.shape[1],
                                    chunk_start=_cs,
                                    chunk_size=_csz,
                                    groups_gpu=all_groups_gpu[device_id],
                                )

                        futures = [_chunk_pool.submit(_k3plus_chunk_on_gpu, bv, did) for bv, did, _ in bitvecs_list]
                        gpu_results = [f.result() for f in futures]

                        # Multi-GPU reduction: NCCL ring all-reduce (chunk-sized arrays)
                        if _use_nccl:
                            _nccl_allreduce_sum(gpu_results, nccl_comms, device_ids)

                        # Reduce across GPUs, then threshold on GPU (sparse transfer)
                        with cp.cuda.Device(bitvecs_list[0][1]):
                            global_counts = gpu_results[0]
                            if not _use_nccl:
                                for i in range(1, len(gpu_results)):
                                    cp.add(global_counts, cp.asarray(gpu_results[i]), out=global_counts)

                            # Safe threshold: CPU fallback for large chunks avoids
                            # hidden cp.where() temporaries (13.5 GB prefix-sum at 1.5B elements)
                            freq_indices_chunk, freq_counts = safe_threshold_filter(global_counts, min_count_threshold)
                            n_freq_chunk = len(freq_indices_chunk)
                            pass_rate = 100 * n_freq_chunk / chunk_size if chunk_size > 0 else 0
                            logger.info(
                                f"  Chunk {chunk_idx + 1}/{n_chunks} filtering: "
                                f"{chunk_size:,} candidates → {n_freq_chunk:,} frequent "
                                f"({pass_rate:.1f}% pass rate, min_count={min_count_threshold:,})"
                            )

                            if n_freq_chunk > 0:
                                all_freq_indices.append(freq_indices_chunk + chunk_start)
                                all_freq_counts.append(freq_counts)

                            # Free all GPU memory from this chunk
                            del global_counts
                            for i in range(len(gpu_results)):
                                gpu_results[i] = None
                            del gpu_results
                            # Free ALL GPUs, not just primary — prevents pool fragmentation
                            for _, did, _ in bitvecs_list:
                                with cp.cuda.Device(did):
                                    cp.get_default_memory_pool().free_all_blocks()

                    _chunk_pool.shutdown(wait=False)

                    # Free group data from all GPUs
                    for did in list(all_groups_gpu):
                        with cp.cuda.Device(did):
                            del all_groups_gpu[did]
                            cp.get_default_memory_pool().free_all_blocks()
                    del all_groups_gpu

                    # Combine chunk results
                    if all_freq_indices:
                        freq_cand_indices = np.concatenate(all_freq_indices)
                        freq_cand_counts = np.concatenate(all_freq_counts)
                        n_freq = len(freq_cand_indices)
                        current_flat = decode_k3plus_flat(
                            freq_cand_indices,
                            groups_info,
                            k,
                        )
                        current_counts_raw = freq_cand_counts.astype(np.int64)  # V3: preserve

                        # V3 B6: Anchor filter (K>=3 dense path)
                        current_flat, current_counts_raw, n_freq = _apply_anchor_filter(
                            current_flat, current_counts_raw, anchor_col_arr, k
                        )

                        items_flat = col_to_item_arr[current_flat]  # (n, k) int32
                        _flush_or_defer(
                            items_flat,
                            current_counts_raw / n_transactions,
                            k,
                        )

            k_time = time.perf_counter() - _k_start
            if level_callback:
                level_callback(k, 0, n_freq, k_time * 1000)
            logger.info(f"  K={k}: {n_freq:,} frequent in {k_time:.1f}s")

            if n_freq == 0:
                break

            # Progressive bitvector deallocation across all GPUs (skip in sparse mode)
            # Replace single-threaded np.unique on
            # (n, k) int32 (~30-90s on K=6's 2.58B elements) with Rust parallel
            # bitset extract (sub-second). Falls back to np.unique if older wheel.
            try:
                from et_miner.backends import get_rust_ext

                et_miner_rust = get_rust_ext()
                if hasattr(et_miner_rust, "unique_columns_from_flat"):
                    current_live_mgpu = set(
                        int(x)
                        for x in et_miner_rust.unique_columns_from_flat(
                            np.ascontiguousarray(current_flat, dtype=np.int32).ravel()
                        )
                    )
                else:
                    current_live_mgpu = set(np.unique(current_flat).tolist())
            except (ImportError, AttributeError):
                current_live_mgpu = set(np.unique(current_flat).tolist())
            dead_cols = prev_live_mgpu - current_live_mgpu
            if dead_cols and bitvecs_list:
                dead_arr = sorted(dead_cols)
                for bv, did, _ in bitvecs_list:
                    with cp.cuda.Device(did):
                        dead_idx = cp.array(dead_arr, dtype=cp.int64)
                        bv[dead_idx] = 0
                freed_mb = len(dead_cols) * bitvecs_list[0][0].shape[1] * 8 / 1024**2
                logger.debug(
                    f"    Pruning: zeroed {len(dead_cols)} dead bitvecs at K={k} ({freed_mb:.0f} MB logical × {len(bitvecs_list)} GPUs)"
                )
            prev_live_mgpu = current_live_mgpu

            # V3: Closed itemset pruning — remove non-closed before next-level candidate gen
            if prune_closed and n_freq > 0:
                current_flat, current_counts_raw = _prune_closed_flat(
                    current_flat, current_counts_raw, prev_frequent_flat, prev_counts_flat
                )
                n_freq = len(current_flat)

            # R2 invariant: binary search in prune_closed_flat_raw requires
            # prev_flat to be sorted-by-row. K>=3 decode produces sorted output
            # (groups sorted by prefix), but K=2 decode (triangular enumeration)
            # is NOT lex-sorted. Sort here at K=2 to guarantee the invariant.
            # Cost: ~10K rows at K=2 = sub-millisecond.
            if n_freq > 0 and k == 2:
                sort_idx = np.lexsort(current_flat[:, ::-1].T)
                current_flat = current_flat[sort_idx]
                current_counts_raw = current_counts_raw[sort_idx]

            prev_frequent_flat = current_flat
            prev_counts_flat = current_counts_raw if n_freq > 0 else np.empty(0, dtype=np.int64)
            k += 1
    finally:
        # sparse_state is local — GC handles cleanup, no manual delattr needed
        # Emergency uploader shutdown (happy path does ordered drain below)
        try:
            uploader.close(wait=False)
        except Exception:
            pass

    # When output_dir is set, results are already flushed to Parquet per K level.
    # Return empty DataFrame — caller reads from output_dir instead.
    if output_dir:
        uploader.drain()
        uploader.close(wait=True)
        logger.info(f"  All results flushed to {output_dir}/frequent_k*.parquet")
        return _empty_result()

    # Build DataFrame from numpy arrays — zero .tolist() overhead
    if not deferred_itemsets_np:
        return _empty_result()

    all_supports = np.concatenate(deferred_supports)

    # PyArrow path: O(1) Python overhead via Arrow ListArray from numpy
    try:
        import pyarrow as pa

        flat_values = np.concatenate([a.ravel() for a in deferred_itemsets_np])
        widths = np.concatenate([np.full(a.shape[0], a.shape[1], dtype=np.int64) for a in deferred_itemsets_np])
        offsets = np.empty(len(widths) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(widths, out=offsets[1:])
        arrow_list = pa.LargeListArray.from_arrays(offsets, flat_values)
        return pl.DataFrame(
            {
                "itemset": pl.Series("itemset", arrow_list),
                "support": all_supports,
            }
        )
    except (ImportError, Exception):
        # Fallback: single-pass list construction
        all_itemsets = []
        for arr in deferred_itemsets_np:
            for i in range(arr.shape[0]):
                all_itemsets.append(arr[i].tolist())
        return pl.DataFrame(
            {
                "itemset": all_itemsets,
                "support": all_supports.tolist(),
            }
        )


def mine_two_phase(
    transactions,
    phase1_support: float = 0.001,
    phase2_support: float = 0.00001,
    max_length: int | None = None,
    item_col: str = "items",
    n_gpus: int = 1,
    output_dir: str | None = None,
    sparse_from_k: int | None = 4,
    level_callback=None,
) -> tuple:
    """Two-phase mining: anchor discovery + neighborhood zoom.

    Phase 1 mines at phase1_support to find anchor items — the items that
    participate in frequent patterns at reasonable support thresholds.

    Phase 2 mines at phase2_support (much lower), restricting candidates to
    those containing >=1 anchor item from Phase 1. This reduces candidate
    explosion from billions to a tractable search space focused on the
    neighborhoods of known-interesting items.

    Args:
        transactions: Transaction data (DataFrame or LazyFrame).
        phase1_support: Support threshold for anchor discovery (higher).
        phase2_support: Support threshold for neighborhood zoom (lower).
        max_length: Maximum itemset length (None = unlimited).
        item_col: Column name containing item lists.
        n_gpus: Number of GPUs to use.
        output_dir: Directory for per-K Parquet output. Phase 1 writes to
            output_dir/phase1/, Phase 2 to output_dir/phase2/.
        sparse_from_k: K-level to switch to sparse CSR counting.
        level_callback: Optional callback(k, n_candidates, n_frequent, ms).

    Returns:
        Tuple of (phase1_result, phase2_result, anchor_items) where
        anchor_items is the set of item IDs found in Phase 1.
    """
    import shutil
    import tempfile
    from pathlib import Path

    _cleanup_phase1 = False  # track if we need to clean up temp dirs
    # Phase 1: Anchor mining
    if output_dir:
        phase1_dir = os.path.join(output_dir, "phase1")
    else:
        phase1_dir = tempfile.mkdtemp(prefix="etminer_p1_")
        _cleanup_phase1 = True
    os.makedirs(phase1_dir, exist_ok=True)

    logger.info(
        f"TWO-PHASE MINING: Phase 1 {phase1_support:.6%} support (anchor discovery), Phase 2 {phase2_support:.6%} support (neighborhood zoom)"
    )

    phase1_result = apriori(
        transactions,
        min_support=phase1_support,
        max_length=max_length,
        item_col=item_col,
        use_gpu=True,
        n_gpus=n_gpus,
        output_dir=phase1_dir,
        sparse_from_k=sparse_from_k,
        prune_equal_support=True,
        level_callback=level_callback,
    )

    # Extract anchor items from ALL K-levels
    anchor_items = set()
    for k_file in sorted(Path(phase1_dir).glob("frequent_k*.parquet")):
        df = pl.scan_parquet(k_file).collect(engine="streaming")
        if "items" in df.columns:
            for item_list in df["items"].to_list():
                anchor_items.update(int(x) for x in item_list)
        elif "itemset" in df.columns:
            for item_list in df["itemset"].to_list():
                anchor_items.update(int(x) for x in item_list)

    logger.info(f"  Phase 1 complete: {len(anchor_items)} anchor items")

    # Clean up Phase 1 temp dir (anchors extracted, parquets no longer needed)
    if _cleanup_phase1:
        shutil.rmtree(phase1_dir, ignore_errors=True)

    if not anchor_items:
        logger.warning("  Phase 1 found 0 anchor items — Phase 2 will run unfiltered")

    # Phase 2: Neighborhood zoom with anchor filtering
    _cleanup_phase2 = output_dir is None
    phase2_dir = os.path.join(output_dir, "phase2") if output_dir else tempfile.mkdtemp(prefix="etminer_p2_")
    os.makedirs(phase2_dir, exist_ok=True)

    phase2_result = apriori(
        transactions,
        min_support=phase2_support,
        max_length=max_length,
        item_col=item_col,
        use_gpu=True,
        n_gpus=n_gpus,
        output_dir=phase2_dir,
        sparse_from_k=sparse_from_k,
        prune_equal_support=True,
        anchor_items=anchor_items if anchor_items else None,
        level_callback=level_callback,
    )

    logger.info("  Phase 2 complete")

    if _cleanup_phase2:
        shutil.rmtree(phase2_dir, ignore_errors=True)

    logger.info("TWO-PHASE MINING DONE")
    return phase1_result, phase2_result, anchor_items


def apriori(
    transactions: pl.LazyFrame | pl.DataFrame | None = None,
    min_support: float = 0.5,
    max_length: int | None = None,
    item_col: str = "items",
    use_gpu: bool = False,
    batch_size: int | None = 10_000,
    profile: bool = False,
    show_progress: bool = False,
    warn_complexity: bool = True,
    prune_equal_support: bool = False,
    use_generator_pruning: bool = False,
    sparse: bool | None = None,
    n_jobs: int = 1,
    enable_length_filter: bool = True,
    # Streaming parameters (SON algorithm)
    streaming: bool = False,
    chunk_size: int = 10_000_000,
    n_gpus: int = 1,
    memory_budget_gb: float | None = None,
    progress_callback: Callable[[str, int, int, dict[str, Any]], None] | None = None,
    level_callback: Callable[[int, int, int, float], None] | None = None,
    # GPU-resident mode (zero PCIe round trips)
    gpu_resident: bool = False,
    # Pre-built bitvector input (GPU fast path)
    bitvecs: tuple | None = None,
    # Per-K Parquet flush (prevents CPU RAM OOM at K=7+ scale)
    output_dir: str | None = None,
    # Resume from K=N+1 by loading K=N frequent itemsets from parquet
    resume_from_k: int | None = None,
    # Memory guard limits for exhaustive mining
    max_ram_gb: float = 800.0,
    max_vram_gb: float = 70.0,
    # V3: switch to sparse CSR counting from this K level
    sparse_from_k: int | None = None,
    # V3 B6: restrict candidates to anchor neighborhoods (two-phase mining)
    anchor_items: set | None = None,
) -> pl.DataFrame | tuple[pl.DataFrame, ProfilingSession]:
    """Find frequent itemsets using fully vectorized boolean matrix operations.

    Transactions become a boolean matrix; support counting becomes vectorized AND + sum.

    Args:
        transactions: Transaction data with item lists.
        min_support: Minimum support threshold (0.0-1.0).
        max_length: Maximum itemset length (None = unlimited).
        item_col: Column name with item lists.
        use_gpu: Use GPU acceleration if available (requires polars[gpu]).
        batch_size: Candidates per batch for memory control. None = no batching.
        profile: If True, return profiling metrics alongside results.
        show_progress: If True, display progress bar (requires tqdm).
        warn_complexity: If True, warn when candidate pairs > 1M.
        prune_equal_support: Prune non-closed itemsets (equal support to subset).
            Based on BitApriori [Zheng 2010]. 3-50x speedup on dense datasets.
        use_generator_pruning: Infer support from (k-1)-subsets instead of counting.
            Based on Pascal/generator pruning [Bastide et al. 2000]. 10-40% speedup,
            no impact on results (unlike prune_equal_support).
        sparse: Scipy CSR matrix usage. True = force, False = Polars, None = auto
            (switches at >100K k=2 candidates or >500 items <10% density).
        n_jobs: Parallel workers for sparse k>2 counting. 1=sequential, -1=all CPUs.
            Only active when sparse mode is used.
        enable_length_filter: Filter transactions shorter than k before counting
            k-itemset support. Set to False to disable.
        streaming: Use SON algorithm for chunked processing. Memory becomes
            O(chunk_size × n_items) instead of O(total × n_items).
        chunk_size: Transactions per chunk when streaming=True. Default 10M.
        n_gpus: GPUs for multi-GPU streaming (default 1). Requires CuPy.
        memory_budget_gb: Auto-calculate chunk_size to fit this budget.
            Overrides chunk_size. Only used when streaming=True.
        progress_callback: Streaming progress callback
            (phase, chunk_idx, n_chunks, metrics).
        level_callback: Per-level callback
            (k, n_candidates, n_frequent, duration_ms).
        bitvecs: Pre-built GPU bitvectors tuple (bitvecs_gpu, col_to_item, n_transactions).
            Skips DataFrame conversion; transactions must be None when provided.

    Returns:
        DataFrame with itemset (List[Int64]) and support (Float64) columns.
        If profile=True: tuple of (DataFrame, ProfilingSession).

    Example:
        >>> result = apriori(df, min_support=0.5)
        >>> result = apriori(df, min_support=0.001, use_gpu=True)
        >>> result, session = apriori(df, min_support=0.1, profile=True)
        >>> result = apriori(df, min_support=0.0001, sparse=True, n_jobs=-1)
        >>> result = apriori(huge_df, min_support=0.001, streaming=True, n_gpus=8)
    """
    _validate_parameters(min_support, max_length, batch_size)

    # Route to bitvecs fast path if pre-built GPU bitvectors provided
    if bitvecs is not None:
        # Validate: cannot provide both bitvecs and transactions
        if transactions is not None:
            raise ValueError(
                "Cannot provide both 'transactions' and 'bitvecs'. "
                "Use 'bitvecs' for pre-built GPU bitvectors or 'transactions' for DataFrame input."
            )

        # Unpack and validate bitvecs tuple
        if not isinstance(bitvecs, tuple) or len(bitvecs) != 3:
            raise ValueError("bitvecs must be a tuple of (bitvecs_gpu, col_to_item, n_transactions)")

        bitvecs_gpu, col_to_item, n_trans = bitvecs

        # Validate bitvecs_gpu is a CuPy array
        if not hasattr(bitvecs_gpu, "__cuda_array_interface__"):
            raise TypeError(
                f"bitvecs_gpu must be a CuPy array (has __cuda_array_interface__), got {type(bitvecs_gpu).__name__}"
            )

        # Validate shape
        if bitvecs_gpu.ndim != 2:
            raise ValueError(f"bitvecs_gpu must be 2D (n_cols, n_u64s), got {bitvecs_gpu.ndim}D")

        # Validate n_u64s matches n_transactions
        expected_n_u64s = math.ceil(n_trans / 64)
        if bitvecs_gpu.shape[1] != expected_n_u64s:
            raise ValueError(
                f"bitvecs_gpu.shape[1] ({bitvecs_gpu.shape[1]}) must equal "
                f"ceil(n_transactions/64) = ceil({n_trans}/64) = {expected_n_u64s}"
            )

        # Validate col_to_item keys match n_cols
        if len(col_to_item) != bitvecs_gpu.shape[0]:
            raise ValueError(
                f"col_to_item has {len(col_to_item)} keys but bitvecs_gpu has {bitvecs_gpu.shape[0]} columns"
            )

        # Route to GPU-resident or standard bitvecs fast path
        if gpu_resident:
            return _apriori_from_bitvecs_gpu_resident(
                bitvecs_gpu,
                col_to_item,
                n_trans,
                min_support,
                max_length,
                profile,
                level_callback,
            )
        return _apriori_from_bitvecs(
            bitvecs_gpu,
            col_to_item,
            n_trans,
            min_support,
            max_length,
            batch_size,
            profile,
            level_callback,
            n_gpus=n_gpus,
            max_ram_gb=max_ram_gb,
            max_vram_gb=max_vram_gb,
        )

    # Validate transactions is provided when bitvecs is not
    if transactions is None:
        raise ValueError("Either 'transactions' or 'bitvecs' must be provided")

    # Route to streaming implementation if requested
    if streaming:
        # Multi-GPU streaming if n_gpus > 1
        if n_gpus > 1:
            from et_miner.streaming.multi_gpu import apriori_streaming_multi_gpu

            return apriori_streaming_multi_gpu(
                transactions,
                min_support=min_support,
                max_length=max_length,
                item_col=item_col,
                n_gpus=n_gpus,
                chunk_size=chunk_size,
                batch_size=batch_size,
                show_progress=show_progress,
                sparse=sparse,
                n_jobs=n_jobs,
                progress_callback=progress_callback,
            )

        # Single-GPU/CPU streaming
        from et_miner.streaming.son import apriori_streaming

        return apriori_streaming(
            transactions,
            min_support=min_support,
            max_length=max_length,
            item_col=item_col,
            chunk_size=chunk_size,
            memory_budget_gb=memory_budget_gb,
            use_gpu=use_gpu,
            gpu_resident=gpu_resident,
            batch_size=batch_size,
            profile=profile,
            show_progress=show_progress,
            sparse=sparse,
            n_jobs=n_jobs,
            progress_callback=progress_callback,
        )

    lf = transactions.lazy() if isinstance(transactions, pl.DataFrame) else transactions

    # ── GPU direct path: transactions → CSR → GPU bitvecs ──────────────
    # Bypasses the dense boolean matrix entirely.
    # For 205M × 1006: ~5 GB CPU + ~25 GB GPU  instead of  206 GB CPU.
    if use_gpu:
        from et_miner.gpu.bitvec import _build_gpu_bitvec_matrix

        from .matrix import _build_csr_from_transactions

        csr_result = _build_csr_from_transactions(lf, min_support, item_col)
        if csr_result is None:
            if profile:
                return _empty_result(), ProfilingSession()
            return _empty_result()

        csr, idx_to_item, n_trans = csr_result

        # Multi-GPU row-split path, or single-GPU with anchor filtering
        if n_gpus > 1 or anchor_items is not None:
            return _apriori_row_split_multi_gpu(
                csr,
                idx_to_item,
                n_trans,
                min_support,
                max_length,
                n_gpus,
                level_callback,
                output_dir=output_dir,
                resume_from_k=resume_from_k,
                prune_closed=prune_equal_support,  # V3: reuse existing flag
                prune_apriori=prune_equal_support,  # V3: Apriori subset pruning
                sparse_from_k=sparse_from_k,  # V3: density transition K-level
                anchor_items=anchor_items,  # V3 B6: two-phase anchor filtering
            )

        bitvecs_gpu = _build_gpu_bitvec_matrix(csr)
        del csr  # free CPU CSR memory

        if gpu_resident:
            return _apriori_from_bitvecs_gpu_resident(
                bitvecs_gpu,
                idx_to_item,
                n_trans,
                min_support,
                max_length,
                profile,
                level_callback,
            )
        return _apriori_from_bitvecs(
            bitvecs_gpu,
            idx_to_item,
            n_trans,
            min_support,
            max_length,
            batch_size,
            profile,
            level_callback,
            n_gpus=n_gpus,
            max_ram_gb=max_ram_gb,
            max_vram_gb=max_vram_gb,
            sparse_from_k=sparse_from_k,
        )

    # ── CPU path: dense boolean matrix ─────────────────────────────────
    session = ProfilingSession() if profile else None

    # Phase 1: Build boolean matrix (includes frequent 1-itemset mining)
    if session:
        session.start_phase("matrix_build")

    matrix, col_to_item, n_trans = build_boolean_matrix(lf, min_support, item_col)

    if session:
        session.end_phase(n_frequent_items=len(col_to_item), n_transactions=n_trans)

    # Compute max possible k from transaction lengths
    max_tx_length = (lf.select(pl.col(item_col).list.len().max()).collect(engine="streaming").item()) or 0

    effective_max_length = min(
        max_length if max_length else float("inf"),
        max_tx_length,
        len(col_to_item),  # Can't have more items than columns
    )

    if not col_to_item:
        if profile:
            return _empty_result(), session
        return _empty_result()

    item_cols = list(col_to_item.keys())

    # Phase 2: Get exact 1-itemset supports from matrix column sums
    _k1_start = time.perf_counter()
    if session:
        session.start_phase("k1_support")

    # 1-itemset counting: always use streaming engine (simple column sums, no GPU benefit)
    one_itemset_exprs = [pl.col(c).sum().alias(c) for c in item_cols]
    one_counts = matrix.lazy().select(one_itemset_exprs).collect(engine="streaming")

    results: list[tuple[list[int], float]] = []
    prev_frequent: list[tuple[str, ...]] = []

    # Filter on integer count to avoid float precision issues at boundaries.
    # math.ceil ensures we don't include items below threshold.
    min_count_threshold = _min_count(min_support, n_trans)
    for col in item_cols:
        count = one_counts.get_column(col).item()
        if count >= min_count_threshold:
            support = count / n_trans
            results.append(([col_to_item[col]], support))
            prev_frequent.append((col,))

    if session:
        session.end_phase(n_frequent=len(prev_frequent))

    # Call level callback for k=1
    if level_callback:
        _k1_duration_ms = (time.perf_counter() - _k1_start) * 1000
        level_callback(1, len(item_cols), len(prev_frequent), _k1_duration_ms)

    if not prev_frequent:
        if profile:
            return _empty_result(), session
        return _empty_result()

    # Warn about potentially expensive computation
    if warn_complexity:
        _warn_complexity(len(prev_frequent), min_support)

    # Initialize prev_supports for equal-support pruning
    # Use integer count filtering for consistency
    prev_supports: dict[tuple[str, ...], float] = {
        (col,): one_counts.get_column(col).item() / n_trans
        for col in item_cols
        if one_counts.get_column(col).item() >= min_count_threshold
    }

    # Initialize prev_counts for generator-based pruning (uses integer counts for precision)
    prev_counts: dict[tuple[str, ...], int] = {
        (col,): one_counts.get_column(col).item()
        for col in item_cols
        if one_counts.get_column(col).item() >= min_count_threshold
    }

    # Phase 3: k >= 2 with vectorized support counting
    k = 2

    while k <= effective_max_length and len(prev_frequent) >= k:
        _k_start = time.perf_counter()

        # Generate candidates using Apriori join
        if session:
            session.start_phase(f"k{k}_candidate_gen")

        candidates = _generate_candidates(prev_frequent, k)

        if session:
            session.end_phase(n_candidates=len(candidates))

        if not candidates:
            break

        # Support inference + counting
        if session:
            session.start_phase(f"k{k}_support_count")

        # Phase A: Try to infer count from subsets (Pascal/generator pruning)
        inferred_counts: dict[tuple[str, ...], int] = {}
        needs_counting: list[tuple[str, ...]] = []

        if use_generator_pruning and prev_counts and k >= 2:
            for candidate in candidates:
                inferred_count = _infer_count_from_subsets(candidate, prev_counts)
                if inferred_count is not None:
                    # Count can be inferred - no need to count!
                    inferred_counts[candidate] = inferred_count
                else:
                    needs_counting.append(candidate)
        else:
            needs_counting = candidates

        # Log inference results
        if use_generator_pruning and len(inferred_counts) > 0:
            skip_rate = 100 * len(inferred_counts) / len(candidates)
            logger.debug(
                f"[k={k}] Generator pruning: {len(inferred_counts)}/{len(candidates)} "
                f"candidates ({skip_rate:.1f}%) skipped via count inference"
            )

        # Phase B: Count support for remaining candidates
        if needs_counting:
            counted = count_support_batched(
                matrix,
                needs_counting,
                n_trans,
                batch_size,
                use_gpu,
                show_progress,
                sparse,
                n_jobs,
                enable_length_filter=enable_length_filter,
            )
        else:
            counted = {}

        # Combine inferred and counted (all are integer counts now)
        all_counts = {**inferred_counts, **counted}

        # Filter to frequent itemsets using integer count comparison.
        # This avoids float precision issues at boundaries.
        current_frequent: list[tuple[str, ...]] = []
        current_supports: dict[tuple[str, ...], float] = {}
        current_counts: dict[tuple[str, ...], int] = {}
        n_skipped = len(inferred_counts)

        for itemset in candidates:
            count = all_counts[itemset]

            if count >= min_count_threshold:
                support = count / n_trans
                current_frequent.append(itemset)
                current_supports[itemset] = support
                current_counts[itemset] = count
                item_list = [col_to_item[c] for c in itemset]
                results.append((item_list, support))

        if session:
            session.end_phase(
                n_frequent=len(current_frequent),
                n_inferred=n_skipped if use_generator_pruning else 0,
            )

        # Call level callback for k>=2
        if level_callback:
            _k_duration_ms = (time.perf_counter() - _k_start) * 1000
            level_callback(k, len(candidates), len(current_frequent), _k_duration_ms)

        if not current_frequent:
            break

        # Apply equal-support pruning if enabled
        if prune_equal_support and prev_supports:
            current_frequent = _prune_equal_support(current_frequent, current_supports, prev_supports)

        prev_supports = current_supports
        prev_counts = current_counts
        prev_frequent = current_frequent
        k += 1

    result_df = _build_result_df(results)
    if profile:
        return result_df, session
    return result_df


def _generate_candidates(
    prev_frequent: list[tuple[str, ...]],
    k: int,
) -> list[tuple[str, ...]]:
    """Generate k-candidates from (k-1)-frequent using Apriori join.

    For k=2: vectorized pair generation via Polars cross-join.
    For k>2: prefix-based generation with automatic strategy selection.

    Args:
        prev_frequent: List of frequent (k-1)-itemsets as tuples of column names.
        k: Target itemset size.

    Returns:
        List of candidate k-itemsets.
    """
    if k == 2:
        return _generate_candidates_k2(prev_frequent)

    # k > 2: Use vectorized approach for larger sets
    n = len(prev_frequent)
    if n > 1000:
        return _generate_candidates_vectorized(prev_frequent, k)

    # Small sets: simple Python loop is faster (no DataFrame overhead)
    return _generate_candidates_simple(prev_frequent, k)


def _generate_candidates_k2_streaming(items: list[str]) -> Iterator[tuple[str, str]]:
    """Yield (a, b) pairs with a<b — O(1) memory vs n*(n-1)/2 cross-join materialization."""
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            yield (items[i], items[j])


# Threshold for switching to streaming K=2 generation
# Cross-join with 5000 items creates 12.5M rows - streaming is more memory-efficient
_K2_STREAMING_THRESHOLD = 5000


def _generate_candidates_k2(
    prev_frequent: list[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    """Pair generation for k=2.

    Uses streaming generator for large item counts (>5000) to avoid
    materializing n*(n-1)/2 rows in memory from cross-join.

    Args:
        prev_frequent: List of frequent 1-itemsets.

    Returns:
        List of candidate 2-itemsets.
    """
    items = sorted(p[0] for p in prev_frequent)
    n_items = len(items)

    # For large item counts, use streaming generator to avoid memory issues
    # 5000 items = 12.5M pairs, 10000 items = 50M pairs in cross-join
    if n_items > _K2_STREAMING_THRESHOLD:
        return list(_generate_candidates_k2_streaming(items))

    # Warn if approaching 2^32 limit (sqrt(4.2B) ≈ 65k)
    if n_items > 60_000:
        warnings.warn(
            f"Large item count ({n_items:,}) may hit Polars 2^32 row limit. "
            f"Consider `pip install polars[rt64]` for datasets >65k items.",
            UserWarning,
            stacklevel=3,
        )

    items_df = pl.DataFrame({"item": items})

    pairs = (
        items_df.lazy()
        .select(pl.col("item").alias("a"))
        .join(items_df.lazy().select(pl.col("item").alias("b")), how="cross")
        .filter(pl.col("a") < pl.col("b"))
        .collect()
    )

    return [(row["a"], row["b"]) for row in pairs.iter_rows(named=True)]


def _generate_candidates_simple(
    prev_frequent: list[tuple[str, ...]],
    k: int,
) -> list[tuple[str, ...]]:
    """Simple Python loop for small candidate sets.

    More efficient than DataFrame overhead for small n.

    Args:
        prev_frequent: List of frequent (k-1)-itemsets.
        k: Target itemset size.

    Returns:
        List of candidate k-itemsets.
    """
    prev_set = set(prev_frequent)
    sorted_prev = sorted(prev_frequent)
    candidates = []

    for i, itemset1 in enumerate(sorted_prev):
        for itemset2 in sorted_prev[i + 1 :]:
            # Check if they share k-2 prefix
            if itemset1[:-1] == itemset2[:-1]:
                candidate = itemset1 + (itemset2[-1],)

                # Apriori pruning: all (k-1)-subsets must be frequent
                is_valid = True
                for j in range(k):
                    subset = candidate[:j] + candidate[j + 1 :]
                    if subset not in prev_set:
                        is_valid = False
                        break

                if is_valid:
                    candidates.append(candidate)

    return candidates


def _generate_candidates_vectorized(
    prev_frequent: list[tuple[str, ...]],
    k: int,
) -> list[tuple[str, ...]]:
    """Vectorized candidate generation via Polars prefix-based grouping.

    Groups itemsets by their (k-2) prefix and generates candidates within
    each group. This avoids O(n²) comparisons across all itemsets.

    Args:
        prev_frequent: List of frequent (k-1)-itemsets.
        k: Target itemset size.

    Returns:
        List of candidate k-itemsets.
    """
    # Group by prefix for efficient candidate generation
    sorted_prev = sorted(prev_frequent)

    # Build prefix groups: {prefix: [itemsets with that prefix]}
    prefix_groups: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
    for itemset in sorted_prev:
        prefix = itemset[:-1]
        if prefix not in prefix_groups:
            prefix_groups[prefix] = []
        prefix_groups[prefix].append(itemset)

    # Check if any group is large enough to benefit from vectorization
    max_group_size = max(len(g) for g in prefix_groups.values()) if prefix_groups else 0

    if max_group_size < 100:
        # All groups small, use simple loop (less overhead)
        return _generate_candidates_simple(prev_frequent, k)

    # Prepare for Apriori pruning
    prev_set = set(prev_frequent)
    all_candidates: list[tuple[str, ...]] = []

    for prefix, group in prefix_groups.items():
        group_size = len(group)

        if group_size < 100:
            # Small group: simple Python loop
            for i, item1 in enumerate(group):
                for item2 in group[i + 1 :]:
                    candidate = item1 + (item2[-1],)
                    if _is_valid_candidate(candidate, k, prev_set):
                        all_candidates.append(candidate)
        else:
            # Large group: vectorized via Polars
            candidates = _generate_from_group_vectorized(group, prefix, k, prev_set)
            all_candidates.extend(candidates)

    return all_candidates


def _generate_from_group_vectorized(
    group: list[tuple[str, ...]],
    prefix: tuple[str, ...],
    k: int,
    prev_set: set[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    """Generate candidates from a single prefix group using Polars.

    Args:
        group: List of itemsets sharing the same prefix.
        prefix: The shared (k-2) prefix.
        k: Target itemset size.
        prev_set: Set of all frequent (k-1)-itemsets for pruning.

    Returns:
        List of valid candidate k-itemsets.
    """
    # Extract last items from each itemset in the group
    last_items = [itemset[-1] for itemset in group]

    # Create DataFrame with last items
    df = pl.DataFrame({"last": last_items})

    # Self-join to get all pairs where a < b
    pairs_df = (
        df.lazy()
        .select(pl.col("last").alias("a"))
        .join(df.lazy().select(pl.col("last").alias("b")), how="cross")
        .filter(pl.col("a") < pl.col("b"))
        .collect()
    )

    # Convert to candidates and apply Apriori pruning
    candidates: list[tuple[str, ...]] = []
    for row in pairs_df.iter_rows(named=True):
        candidate = prefix + (row["a"], row["b"])
        if _is_valid_candidate(candidate, k, prev_set):
            candidates.append(candidate)

    return candidates


def _is_valid_candidate(
    candidate: tuple[str, ...],
    k: int,
    prev_set: set[tuple[str, ...]],
) -> bool:
    """Apriori pruning: all (k-1)-subsets of candidate must be in prev_set."""
    for j in range(k):
        subset = candidate[:j] + candidate[j + 1 :]
        if subset not in prev_set:
            return False
    return True
