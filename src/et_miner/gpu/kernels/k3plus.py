"""K>=3 prefix groups (build, select, upload) and the per-candidate dense count."""

from __future__ import annotations

from collections import namedtuple

import numpy as np
from loguru import logger

from .loader import _assert_bitvecs, _grid_dims, get_cuda_kernel


# ── Dense counting for row-split multi-GPU ────────────────────────────
# Eliminates the recount phase by outputting counts for ALL candidates.
# In row-split mode, all GPUs generate the same candidates (same prev_frequent),
# so element-wise sum of dense count arrays = exact global counts.

K3PlusGroups = namedtuple(
    "K3PlusGroups",
    [
        "prefix_items",
        "prefix_offsets",
        "suffixes",
        "suffix_offsets",
        "cumulative_pairs",
        "total_candidates",
        "groups",
    ],
)

_STALE_RUST_WARNED = False


def _warn_stale_rust_once(what: str) -> None:
    """One-time warning when the installed et_miner_rust predates an API it
    is being called with; callers then take the numpy/Python fallbacks."""
    global _STALE_RUST_WARNED
    if not _STALE_RUST_WARNED:
        _STALE_RUST_WARNED = True
        from et_miner.backends import BUILD_COMMAND

        logger.warning(
            f"et_miner_rust is stale ({what}) — rebuild with `{BUILD_COMMAND}`; "
            "using numpy/Python fallbacks meanwhile"
        )


_MISSING_RUST_WARNED = False


def _warn_missing_rust_once(what: str) -> None:
    """One-time warning when a Rust fast path falls back for want of the
    extension.

    The itemsets are identical on the fallback; the cost is not. Both callers
    sit on the downward-closure row-split path, where candidate generation is
    the dominant term -- measured at 93% of mining time over a nine-level run (K=2..K=10),
    with the two Rust entry points 9.3x apart from their fallbacks. Without
    this line the whole difference is invisible: the extension is pruned by a
    routine `uv sync`, the import fails, and the run is simply slower.
    """
    global _MISSING_RUST_WARNED
    if not _MISSING_RUST_WARNED:
        _MISSING_RUST_WARNED = True
        from et_miner.backends import BUILD_COMMAND, RUST_DISABLED

        if RUST_DISABLED:
            logger.warning(
                f"et_miner_rust is disabled by ET_MINER_DISABLE_RUST=1 — {what} is taking the Python fallback."
            )
            return
        logger.warning(
            f"et_miner_rust is not installed — {what} is taking the Python fallback. "
            f"Same itemsets, ~9x the candidate-generation time. Build it with `{BUILD_COMMAND}`"
        )


def _warn_rust_failed(what: str, exc: BaseException) -> None:
    """Warn when a Rust call raised and the caller degraded to the fallback.

    Not rate-limited and not once-only: this is a malfunction rather than a
    build state, every occurrence is worth seeing, and the fallback makes it
    otherwise indistinguishable from success.
    """
    logger.warning(f"et_miner_rust.{what} raised {type(exc).__name__}: {exc} — using the Python fallback")


def select_k3plus_groups(groups_info, keep):
    """The groups where ``keep`` is True, as their own K3PlusGroups.

    Candidate indices restart at 0 in the selection, in the same order, so
    decode_k3plus_flat decodes its survivors against the selection.
    """
    keep = np.asarray(keep, dtype=bool)
    idx = np.nonzero(keep)[0]
    po = np.asarray(groups_info.prefix_offsets, dtype=np.int64)
    so = np.asarray(groups_info.suffix_offsets, dtype=np.int64)
    p_len = po[idx + 1] - po[idx]
    s_len = so[idx + 1] - so[idx]

    def _gather(starts, lengths):
        offsets = np.zeros(len(lengths), dtype=np.int64)
        np.cumsum(lengths[:-1], out=offsets[1:])
        return np.repeat(starts - offsets, lengths) + np.arange(int(lengths.sum()), dtype=np.int64)

    cumulative_pairs = np.zeros(len(idx) + 1, dtype=np.int64)
    np.cumsum(s_len * (s_len - 1) // 2, out=cumulative_pairs[1:])
    return K3PlusGroups(
        prefix_items=np.asarray(groups_info.prefix_items)[_gather(po[idx], p_len)],
        prefix_offsets=np.concatenate([[0], np.cumsum(p_len)]).astype(np.int64),
        suffixes=np.asarray(groups_info.suffixes)[_gather(so[idx], s_len)],
        suffix_offsets=np.concatenate([[0], np.cumsum(s_len)]).astype(np.int64),
        cumulative_pairs=cumulative_pairs,
        total_candidates=int(cumulative_pairs[-1]),
        groups=None,
    )


def upload_k3plus_groups(groups_info, device_id):
    """Upload K3+ group data to GPU once, keep resident across chunks — ~40 GB at K=8.

    Includes "ctp" (cumulative tile-pairs) for the shared/tiled kernel;
    negligible extra bytes (one int64 per group + 1) for the legacy path.
    "tc" carries total_candidates for range checks.
    """
    import cupy as cp

    from .shared_tiled import compute_cumulative_tilepairs

    with cp.cuda.Device(device_id):
        out = {
            "gpi": cp.array(groups_info.prefix_items, dtype=cp.int32),
            "gpo": cp.array(groups_info.prefix_offsets, dtype=cp.int64),
            "gs": cp.array(groups_info.suffixes, dtype=cp.int32),
            "gso": cp.array(groups_info.suffix_offsets, dtype=cp.int64),
            "cp": cp.array(groups_info.cumulative_pairs, dtype=cp.int64),
            "ctp": cp.array(compute_cumulative_tilepairs(groups_info.suffix_offsets), dtype=cp.int64),
            "tc": int(groups_info.total_candidates),
        }
        return out


def count_k3plus_per_candidate(bitvecs_gpu, groups_info, n_u64s, chunk_start=0, chunk_size=None, groups_gpu=None):
    """Dense K>=3 counting, one thread per candidate: support counts for a range.

    Unlike the tiled kernel it serves any candidate range, so it also counts a
    prefix group too large for one chunk.

    Takes pre-built groups_info from build_k3plus_groups_from_flat().
    No threshold filtering — outputs counts for every candidate in range.

    Supports candidate-range chunking: when chunk_start/chunk_size are set,
    only processes candidates [chunk_start, chunk_start + chunk_size).
    The CUDA kernel uses candidate_offset for chunk-relative output indexing:
    result_counts[cand_idx - candidate_offset] instead of result_counts[cand_idx].

    When groups_gpu is provided, skips group data upload (already resident).
    This is critical for K=8+: ~40 GB group data uploaded once, not per chunk.

    Memory: chunk_size × 4 bytes (int32, not total_candidates × 8 — counts
    are bounded by n_transactions, guarded to < 2^31 by the row-split caller,
    and the ≤8-GPU partial sum is bounded by the same n_transactions).

    Args:
        bitvecs_gpu: CuPy array of shape (n_cols, n_u64s).
        groups_info: K3PlusGroups namedtuple from build_k3plus_groups_from_flat().
        n_u64s: Number of uint64 words per bitvector.
        chunk_start: First candidate index to process (default: 0).
        chunk_size: Number of candidates to process (default: all).
        groups_gpu: Pre-uploaded group data dict from upload_k3plus_groups().
            If None, uploads fresh.

    Returns:
        CuPy int32 array of shape (chunk_size,) with counts — stays in VRAM.
    """
    from .shared_tiled import _assert_k_cap

    _assert_k_cap(groups_info)
    import cupy as cp

    tc = groups_info.total_candidates
    if chunk_size is None:
        chunk_size = tc - chunk_start

    _assert_bitvecs("count_k3plus_per_candidate", bitvecs_gpu, **({} if groups_gpu is None else {"groups_gpu": groups_gpu["gpi"]}))
    device_id = bitvecs_gpu.device.id
    if groups_gpu is None:
        groups_gpu = upload_k3plus_groups(groups_info, device_id)

    with cp.cuda.Device(device_id):
        result_counts = cp.zeros(chunk_size, dtype=cp.int32)

        kernel = get_cuda_kernel("count_k3plus_dense")
        block_size = 256
        grid = _grid_dims(chunk_size)

        kernel(
            grid,
            (block_size,),
            (
                bitvecs_gpu,
                groups_gpu["gpi"],
                groups_gpu["gpo"],
                groups_gpu["gs"],
                groups_gpu["gso"],
                groups_gpu["cp"],
                np.int64(n_u64s),
                np.int64(len(groups_info.cumulative_pairs) - 1),
                np.int64(chunk_start + chunk_size),
                result_counts,
                np.int64(chunk_start),
            ),
        )
        cp.cuda.Stream.null.synchronize()

    return result_counts  # stays in VRAM — no .get()




def build_k3plus_groups_from_flat(freq_flat):
    """Build prefix group arrays from flat (n_freq, k) numpy array.

    Uses Rust/Rayon parallel sort when available (10-100x faster, 9x less memory).
    Falls back to vectorized numpy if Rust extension not built (or is a stale
    wheel — a one-time warning names the rebuild command).

    Args:
        freq_flat: numpy int32 array of shape (n_freq, k).

    Returns:
        K3PlusGroups namedtuple or None if no valid groups.
    """
    n, k = freq_flat.shape
    if n < 2:
        return None

    # ── Rust fast path: Rayon parallel sort, GIL-free ──────────────
    try:
        from et_miner.backends import get_rust_ext

        et_miner_rust = get_rust_ext()
        if et_miner_rust is None:
            _warn_missing_rust_once("build_k3plus_groups_from_flat")
        if hasattr(et_miner_rust, "build_k3plus_groups_from_flat"):
            arr = np.ascontiguousarray(freq_flat, dtype=np.int32)
            try:
                result = et_miner_rust.build_k3plus_groups_from_flat(arr, False)
            except TypeError:  # pre-0.2.0 wheel: no with_src_rows argument
                result = None
                _warn_stale_rust_once("build_k3plus_groups_from_flat has no with_src_rows")
            else:
                if result is None:
                    return None
                if len(result) != 7:  # pre-0.2.0 wheel: 6-tuple
                    _warn_stale_rust_once("build_k3plus_groups_from_flat returned a 6-tuple")
                    result = None
            if result is not None:
                prefix_items, prefix_offsets, suffixes, suffix_offsets, cumulative_pairs, _src_rows, total = result
                return K3PlusGroups(
                    prefix_items=prefix_items,
                    prefix_offsets=prefix_offsets,
                    suffixes=suffixes,
                    suffix_offsets=suffix_offsets,
                    cumulative_pairs=cumulative_pairs,
                    total_candidates=int(total),
                    groups=None,
                )
    except Exception as exc:
        # (ImportError, Exception) was the same blanket catch written twice.
        # The catch stays -- the numpy fallback gives the same answer, so a
        # failing extension should not take a campaign down -- but it no
        # longer hides the failure that made it necessary.
        _warn_rust_failed("build_k3plus_groups_from_flat", exc)

    return _build_k3plus_groups_numpy(freq_flat)


def _build_k3plus_groups_numpy(freq_flat):
    """Vectorized numpy group builder (the no-Rust fallback; directly testable).

    Sorts by the FULL row (prefix columns, then suffix) like the Rust
    builder, so suffixes are ascending within every group.
    """
    n, k = freq_flat.shape
    if n < 2:
        return None
    prefixes = freq_flat[:, :-1]  # (n, k-1) — group key
    suffixes_col = freq_flat[:, -1]  # (n,) — suffix values

    # Lexicographic sort by prefix, then suffix. np.lexsort sorts by its LAST
    # key first, so the suffix goes first (least significant) and the prefix
    # columns follow in reverse order.
    sort_keys = (suffixes_col,) + tuple(prefixes[:, i] for i in range(prefixes.shape[1] - 1, -1, -1))
    order = np.lexsort(sort_keys)
    prefixes_sorted = prefixes[order]
    suffixes_sorted = suffixes_col[order]

    # Find group boundaries: where prefix changes
    diff = np.any(prefixes_sorted[1:] != prefixes_sorted[:-1], axis=1)
    boundary_mask = np.concatenate([[True], diff])
    boundaries = np.where(boundary_mask)[0]
    group_starts = boundaries
    group_ends = np.concatenate([boundaries[1:], [n]])
    group_sizes = group_ends - group_starts

    # Filter groups with >= 2 suffixes
    valid = group_sizes >= 2
    if not np.any(valid):
        return None

    valid_starts = group_starts[valid]
    valid_ends = group_ends[valid]
    valid_sizes = group_sizes[valid]
    n_groups = len(valid_starts)

    # Build flat prefix items array
    prefix_len = prefixes_sorted.shape[1]
    group_prefix_items = prefixes_sorted[valid_starts].ravel().astype(np.int32)
    group_prefix_offsets = np.arange(0, (n_groups + 1) * prefix_len, prefix_len, dtype=np.int64)

    # Build flat suffix array — vectorized gather, no Python loop
    total_suffixes = int(valid_sizes.sum())
    group_suffix_offsets = np.zeros(n_groups + 1, dtype=np.int64)
    np.cumsum(valid_sizes, out=group_suffix_offsets[1:])

    # Vectorized index construction: repeat group starts, add range offsets
    offsets_within = np.arange(total_suffixes, dtype=np.int64)
    group_ids = np.searchsorted(group_suffix_offsets[1:], offsets_within, side="right")
    src_indices = valid_starts[group_ids] + offsets_within - group_suffix_offsets[:-1][group_ids]
    group_suffixes = suffixes_sorted[src_indices].astype(np.int32)

    # Cumulative pairs
    pairs_per_group = valid_sizes * (valid_sizes - 1) // 2
    cumulative_pairs = np.zeros(n_groups + 1, dtype=np.int64)
    np.cumsum(pairs_per_group, out=cumulative_pairs[1:])
    total_candidates = int(cumulative_pairs[-1])

    if total_candidates == 0:
        return None

    return K3PlusGroups(
        prefix_items=group_prefix_items,
        prefix_offsets=group_prefix_offsets,
        suffixes=group_suffixes,
        suffix_offsets=group_suffix_offsets,  # int64 — prevents overflow at >2B suffixes
        cumulative_pairs=cumulative_pairs,
        total_candidates=total_candidates,
        groups=None,  # not needed for dense counting
    )
