"""Host-side level helpers for the row-split GPU miner.

The free-set (non-free) prune with its Rust fast path, the anchor output
selector and the sortedness check. Import-safe without CuPy or the Rust
extension.
"""

from __future__ import annotations

from loguru import logger


def _rust_prune_fn(et_miner_rust, name: str):
    """Resolve a free-set prune entry point, tolerating a pre-0.3.0 wheel.

    0.3.0 renamed ``prune_closed_flat*`` to ``prune_non_free_flat*`` — the test
    was always the free-set one, only the word was wrong. The old name is tried
    second so an unrebuilt wheel keeps working (identical semantics).
    """
    return getattr(et_miner_rust, name, None) or getattr(et_miner_rust, name.replace("non_free", "closed"), None)


def _rows_sorted(flat) -> bool:
    """True iff the rows of an (n, k) integer array are in STRICT lexicographic
    ascending order — vectorized O(n·k), no sort.

    Semantics: for every adjacent pair the first differing column must be
    ascending. Duplicate adjacent rows make ``argmax`` over ``a != b`` return
    column 0, where ``a < b`` is false, so duplicates count as UNSORTED and
    merely trigger a (harmless) redundant sort. Rows are unique itemsets, so
    this is by design, not a defect. Used to skip the level-end lexsort that
    the free-set-prune binary search requires when the level is already sorted.
    """
    import numpy as np

    n = len(flat)
    if n < 2:
        return True
    a = np.asarray(flat[:-1])
    b = np.asarray(flat[1:])
    neq = a != b
    first = neq.argmax(axis=1)
    rows = np.arange(n - 1)
    return bool(np.all(a[rows, first] < b[rows, first]))


def _prune_non_free_mask_python(current_flat, current_counts, prev_flat, prev_counts):
    """Pure-Python free-set keep-mask (True = keep). Dict-based, so it
    does not need prev_flat sorted."""
    import numpy as np

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
            f"    Free-set pruning: {n_before:,} → {n_after:,} ({100 * (1 - n_after / n_before):.1f}% non-free removed)"
        )
    return mask


def _prune_non_free_mask(current_flat, current_counts, prev_flat, prev_counts):
    """Free-set keep-mask of a level (True = keep).

    An itemset is a free-set (generator) when no (k-1)-subset has the same
    count [Bastide et al. 2000]. ``prev_flat``/``prev_counts`` must be the
    COMPLETE previous level, or the test misses subsets that were themselves
    pruned and under-prunes. A mask rather than filtered arrays, because the
    sparse-CSR path also carries the survivor index array and filters it in
    lockstep. Rust ``prune_non_free_flat`` when available (``prev_flat`` must be
    row-sorted), else the dict-based Python fallback.
    """
    import numpy as np

    n = len(current_flat)
    if prev_flat is None or prev_counts is None or len(prev_flat) == 0 or n == 0:
        return np.ones(n, dtype=bool)
    try:
        from et_miner.backends import get_rust_ext

        et_miner_rust = get_rust_ext()
        if et_miner_rust is None:
            raise ImportError("et_miner_rust not built")
        _mask_fn = _rust_prune_fn(et_miner_rust, "prune_non_free_flat")
        if _mask_fn is None:
            raise AttributeError("et_miner_rust exposes no free-set prune")
        mask = np.asarray(
            _mask_fn(
                np.ascontiguousarray(current_flat, dtype=np.int32),
                np.ascontiguousarray(current_counts, dtype=np.int64),
                np.ascontiguousarray(prev_flat, dtype=np.int32),
                np.ascontiguousarray(prev_counts, dtype=np.int64),
            ),
            dtype=bool,
        )
        n_after = int(mask.sum())
        if n > n_after:
            logger.debug(
                f"    Free-set pruning: {n:,} → {n_after:,} ({100 * (1 - n_after / n):.1f}% non-free removed) [rust-mask]"
            )
        return mask
    except (ImportError, AttributeError):
        pass
    return _prune_non_free_mask_python(current_flat, current_counts, prev_flat, prev_counts)


def _anchor_keep_mask(current_flat, anchor_col_arr, k):
    """Boolean keep-mask for itemsets containing >= 1 anchor column, or None
    when no filtering applies (no anchors, or nothing to filter).

    K=1 is masked like any other level. The `k < 2` early return this used to
    carry made a two-phase run emit every frequent item at K=1 while filtering
    every deeper level -- an inconsistency that only made sense while anchoring
    was a mining gate. As an output selector it is uniform, and free: the
    generation base is a different array.
    """
    import numpy as np

    if anchor_col_arr is None or len(current_flat) == 0:
        return None
    anchor_mask = np.zeros(len(current_flat), dtype=bool)
    for col in range(k):
        anchor_mask |= np.isin(current_flat[:, col], anchor_col_arr)
    return anchor_mask
