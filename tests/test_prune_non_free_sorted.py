"""CPU tests for the free-set prune's sortedness invariant.

The Rust free-set prune binary-searches the previous level, so that table must
be lexicographically sorted by row. `_rows_sorted` lets the miners skip the
level-end sort when a level is already sorted, and the Rust pyfunctions refuse
an unsorted table instead of silently under-pruning.
"""

from __future__ import annotations

import numpy as np
import pytest

from et_miner.backends import get_rust_ext
from et_miner.gpu.mining import _prune_non_free_flat, _rust_prune_fn, _rows_sorted


def _fresh_rust():
    ext = get_rust_ext()
    if ext is None:
        return None
    version = tuple(int(x) for x in str(getattr(ext, "__version__", "0.0.0")).split(".")[:2])
    return ext if version >= (0, 2) else None


def _prune_fns(ext):
    """The mask and compact entry points under whichever name the wheel has —
    0.3.0 renamed prune_closed_flat* to prune_non_free_flat*."""
    return _rust_prune_fn(ext, "prune_non_free_flat"), _rust_prune_fn(ext, "prune_non_free_flat_compact")


def test_rows_sorted_semantics():
    assert _rows_sorted(np.empty((0, 3), np.int32))
    assert _rows_sorted(np.array([[1, 2, 3]], np.int32))
    assert _rows_sorted(np.array([[1, 2], [1, 3], [2, 0]], np.int32))
    assert _rows_sorted(np.array([[5], [6], [9]], np.int32))
    assert not _rows_sorted(np.array([[1, 3], [1, 2]], np.int32))
    assert not _rows_sorted(np.array([[2, 0], [1, 9]], np.int32))
    # j-major group order (groups of >= 4 suffixes) is NOT lex order
    assert not _rows_sorted(np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [0, 1, 4]], np.int32))
    # strict: duplicate adjacent rows count as unsorted by design (rows are unique itemsets)
    assert not _rows_sorted(np.array([[1, 2], [1, 2]], np.int32))


def test_rows_sorted_agrees_with_lexsort_on_random_tables():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n, k = int(rng.integers(1, 40)), int(rng.integers(1, 5))
        flat = rng.integers(0, 6, size=(n, k)).astype(np.int32)
        flat = np.unique(flat, axis=0)  # unique rows, sorted
        assert _rows_sorted(flat)
        if len(flat) > 1:
            perm = rng.permutation(len(flat))
            shuffled = flat[perm]
            expected = bool(np.array_equal(shuffled, flat))
            assert _rows_sorted(shuffled) == expected


@pytest.mark.skipif(_fresh_rust() is None, reason="et_miner_rust >= 0.2.0 not built")
def test_rust_free_set_prune_rejects_unsorted_prev():
    ext = _fresh_rust()
    mask_fn, compact_fn = _prune_fns(ext)
    cur = np.array([[0, 1, 4]], np.int32)
    cc = np.array([5], np.int64)
    prev_unsorted = np.array([[0, 1], [0, 2], [1, 2], [0, 3], [0, 4]], np.int32)  # j-major order
    pc = np.array([5, 5, 5, 5, 5], np.int64)
    with pytest.raises(ValueError, match="sorted"):
        mask_fn(cur, cc, prev_unsorted, pc)
    with pytest.raises(ValueError, match="sorted"):
        compact_fn(cur, cc, prev_unsorted, pc)

    order = np.lexsort(prev_unsorted[:, ::-1].T)
    prev_sorted, pc_sorted = prev_unsorted[order], pc[order]
    # (0,1,4) has subset (0,1) with the same count -> not free -> pruned
    assert mask_fn(cur, cc, prev_sorted, pc_sorted).tolist() == [False]
    flat_1d, counts, n_kept = compact_fn(cur, cc, prev_sorted, pc_sorted)
    assert n_kept == 0 and len(counts) == 0

    # the Python wrapper goes through the compact path and keeps the free rows
    cur2 = np.array([[0, 1, 4], [0, 2, 4]], np.int32)
    cc2 = np.array([5, 3], np.int64)
    kept_flat, kept_counts = _prune_non_free_flat(cur2, cc2, prev_sorted, pc_sorted)
    assert kept_flat.tolist() == [[0, 2, 4]] and kept_counts.tolist() == [3]
