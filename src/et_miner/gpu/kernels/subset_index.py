"""Device index of a mining level for the counting kernels' (k-1)-subset test.

The K>=3 counting kernels (per-candidate, tiled dense and fused, sparse CSR)
take the previous level as sorted int32 rows with per-row counts and free
flags (``_src/_subset_index.cu``). With ``SUBSET_PRUNE`` a candidate with a
(k-1)-subset missing from the index is not counted and keeps a zero count;
with ``SUBSET_INFER`` as well, a candidate with a non-free (k-1)-subset gets
the minimum of its subset counts instead of a count. Candidate indices never
change, so chunk plans, the cross-GPU reduce and the decode are unaffected.

Usage:
    index = upload_subset_index(rows, counts, mode=SUBSET_PRUNE, device_id=0, write_inferred=True)
    count_k3plus_per_candidate(..., index=index)

Options:
    mode            SUBSET_PRUNE, optionally | SUBSET_INFER (needs ``free``)
    free            per-row free flags (bool), required by SUBSET_INFER
    write_inferred  whether this device writes inferred counts; exactly one
                    device of a reduce does, the others leave zeros
"""

from __future__ import annotations

import numpy as np

#: Skip candidates with a (k-1)-subset missing from the index.
SUBSET_PRUNE = 1
#: Report the count of candidates with a non-free (k-1)-subset (needs free flags).
SUBSET_INFER = 2

_DISABLED: dict[int, tuple] = {}


def index_nbytes(n_rows: int, width: int) -> int:
    """Device bytes of an index of ``n_rows`` itemsets of ``width`` items."""
    return int(n_rows) * (4 * int(width) + 4 + 1)


def _strictly_sorted(rows: np.ndarray) -> bool:
    if len(rows) < 2:
        return True
    a, b = rows[:-1], rows[1:]
    neq = a != b
    first = neq.argmax(axis=1)
    idx = np.arange(len(a))
    return bool(np.all(neq[idx, first] & (a[idx, first] < b[idx, first])))


def upload_subset_index(rows, counts, free=None, *, mode: int, device_id: int, write_inferred: bool) -> dict:
    """Upload one level's index to ``device_id``.

    Args:
        rows: (n, k-1) integer array of column ids, rows strictly ascending in
            lexicographic order (the kernels binary-search it).
        counts: (n,) global counts of the rows (below 2**31).
        free: (n,) bool, True where the row has no equal-count (k-2)-subset;
            required when ``mode`` has SUBSET_INFER.
        mode: SUBSET_PRUNE, optionally | SUBSET_INFER.
        device_id: CUDA device to upload to.
        write_inferred: whether this device writes inferred counts.

    Returns:
        The ``index=`` argument of the counting wrappers for this device.
    """
    import cupy as cp

    rows = np.ascontiguousarray(rows, dtype=np.int32)
    if rows.ndim != 2 or not mode & SUBSET_PRUNE:
        raise ValueError(f"an index needs (n, k-1) rows and SUBSET_PRUNE, got shape {rows.shape} and mode {mode}")
    if not _strictly_sorted(rows):
        raise ValueError("index rows must be unique and in ascending lexicographic order")
    counts = np.asarray(counts, dtype=np.int64)
    if len(counts) != len(rows) or (len(counts) and int(counts.max()) > np.iinfo(np.int32).max):
        raise ValueError("index counts must match the rows and fit int32")
    if mode & SUBSET_INFER:
        if free is None or len(free) != len(rows):
            raise ValueError("SUBSET_INFER needs one free flag per index row")
        free_u8 = np.asarray(free, dtype=np.uint8)
    else:
        free_u8 = np.ones(len(rows), dtype=np.uint8)
    with cp.cuda.Device(device_id):
        return {
            "rows": cp.asarray(rows if len(rows) else np.zeros((1, max(1, rows.shape[1])), np.int32)),
            "n": int(len(rows)),
            "counts": cp.asarray(counts.astype(np.int32) if len(rows) else np.zeros(1, np.int32)),
            "free": cp.asarray(free_u8 if len(rows) else np.ones(1, np.uint8)),
            "mode": int(mode),
            "write_inferred": int(bool(write_inferred)),
            "device_id": int(device_id),
        }


def kernel_args(index, device_id: int) -> tuple:
    """The kernels' trailing index arguments; mode 0 (count everything) when ``index`` is None."""
    if index is None:
        if device_id not in _DISABLED:
            import cupy as cp

            with cp.cuda.Device(device_id):
                _DISABLED[device_id] = (
                    cp.zeros(1, dtype=cp.int32),
                    np.int64(0),
                    cp.zeros(1, dtype=cp.int32),
                    cp.zeros(1, dtype=cp.uint8),
                    np.int32(0),
                    np.int32(0),
                )
        return _DISABLED[device_id]
    if index["device_id"] != device_id:
        raise ValueError(f"subset index lives on device {index['device_id']}, the launch on device {device_id}")
    return (
        index["rows"],
        np.int64(index["n"]),
        index["counts"],
        index["free"],
        np.int32(index["mode"]),
        np.int32(index["write_inferred"]),
    )
