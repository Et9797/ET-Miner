"""#30 -- the row-split miner claimed to free VRAM it does not own.

Two sites, one rule. `_apriori_from_bitvecs` did `del bitvecs_gpu` on its own
PARAMETER and logged "Freed bitvec VRAM"; `_apriori_row_split_multi_gpu` did
`del bv` on a loop target while the list still held the array, called
`free_all_blocks()` before dropping the references, mutated a caller-supplied
list, and logged "Freed bitvec VRAM across all GPUs".

Neither freed anything, and on the route `core/apriori.py:600` takes the arrays
belong to the caller regardless -- it forwards the user's own `bitvecs=` array.

This is a logging and ownership fix, NOT a memory-behaviour fix: the
`free_all_blocks()` call is not scoped to allocations made in these functions,
and the sibling `try/finally` around the K>=3 group arrays depends on it. So
the test pinned that the call was still made on both branches, and that only
the claim differed.

The release sat in the dense-to-sparse transition, which went with ESCO
(route consolidation G6): the route now releases no bitvecs, owned or not.
What stays pinned is the contract the fix was about: the caller's array is
forwarded by identity and comes back unchanged, and a caller's list is not
mutated.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cupy")


def _gpu_count() -> int:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def _dense_csr(n_rows=4000, n_cols=14, seed=5):
    """Dense enough that K>=3 levels exist, small enough to be quick."""
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for r in range(n_rows):
        picked = rng.choice(n_cols, size=int(rng.integers(4, 8)), replace=False)
        if 3 in picked:
            picked = np.append(picked, 2)  # nested vocabulary
        for c in sorted(set(picked.tolist())):
            rows.append(r)
            cols.append(c)
    data = np.ones(len(rows), dtype=np.int8)
    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))


@pytest.mark.gpu
class TestBorrowedBitvecsAreNotFreed:
    def test_route_is_the_one_this_fix_touches(self, monkeypatch):
        """Assert the route, do not reason about it.

        `apriori(bitvecs=...)` reaches core/apriori.py:600 only when
        `_route_for_pruning` holds (:547); otherwise it lands on the
        single-GPU bitvec routes at :616/:628, which this commit does not
        touch. test_single_gpu_legacy_matches_oracle is the standing example of
        a test that claimed a route it did not exercise.
        """
        import cupy as cp

        from et_miner import apriori
        from et_miner.gpu.bitvec import _build_gpu_bitvec_matrix

        csr = _dense_csr()
        bitvecs = _build_gpu_bitvec_matrix(csr)
        col_to_item = {c: c + 100 for c in range(csr.shape[1])}

        import et_miner.gpu.row_split as rs

        seen = {}
        real_fn = rs._apriori_row_split_multi_gpu

        def _spy(*args, **kwargs):
            # Snapshot INSIDE the call. Capturing the list and inspecting it
            # afterwards would make this test fail for the list-mutation defect
            # too, and a test should fail only for the reason it is named after.
            seen["called"] = True
            bl = kwargs.get("bitvecs_list")
            seen["n"] = None if bl is None else len(bl)
            seen["first"] = None if not bl else bl[0][0]
            return real_fn(*args, **kwargs)

        monkeypatch.setattr(rs, "_apriori_row_split_multi_gpu", _spy)

        apriori(
            None,
            min_support=0.05,
            bitvecs=(bitvecs, col_to_item, csr.shape[0]),
            use_gpu=True,
            prune_equal_support=True,
        )

        assert seen.get("called"), "expected the row-split route (core/apriori.py:600)"
        assert seen["n"] == 1
        assert seen["first"] is bitvecs, "the route must forward the CALLER'S array, by identity"
        assert cp is not None

    def test_callers_array_survives_the_run(self, monkeypatch):
        """The array must still be usable, with the same contents, after a run
        that used to log that it had been freed."""
        import cupy as cp

        from et_miner import apriori
        from et_miner.gpu.bitvec import _build_gpu_bitvec_matrix

        csr = _dense_csr()
        bitvecs = _build_gpu_bitvec_matrix(csr)
        col_to_item = {c: c + 100 for c in range(csr.shape[1])}
        before = int(cp.sum(cp.unpackbits(bitvecs.view(cp.uint8))))

        apriori(
            None,
            min_support=0.05,
            bitvecs=(bitvecs, col_to_item, csr.shape[0]),
            use_gpu=True,
            prune_equal_support=True,
        )

        after = int(cp.sum(cp.unpackbits(bitvecs.view(cp.uint8))))
        assert after == before, "the caller's bitvecs were mutated or released"
        assert int(bitvecs.sum()) > 0

    def test_caller_list_is_not_mutated(self):
        """CONTROL: pre-fix this list was emptied by .clear() at the transition."""
        from et_miner.gpu.bitvec import _build_gpu_bitvec_matrix
        from et_miner.gpu.row_split import _apriori_row_split_multi_gpu

        csr = _dense_csr()
        bitvecs = _build_gpu_bitvec_matrix(csr)
        col_to_item = {c: c + 100 for c in range(csr.shape[1])}
        caller_list = [(bitvecs, int(bitvecs.device.id), csr.shape[0])]

        _apriori_row_split_multi_gpu(
            None,
            col_to_item,
            csr.shape[0],
            0.05,
            None,
            1,
            bitvecs_list=caller_list,
            prune_non_free=True,
        )

        assert len(caller_list) == 1, "a caller-supplied list must not be mutated"
        assert caller_list[0][0] is bitvecs
