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
the test pins that the call is still made on both branches, and that only the
claim differs. A "cleanup" that quietly changed VRAM pressure would be a
different commit with a VRAM measurement behind it.
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


class _PoolSpy:
    """Delegates to the real pool, counts free_all_blocks() calls.

    Wrapping rather than replacing: other code on this path legitimately uses
    the pool, and a stub that answered differently would change what is being
    measured.
    """

    def __init__(self, real):
        self._real = real
        self.free_calls = 0

    def free_all_blocks(self, *a, **kw):
        self.free_calls += 1
        return self._real.free_all_blocks(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._real, name)


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
            sparse_from_k=3,
        )

        assert seen.get("called"), "expected the row-split route (core/apriori.py:600)"
        assert seen["n"] == 1
        assert seen["first"] is bitvecs, "the route must forward the CALLER'S array, by identity"
        assert cp is not None

    def test_callers_array_survives_the_density_transition(self, monkeypatch):
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
            sparse_from_k=3,
        )

        after = int(cp.sum(cp.unpackbits(bitvecs.view(cp.uint8))))
        assert after == before, "the caller's bitvecs were mutated or released"
        assert int(bitvecs.sum()) > 0

    def test_caller_list_is_not_mutated_and_the_claim_is_honest(self, monkeypatch, caplog):
        """CONTROL: pre-fix this list is emptied by .clear() and the log claims
        the VRAM was freed."""
        import cupy as cp
        from loguru import logger

        from et_miner.gpu.bitvec import _build_gpu_bitvec_matrix
        from et_miner.gpu.row_split import _apriori_row_split_multi_gpu

        csr = _dense_csr()
        bitvecs = _build_gpu_bitvec_matrix(csr)
        col_to_item = {c: c + 100 for c in range(csr.shape[1])}
        caller_list = [(bitvecs, int(bitvecs.device.id), csr.shape[0])]

        real_pool = cp.get_default_memory_pool()
        spy = _PoolSpy(real_pool)
        monkeypatch.setattr(cp, "get_default_memory_pool", lambda: spy)

        messages: list[str] = []
        sink = logger.add(lambda m: messages.append(str(m)), level="DEBUG")
        try:
            _apriori_row_split_multi_gpu(
                None,
                col_to_item,
                csr.shape[0],
                0.05,
                None,
                1,
                bitvecs_list=caller_list,
                prune_non_free=True,
                prune_apriori=True,
                sparse_from_k=3,
            )
        finally:
            logger.remove(sink)

        joined = "".join(messages)
        assert "DENSITY TRANSITION" in joined, "fixture must reach the transition"

        assert len(caller_list) == 1, "a caller-supplied list must not be mutated"
        assert caller_list[0][0] is bitvecs

        assert "caller-owned and were not released" in joined
        assert "Freed bitvec VRAM across all GPUs" not in joined

        # The pool call is NOT removed -- see the module docstring.
        assert spy.free_calls > 0, "free_all_blocks() must still run on the borrowed branch"


@pytest.mark.gpu
def test_owned_bitvecs_are_freed_and_say_so(monkeypatch):
    """The internally-built path does own the arrays, and its message is the
    one that is now true."""
    import cupy as cp
    from loguru import logger

    from et_miner.gpu.row_split import _apriori_row_split_multi_gpu

    csr = _dense_csr()
    col_to_item = {c: c + 100 for c in range(csr.shape[1])}

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="DEBUG")
    try:
        _apriori_row_split_multi_gpu(
            csr,
            col_to_item,
            csr.shape[0],
            0.05,
            None,
            1,
            prune_non_free=True,
            prune_apriori=True,
            sparse_from_k=3,
        )
    finally:
        logger.remove(sink)

    joined = "".join(messages)
    assert "DENSITY TRANSITION" in joined
    assert "Freed bitvec VRAM across all GPUs" in joined
    assert "caller-owned" not in joined
    assert cp is not None
