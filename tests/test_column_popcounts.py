"""K=1 support of the row-split miner: popcount a block of columns at a time.

A whole-matrix popcount allocates a uint64 temporary as large as the bitvec
matrix itself, so a matrix above half of free VRAM ran out of memory at K=1
(stress_k2's 8.75 GB matrix on a 12 GB card).
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("max_temp_bytes", [8, 8 * 53 * 3, 1 << 28])
def test_blocked_popcount_equals_the_direct_count(max_temp_bytes):
    from et_miner.gpu.kernels import column_popcounts

    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2**63, size=(37, 53), dtype=np.uint64)
    want = np.array([sum(bin(int(w)).count("1") for w in row) for row in bits], dtype=np.int64)
    got = column_popcounts(cp.asarray(bits), max_temp_bytes=max_temp_bytes).get()
    assert got.tolist() == want.tolist()


def test_the_temporary_is_bounded_by_the_block(monkeypatch):
    from et_miner.gpu.kernels import loader

    shapes = []
    real = loader.get_popcount_kernel

    def spy():
        kernel = real()

        def run(arr):
            shapes.append(arr.shape)
            return kernel(arr)

        return run

    monkeypatch.setattr(loader, "get_popcount_kernel", spy)
    bv = cp.full((10, 16), np.uint64(0xFFFFFFFFFFFFFFFF), dtype=cp.uint64)
    counts = loader.column_popcounts(bv, max_temp_bytes=3 * 16 * 8).get()
    assert counts.tolist() == [16 * 64] * 10
    assert [s[0] for s in shapes] == [3, 3, 3, 1]
