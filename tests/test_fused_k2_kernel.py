"""Tests for fused k=2 CUDA kernel.

Verifies that the single-launch pair generation + AND + popcount + filter
kernel produces identical results to the Python-based candidate pipeline.

Skip automatically if CuPy is not available.
"""

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

# Belt for cupy-installed-but-no-device boxes: auto-skipped via the gpu mark.
pytestmark = pytest.mark.gpu

from et_miner.gpu.kernels import (
    count_pairs_fused_k2,
    count_itemsets_cuda,
    get_cuda_kernel,
)


def _make_bitvecs(n_cols, n_rows, density=0.3, seed=42):
    """Create random bitvec matrix on GPU for testing.

    Returns (bitvecs_gpu, n_u64s) where bitvecs_gpu has shape (n_cols, n_u64s).
    """
    rng = np.random.RandomState(seed)
    n_u64s = (n_rows + 63) // 64

    # Build packed bitvectors
    bitvecs = np.zeros((n_cols, n_u64s), dtype=np.uint64)
    for col in range(n_cols):
        bits = rng.random(n_rows) < density
        for row in range(n_rows):
            if bits[row]:
                word = row // 64
                bit = row % 64
                bitvecs[col, word] |= np.uint64(1) << np.uint64(bit)

    return cupy.array(bitvecs), n_u64s, n_rows


def _python_k2_reference(bitvecs_gpu, freq_cols, n_u64s, min_count):
    """Reference implementation: Python candidate gen + batch CUDA count."""
    n_freq = len(freq_cols)
    if n_freq < 2:
        return [], np.array([], dtype=np.int64)

    # Generate all pairs (Python path)
    candidates = []
    for i, col_a in enumerate(freq_cols):
        for col_b in freq_cols[i + 1:]:
            candidates.append((col_a, col_b))

    if not candidates:
        return [], np.array([], dtype=np.int64)

    # Count via existing batch kernel
    itemsets_np = [np.array(list(c), dtype=np.int32) for c in candidates]
    counts = count_itemsets_cuda(bitvecs_gpu, itemsets_np, use_batch=True)

    # Filter
    pairs = []
    filtered_counts = []
    for idx, (col_a, col_b) in enumerate(candidates):
        if counts[idx] >= min_count:
            pairs.append((col_a, col_b))
            filtered_counts.append(counts[idx])

    return pairs, np.array(filtered_counts, dtype=np.int64)


class TestFusedK2Kernel:
    """Test the fused k=2 pair counting kernel."""

    def test_kernel_compiles(self):
        """Verify the fused k=2 kernel compiles without errors."""
        kernel = get_cuda_kernel('count_pairs_fused_k2')
        assert kernel is not None

    def test_small_known_answer(self):
        """Known-answer test with 4 items and predictable bitvectors."""
        # 4 columns, 128 rows. Set specific bit patterns.
        n_rows = 128
        n_u64s = 2  # 128 bits = 2 uint64s
        bitvecs = np.zeros((4, 2), dtype=np.uint64)

        # Col 0: rows 0-63 all set (64 bits)
        bitvecs[0, 0] = np.uint64(0xFFFFFFFFFFFFFFFF)
        # Col 1: rows 0-63 all set (64 bits) -> AND with col0 = 64
        bitvecs[1, 0] = np.uint64(0xFFFFFFFFFFFFFFFF)
        # Col 2: rows 0-31 set (32 bits) -> AND with col0 = 32, AND with col1 = 32
        bitvecs[2, 0] = np.uint64(0x00000000FFFFFFFF)
        # Col 3: no bits set -> AND with anything = 0
        bitvecs[3, 0] = np.uint64(0)

        bitvecs_gpu = cupy.array(bitvecs)
        freq_cols = [0, 1, 2, 3]

        # min_count = 30: should find (0,1)=64, (0,2)=32, (1,2)=32
        pairs, counts = count_pairs_fused_k2(bitvecs_gpu, freq_cols, n_u64s, min_count=30)

        # Sort for comparison
        pair_count_map = {p: int(c) for p, c in zip(pairs, counts)}

        assert (0, 1) in pair_count_map
        assert pair_count_map[(0, 1)] == 64
        assert (0, 2) in pair_count_map
        assert pair_count_map[(0, 2)] == 32
        assert (1, 2) in pair_count_map
        assert pair_count_map[(1, 2)] == 32

        # Col 3 has 0 bits, so no pair with col3 should pass min_count=30
        for p in pairs:
            assert 3 not in p

    def test_matches_python_reference(self):
        """Fused kernel output must exactly match Python reference for random data."""
        bitvecs_gpu, n_u64s, n_rows = _make_bitvecs(n_cols=20, n_rows=1000, density=0.3)
        freq_cols = list(range(20))
        min_count = 50

        fused_pairs, fused_counts = count_pairs_fused_k2(
            bitvecs_gpu, freq_cols, n_u64s, min_count
        )
        ref_pairs, ref_counts = _python_k2_reference(
            bitvecs_gpu, freq_cols, n_u64s, min_count
        )

        # Convert to sorted sets for comparison
        fused_set = {(p, int(c)) for p, c in zip(fused_pairs, fused_counts)}
        ref_set = {(p, int(c)) for p, c in zip(ref_pairs, ref_counts)}

        assert fused_set == ref_set, (
            f"Mismatch: fused has {len(fused_set)} pairs, ref has {len(ref_set)} pairs. "
            f"Extra in fused: {fused_set - ref_set}, Missing from fused: {ref_set - fused_set}"
        )

    def test_larger_scale_matches(self):
        """Test with more items (100 cols) to stress triangular indexing."""
        bitvecs_gpu, n_u64s, n_rows = _make_bitvecs(n_cols=100, n_rows=5000, density=0.2)
        freq_cols = list(range(100))
        min_count = 200

        fused_pairs, fused_counts = count_pairs_fused_k2(
            bitvecs_gpu, freq_cols, n_u64s, min_count
        )
        ref_pairs, ref_counts = _python_k2_reference(
            bitvecs_gpu, freq_cols, n_u64s, min_count
        )

        fused_set = {(p, int(c)) for p, c in zip(fused_pairs, fused_counts)}
        ref_set = {(p, int(c)) for p, c in zip(ref_pairs, ref_counts)}
        assert fused_set == ref_set

    def test_zero_frequent_items(self):
        """Edge case: 0 frequent items should return empty."""
        bitvecs_gpu, n_u64s, _ = _make_bitvecs(n_cols=5, n_rows=64)
        pairs, counts = count_pairs_fused_k2(bitvecs_gpu, [], n_u64s, min_count=1)
        assert pairs == []
        assert len(counts) == 0

    def test_one_frequent_item(self):
        """Edge case: 1 frequent item -> 0 pairs."""
        bitvecs_gpu, n_u64s, _ = _make_bitvecs(n_cols=5, n_rows=64)
        pairs, counts = count_pairs_fused_k2(bitvecs_gpu, [0], n_u64s, min_count=1)
        assert pairs == []
        assert len(counts) == 0

    def test_high_min_count_filters_all(self):
        """If min_count is higher than n_rows, no pairs should be found."""
        bitvecs_gpu, n_u64s, n_rows = _make_bitvecs(n_cols=10, n_rows=100, density=0.5)
        freq_cols = list(range(10))
        # min_count > n_rows means impossible to satisfy
        pairs, counts = count_pairs_fused_k2(
            bitvecs_gpu, freq_cols, n_u64s, min_count=n_rows + 1
        )
        assert pairs == []
        assert len(counts) == 0

    def test_subset_of_columns(self):
        """Only a subset of columns are frequent — kernel should only process those."""
        bitvecs_gpu, n_u64s, _ = _make_bitvecs(n_cols=50, n_rows=2000, density=0.4)
        # Only columns 10, 20, 30, 40 are "frequent"
        freq_cols = [10, 20, 30, 40]
        min_count = 100

        fused_pairs, fused_counts = count_pairs_fused_k2(
            bitvecs_gpu, freq_cols, n_u64s, min_count
        )
        ref_pairs, ref_counts = _python_k2_reference(
            bitvecs_gpu, freq_cols, n_u64s, min_count
        )

        fused_set = {(p, int(c)) for p, c in zip(fused_pairs, fused_counts)}
        ref_set = {(p, int(c)) for p, c in zip(ref_pairs, ref_counts)}
        assert fused_set == ref_set

    def test_many_u64_words(self):
        """Test with large n_u64s to verify word-parallel reduction works."""
        # 100K rows = 1563 u64s per column — stresses the thread-parallel popcount
        bitvecs_gpu, n_u64s, _ = _make_bitvecs(n_cols=10, n_rows=100_000, density=0.1)
        freq_cols = list(range(10))
        min_count = 500

        fused_pairs, fused_counts = count_pairs_fused_k2(
            bitvecs_gpu, freq_cols, n_u64s, min_count
        )
        ref_pairs, ref_counts = _python_k2_reference(
            bitvecs_gpu, freq_cols, n_u64s, min_count
        )

        fused_set = {(p, int(c)) for p, c in zip(fused_pairs, fused_counts)}
        ref_set = {(p, int(c)) for p, c in zip(ref_pairs, ref_counts)}
        assert fused_set == ref_set
