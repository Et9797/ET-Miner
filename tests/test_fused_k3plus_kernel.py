"""Tests for fused k>=3 CUDA kernel.

Verifies that the single-launch candidate counting + in-kernel filtering
produces identical results to the legacy count_itemsets_cuda + Python filter.

Skip automatically if CuPy is not available.
"""

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

from et_miner.cuda_kernels import (
    count_itemsets_fused_k3plus,
    count_itemsets_fused_k3plus_multi_gpu,
    count_itemsets_cuda,
    get_cuda_kernel,
    clear_kernel_cache,
)


def _make_bitvecs(n_cols, n_rows, density=0.3, seed=42):
    """Create random bitvec matrix on GPU for testing."""
    rng = np.random.RandomState(seed)
    n_u64s = (n_rows + 63) // 64
    bitvecs_np = np.zeros((n_cols, n_u64s), dtype=np.uint64)
    for col in range(n_cols):
        bits = rng.random(n_rows) < density
        for i, bit in enumerate(bits):
            if bit:
                word_idx = i // 64
                bit_idx = i % 64
                bitvecs_np[col, word_idx] |= np.uint64(1 << bit_idx)
    return cupy.array(bitvecs_np, dtype=cupy.uint64), n_u64s


def _legacy_count(bitvecs_gpu, candidates, min_count):
    """Count using legacy approach for reference."""
    itemsets_np = [np.array(list(c), dtype=np.int32) for c in candidates]
    counts = count_itemsets_cuda(bitvecs_gpu, itemsets_np, use_batch=True)
    freq = []
    freq_counts = []
    for idx, c in enumerate(candidates):
        if counts[idx] >= min_count:
            freq.append(c)
            freq_counts.append(int(counts[idx]))
    return freq, freq_counts


class TestFusedK3PlusKernel:
    """Tests for the fused k>=3 CUDA kernel."""

    def test_kernel_compiles(self):
        """Verify kernel compiles without errors."""
        clear_kernel_cache()
        kernel = get_cuda_kernel('count_itemsets_fused_k3plus')
        assert kernel is not None

    def test_small_known_answer_k3(self):
        """k=3 on small data: fused matches legacy."""
        bitvecs_gpu, n_u64s = _make_bitvecs(20, 100, density=0.5, seed=1)
        candidates = [(0, 1, 2), (3, 4, 5), (0, 5, 10), (1, 2, 3)]
        min_count = 1

        freq_new, counts_new = count_itemsets_fused_k3plus(
            bitvecs_gpu, candidates, n_u64s, min_count
        )
        freq_old, counts_old = _legacy_count(bitvecs_gpu, candidates, min_count)

        assert set(freq_new) == set(freq_old)

    def test_matches_legacy_k3(self):
        """k=3 with many candidates: fused matches legacy exactly."""
        bitvecs_gpu, n_u64s = _make_bitvecs(30, 200, density=0.4, seed=42)
        candidates = []
        for i in range(15):
            for j in range(i + 1, 20):
                for m in range(j + 1, 25):
                    candidates.append((i, j, m))

        for min_count in [1, 20, 50, 100]:
            freq_new, counts_new = count_itemsets_fused_k3plus(
                bitvecs_gpu, candidates, n_u64s, min_count
            )
            freq_old, counts_old = _legacy_count(bitvecs_gpu, candidates, min_count)
            assert set(freq_new) == set(freq_old), f"Mismatch at min_count={min_count}"

    def test_matches_legacy_k4(self):
        """k=4 candidates: fused matches legacy."""
        bitvecs_gpu, n_u64s = _make_bitvecs(20, 100, density=0.5, seed=7)
        candidates = []
        for i in range(8):
            for j in range(i + 1, 10):
                for m in range(j + 1, 12):
                    for n in range(m + 1, 14):
                        candidates.append((i, j, m, n))

        min_count = 10
        freq_new, counts_new = count_itemsets_fused_k3plus(
            bitvecs_gpu, candidates, n_u64s, min_count
        )
        freq_old, counts_old = _legacy_count(bitvecs_gpu, candidates, min_count)
        assert set(freq_new) == set(freq_old)

    def test_counts_match_exactly(self):
        """Verify support counts match legacy, not just sets."""
        bitvecs_gpu, n_u64s = _make_bitvecs(20, 200, density=0.5, seed=99)
        candidates = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6)]
        min_count = 1

        freq_new, counts_new = count_itemsets_fused_k3plus(
            bitvecs_gpu, candidates, n_u64s, min_count
        )
        itemsets_np = [np.array(list(c), dtype=np.int32) for c in candidates]
        legacy_counts = count_itemsets_cuda(bitvecs_gpu, itemsets_np, use_batch=True)

        # Build count dict for new results
        new_dict = {c: int(counts_new[i]) for i, c in enumerate(freq_new)}

        for idx, c in enumerate(candidates):
            if legacy_counts[idx] >= min_count:
                assert c in new_dict, f"{c} missing from fused output"
                assert new_dict[c] == int(legacy_counts[idx]), (
                    f"Count mismatch for {c}: fused={new_dict[c]}, legacy={legacy_counts[idx]}"
                )

    def test_empty_candidates(self):
        """Empty candidates returns empty results."""
        bitvecs_gpu, n_u64s = _make_bitvecs(10, 100, seed=1)
        freq, counts = count_itemsets_fused_k3plus(bitvecs_gpu, [], n_u64s, 1)
        assert len(freq) == 0
        assert len(counts) == 0

    def test_high_threshold_filters_all(self):
        """Very high min_count filters everything."""
        bitvecs_gpu, n_u64s = _make_bitvecs(10, 100, density=0.1, seed=1)
        candidates = [(0, 1, 2), (3, 4, 5)]
        freq, counts = count_itemsets_fused_k3plus(
            bitvecs_gpu, candidates, n_u64s, 999999
        )
        assert len(freq) == 0

    def test_single_candidate(self):
        """Single candidate works correctly."""
        bitvecs_gpu, n_u64s = _make_bitvecs(10, 100, density=0.5, seed=1)
        candidates = [(0, 1, 2)]
        freq, counts = count_itemsets_fused_k3plus(
            bitvecs_gpu, candidates, n_u64s, 1
        )
        assert len(freq) <= 1

    def test_larger_scale(self):
        """Larger scale: 5000+ candidates, verify correctness."""
        bitvecs_gpu, n_u64s = _make_bitvecs(50, 500, density=0.3, seed=42)
        candidates = []
        for i in range(20):
            for j in range(i + 1, 30):
                for m in range(j + 1, 40):
                    candidates.append((i, j, m))

        min_count = 50
        freq_new, _ = count_itemsets_fused_k3plus(
            bitvecs_gpu, candidates, n_u64s, min_count
        )
        freq_old, _ = _legacy_count(bitvecs_gpu, candidates, min_count)
        assert set(freq_new) == set(freq_old)


class TestFusedK3PlusMultiGpu:
    """Tests for multi-GPU variant (runs on single GPU if only 1 available)."""

    def test_matches_single_gpu(self):
        """Multi-GPU gives same results as single-GPU."""
        bitvecs_gpu, n_u64s = _make_bitvecs(30, 200, density=0.4, seed=42)
        candidates = []
        for i in range(10):
            for j in range(i + 1, 15):
                for m in range(j + 1, 20):
                    candidates.append((i, j, m))

        min_count = 20
        freq_single, counts_single = count_itemsets_fused_k3plus(
            bitvecs_gpu, candidates, n_u64s, min_count
        )
        n_gpus = cupy.cuda.runtime.getDeviceCount()
        freq_multi, counts_multi = count_itemsets_fused_k3plus_multi_gpu(
            bitvecs_gpu, candidates, n_u64s, min_count, n_gpus
        )
        assert set(freq_single) == set(freq_multi)

    def test_empty_candidates(self):
        """Multi-GPU handles empty candidates."""
        bitvecs_gpu, n_u64s = _make_bitvecs(10, 100, seed=1)
        freq, counts = count_itemsets_fused_k3plus_multi_gpu(
            bitvecs_gpu, [], n_u64s, 1, 4
        )
        assert len(freq) == 0
