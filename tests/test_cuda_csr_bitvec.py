"""Tests for CUDA CSR→bitvec kernel.

These tests verify the GPU kernel correctly converts CSR format to
column-oriented bitvectors.

Skip automatically if CuPy is not available.
"""

import numpy as np
import pytest

# Skip all tests if CuPy not available
cupy = pytest.importorskip("cupy")

# Belt for cupy-installed-but-no-device boxes: auto-skipped via the gpu mark.
pytestmark = pytest.mark.gpu


from et_miner.gpu.csr_bitvec import (
    build_bitvecs_gpu,
    build_bitvecs_gpu_from_scipy,
    get_csr_to_bitvec_kernel,
    clear_csr_kernel_cache,
)


class TestCSRToBitvecKernel:
    """Test the CUDA CSR→bitvec conversion kernel."""

    def test_kernel_compiles(self):
        """Verify the kernel compiles without errors."""
        kernel = get_csr_to_bitvec_kernel()
        assert kernel is not None

    def test_kernel_caching(self):
        """Verify kernel caching works correctly."""
        clear_csr_kernel_cache()
        kernel1 = get_csr_to_bitvec_kernel()
        kernel2 = get_csr_to_bitvec_kernel()
        assert kernel1 is kernel2  # Same object

        clear_csr_kernel_cache()
        kernel3 = get_csr_to_bitvec_kernel()
        # After clearing, should be new compilation (but functionally equivalent)
        assert kernel3 is not kernel1

    def test_simple_conversion(self):
        """Test basic CSR→bitvec conversion with small data."""
        # CSR for 4 transactions, 3 items:
        # Row 0: items [0, 1]
        # Row 1: items [1, 2]
        # Row 2: items [0, 2]
        # Row 3: items [0, 1, 2]
        indptr = np.array([0, 2, 4, 6, 9], dtype=np.int64)
        indices = np.array([0, 1, 1, 2, 0, 2, 0, 1, 2], dtype=np.int64)
        n_rows = 4
        n_cols = 3

        bitvecs = build_bitvecs_gpu(indptr, indices, n_rows, n_cols)

        # Verify shape: [n_cols, n_u64s] where n_u64s = ceil(4/64) = 1
        assert bitvecs.shape == (3, 1)

        # Convert to CPU for checking
        bitvecs_cpu = cupy.asnumpy(bitvecs)

        # Item 0 appears in rows 0, 2, 3 → bits 0, 2, 3 set → 0b1101 = 13
        assert bitvecs_cpu[0, 0] == 13, f"Expected 13, got {bitvecs_cpu[0, 0]}"

        # Item 1 appears in rows 0, 1, 3 → bits 0, 1, 3 set → 0b1011 = 11
        assert bitvecs_cpu[1, 0] == 11, f"Expected 11, got {bitvecs_cpu[1, 0]}"

        # Item 2 appears in rows 1, 2, 3 → bits 1, 2, 3 set → 0b1110 = 14
        assert bitvecs_cpu[2, 0] == 14, f"Expected 14, got {bitvecs_cpu[2, 0]}"

    def test_empty_matrix(self):
        """Test edge case: no transactions."""
        indptr = np.array([0], dtype=np.int64)
        indices = np.array([], dtype=np.int64)

        bitvecs = build_bitvecs_gpu(indptr, indices, n_rows=0, n_cols=5)

        # Shape should be [5, 0] or [5, 1] depending on ceil(0/64)
        # ceil(0/64) = 0, but we might want at least 1 for safety
        # Actually (0 + 63) // 64 = 0, so shape is [5, 0]
        assert bitvecs.shape[0] == 5

    def test_single_transaction(self):
        """Test edge case: single transaction."""
        # One transaction with items [0, 2]
        indptr = np.array([0, 2], dtype=np.int64)
        indices = np.array([0, 2], dtype=np.int64)

        bitvecs = build_bitvecs_gpu(indptr, indices, n_rows=1, n_cols=4)

        bitvecs_cpu = cupy.asnumpy(bitvecs)

        # Shape: [4, 1]
        assert bitvecs.shape == (4, 1)

        # Only row 0, so bit 0 = 1 for items 0 and 2
        assert bitvecs_cpu[0, 0] == 1  # Item 0 in row 0
        assert bitvecs_cpu[1, 0] == 0  # Item 1 not present
        assert bitvecs_cpu[2, 0] == 1  # Item 2 in row 0
        assert bitvecs_cpu[3, 0] == 0  # Item 3 not present

    def test_multiple_u64s(self):
        """Test with >64 rows to verify multi-word bitvecs."""
        # 100 transactions, each containing item 0
        n_rows = 100
        indptr = np.arange(n_rows + 1, dtype=np.int64)  # [0, 1, 2, ..., 100]
        indices = np.zeros(n_rows, dtype=np.int64)  # All item 0

        bitvecs = build_bitvecs_gpu(indptr, indices, n_rows=n_rows, n_cols=3)

        # n_u64s = ceil(100/64) = 2
        assert bitvecs.shape == (3, 2)

        bitvecs_cpu = cupy.asnumpy(bitvecs)

        # Item 0 should have all bits set for rows 0-99
        # Word 0: bits 0-63 → all 1s = 0xFFFFFFFFFFFFFFFF
        # Word 1: bits 64-99 (36 bits) → 0xFFFFFFFFF (but only 36 bits)
        assert bitvecs_cpu[0, 0] == 0xFFFFFFFFFFFFFFFF
        # 36 bits set: 2^36 - 1 = 0xFFFFFFFFF
        expected_word1 = (1 << 36) - 1
        assert bitvecs_cpu[0, 1] == expected_word1, f"Expected {expected_word1}, got {bitvecs_cpu[0, 1]}"

        # Items 1 and 2 should have no bits set
        assert bitvecs_cpu[1, 0] == 0
        assert bitvecs_cpu[1, 1] == 0
        assert bitvecs_cpu[2, 0] == 0
        assert bitvecs_cpu[2, 1] == 0

    def test_sparse_pattern(self):
        """Test with sparse data pattern (typical market basket)."""
        # 10 transactions, 5 items, sparse pattern
        # Row 0: [0]
        # Row 1: [1, 2]
        # Row 2: [0, 1]
        # Row 3: [3]
        # Row 4: [2, 4]
        # Row 5: [0, 1, 2]
        # Row 6: [4]
        # Row 7: [0, 3]
        # Row 8: [1]
        # Row 9: [2, 3, 4]
        indptr = np.array([0, 1, 3, 5, 6, 8, 11, 12, 14, 15, 18], dtype=np.int64)
        indices = np.array([0, 1, 2, 0, 1, 3, 2, 4, 0, 1, 2, 4, 0, 3, 1, 2, 3, 4], dtype=np.int64)

        bitvecs = build_bitvecs_gpu(indptr, indices, n_rows=10, n_cols=5)

        bitvecs_cpu = cupy.asnumpy(bitvecs)

        # Verify each item's bitvec
        # Item 0: rows 0, 2, 5, 7 → bits 0, 2, 5, 7 → 0b10100101 = 165
        assert bitvecs_cpu[0, 0] == 165, f"Item 0: expected 165, got {bitvecs_cpu[0, 0]}"

        # Item 1: rows 1, 2, 5, 8 → bits 1, 2, 5, 8 → 0b100100110 = 294
        assert bitvecs_cpu[1, 0] == 294, f"Item 1: expected 294, got {bitvecs_cpu[1, 0]}"

        # Item 2: rows 1, 4, 5, 9 → bits 1, 4, 5, 9 → 0b1000110010 = 562
        assert bitvecs_cpu[2, 0] == 562, f"Item 2: expected 562, got {bitvecs_cpu[2, 0]}"

        # Item 3: rows 3, 7, 9 → bits 3, 7, 9 → 0b1010001000 = 648
        assert bitvecs_cpu[3, 0] == 648, f"Item 3: expected 648, got {bitvecs_cpu[3, 0]}"

        # Item 4: rows 4, 6, 9 → bits 4, 6, 9 → 0b1001010000 = 592
        assert bitvecs_cpu[4, 0] == 592, f"Item 4: expected 592, got {bitvecs_cpu[4, 0]}"

    def test_int32_input_converted(self):
        """Test that int32 inputs are correctly converted to int64."""
        # Same as simple test but with int32 arrays
        indptr = np.array([0, 2, 4, 6, 9], dtype=np.int32)
        indices = np.array([0, 1, 1, 2, 0, 2, 0, 1, 2], dtype=np.int32)

        bitvecs = build_bitvecs_gpu(indptr, indices, n_rows=4, n_cols=3)

        bitvecs_cpu = cupy.asnumpy(bitvecs)

        # Same expected results as simple test
        assert bitvecs_cpu[0, 0] == 13
        assert bitvecs_cpu[1, 0] == 11
        assert bitvecs_cpu[2, 0] == 14


class TestScipyIntegration:
    """Test integration with scipy sparse matrices."""

    def test_from_scipy_csr(self):
        """Test build_bitvecs_gpu_from_scipy helper."""
        from scipy.sparse import csr_matrix

        # Create sparse matrix manually
        # 4 rows, 3 cols, same pattern as simple test
        data = np.ones(9, dtype=np.int32)
        indices = np.array([0, 1, 1, 2, 0, 2, 0, 1, 2], dtype=np.int32)
        indptr = np.array([0, 2, 4, 6, 9], dtype=np.int32)
        csr = csr_matrix((data, indices, indptr), shape=(4, 3))

        bitvecs = build_bitvecs_gpu_from_scipy(csr)

        bitvecs_cpu = cupy.asnumpy(bitvecs)

        # Same expected results
        assert bitvecs_cpu[0, 0] == 13
        assert bitvecs_cpu[1, 0] == 11
        assert bitvecs_cpu[2, 0] == 14


class TestLargeScale:
    """Test with larger data to verify scalability."""

    @pytest.mark.slow
    def test_million_rows(self):
        """Test with 1 million rows (realistic scale)."""
        n_rows = 1_000_000
        n_cols = 100
        avg_items_per_row = 5

        # Generate random sparse pattern
        rng = np.random.default_rng(42)
        items_per_row = rng.poisson(avg_items_per_row, n_rows)
        items_per_row = np.clip(items_per_row, 0, n_cols)

        nnz = items_per_row.sum()
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(items_per_row)

        indices = np.concatenate([
            rng.choice(n_cols, size=n, replace=False)
            for n in items_per_row
        ]).astype(np.int64)

        bitvecs = build_bitvecs_gpu(indptr, indices, n_rows, n_cols)

        # Verify shape
        n_u64s = (n_rows + 63) // 64
        assert bitvecs.shape == (n_cols, n_u64s)

        # Verify dtype
        assert bitvecs.dtype == np.uint64

    @pytest.mark.slow
    def test_many_columns(self):
        """Test with many columns (wide matrix)."""
        n_rows = 10_000
        n_cols = 5_000  # Many items
        avg_items_per_row = 20

        rng = np.random.default_rng(123)
        items_per_row = rng.poisson(avg_items_per_row, n_rows)
        items_per_row = np.clip(items_per_row, 0, n_cols)

        nnz = items_per_row.sum()
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(items_per_row)

        indices = np.concatenate([
            rng.choice(n_cols, size=n, replace=False)
            for n in items_per_row
        ]).astype(np.int64)

        bitvecs = build_bitvecs_gpu(indptr, indices, n_rows, n_cols)

        # Verify shape
        n_u64s = (n_rows + 63) // 64
        assert bitvecs.shape == (n_cols, n_u64s)
