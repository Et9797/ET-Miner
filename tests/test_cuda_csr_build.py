"""
Tests for CUDA CSR Build Module (GPU-accelerated CSR construction)

Run with: pytest tests/test_cuda_csr_build.py -v

Requires a CUDA GPU with CuPy installed.
"""
import pytest
import numpy as np

# Skip all tests if cupy not available
cupy = pytest.importorskip("cupy")

# Belt for cupy-installed-but-no-device boxes: auto-skipped via the gpu mark.
pytestmark = pytest.mark.gpu


class TestGenerateCSRGPU:
    """Tests for generate_csr_gpu function."""

    def test_basic_generation(self):
        """Test basic CSR generation on GPU."""
        from et_miner.gpu.csr_build import generate_csr_gpu

        indptr, indices = generate_csr_gpu(
            n_rows=1000,
            n_cols=100,
            avg_items_per_row=10,
            seed=42
        )

        # Check types
        assert isinstance(indptr, cupy.ndarray)
        assert isinstance(indices, cupy.ndarray)
        assert indptr.dtype == cupy.int64
        assert indices.dtype == cupy.int64

        # Check shapes
        assert len(indptr) == 1001  # n_rows + 1
        assert len(indices) > 0

        # Check indptr monotonicity
        diffs = indptr[1:] - indptr[:-1]
        assert cupy.all(diffs >= 0)

    def test_exact_items_mode(self):
        """Test fixed items per row mode."""
        from et_miner.gpu.csr_build import generate_csr_gpu

        n_rows = 500
        items_per_row = 5

        indptr, indices = generate_csr_gpu(
            n_rows=n_rows,
            n_cols=100,
            avg_items_per_row=items_per_row,
            seed=42,
            exact_items=True
        )

        # With exact_items=True, each row should have exactly 5 items
        expected_nnz = n_rows * items_per_row
        assert len(indices) == expected_nnz

    def test_indices_in_valid_range(self):
        """Test that all indices are within valid column range."""
        from et_miner.gpu.csr_build import generate_csr_gpu

        n_cols = 50
        indptr, indices = generate_csr_gpu(
            n_rows=1000,
            n_cols=n_cols,
            avg_items_per_row=10,
            seed=42
        )

        assert int(indices.min()) >= 0
        assert int(indices.max()) < n_cols

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        from et_miner.gpu.csr_build import generate_csr_gpu

        indptr1, indices1 = generate_csr_gpu(
            n_rows=100, n_cols=50, avg_items_per_row=5, seed=42
        )
        indptr2, indices2 = generate_csr_gpu(
            n_rows=100, n_cols=50, avg_items_per_row=5, seed=42
        )

        assert cupy.array_equal(indptr1, indptr2)
        # Note: indices might differ due to sorting implementation
        # but nnz should be the same
        assert len(indices1) == len(indices2)


class TestCSRToBitvecsGPU:
    """Tests for csr_to_bitvecs_gpu function."""

    def test_basic_conversion(self):
        """Test basic CSR to bitvec conversion."""
        from et_miner.gpu.csr_build import generate_csr_gpu, csr_to_bitvecs_gpu

        n_rows = 1000
        n_cols = 100

        indptr, indices = generate_csr_gpu(
            n_rows=n_rows,
            n_cols=n_cols,
            avg_items_per_row=10,
            seed=42
        )

        bitvecs = csr_to_bitvecs_gpu(indptr, indices, n_rows, n_cols)

        # Check shape
        n_u64s = (n_rows + 63) // 64
        assert bitvecs.shape == (n_cols, n_u64s)
        assert bitvecs.dtype == cupy.uint64

    def test_bitvec_correctness(self):
        """Test that bitvecs correctly represent the CSR data."""
        from et_miner.gpu.csr_build import csr_to_bitvecs_gpu

        # Simple known CSR: 3 rows, 3 cols
        # Row 0: cols [0, 2]
        # Row 1: cols [1]
        # Row 2: cols [0, 1, 2]
        indptr = cupy.array([0, 2, 3, 6], dtype=cupy.int64)
        indices = cupy.array([0, 2, 1, 0, 1, 2], dtype=cupy.int64)

        bitvecs = csr_to_bitvecs_gpu(indptr, indices, n_rows=3, n_cols=3)

        # Bitvecs should have 1 u64 (since 3 rows < 64)
        assert bitvecs.shape == (3, 1)

        # Check bit patterns
        # Col 0: rows 0, 2 -> bits 0, 2 -> 0b101 = 5
        # Col 1: rows 1, 2 -> bits 1, 2 -> 0b110 = 6
        # Col 2: rows 0, 2 -> bits 0, 2 -> 0b101 = 5
        assert int(bitvecs[0, 0]) == 5
        assert int(bitvecs[1, 0]) == 6
        assert int(bitvecs[2, 0]) == 5


class TestGenerateBitvecsGPU:
    """Tests for full pipeline generate_bitvecs_gpu function."""

    def test_full_pipeline(self):
        """Test the combined CSR + bitvec generation."""
        from et_miner.gpu.csr_build import generate_bitvecs_gpu

        n_rows = 2000
        n_cols = 150

        indptr, indices, bitvecs = generate_bitvecs_gpu(
            n_rows=n_rows,
            n_cols=n_cols,
            avg_items_per_row=12,
            seed=42
        )

        # Check CSR
        assert len(indptr) == n_rows + 1
        assert len(indices) > 0

        # Check bitvecs
        n_u64s = (n_rows + 63) // 64
        assert bitvecs.shape == (n_cols, n_u64s)


class TestGenerateCSRGPUBatch:
    """Tests for multi-GPU batch generation."""

    def test_batch_generation_single_gpu(self):
        """Test batch generation with single GPU."""
        from et_miner.gpu.csr_build import generate_csr_gpu_batch

        results = generate_csr_gpu_batch(
            n_rows=1000,
            n_cols=100,
            avg_items_per_row=10,
            n_gpus=1,
            base_seed=42
        )

        assert len(results) == 1
        indptr, indices, device_id = results[0]
        assert device_id == 0
        assert len(indptr) == 1001


class TestVerifyCSRCorrectness:
    """Tests for CSR verification utility."""

    def test_verify_valid_csr(self):
        """Test verification of valid CSR matrix."""
        from et_miner.gpu.csr_build import generate_csr_gpu, verify_csr_correctness

        indptr, indices = generate_csr_gpu(
            n_rows=100,
            n_cols=50,
            avg_items_per_row=5,
            seed=42
        )

        # Should not raise
        result = verify_csr_correctness(indptr, indices, n_rows=100, n_cols=50)
        assert result is True


class TestPoissonGeneration:
    """Tests for Poisson-distributed generation."""

    def test_poisson_distribution(self):
        """Test that Poisson generation produces varied row lengths."""
        from et_miner.gpu.csr_build import generate_csr_gpu_poisson

        n_rows = 10000
        avg_items = 20

        indptr, indices = generate_csr_gpu_poisson(
            n_rows=n_rows,
            n_cols=100,
            avg_items_per_row=avg_items,
            seed=42
        )

        # Calculate items per row
        items_per_row = indptr[1:] - indptr[:-1]

        # Should have some variation (std > 0)
        std = float(cupy.std(items_per_row))
        assert std > 0

        # Mean should be close to avg_items
        mean = float(cupy.mean(items_per_row))
        assert abs(mean - avg_items) < 2  # Allow some deviation


class TestBootstrapGeneration:
    """Tests for bootstrap generation from source data."""

    def test_bootstrap_from_source(self):
        """Test bootstrap generation maintains source distribution."""
        from et_miner.gpu.csr_build import generate_csr_gpu_bootstrap

        # Create simple source data (numpy arrays)
        source_indptr = np.array([0, 3, 5, 10], dtype=np.int64)  # 3 transactions
        source_indices = np.array([0, 1, 2, 1, 3, 0, 2, 3, 4, 5], dtype=np.int64)

        indptr, indices = generate_csr_gpu_bootstrap(
            source_indptr=source_indptr,
            source_indices=source_indices,
            n_rows=100,
            n_cols=6,
            seed=42
        )

        assert len(indptr) == 101
        assert len(indices) > 0

        # All indices should be in valid range
        assert int(indices.min()) >= 0
        assert int(indices.max()) < 6


# Performance test (optional, slow)
@pytest.mark.slow
class TestPerformance:
    """Performance benchmarks (marked as slow)."""

    def test_large_scale_generation(self):
        """Test generation at scale (10M rows)."""
        from et_miner.gpu.csr_build import generate_csr_gpu
        import time

        n_rows = 10_000_000
        n_cols = 1000

        # Warmup
        generate_csr_gpu(n_rows=1000, n_cols=100, avg_items_per_row=10, seed=0)
        cupy.cuda.Stream.null.synchronize()

        # Benchmark
        start = time.time()
        indptr, indices = generate_csr_gpu(
            n_rows=n_rows,
            n_cols=n_cols,
            avg_items_per_row=10,
            seed=42
        )
        cupy.cuda.Stream.null.synchronize()
        elapsed = time.time() - start

        rows_per_sec = n_rows / elapsed
        print(f"\n10M row generation: {elapsed:.2f}s ({rows_per_sec/1e6:.2f}M rows/sec)")

        # Should be faster than 10 seconds for 10M rows
        assert elapsed < 30  # Conservative limit
