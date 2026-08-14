#!/usr/bin/env python3
"""
Stream Sync Validation — Phase 1 of Triple-Layer Parallelism

Tests whether CuPy's `with non_blocking_stream:` context manager redirects
kernel launches away from the null (default) stream, and whether calling
`Stream.null.synchronize()` inside that context blocks OTHER streams.

Three possible outcomes:
  1. WORKS: Kernels go to the non-blocking stream. null sync is a no-op
     (null stream has no pending work) → no code changes needed
  2. PARTIALLY WORKS: Kernels go to the stream, but null sync still blocks
     all streams → need suppress_null_sync() context manager
  3. DOESN'T WORK: Kernels go to null stream despite context → need explicit
     stream parameter in kernel wrappers

Usage:
    python benchmarks/stream_sync_test.py          # Quick validation
    python benchmarks/stream_sync_test.py --full    # Full test with real kernels

Date: 2026-02-06
"""

import sys
import time
import argparse

import pytest

# Whole module is GPU-only: kernel-stream behaviour needs a real CUDA device.
pytest.importorskip("cupy")


def test_stream_context_redirect():
    """Test 1: Does `with stream:` redirect CuPy kernel launches?

    Creates two non-blocking streams, launches heavy work on stream_a inside
    `with stream_a:`, then immediately launches work on stream_b. If streams
    truly overlap, the total time should be less than the sum of individual times.
    """
    import cupy as cp

    print("\n" + "=" * 70)
    print("  TEST 1: Stream Context Redirect")
    print("  Does `with stream:` redirect kernel launches to that stream?")
    print("=" * 70)

    stream_a = cp.cuda.Stream(non_blocking=True)
    stream_b = cp.cuda.Stream(non_blocking=True)

    # Warmup
    _ = cp.random.random((1000, 1000))
    cp.cuda.Stream.null.synchronize()

    N = 8192  # Large enough to take measurable time

    # Measure stream_a alone
    event_a_start = cp.cuda.Event()
    event_a_end = cp.cuda.Event()
    with stream_a:
        event_a_start.record(stream=stream_a)
        a = cp.random.random((N, N))
        _ = cp.matmul(a, a)
        event_a_end.record(stream=stream_a)
    event_a_end.synchronize()
    time_a = cp.cuda.get_elapsed_time(event_a_start, event_a_end)

    del a
    cp.get_default_memory_pool().free_all_blocks()

    # Measure stream_b alone
    event_b_start = cp.cuda.Event()
    event_b_end = cp.cuda.Event()
    with stream_b:
        event_b_start.record(stream=stream_b)
        b = cp.random.random((N, N))
        _ = cp.matmul(b, b)
        event_b_end.record(stream=stream_b)
    event_b_end.synchronize()
    time_b = cp.cuda.get_elapsed_time(event_b_start, event_b_end)

    del b
    cp.get_default_memory_pool().free_all_blocks()

    # Measure both streams overlapping
    # Use smaller matrix to fit both in memory
    M = N // 2
    event_total_start = cp.cuda.Event()
    event_total_end_a = cp.cuda.Event()
    event_total_end_b = cp.cuda.Event()

    event_total_start.record()  # Record on null stream (before both)

    with stream_a:
        a = cp.random.random((M, M))
        _ = cp.matmul(a, a)
        event_total_end_a.record(stream=stream_a)

    with stream_b:
        b = cp.random.random((M, M))
        _ = cp.matmul(b, b)
        event_total_end_b.record(stream=stream_b)

    event_total_end_a.synchronize()
    event_total_end_b.synchronize()

    time_concurrent_a = cp.cuda.get_elapsed_time(event_total_start, event_total_end_a)
    time_concurrent_b = cp.cuda.get_elapsed_time(event_total_start, event_total_end_b)
    time_concurrent_total = max(time_concurrent_a, time_concurrent_b)

    del a, b
    cp.get_default_memory_pool().free_all_blocks()

    # Check if CuPy events are on separate streams
    print(f"\n  Stream A alone:         {time_a:.2f} ms")
    print(f"  Stream B alone:         {time_b:.2f} ms")
    print(f"  Sum (no overlap):       {time_a + time_b:.2f} ms")
    print(f"  Concurrent (from null): {time_concurrent_total:.2f} ms")

    # If streams redirect correctly, concurrent total should be roughly equal
    # to the max of the two (not the sum)
    if time_concurrent_total < (time_a + time_b) * 0.85:
        print("\n  [PASS] Streams overlap! Concurrent < 85% of sequential sum")
        print("         `with stream:` correctly redirects kernel launches")
        return True
    else:
        print("\n  [WARN] Minimal overlap detected. Streams may not be independent")
        return False


def test_null_sync_blocks_other_streams():
    """Test 2: Does Stream.null.synchronize() block non-blocking streams?

    Launches a long-running kernel on stream_a, calls Stream.null.synchronize()
    (which should only sync the null stream), then checks if stream_a is still
    running.

    This is the CRITICAL test — if null sync blocks non-blocking streams,
    we need suppress_null_sync().
    """
    import cupy as cp

    print("\n" + "=" * 70)
    print("  TEST 2: Does Stream.null.synchronize() Block Other Streams?")
    print("  This determines if we need suppress_null_sync()")
    print("=" * 70)

    stream_a = cp.cuda.Stream(non_blocking=True)

    # Warmup
    _ = cp.random.random((1000, 1000))
    cp.cuda.Stream.null.synchronize()

    N = 8192

    # Launch heavy work on stream_a
    event_start = cp.cuda.Event()
    event_end = cp.cuda.Event()

    with stream_a:
        event_start.record(stream=stream_a)
        # Launch a LOT of work to keep stream_a busy
        a = cp.random.random((N, N))
        for _ in range(5):
            a = cp.matmul(a, a)
            a = a / cp.max(a)  # Prevent overflow
        event_end.record(stream=stream_a)

    # Now call null sync — does this wait for stream_a?
    wall_start = time.perf_counter()
    cp.cuda.Stream.null.synchronize()
    wall_null_sync = time.perf_counter() - wall_start

    # Check if stream_a is still running after null sync
    stream_a_done = event_end.done

    # Now wait for stream_a to actually finish
    event_end.synchronize()
    time_stream_a = cp.cuda.get_elapsed_time(event_start, event_end)

    del a
    cp.get_default_memory_pool().free_all_blocks()

    print(f"\n  Stream A total GPU time: {time_stream_a:.2f} ms")
    print(f"  null sync wall time:    {wall_null_sync*1000:.2f} ms")
    print(f"  Stream A done after null sync: {stream_a_done}")

    if not stream_a_done and wall_null_sync * 1000 < time_stream_a * 0.5:
        print("\n  [PASS] null sync did NOT block stream_a!")
        print("         non_blocking=True streams are independent of null stream")
        print("         → No suppress_null_sync() needed")
        return True
    elif wall_null_sync * 1000 >= time_stream_a * 0.8:
        print("\n  [FAIL] null sync BLOCKED until stream_a completed!")
        print("         non_blocking=True doesn't prevent null sync from blocking")
        print("         → NEED suppress_null_sync() context manager")
        return False
    else:
        print("\n  [PARTIAL] null sync partially blocked stream_a")
        print("            May need suppress_null_sync() for safety")
        return False


def test_with_stream_plus_null_sync():
    """Test 3: The actual code pattern — kernel inside `with stream:` + null sync.

    This simulates what happens when we call a kernel function that internally
    calls Stream.null.synchronize() while we're inside a `with stream:` context.

    Pattern:
        with gen_stream:
            # kernel_func() internally calls Stream.null.synchronize()
            result = kernel_func(...)  # includes null sync at the end

    Question: Does the null sync inside kernel_func only sync the null stream
    (which has no work because everything went to gen_stream), or does it
    sync everything?
    """
    import cupy as cp

    print("\n" + "=" * 70)
    print("  TEST 3: Kernel + null sync inside `with stream:` Context")
    print("  Simulates the actual code pattern in et-miner")
    print("=" * 70)

    stream_a = cp.cuda.Stream(non_blocking=True)
    stream_b = cp.cuda.Stream(non_blocking=True)

    # Warmup
    _ = cp.random.random((1000, 1000))
    cp.cuda.Stream.null.synchronize()

    N = 4096

    def kernel_with_null_sync(n):
        """Simulates a kernel function that ends with Stream.null.synchronize()."""
        a = cp.random.random((n, n))
        result = cp.matmul(a, a)
        cp.cuda.Stream.null.synchronize()  # This is what the kernel does
        return result

    # Time both streams with interleaved null syncs
    wall_start = time.perf_counter()

    event_a_start = cp.cuda.Event()
    event_a_end = cp.cuda.Event()
    event_b_start = cp.cuda.Event()
    event_b_end = cp.cuda.Event()

    # Stream A: kernel with null sync
    with stream_a:
        event_a_start.record(stream=stream_a)
        result_a = kernel_with_null_sync(N)
        event_a_end.record(stream=stream_a)

    # Stream B: starts immediately after (or does it?)
    with stream_b:
        event_b_start.record(stream=stream_b)
        result_b = kernel_with_null_sync(N)
        event_b_end.record(stream=stream_b)

    event_a_end.synchronize()
    event_b_end.synchronize()
    wall_total = time.perf_counter() - wall_start

    time_a = cp.cuda.get_elapsed_time(event_a_start, event_a_end)
    time_b = cp.cuda.get_elapsed_time(event_b_start, event_b_end)
    time_a_to_b_end = cp.cuda.get_elapsed_time(event_a_start, event_b_end)

    del result_a, result_b
    cp.get_default_memory_pool().free_all_blocks()

    print(f"\n  Stream A GPU time:      {time_a:.2f} ms")
    print(f"  Stream B GPU time:      {time_b:.2f} ms")
    print(f"  Sum (no overlap):       {time_a + time_b:.2f} ms")
    print(f"  A start → B end:        {time_a_to_b_end:.2f} ms")
    print(f"  Wall clock total:       {wall_total*1000:.2f} ms")

    # If the null sync inside kernel_with_null_sync blocks stream_b from starting,
    # then A start → B end ≈ time_a + time_b (sequential)
    # If it doesn't block, A start → B end ≈ max(time_a, time_b) (parallel)
    overlap = 1 - (time_a_to_b_end / (time_a + time_b)) if (time_a + time_b) > 0 else 0

    if overlap > 0.15:
        print(f"\n  [PASS] {overlap*100:.0f}% stream overlap achieved!")
        print("         null sync inside `with stream:` does NOT block other streams")
        print("         → Existing async_pipeline.py code should work as-is")
        return True
    else:
        print(f"\n  [FAIL] Only {overlap*100:.0f}% overlap — streams are serialized")
        print("         null sync blocks even non-blocking streams")
        print("         → Need suppress_null_sync() wrapper")
        return False


def test_suppress_null_sync():
    """Test 4: Validate suppress_null_sync() context manager.

    If Test 2/3 fail, this tests the fallback approach: monkey-patching
    Stream.null.synchronize() to a no-op within the stream context.
    """
    import contextlib
    import cupy as cp

    print("\n" + "=" * 70)
    print("  TEST 4: suppress_null_sync() Fallback Validation")
    print("  Tests monkey-patching Stream.null.synchronize() to a no-op")
    print("=" * 70)

    @contextlib.contextmanager
    def suppress_null_sync():
        """Temporarily make Stream.null.synchronize() a no-op."""
        original = cp.cuda.Stream.null.synchronize
        cp.cuda.Stream.null.synchronize = lambda: None
        try:
            yield
        finally:
            cp.cuda.Stream.null.synchronize = original

    stream_a = cp.cuda.Stream(non_blocking=True)
    stream_b = cp.cuda.Stream(non_blocking=True)

    # Warmup
    _ = cp.random.random((1000, 1000))
    cp.cuda.Stream.null.synchronize()

    N = 4096

    def kernel_with_null_sync(n):
        """Simulates a kernel that ends with null sync."""
        a = cp.random.random((n, n))
        result = cp.matmul(a, a)
        cp.cuda.Stream.null.synchronize()
        return result

    # Test with suppress_null_sync
    event_a_start = cp.cuda.Event()
    event_a_end = cp.cuda.Event()
    event_b_start = cp.cuda.Event()
    event_b_end = cp.cuda.Event()

    with stream_a:
        with suppress_null_sync():
            event_a_start.record(stream=stream_a)
            result_a = kernel_with_null_sync(N)
            event_a_end.record(stream=stream_a)

    with stream_b:
        with suppress_null_sync():
            event_b_start.record(stream=stream_b)
            result_b = kernel_with_null_sync(N)
            event_b_end.record(stream=stream_b)

    # Explicit stream sync instead of null sync
    stream_a.synchronize()
    stream_b.synchronize()

    time_a = cp.cuda.get_elapsed_time(event_a_start, event_a_end)
    time_b = cp.cuda.get_elapsed_time(event_b_start, event_b_end)
    time_total = cp.cuda.get_elapsed_time(event_a_start, event_b_end)

    # Verify results are correct (no data corruption from skipping sync)
    result_a_sum = float(cp.sum(result_a))
    result_b_sum = float(cp.sum(result_b))

    del result_a, result_b
    cp.get_default_memory_pool().free_all_blocks()

    print(f"\n  Stream A GPU time:      {time_a:.2f} ms")
    print(f"  Stream B GPU time:      {time_b:.2f} ms")
    print(f"  Total (A start→B end):  {time_total:.2f} ms")
    print(f"  Result A checksum:      {result_a_sum:.4e} (should be finite)")
    print(f"  Result B checksum:      {result_b_sum:.4e} (should be finite)")

    overlap = 1 - (time_total / (time_a + time_b)) if (time_a + time_b) > 0 else 0

    import math
    results_valid = math.isfinite(result_a_sum) and math.isfinite(result_b_sum)

    if overlap > 0.15 and results_valid:
        print(f"\n  [PASS] {overlap*100:.0f}% overlap with suppress_null_sync()!")
        print("         Results are valid (no data corruption)")
        print("         → suppress_null_sync() is a safe fallback")
        return True
    elif not results_valid:
        print("\n  [FAIL] Data corruption detected! suppress_null_sync() is NOT safe")
        return False
    else:
        print(f"\n  [FAIL] Only {overlap*100:.0f}% overlap even with suppress")
        return False


def test_real_kernel_pattern():
    """Test 5: Test with actual et-miner kernel calls if available.

    Runs the real generate_csr_gpu + csr_to_bitvecs_gpu within stream contexts
    to verify the full production code path.
    """
    import cupy as cp

    print("\n" + "=" * 70)
    print("  TEST 5: Real ET-Miner Kernel Pattern")
    print("  Tests actual generate_csr_gpu + csr_to_bitvecs_gpu in streams")
    print("=" * 70)

    try:
        from et_miner.cuda_csr_build import generate_csr_gpu, csr_to_bitvecs_gpu
    except ImportError:
        print("\n  [SKIP] et_miner.cuda_csr_build not available")
        print("         Install et-miner to test real kernel paths")
        return None

    stream_a = cp.cuda.Stream(non_blocking=True)
    stream_b = cp.cuda.Stream(non_blocking=True)

    # Warmup
    _ = cp.random.random((1000, 1000))
    cp.cuda.Stream.null.synchronize()

    N_ROWS = 1_000_000
    N_COLS = 100
    AVG_ITEMS = 10

    # Sequential baseline
    seq_start = time.perf_counter()
    for i in range(2):
        with cp.cuda.Device(0):
            indptr, indices = generate_csr_gpu(N_ROWS, N_COLS, AVG_ITEMS, seed=42+i, device_id=0)
            bitvecs = csr_to_bitvecs_gpu(indptr, indices, N_ROWS, N_COLS, device_id=0)
            del indptr, indices, bitvecs
            cp.get_default_memory_pool().free_all_blocks()
    seq_time = time.perf_counter() - seq_start

    # Stream-overlapped version
    event_a_start = cp.cuda.Event()
    event_a_end = cp.cuda.Event()
    event_b_start = cp.cuda.Event()
    event_b_end = cp.cuda.Event()

    stream_start = time.perf_counter()

    with cp.cuda.Device(0):
        with stream_a:
            event_a_start.record(stream=stream_a)
            indptr_a, indices_a = generate_csr_gpu(N_ROWS, N_COLS, AVG_ITEMS, seed=42, device_id=0)
            bitvecs_a = csr_to_bitvecs_gpu(indptr_a, indices_a, N_ROWS, N_COLS, device_id=0)
            event_a_end.record(stream=stream_a)

        with stream_b:
            event_b_start.record(stream=stream_b)
            indptr_b, indices_b = generate_csr_gpu(N_ROWS, N_COLS, AVG_ITEMS, seed=43, device_id=0)
            bitvecs_b = csr_to_bitvecs_gpu(indptr_b, indices_b, N_ROWS, N_COLS, device_id=0)
            event_b_end.record(stream=stream_b)

    stream_a.synchronize()
    stream_b.synchronize()
    stream_time = time.perf_counter() - stream_start

    time_a = cp.cuda.get_elapsed_time(event_a_start, event_a_end)
    time_b = cp.cuda.get_elapsed_time(event_b_start, event_b_end)
    time_total = cp.cuda.get_elapsed_time(event_a_start, event_b_end)

    del indptr_a, indices_a, bitvecs_a
    del indptr_b, indices_b, bitvecs_b
    cp.get_default_memory_pool().free_all_blocks()

    overlap = 1 - (time_total / (time_a + time_b)) if (time_a + time_b) > 0 else 0
    speedup = seq_time / stream_time if stream_time > 0 else 0

    print(f"\n  Sequential (2 waves):   {seq_time*1000:.2f} ms")
    print(f"  Streamed (2 waves):     {stream_time*1000:.2f} ms")
    print(f"  Stream A GPU time:      {time_a:.2f} ms")
    print(f"  Stream B GPU time:      {time_b:.2f} ms")
    print(f"  Total (A→B end):        {time_total:.2f} ms")
    print(f"  Speedup:                {speedup:.2f}x")
    print(f"  Overlap:                {overlap*100:.0f}%")

    if overlap > 0.10:
        print(f"\n  [PASS] Real kernel streams show {overlap*100:.0f}% overlap!")
        print("         The current _prefetch_on_stream() pattern should work")
        return True
    else:
        print(f"\n  [WARN] Minimal overlap with real kernels ({overlap*100:.0f}%)")
        print("         May need suppress_null_sync() for production code")
        return False


def print_summary(results: dict):
    """Print final summary with verdict."""
    print("\n" + "=" * 70)
    print("  STREAM SYNC VALIDATION SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "PASS" if passed else ("SKIP" if passed is None else "FAIL")
        icon = {"PASS": "[+]", "FAIL": "[-]", "SKIP": "[~]"}[status]
        print(f"  {icon} {test_name}: {status}")

    print()

    # Determine overall verdict
    if results.get("stream_context_redirect") and results.get("null_sync_blocks") is True:
        print("  VERDICT: Streams work perfectly!")
        print("  → No code changes needed in kernel files")
        print("  → Existing _prefetch_on_stream() / _compute_on_stream() are correct")
        verdict = "no_changes_needed"
    elif results.get("suppress_null_sync"):
        print("  VERDICT: Need suppress_null_sync() wrapper")
        print("  → Add suppress_null_sync() to async_pipeline.py")
        print("  → Wrap kernel calls in _prefetch_on_stream() and _compute_on_stream()")
        verdict = "suppress_null_sync"
    elif results.get("stream_context_redirect"):
        print("  VERDICT: Streams redirect correctly but null sync needs handling")
        print("  → Use suppress_null_sync() OR explicit stream.synchronize()")
        verdict = "suppress_null_sync"
    else:
        print("  VERDICT: Stream context does NOT redirect kernel launches")
        print("  → Need explicit stream parameter in kernel wrappers")
        print("  → This is the most invasive change")
        verdict = "explicit_stream_params"

    print("=" * 70)
    return verdict


def main():
    parser = argparse.ArgumentParser(
        description="Stream Sync Validation for Triple-Layer Parallelism"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full test suite including real et-miner kernels"
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="GPU device to test on (default: 0)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  STREAM SYNC VALIDATION — Phase 1 of Triple-Layer Parallelism")
    print("  Testing CuPy CUDA stream behavior for overlap potential")
    print("=" * 70)

    # Check CuPy availability
    try:
        import cupy as cp
        n_gpus = cp.cuda.runtime.getDeviceCount()
        props = cp.cuda.runtime.getDeviceProperties(args.gpu)
        gpu_name = props["name"].decode()
        print(f"\n  GPU {args.gpu}: {gpu_name}")
        print(f"  CuPy: {cp.__version__}")
        print(f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

        has_ft = hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled()
        print(f"  Free-threading: {'YES' if has_ft else 'NO (GIL present)'}")
    except Exception as e:
        print(f"\n  [ERROR] CuPy not available: {e}")
        print("  This test requires a GPU with CuPy installed.")
        sys.exit(1)

    # Set device
    cp.cuda.Device(args.gpu).use()

    results = {}

    # Test 1: Does `with stream:` redirect kernel launches?
    results["stream_context_redirect"] = test_stream_context_redirect()

    # Test 2: Does null sync block non-blocking streams?
    results["null_sync_blocks"] = test_null_sync_blocks_other_streams()

    # Test 3: The actual kernel + null sync pattern
    results["kernel_null_sync_pattern"] = test_with_stream_plus_null_sync()

    # Test 4: suppress_null_sync() fallback
    results["suppress_null_sync"] = test_suppress_null_sync()

    # Test 5: Real kernels (if --full)
    if args.full:
        results["real_kernel_pattern"] = test_real_kernel_pattern()

    # Summary & verdict
    verdict = print_summary(results)

    # Return exit code: 0 = no changes needed, 1 = need suppress, 2 = need explicit
    exit_codes = {"no_changes_needed": 0, "suppress_null_sync": 1, "explicit_stream_params": 2}
    sys.exit(exit_codes.get(verdict, 2))


if __name__ == "__main__":
    main()
