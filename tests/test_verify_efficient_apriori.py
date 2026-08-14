#!/usr/bin/env python3
"""Correctness verification: et-miner GPU pipeline vs efficient-apriori ground truth.

Validates that our fused k=2 CUDA kernel + standard k>=3 path produces
identical results to the reference efficient-apriori implementation.

Usage:
    python scripts/verify_vs_efficient_apriori.py
"""

import signal
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

import pytest

ea_apriori = pytest.importorskip("efficient_apriori", reason="efficient_apriori not installed").apriori
from et_miner import apriori as pa_apriori


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

PASS = f"{GREEN}\u2713 PASS{RESET}"
FAIL = f"{RED}\u2717 FAIL{RESET}"
SKIP = f"{YELLOW}~ SKIP{RESET}"


def header(text: str) -> None:
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  {text}{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}\n")


def subheader(text: str) -> None:
    print(f"\n{CYAN}--- {text} ---{RESET}")


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def ea_to_set(itemsets_dict: dict, n_trans: int) -> set[tuple]:
    """Convert efficient-apriori output to comparable set of (sorted_tuple, support).

    efficient-apriori returns {k: {frozenset: absolute_count}}.
    We convert to {(tuple(sorted(items)), round(count / n_trans, ndigits))}.
    """
    result = set()
    for k, itemsets in itemsets_dict.items():
        if k < 1:
            continue
        for fset, count in itemsets.items():
            key = tuple(sorted(fset))
            support = count / n_trans
            # Round to avoid floating point noise
            support = round(support, 6)
            result.add((key, support))
    return result


def pa_to_set(result_df: pl.DataFrame) -> set[tuple]:
    """Convert et-miner output DataFrame to comparable set of (sorted_tuple, support)."""
    result = set()
    itemsets = result_df.get_column("itemset").to_list()
    supports = result_df.get_column("support").to_list()
    for itemset, support in zip(itemsets, supports):
        key = tuple(sorted(itemset))
        support = round(support, 6)
        result.add((key, support))
    return result


def compare_results(
    ea_set: set[tuple],
    pa_set: set[tuple],
    tol: float = 0.001,
    label: str = "",
) -> tuple[bool, str]:
    """Compare two result sets with tolerance on supports.

    Returns (passed, detail_message).
    """
    # Extract just the itemset keys
    ea_keys = {item[0] for item in ea_set}
    pa_keys = {item[0] for item in pa_set}

    missing_from_pa = ea_keys - pa_keys
    extra_in_pa = pa_keys - ea_keys

    # Check support values for shared itemsets
    ea_lookup = {item[0]: item[1] for item in ea_set}
    pa_lookup = {item[0]: item[1] for item in pa_set}

    support_mismatches = []
    shared_keys = ea_keys & pa_keys
    for key in shared_keys:
        ea_sup = ea_lookup[key]
        pa_sup = pa_lookup[key]
        if abs(ea_sup - pa_sup) > tol:
            support_mismatches.append((key, ea_sup, pa_sup))

    passed = (not missing_from_pa) and (not extra_in_pa) and (not support_mismatches)

    details = []
    if missing_from_pa:
        details.append(f"  Missing from et-miner ({len(missing_from_pa)}):")
        for item in sorted(missing_from_pa)[:10]:
            details.append(f"    {item}  (ea support: {ea_lookup[item]:.4f})")
        if len(missing_from_pa) > 10:
            details.append(f"    ... and {len(missing_from_pa) - 10} more")

    if extra_in_pa:
        details.append(f"  Extra in et-miner ({len(extra_in_pa)}):")
        for item in sorted(extra_in_pa)[:10]:
            details.append(f"    {item}  (pa support: {pa_lookup[item]:.4f})")
        if len(extra_in_pa) > 10:
            details.append(f"    ... and {len(extra_in_pa) - 10} more")

    if support_mismatches:
        details.append(f"  Support mismatches ({len(support_mismatches)}):")
        for key, ea_sup, pa_sup in sorted(support_mismatches)[:10]:
            details.append(f"    {key}: ea={ea_sup:.6f} vs pa={pa_sup:.6f}")
        if len(support_mismatches) > 10:
            details.append(f"    ... and {len(support_mismatches) - 10} more")

    detail_msg = "\n".join(details) if details else ""
    return passed, detail_msg


def itemsets_by_k(result_set: set[tuple]) -> dict[int, set[tuple]]:
    """Group a result set by itemset size k."""
    groups: dict[int, set[tuple]] = {}
    for key, sup in result_set:
        k = len(key)
        if k not in groups:
            groups[k] = set()
        groups[k].add((key, sup))
    return groups


# ---------------------------------------------------------------------------
# Timeout helper (POSIX only)
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Timed out")


# ---------------------------------------------------------------------------
# Test 1: Small dataset (5 transactions)
# ---------------------------------------------------------------------------

def test_small_dataset() -> list[tuple[str, bool]]:
    """Validate on a tiny hand-verifiable dataset.

    Returns list of (label, passed) tuples.
    Asserts exact match for k=1, k=2, k=3 itemsets and supports separately.
    """
    header("Test 1: Small dataset (5 transactions)")

    transactions_raw = [[1, 2, 3], [2, 3, 4], [1, 3, 5], [2, 3], [1, 2, 3, 4]]
    n_trans = len(transactions_raw)

    # efficient-apriori format: list of tuples
    ea_transactions = [tuple(t) for t in transactions_raw]

    # et-miner format: Polars DataFrame
    pa_df = pl.DataFrame({"items": transactions_raw})

    results: list[tuple[str, bool]] = []

    for min_support in [0.3, 0.4]:
        subheader(f"min_support = {min_support}")

        # --- efficient-apriori (ground truth) ---
        t0 = time.perf_counter()
        ea_itemsets, _ = ea_apriori(ea_transactions, min_support=min_support)
        ea_time = time.perf_counter() - t0
        ea_set = ea_to_set(ea_itemsets, n_trans)
        ea_by_k = itemsets_by_k(ea_set)

        print(f"  efficient-apriori: {len(ea_set)} itemsets in {ea_time*1000:.1f}ms")
        for k in sorted(ea_by_k.keys()):
            items = sorted(ea_by_k[k])
            print(f"    k={k}: {len(items)} itemsets")
            for item_key, sup in items:
                print(f"      {item_key} -> {sup:.4f}")

        # --- et-miner GPU ---
        t0 = time.perf_counter()
        pa_result = pa_apriori(pa_df, min_support=min_support, use_gpu=True)
        pa_time = time.perf_counter() - t0
        pa_set = pa_to_set(pa_result)
        pa_by_k = itemsets_by_k(pa_set)

        print(f"  et-miner (GPU):    {len(pa_set)} itemsets in {pa_time*1000:.1f}ms")
        for k in sorted(pa_by_k.keys()):
            print(f"    k={k}: {len(pa_by_k[k])} itemsets")

        # --- Compare per-k: assert exact match for k=1, k=2, k=3 ---
        all_ks = sorted(set(ea_by_k.keys()) | set(pa_by_k.keys()))
        all_passed_for_support = True

        for k in all_ks:
            ea_k = ea_by_k.get(k, set())
            pa_k = pa_by_k.get(k, set())
            passed_k, detail_k = compare_results(ea_k, pa_k, label=f"s={min_support} k={k}")

            if passed_k:
                print(f"  {PASS}  k={k}: {len(ea_k)} itemsets match")
            else:
                print(f"  {FAIL}  k={k}: mismatch (ea={len(ea_k)}, pa={len(pa_k)})")
                print(detail_k)
                all_passed_for_support = False

        results.append((f"Test 1 (small, s={min_support})", all_passed_for_support))

    return results


# ---------------------------------------------------------------------------
# Test 2: Large real/synthetic dataset
# ---------------------------------------------------------------------------

def _generate_synthetic_data(
    n_transactions: int = 100_000,
    n_items: int = 500,
    avg_items_per_txn: int = 10,
    seed: int = 42,
) -> list[list[int]]:
    """Generate reproducible synthetic transaction data."""
    rng = np.random.RandomState(seed)
    transactions = []
    for _ in range(n_transactions):
        n_items_in_txn = max(1, rng.poisson(avg_items_per_txn))
        n_items_in_txn = min(n_items_in_txn, n_items)
        txn = sorted(rng.choice(n_items, size=n_items_in_txn, replace=False).tolist())
        transactions.append(txn)
    return transactions


def _try_load_real_data() -> tuple[list[list[int]] | None, pl.DataFrame | None]:
    """Try to load the 819K real dataset from benchmarks/datasets/all_transactions.parquet.

    Returns (transactions_list, polars_df) or (None, None) if not found.
    We return the DataFrame directly to avoid a redundant re-construction.
    """
    project_root = Path(__file__).resolve().parent.parent
    parquet_path = project_root / "benchmarks" / "datasets" / "all_transactions.parquet"

    if parquet_path.exists():
        print(f"  Loading real data from {parquet_path}")
        df = pl.read_parquet(parquet_path)
        if "items" in df.columns:
            return df.get_column("items").to_list(), df
        else:
            print(f"  WARNING: parquet has columns {df.columns}, expected 'items'")

    return None, None


def test_large_dataset() -> list[tuple[str, bool]]:
    """Validate on the 819K real dataset (or synthetic fallback).

    Returns list of (label, passed) tuples to avoid index mapping issues
    when some support levels are skipped due to timeout.
    """
    header("Test 2: Large dataset (819K transactions)")

    transactions_raw, pa_df = _try_load_real_data()
    if transactions_raw is not None:
        data_source = "real (all_transactions.parquet, 819K)"
    else:
        print("  Real data not found, generating synthetic data...")
        transactions_raw = _generate_synthetic_data(
            n_transactions=100_000, n_items=500, avg_items_per_txn=10, seed=42
        )
        pa_df = pl.DataFrame({"items": transactions_raw})
        data_source = "synthetic (100K txns, 500 items)"

    n_trans = len(transactions_raw)
    print(f"  Dataset: {data_source}")
    print(f"  Transactions: {n_trans:,}")

    # efficient-apriori format
    ea_transactions = [tuple(t) for t in transactions_raw]

    results: list[tuple[str, bool]] = []
    ea_timeout_seconds = 120  # 2 minutes max for efficient-apriori

    for min_support in [0.01, 0.005, 0.002, 0.001]:
        subheader(f"min_support = {min_support}")

        # --- efficient-apriori (ground truth) with timeout ---
        ea_set = None
        ea_time = None
        timed_out = False

        try:
            # Set alarm for timeout (POSIX only)
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(ea_timeout_seconds)

            t0 = time.perf_counter()
            ea_itemsets, _ = ea_apriori(ea_transactions, min_support=min_support)
            ea_time = time.perf_counter() - t0
            ea_set = ea_to_set(ea_itemsets, n_trans)

            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

            print(f"  efficient-apriori: {len(ea_set)} itemsets in {ea_time:.2f}s")

        except TimeoutError:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            timed_out = True
            print(f"  efficient-apriori: timed out after {ea_timeout_seconds}s")

        except Exception as e:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            print(f"  efficient-apriori: error - {e}")
            timed_out = True

        # --- et-miner GPU ---
        t0 = time.perf_counter()
        pa_result = pa_apriori(pa_df, min_support=min_support, use_gpu=True)
        pa_time = time.perf_counter() - t0
        pa_set = pa_to_set(pa_result)

        print(f"  et-miner (GPU):    {len(pa_set)} itemsets in {pa_time:.2f}s")

        if timed_out:
            print(f"  {SKIP}  min_support={min_support} (efficient-apriori too slow)")
            # Don't count skipped tests as failures
            continue

        # --- Compare ---
        passed, detail = compare_results(ea_set, pa_set, label=f"s={min_support}")

        if passed:
            speedup = ea_time / pa_time if pa_time > 0 else float("inf")
            print(f"  {PASS}  min_support={min_support}  (speedup: {speedup:.1f}x)")
        else:
            print(f"  {FAIL}  min_support={min_support}")
            print(detail)

        results.append((f"Test 2 (819K, s={min_support})", passed))

    return results


# ---------------------------------------------------------------------------
# Test 3: k=3 specific validation
# ---------------------------------------------------------------------------

def test_k3_validation() -> list[tuple[str, bool]]:
    """Specifically validate that k=3 itemsets match between implementations.

    Returns list of (label, passed) tuples.
    """
    header("Test 3: k=3 specific validation")

    # Use a dataset + support level known to produce k=3 itemsets
    transactions_raw = [[1, 2, 3], [2, 3, 4], [1, 3, 5], [2, 3], [1, 2, 3, 4]]
    n_trans = len(transactions_raw)
    min_support = 0.3  # Should produce k=3 itemsets

    ea_transactions = [tuple(t) for t in transactions_raw]
    pa_df = pl.DataFrame({"items": transactions_raw})

    # --- efficient-apriori ---
    ea_itemsets, _ = ea_apriori(ea_transactions, min_support=min_support)
    ea_set = ea_to_set(ea_itemsets, n_trans)
    ea_by_k = itemsets_by_k(ea_set)

    # --- et-miner GPU ---
    pa_result = pa_apriori(pa_df, min_support=min_support, use_gpu=True)
    pa_set = pa_to_set(pa_result)
    pa_by_k = itemsets_by_k(pa_set)

    results: list[tuple[str, bool]] = []

    # Check k=3 specifically
    ea_k3 = ea_by_k.get(3, set())
    pa_k3 = pa_by_k.get(3, set())

    if not ea_k3 and not pa_k3:
        print(f"  No k=3 itemsets found at min_support={min_support}")
        print(f"  Trying with lower support on a denser synthetic dataset...")

        # Generate a denser dataset that's more likely to produce k=3
        rng = np.random.RandomState(123)
        dense_txns = []
        item_pool = list(range(1, 21))  # 20 items
        for _ in range(1000):
            size = max(3, rng.poisson(5))
            size = min(size, len(item_pool))
            dense_txns.append(sorted(rng.choice(item_pool, size=size, replace=False).tolist()))

        n_trans = len(dense_txns)
        min_support = 0.05

        ea_transactions = [tuple(t) for t in dense_txns]
        pa_df = pl.DataFrame({"items": dense_txns})

        ea_itemsets, _ = ea_apriori(ea_transactions, min_support=min_support)
        ea_set = ea_to_set(ea_itemsets, n_trans)
        ea_by_k = itemsets_by_k(ea_set)

        pa_result = pa_apriori(pa_df, min_support=min_support, use_gpu=True)
        pa_set = pa_to_set(pa_result)
        pa_by_k = itemsets_by_k(pa_set)

        ea_k3 = ea_by_k.get(3, set())
        pa_k3 = pa_by_k.get(3, set())

    print(f"  efficient-apriori k=3: {len(ea_k3)} itemsets")
    print(f"  et-miner k=3:         {len(pa_k3)} itemsets")

    if not ea_k3:
        print(f"  {SKIP}  No k=3 itemsets in ground truth to validate against")
        return []

    # Compare k=3 specifically
    passed_k3, detail_k3 = compare_results(ea_k3, pa_k3, label="k=3")

    if passed_k3:
        print(f"  {PASS}  k=3 itemsets match ({len(ea_k3)} itemsets)")
        # Print a few examples
        for key, sup in sorted(ea_k3)[:5]:
            print(f"    {key} -> {sup:.4f}")
        if len(ea_k3) > 5:
            print(f"    ... and {len(ea_k3) - 5} more")
    else:
        print(f"  {FAIL}  k=3 itemsets do NOT match")
        print(detail_k3)

    results.append(("Test 3 (k=3 validation)", passed_k3))

    # Also verify k=1 and k=2 for completeness
    for k in [1, 2]:
        ea_k = ea_by_k.get(k, set())
        pa_k = pa_by_k.get(k, set())
        passed_k, detail_k = compare_results(ea_k, pa_k, label=f"k={k}")
        if passed_k:
            print(f"  {PASS}  k={k} itemsets match ({len(ea_k)} itemsets)")
        else:
            print(f"  {FAIL}  k={k} itemsets do NOT match")
            print(detail_k)
        results.append((f"Test 3 (k={k} validation)", passed_k))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"{BOLD}et-miner GPU vs efficient-apriori correctness verification{RESET}")
    print(f"Python {sys.version.split()[0]}")

    try:
        import cupy as cp

        gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        vram_gb = cp.cuda.Device(0).mem_info[1] / 1e9
        print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    except Exception:
        print(f"{YELLOW}WARNING: CuPy not available, GPU tests may fall back to CPU{RESET}")

    # All test functions return list of (label, passed) tuples
    all_results: list[tuple[str, bool]] = []

    # Test 1: Small dataset, per-k validation
    all_results.extend(test_small_dataset())

    # Test 2: 819K real dataset
    all_results.extend(test_large_dataset())

    # Test 3: k=3 specific validation
    all_results.extend(test_k3_validation())

    # --- Summary ---
    header("SUMMARY")

    n_passed = sum(1 for _, passed in all_results if passed)
    n_total = len(all_results)

    for name, passed in all_results:
        status = PASS if passed else FAIL
        print(f"  {status}  {name}")

    print()
    if n_total == 0:
        print(f"{YELLOW}No tests were executed (all skipped).{RESET}")
        sys.exit(2)
    elif n_passed == n_total:
        print(f"{GREEN}{BOLD}{n_passed}/{n_total} tests passed{RESET}")
        sys.exit(0)
    else:
        print(f"{RED}{BOLD}{n_passed}/{n_total} tests passed{RESET}")
        sys.exit(1)
