#!/usr/bin/env python3
"""Pre-flight (Phase 0.3b): validate the multi-GPU row-split engine.

The null-model at min_count=8 runs 100 permutations through
``_apriori_row_split_multi_gpu`` (all GPUs per permutation, row-split NCCL sum).
That path has no end-to-end test in the suite (``test_gpu_dispatch`` is mock-only),
so this script IS the test: mine a subset single-GPU and multi-GPU and assert the
two produce byte-identical frequent itemsets + support counts.

Rationale: ``apriori(n_gpus=1)`` uses the single-GPU counting path;
``apriori(n_gpus>1)`` dispatches to ``_apriori_row_split_multi_gpu`` (apriori.py:2848).
Comparing the two exercises the exact production entry point the null uses.

Exit 0 (GREEN)  → row-split is exact; run the 100-perm null with --n-gpus N.
Exit 1 (RED)    → row-split diverges; fall back to --n-gpus 1 for the null.

Usage (on the GPU box, needs >= 2 GPUs):
    python3 validate_row_split.py \
        --data /workspace/data/transactions_214m_base.parquet \
        --subset-size 1000000 --min-count 50 --n-gpus 2 -v
"""

import argparse
import sys
import time
from pathlib import Path

import polars as pl
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import k_distribution  # noqa: E402

from et_miner import apriori  # noqa: E402


def _canonical(result: pl.DataFrame) -> pl.DataFrame:
    """Sort items within each itemset and rows deterministically for comparison.

    Item order inside an itemset and row order are implementation details of the
    counting engine; equality must hold on the *set* of (itemset, support) pairs.
    """
    return (
        result.with_columns(pl.col("itemset").list.sort().alias("itemset"))
        .with_columns(pl.col("itemset").cast(pl.List(pl.Int64)))
        .sort(by=[pl.col("itemset").list.len(), "itemset", "support"])
        .select("itemset", "support")
    )


def _first_divergence(ref: pl.DataFrame, got: pl.DataFrame, limit: int = 10) -> None:
    """Log the first few itemsets that differ between the two engines."""
    ref_map = {tuple(r): s for r, s in zip(ref["itemset"].to_list(), ref["support"].to_list())}
    got_map = {tuple(r): s for r, s in zip(got["itemset"].to_list(), got["support"].to_list())}

    only_ref = sorted(set(ref_map) - set(got_map), key=lambda t: (len(t), t))
    only_got = sorted(set(got_map) - set(ref_map), key=lambda t: (len(t), t))
    mismatched = sorted(
        (t for t in set(ref_map) & set(got_map) if ref_map[t] != got_map[t]),
        key=lambda t: (len(t), t),
    )

    if only_ref:
        logger.error(f"  {len(only_ref):,} itemsets ONLY in single-GPU (missing from row-split):")
        for t in only_ref[:limit]:
            logger.error(f"    K={len(t)} {list(t)} support={ref_map[t]}")
    if only_got:
        logger.error(f"  {len(only_got):,} itemsets ONLY in row-split (spurious):")
        for t in only_got[:limit]:
            logger.error(f"    K={len(t)} {list(t)} support={got_map[t]}")
    if mismatched:
        logger.error(f"  {len(mismatched):,} itemsets with DIFFERENT support counts:")
        for t in mismatched[:limit]:
            logger.error(f"    K={len(t)} {list(t)} single={ref_map[t]} row-split={got_map[t]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate multi-GPU row-split == single-GPU")
    parser.add_argument("--data", required=True, help="Transactions parquet (items: List[Int])")
    parser.add_argument(
        "--subset-size", type=int, default=1_000_000,
        help="First N transactions to mine (default: %(default)s)",
    )
    parser.add_argument(
        "--min-count", type=int, default=50,
        help="Absolute support threshold on the subset (default: %(default)s). "
        "Lower reaches deeper K but costs more; the depth only needs K>=3 to "
        "exercise the row-split candidate merge.",
    )
    parser.add_argument("--n-gpus", type=int, default=2, help="GPUs for the row-split path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Pre-flight gate: the Rust extension must import, else closed-filtering falls
    # back to the slow Python loop at 26M+ itemsets. Fail loud here, not mid-run.
    try:
        import et_miner_rust  # noqa: F401
        logger.info("Rust extension (et_miner_rust): OK")
    except ImportError as e:
        logger.error(f"et_miner_rust import FAILED: {e}")
        logger.error("maturin build did not land — closed-filtering would be O(n^2) Python. Abort.")
        return 1

    import cupy as cp

    n_avail = cp.cuda.runtime.getDeviceCount()
    if n_avail < args.n_gpus:
        logger.error(f"Need {args.n_gpus} GPUs for the row-split path, only {n_avail} present.")
        return 1
    logger.info(f"GPUs available: {n_avail} (validating row-split with n_gpus={args.n_gpus})")

    logger.info(f"Loading first {args.subset_size:,} transactions from {args.data}...")
    df = (
        pl.scan_parquet(args.data)
        .select("items")
        .head(args.subset_size)
        .collect()
    )
    n_tx = len(df)
    if n_tx == 0:
        logger.error(f"No transactions loaded from {args.data} (empty or wrong path). Abort.")
        return 1
    min_support = args.min_count / n_tx
    logger.info(
        f"Subset: {n_tx:,} transactions | min_count={args.min_count} "
        f"→ min_support={min_support:.3e}"
    )

    # ── Reference: single-GPU ────────────────────────────────────────────
    logger.info("Mining single-GPU (n_gpus=1) — reference truth...")
    t0 = time.perf_counter()
    ref = apriori(df, min_support=min_support, use_gpu=True, max_length=None, n_gpus=1)
    logger.info(f"  {len(ref):,} itemsets in {time.perf_counter() - t0:.1f}s")

    # ── Candidate: multi-GPU row-split ───────────────────────────────────
    logger.info(f"Mining row-split (n_gpus={args.n_gpus}) — via _apriori_row_split_multi_gpu...")
    t0 = time.perf_counter()
    got = apriori(df, min_support=min_support, use_gpu=True, max_length=None, n_gpus=args.n_gpus)
    logger.info(f"  {len(got):,} itemsets in {time.perf_counter() - t0:.1f}s")

    # ── Compare ──────────────────────────────────────────────────────────
    ref_dist, got_dist = k_distribution(ref), k_distribution(got)
    logger.info("K-distribution (single-GPU vs row-split):")
    for k in sorted(set(ref_dist) | set(got_dist)):
        flag = "" if ref_dist.get(k, 0) == got_dist.get(k, 0) else "  <-- MISMATCH"
        logger.info(f"  K={k:>2}: {ref_dist.get(k, 0):>10,} vs {got_dist.get(k, 0):>10,}{flag}")

    # Gate: the comparison is only meaningful if the reference actually exercised
    # the row-split candidate-merge path. Empty==empty and shallow (K=1-only)
    # results compare equal, so a too-high min_count would print a false GREEN.
    if len(ref) == 0:
        logger.error(
            "INCONCLUSIVE: single-GPU reference produced ZERO itemsets — min_count too "
            "high (or subset too small). The row-split path was NOT validated "
            "(empty==empty compares equal). Lower --min-count or raise --subset-size. "
            "Falling back to --n-gpus 1."
        )
        return 1
    ref_kmax = max(ref_dist)  # ref non-empty here
    if ref_kmax < 3:
        logger.error(
            f"INCONCLUSIVE: reference reached only K={ref_kmax}; the K>=3 row-split "
            "candidate merge (count_k3plus / build_k3plus_groups) was never exercised, "
            "so a bug isolated to that path is invisible. Lower --min-count to reach "
            "K>=3. Falling back to --n-gpus 1."
        )
        return 1

    ref_c, got_c = _canonical(ref), _canonical(got)
    identical = ref_c.equals(got_c)

    if identical:
        logger.success(
            f"GREEN: row-split (n_gpus={args.n_gpus}) is EXACT vs single-GPU "
            f"({len(ref):,} itemsets, all support counts match). "
            f"Safe to run the 100-perm null with --n-gpus {args.n_gpus}."
        )
        return 0

    logger.error(f"RED: row-split (n_gpus={args.n_gpus}) DIVERGES from single-GPU.")
    _first_divergence(ref_c, got_c)
    logger.error("Fall back to --n-gpus 1 for the null-model runs (no NCCL, serial, slower).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
