#!/usr/bin/env python3
"""Experiment: Null model permutation test for K-distribution significance.

Tests whether the observed K-distribution peak at K=9 (3.53M itemsets, 13.15%)
is biologically significant or a statistical artifact of the dataset structure.

Null model design:
  - Preserve: per-protein feature COUNT (each protein keeps same number of features)
  - Preserve: per-feature global frequency (each feature appears same total times)
  - Randomize: which features are assigned to which protein
  - Method: Explode transactions → shuffle item column → re-aggregate → dedup

The global shuffle preserves both marginals simultaneously (Fisher-Yates on the
flat item vector). After shuffle, duplicate items within a protein are removed
(expected ~0.3% of proteins affected given avg 4 items from 1,002 unique).

Reference K-distribution (Opus run, min_count=8, 26.8M itemsets):
  K=1:    1,002   K=12: 1,996,772
  K=2:   73,786   K=13: 1,259,045
  K=3:  452,777   K=14:   679,471
  K=4: 1,184,461  K=15:   310,527
  K=5: 1,974,126  K=16:   118,659
  K=6: 2,626,332  K=17:    37,261
  K=7: 3,118,459  K=18:     9,375
  K=8: 3,442,954  K=19:     1,818
  K=9: 3,529,257  K=20:       255
  K=10: 3,293,612 K=21:        23
  K=11: 2,739,532 K=22:         1
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from et_miner import apriori

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_transactions, k_distribution


def _prepare_gpu_resident_data(df: pl.DataFrame) -> tuple:
    """Build GPU-persistent CSR arrays from transactions DataFrame.

    Explodes item lists into flat CSR format (indptr + indices). These arrays
    persist on GPU across permutations — only indices_gpu gets shuffled each time.

    Returns:
        (indptr_np, items_np, n_cols, col_to_item, n_transactions)
    """
    lengths = df["items"].list.len().to_numpy()
    indptr = np.zeros(len(df) + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])

    items = df.explode("items")["items"].to_numpy().astype(np.int64)

    n_cols = int(items.max()) + 1
    col_to_item = {i: i for i in range(n_cols)}
    n_transactions = len(df)

    logger.info(f"  GPU-resident prep: {n_transactions:,} proteins, {len(items):,} items, "
          f"{n_cols} unique features, {items.nbytes / 1e9:.1f} GB items + "
          f"{indptr.nbytes / 1e9:.1f} GB indptr")

    return indptr, items, n_cols, col_to_item, n_transactions


def shuffle_transactions(df: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Create null model by shuffling feature identities while preserving marginals.

    Explodes all transactions into flat (row_idx, item) pairs, shuffles the item
    column via Fisher-Yates permutation, then re-aggregates per protein. This
    preserves both per-protein feature counts AND per-feature global frequencies.

    Uses CuPy GPU shuffle when available (~0.1s vs ~22s on CPU for 316M items).
    Falls back to NumPy CPU shuffle if CuPy is unavailable.

    After shuffle, duplicate items within a protein are removed (dedup). This
    slightly reduces some proteins' feature counts (~0.3% affected).
    """
    t0 = time.perf_counter()

    # Add row index for re-aggregation
    indexed = df.with_row_index("idx")

    # Explode to flat (idx, item) pairs
    flat = indexed.explode("items")
    n_pairs = len(flat)

    # Shuffle item column — GPU path (CuPy) or CPU fallback (NumPy)
    items_np = flat["items"].to_numpy().copy()

    try:
        import cupy as cp
        # Derive deterministic seed from numpy rng (advances rng state)
        gpu_seed = int(rng.integers(0, 2**63))
        items_gpu = cp.asarray(items_np)
        # CuPy 14's Generator (default_rng) has no shuffle/permutation; RandomState
        # does and is deterministic per seed — identical across the worker and
        # sequential paths since both derive gpu_seed the same way.
        gpu_rng = cp.random.RandomState(gpu_seed)
        gpu_rng.shuffle(items_gpu)
        items_np = cp.asnumpy(items_gpu)
        del items_gpu
        cp.get_default_memory_pool().free_all_blocks()
        shuffle_method = "GPU"
    except (ImportError, Exception) as e:
        rng.shuffle(items_np)
        shuffle_method = f"CPU (fallback: {e})" if not isinstance(e, ImportError) else "CPU"

    # Replace items column with shuffled values
    flat = flat.with_columns(pl.Series("items", items_np))

    # Re-aggregate per protein, dedup items within each protein
    null_df = (
        flat
        .group_by("idx")
        .agg(pl.col("items").unique())
        .sort("idx")
        .select("items")
    )

    elapsed = time.perf_counter() - t0

    # Report dedup stats
    original_lengths = df["items"].list.len()
    null_lengths = null_df["items"].list.len()
    n_affected = (original_lengths != null_lengths).sum()
    pct_affected = n_affected / len(df) * 100

    logger.info(f"  Shuffle [{shuffle_method}]: {n_pairs:,} pairs shuffled in {elapsed:.1f}s "
          f"(dedup affected {n_affected:,} proteins, {pct_affected:.2f}%)")

    return null_df


def compute_statistics(
    real_dist: dict[int, int],
    null_dists: list[dict[int, int]],
) -> dict[int, dict]:
    """Compute per-K z-scores and p-values from null distribution."""
    all_k = sorted(set(real_dist.keys()) | {k for d in null_dists for k in d})
    stats = {}

    for k in all_k:
        real_count = real_dist.get(k, 0)
        null_counts = [d.get(k, 0) for d in null_dists]
        null_mean = np.mean(null_counts)
        null_std = np.std(null_counts, ddof=1) if len(null_counts) > 1 else 0.0

        if null_std > 0:
            z_score = (real_count - null_mean) / null_std
        else:
            z_score = float("inf") if real_count > null_mean else (
                float("-inf") if real_count < null_mean else 0.0
            )

        # One-sided p-value (normal approximation): P(X >= real | null)
        if null_std > 0:
            from scipy.stats import norm
            p_value = 1 - norm.cdf(z_score)
        else:
            p_value = 0.0 if real_count > null_mean else 1.0

        stats[k] = {
            "real": real_count,
            "null_mean": round(float(null_mean), 1),
            "null_std": round(float(null_std), 1),
            "z_score": round(float(z_score), 2) if np.isfinite(z_score) else str(z_score),
            "p_value": float(p_value),
            "direction": "enriched" if z_score > 0 else "depleted",
            "significant": bool(p_value < 0.05) if np.isfinite(z_score) else (real_count > null_mean),
        }

    return stats


def _worker_run_permutations(args):
    """Run permutations on a single GPU — fully GPU-resident.

    Keeps items + indptr persistent in VRAM across all permutations.
    Per permutation: GPU shuffle → GPU bitvec build → apriori bitvecs fast path.
    Dedup is free via atomicOr idempotency in the bitvec kernel.
    """
    gpu_id, seeds, data_path, min_support = args

    # Pin GPU BEFORE any CUDA import/initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import cupy as cp
    from et_miner.gpu.csr_bitvec import build_bitvecs_from_gpu_arrays

    df = load_transactions(data_path, min_items=2)
    indptr_np, items_np, n_cols, col_to_item, n_transactions = _prepare_gpu_resident_data(df)
    del df  # free CPU DataFrame

    # Transfer to GPU once — persistent across all permutations
    indptr_gpu = cp.asarray(indptr_np)
    items_orig_gpu = cp.asarray(items_np)
    items_gpu = cp.empty_like(items_orig_gpu)  # working copy for shuffle
    del indptr_np, items_np

    logger.info(f"  [GPU {gpu_id}] GPU-resident: {items_orig_gpu.nbytes / 1e9:.1f} GB items + "
          f"{indptr_gpu.nbytes / 1e9:.1f} GB indptr persistent in VRAM")

    results = []

    for i, seed in enumerate(seeds):
        perm_t0 = time.perf_counter()

        # Restore original items and shuffle on GPU
        cp.copyto(items_gpu, items_orig_gpu)
        rng = np.random.default_rng(seed)
        gpu_seed = int(rng.integers(0, 2**63))
        # CuPy 14's Generator (default_rng) has no shuffle/permutation; RandomState
        # does and is deterministic per seed — identical across the worker and
        # sequential paths since both derive gpu_seed the same way.
        gpu_rng = cp.random.RandomState(gpu_seed)
        gpu_rng.shuffle(items_gpu)

        # Build bitvecs from GPU-resident arrays (atomicOr handles dedup)
        bitvecs_gpu = build_bitvecs_from_gpu_arrays(
            indptr_gpu, items_gpu, n_transactions, n_cols,
        )

        logger.info(f"  [GPU {gpu_id}] Perm {i+1}/{len(seeds)} (seed={seed}): "
              f"shuffle+bitvec {time.perf_counter() - perm_t0:.2f}s, mining...")

        # Mine using bitvecs fast path — skips DataFrame/CSR/transfer pipeline
        t0 = time.perf_counter()
        null_result = apriori(
            bitvecs=(bitvecs_gpu, col_to_item, n_transactions),
            min_support=min_support,
            max_length=None,
        )
        mine_time = time.perf_counter() - t0

        del bitvecs_gpu
        cp.get_default_memory_pool().free_all_blocks()

        null_dist = k_distribution(null_result)
        null_total = len(null_result)
        max_k = max(null_dist.keys()) if null_dist else 0
        total_time = time.perf_counter() - perm_t0

        results.append({
            "seed": int(seed),
            "k_distribution": null_dist,
            "total_itemsets": null_total,
            "time_seconds": round(total_time, 2),
            "mine_seconds": round(mine_time, 2),
            "max_k": max_k,
            "gpu_id": gpu_id,
        })

        logger.info(f"  [GPU {gpu_id}] seed={seed}: {null_total:,} itemsets, "
              f"max K={max_k}, mine={mine_time:.1f}s, total={total_time:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Null model permutation test for K-distribution significance",
    )
    parser.add_argument(
        "--data", default="/workspace/data/transactions_35k.parquet",
        help="Path to transactions parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of null model permutations (default: %(default)s)",
    )
    parser.add_argument(
        "--min-count", type=int, default=1090,
        help="Minimum count threshold (default: 1090 = 0.001%% of 109M proteins)",
    )
    parser.add_argument(
        "--output-dir", default="results_35k",
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--n-gpus", type=int, default=1,
        help="Number of GPUs for parallel permutations (default: %(default)s)",
    )
    parser.add_argument(
        "--perm-per-gpu", action="store_true",
        help="Run one permutation per GPU across n_gpus workers (spawn Pool), instead of "
        "row-split (all GPUs per perm, sequential). ~n_gpus x faster when the per-perm "
        "bitvec fits on a single GPU (base vocab). Row-split remains the default for "
        "large-vocab runs whose bitvec needs splitting across GPUs.",
    )
    parser.add_argument(
        "--checkpoint", action="store_true",
        help="Enable JSONL checkpointing for crash recovery",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint file, skipping completed permutations",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )
    args = parser.parse_args()

    master_rng = np.random.default_rng(args.seed)

    # Pre-generate deterministic seeds for all permutations.
    # This ensures identical results regardless of n_gpus.
    permutation_seeds = [int(master_rng.integers(0, 2**63)) for _ in range(args.runs)]

    # Checkpoint setup
    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / "null_model_checkpoint.jsonl"
    completed_seeds: set[int] = set()
    resumed_results: list[dict] = []

    if args.resume and checkpoint_file.exists():
        for line in checkpoint_file.read_text().strip().split("\n"):
            if line:
                entry = json.loads(line)
                completed_seeds.add(entry["seed"])
                resumed_results.append(entry)
        logger.info(f"  Resumed: {len(completed_seeds)} completed permutations from checkpoint")

    # Filter out already-completed seeds
    remaining_seeds = [s for s in permutation_seeds if s not in completed_seeds]
    if not remaining_seeds and completed_seeds:
        logger.info(f"  All {len(permutation_seeds)} permutations already completed!")
    elif completed_seeds:
        logger.info(f"  Remaining: {len(remaining_seeds)}/{len(permutation_seeds)} permutations")

    # Load and filter (main process — for sequential path and real mining)
    df = load_transactions(args.data, min_items=2)
    n_transactions = len(df)

    min_count = args.min_count
    min_support = min_count / n_transactions

    logger.info(f"\n{'='*60}")
    logger.info("Null Model Permutation Test")
    logger.info(f"{'='*60}")
    logger.info(f"  min_count:      {min_count}")
    logger.info(f"  min_support:    {min_support:.10f} ({min_support*100:.4f}%)")
    logger.info(f"  n_transactions: {n_transactions:,}")
    logger.info(f"  permutations:   {args.runs}")
    logger.info(f"  n_gpus:         {args.n_gpus}")
    logger.info(f"  seed:           {args.seed}")

    # Real K-distribution — always mine (no hardcoded lookup, works with any feature set).
    # Force single-GPU: the multi-GPU row-split diverges from single-GPU at K>=8 (the
    # pre-flight caught over-counting), so the row-split path is unsafe for exact counts.
    # The real-dist bitvec fits on one 80GB H100, and the null perms use the single-GPU
    # per-worker path (--perm-per-gpu), so real and null are both exact and comparable.
    logger.info(f"\nMining REAL distribution (min_count={min_count}, single-GPU for exactness)...")
    t0 = time.perf_counter()
    real_result = apriori(df, min_support=min_support, use_gpu=True, max_length=None,
                          n_gpus=1)
    real_time = time.perf_counter() - t0
    real_dist = k_distribution(real_result)
    real_total = len(real_result)
    logger.info(f"  Real mining: {real_total:,} itemsets in {real_time:.1f}s")

    # Print real distribution
    logger.info("\n  Real K-distribution:")
    for k in sorted(real_dist.keys()):
        pct = real_dist[k] / real_total * 100
        logger.info(f"    K={k:>2}: {real_dist[k]:>10,} ({pct:>5.2f}%)")

    # Null model permutations — integrate resumed results
    null_dists: list[dict[int, int]] = []
    null_totals: list[int] = []
    null_times: list[float] = []

    for entry in resumed_results:
        # JSON deserialize turns int keys to strings — convert back
        null_dists.append({int(k): v for k, v in entry["k_distribution"].items()})
        null_totals.append(entry["total_itemsets"])
        null_times.append(entry["time_seconds"])

    wall_t0 = time.perf_counter()

    if not remaining_seeds:
        logger.info("\nAll permutations completed (from checkpoint). Skipping to statistics.")

    elif args.n_gpus > 1 and args.perm_per_gpu:
        # ── ONE-PERMUTATION-PER-GPU: n_gpus workers, each mining a disjoint seed
        # subset on a single pinned GPU. For base vocab the per-perm bitvec (~10-11 GB)
        # fits on one 80 GB H100, so this is ~n_gpus x faster than row-split
        # (all-GPUs-per-perm, sequential): no NCCL, no cross-GPU barrier, no
        # shared-interpreter stall during serial per-level orchestration. (workflow finding 3)
        import cupy as cp

        # Free the parent's real-distribution mining VRAM before spawning workers so
        # each single-GPU worker gets full per-GPU headroom.
        del real_result
        for _did in range(args.n_gpus):
            try:
                with cp.cuda.Device(_did):
                    cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        del df  # each worker reloads from data_path itself

        # Round-robin seed assignment: deterministic and balanced across GPUs.
        seeds_per_gpu = [remaining_seeds[g::args.n_gpus] for g in range(args.n_gpus)]
        worker_args = [
            (g, seeds_per_gpu[g], args.data, min_support)
            for g in range(args.n_gpus) if seeds_per_gpu[g]
        ]
        logger.info(
            f"\nRunning {len(remaining_seeds)} permutations one-per-GPU across "
            f"{args.n_gpus} GPUs ({[len(s) for s in seeds_per_gpu]} seeds/GPU)..."
        )

        # spawn (NOT fork): CUDA contexts cannot be forked; the worker's
        # CUDA_VISIBLE_DEVICES pin only takes effect in a fresh interpreter.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(worker_args)) as pool:
            for batch in pool.imap_unordered(_worker_run_permutations, worker_args):
                for entry in batch:
                    null_dists.append(entry["k_distribution"])
                    null_totals.append(entry["total_itemsets"])
                    # Use MINE time (not total) for parity with the sequential/row-split
                    # entries, which store mine_time in null_times + checkpoint.
                    null_times.append(entry["mine_seconds"])
                    if args.checkpoint or args.resume:
                        ckpt = {
                            "seed": entry["seed"],
                            "k_distribution": entry["k_distribution"],
                            "total_itemsets": entry["total_itemsets"],
                            "time_seconds": entry["mine_seconds"],
                            "max_k": entry["max_k"],
                        }
                        with open(checkpoint_file, "a") as f:
                            f.write(json.dumps(ckpt, default=str) + "\n")
                logger.info(f"  GPU batch done: {len(batch)} perms "
                            f"(collected {len(null_dists)}/{args.runs})")

    elif args.n_gpus > 1:
        # ── ROW-SPLIT: GPU-resident shuffle + direct bitvec build ──
        # GPU shuffle ~0.9s vs Polars explode ~84s = 93x faster per perm.
        # Items shuffled on GPU, D2H'd, then bitvecs built row-split from raw arrays.
        # atomicOr in bitvec kernel handles dedup implicitly (idempotent bit-set).
        import cupy as cp
        from et_miner.gpu.csr_bitvec import build_bitvecs_row_split_from_arrays
        from et_miner.gpu.row_split import _apriori_row_split_multi_gpu

        logger.info(f"\nRunning {len(remaining_seeds)} permutations using {args.n_gpus}-GPU "
              f"row-split (GPU-resident shuffle, all GPUs per perm)...")

        # One-time CSR prep — items_np stays on CPU, copied to GPU per perm
        indptr_np, items_np, n_cols, col_to_item, n_tx = _prepare_gpu_resident_data(df)
        del df

        n_done = len(completed_seeds)
        for run_idx, seed in enumerate(remaining_seeds):
            global_idx = n_done + run_idx
            logger.info(f"\n{'─'*60}")
            logger.info(f"Null model run {global_idx + 1}/{args.runs} (seed={seed})")
            logger.info(f"{'─'*60}")

            perm_t0 = time.perf_counter()

            # GPU shuffle: H2D → shuffle → D2H (~0.9s for 1.36B items)
            rng = np.random.default_rng(seed)
            gpu_seed = int(rng.integers(0, 2**63))
            items_gpu = cp.asarray(items_np)
            gpu_rng = cp.random.RandomState(gpu_seed)  # CuPy 14 Generator lacks shuffle
            gpu_rng.shuffle(items_gpu)
            items_shuffled = cp.asnumpy(items_gpu)
            del items_gpu
            cp.get_default_memory_pool().free_all_blocks()
            shuffle_time = time.perf_counter() - perm_t0
            logger.info(f"  GPU shuffle+D2H: {shuffle_time:.1f}s")

            # Build row-split bitvecs from raw arrays (no scipy, no Polars)
            t_bv = time.perf_counter()
            bitvecs_list = build_bitvecs_row_split_from_arrays(
                indptr_np, items_shuffled, n_tx, n_cols, args.n_gpus,
            )
            del items_shuffled
            logger.info(f"  Bitvec build: {time.perf_counter() - t_bv:.1f}s")

            # Mine using row-split with pre-built bitvecs
            t0 = time.perf_counter()
            null_result = _apriori_row_split_multi_gpu(
                csr=None, col_to_item=col_to_item, n_transactions=n_tx,
                min_support=min_support, max_length=None, n_gpus=args.n_gpus,
                bitvecs_list=bitvecs_list,
            )
            mine_time = time.perf_counter() - t0

            # Free bitvecs VRAM before next permutation
            del bitvecs_list
            for did in range(args.n_gpus):
                with cp.cuda.Device(did):
                    cp.get_default_memory_pool().free_all_blocks()

            null_dist = k_distribution(null_result)
            null_total = len(null_result)
            max_k_null = max(null_dist.keys()) if null_dist else 0
            total_time = time.perf_counter() - perm_t0

            null_times.append(mine_time)
            null_dists.append(null_dist)
            null_totals.append(null_total)

            logger.info(f"  {null_total:,} itemsets, max K={max_k_null}, "
                  f"mine={mine_time:.1f}s, total={total_time:.1f}s")

            # Checkpoint this permutation
            if args.checkpoint or args.resume:
                entry = {
                    "seed": int(seed),
                    "k_distribution": null_dist,
                    "total_itemsets": null_total,
                    "time_seconds": round(mine_time, 2),
                    "max_k": max_k_null,
                }
                with open(checkpoint_file, "a") as f:
                    f.write(json.dumps(entry, default=str) + "\n")

    else:
        # ── SEQUENTIAL: GPU-resident single-GPU path ──
        try:
            import cupy as cp
            from et_miner.gpu.csr_bitvec import build_bitvecs_from_gpu_arrays
            _has_cupy = True
        except ImportError:
            _has_cupy = False

        if _has_cupy:
            cp.get_default_memory_pool().free_all_blocks()
            indptr_np, items_np, n_cols, col_to_item, n_tx = _prepare_gpu_resident_data(df)
            indptr_gpu = cp.asarray(indptr_np)
            items_orig_gpu = cp.asarray(items_np)
            items_gpu = cp.empty_like(items_orig_gpu)
            del indptr_np, items_np
            logger.info(f"  GPU-resident mode: {items_orig_gpu.nbytes / 1e9:.1f} GB items in VRAM")

        n_done = len(completed_seeds)
        for run_idx, seed in enumerate(remaining_seeds):
            global_idx = n_done + run_idx
            logger.info(f"\n{'─'*60}")
            logger.info(f"Null model run {global_idx + 1}/{args.runs} (seed={seed})")
            logger.info(f"{'─'*60}")

            if _has_cupy:
                # GPU-resident path
                perm_t0 = time.perf_counter()
                cp.copyto(items_gpu, items_orig_gpu)
                rng = np.random.default_rng(seed)
                gpu_seed = int(rng.integers(0, 2**63))
                gpu_rng = cp.random.RandomState(gpu_seed)  # CuPy 14 Generator lacks shuffle
                gpu_rng.shuffle(items_gpu)

                bitvecs_gpu = build_bitvecs_from_gpu_arrays(
                    indptr_gpu, items_gpu, n_tx, n_cols,
                )
                logger.info(f"  Shuffle+bitvec: {time.perf_counter() - perm_t0:.2f}s")

                t0 = time.perf_counter()
                null_result = apriori(
                    bitvecs=(bitvecs_gpu, col_to_item, n_tx),
                    min_support=min_support, max_length=None,
                )
                mine_time = time.perf_counter() - t0
                del bitvecs_gpu
                cp.get_default_memory_pool().free_all_blocks()
            else:
                # CPU fallback path
                rng = np.random.default_rng(seed)
                null_df = shuffle_transactions(df, rng)
                t0 = time.perf_counter()
                null_result = apriori(
                    null_df, min_support=min_support, use_gpu=True,
                    max_length=None, n_gpus=args.n_gpus,
                )
                mine_time = time.perf_counter() - t0

            null_dist = k_distribution(null_result)
            null_total = len(null_result)
            max_k_null = max(null_dist.keys()) if null_dist else 0

            null_times.append(mine_time)
            null_dists.append(null_dist)
            null_totals.append(null_total)

            # Checkpoint this permutation
            if args.checkpoint or args.resume:
                entry = {
                    "seed": int(seed),
                    "k_distribution": null_dist,
                    "total_itemsets": null_total,
                    "time_seconds": round(mine_time, 2),
                    "max_k": max_k_null,
                }
                with open(checkpoint_file, "a") as f:
                    f.write(json.dumps(entry, default=str) + "\n")

            logger.info(f"  Null mining: {null_total:,} itemsets in {mine_time:.1f}s")
            logger.info(f"  Max K: {max_k_null}")

            for k in sorted(null_dist.keys()):
                pct = null_dist[k] / null_total * 100 if null_total > 0 else 0
                logger.info(f"    K={k:>2}: {null_dist[k]:>10,} ({pct:>5.2f}%)")

    wall_time = time.perf_counter() - wall_t0
    logger.info(f"\n  Wall-clock time for all permutations: {wall_time:.1f}s")

    # Compute statistics
    logger.info(f"\n{'='*60}")
    logger.info("Statistical Analysis")
    logger.info(f"{'='*60}")

    stats = compute_statistics(real_dist, null_dists)

    # Print comparison table
    header = (f"{'K':>3} {'Real':>12} {'Null Mean':>12} {'Null Std':>12} "
              f"{'Z-score':>10} {'p-value':>12} {'Sig':>5}")
    logger.info(f"\n{header}")
    logger.info(f"{'-'*len(header)}")

    for k in sorted(stats.keys()):
        s = stats[k]
        z_str = f"{s['z_score']:>10.2f}" if isinstance(s["z_score"], float) else f"{s['z_score']:>10}"
        p_str = f"{s['p_value']:>12.2e}" if s["p_value"] > 0 else f"{'<1e-300':>12}"
        dir_char = "+" if s["direction"] == "enriched" else "-"
        sig_str = f"{dir_char}***" if s["significant"] else ""
        logger.info(f"{k:>3} {s['real']:>12,} {s['null_mean']:>12.1f} {s['null_std']:>12.1f} "
              f"{z_str} {p_str} {sig_str:>5}")

    # Summary
    real_total_count = sum(real_dist.values())
    null_mean_total = np.mean(null_totals)
    total_ratio = real_total_count / null_mean_total if null_mean_total > 0 else float("inf")

    logger.info(f"\n{'='*60}")
    logger.info("Summary")
    logger.info(f"{'='*60}")
    logger.info(f"  Real total itemsets:      {real_total_count:>12,}")
    logger.info(f"  Null mean total itemsets: {null_mean_total:>12,.1f}")
    logger.info(f"  Ratio (real/null):        {total_ratio:>12.1f}x")
    logger.info(f"  Avg null mining time:     {np.mean(null_times):>12.1f}s")
    logger.info(f"  Total GPU-seconds:        {sum(null_times):>12.1f}s")
    logger.info(f"  Wall-clock time:          {wall_time:>12.1f}s")
    if args.n_gpus > 1:
        logger.info(f"  GPUs used:                {args.n_gpus:>12}")
        logger.info(f"  Parallel speedup:         {sum(null_times)/wall_time:>12.1f}x")

    # Interpretation — find the real peak K and analyze it
    if real_dist:
        peak_k = max(real_dist, key=real_dist.get)
        peak_stats = stats.get(peak_k, {})
        if peak_stats:
            logger.info(f"\n  Peak analysis (K={peak_k}, highest itemset count):")
            logger.info(f"    Real:      {peak_stats['real']:>12,}")
            logger.info(f"    Null mean: {peak_stats['null_mean']:>12.1f}")
            logger.info(f"    Z-score:   {peak_stats['z_score']}")
            z = peak_stats["z_score"]
            if isinstance(z, (int, float)) and z > 2:
                logger.info(f"    → K={peak_k} peak is BIOLOGICALLY SIGNIFICANT (Z > 2)")
            elif isinstance(z, str) and "inf" in z:
                logger.info(f"    → K={peak_k} peak is BIOLOGICALLY SIGNIFICANT (Z = inf)")
            else:
                logger.info(f"    → K={peak_k} peak is NOT significant (could be structural artifact)")

    # K=1 sanity check — null K=1 should match real K=1 (marginals preserved)
    real_k1 = real_dist.get(1, 0)
    k1_null_counts = [d.get(1, 0) for d in null_dists]
    k1_mean = np.mean(k1_null_counts)
    k1_tolerance = max(50, real_k1 * 0.05)  # 5% or ±50, whichever is larger
    logger.info(f"\n  K=1 sanity check (should be ~{real_k1:,}):")
    logger.info(f"    Real K=1:      {real_k1:,}")
    logger.info(f"    Null K=1 mean: {k1_mean:,.0f}")
    if abs(k1_mean - real_k1) < k1_tolerance:
        logger.info("    → Marginals correctly preserved ✓")
    else:
        logger.info(f"    → WARNING: K=1 deviates from expected {real_k1:,} — check shuffle")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"experiment_null_model_{timestamp}.json"

    output = {
        "experiment": "null_model_permutation_test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "min_support": min_support,
            "min_count": min_count,
            "n_transactions": n_transactions,
            "n_permutations": args.runs,
            "n_gpus": args.n_gpus,
            "seed": args.seed,
            "permutation_seeds": permutation_seeds,
            "recomputed_real": True,
        },
        "real_distribution": {str(k): v for k, v in real_dist.items()},
        "real_total": real_total_count,
        "null_runs": [
            {
                "run": i + 1,
                "total_itemsets": null_totals[i],
                "time_seconds": round(null_times[i], 2),
                "k_distribution": {str(k): v for k, v in null_dists[i].items()},
            }
            for i in range(len(null_dists))
        ],
        "statistics": {str(k): v for k, v in stats.items()},
        "summary": {
            "real_total": real_total_count,
            "null_mean_total": round(float(null_mean_total), 1),
            "ratio_real_vs_null": round(float(total_ratio), 2),
            "avg_null_mining_seconds": round(float(np.mean(null_times)), 2),
            "total_gpu_seconds": round(float(sum(null_times)), 2),
            "wall_clock_seconds": round(float(wall_time), 2),
            "n_gpus": args.n_gpus,
        },
    }

    output["real_mining_seconds"] = round(real_time, 2)

    output_file.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
