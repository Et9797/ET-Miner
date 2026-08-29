"""Association rule generation from frequent itemsets.

This module contains functions for generating association rules
from frequent itemsets with confidence and lift metrics.

Scalable functions (generate_rules_drop1, compute_self_sufficiency) use
PyArrow row-group iteration to process parquets that exceed u32 row limits
(K=8: 12B rows, 67 GB). The "drop-1" approach generates K rules per K-itemset
by dropping each item in turn, avoiding combinatorial explosion.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, combinations
from pathlib import Path

import polars as pl
from loguru import logger


@dataclass
class Rule:
    """An association rule with metrics."""

    lhs: list[int]  # Left-hand side (antecedent)
    rhs: list[int]  # Right-hand side (consequent)
    support: float
    confidence: float
    lift: float


def _powerset_nonempty(iterable: Iterable[int]) -> Iterator[tuple[int, ...]]:
    """Non-empty proper subsets (excludes the full set itself)."""
    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(1, len(items)))


def _build_support_lookup(frequent_itemsets: pl.DataFrame) -> dict[tuple[int, ...], float]:
    """Dict of itemset tuple -> support for O(1) subset lookups during rule generation."""
    # iter_rows() is 3x faster than to_dicts() at 100K itemsets (benched 2026-01-18)
    return {tuple(row["itemset"]): row["support"] for row in frequent_itemsets.iter_rows(named=True)}


def generate_rules(
    frequent_itemsets: pl.DataFrame,
    min_confidence: float = 0.5,
) -> list[Rule]:
    """Generate association rules from frequent itemsets.

    For each itemset of size >= 2, generates all possible rules by partitioning
    the itemset into antecedent (LHS) and consequent (RHS).

    Note: This function uses iter_rows() to iterate over itemsets because rule
    generation requires Python-level operations (subset enumeration) that cannot
    be expressed as Polars operations.

    Args:
        frequent_itemsets: DataFrame with columns "itemset" (List[Int64]) and "support" (Float64)
        min_confidence: Minimum confidence threshold (0.0-1.0)

    Returns:
        List of Rule objects meeting the confidence threshold
    """
    support_map = _build_support_lookup(frequent_itemsets)
    rules: list[Rule] = []

    for row in frequent_itemsets.iter_rows(named=True):
        itemset = row["itemset"]
        itemset_support = row["support"]

        if len(itemset) < 2:
            continue

        for lhs in _powerset_nonempty(itemset):
            rhs = tuple(x for x in itemset if x not in lhs)
            if not rhs:
                continue

            lhs_support = support_map.get(tuple(sorted(lhs)), 0.0)
            if lhs_support == 0.0:
                continue

            confidence = itemset_support / lhs_support
            if confidence < min_confidence:
                continue

            rhs_support = support_map.get(tuple(sorted(rhs)), 0.0)
            lift = confidence / rhs_support if rhs_support > 0.0 else 0.0

            rules.append(
                Rule(
                    lhs=list(sorted(lhs)),
                    rhs=list(sorted(rhs)),
                    support=itemset_support,
                    confidence=confidence,
                    lift=lift,
                )
            )

    return rules


# ---------------------------------------------------------------------------
# Scalable Polars-native rule generation for billion-row parquets
# ---------------------------------------------------------------------------


def _iter_row_groups(parquet_path: Path, chunk_size: int) -> Iterator[pl.DataFrame]:
    """Iterate over a parquet file (or partitioned directory) in row-group chunks.

    Supports both single-file and partitioned directory output
    (parallel flush: frequent_k{k}/part_*.parquet).

    Yields:
        pl.DataFrame chunks with the parquet's original schema.
    """
    import pyarrow.parquet as pq

    path = str(parquet_path)
    if Path(path).is_dir():
        # Partitioned output — iterate over each part file
        part_files = sorted(Path(path).glob("part_*.parquet"))
        for part_file in part_files:
            yield from _iter_single_file_row_groups(pq, part_file, chunk_size)
    else:
        yield from _iter_single_file_row_groups(pq, path, chunk_size)


def _iter_single_file_row_groups(pq, file_path, chunk_size: int) -> Iterator[pl.DataFrame]:
    """Iterate row groups from a single parquet file."""
    pf = pq.ParquetFile(str(file_path))
    n_groups = pf.metadata.num_row_groups

    batch_groups: list[int] = []
    batch_rows: int = 0

    for i in range(n_groups):
        rg_rows = pf.metadata.row_group(i).num_rows
        batch_groups.append(i)
        batch_rows += rg_rows

        if batch_rows >= chunk_size:
            table = pf.read_row_groups(batch_groups)
            yield pl.from_arrow(table)
            del table
            batch_groups = []
            batch_rows = 0

    if batch_groups:
        table = pf.read_row_groups(batch_groups)
        yield pl.from_arrow(table)
        del table


def _list_to_scalar_cols(col_name: str, k: int, prefix: str = "i") -> list[pl.Expr]:
    """Unpack fixed-length list column into K scalar cols — scalar hash joins >> list joins."""
    return [pl.col(col_name).list.get(j).alias(f"{prefix}{j}") for j in range(k)]


def _explode_drop1(chunk: pl.DataFrame, k: int) -> pl.DataFrame:
    """Explode a chunk of K-itemsets into K rows each, dropping one item per row.

    For an itemset [a, b, c, d, ...] of length K, produces K rows:
        row 0: dropped_item=a, antecedent=[b, c, d, ...]
        row 1: dropped_item=b, antecedent=[a, c, d, ...]
        ...

    This is the core of the "drop-1" approach: each row becomes a rule
    antecedent -> dropped_item with the original itemset's support.

    The antecedent list is already sorted because the input itemsets are
    sorted and we remove one element while preserving order.

    Args:
        chunk: DataFrame with columns "itemset" (list[i32]) and "support" (f64).
        k: Length of itemsets (determined once from first row, passed in).

    Returns:
        DataFrame with columns:
            - itemset: original K-itemset (list[i32])
            - support: original support (f64)
            - drop_idx: which position was dropped (i32)
            - dropped_item: the single dropped item (i32)
            - antecedent: the (K-1)-subset after removal (list[i32])
    """
    # Create K copies of each row, one per drop position
    drop_indices = pl.DataFrame(
        {"drop_idx": list(range(k))},
        schema={"drop_idx": pl.Int32},
    )
    exploded = chunk.join(drop_indices, how="cross")

    # Extract the dropped item: itemset[drop_idx]
    exploded = exploded.with_columns(
        pl.col("itemset").list.get(pl.col("drop_idx")).alias("dropped_item"),
    )

    # Build antecedent by removing the item at drop_idx.
    # For each drop position d, gather all indices except d.
    # This uses a chained when/then/otherwise to select the right gather mask.
    all_indices = list(range(k))

    # Start building the chained expression from the last drop position
    # so the first when() is drop_idx == 0 (most readable).
    keep_lists = {d: [i for i in all_indices if i != d] for d in range(k)}

    expr = pl.when(pl.col("drop_idx") == 0).then(pl.col("itemset").list.gather(keep_lists[0]))
    for d in range(1, k):
        expr = expr.when(pl.col("drop_idx") == d).then(pl.col("itemset").list.gather(keep_lists[d]))
    # Polars requires a final .otherwise(); this branch is unreachable.
    expr = expr.otherwise(pl.col("itemset").list.head(k - 1))

    exploded = exploded.with_columns(expr.alias("antecedent"))

    return exploded.select("itemset", "support", "drop_idx", "dropped_item", "antecedent")


def _detect_k(parquet_path: Path) -> int:
    """Detect K (itemset length) from the first row of a parquet file.

    Reads only the first row group's first row to determine K, avoiding
    any full-file scan.

    Args:
        parquet_path: Path to a frequent itemsets parquet.

    Returns:
        Integer K (itemset length).
    """
    import pyarrow.parquet as pq

    path = str(parquet_path)
    if Path(path).is_dir():
        # Partitioned output — read first part file
        first_part = sorted(Path(path).glob("part_*.parquet"))[0]
        pf = pq.ParquetFile(str(first_part))
    else:
        pf = pq.ParquetFile(path)
    first_rg = pf.read_row_groups([0], columns=["itemset"])
    first_list = first_rg.column("itemset")[0].as_py()
    return len(first_list)


def generate_rules_drop1(
    k_parquet: Path,
    k_minus1_parquet: Path,
    k1_parquet: Path | None = None,
    min_confidence: float = 0.5,
    chunk_size: int = 1_000_000,
) -> pl.DataFrame:
    """Generate association rules using the drop-1 approach on parquet files.

    For each K-itemset, generates K rules by dropping each item in turn.
    The antecedent is the (K-1)-subset, the consequent is the single dropped item.
    Joins with K-1 parquet to compute confidence, and optionally with K=1 for lift.

    This function is designed for billion-row parquets (K=8: 12B rows, 67 GB)
    that exceed Polars' u32 row limit. It uses PyArrow row-group iteration
    and processes chunks independently to keep memory bounded.

    The K and K-1 parquets must be sorted lexicographically (which is the default
    output of et-miner's apriori pipeline).

    Join strategy:
        The K-1 parquet is loaded once and its itemset list is unpacked into
        scalar columns (i0, i1, ..., i{K-2}) for efficient hash joining.
        For K-1 = K=7 (3.5B rows, 16.6 GB), this is loaded once via streaming
        collect and reused for every chunk. The scalar-column join avoids
        string serialization overhead that would be prohibitive at this scale.

    Args:
        k_parquet: Path to the K-level frequent itemsets parquet.
            Schema: itemset (list[i32]), support (f64).
        k_minus1_parquet: Path to the (K-1)-level frequent itemsets parquet.
            Schema: itemset (list[i32]), support (f64).
        k1_parquet: Optional path to K=1 parquet for single-item support (lift).
            Schema: itemset (list[i32]), support (f64).
        min_confidence: Minimum confidence threshold for filtering rules.
        chunk_size: Number of K-itemset rows per processing chunk.
            Controls peak memory. Default 1M rows.

    Returns:
        pl.DataFrame with columns:
            - itemset: original K-itemset (list[i32])
            - dropped_item: the consequent item (i32)
            - antecedent: the (K-1)-subset (list[i32])
            - support: K-itemset support (f64)
            - antecedent_support: (K-1)-subset support (f64)
            - confidence: support / antecedent_support (f64)
            - lift: confidence / item_support, or null if k1_parquet not provided (f64)
    """
    # ── Detect K from the parquet ──
    k = _detect_k(k_parquet)
    km1 = k - 1
    logger.info(f"[rules] Detected K={k}, K-1={km1}")

    # ── Load K-1 ONCE, unpack list into scalar columns for fast joins ──
    # K-1 can be large (K=7: 3.5B, 16.6 GB) but we need it in memory for
    # hash joins. Scalar columns use native Polars join (no string keys).
    logger.info(f"[rules] Loading K-1 parquet: {k_minus1_parquet}")
    join_cols = [f"i{j}" for j in range(km1)]

    km1_df: pl.DataFrame = (
        pl.scan_parquet(k_minus1_parquet)
        .with_columns(_list_to_scalar_cols("itemset", km1))
        .select([*join_cols, pl.col("support").alias("antecedent_support")])
        .collect(engine="streaming")
    )
    logger.info(f"[rules] K-1 loaded: {len(km1_df):,} itemsets")

    # ── Load K=1 support lookup (small: ~35K rows) ──
    item_support_map: dict[int, float] | None = None
    if k1_parquet is not None:
        logger.info(f"[rules] Loading K=1 parquet: {k1_parquet}")
        k1_df = pl.scan_parquet(k1_parquet).collect(engine="streaming")
        item_support_map = {row["itemset"][0]: row["support"] for row in k1_df.iter_rows(named=True)}
        logger.info(f"[rules] K=1 lookup: {len(item_support_map):,} items")
        del k1_df

    # ── Process K parquet in chunks ──
    result_chunks: list[pl.DataFrame] = []
    total_rules = 0
    chunk_idx = 0

    for chunk in _iter_row_groups(k_parquet, chunk_size):
        chunk_idx += 1
        n_rows = len(chunk)
        logger.debug(f"[rules] Chunk {chunk_idx}: {n_rows:,} K-itemsets")

        # Step 1: Explode into K rows per itemset (drop-1)
        exploded = _explode_drop1(chunk, k)
        del chunk

        # Step 2: Unpack antecedent list into scalar join columns
        exploded = exploded.with_columns(_list_to_scalar_cols("antecedent", km1))

        # Step 3: Hash join with K-1 on scalar columns to get antecedent support
        joined = exploded.join(km1_df, on=join_cols, how="inner")

        # Drop the scalar join columns — no longer needed
        joined = joined.drop(join_cols)
        del exploded

        # Step 4: Compute confidence
        joined = joined.with_columns(
            (pl.col("support") / pl.col("antecedent_support")).alias("confidence"),
        )

        # Step 5: Filter by min_confidence
        joined = joined.filter(pl.col("confidence") >= min_confidence)

        # Step 6: Compute lift if K=1 support available
        if item_support_map is not None:
            # Map dropped_item to its K=1 support via a small lookup DataFrame
            items_in_chunk = joined["dropped_item"].unique().to_list()
            if items_in_chunk:
                lift_lookup = pl.DataFrame(
                    {
                        "dropped_item": items_in_chunk,
                        "item_support": [item_support_map.get(int(i), 0.0) for i in items_in_chunk],
                    }
                ).cast({"dropped_item": joined["dropped_item"].dtype})

                joined = joined.join(lift_lookup, on="dropped_item", how="left")
                joined = joined.with_columns(
                    pl.when(pl.col("item_support") > 0.0)
                    .then(pl.col("confidence") / pl.col("item_support"))
                    .otherwise(pl.lit(None, dtype=pl.Float64))
                    .alias("lift"),
                )
                joined = joined.drop("item_support")
                del lift_lookup
            else:
                joined = joined.with_columns(
                    pl.lit(None, dtype=pl.Float64).alias("lift"),
                )
        else:
            joined = joined.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("lift"),
            )

        # Select final columns, drop internal columns (drop_idx etc.)
        result = joined.select(
            "itemset",
            "dropped_item",
            "antecedent",
            "support",
            "antecedent_support",
            "confidence",
            "lift",
        )

        n_rules = len(result)
        total_rules += n_rules
        logger.debug(f"[rules] Chunk {chunk_idx}: {n_rules:,} rules (confidence >= {min_confidence})")

        if n_rules > 0:
            result_chunks.append(result)

        del joined, result

    del km1_df

    # ── Concatenate all chunk results ──
    if not result_chunks:
        logger.warning("[rules] No rules found above confidence threshold.")
        return pl.DataFrame(
            schema={
                "itemset": pl.List(pl.Int32),
                "dropped_item": pl.Int32,
                "antecedent": pl.List(pl.Int32),
                "support": pl.Float64,
                "antecedent_support": pl.Float64,
                "confidence": pl.Float64,
                "lift": pl.Float64,
            }
        )

    logger.info(f"[rules] Total: {total_rules:,} rules from {chunk_idx} chunks")
    return pl.concat(result_chunks)


def generate_rules_top_n(
    frequent_itemsets: pl.DataFrame,
    n: int = 10_000,
    min_confidence: float = 0.5,
) -> list[Rule]:
    """Generate rules from the top-N frequent itemsets by support.

    Simple convenience wrapper for exploratory analysis: takes the N highest-
    support itemsets and feeds them to the existing generate_rules() function.
    Useful for quick inspection before running the full drop-1 pipeline.

    Args:
        frequent_itemsets: DataFrame with columns "itemset" (List[Int64])
            and "support" (Float64).
        n: Number of top itemsets to consider (sorted by support descending).
        min_confidence: Minimum confidence threshold (0.0-1.0).

    Returns:
        List of Rule objects meeting the confidence threshold, derived from
        the top-N itemsets.
    """
    top_n = frequent_itemsets.sort("support", descending=True).head(n)
    logger.info(f"[rules] Top-{n} itemsets: support range [{top_n['support'].min():.6f}, {top_n['support'].max():.6f}]")
    return generate_rules(top_n, min_confidence=min_confidence)


def compute_self_sufficiency(
    k_parquet: Path,
    k_minus1_parquet: Path,
    chunk_size: int = 1_000_000,
) -> pl.DataFrame:
    """Compute self-sufficiency ratio for K-itemsets vs their (K-1)-subsets.

    For each K-itemset, computes:
        max_k_minus1_support = max(support of all K-1 subsets)
        self_sufficiency_ratio = support_K / max_k_minus1_support

    A ratio close to 1.0 means the K-th item adds almost no information
    beyond what the (K-1)-subset already captures -- the itemset is
    "near-closed" and can be filtered out to reduce redundancy.

    Ratios well below 1.0 indicate genuine combinatorial signal: the
    K-itemset's co-occurrence is notably less frequent than any of its
    subsets, meaning the combination is informative.

    Processes K parquet in chunks via PyArrow row-group iteration to
    handle billion-row files without exceeding memory or u32 limits.

    Args:
        k_parquet: Path to the K-level frequent itemsets parquet.
            Schema: itemset (list[i32]), support (f64).
        k_minus1_parquet: Path to the (K-1)-level frequent itemsets parquet.
            Schema: itemset (list[i32]), support (f64).
        chunk_size: Number of K-itemset rows per processing chunk.

    Returns:
        pl.DataFrame with columns:
            - itemset: the K-itemset (list[i32])
            - support: K-itemset support (f64)
            - max_k_minus1_support: max support across all (K-1)-subsets (f64)
            - self_sufficiency_ratio: support / max_k_minus1_support (f64)
    """
    # ── Detect K ──
    k = _detect_k(k_parquet)
    km1 = k - 1
    logger.info(f"[self-sufficiency] Detected K={k}, K-1={km1}")

    # ── Load K-1 ONCE with scalar join columns ──
    logger.info(f"[self-sufficiency] Loading K-1 parquet: {k_minus1_parquet}")
    join_cols = [f"i{j}" for j in range(km1)]

    km1_df: pl.DataFrame = (
        pl.scan_parquet(k_minus1_parquet)
        .with_columns(_list_to_scalar_cols("itemset", km1))
        .select([*join_cols, pl.col("support").alias("km1_support")])
        .collect(engine="streaming")
    )
    logger.info(f"[self-sufficiency] K-1 loaded: {len(km1_df):,} itemsets")

    # ── Process K in chunks ──
    result_chunks: list[pl.DataFrame] = []
    chunk_idx = 0

    for chunk in _iter_row_groups(k_parquet, chunk_size):
        chunk_idx += 1
        n_rows = len(chunk)
        logger.debug(f"[self-sufficiency] Chunk {chunk_idx}: {n_rows:,} K-itemsets")

        # Add a row index to group back after explode + join
        chunk = chunk.with_row_index("_row_idx")

        # Explode to drop-1 rows (K subsets per itemset)
        exploded = _explode_drop1(chunk.drop("_row_idx"), k)

        # Re-attach row index: exploded has K rows per original row,
        # in order (cross join preserves order). Compute from drop_idx count.
        # Actually, _explode_drop1 doesn't carry _row_idx. We need to add it.
        # Since cross join produces rows in order [row0*K copies, row1*K copies, ...],
        # we can reconstruct it:
        n_exploded = len(exploded)
        row_indices = pl.Series(
            "_row_idx",
            [i // k for i in range(n_exploded)],
            dtype=pl.UInt32,
        )
        exploded = exploded.with_columns(row_indices)

        # Unpack antecedent into scalar columns for join
        exploded = exploded.with_columns(_list_to_scalar_cols("antecedent", km1))

        # Join with K-1 to get subset support
        joined = exploded.join(km1_df, on=join_cols, how="inner")
        joined = joined.drop(join_cols)
        del exploded

        # Group by _row_idx (= original itemset) and take max K-1 support
        grouped = joined.group_by("_row_idx").agg(pl.col("km1_support").max().alias("max_k_minus1_support"))
        del joined

        # Re-attach original itemset and support from the chunk
        result = chunk.join(grouped, on="_row_idx", how="inner").select("itemset", "support", "max_k_minus1_support")
        del grouped, chunk

        result_chunks.append(result)

    del km1_df

    if not result_chunks:
        logger.warning("[self-sufficiency] No results.")
        return pl.DataFrame(
            schema={
                "itemset": pl.List(pl.Int32),
                "support": pl.Float64,
                "max_k_minus1_support": pl.Float64,
                "self_sufficiency_ratio": pl.Float64,
            }
        )

    combined = pl.concat(result_chunks)

    # Compute ratio
    combined = combined.with_columns(
        (pl.col("support") / pl.col("max_k_minus1_support")).alias("self_sufficiency_ratio"),
    )

    logger.info(
        f"[self-sufficiency] Done: {len(combined):,} itemsets, "
        f"ratio range [{combined['self_sufficiency_ratio'].min():.6f}, "
        f"{combined['self_sufficiency_ratio'].max():.6f}]"
    )

    return combined
