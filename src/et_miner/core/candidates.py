"""CPU candidate generation for the Apriori join step.

k=2 uses vectorized Polars cross-join generation (streaming above
_K2_STREAMING_THRESHOLD items); k>2 uses prefix-group generation with
automatic strategy selection. _generate_candidates is the entry point.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import polars as pl


def _generate_candidates(
    prev_frequent: list[tuple[str, ...]],
    k: int,
) -> list[tuple[str, ...]]:
    """Generate k-candidates from (k-1)-frequent using Apriori join.

    For k=2: vectorized pair generation via Polars cross-join.
    For k>2: prefix-based generation with automatic strategy selection.

    Args:
        prev_frequent: List of frequent (k-1)-itemsets as tuples of column names.
        k: Target itemset size.

    Returns:
        List of candidate k-itemsets.
    """
    if k == 2:
        return _generate_candidates_k2(prev_frequent)

    # k > 2: Use vectorized approach for larger sets
    n = len(prev_frequent)
    if n > 1000:
        return _generate_candidates_vectorized(prev_frequent, k)

    # Small sets: simple Python loop is faster (no DataFrame overhead)
    return _generate_candidates_simple(prev_frequent, k)


def _generate_candidates_k2_streaming(items: list[str]) -> Iterator[tuple[str, str]]:
    """Yield (a, b) pairs with a<b, avoiding the Polars cross-join's intermediates.

    The generator itself is O(1), but **the caller drains it into a list**
    (`_generate_candidates_k2` below), so the pair list is fully materialised
    either way: measured 1,242 MB peak RSS at 6,000 items (17,997,000 pairs),
    and the stress_k2 preset's 35,000 items would be ~600M pairs. The saving
    over the cross-join branch is real but partial — it avoids the intermediate
    DataFrame, not the list.

    The laziness cannot be used without reworking the caller:
    `_generate_candidates` is typed `-> list[tuple[str, ...]]`, and
    `core/apriori.py`'s level loop takes `len()` of the result and then iterates
    it twice. Making the O(1) claim true means folding that into a single pass.
    """
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            yield (items[i], items[j])


# Threshold for switching to streaming K=2 generation
# Cross-join with 5000 items creates 12.5M rows - streaming is more memory-efficient
_K2_STREAMING_THRESHOLD = 5000


def _generate_candidates_k2(
    prev_frequent: list[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    """Pair generation for k=2.

    Uses streaming generator for large item counts (>5000) to avoid
    materializing n*(n-1)/2 rows in memory from cross-join.

    Args:
        prev_frequent: List of frequent 1-itemsets.

    Returns:
        List of candidate 2-itemsets.
    """
    items = sorted(p[0] for p in prev_frequent)
    n_items = len(items)

    # For large item counts, use streaming generator to avoid memory issues
    # 5000 items = 12.5M pairs, 10000 items = 50M pairs in cross-join
    if n_items > _K2_STREAMING_THRESHOLD:
        return list(_generate_candidates_k2_streaming(items))

    # Warn if approaching 2^32 limit (sqrt(4.2B) ≈ 65k)
    if n_items > 60_000:
        warnings.warn(
            f"Large item count ({n_items:,}) may hit Polars 2^32 row limit. "
            f"Consider `pip install polars[rt64]` for datasets >65k items.",
            UserWarning,
            stacklevel=3,
        )

    items_df = pl.DataFrame({"item": items})

    pairs = (
        items_df.lazy()
        .select(pl.col("item").alias("a"))
        .join(items_df.lazy().select(pl.col("item").alias("b")), how="cross")
        .filter(pl.col("a") < pl.col("b"))
        .collect()
    )

    return [(row["a"], row["b"]) for row in pairs.iter_rows(named=True)]


def _generate_candidates_simple(
    prev_frequent: list[tuple[str, ...]],
    k: int,
) -> list[tuple[str, ...]]:
    """Simple Python loop for small candidate sets.

    More efficient than DataFrame overhead for small n.

    Args:
        prev_frequent: List of frequent (k-1)-itemsets.
        k: Target itemset size.

    Returns:
        List of candidate k-itemsets.
    """
    prev_set = set(prev_frequent)
    sorted_prev = sorted(prev_frequent)
    candidates = []

    for i, itemset1 in enumerate(sorted_prev):
        for itemset2 in sorted_prev[i + 1 :]:
            # Check if they share k-2 prefix
            if itemset1[:-1] == itemset2[:-1]:
                candidate = itemset1 + (itemset2[-1],)

                # Apriori pruning: all (k-1)-subsets must be frequent
                is_valid = True
                for j in range(k):
                    subset = candidate[:j] + candidate[j + 1 :]
                    if subset not in prev_set:
                        is_valid = False
                        break

                if is_valid:
                    candidates.append(candidate)

    return candidates


def _generate_candidates_vectorized(
    prev_frequent: list[tuple[str, ...]],
    k: int,
) -> list[tuple[str, ...]]:
    """Vectorized candidate generation via Polars prefix-based grouping.

    Groups itemsets by their (k-2) prefix and generates candidates within
    each group. This avoids O(n²) comparisons across all itemsets.

    Args:
        prev_frequent: List of frequent (k-1)-itemsets.
        k: Target itemset size.

    Returns:
        List of candidate k-itemsets.
    """
    # Group by prefix for efficient candidate generation
    sorted_prev = sorted(prev_frequent)

    # Build prefix groups: {prefix: [itemsets with that prefix]}
    prefix_groups: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
    for itemset in sorted_prev:
        prefix = itemset[:-1]
        if prefix not in prefix_groups:
            prefix_groups[prefix] = []
        prefix_groups[prefix].append(itemset)

    # Check if any group is large enough to benefit from vectorization
    max_group_size = max(len(g) for g in prefix_groups.values()) if prefix_groups else 0

    if max_group_size < 100:
        # All groups small, use simple loop (less overhead)
        return _generate_candidates_simple(prev_frequent, k)

    # Prepare for Apriori pruning
    prev_set = set(prev_frequent)
    all_candidates: list[tuple[str, ...]] = []

    for prefix, group in prefix_groups.items():
        group_size = len(group)

        if group_size < 100:
            # Small group: simple Python loop
            for i, item1 in enumerate(group):
                for item2 in group[i + 1 :]:
                    candidate = item1 + (item2[-1],)
                    if _is_valid_candidate(candidate, k, prev_set):
                        all_candidates.append(candidate)
        else:
            # Large group: vectorized via Polars
            candidates = _generate_from_group_vectorized(group, prefix, k, prev_set)
            all_candidates.extend(candidates)

    return all_candidates


def _generate_from_group_vectorized(
    group: list[tuple[str, ...]],
    prefix: tuple[str, ...],
    k: int,
    prev_set: set[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    """Generate candidates from a single prefix group using Polars.

    Args:
        group: List of itemsets sharing the same prefix.
        prefix: The shared (k-2) prefix.
        k: Target itemset size.
        prev_set: Set of all frequent (k-1)-itemsets for pruning.

    Returns:
        List of valid candidate k-itemsets.
    """
    # Extract last items from each itemset in the group
    last_items = [itemset[-1] for itemset in group]

    # Create DataFrame with last items
    df = pl.DataFrame({"last": last_items})

    # Self-join to get all pairs where a < b
    pairs_df = (
        df.lazy()
        .select(pl.col("last").alias("a"))
        .join(df.lazy().select(pl.col("last").alias("b")), how="cross")
        .filter(pl.col("a") < pl.col("b"))
        .collect()
    )

    # Convert to candidates and apply Apriori pruning
    candidates: list[tuple[str, ...]] = []
    for row in pairs_df.iter_rows(named=True):
        candidate = prefix + (row["a"], row["b"])
        if _is_valid_candidate(candidate, k, prev_set):
            candidates.append(candidate)

    return candidates


def _is_valid_candidate(
    candidate: tuple[str, ...],
    k: int,
    prev_set: set[tuple[str, ...]],
) -> bool:
    """Apriori pruning: all (k-1)-subsets of candidate must be in prev_set."""
    for j in range(k):
        subset = candidate[:j] + candidate[j + 1 :]
        if subset not in prev_set:
            return False
    return True
