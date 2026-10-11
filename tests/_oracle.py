"""efficient-apriori under the oracle convention of CLAUDE.md, for the test modules that compare against it.

tests/test_tier_equivalence.py keeps its own call: it is the gate, and its
oracle is a session fixture over the smoke preset.
"""

from __future__ import annotations

import polars as pl
import pytest

from et_miner.core.result import _min_count


def ea_counts(transactions, min_support: float, max_length: int | None = None) -> dict[tuple, int]:
    """efficient-apriori's {sorted itemset: count}.

    ``(min_count - 0.5) / N`` makes its float threshold keep exactly the counts
    the miner keeps (``count >= min_count``). ``max_length`` defaults to the
    longest transaction, since efficient-apriori otherwise stops at 8. Null
    items and null baskets are dropped, as the miner drops them.
    """
    ea = pytest.importorskip("efficient_apriori")
    tx = [tuple(x for x in (t or []) if x is not None) for t in transactions]
    n = len(tx)
    max_length = max_length or max((len(set(t)) for t in tx), default=1) or 1
    itemsets, _ = ea.itemsets_from_transactions(tx, (_min_count(min_support, n) - 0.5) / n, max_length=max_length)
    return {tuple(sorted(k)): v for level in itemsets.values() for k, v in level.items()}


def mined_counts(res: pl.DataFrame, n_rows: int) -> dict[tuple, int]:
    """A miner result's {sorted itemset: count}."""
    return {tuple(sorted(s)): round(p * n_rows) for s, p in zip(res["itemset"].to_list(), res["support"].to_list())}
