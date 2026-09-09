"""Shared result-frame helpers.

Semi-stable internal API: keep the signatures put even as the mining modules
evolve.

**Not every path routes its output through _build_result_df, despite what an
earlier version of this docstring said.** The direct CPU path
(core/apriori.py), gpu/mining.py and both streaming paths do. The multi-GPU
row-split miner does not: gpu/row_split.py imports _build_result_df but calls
it only for the empty-result early return, and builds its two real return
frames directly from numpy via Arrow.

That matters because the claim was two-thirds true -- row_split *does* use
_min_count and _empty_result -- so spot-checking "does row_split use the result
helpers?" answers yes and the false part is the narrower, better-hidden one.
Two reviewers acted on it and proposed enforcing the emitted-itemset order here;
that would have missed the one route which already gets the order right.
"""

from __future__ import annotations

import math

import polars as pl


def _min_count(min_support: float, n_transactions: int) -> int:
    """Calculate minimum count threshold from support."""
    return math.ceil(min_support * n_transactions)


def _empty_result() -> pl.DataFrame:
    """Return empty result DataFrame with correct schema."""
    return pl.DataFrame(schema={"itemset": pl.List(pl.Int64), "support": pl.Float64})


def _build_result_df(results: list[tuple[list, float]]) -> pl.DataFrame:
    """Build result DataFrame from collected itemsets.

    Preserves the original item types (int, str, etc.) from the input data.
    """
    if not results:
        return _empty_result()
    # Don't force cast to Int64 - let Polars infer the type from the actual items
    return pl.DataFrame(
        {
            "itemset": [r[0] for r in results],
            "support": [float(r[1]) for r in results],
        }
    )
