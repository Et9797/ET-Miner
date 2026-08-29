"""Shared result-frame helpers.

Semi-stable internal API: the direct, SON-streaming, and multi-GPU mining
paths (and their tests) all build their output through these three helpers,
so their signatures should stay put even as the mining modules evolve.
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
