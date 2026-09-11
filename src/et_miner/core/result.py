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
from fractions import Fraction

import polars as pl


def _min_count(min_support: float, n_transactions: int) -> int:
    """Minimum absolute count for an itemset to be frequent: ceil(s * N), exactly.

    ``min_support`` denotes the **shortest decimal that round-trips to the given
    float**, not the exact binary value of that float. ``Fraction(str(s))`` is
    that decimal, exactly: ``str()`` on a float is the shortest round-tripping
    representation, so ``Fraction(str(0.07)) == 7/100``.

    That interpretation is a contract, not a law, and it is shared by
    ``rust_ext::core::apriori::exact_min_count`` and by
    ``synthetic.SynthSpec.min_count``. **All three must move together** --
    tests/fixtures/min_count_cases.json is the single table they are checked
    against, because two hand-maintained tables drift.

    The naive ``math.ceil(min_support * n_transactions)`` is wrong on the
    boundary: 0.07 has no exact binary64 form, so ``0.07 * 10000`` evaluates to
    700.0000000000001 and ceils to 701 where the exact ceiling is 700. The
    excess exceeds half an ulp there, so it cannot round back. An itemset whose
    count is exactly 700 was then dropped -- and with it the whole cone of
    itemsets above it -- while efficient-apriori, the oracle CLAUDE.md mandates,
    keeps it (``700/10000 >= 0.07``).
    """
    return math.ceil(Fraction(str(min_support)) * n_transactions)


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
