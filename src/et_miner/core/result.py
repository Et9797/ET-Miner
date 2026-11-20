"""Shared result-frame helpers.

Semi-stable internal API: keep the signatures put even as the mining modules
evolve.

**Not every path routes its output through _build_result_df, despite what an
earlier version of this docstring said.** Both streaming paths (SON and
multi-GPU) do. The CPU route does not: core/cpu_miner.py builds its frame in
_emit from arrays and uses _min_count, _empty_result and the item-dtype rule
(_INT64_ITEM_DTYPES, _result_item_dtype) from here, not _build_result_df. The
multi-GPU row-split miner does not either: gpu/row_split.py imports
_build_result_df but calls it only for the empty-result early return, and
builds its two real return frames directly from numpy via Arrow.

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

#: Integer item dtypes Int64 holds every value of: a result carries their ids as Int64.
_INT64_ITEM_DTYPES = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32)
#: Integer item dtypes with values beyond Int64: a result keeps them.
_WIDE_ITEM_DTYPES = (pl.UInt64, pl.Int128, pl.UInt128)
#: Every integer item dtype.
_INT_ITEM_DTYPES = _INT64_ITEM_DTYPES + _WIDE_ITEM_DTYPES


def _item_dtype(lf: pl.LazyFrame, item_col: str) -> pl.DataType | pl.datatypes.DataTypeClass | None:
    """The inner dtype of the list column ``item_col``; None when it is not a list."""
    dtype = lf.collect_schema()[item_col]
    return dtype.inner if isinstance(dtype, pl.List) else None


def _result_item_dtype(
    item_dtype: pl.DataType | pl.datatypes.DataTypeClass | None,
) -> pl.DataType | pl.datatypes.DataTypeClass | None:
    """A result's item dtype for integer items: Int64 where it holds every value, the dtype itself otherwise.

    None for any other dtype (the caller's own rule applies).
    """
    if item_dtype in _INT64_ITEM_DTYPES:
        return pl.Int64
    if item_dtype in _WIDE_ITEM_DTYPES:
        return item_dtype
    return None


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


def _emitted_items(items: pl.Series) -> pl.Series:
    """``items`` as a result emits them: integers as ``_result_item_dtype`` gives them, floats as Float64,
    categoricals and enums as String, any other kind as it is."""
    dtype = items.dtype
    if (out := _result_item_dtype(dtype)) is not None:
        return items.cast(out)
    if dtype.is_float():
        return items.cast(pl.Float64)
    if dtype.base_type() in (pl.Categorical, pl.Enum):
        return items.cast(pl.String)
    return items


def _empty_result(item_dtype: pl.DataType | pl.datatypes.DataTypeClass | None = None) -> pl.DataFrame:
    """An empty result frame: itemsets as ``_result_item_dtype(item_dtype)`` for integer items, else Int64."""
    return pl.DataFrame(schema={"itemset": pl.List(_result_item_dtype(item_dtype) or pl.Int64), "support": pl.Float64})


def _build_result_df(
    results: list[tuple[list, float]], item_dtype: pl.DataType | pl.datatypes.DataTypeClass | None = None
) -> pl.DataFrame:
    """Build result DataFrame from collected itemsets.

    Integer items take ``_result_item_dtype(item_dtype)``, the input's item
    dtype; other items keep the type Polars infers from the values (str etc.).
    """
    if not results:
        return _empty_result(item_dtype)
    out = _result_item_dtype(item_dtype)
    return pl.DataFrame(
        {
            "itemset": pl.Series([r[0] for r in results], dtype=pl.List(out) if out is not None else None),
            "support": [float(r[1]) for r in results],
        }
    )
