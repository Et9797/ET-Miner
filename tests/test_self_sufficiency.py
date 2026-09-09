"""The self-sufficiency aggregate, pinned.

`compute_self_sufficiency` changed from `max` to `min` over the (K-1)-subset
supports, which alters **every row** of a public output. Nothing pinned it: a
council reviewer mutated `.min()` back to `.max()` and the full suite stayed
green at 584 passed, byte-identical. The only test touching the function
asserted route-insensitivity (`a == b`), which is invariant under the aggregate.

That is the same failure mode the change itself was filed against — #12 existed
because three sites carried one expression and agreed with each other while all
three were wrong. An untested aggregate is that with N=1.
"""

from __future__ import annotations

import polars as pl
import pytest

from et_miner.core.rules import compute_self_sufficiency


def _write(tmp_path, k2_rows, k3_rows):
    k2, k3 = tmp_path / "k2.parquet", tmp_path / "k3.parquet"
    pl.DataFrame(k2_rows, schema={"itemset": pl.List(pl.Int32), "support": pl.Float64}).write_parquet(k2)
    pl.DataFrame(k3_rows, schema={"itemset": pl.List(pl.Int32), "support": pl.Float64}).write_parquet(k3)
    return k3, k2


def test_aggregates_with_min_not_max(tmp_path):
    """Two itemsets that are EQUALLY redundant under the engine's own predicate
    must score the same.

    The predicate is `support_K == min(subset supports)` -- the same test as
    `core/apriori.py::_prune_equal_support` and `groups.rs`. Both itemsets below
    satisfy it, so both are maximally redundant and both must score 1.0. Under
    `max` they score 0.30/0.32 = 0.9375 and 0.30/0.95 = 0.3157894736842105:
    identical redundancy, opposite readings, opposite recommended actions. That
    is why no cutoff recalibrates it -- it is the wrong *kind* of aggregation.
    """
    k3, k2 = _write(
        tmp_path,
        {"itemset": [[1, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 6]],
         "support": [0.30, 0.90, 0.95, 0.30, 0.32, 0.32]},
        {"itemset": [[1, 2, 3], [4, 5, 6]], "support": [0.30, 0.30]},
    )
    out = compute_self_sufficiency(k3, k2).sort("itemset")
    assert out["min_k_minus1_support"].to_list() == [0.30, 0.30]
    assert out["self_sufficiency_ratio"].to_list() == [1.0, 1.0]


def test_a_redundant_itemset_scores_near_one(tmp_path):
    """The docstring's own semantics: a ratio close to 1.0 means the K-th item
    adds almost nothing. Item 3 is fully implied by {1,2}, so {1,2,3} is
    maximally redundant -- it scored 0.32 under max and read as "genuine
    combinatorial signal", exactly backwards."""
    k3, k2 = _write(
        tmp_path,
        {"itemset": [[1, 2], [1, 3], [2, 3]], "support": [0.30, 0.90, 0.95]},
        {"itemset": [[1, 2, 3]], "support": [0.30]},
    )
    out = compute_self_sufficiency(k3, k2)
    assert out["self_sufficiency_ratio"][0] == pytest.approx(1.0)


def test_a_genuinely_informative_itemset_scores_well_below_one(tmp_path):
    """The other direction, so the test cannot pass by always returning 1.0."""
    k3, k2 = _write(
        tmp_path,
        {"itemset": [[1, 2], [1, 3], [2, 3]], "support": [0.80, 0.85, 0.90]},
        {"itemset": [[1, 2, 3]], "support": [0.20]},
    )
    out = compute_self_sufficiency(k3, k2)
    assert out["self_sufficiency_ratio"][0] == pytest.approx(0.25)


def test_ratio_is_bounded_by_one(tmp_path):
    """Under min the ratio is bounded in (0, 1] by construction, since
    support_K <= support(W) for every subset W. Under max it was not."""
    k3, k2 = _write(
        tmp_path,
        {"itemset": [[1, 2], [1, 3], [2, 3]], "support": [0.31, 0.60, 0.99]},
        {"itemset": [[1, 2, 3]], "support": [0.30]},
    )
    assert compute_self_sufficiency(k3, k2)["self_sufficiency_ratio"][0] <= 1.0
