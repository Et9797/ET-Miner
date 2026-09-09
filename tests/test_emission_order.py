"""The emitted-itemset order contract, and the two parquet consumers that join on it.

`apriori()` documents its result as a set of itemsets, and both parquet
consumers (`generate_rules_drop1`, `compute_self_sufficiency`) join K against
K-1 **positionally** -- `_list_to_scalar_cols` unpacks a list column into scalar
join keys by index. That only works if every producer emits the same canonical
element order. It was not stated anywhere and no test checked it, so the CPU
route emitted lexicographic column-NAME order ("i_10" < "i_2") while the GPU
route emitted ascending item ids, and a mixed-tier artifact silently lost rows
and corrupted numbers.

These tests are the contract, stated producer-side.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from et_miner import apriori
from et_miner.core.matrix import build_boolean_matrix
from et_miner.core.rules import compute_self_sufficiency, generate_rules, generate_rules_drop1


def _wide_frame(n_rows: int = 300, n_items: int = 15, seed: int = 0) -> pl.DataFrame:
    """Enough items that the unpadded name order diverges from item-id order."""
    rng = np.random.default_rng(seed)
    rows = [
        sorted(rng.choice(n_items, size=int(rng.integers(4, 9)), replace=False).tolist())
        for _ in range(n_rows)
    ]
    return pl.DataFrame({"items": rows})


class TestColumnNaming:
    def test_column_names_are_zero_padded_to_a_constant_width(self):
        """The control for the emission-order fix.

        Padding is what makes lexicographic name order equal item-id order. A
        bare green on the ordering tests is indistinguishable from a patch that
        never applied, so assert the mechanism as well as the outcome.
        """
        _, col_to_item, _ = build_boolean_matrix(_wide_frame().lazy(), min_support=0.05)
        names = list(col_to_item)
        assert len(names) == 15, f"fixture must have >9 frequent items, got {len(names)}"
        assert names[0] == "i_00" and names[-1] == "i_14", names[:3]
        assert len({len(n) for n in names}) == 1, "width must be constant within a run"

    def test_padded_name_order_equals_item_id_order(self):
        """The invariant the width exists to produce."""
        _, col_to_item, _ = build_boolean_matrix(_wide_frame().lazy(), min_support=0.05)
        by_name = [col_to_item[n] for n in sorted(col_to_item)]
        assert by_name == sorted(by_name)


class TestEmissionOrder:
    @pytest.mark.parametrize("kwargs", [{}, {"sparse": True}])
    def test_itemsets_are_emitted_in_ascending_item_id_order(self, kwargs):
        res = apriori(_wide_frame(), min_support=0.05, **kwargs)
        bad = [list(x) for x in res["itemset"].to_list() if list(x) != sorted(x)]
        assert not bad, f"{len(bad)} non-ascending itemsets, e.g. {bad[:5]}"

    def test_order_is_stable_across_thresholds(self):
        """A column index depends on how many OTHER items cleared the threshold,
        so an unpadded name order changes with min_support: the same itemset came
        back as [2, 10] at one threshold and [10, 2] at another, which silently
        breaks any stored artifact or frozen digest keyed on the emitted list."""
        rng = np.random.default_rng(1)
        rows = []
        for i in range(300):
            r = {2, 10}
            if i % 10 < 4:
                r |= set(rng.choice(15, size=5, replace=False).tolist())
            rows.append(sorted(r))
        df = pl.DataFrame({"items": rows})

        def emitted(ms: float) -> dict[tuple, list]:
            return {tuple(sorted(s)): list(s) for s in apriori(df, min_support=ms)["itemset"].to_list()}

        few, many = emitted(0.8), emitted(0.1)
        shared = set(few) & set(many)
        assert shared, "fixture must share itemsets across the two thresholds"
        differing = {k: (few[k], many[k]) for k in shared if few[k] != many[k]}
        assert not differing, f"emitted order moved with min_support: {differing}"


class TestSupportLookupIsOrderInsensitive:
    def test_generate_rules_matches_the_canonicalised_frame(self):
        """_build_support_lookup keys sorted and both callers query sorted, so a
        producer's tuple order cannot drop rules or zero out lift."""
        res = apriori(_wide_frame(), min_support=0.05, max_length=3)
        as_emitted = generate_rules(res, min_confidence=0.0)
        canonical = generate_rules(
            res.with_columns(pl.col("itemset").list.sort()), min_confidence=0.0
        )
        assert len(as_emitted) == len(canonical)
        assert sum(1 for r in as_emitted if r.lift == 0.0) == 0

    def test_lookup_survives_a_reversed_producer(self):
        """The helper must not depend on its producer's ordering -- the next
        producer need not know that rule."""
        res = apriori(_wide_frame(), min_support=0.05, max_length=3)
        reversed_frame = res.with_columns(pl.col("itemset").list.reverse())
        assert len(generate_rules(reversed_frame, min_confidence=0.0)) == len(
            generate_rules(res, min_confidence=0.0)
        )


class TestMixedTierParquetJoins:
    """#3 -- both parquet consumers join positionally, so a K level from one
    route and a K-1 level from another must still line up. Neither function had
    any test coverage at all before this."""

    @staticmethod
    def _write_levels(tmp_path, df, k_route: dict, km1_route: dict):
        paths = {}
        for k, route in ((3, k_route), (2, km1_route)):
            res = apriori(df, min_support=0.05, max_length=k, **route)
            level = res.filter(pl.col("itemset").list.len() == k)
            p = tmp_path / f"k{k}_{'gpu' if route.get('use_gpu') else 'cpu'}.parquet"
            level.write_parquet(p)
            paths[k] = p
        return paths

    def test_drop1_is_route_insensitive(self, tmp_path):
        df = _wide_frame()
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        clean = self._write_levels(tmp_path / "a", df, {}, {})
        mixed = self._write_levels(tmp_path / "b", df, {}, {"sparse": True})

        n_clean = len(generate_rules_drop1(clean[3], clean[2], min_confidence=0.0))
        n_mixed = len(generate_rules_drop1(mixed[3], mixed[2], min_confidence=0.0))
        assert n_clean > 0, "fixture produced no drop-1 rules"
        assert n_mixed == n_clean, f"mixing routes lost {n_clean - n_mixed} of {n_clean} rules"

    def test_self_sufficiency_is_route_insensitive(self, tmp_path):
        df = _wide_frame()
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        clean = self._write_levels(tmp_path / "a", df, {}, {})
        mixed = self._write_levels(tmp_path / "b", df, {}, {"sparse": True})

        a = compute_self_sufficiency(clean[3], clean[2]).sort("itemset")
        b = compute_self_sufficiency(mixed[3], mixed[2]).sort("itemset")
        assert a.height > 0, "fixture produced no self-sufficiency rows"
        assert a.height == b.height, f"mixing routes dropped {a.height - b.height} rows"
        assert a["max_k_minus1_support"].to_list() == b["max_k_minus1_support"].to_list()

    @pytest.mark.gpu
    def test_drop1_survives_a_cpu_k_level_against_a_gpu_k_minus_1(self, tmp_path):
        """The configuration #3 was measured on: the CPU route emitted
        lexicographic column-name order and the GPU route ascending item ids, so
        a (K-1)-subset extracted from a CPU K-itemset missed its GPU-stored twin
        and the inner join silently dropped 38% of the drop-1 rules. The routes
        agree on element order now, so mixing them must be lossless."""
        df = _wide_frame()
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        clean = self._write_levels(tmp_path / "a", df, {}, {})
        mixed = self._write_levels(tmp_path / "b", df, {}, {"use_gpu": True})

        n_clean = len(generate_rules_drop1(clean[3], clean[2], min_confidence=0.0))
        n_mixed = len(generate_rules_drop1(mixed[3], mixed[2], min_confidence=0.0))
        assert n_clean > 0
        assert n_mixed == n_clean, f"CPU K3 + GPU K2 lost {n_clean - n_mixed} of {n_clean} rules"
