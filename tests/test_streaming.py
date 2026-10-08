"""Tests for streaming Apriori implementation (SON algorithm).

These tests verify that:
1. streaming=True produces identical results to streaming=False
2. Memory stays within expected bounds
3. Edge cases are handled correctly
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import polars as pl
import pytest

from et_miner import apriori, apriori_streaming


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def small_transactions():
    """Small test dataset for basic functionality."""
    return pl.DataFrame(
        {
            "items": [
                [1, 2, 3],
                [2, 3, 4],
                [1, 2, 3, 4],
                [1, 3, 5],
                [2, 3, 5],
                [1, 2, 3, 5],
            ]
        }
    )


@pytest.fixture
def medium_transactions():
    """Medium dataset with ~1000 transactions for testing chunking."""
    import random

    random.seed(42)
    items = list(range(1, 51))  # 50 unique items

    transactions = []
    for _ in range(1000):
        n_items = random.randint(3, 10)
        tx = sorted(random.sample(items, n_items))
        transactions.append(tx)

    return pl.DataFrame({"items": transactions})


@pytest.fixture
def large_transactions():
    """Larger dataset with ~10000 transactions."""
    import random

    random.seed(42)
    items = list(range(1, 101))  # 100 unique items

    transactions = []
    for _ in range(10000):
        n_items = random.randint(3, 15)
        tx = sorted(random.sample(items, n_items))
        transactions.append(tx)

    return pl.DataFrame({"items": transactions})


# =============================================================================
# Correctness Tests
# =============================================================================


class TestStreamingCorrectness:
    """Verify streaming produces identical results to standard apriori."""

    def test_identical_results_small_dataset(self, small_transactions):
        """Streaming should produce same itemsets as standard on small data."""
        min_support = 0.3

        # Standard apriori
        standard_result = apriori(small_transactions, min_support=min_support)

        # Streaming with small chunks to force multiple chunks
        streaming_result = apriori(
            small_transactions,
            min_support=min_support,
            streaming=True,
            chunk_size=3,  # Force 2 chunks
        )

        # Sort both for comparison
        standard_sorted = standard_result.sort("itemset")
        streaming_sorted = streaming_result.sort("itemset")

        # Compare itemsets
        assert standard_sorted.height == streaming_sorted.height, (
            f"Different number of itemsets: standard={standard_sorted.height}, "
            f"streaming={streaming_sorted.height}"
        )

        # Compare each itemset
        for i in range(standard_sorted.height):
            std_itemset = standard_sorted["itemset"][i].to_list()
            str_itemset = streaming_sorted["itemset"][i].to_list()
            assert std_itemset == str_itemset, (
                f"Itemset mismatch at row {i}: standard={std_itemset}, streaming={str_itemset}"
            )

    def test_identical_results_medium_dataset(self, medium_transactions):
        """Streaming should produce same itemsets on medium data."""
        min_support = 0.05

        standard_result = apriori(medium_transactions, min_support=min_support)
        streaming_result = apriori(
            medium_transactions,
            min_support=min_support,
            streaming=True,
            chunk_size=200,  # ~5 chunks
        )

        standard_sorted = standard_result.sort("itemset")
        streaming_sorted = streaming_result.sort("itemset")

        assert standard_sorted.height == streaming_sorted.height

    def test_single_chunk_optimization(self, small_transactions):
        """When data fits in one chunk, should use standard apriori path."""
        result = apriori(
            small_transactions,
            min_support=0.3,
            streaming=True,
            chunk_size=1000,  # Larger than dataset
        )

        # Should still produce correct results
        assert result.height > 0

    def test_support_values_match(self, small_transactions):
        """Support values should be identical between streaming and standard."""
        min_support = 0.3

        standard = apriori(small_transactions, min_support=min_support)
        streaming = apriori(
            small_transactions,
            min_support=min_support,
            streaming=True,
            chunk_size=3,
        )

        # Create lookup dict
        standard_supports = {
            tuple(row["itemset"]): row["support"]
            for row in standard.iter_rows(named=True)
        }
        streaming_supports = {
            tuple(row["itemset"]): row["support"]
            for row in streaming.iter_rows(named=True)
        }

        for itemset, support in standard_supports.items():
            assert itemset in streaming_supports, f"Missing itemset {itemset}"
            assert abs(support - streaming_supports[itemset]) < 1e-9, (
                f"Support mismatch for {itemset}: "
                f"standard={support}, streaming={streaming_supports[itemset]}"
            )


# =============================================================================
# Edge Cases
# =============================================================================


class TestStreamingEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_transactions(self):
        """Handle empty input gracefully."""
        empty_df = pl.DataFrame({"items": []}, schema={"items": pl.List(pl.Int64)})

        result = apriori(empty_df, min_support=0.5, streaming=True, chunk_size=100)
        assert result.height == 0

    def test_no_frequent_itemsets(self):
        """Handle case where no itemsets meet threshold."""
        df = pl.DataFrame(
            {
                "items": [
                    [1],
                    [2],
                    [3],
                    [4],
                    [5],
                ]
            }
        )

        result = apriori(df, min_support=0.5, streaming=True, chunk_size=2)
        assert result.height == 0

    def test_all_same_transaction(self):
        """All transactions identical should find all subsets."""
        df = pl.DataFrame(
            {
                "items": [
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                ]
            }
        )

        result = apriori(df, min_support=0.5, streaming=True, chunk_size=2)
        # Should find: {1}, {2}, {3}, {1,2}, {1,3}, {2,3}, {1,2,3}
        assert result.height == 7

    def test_max_length_respected(self, medium_transactions):
        """max_length should limit itemset size in streaming mode."""
        result = apriori(
            medium_transactions,
            min_support=0.05,
            max_length=2,
            streaming=True,
            chunk_size=200,
        )

        max_itemset_len = result.select(
            pl.col("itemset").list.len().max()
        ).item()
        assert max_itemset_len <= 2


# =============================================================================
# Memory Tests
# =============================================================================


class TestStreamingMemory:
    """Verify memory bounds are respected."""

    @pytest.mark.slow
    def test_memory_stays_bounded(self, large_transactions):
        """Memory should not grow linearly with dataset size."""
        tracemalloc.start()

        # Run streaming with small chunks
        _ = apriori(
            large_transactions,
            min_support=0.01,
            streaming=True,
            chunk_size=500,
        )

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Peak memory should be reasonable (< 100MB for this dataset)
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 100, f"Peak memory {peak_mb:.1f}MB exceeds 100MB limit"

    def test_memory_budget_parameter(self, medium_transactions):
        """memory_budget_gb should automatically calculate chunk size."""
        # Very small budget should still work
        result = apriori(
            medium_transactions,
            min_support=0.1,
            streaming=True,
            memory_budget_gb=0.001,  # 1MB budget
        )

        assert result.height > 0


# =============================================================================
# API Tests
# =============================================================================


class TestStreamingAPI:
    """Test API surface and parameter handling."""

    def test_lazyframe_input(self, small_transactions):
        """Should work with LazyFrame input."""
        lf = small_transactions.lazy()

        result = apriori(lf, min_support=0.3, streaming=True, chunk_size=3)
        assert result.height > 0

    def test_profiling_returns_session(self, small_transactions):
        """profile=True should return ProfilingSession."""
        result, session = apriori(
            small_transactions,
            min_support=0.3,
            streaming=True,
            chunk_size=3,
            profile=True,
        )

        assert result.height > 0
        assert session is not None
        # Session should have streaming-specific phases
        assert "count_transactions" in str(session)

    def test_apriori_streaming_direct_call(self, small_transactions):
        """apriori_streaming can be called directly."""
        result = apriori_streaming(
            small_transactions,
            min_support=0.3,
            chunk_size=3,
        )

        assert result.height > 0

    @pytest.mark.parametrize("sparse", [True, False])
    def test_sparse_is_deprecated_and_changes_nothing(self, medium_transactions, sparse):
        """SON's CPU passes select no counter by sparse=: a value warns and mines the same lattice."""
        plain = apriori(medium_transactions, min_support=0.05, streaming=True, chunk_size=200)
        with pytest.warns(DeprecationWarning, match="sparse= no longer selects"):
            result = apriori(medium_transactions, min_support=0.05, streaming=True, chunk_size=200, sparse=sparse)
        assert result.height > 0
        assert result.sort("itemset").equals(plain.sort("itemset"))

    def test_sparse_warns_once_on_a_direct_call(self, medium_transactions):
        with pytest.warns(DeprecationWarning, match="sparse= no longer selects") as record:
            apriori_streaming(medium_transactions, min_support=0.05, chunk_size=200, sparse=True, show_progress=False)
        assert sum(issubclass(w.category, DeprecationWarning) for w in record) == 1

    def test_n_jobs_parameter_forwarded(self, medium_transactions):
        """n_jobs parameter should work in streaming mode."""
        result = apriori(
            medium_transactions,
            min_support=0.05,
            streaming=True,
            chunk_size=200,
            n_jobs=2,
        )

        assert result.height > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestStreamingIntegration:
    """Integration tests combining multiple features."""

    def test_streaming_with_all_parameters(self, medium_transactions):
        """All parameters should work together."""
        result = apriori(
            medium_transactions,
            min_support=0.05,
            max_length=3,
            streaming=True,
            chunk_size=200,
            n_jobs=2,
            show_progress=False,
        )

        assert result.height > 0
        max_len = result.select(pl.col("itemset").list.len().max()).item()
        assert max_len <= 3

    @pytest.mark.slow
    def test_streaming_deterministic(self, large_transactions):
        """Multiple runs should produce identical results."""
        results = []
        for _ in range(3):
            result = apriori(
                large_transactions,
                min_support=0.01,
                streaming=True,
                chunk_size=500,
            )
            results.append(result.sort("itemset"))

        # All runs should be identical
        for i in range(1, len(results)):
            assert results[0].height == results[i].height
            assert results[0]["itemset"].to_list() == results[i]["itemset"].to_list()


# =============================================================================
# SON's CPU passes on the array miner
# =============================================================================


def _messy_rows(seed: int, n_rows: int = 600, n_items: int = 20) -> list:
    """Item ids with unsorted rows, repeats, empty and null lists and null items."""
    import random

    rng = random.Random(seed)
    rows: list = []
    for _ in range(n_rows):
        row = rng.sample(range(n_items), rng.randint(0, 7))
        r = rng.random()
        if r < 0.04:
            row = None
        elif r < 0.08 and row:
            row = row + [row[0]]
        elif r < 0.11:
            row = row + [None]
        rows.append(row)
    return rows


def _counted(df: pl.DataFrame, n_rows: int) -> dict[tuple, int]:
    """{itemset: count}, asserting every itemset is emitted in ascending item order."""
    out = {}
    for s, p in zip(df["itemset"].to_list(), df["support"].to_list()):
        assert list(s) == sorted(s), f"itemset {s} not ascending"
        out[tuple(s)] = round(p * n_rows)
    return out


def _same(son: pl.DataFrame, core: pl.DataFrame, n_rows: int) -> None:
    assert len(son) == len(core), "an itemset emitted twice, or one missing"
    assert _counted(son, n_rows) == _counted(core, n_rows)


class TestArrayPasses:
    """SON's CPU passes give exactly the in-core CPU route's itemsets and counts."""

    @pytest.mark.parametrize("chunk_size", [7, 150, 599])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_messy_rows_match_the_cpu_route(self, seed, chunk_size):
        df = pl.DataFrame({"items": _messy_rows(seed)}, schema={"items": pl.List(pl.Int64)})
        son = apriori(df, min_support=0.03, streaming=True, chunk_size=chunk_size, show_progress=False)
        _same(son, apriori(df, min_support=0.03), df.height)

    @pytest.mark.parametrize(
        "rows",
        [
            pytest.param([["b", "a"], ["a", "b", "c"], ["c", "b"], ["d"]] * 30, id="strings"),
            pytest.param([[2**40, 1], [1, 2**40, 3], [2**40, 3]] * 30, id="ids-beyond-int32"),
            pytest.param([[1, 2], [], [1, 2, 3], None] * 30, id="empty-and-null-baskets"),
        ],
    )
    def test_item_kinds_match_the_cpu_route(self, rows):
        df = pl.DataFrame({"items": rows})
        son = apriori(df, min_support=0.1, streaming=True, chunk_size=25, show_progress=False)
        core = apriori(df, min_support=0.1)
        assert len(son) > 0
        assert son.schema == core.schema
        assert sorted(map(tuple, son["itemset"].to_list())) == sorted(map(tuple, core["itemset"].to_list()))
        assert sorted(son["support"].to_list()) == sorted(core["support"].to_list())

    def test_items_first_seen_out_of_order_are_renumbered(self):
        """The first chunk holds only the large ids, the second only the small ones."""
        rows = [[9, 7], [9, 7, 8]] * 10 + [[1, 3], [1, 2, 3]] * 10 + [[1, 9], [3, 7]] * 5
        df = pl.DataFrame({"items": rows})
        son = apriori(df, min_support=0.1, streaming=True, chunk_size=20, show_progress=False)
        _same(son, apriori(df, min_support=0.1), df.height)

    @pytest.mark.parametrize("max_length", [1, 2, 3])
    def test_max_length_matches_the_cpu_route(self, medium_transactions, max_length):
        son = apriori(
            medium_transactions, min_support=0.005, max_length=max_length, streaming=True, chunk_size=300,
            show_progress=False,
        )
        assert son["itemset"].list.len().max() == max_length
        _same(son, apriori(medium_transactions, min_support=0.005, max_length=max_length), 1000)

    def test_n_jobs_does_not_change_the_result(self, medium_transactions):
        runs = [
            apriori(medium_transactions, min_support=0.02, streaming=True, chunk_size=300, show_progress=False, n_jobs=n)
            for n in (3, 1)
        ]
        _same(runs[0], runs[1], 1000)

    def test_profile_and_progress_report_the_deduplicated_union(self, medium_transactions):
        seen = []
        res, session = apriori_streaming(
            medium_transactions, min_support=0.05, chunk_size=300, show_progress=False, profile=True,
            progress_callback=lambda phase, i, n, m: seen.append((phase, i, n, m)),
        )
        pass1 = {p.name: p.extra for p in session.phases}["pass1_local_mining"]
        assert [s[:3] for s in seen] == [("pass1", i, 4) for i in range(4)] + [("pass2", i, 4) for i in range(4)]
        assert seen[3][3]["candidates"] >= pass1["n_candidates"] >= len(res)
        assert seen[3][3]["items"] == pass1["n_items"]

    @pytest.mark.parametrize("chunk_size", [7, 150])
    def test_merging_the_union_during_pass_1_changes_nothing(self, monkeypatch, chunk_size):
        """With no pending budget, each chunk is merged into the union as soon as it outgrows it."""
        from et_miner.streaming import son

        df = pl.DataFrame({"items": _messy_rows(3)}, schema={"items": pl.List(pl.Int64)})
        kw = {"min_support": 0.03, "chunk_size": chunk_size, "show_progress": False}
        merged_once = apriori_streaming(df, **kw)
        monkeypatch.setattr(son, "UNION_PENDING_BYTES", 0)
        merges = []
        real = son.sum_rows
        monkeypatch.setattr(son, "sum_rows", lambda *a: merges.append(1) or real(*a))
        _same(apriori_streaming(df, **kw), merged_once, df.height)
        _same(merged_once, apriori(df, min_support=0.03), df.height)
        assert len(merges) > len(merged_once["itemset"].list.len().unique())

    def test_the_bound_drops_candidates_and_keeps_exact_counts(self):
        """Pass 1's counts drop union candidates and spare the exact ones from pass 2; the result stays the CPU route's."""
        df = pl.DataFrame({"items": _messy_rows(4)}, schema={"items": pl.List(pl.Int64)})
        son, session = apriori_streaming(df, min_support=0.03, chunk_size=150, show_progress=False, profile=True)
        extra = {p.name: p.extra for p in session.phases}
        pass1, pass2 = extra["pass1_local_mining"], extra["pass2_global_counting"]
        assert pass1["n_bounded"] < pass1["n_candidates"]
        assert pass1["n_exact"] > 0
        assert pass2["n_counted"] == pass1["n_bounded"] - pass1["n_exact"] > 0
        _same(son, apriori(df, min_support=0.03), df.height)

    def test_pass_2_reads_no_chunk_when_every_count_is_exact(self, monkeypatch):
        """A local min_count of 1 leaves no slack: pass 1's counts are the global ones."""
        from et_miner.streaming import son as son_mod

        def fail(*a, **k):
            raise AssertionError("pass 2 read a chunk")

        df = pl.DataFrame({"items": _messy_rows(5)}, schema={"items": pl.List(pl.Int64)})
        seen = []
        monkeypatch.setattr(son_mod, "_count_chunk", fail)
        son = apriori_streaming(
            df, min_support=0.03, chunk_size=25, show_progress=False,
            progress_callback=lambda phase, i, n, m: seen.append(phase),
        )
        assert "pass2" not in seen
        _same(son, apriori(df, min_support=0.03), df.height)

    @pytest.mark.parametrize("factor", [0.5, 1.0])
    def test_the_local_support_factor_does_not_change_the_result(self, factor):
        df = pl.DataFrame({"items": _messy_rows(6)}, schema={"items": pl.List(pl.Int64)})
        son = apriori_streaming(df, min_support=0.03, chunk_size=130, show_progress=False, local_support_factor=factor)
        _same(son, apriori(df, min_support=0.03), df.height)


def test_bound_per_candidate():
    """Two chunks of slack 2 each, min_count 5."""
    from et_miner.streaming.son import _bound

    rows = np.arange(8, dtype=np.int32).reshape(4, 2)
    known = np.array([6, 3, 2, 4], dtype=np.int32)
    slack = np.array([4, 2, 2, 4], dtype=np.int32)
    kept, counts, need = _bound({2: (rows, known, slack)}, total_slack=4, min_count=5)[2]
    assert kept.tolist() == [[0, 1], [2, 3]]
    assert counts.tolist() == [6, 0]
    assert need.tolist() == [False, True]


def test_union_sums_in_int64_from_2_31_transactions():
    from et_miner.streaming.son import _CandidateUnion

    union = _CandidateUnion(2**31)
    big = np.array([2**30, 2**30], dtype=np.int64)
    for _ in range(2):
        union.add(pl.Series([5, 9]), [(np.array([[0], [1]], dtype=np.int32), big)], 2**30)
    items, merged = union.finish()
    rows, known, slack = merged[1]
    assert items.to_list() == [5, 9]
    assert known.dtype == slack.dtype == np.int64
    assert known.tolist() == [2**31, 2**31]
    assert slack.tolist() == [2**31, 2**31]
