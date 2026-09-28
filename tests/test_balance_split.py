"""Tests for the multi-GPU row-split cut computation.

The cut math is pure numpy (no marker). The nnz-balanced cut was removed
after the 2-GPU measurements (it tied the rows split on `skewed_rows` and
`deep_sparse_large`); asking for it, by parameter or by
`ET_MINER_ROW_BALANCE`, must raise a ValueError naming the rows split.
"""

import numpy as np
import polars as pl
import pytest

from et_miner import _env
from et_miner.gpu.csr_bitvec import _row_split_cuts, build_bitvecs_row_split_from_arrays


def _indptr(row_nnz):
    return np.concatenate([[0], np.cumsum(row_nnz)]).astype(np.int64)


class TestRowSplitCuts:
    def test_rows_mode_equal_counts(self):
        cuts = _row_split_cuts(10, 2, balance="rows")
        assert cuts == [(0, 5), (5, 10)]

    def test_rows_mode_remainder(self):
        cuts = _row_split_cuts(10, 3, balance="rows")
        assert cuts == [(0, 4), (4, 8), (8, 10)]

    def test_default_is_rows(self):
        assert _row_split_cuts(10, 2) == [(0, 5), (5, 10)]

    def test_single_gpu_is_whole_range(self):
        assert _row_split_cuts(7, 1, balance="rows") == [(0, 7)]

    def test_contiguous_and_covering(self):
        cuts = _row_split_cuts(1000, 4, balance="rows")
        assert cuts[0][0] == 0 and cuts[-1][1] == 1000
        assert all(a[1] == b[0] for a, b in zip(cuts, cuts[1:]))

    def test_no_empty_shards(self):
        cuts = _row_split_cuts(5, 4, balance="rows")
        assert all(e > s for s, e in cuts)
        assert cuts[0][0] == 0 and cuts[-1][1] == 5

    def test_more_gpus_than_rows(self):
        cuts = _row_split_cuts(2, 8, balance="rows")
        assert sum(e - s for s, e in cuts) == 2

    def test_removed_nnz_raises_naming_rows(self):
        with pytest.raises(ValueError, match="balance='nnz' was removed"):
            _row_split_cuts(4, 2, balance="nnz")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="must be 'rows'"):
            _row_split_cuts(4, 2, balance="roundrobin")

    def test_empty_input(self):
        assert _row_split_cuts(0, 2, balance="rows") == []


class TestRemovedNnzBalance:
    def test_build_from_arrays_rejects_nnz_before_touching_a_device(self):
        # The check runs before cupy is imported, so it holds on any box.
        with pytest.raises(ValueError, match="balance='nnz' was removed"):
            build_bitvecs_row_split_from_arrays(_indptr([1] * 4), np.zeros(4, np.int64), 4, 2, 2, balance="nnz")

    def test_env_unset_or_rows_passes(self, monkeypatch):
        monkeypatch.delenv("ET_MINER_ROW_BALANCE", raising=False)
        _env.reject_removed_knobs()
        monkeypatch.setenv("ET_MINER_ROW_BALANCE", "rows")
        _env.reject_removed_knobs()

    def test_env_empty_is_unset(self, monkeypatch):
        monkeypatch.setenv("ET_MINER_ROW_BALANCE", "")
        _env.reject_removed_knobs()

    @pytest.mark.parametrize("value", ["nnz", "NNZ", " nnz "])
    def test_env_nnz_raises_naming_rows(self, monkeypatch, value):
        monkeypatch.setenv("ET_MINER_ROW_BALANCE", value)
        with pytest.raises(ValueError, match="ET_MINER_ROW_BALANCE='nnz' was removed"):
            _env.reject_removed_knobs()

    def test_env_nnz_raises_at_apriori_entry(self, monkeypatch):
        from et_miner.core.apriori import apriori

        monkeypatch.setenv("ET_MINER_ROW_BALANCE", "nnz")
        df = pl.DataFrame({"items": [[1, 2], [1, 2, 3], [2, 3]]})
        with pytest.raises(ValueError, match="ET_MINER_ROW_BALANCE"):
            apriori(df, min_support=0.5, item_col="items")
