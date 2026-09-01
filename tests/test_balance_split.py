"""Tests for the multi-GPU row-split cut computation (rows vs nnz balance).

The cut math is pure numpy (no marker); end-to-end shard equivalence on
real devices is covered by the gpu-marked campaign tests.
"""

import numpy as np
import pytest

from et_miner.gpu.csr_bitvec import _row_split_cuts


def _indptr(row_nnz):
    return np.concatenate([[0], np.cumsum(row_nnz)]).astype(np.int64)


class TestRowSplitCuts:
    def test_rows_mode_equal_counts(self):
        cuts = _row_split_cuts(_indptr([1] * 10), 10, 2, balance="rows")
        assert cuts == [(0, 5), (5, 10)]

    def test_rows_mode_remainder(self):
        cuts = _row_split_cuts(_indptr([1] * 10), 10, 3, balance="rows")
        assert cuts == [(0, 4), (4, 8), (8, 10)]

    def test_single_gpu_is_whole_range(self):
        for mode in ("rows", "nnz"):
            assert _row_split_cuts(_indptr([2] * 7), 7, 1, balance=mode) == [(0, 7)]

    def test_nnz_mode_balances_clustered_data(self):
        # First 2 rows carry almost all nnz; rows mode would give GPU 0
        # nearly everything, nnz mode pushes the cut to the front.
        row_nnz = [100, 100, 1, 1, 1, 1, 1, 1]
        indptr = _indptr(row_nnz)
        cuts = _row_split_cuts(indptr, 8, 2, balance="nnz")
        assert len(cuts) == 2
        nnz_per_shard = [int(indptr[e] - indptr[s]) for s, e in cuts]
        total = sum(row_nnz)
        # Each shard within a row's worth of the ideal half.
        assert all(abs(x - total / 2) <= max(row_nnz) for x in nnz_per_shard)
        # And distinctly better than the rows split, which is 203 vs 3.
        rows_cuts = _row_split_cuts(indptr, 8, 2, balance="rows")
        rows_nnz = [int(indptr[e] - indptr[s]) for s, e in rows_cuts]
        assert max(nnz_per_shard) < max(rows_nnz)

    def test_nnz_mode_contiguous_and_covering(self):
        rng = np.random.default_rng(3)
        row_nnz = rng.integers(0, 50, size=1000)
        indptr = _indptr(row_nnz)
        cuts = _row_split_cuts(indptr, 1000, 4, balance="nnz")
        assert cuts[0][0] == 0 and cuts[-1][1] == 1000
        assert all(a[1] == b[0] for a, b in zip(cuts, cuts[1:]))

    def test_nnz_mode_zero_nnz_falls_back(self):
        cuts = _row_split_cuts(_indptr([0] * 8), 8, 2, balance="nnz")
        assert cuts == [(0, 4), (4, 8)]

    def test_no_empty_shards(self):
        # One row hoards everything: nnz targets collapse onto the same cut;
        # empty ranges must be dropped, not emitted.
        cuts = _row_split_cuts(_indptr([1000, 1, 1, 1]), 4, 4, balance="nnz")
        assert all(e > s for s, e in cuts)
        assert cuts[0][0] == 0 and cuts[-1][1] == 4

    def test_more_gpus_than_rows(self):
        cuts = _row_split_cuts(_indptr([1, 1]), 2, 8, balance="rows")
        assert sum(e - s for s, e in cuts) == 2

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="balance"):
            _row_split_cuts(_indptr([1]), 1, 1, balance="roundrobin")

    def test_empty_input(self):
        assert _row_split_cuts(_indptr([]), 0, 2, balance="rows") == []


class TestRowBalanceEnv:
    def test_default_rows(self, monkeypatch):
        monkeypatch.delenv("ET_MINER_ROW_BALANCE", raising=False)
        from et_miner import _env

        assert _env.row_balance() == "rows"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ET_MINER_ROW_BALANCE", "NNZ")
        from et_miner import _env

        assert _env.row_balance() == "nnz"


@pytest.mark.gpu
class TestBalanceSplitGPU:
    """On clustered (skewed_rows-shaped) data, the nnz split must shard nnz
    near-evenly AND produce mining results identical to the rows split."""

    @pytest.fixture(scope="class")
    def skewed(self):
        pytest.importorskip("cupy")
        from et_miner.synthetic import SynthSpec, generate_transactions

        spec = SynthSpec(
            name="skew_mini",
            n_rows=80_000,
            vocab_size=1_200,
            zipf_a=1.1,
            row_len_mean=8,
            row_len_max=64,
            skew_frac=0.2,
            skew_mult=5.0,
            min_support=0.01,
            seed=31,
        )
        df, data = generate_transactions(spec)
        return spec, df, data

    def test_nnz_shards_balanced_and_correct(self, skewed):
        import cupy as cp

        from et_miner.gpu.csr_bitvec import build_bitvecs_row_split_from_arrays

        if cp.cuda.runtime.getDeviceCount() < 2:
            pytest.skip("needs 2 CUDA devices")
        spec, _, data = skewed
        shards = build_bitvecs_row_split_from_arrays(
            data.indptr, data.indices.astype(np.int64), data.n_rows, data.n_cols, n_gpus=2, balance="nnz"
        )
        assert sum(rows for _, _, rows in shards) == data.n_rows
        total_nnz = int(data.indptr[-1])
        # Reconstruct per-shard nnz from the contiguous row ranges.
        row0 = shards[0][2]
        nnz0 = int(data.indptr[row0])
        assert abs(nnz0 - total_nnz / 2) < 0.1 * total_nnz, "nnz split failed to balance clustered data"
        for bv, did, _ in shards:
            with cp.cuda.Device(did):
                del bv
                cp.get_default_memory_pool().free_all_blocks()

    def test_nnz_results_equal_rows_results(self, skewed, monkeypatch):
        from et_miner.core.apriori import apriori

        spec, df, _ = skewed

        def run():
            res = apriori(df, min_support=spec.min_support, item_col="items", use_gpu=True, n_gpus=2)
            return {
                (tuple(sorted(int(i) for i in s)), round(sup * spec.n_rows))
                for s, sup in zip(res["itemset"].to_list(), res["support"].to_list())
            }

        monkeypatch.setenv("ET_MINER_ROW_BALANCE", "rows")
        rows_result = run()
        monkeypatch.setenv("ET_MINER_ROW_BALANCE", "nnz")
        assert run() == rows_result
        assert len(rows_result) > 0
