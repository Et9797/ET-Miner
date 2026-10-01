"""The benchmark must forward ESCO settings and compare the same counted lattice."""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from consolidation_matrix import build_esco_matrix
from consolidation_run import _mine
from child_run import result_signatures


@pytest.mark.parametrize("sparse_from_k", [3, "auto"])
def test_benchmark_forwards_esco_setting(monkeypatch, sparse_from_k):
    import importlib

    api = importlib.import_module("et_miner.core.apriori")
    seen = {}

    def spy(*a, **kw):
        seen.update(kw)
        return None

    monkeypatch.setattr(api, "apriori", spy)
    _mine({"route": "C", "sparse_from_k": sparse_from_k}, None, 0.0001, 3, None, None, {})
    assert seen["sparse_from_k"] == sparse_from_k
    assert seen["use_gpu"] is True
    assert seen["max_length"] == 3


def test_low_support_matrix_has_controls_and_equal_workloads():
    matrix = build_esco_matrix(2, retail_low=True)
    assert len({c["id"] for c in matrix}) == len(matrix)
    for s in (0.0001, 0.00005):
        for max_length in (2, 3, 4):
            for n_gpus in (1, 2):
                configs = [c for c in matrix if c["min_support"] == s and c["max_length"] == max_length
                           and c["n_gpus"] == n_gpus and c["route"] == "C" and c["rep"] == 0]
                assert len(configs) == 5
                assert {c.get("sparse_from_k") for c in configs} == {None, 3, "auto"}
                assert all(c["dataset"] == "online_retail" for c in configs)
                assert all(c["expect_transition"] == (max_length >= 3) for c in configs if c.get("sparse_from_k") == 3)
    assert {c["route"] for c in matrix} == {"C", "F", "EA"}


def test_retail_oracle_route_keeps_the_integer_boundary():
    from et_miner import apriori

    # Same row count and threshold counts as Online Retail, with a small
    # vocabulary so both supports can be checked exactly in CPU CI.
    df = pl.DataFrame({"items": [[0, 1, 2]] * 4 + [[3, 4, 5]] * 2 + [[99]] * (36_422 - 6)})
    for s in (0.0001, 0.00005):
        oracle = _mine({"route": "EA"}, df, s, 6, None, None, {})
        cpu = apriori(df, min_support=s, max_length=6)
        assert result_signatures(oracle, df.height) == result_signatures(cpu, df.height)


@pytest.mark.gpu
@pytest.mark.parametrize("min_support", [0.0001, 0.00005])
@pytest.mark.parametrize("sparse_from_k", [3, "auto"])
def test_esco_keeps_extreme_support_boundaries(monkeypatch, min_support, sparse_from_k):
    from et_miner import apriori
    from et_miner.gpu import row_split

    df = pl.DataFrame({"items": [[0, 1, 2, 3]] * 4 + [[4, 5, 6, 7]] * 2 + [[99]] * (36_422 - 6)})
    transitioned = []
    convert = row_split.convert_shards_to_csr

    def spy(*a, **kw):
        transitioned.append(1)
        return convert(*a, **kw)

    monkeypatch.setattr(row_split, "convert_shards_to_csr", spy)
    oracle = _mine({"route": "EA"}, df, min_support, 4, None, None, {})
    dense = apriori(df, min_support=min_support, max_length=4, use_gpu=True)
    sparse = apriori(df, min_support=min_support, max_length=4, use_gpu=True, sparse_from_k=sparse_from_k)
    assert result_signatures(dense, df.height) == result_signatures(oracle, df.height)
    assert result_signatures(sparse, df.height) == result_signatures(oracle, df.height)
    assert transitioned, "low-support boundary test must reach ESCO"
