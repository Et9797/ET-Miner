"""Integer item ids keep one dtype rule on every route.

Int8..Int64 and UInt8..UInt32 come back as Int64; UInt64, Int128 and UInt128
come back as themselves, ids beyond the Int64 range included. Before, the CPU
route crashed on 128-bit ids outside Int64, the GPU route on any id outside it
(and returned UInt64 ids as Int64), and SON on the GPU inferred the dtype from
the values.
"""

from __future__ import annotations

import polars as pl
import pytest

from et_miner import apriori
from et_miner.core.result import _build_result_df
from et_miner.streaming.son import apriori_streaming

BIG = 2**70

CASES = [
    (pl.Int32, pl.Int64, (5, 7, 9)),
    (pl.UInt32, pl.Int64, (5, 7, 9)),
    (pl.Int64, pl.Int64, (-(2**63), 7, 2**63 - 1)),
    (pl.UInt64, pl.UInt64, (5, 7, 9)),
    (pl.UInt64, pl.UInt64, (2**63 + 5, 2**63 + 7, 2**64 - 1)),
    (pl.Int128, pl.Int128, (5, 7, 9)),
    (pl.Int128, pl.Int128, (-BIG, BIG, BIG + 1)),
    (pl.UInt128, pl.UInt128, (5, 7, 9)),
    (pl.UInt128, pl.UInt128, (BIG, 2**127, 2**128 - 1)),
]
IDS = [f"{dt}-{'beyond' if max(abs(v) for v in ids) >= 2**63 else 'within'}" for dt, _, ids in CASES]


def _frame(dtype, ids) -> pl.DataFrame:
    a, b, c = ids
    rows = [[a, b]] * 4 + [[b, a]] * 2 + [[a]] * 2 + [[c]] * 2
    return pl.DataFrame({"items": rows}, schema={"items": pl.List(dtype)})


def _expected(ids) -> dict[tuple, int]:
    a, b, _ = ids
    return {(a,): 8, (b,): 6, tuple(sorted((a, b))): 6}


def _counts(res: pl.DataFrame, n: int) -> dict[tuple, int]:
    return {tuple(sorted(s)): round(p * n) for s, p in zip(res["itemset"].to_list(), res["support"].to_list())}


def _check(res: pl.DataFrame, want, ids) -> None:
    assert res.schema["itemset"] == pl.List(want)
    assert _counts(res, 10) == _expected(ids)


@pytest.mark.parametrize("dtype,want,ids", CASES, ids=IDS)
def test_cpu_route(dtype, want, ids):
    _check(apriori(_frame(dtype, ids), min_support=0.5), want, ids)


@pytest.mark.parametrize("dtype,want,ids", CASES, ids=IDS)
def test_son_on_the_cpu(dtype, want, ids):
    _check(apriori_streaming(_frame(dtype, ids), min_support=0.5, chunk_size=4, show_progress=False), want, ids)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype,want,ids", CASES, ids=IDS)
def test_gpu_route(dtype, want, ids):
    _check(apriori(_frame(dtype, ids), min_support=0.5, use_gpu=True), want, ids)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype,want,ids", CASES, ids=IDS)
def test_son_on_the_gpu(dtype, want, ids):
    res = apriori_streaming(_frame(dtype, ids), min_support=0.5, chunk_size=4, use_gpu=True, show_progress=False)
    _check(res, want, ids)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype,want,ids", CASES, ids=IDS)
def test_multi_gpu_son(dtype, want, ids):
    from et_miner.streaming.multi_gpu import apriori_streaming_multi_gpu

    res = apriori_streaming_multi_gpu(_frame(dtype, ids), min_support=0.5, chunk_size=4, n_gpus=1, show_progress=False)
    _check(res, want, ids)


@pytest.mark.gpu
def test_gpu_route_profile_keeps_the_dtype():
    ids = (2**63 + 5, 2**63 + 7, 2**64 - 1)
    res, session = apriori(_frame(pl.UInt64, ids), min_support=0.5, use_gpu=True, profile=True)
    _check(res, pl.UInt64, ids)
    assert session is not None


@pytest.mark.gpu
@pytest.mark.parametrize("kwargs", [{"output_dir": "OUT"}, {"anchor_items": {BIG}}])
def test_gpu_route_rejects_ids_beyond_int64_where_it_stores_them(kwargs, tmp_path):
    """Per-K parquet and anchors hold item ids in 64 bits; ids beyond Int64 raise there instead of overflowing."""
    if "output_dir" in kwargs:
        kwargs = {"output_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="outside the Int64 range"):
        apriori(_frame(pl.Int128, (-BIG, BIG, BIG + 1)), min_support=0.5, use_gpu=True, **kwargs)


@pytest.mark.gpu
def test_gpu_route_keeps_output_dir_for_wide_ids_within_int64(tmp_path):
    ids = (5, 7, 9)
    res = apriori(_frame(pl.UInt64, ids), min_support=0.5, use_gpu=True, output_dir=str(tmp_path))
    assert res.height == 0
    assert res.schema["itemset"] == pl.List(pl.UInt64)
    flushed = pl.concat([pl.read_parquet(p) for p in sorted(tmp_path.glob("frequent_k*.parquet"))])
    assert _counts(flushed, 10) == _expected(ids)


def test_build_result_df_takes_the_input_dtype():
    assert _build_result_df([([5], 0.5)], pl.UInt64).schema["itemset"] == pl.List(pl.UInt64)
    assert _build_result_df([([5], 0.5)], pl.Int16).schema["itemset"] == pl.List(pl.Int64)
    assert _build_result_df([([2**127], 0.5)], pl.UInt128)["itemset"].to_list() == [[2**127]]
    assert _build_result_df([(["a"], 0.5)], pl.String).schema["itemset"] == pl.List(pl.String)
