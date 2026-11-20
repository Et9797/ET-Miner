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


@pytest.mark.parametrize("kwargs", [{"output_dir": "OUT"}, {"anchor_items": {BIG}}, {"resume_from_k": 2}])
def test_gpu_route_rejects_ids_beyond_int64_where_it_stores_them(kwargs, tmp_path, monkeypatch):
    """Per-K parquet, resume and anchors hold item ids in 64 bits: ids beyond Int64 raise, before any CSR is built."""
    from et_miner.core import matrix

    def no_csr(*args, **kwargs):
        raise AssertionError("the CSR was built before the ids were checked")

    monkeypatch.setattr(matrix, "coo_matrix", no_csr)
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


ROUTES = {
    "cpu": lambda df, s: apriori(df, min_support=s),
    "son-cpu": lambda df, s: apriori_streaming(df, min_support=s, chunk_size=4, show_progress=False),
}
GPU_ROUTES = {
    "gpu": lambda df, s: apriori(df, min_support=s, use_gpu=True),
    "son-gpu": lambda df, s: apriori_streaming(df, min_support=s, chunk_size=4, use_gpu=True, show_progress=False),
}
EMPTY_CASES = [
    (pl.Int32, pl.Int64, (5, 7, 9)),
    (pl.UInt64, pl.UInt64, (5, 7, 9)),
    (pl.UInt128, pl.UInt128, (BIG, 2**127, 2**128 - 1)),
]


def _empty_and_full(route, dtype, want, ids) -> None:
    """Nothing frequent, an empty frame, and a result: all three carry the same itemset dtype and concatenate."""
    df = _frame(dtype, ids)
    none_frequent = route(df, 0.95)
    no_rows = route(df.clear(), 0.5)
    full = route(df, 0.5)
    for res in (none_frequent, no_rows, full):
        assert res.schema["itemset"] == pl.List(want)
    assert pl.concat([full, none_frequent, no_rows]).height == full.height


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("dtype,want,ids", EMPTY_CASES, ids=[str(c[0]) for c in EMPTY_CASES])
def test_an_empty_result_keeps_the_dtype(route, dtype, want, ids):
    _empty_and_full(ROUTES[route], dtype, want, ids)


@pytest.mark.gpu
@pytest.mark.parametrize("route", GPU_ROUTES)
@pytest.mark.parametrize("dtype,want,ids", EMPTY_CASES, ids=[str(c[0]) for c in EMPTY_CASES])
def test_an_empty_gpu_result_keeps_the_dtype(route, dtype, want, ids):
    _empty_and_full(GPU_ROUTES[route], dtype, want, ids)


def test_mine_two_phase_removes_its_temp_dir_when_phase_1_raises(tmp_path, monkeypatch):
    """Phase 1 stores ids in 64 bits, so ids beyond Int64 raise there; the temp dir it made goes with it."""
    import tempfile

    from et_miner.gpu.row_split import mine_two_phase

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with pytest.raises(ValueError, match="outside the Int64 range"):
        mine_two_phase(_frame(pl.Int128, (-BIG, BIG, BIG + 1)), phase1_support=0.5, phase2_support=0.4)
    assert list(tmp_path.iterdir()) == []


NULL_ROWS = [[1, None, 2]] * 4 + [[1, None]] * 2 + [[]] * 3 + [None] * 2


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [pl.Int64, pl.UInt64])
def test_null_items_are_no_items_on_the_gpu(dtype):
    """The CPU route drops null items and empty baskets' nulls; the GPU paths raised on them."""
    from et_miner.streaming.multi_gpu import apriori_streaming_multi_gpu

    df = pl.DataFrame({"items": NULL_ROWS}, schema={"items": pl.List(dtype)})
    want = _counts(apriori(df, min_support=0.3), df.height)
    assert want == {(1,): 6, (2,): 4, (1, 2): 4}
    runs = {
        "gpu": apriori(df, min_support=0.3, use_gpu=True),
        "son-gpu": apriori_streaming(df, min_support=0.3, chunk_size=4, use_gpu=True, show_progress=False),
        "multi-gpu": apriori_streaming_multi_gpu(df, min_support=0.3, chunk_size=4, n_gpus=1, show_progress=False),
    }
    for name, res in runs.items():
        assert _counts(res, df.height) == want, name


#: Items that are not integer ids: (input dtype, the dtype every route emits, values, values on the CPU route).
NON_INT_CASES = [
    (pl.String, pl.String, ("10", "9", "b")),
    (pl.Float64, pl.Float64, (1.2, 1.7, -0.5)),
    (pl.Float32, pl.Float64, (1.25, 1.75, -0.5)),
    (pl.Categorical, pl.String, ("10", "9", "b")),
]


def _ordered(res: pl.DataFrame, n: int) -> dict[tuple, int]:
    """Itemsets in the order the route emits their items, with integer counts."""
    return {tuple(s): round(p * n) for s, p in zip(res["itemset"].to_list(), res["support"].to_list())}


@pytest.mark.gpu
@pytest.mark.parametrize("dtype,want,ids", NON_INT_CASES, ids=[str(c[0]) for c in NON_INT_CASES])
def test_gpu_route_mines_non_integer_items_as_the_cpu_route_does(dtype, want, ids):
    """Strings, floats and categoricals mine as column ids and map back: same items, item order, dtype and counts
    as the CPU route. They used to be copied into an int64 array: strings raised or came back as Int64, floats
    were truncated (1.2 and 1.7 both became 1)."""
    df = _frame(dtype, ids)
    cpu = apriori(df, min_support=0.5)
    for layer in ("host", None):
        with pytest.MonkeyPatch.context() as mp:
            if layer:
                mp.setenv("ET_MINER_INPUT_LAYER", layer)
            gpu = apriori(df, min_support=0.5, use_gpu=True)
        assert gpu.schema["itemset"] == cpu.schema["itemset"] == pl.List(want)
        assert _ordered(gpu, 10) == _ordered(cpu, 10)
        assert _counts(gpu, 10) == _expected(ids)


@pytest.mark.parametrize("kwargs", [{"output_dir": "OUT"}, {"anchor_items": {"10"}}, {"resume_from_k": 2}])
def test_gpu_route_rejects_non_integer_items_where_it_stores_ids(kwargs, tmp_path):
    """Per-K parquet, resume and anchors hold integer item ids: other items raise before anything is built."""
    if "output_dir" in kwargs:
        kwargs = {"output_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="not integer ids"):
        apriori(_frame(pl.String, ("10", "9", "b")), min_support=0.5, use_gpu=True, **kwargs)
