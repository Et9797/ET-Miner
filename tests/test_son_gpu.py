"""SON's GPU passes (the CPU passes' array skeleton on the row-split miner) mine what the CPU route mines."""

from __future__ import annotations

import polars as pl
import pytest

from et_miner.core.apriori import apriori
from et_miner.synthetic import PRESETS, generate_transactions

pytestmark = pytest.mark.gpu

SPEC = PRESETS["smoke"]


def _gpu_count() -> int:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


@pytest.fixture(scope="module")
def smoke():
    df, _ = generate_transactions(SPEC)
    return df


def _ordered(res: pl.DataFrame) -> list:
    return sorted((tuple(s), round(p * SPEC.n_rows)) for s, p in zip(res["itemset"].to_list(), res["support"].to_list()))


@pytest.mark.parametrize("n_gpus", [1, 2])
@pytest.mark.parametrize("cast", [None, pl.String])
def test_gpu_son_matches_the_cpu_route(smoke, monkeypatch, n_gpus, cast):
    if n_gpus > _gpu_count():
        pytest.skip(f"needs {n_gpus} CUDA devices")
    from et_miner.streaming import son

    df = smoke if cast is None else smoke.select(pl.col("items").list.eval(pl.element().cast(cast)))
    calls = {"mine": [], "count": []}
    for name, key in (("_local_levels_gpu", "mine"), ("_count_chunk_gpu", "count")):
        def spy(*a, _real=getattr(son, name), _key=key, **k):
            calls[_key].append(a[-1])
            return _real(*a, **k)

        monkeypatch.setattr(son, name, spy)
    events = []
    got = apriori(df, min_support=SPEC.min_support, streaming=True, chunk_size=SPEC.n_rows // 4, use_gpu=True,
                  n_gpus=n_gpus, progress_callback=lambda phase, i, n, m: events.append((phase, n)))
    cpu = apriori(df, min_support=SPEC.min_support)
    assert got.schema == cpu.schema
    assert _ordered(got) == _ordered(cpu)
    assert calls["mine"] == [n_gpus] * 4, calls
    assert set(calls["count"]) <= {n_gpus}
    assert {n for _, n in events} == {4}


def test_gpu_son_profiles_both_passes(smoke):
    got, session = apriori(smoke, min_support=SPEC.min_support, streaming=True, chunk_size=SPEC.n_rows // 4,
                           use_gpu=True, profile=True)
    assert _ordered(got) == _ordered(apriori(smoke, min_support=SPEC.min_support))
    names = [p.name for p in session.phases]
    assert "pass1_local_mining" in names and "pass2_global_counting" in names
