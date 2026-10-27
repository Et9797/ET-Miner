"""The bench-only input layers ("rebuilt", "coded") mine what the CPU route mines, through the builder they name."""

from __future__ import annotations

import polars as pl
import pytest

from et_miner.core.apriori import apriori
from et_miner.synthetic import PRESETS, generate_transactions

pytestmark = pytest.mark.gpu

SPEC = PRESETS["smoke"]
VARIANTS = {
    "int": lambda df: df,
    "str": lambda df: df.select(pl.col("items").list.eval(pl.element().cast(pl.String))),
    "wide": lambda df: df.select(pl.col("items").list.eval(pl.element() * 10_000_019)),
    "u64": lambda df: df.select(pl.col("items").cast(pl.List(pl.UInt64))),
}


@pytest.fixture(scope="module")
def smoke():
    df, _ = generate_transactions(SPEC)
    return df


def _ordered(res: pl.DataFrame) -> list:
    return sorted((tuple(s), round(p * SPEC.n_rows)) for s, p in zip(res["itemset"].to_list(), res["support"].to_list()))


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("layer", ["rebuilt", "coded"])
def test_input_arm_matches_the_cpu_route(smoke, monkeypatch, layer, variant):
    from et_miner.core import cpu_miner, matrix
    from et_miner.gpu import input_layer

    df = VARIANTS[variant](smoke)
    ran = []
    for mod, name in ((input_layer, "build_device_input"), (matrix, "_build_csr_from_transactions"),
                      (cpu_miner, "build_transaction_csr")):
        def spy(*a, _real=getattr(mod, name), _name=name, **k):
            ran.append(_name)
            return _real(*a, **k)

        monkeypatch.setattr(mod, name, spy)
    cpu = apriori(df, min_support=SPEC.min_support)
    ran.clear()
    monkeypatch.setenv("ET_MINER_INPUT_LAYER", layer)
    gpu = apriori(df, min_support=SPEC.min_support, use_gpu=True)
    assert gpu.schema == cpu.schema
    assert _ordered(gpu) == _ordered(cpu)
    if layer == "rebuilt":
        assert ran == ["build_transaction_csr"], ran
    else:
        assert ran[-1] == "build_device_input" and "_build_csr_from_transactions" not in ran, ran
