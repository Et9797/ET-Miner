"""The GPU route's input layer: the device layer for the integer ids it serves, every other column coded to
integer ids and mined there too, the host CSR only where neither can (or when pinned)."""

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


@pytest.fixture
def builders(monkeypatch):
    """The input builders that ran, in order."""
    from et_miner.core import matrix
    from et_miner.gpu import input_layer

    ran = []
    for mod, name in ((input_layer, "build_device_input"), (matrix, "_build_csr_from_transactions")):
        def spy(*a, _real=getattr(mod, name), _name=name, **k):
            ran.append(_name)
            return _real(*a, **k)

        monkeypatch.setattr(mod, name, spy)
    return ran


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_every_column_mines_on_the_device_layer_by_default(smoke, builders, variant):
    df = VARIANTS[variant](smoke)
    cpu = apriori(df, min_support=SPEC.min_support)
    builders.clear()
    gpu = apriori(df, min_support=SPEC.min_support, use_gpu=True)
    assert gpu.schema == cpu.schema
    assert _ordered(gpu) == _ordered(cpu)
    assert builders[-1] == "build_device_input" and "_build_csr_from_transactions" not in builders, builders


def test_more_distinct_items_than_the_coded_span_take_the_host_csr(smoke, builders, monkeypatch):
    from et_miner.core import cpu_miner

    df = VARIANTS["str"](smoke)
    monkeypatch.setattr(cpu_miner, "INT_SPAN_LIMIT", 3)
    cpu = apriori(df, min_support=SPEC.min_support)
    builders.clear()
    assert _ordered(apriori(df, min_support=SPEC.min_support, use_gpu=True)) == _ordered(cpu)
    assert builders == ["_build_csr_from_transactions"], builders


def test_stored_ids_on_an_unserved_column_take_the_host_csr(smoke, builders, tmp_path):
    """output_dir stores item ids, which the coded layer does not keep: wide ids go through the host CSR."""
    df = VARIANTS["wide"](smoke)
    builders.clear()
    apriori(df, min_support=SPEC.min_support, use_gpu=True, output_dir=str(tmp_path))
    assert builders[-1] == "_build_csr_from_transactions", builders


def test_the_host_pin_takes_the_host_csr(smoke, builders, monkeypatch):
    monkeypatch.setenv("ET_MINER_INPUT_LAYER", "host")
    builders.clear()
    apriori(VARIANTS["int"](smoke), min_support=SPEC.min_support, use_gpu=True)
    assert builders == ["_build_csr_from_transactions"], builders


@pytest.mark.parametrize("value", ["rebuilt", "coded"])
def test_the_removed_bench_pins_raise(monkeypatch, value):
    from et_miner import _env

    monkeypatch.setenv("ET_MINER_INPUT_LAYER", value)
    with pytest.raises(ValueError, match="ET_MINER_INPUT_LAYER"):
        _env.input_layer()
