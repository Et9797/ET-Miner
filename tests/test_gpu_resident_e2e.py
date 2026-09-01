"""GPU-resident single-GPU path (``apriori(gpu_resident=True)``) vs the CPU
tier on the smoke preset.

Covers count_k3plus_gpu_resident and decode_candidates_gpu — the kernels
that decode candidates from explicit group sizes through the shared decode
prelude (_decode_common.cu) — which no other test exercised.
"""

import polars as pl
import pytest

cp = pytest.importorskip("cupy", reason="cupy not installed")

pytestmark = pytest.mark.gpu

from et_miner.core.apriori import apriori
from et_miner.synthetic import PRESETS, generate_transactions

SPEC = PRESETS["smoke"]


def _counted(df: pl.DataFrame, n_rows: int) -> set:
    return {
        (tuple(sorted(int(i) for i in itemset)), round(sup * n_rows))
        for itemset, sup in zip(df["itemset"].to_list(), df["support"].to_list())
    }


def test_gpu_resident_matches_cpu_reference():
    df, _ = generate_transactions(SPEC)
    cpu = _counted(apriori(df, min_support=SPEC.min_support, item_col="items"), SPEC.n_rows)
    got = _counted(
        apriori(df, min_support=SPEC.min_support, item_col="items", use_gpu=True, gpu_resident=True), SPEC.n_rows
    )
    assert len(cpu) > 0
    assert got == cpu
