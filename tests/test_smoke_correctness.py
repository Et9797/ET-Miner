"""Smoke test: verify et_miner CPU, GPU, and efficient_apriori produce identical results.

Uses the real Online Retail II dataset at multiple support levels. Catches
regressions in kernel correctness, pruning logic, and counting arithmetic.

The dataset is generated locally (it is gitignored):

    python datasets/prepare_online_retail.py
"""

from pathlib import Path

import pytest

from tests._oracle import ea_counts, mined_counts

pytest.importorskip("efficient_apriori", reason="efficient_apriori not installed")

DATASET = Path(__file__).parent.parent / "datasets" / "online_retail_ii" / "transactions.parquet"
SKIP_NO_DATASET = pytest.mark.skipif(
    not DATASET.exists(),
    reason=f"Dataset not found: {DATASET} — generate it with: python datasets/prepare_online_retail.py",
)


def _run_et_miner(lf, min_support, use_gpu=False) -> dict[tuple, int]:
    """{sorted itemset: absolute count}."""
    import polars as pl

    from et_miner import apriori
    res = apriori(lf, min_support=min_support, item_col="items", use_gpu=use_gpu)
    return mined_counts(res, lf.select(pl.len()).collect().item())


def _run_efficient_apriori(lf, min_support) -> dict[tuple, int]:
    return ea_counts(lf.select("items").collect(engine="streaming").to_series().to_list(), min_support)


@SKIP_NO_DATASET
class TestCPUvsEfficientApriori:
    """Verify et_miner CPU matches efficient_apriori exactly."""

    def _load(self):
        import polars as pl
        return pl.scan_parquet(DATASET)

    def test_support_005(self):
        lf = self._load()
        cpu = _run_et_miner(lf, 0.005)
        ea = _run_efficient_apriori(lf, 0.005)
        assert cpu == ea
        assert cpu

    @pytest.mark.slow
    def test_support_001(self):
        lf = self._load()
        cpu = _run_et_miner(lf, 0.001)
        ea = _run_efficient_apriori(lf, 0.001)
        assert cpu == ea
        assert cpu

    @pytest.mark.slow
    def test_support_0001(self):
        lf = self._load()
        cpu = _run_et_miner(lf, 0.0001)
        ea = _run_efficient_apriori(lf, 0.0001)
        assert cpu == ea
        assert cpu


@SKIP_NO_DATASET
@pytest.mark.gpu
class TestGPUvsCPU:
    """Verify the GPU path produces the CPU's itemsets and counts."""

    def _load(self):
        import polars as pl
        return pl.scan_parquet(DATASET)

    def test_support_005(self):
        lf = self._load()
        cpu = _run_et_miner(lf, 0.005, use_gpu=False)
        gpu = _run_et_miner(lf, 0.005, use_gpu=True)
        assert cpu == gpu
        assert cpu

    @pytest.mark.slow
    def test_support_001(self):
        lf = self._load()
        cpu = _run_et_miner(lf, 0.001, use_gpu=False)
        gpu = _run_et_miner(lf, 0.001, use_gpu=True)
        assert cpu == gpu
        assert cpu

    @pytest.mark.slow
    def test_support_0001(self):
        lf = self._load()
        cpu = _run_et_miner(lf, 0.0001, use_gpu=False)
        gpu = _run_et_miner(lf, 0.0001, use_gpu=True)
        assert cpu == gpu
        assert cpu
