"""Smoke test: verify et_miner CPU, GPU, and efficient_apriori produce identical results.

Uses the real Online Retail II dataset at multiple support levels. Catches
regressions in kernel correctness, pruning logic, and counting arithmetic.

The dataset is generated locally (it is gitignored):

    python datasets/prepare_online_retail.py
"""

from pathlib import Path

import pytest

pytest.importorskip("efficient_apriori", reason="efficient_apriori not installed")

DATASET = Path(__file__).parent.parent / "datasets" / "online_retail_ii" / "transactions.parquet"
SKIP_NO_DATASET = pytest.mark.skipif(
    not DATASET.exists(),
    reason=f"Dataset not found: {DATASET} — generate it with: python datasets/prepare_online_retail.py",
)


def _run_et_miner(lf, min_support, use_gpu=False):
    from et_miner import apriori
    return apriori(lf, min_support=min_support, item_col="items", use_gpu=use_gpu)


def _run_efficient_apriori(lf, min_support):
    from efficient_apriori import apriori as ea_apriori
    transactions = lf.select("items").collect(engine="streaming").to_series().to_list()
    itemsets, _ = ea_apriori(transactions, min_support=min_support, max_length=100)
    return sum(len(v) for v in itemsets.values())


@SKIP_NO_DATASET
class TestCPUvsEfficientApriori:
    """Verify et_miner CPU matches efficient_apriori exactly."""

    def _load(self):
        import polars as pl
        return pl.scan_parquet(DATASET)

    def test_support_005(self):
        lf = self._load()
        cpu = _run_et_miner(lf, 0.005).height
        ea = _run_efficient_apriori(lf, 0.005)
        assert cpu == ea
        assert cpu > 0

    def test_support_001(self):
        lf = self._load()
        cpu = _run_et_miner(lf, 0.001).height
        ea = _run_efficient_apriori(lf, 0.001)
        assert cpu == ea
        assert cpu > 0

    @pytest.mark.slow
    def test_support_0001(self):
        lf = self._load()
        cpu = _run_et_miner(lf, 0.0001).height
        ea = _run_efficient_apriori(lf, 0.0001)
        assert cpu == ea
        assert cpu > 0


@SKIP_NO_DATASET
@pytest.mark.gpu
class TestGPUvsCPU:
    """Verify GPU path produces identical counts to CPU."""

    def _load(self):
        import polars as pl
        return pl.scan_parquet(DATASET)

    def test_support_005(self):
        lf = self._load()
        cpu = _run_et_miner(lf, 0.005, use_gpu=False).height
        gpu = _run_et_miner(lf, 0.005, use_gpu=True).height
        assert cpu == gpu

    def test_support_001(self):
        lf = self._load()
        cpu = _run_et_miner(lf, 0.001, use_gpu=False).height
        gpu = _run_et_miner(lf, 0.001, use_gpu=True).height
        assert cpu == gpu

    @pytest.mark.slow
    def test_support_0001(self):
        lf = self._load()
        cpu = _run_et_miner(lf, 0.0001, use_gpu=False).height
        gpu = _run_et_miner(lf, 0.0001, use_gpu=True).height
        assert cpu == gpu
