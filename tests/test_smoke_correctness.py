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


def _run_et_miner(lf, min_support, use_gpu=False) -> dict[tuple, int]:
    """{sorted itemset: absolute count}."""
    import polars as pl

    from et_miner import apriori
    res = apriori(lf, min_support=min_support, item_col="items", use_gpu=use_gpu)
    n = lf.select(pl.len()).collect().item()
    return {tuple(sorted(s)): round(p * n) for s, p in zip(res["itemset"].to_list(), res["support"].to_list())}


def _run_efficient_apriori(lf, min_support) -> dict[tuple, int]:
    """The oracle with the convention of CLAUDE.md: (min_count - 0.5) / N and an explicit max_length."""
    from efficient_apriori import itemsets_from_transactions

    from et_miner.core.result import _min_count
    transactions = [tuple(t) for t in lf.select("items").collect(engine="streaming").to_series().to_list()]
    n = len(transactions)
    max_length = max(len(set(t)) for t in transactions)
    itemsets, _ = itemsets_from_transactions(transactions, (_min_count(min_support, n) - 0.5) / n, max_length=max_length)
    return {tuple(sorted(k)): v for level in itemsets.values() for k, v in level.items()}


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
