"""Shared pytest fixtures for et-miner tests."""

import random

import polars as pl
import pytest


def _gpu_count() -> int:
    """CUDA device count, 0 when CuPy is absent or no device is usable.

    Kept self-contained (not imported from et_miner) so test collection
    still works even when the package itself is broken.
    """
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def pytest_collection_modifyitems(config, items):
    if _gpu_count() == 0:
        skip_gpu = pytest.mark.skip(reason="no CUDA device available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture
def sample_transactions() -> pl.DataFrame:
    """Basic 4-transaction dataset for tests.

    Items and their support:
    - Item 1: appears in txs 0, 2 -> support = 0.5
    - Item 2: appears in txs 0, 1, 3 -> support = 0.75
    - Item 3: appears in all 4 txs -> support = 1.0
    - Item 4: appears in tx 1 only -> support = 0.25
    - Item 5: appears in tx 2 only -> support = 0.25

    2-itemsets:
    - [1, 3]: txs 0, 2 -> support = 0.5
    - [2, 3]: txs 0, 1, 3 -> support = 0.75
    """
    return pl.DataFrame({
        "items": [[1, 2, 3], [2, 3, 4], [1, 3, 5], [2, 3]]
    })


@pytest.fixture
def single_transaction() -> pl.DataFrame:
    """Single transaction for edge case testing."""
    return pl.DataFrame({
        "items": [[1, 2, 3]]
    })


@pytest.fixture
def empty_transactions() -> pl.DataFrame:
    """Empty DataFrame with correct schema."""
    return pl.DataFrame({
        "items": pl.Series([], dtype=pl.List(pl.Int64))
    })


@pytest.fixture
def custom_column_transactions() -> pl.DataFrame:
    """Transactions with custom column name."""
    return pl.DataFrame({
        "transaction_id": [1, 2, 3],
        "products": [[10, 20], [20, 30], [10, 20, 30]]
    })


@pytest.fixture
def large_transactions() -> pl.DataFrame:
    """1000 random transactions for performance/integration tests."""
    random.seed(42)
    transactions = []
    for _ in range(1000):
        n_items = random.randint(2, 8)
        items = sorted(random.sample(range(1, 51), n_items))
        transactions.append(items)
    return pl.DataFrame({"items": transactions})


@pytest.fixture
def known_frequent_itemsets() -> pl.DataFrame:
    """Pre-computed frequent itemsets for rule generation tests.

    Based on sample_transactions with min_support=0.5:
    - [1]: 0.5
    - [2]: 0.75
    - [3]: 1.0
    - [1, 3]: 0.5
    - [2, 3]: 0.75
    """
    return pl.DataFrame({
        "itemset": [[1], [2], [3], [1, 3], [2, 3]],
        "support": [0.5, 0.75, 1.0, 0.5, 0.75]
    })


@pytest.fixture
def sample_parquet_path(tmp_path):
    """Create a temporary parquet file with sample transactions."""
    df = pl.DataFrame({
        "items": [[1, 2, 3], [2, 3, 4], [1, 2, 4], [1, 3, 4], [2, 3]]
    })
    path = tmp_path / "transactions.parquet"
    df.write_parquet(path)
    return path
