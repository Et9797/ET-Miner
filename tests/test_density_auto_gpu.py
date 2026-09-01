"""GPU validation of the density-adaptive dense→sparse transition.

The decision logic is CPU-tested in test_density_transition.py; these tests
prove on-device that (a) mining results are identical whether the CSR
transition never fires, fires at a fixed K, or fires adaptively, and
(b) the "auto" mode actually transitions on data shaped to cross the n/32
crossover.
"""

import pytest

cp = pytest.importorskip("cupy", reason="cupy not installed")

pytestmark = pytest.mark.gpu

from loguru import logger

from et_miner.core.apriori import apriori
from et_miner.synthetic import SynthSpec, generate_transactions

# Sparse-shaped: K>=2 survivor mean support sits well under the n/32
# (~3.1%) crossover, so "auto" should convert to CSR tidsets at K=3.
SPEC = SynthSpec(
    name="density_auto",
    n_rows=50_000,
    vocab_size=1_500,
    zipf_a=1.3,
    row_len_mean=10,
    row_len_max=40,
    motif_count=4,
    motif_size=6,
    motif_penetration=0.015,
    min_support=0.004,
    seed=23,
)


@pytest.fixture(scope="module")
def dataset():
    df, data = generate_transactions(SPEC)
    return df, data


def _mine(df, sparse_from_k):
    res = apriori(
        df,
        min_support=SPEC.min_support,
        item_col="items",
        use_gpu=True,
        n_gpus=2,
        sparse_from_k=sparse_from_k,
    )
    return {
        (tuple(sorted(int(i) for i in s)), round(sup * SPEC.n_rows))
        for s, sup in zip(res["itemset"].to_list(), res["support"].to_list())
    }


def test_auto_equals_never(dataset):
    df, _ = dataset
    dense_only = _mine(df, None)
    assert len(dense_only) > 0
    assert _mine(df, "auto") == dense_only


def test_fixed_k_equals_never(dataset):
    """Forcing the CSR path from K=3 must not change results either —
    dense-vs-sparse counting equivalence on device."""
    df, _ = dataset
    assert _mine(df, 3) == _mine(df, None)


def test_auto_transition_fires(dataset):
    """On this sparse-shaped preset the measured mean support falls under
    the n/32 crossover, so auto must actually convert (observed via the
    DENSITY TRANSITION log line, captured with a loguru sink)."""
    df, _ = dataset
    records: list[str] = []
    sink_id = logger.add(lambda m: records.append(str(m)), level="INFO")
    try:
        _mine(df, "auto")
    finally:
        logger.remove(sink_id)
    transitions = [r for r in records if "DENSITY TRANSITION" in r]
    assert transitions, "auto mode never transitioned on sparse-shaped data"
    assert any("measured mean support" in r for r in transitions)
