"""SON completeness: a chunk that fails must fail the run.

SON's guarantee is that every globally frequent itemset is locally frequent in
at least one chunk (pass 1) and that pass 2 counts every chunk. A chunk that is
logged and skipped breaks both halves silently: its locally frequent itemsets
never become candidates, and its counts never reach the global total while
support is still divided by the full row count.

The GPU halves mine each chunk on the row-split miner and count pass 2 with the
batched itemset kernel. The multi-GPU half also checks two things that made its
chunks fail inside a worker, where the failure was then dropped: GPU counting
needs no Rust extension (the bitvec build and the count are CUDA), and a
worker's bitvecs are built on that worker's device (without peer access, a
chunk counted on device 1 against bitvecs on device 0 raises).
"""

from __future__ import annotations

import random

import numpy as np
import polars as pl
import pytest

from et_miner import apriori
from et_miner.streaming import son

N_ROWS = 3_000
CHUNK = 1_000
MIN_SUPPORT = 0.05


@pytest.fixture(scope="module")
def transactions() -> pl.DataFrame:
    rng = random.Random(7)
    rows = []
    for _ in range(N_ROWS):
        size = rng.randint(2, 7)
        rows.append(sorted(rng.sample(range(24), size)))
    return pl.DataFrame({"items": rows})


def _as_set(df: pl.DataFrame) -> set[tuple[tuple[int, ...], int]]:
    return {
        (tuple(sorted(int(i) for i in s)), round(sup * N_ROWS))
        for s, sup in zip(df["itemset"].to_list(), df["support"].to_list())
    }


class _FailOnCall:
    """Wrap a callable so that its `n`-th call raises."""

    def __init__(self, fn, n: int):
        self.fn, self.n, self.calls = fn, n, 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls == self.n:
            raise RuntimeError(f"injected failure on call {self.n}")
        return self.fn(*args, **kwargs)


@pytest.mark.parametrize("target", ["_local_levels", "_count_chunk"])
def test_cpu_son_raises_when_a_chunk_fails(transactions, monkeypatch, target):
    """Pass 1 (_local_levels) and pass 2 (_count_chunk) each fail the run on their second chunk."""
    failing = _FailOnCall(getattr(son, target), 2)
    monkeypatch.setattr(son, target, failing)
    with pytest.raises(RuntimeError, match="injected failure on call 2"):
        son.apriori_streaming(transactions, min_support=MIN_SUPPORT, chunk_size=CHUNK, show_progress=False)
    assert failing.calls == 2


def test_cpu_son_matches_in_core_when_nothing_fails(transactions):
    got = son.apriori_streaming(transactions, min_support=MIN_SUPPORT, chunk_size=CHUNK, show_progress=False)
    assert _as_set(got) == _as_set(apriori(transactions, min_support=MIN_SUPPORT))


def test_gpu_resident_is_refused(transactions):
    with pytest.raises(ValueError, match="gpu_resident was removed"):
        son.apriori_streaming(transactions, min_support=MIN_SUPPORT, chunk_size=CHUNK, gpu_resident=True)


@pytest.mark.gpu
class TestSingleGpuSonOnTheGpu:
    def _run(self, transactions):
        return son.apriori_streaming(
            transactions, min_support=MIN_SUPPORT, chunk_size=CHUNK, use_gpu=True, show_progress=False
        )

    def test_matches_in_core(self, transactions):
        assert _as_set(self._run(transactions)) == _as_set(apriori(transactions, min_support=MIN_SUPPORT))

    @pytest.mark.parametrize(
        "rows",
        [
            pytest.param([["a", "b"], ["a", "b", "c"], ["b", "c"]] * 20, id="string-items"),
            pytest.param([[1, 2**40], [1, 2**40, 3], [2**40, 3]] * 20, id="ids-beyond-int32"),
            pytest.param([[1, 2], [1, 2, 3], [2, 3]] * 20 + [[]] * 40, id="empty-baskets"),
        ],
    )
    def test_items_of_any_type_mine_as_on_the_cpu(self, rows):
        """The chunk miner runs on column indices and maps the items back."""
        df = pl.DataFrame({"items": rows})
        kw = dict(min_support=0.1, chunk_size=30, show_progress=False)
        gpu = son.apriori_streaming(df, use_gpu=True, **kw)
        cpu = son.apriori_streaming(df, **kw)
        assert sorted(map(tuple, gpu["itemset"].to_list())) == sorted(map(tuple, cpu["itemset"].to_list()))
        assert len(cpu) > 0

    @pytest.mark.parametrize("target", ["_mine_chunk_gpu", "_count_candidates_gpu"])
    def test_a_failed_chunk_raises(self, transactions, monkeypatch, target):
        failing = _FailOnCall(getattr(son, target), 2)
        monkeypatch.setattr(son, target, failing)
        with pytest.raises(RuntimeError, match="injected failure on call 2"):
            self._run(transactions)
        assert failing.calls == 2


@pytest.mark.gpu
class TestMultiGpuSon:
    @pytest.fixture(autouse=True)
    def _module(self):
        from et_miner.streaming import multi_gpu

        self.mg = multi_gpu

    def _run(self, transactions):
        return self.mg.apriori_streaming_multi_gpu(
            transactions, min_support=MIN_SUPPORT, n_gpus=2, chunk_size=CHUNK, show_progress=False
        )

    @pytest.mark.parametrize("target", ["build_boolean_matrix", "_mine_chunk_gpu"])
    def test_a_failed_pass1_chunk_raises(self, transactions, monkeypatch, target):
        monkeypatch.setattr(self.mg, target, _FailOnCall(getattr(self.mg, target), 2))
        with pytest.raises(RuntimeError, match="injected failure"):
            self._run(transactions)

    @pytest.mark.parametrize("target", ["_build_matrix_for_items", "_count_candidates_gpu"])
    def test_a_failed_pass2_chunk_raises(self, transactions, monkeypatch, target):
        monkeypatch.setattr(self.mg, target, _FailOnCall(getattr(self.mg, target), 2))
        with pytest.raises(RuntimeError, match="injected failure"):
            self._run(transactions)

    def test_matches_in_core_without_the_rust_extension(self, transactions, monkeypatch):
        from et_miner.gpu import bitvec

        monkeypatch.setattr(bitvec, "RUST_INSTALLED", False)
        monkeypatch.setattr(bitvec, "get_rust_ext", lambda: None)
        got = self._run(transactions)
        assert _as_set(got) == _as_set(apriori(transactions, min_support=MIN_SUPPORT))


@pytest.mark.gpu
def test_gpu_bitvec_counting_needs_no_rust_extension(transactions, monkeypatch):
    from et_miner.core.matrix import build_boolean_matrix, count_support_vectorized
    from et_miner.gpu import bitvec

    monkeypatch.setattr(bitvec, "RUST_INSTALLED", False)
    monkeypatch.setattr(bitvec, "get_rust_ext", lambda: None)
    matrix, col_to_item, _ = build_boolean_matrix(transactions.lazy(), MIN_SUPPORT)
    cols = list(col_to_item)
    itemsets = [(cols[0], cols[1]), (cols[1], cols[2], cols[3]), (cols[4],)]
    assert bitvec.count_support_gpu_bitvec(matrix, itemsets) == count_support_vectorized(matrix, itemsets)


@pytest.mark.gpu
def test_gpu_counting_handles_no_rows_and_rejects_the_empty_itemset(transactions):
    """A length filter can leave no row at all; the batched kernel is then not launched."""
    import cupy as cp
    from et_miner.core.matrix import build_boolean_matrix, count_support_batched
    from et_miner.gpu.kernels import count_itemsets_cuda

    rows = pl.DataFrame({"items": [[1], [2], [1], [2], [3]]})
    matrix, col_to_item, _ = build_boolean_matrix(rows.lazy(), 0.0)
    cols = list(col_to_item)
    got = count_support_batched(matrix, [(cols[0], cols[1])], rows.height, use_gpu=True)
    assert got == {(cols[0], cols[1]): 0}
    assert count_itemsets_cuda(cp.zeros((3, 0), dtype=cp.uint64), [np.array([0, 1], np.int32)]).tolist() == [0]
    with pytest.raises(ValueError, match="at least one item"):
        count_itemsets_cuda(cp.zeros((3, 2), dtype=cp.uint64), [np.array([], np.int32)])


@pytest.mark.gpu
@pytest.mark.multigpu
def test_bitvecs_are_built_on_the_current_device(transactions):
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("needs 2 CUDA devices")
    from et_miner.core.matrix import _build_csr_from_transactions
    from et_miner.gpu.bitvec import _build_gpu_bitvec_matrix

    csr, _, _ = _build_csr_from_transactions(transactions.lazy(), MIN_SUPPORT)
    with cp.cuda.Device(1):
        bv = _build_gpu_bitvec_matrix(csr)
    assert bv.device.id == 1
