"""The Rust fast paths may fall back, but never in silence.

``et_miner_rust`` is optional in the sense that the itemsets are identical
without it, not in the sense that the run is: ``prune_groups_apriori`` and
``build_k3plus_groups_from_flat`` back candidate generation on the
downward-closure row-split path, which was measured at 93% of mining time
over a nine-level run (K=2..K=10). A routine ``uv sync`` in a consumer project prunes the
extension, the import fails, and before these warnings nothing in any log
said the run had changed cost.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from loguru import logger

import et_miner.backends as backends
from et_miner.backends import BUILD_COMMAND
import et_miner.gpu.kernels.k3plus as k3plus
from et_miner.gpu.kernels.k3plus import build_k3plus_groups_from_flat
from et_miner.gpu.mining import _prune_groups_apriori


@pytest.fixture
def warnings_captured():
    """Collect loguru WARNING messages -- they do not route through caplog."""
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    yield messages
    logger.remove(sink)


@pytest.fixture(autouse=True)
def _reset_once_flags():
    """The warnings are once-per-process; each test needs its own first time."""
    k3plus._MISSING_RUST_WARNED = False
    k3plus._STALE_RUST_WARNED = False
    yield
    k3plus._MISSING_RUST_WARNED = False
    k3plus._STALE_RUST_WARNED = False


def _prev_flat(k: int = 3) -> np.ndarray:
    """Sorted (k)-itemsets over one prefix, enough to build groups from."""
    rows = [(0, 1, s) for s in range(2, 8)] if k == 3 else [(0, s) for s in range(1, 8)]
    return np.array(rows, dtype=np.int32)


def test_absent_extension_warns_in_the_group_builder(monkeypatch, warnings_captured):
    monkeypatch.setattr(backends, "_rust_ext", None)
    groups = build_k3plus_groups_from_flat(_prev_flat(), with_src_rows=True)
    assert groups is not None, "the numpy fallback must still answer"
    assert len(warnings_captured) == 1
    assert "et_miner_rust is not installed" in warnings_captured[0]
    assert BUILD_COMMAND in warnings_captured[0], "the warning must quote the one recipe in backends.py"


def test_absent_extension_warns_in_the_apriori_prune(monkeypatch, warnings_captured):
    # A clique, so the prune has survivors: every 3-subset of {0..4} is
    # present, and the 4-itemsets built from them keep all four subsets.
    flat = np.array(list(itertools.combinations(range(5), 3)), dtype=np.int32)
    groups = k3plus._build_k3plus_groups_numpy(flat, with_src_rows=True)
    monkeypatch.setattr(backends, "_rust_ext", None)
    pruned = _prune_groups_apriori(groups, None, 4, prev_flat_np=flat)
    assert pruned is not None and pruned.total_candidates > 0
    assert len(warnings_captured) == 1
    assert "et_miner_rust is not installed" in warnings_captured[0]


def test_the_warning_fires_once_not_once_per_level(monkeypatch, warnings_captured):
    """One line per process. Sixteen levels must not be sixteen warnings."""
    monkeypatch.setattr(backends, "_rust_ext", None)
    for _ in range(5):
        build_k3plus_groups_from_flat(_prev_flat(), with_src_rows=True)
    assert len(warnings_captured) == 1


def test_a_raising_extension_is_reported_every_time(monkeypatch, warnings_captured):
    """A malfunction is not a build state: the fallback keeps the campaign
    alive, but each occurrence has to be visible."""

    class Exploding:
        def build_k3plus_groups_from_flat(self, *args):
            raise RuntimeError("boom")

    monkeypatch.setattr(backends, "_rust_ext", Exploding())
    flat = _prev_flat()
    expected = k3plus._build_k3plus_groups_numpy(flat, with_src_rows=True)

    for _ in range(3):
        groups = build_k3plus_groups_from_flat(flat, with_src_rows=True)
        assert groups is not None
        np.testing.assert_array_equal(groups.suffixes, expected.suffixes)

    assert len(warnings_captured) == 3
    assert all("RuntimeError: boom" in m for m in warnings_captured)
    assert all("build_k3plus_groups_from_flat" in m for m in warnings_captured)
