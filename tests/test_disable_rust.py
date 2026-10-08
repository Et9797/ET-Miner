"""ET_MINER_DISABLE_RUST=1 makes every Rust role take its fallback.

The switch is read once, when et_miner.backends is imported, so each case runs
in a fresh interpreter with the variable set before the first et_miner import.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("et_miner_rust", reason="the switch only matters when the extension is built")

_PROBE = r"""
import json
import numpy as np
import polars as pl

import et_miner
from et_miner import backends, build_boolean_matrix, count_support_batched
from et_miner.core import sparse
from et_miner.gpu import bitvec
from et_miner.gpu.kernels.k3plus import _build_k3plus_groups_numpy, build_k3plus_groups_from_flat

flat = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]], dtype=np.int32)
g = build_k3plus_groups_from_flat(flat)
ref = _build_k3plus_groups_numpy(flat)
rows = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3], [0, 1]] * 20
matrix, col_to_item, n = build_boolean_matrix(pl.DataFrame({"items": rows}).lazy(), 0.3)
cols = sorted(col_to_item)
itemsets = [(a,) for a in cols] + [(a, b) for a in cols for b in cols if a < b]
itemsets += [(a, b, c) for a in cols for b in cols for c in cols if a < b < c]
counts = count_support_batched(matrix, itemsets, n, sparse=True, enable_length_filter=False)
print(json.dumps({
    "backends": [backends.RUST_INSTALLED, backends.get_rust_ext() is None, backends.RUST_DISABLED],
    "copies": [et_miner.HAS_RUST, sparse.RUST_INSTALLED, bitvec.RUST_INSTALLED],
    "groups_equal": bool(np.array_equal(g.suffixes, ref.suffixes)
                         and np.array_equal(g.cumulative_pairs, ref.cumulative_pairs)),
    "counts": sorted((sorted(col_to_item[c] for c in k), int(v)) for k, v in counts.items()),
}))
"""


def _probe(disable: bool) -> tuple[dict, str]:
    env = {k: v for k, v in os.environ.items() if k != "ET_MINER_DISABLE_RUST"}
    if disable:
        env["ET_MINER_DISABLE_RUST"] = "1"
    proc = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1]), proc.stderr


def test_the_switch_disables_every_detection_point_and_changes_no_result():
    off, off_log = _probe(disable=True)
    on, _ = _probe(disable=False)

    assert off["backends"] == [False, True, True]
    assert off["copies"] == [False, False, False]
    assert on["backends"] == [True, False, False]
    assert on["copies"] == [True, True, True]

    assert off["groups_equal"] and on["groups_equal"]
    assert off["counts"] == on["counts"]
    assert len(off["counts"]) == 14
    assert "disabled by ET_MINER_DISABLE_RUST=1" in off_log


def test_apriori_from_csr_names_the_switch():
    code = "import et_miner\ntry:\n    et_miner.apriori_from_csr()\nexcept Exception as e:\n    print(type(e).__name__, e)"
    env = {**os.environ, "ET_MINER_DISABLE_RUST": "1"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120).stdout
    assert "MiningError" in out and "ET_MINER_DISABLE_RUST=1" in out
