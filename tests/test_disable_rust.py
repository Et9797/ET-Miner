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
from et_miner import apriori, backends
from et_miner.core import sparse
from et_miner.gpu import bitvec
from et_miner.gpu.kernels.k3plus import _build_k3plus_groups_numpy, build_k3plus_groups_from_flat

flat = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]], dtype=np.int32)
g = build_k3plus_groups_from_flat(flat)
ref = _build_k3plus_groups_numpy(flat)
rows = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3], [0, 1]] * 20
res = apriori(pl.DataFrame({"items": rows}), min_support=0.3, sparse=True)
print(json.dumps({
    "backends": [backends.RUST_INSTALLED, backends.get_rust_ext() is None, backends.RUST_DISABLED],
    "copies": [et_miner.HAS_RUST, sparse.RUST_INSTALLED, bitvec.RUST_INSTALLED],
    "groups_equal": bool(np.array_equal(g.suffixes, ref.suffixes)
                         and np.array_equal(g.cumulative_pairs, ref.cumulative_pairs)),
    "lattice": sorted((sorted(s), round(p * len(rows))) for s, p in zip(res["itemset"].to_list(), res["support"].to_list())),
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
    assert off["lattice"] == on["lattice"]
    assert "disabled by ET_MINER_DISABLE_RUST=1" in off_log


def test_apriori_from_csr_names_the_switch():
    code = "import et_miner\ntry:\n    et_miner.apriori_from_csr()\nexcept Exception as e:\n    print(type(e).__name__, e)"
    env = {**os.environ, "ET_MINER_DISABLE_RUST": "1"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120).stdout
    assert "MiningError" in out and "ET_MINER_DISABLE_RUST=1" in out
