"""Moonshot campaign tooling: M1's matrices pin what bench/moonshots/PROTOCOL-M1.md fixes, decide.py applies
rules 1 and 2 mechanically, and the level split times the device input layer's phases."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "moonshots"))

from consolidation_matrix import (
    MOONSHOT_REGIMES,
    build_moonshots_m1,
    build_moonshots_m1_final,
    build_moonshots_m2,
    build_moonshots_m2_final,
)
from decide import decide
from level_split import LevelSplit
from runner import build_matrix

THREADS = ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")


@pytest.mark.parametrize("n_dev", [1, 2])
def test_m1_matrix_has_every_regime_arm_and_gpu_count_rep_major(n_dev):
    m = build_moonshots_m1(n_dev)
    gpus = [1, 2][:n_dev]
    assert len(m) == len(MOONSHOT_REGIMES) * 2 * len(gpus) * 3
    assert [c["rep"] for c in m] == sorted(c["rep"] for c in m)
    assert len({c["id"] for c in m}) == len(m)
    per_rep = [c["base_id"] for c in m if c["rep"] == 0]
    assert per_rep == [c["base_id"] for c in m if c["rep"] == 2]
    assert {c["n_gpus"] for c in m} == set(gpus)
    assert build_matrix("moonshots-m1", n_dev, []) == m


def test_m1_arms_pin_the_layer_threads_and_devices():
    for c in build_moonshots_m1(2):
        arm = c["base_id"].rsplit("-", 1)[1]
        assert c["env"]["ET_MINER_INPUT_LAYER"] == arm and arm in ("host", "device")
        assert all(c["env"][t] == "16" for t in THREADS)
        assert c["env"]["NCCL_P2P_DISABLE"] == "1" and c["level_split"]
        if c["n_gpus"] == 1:
            assert c["env"]["CUDA_VISIBLE_DEVICES"] == "0" and c["env"]["ET_MINER_DISABLE_NCCL"] == "1"
        knobs = {k for k in c["env"] if k.startswith("ET_MINER_")} - {"ET_MINER_INPUT_LAYER", "ET_MINER_DISABLE_NCCL"}
        assert not knobs, f"{c['id']} pins {knobs}"


def test_m1_final_runs_every_regime_once_with_the_layer_unset():
    m = build_moonshots_m1_final(2)
    assert len(m) == len(MOONSHOT_REGIMES) * 2
    assert all(c["rep"] == 0 and "ET_MINER_INPUT_LAYER" not in c["env"] for c in m)
    assert build_matrix("moonshots-m1-final", 2, []) == m


@pytest.mark.parametrize("n_dev", [1, 2])
def test_m2_matrix_pins_the_k3_kernel_per_arm_rep_major(n_dev):
    m = build_moonshots_m2(n_dev)
    assert len(m) == len(MOONSHOT_REGIMES) * 2 * n_dev * 3
    assert [c["rep"] for c in m] == sorted(c["rep"] for c in m)
    for c in m:
        arm = c["base_id"].rsplit("-", 1)[1]
        assert c["env"]["ET_MINER_K3_KERNEL"] == arm and arm in ("dense", "rows")
        assert all(c["env"][t] == "16" for t in THREADS) and c["level_split"]
        assert "ET_MINER_INPUT_LAYER" not in c["env"]
    final = build_moonshots_m2_final(n_dev)
    assert len(final) == len(MOONSHOT_REGIMES) * n_dev
    assert all(c["rep"] == 0 and "ET_MINER_K3_KERNEL" not in c["env"] for c in final)
    assert build_matrix("moonshots-m2", n_dev, []) == m and build_matrix("moonshots-m2-final", n_dev, []) == final


def _row(id_, wall, sig=("n", 1, "h"), fallbacks=(), status="ok", free=False):
    return {
        "id": id_,
        "status": status,
        "wall_s": wall,
        "n_itemsets": sig[0],
        "sum_counts": sig[1],
        "itemset_hash": sig[2],
        "fallbacks": list(fallbacks),
        "timings": {"span": {"input": wall / 2}},
        "config": {
            "dataset": id_.split("-")[0],
            "min_support": 0.01,
            "max_length": None,
            "prune_equal_support": free,
            "allow_fallback": ["NCCL unavailable"],
        },
    }


def test_decide_applies_rule_2_to_each_regime():
    rows = []
    for rep, (h, d) in enumerate([(10.0, 5.0), (10.2, 5.1), (9.9, 4.9)]):
        rows += [_row(f"dsl-C1-host#r{rep}", h), _row(f"dsl-C1-device#r{rep}", d)]
    for rep, (h, d) in enumerate([(0.30, 0.20), (0.31, 0.21), (0.29, 0.19)]):
        rows += [_row(f"smoke-C1-host#r{rep}", h), _row(f"smoke-C1-device#r{rep}", d)]
    for rep, (h, d) in enumerate([(5.0, 7.0), (5.1, 7.1), (4.9, 6.9)]):
        rows += [_row(f"sk2ml3-C2-host#r{rep}", h), _row(f"sk2ml3-C2-device#r{rep}", d)]
    res = decide(rows, ["host", "device"])
    verdict = {(e["workload"], e["n_gpus"]): e["arms"]["device"]["vs_host"] for e in res["regimes"]}
    assert verdict[("dsl", 1)]["wins"] and verdict[("dsl", 1)]["saved_s"] == pytest.approx(5.0)
    assert not verdict[("smoke", 1)]["wins"] and not verdict[("smoke", 1)]["one_s"]
    assert verdict[("sk2ml3", 2)]["loses"] and not verdict[("sk2ml3", 2)]["wins"]
    assert res["must_fix"] == []


def test_decide_reads_labels_and_overlapping_ranges_tie():
    rows = []
    for rep, (h, d) in enumerate([(10.0, 8.0), (10.5, 8.2), (7.9, 8.1)]):
        rows += [_row(f"dsl-C1-infer-host#r{rep}", h), _row(f"dsl-C1-infer-device#r{rep}", d)]
    (e,) = decide(rows, ["host", "device"])["regimes"]
    assert e["label"] == "infer-" and not e["arms"]["device"]["vs_host"]["disjoint"]
    assert not e["arms"]["device"]["vs_host"]["wins"]


def test_decide_flags_signature_mismatches_fallbacks_and_errors():
    rows = [
        _row("dsl-C1-host#r0", 10.0),
        _row("dsl-C2-device#r0", 5.0, sig=("n", 2, "x")),
        _row("dsl-C1-device#r0", 5.0, fallbacks=["Rust role fell back"]),
        _row("dsl-C2-host#r0", 5.0, fallbacks=["NCCL unavailable, staged reduce"]),
        _row("deepk-C1-device#r0", 1.0, status="error: OutOfMemoryError"),
    ]
    fix = decide(rows, ["host", "device"])["must_fix"]
    assert any("signature mismatch" in f for f in fix)
    assert any("Rust role fell back" in f for f in fix)
    assert not any("NCCL" in f for f in fix)
    assert any("deepk-C1-device#r0" in f for f in fix)


@pytest.mark.gpu
def test_level_split_times_the_device_input_layer(monkeypatch):
    from et_miner import apriori

    monkeypatch.setenv("ET_MINER_INPUT_LAYER", "device")
    monkeypatch.setenv("ET_MINER_K2_KERNEL", "rows")
    rng = np.random.default_rng(3)
    rows = [sorted(set(rng.integers(0, 30, size=int(rng.integers(1, 8))).tolist())) for _ in range(5_000)]
    df = pl.DataFrame({"items": rows})
    split = LevelSplit()
    split.install()
    try:
        split.start()
        apriori(df, min_support=0.02, use_gpu=True, level_callback=split.callback())
        split.finish()
    finally:
        split.uninstall()
    for phase in ("input_dev_chunks", "input_dev_copy", "input_dev_k1", "input_dev_bitvec"):
        assert split.span[phase] > 0, phase
    assert split.span["input_explode"] == 0 and split.span["input_csr"] == 0
    assert split.levels[2]["k2_dispatch"] > 0 and split.levels[2]["k2_input"] > 0


@pytest.mark.gpu
def test_level_split_times_the_row_wise_k3(monkeypatch):
    from et_miner import apriori

    monkeypatch.setenv("ET_MINER_K3_KERNEL", "rows")
    rng = np.random.default_rng(5)
    rows = [sorted(set(rng.integers(0, 25, size=int(rng.integers(2, 9))).tolist())) for _ in range(6_000)]
    split = LevelSplit()
    split.install()
    try:
        split.start()
        apriori(
            pl.DataFrame({"items": rows}), min_support=0.01, max_length=3, use_gpu=True, level_callback=split.callback()
        )
        split.finish()
    finally:
        split.uninstall()
    for phase in ("k2_dispatch", "k2_input", "k3_graph", "count_rows", "decode"):
        assert split.levels[3][phase] > 0, phase
    assert split.levels[3]["count_tiled"] == split.levels[3]["count_group"] == 0
