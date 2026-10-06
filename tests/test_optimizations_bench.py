"""Phase A and B campaign tooling: the matrices pin what bench/optimizations/PROTOCOL.md and PROTOCOL-B.md fix,
the K=2 sweep's r* rule, and the level split's CSR phase."""

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from consolidation_matrix import (
    O5_ESCO_CAP_S,
    OPTIMIZATION_ARMS,
    WORKLOADS,
    build_o5_calibration_matrix,
    build_optimizations_calibration,
    build_optimizations_final,
    build_optimizations_matrix,
    calibration_id,
)
from k2_crossover import N_ROWS, check_points, crossover, generate_rows, grid, r_of
from level_split import LevelSplit
from runner import build_matrix

KNOBS = ("ET_MINER_K2_KERNEL", "ET_MINER_REDUCE", "ET_MINER_ESCO_MATERIALIZE")
THREADS = ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")


def _calibrated(*statuses):
    return [{"id": calibration_id(n), "status": s} for n, s in enumerate(statuses, start=1)]


def _arm(c):
    return next(a for a, env in OPTIMIZATION_ARMS.items() if all(c["env"][k] == env[k] for k in KNOBS))


def _regime(c):
    w = c["base_id"].split("-C", 1)[0]
    variant = {3: "-esco", "auto": "-esco"}.get(c.get("sparse_from_k"), "")
    variant += "-infer" if c.get("use_generator_pruning") else ""
    return f"{w}{variant}", c["n_gpus"]


def test_matrix_refuses_to_start_before_the_calibration():
    with pytest.raises(ValueError, match="optimizations-calibration"):
        build_optimizations_matrix(2, _calibrated("ok"))


def test_matrix_has_the_protocol_regimes_and_arms():
    matrix = build_optimizations_matrix(2, _calibrated("ok", "timeout"))
    assert len({c["id"] for c in matrix}) == len(matrix)
    arms: dict[tuple, set] = {}
    for c in matrix:
        arms.setdefault(_regime(c), set()).add(_arm(c))
    one, two = {"base", "rows"}, {"base", "rows", "compact"}
    expected = {(w, 1): one for w in ("smoke", "deepk", "skew", "or003", "or002", "dsl", "oom2ml3", "sk2ml3")}
    expected |= {(w, 2): two for w in ("dsl", "oom2ml3", "sk2ml3")}
    expected[("dsl-infer", 2)] = {"base", "compact"}
    expected |= {(f"{w}-esco", n): {"base", "reuse"} for w in ("deepk", "or002") for n in (1, 2)}
    expected[("dsl-esco", 1)] = {"base", "reuse"}
    assert arms == expected
    assert {c["rep"] for c in matrix} == {0, 1, 2}


def test_matrix_drops_dsl_esco_where_the_calibration_failed():
    regimes = {_regime(c) for c in build_optimizations_matrix(2, _calibrated("error: OOM", "ok"))}
    assert ("dsl-esco", 2) in regimes and ("dsl-esco", 1) not in regimes
    assert all(r[1] == 1 for r in {_regime(c) for c in build_optimizations_matrix(1, _calibrated("ok"))})


def test_every_config_pins_the_knobs_threads_and_devices():
    configs = build_optimizations_matrix(2, _calibrated("ok", "ok")) + build_optimizations_calibration(2)
    for c in configs:
        assert all(c["env"][k] for k in KNOBS), c["id"]
        assert all(c["env"][t] == "6" for t in THREADS), c["id"]
        assert c["env"]["NCCL_P2P_DISABLE"] == "1"
        assert c["level_split"] and c["route"] == "C"
        if c["n_gpus"] == 1:
            assert c["env"]["CUDA_VISIBLE_DEVICES"] == "0" and c["env"]["ET_MINER_DISABLE_NCCL"] == "1"
        else:
            assert "CUDA_VISIBLE_DEVICES" not in c["env"] and "ET_MINER_DISABLE_NCCL" not in c["env"]


def test_final_check_runs_every_campaign_regime_once_with_the_knobs_unset():
    final = build_optimizations_final(2)
    campaign = {_regime(c) for c in build_optimizations_matrix(2, _calibrated("error: OOM", "error: OOM"))}
    regimes = [_regime(c) for c in final]
    assert len(set(regimes)) == len(regimes) and set(regimes) == campaign
    assert all(c["rep"] == 0 and c["level_split"] for c in final)
    assert not any(k in c["env"] for c in final for k in KNOBS)
    assert all(_regime(c)[1] == 1 for c in build_optimizations_final(1))


def test_calibration_is_dsl_esco_base_once_per_gpu_count_capped():
    cal = build_optimizations_calibration(2)
    assert [c["id"] for c in cal] == [calibration_id(1), calibration_id(2)]
    assert all(_arm(c) == "base" and c["sparse_from_k"] == "auto" and c["timeout_s"] == 600 for c in cal)
    assert all(_regime(c)[0] == "dsl-esco" for c in cal)


def test_order_is_rep_major_short_then_esco_then_long():
    matrix = build_optimizations_matrix(2, _calibrated("ok", "ok"))
    assert [c["rep"] for c in matrix] == sorted(c["rep"] for c in matrix)
    rank = {"short": 0, "esco": 1, "dsl": 2, "oom2ml3": 3, "sk2ml3": 4}

    def stage(c):
        w = _regime(c)[0]
        if w.endswith("-esco"):
            return rank["esco"]
        return rank.get(w.removesuffix("-infer"), rank["short"])

    rep0 = [stage(c) for c in matrix if c["rep"] == 0]
    assert rep0 == sorted(rep0)


@pytest.mark.parametrize(("dist", "n_items", "row_len"), [("uniform", 10, 8), ("uniform", 1000, 5), ("zipf", 30, 20)])
def test_generated_rows_hold_distinct_sorted_items(dist, n_items, row_len):
    rows = generate_rows(dist, 2000, n_items, row_len, seed=1)
    assert rows.shape == (2000, row_len) and rows.dtype == np.int32
    assert (np.diff(rows, axis=1) > 0).all()
    assert rows.min() >= 0 and rows.max() < n_items
    assert np.array_equal(rows, generate_rows(dist, 2000, n_items, row_len, seed=1))


def test_zipf_rows_favour_low_ranks():
    counts = np.bincount(generate_rows("zipf", 5000, 1000, 5, seed=2).ravel(), minlength=1000)
    assert counts[:10].sum() > 10 * counts[-10:].sum()


def test_grid_and_r_match_the_protocol():
    pts = grid()
    assert len(pts) == 6 * 4 + 6 and all(p["n_rows"] == N_ROWS for p in pts)
    assert math.isclose(r_of(N_ROWS, 100, 5), N_ROWS * 10 / (4950 * 15_625))


def _pt(dist, f, length, ratio, n_rows=N_ROWS):
    return {"id": f"{dist}-F{f}-L{length}-N{n_rows}", "dist": dist, "n_rows": n_rows, "n_items": f,
            "row_len": length, "status": "ok", "r": r_of(n_rows, f, length), "ratio": ratio}


def test_r_star_is_the_geometric_mean_at_the_crossing():
    rows = [_pt("uniform", 300, 5, 1.4), _pt("uniform", 1000, 5, 0.7), _pt("uniform", 3000, 5, 0.2)]
    out = crossover(rows)
    assert out["monotone"] and out["uniform_up"] == [(rows[1]["id"], rows[0]["id"])]
    assert math.isclose(out["r_star"], math.sqrt(rows[0]["r"] * rows[1]["r"]))
    assert [p["n_items"] for p in check_points(rows)] == [1000, 300]
    assert all(p["n_rows"] == 4_000_000 for p in check_points(rows))


def test_r_star_takes_the_lower_zipf_crossing_and_flags_disorder():
    rows = [_pt("uniform", 300, 5, 1.4), _pt("uniform", 1000, 5, 0.7), _pt("uniform", 3000, 5, 1.1),
            _pt("zipf", 30000, 20, 0.9), _pt("zipf", 10000, 20, 1.2)]
    out = crossover(rows)
    assert not out["monotone"] and len(out["uniform_up"]) == 1 and len(out["uniform_down"]) == 1
    zipf = math.sqrt(rows[3]["r"] * rows[4]["r"])
    assert zipf < math.sqrt(rows[0]["r"] * rows[1]["r"])
    assert math.isclose(out["r_star"], zipf)


def test_no_crossing_leaves_r_star_unset_and_checks_the_closest_points():
    rows = [_pt("uniform", 300, 5, 0.9), _pt("uniform", 1000, 5, 0.5), _pt("uniform", 3000, 5, 0.95)]
    assert crossover(rows)["r_star"] is None
    assert sorted(p["n_items"] for p in check_points(rows)) == [300, 3000]


def test_o5_calibration_has_the_protocol_configs_in_order():
    matrix = build_o5_calibration_matrix(2)
    assert [c["id"] for c in matrix] == [
        f"{w}-C{n}-{a}#r0" for w in ("oom2ml3", "sk2ml3") for n in (1, 2) for a in ("dense", "esco")
    ]
    assert build_matrix("o5-calibration", 2) == matrix
    assert [c["id"] for c in build_o5_calibration_matrix(1)] == [
        f"{w}-C1-{a}#r0" for w in ("oom2ml3", "sk2ml3") for a in ("dense", "esco")
    ]
    for c in matrix:
        w = c["base_id"].split("-C", 1)[0]
        esco = c["base_id"].endswith("-esco")
        assert (c["dataset"], c["min_support"], c["max_length"]) == WORKLOADS[w]
        assert c.get("sparse_from_k") == (3 if esco else None), c["id"]
        assert bool(c.get("expect_transition")) == esco, c["id"]
        assert c["timeout_s"] == (O5_ESCO_CAP_S[w] if esco else 600), c["id"]
        assert c["rep"] == 0 and c["level_split"] and c["route"] == "C"


def test_o5_calibration_pins_threads_and_devices_and_leaves_the_knobs_at_their_defaults():
    for c in build_o5_calibration_matrix(2):
        assert all(c["env"][t] == "6" for t in THREADS), c["id"]
        assert c["env"]["NCCL_P2P_DISABLE"] == "1"
        assert not any(k in c["env"] for k in KNOBS)
        assert not c.get("use_generator_pruning") and c.get("prune_apriori", True)
        if c["n_gpus"] == 1:
            assert c["env"]["CUDA_VISIBLE_DEVICES"] == "0" and c["env"]["ET_MINER_DISABLE_NCCL"] == "1"
        else:
            assert "CUDA_VISIBLE_DEVICES" not in c["env"] and "ET_MINER_DISABLE_NCCL" not in c["env"]


def test_level_split_times_csr_launches_as_count_csr(monkeypatch):
    import et_miner.gpu.sparse_csr as sparse_csr

    class Chunk:
        per_candidate = False

    def chunk_loop(bitvecs_list, chunks, launch_chunk, *args, **kwargs):
        for chunk in chunks:
            launch_chunk(None, 0, chunk)

    monkeypatch.setattr(sparse_csr, "run_chunked_dense_level", chunk_loop)
    split = LevelSplit()
    split.install()
    try:
        sparse_csr.run_chunked_dense_level([], [Chunk(), Chunk()], lambda bv, did, chunk: time.sleep(0.01))
        split.callback()(3, 2, 1, 100.0)
    finally:
        split.uninstall()
    assert sparse_csr.run_chunked_dense_level is chunk_loop
    assert split.levels[3]["count_csr"] >= 0.02
    assert split.levels[3]["count_tiled"] == split.levels[3]["count_percand"] == 0


@pytest.mark.gpu
def test_copy_bandwidth_is_positive_on_a_device():
    pytest.importorskip("cupy")
    from copy_bandwidth import measure

    assert measure(0, nbytes=1 << 24, reps=2) > 0
