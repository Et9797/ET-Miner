"""The route consolidation's harness: the matrices and the mechanical decision rule (no GPU)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))

routes_matrix = pytest.importorskip("routes_matrix")
# By path, under its own name: bench/moonshots, bench/optimizations and bench/pruning have a decide.py too.
_spec = importlib.util.spec_from_file_location("routes_decide", BENCH / "routes" / "decide.py")
decide = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(decide)


@pytest.mark.parametrize("mode", ["routes-cpu", "routes", "routes-census"])
def test_matrix_ids_are_unique_and_every_arm_is_pinned(mode):
    from et_miner import _env

    configs = routes_matrix.build_routes_matrix(mode, 2)
    ids = [c["id"] for c in configs]
    assert len(ids) == len(set(ids))
    for c in configs:
        assert c["env"]["NCCL_P2P_DISABLE"] == "1"
        assert all(c["env"][k] == routes_matrix.THREADS for k in routes_matrix.POOLS)
        if c["route"] in ("C", "C-bitvecs", "E") and c["n_gpus"] == 1:
            assert c["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        if (v := c["env"].get("ET_MINER_LEAVE_OUT")) is not None:
            assert set(v.split(",")) <= set(_env.LEAVE_OUT)
        if c["route"] in ("D", "E"):
            assert c["expect_chunks"] == 4
        assert c["id"].startswith(c["regime"] + "-")
        assert decide._ID.match(c["id"]), c["id"]


def test_one_gpu_only_members_have_no_two_gpu_arm_or_lone_default():
    ids = {c["base_id"] for c in routes_matrix.build_routes_matrix("routes", 2)}
    assert not [i for i in ids if "2-no-k2fused" in i or "2-no-fused" in i]
    assert "sk2ml2-bv-C1-no-k2fused" in ids and "sk2ml3-bv-C1-no-fused" in ids
    assert "sk2ml2-bv-C2-default" not in ids and "sk2ml3-bv-C2-default" in ids


def test_a_failed_rep_0_is_never_repeated():
    cfg = {"mode": "routes", "base_id": "sk2ml3-bv-C1-no-tiled", "rep": 1}
    rows = [{"id": "sk2ml3-bv-C1-no-tiled#r0", "status": "timeout"}]
    assert routes_matrix.rep0_failed_skip(cfg, rows) == "skipped: rep 0 timeout"
    rows.append({"id": "sk2ml3-bv-C1-no-tiled#r0", "status": "ok"})
    assert routes_matrix.rep0_failed_skip(cfg, rows) is None
    assert routes_matrix.rep0_failed_skip({**cfg, "rep": 0}, rows[:1]) is None


def test_reps_are_rep_major_with_the_sub_second_reps_last():
    configs = routes_matrix.build_routes_matrix("routes", 2)
    reps = [c["rep"] for c in configs]
    first_rep1 = reps.index(1)
    assert set(reps[:first_rep1]) == {0}
    tail = [c for c in configs[first_rep1:] if c["workload"] in routes_matrix.SUBSECOND and c["route"].startswith("C")]
    assert tail and configs[-len(tail):] == tail


def _row(id_, wall, status="ok", sig=(10, 100, "h"), vram=100, rss=100):
    return {"id": id_, "status": status, "wall_s": wall, "peak_vram_mb": {"0": vram}, "peak_rss_mb": rss,
            "n_itemsets": sig[0], "sum_counts": sig[1], "itemset_hash": sig[2],
            "config": {"dataset": "d", "min_support": 0.1, "max_length": None}}


def _write(tmp_path, rows):
    p = tmp_path / "raw.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(p)


def test_rule_2_needs_ten_percent_disjoint_ranges_and_a_second():
    a, b = decide.Cell(), decide.Cell()
    a.walls, b.walls = [10.0, 10.1, 10.2], [12.0, 12.1, 12.2]
    assert decide.wins(a, b) and not decide.wins(b, a)
    b.walls = [10.9, 11.0, 11.1]
    assert not decide.wins(a, b)  # ranges apart and 0.9 s saved, but under 10 %
    b.walls = [10.15, 11.5, 12.0]
    assert not decide.wins(a, b)  # ranges overlap
    a.walls, b.walls = [1.0] * 3, [1.5] * 3
    assert not decide.wins(a, b)  # under 1 s saved


def test_leave_one_out_keeps_a_member_the_full_dispatch_needs_and_drops_a_tie(tmp_path):
    rows = []
    for rep in range(3):
        rows += [_row(f"r1-C1-default#r{rep}", 10 + rep * 0.1), _row(f"r1-C1-no-tiled#r{rep}", 20 + rep * 0.1),
                 _row(f"r1-C1-no-group#r{rep}", 10.05 + rep * 0.1)]
    table, must_fix = decide.cells(decide._load([_write(tmp_path, rows)]))
    assert not must_fix
    by = {e["member"]: e["decision"] for e in decide.leave_one_out(table, "G5")}
    assert by["tiled"].startswith("stays")
    assert by["group"].startswith("goes") and "neither wins" in by["group"]


def test_a_failure_without_the_member_blocks_its_removal(tmp_path):
    rows = [_row(f"r1-C1-default#r{rep}", 10.0) for rep in range(3)]
    rows += [_row("r1-C1-no-fused#r0", 0, status="error: OutOfMemoryError")]
    table, _ = decide.cells(decide._load([_write(tmp_path, rows)]))
    fused = next(e for e in decide.leave_one_out(table, "G5") if e["member"] == "fused")
    assert fused["decision"].startswith("stays")


def test_a_signature_mismatch_is_a_must_fix(tmp_path):
    rows = [_row("r1-C1-default#r0", 1.0), _row("r1-C1-keep#r0", 1.0, sig=(11, 100, "x"))]
    _, must_fix = decide.cells(decide._load([_write(tmp_path, rows)]))
    assert must_fix and "signature mismatch" in must_fix[0]


def test_head_to_head_names_the_winning_path(tmp_path):
    rows = []
    for rep in range(3):
        rows += [_row(f"dsl-C1-default#r{rep}", 3 + rep * 0.01), _row(f"dsl-C1-host#r{rep}", 25 + rep * 0.01),
                 _row(f"dsl-C1-rebuilt#r{rep}", 15 + rep * 0.01)]
    table, _ = decide.cells(decide._load([_write(tmp_path, rows)]))
    (g1,) = decide.head_to_head(table, "G1")
    assert g1["regimes"][0]["winner"] == "device"
    assert g1["decisions"]["host"].startswith("wins no regime")


def test_head_to_head_compares_arms_at_one_gpu_count(tmp_path):
    rows = []
    for rep in range(3):
        rows += [_row(f"dsl-C1-default#r{rep}", 10 + rep * 0.01), _row(f"dsl-C2-default#r{rep}", 2 + rep * 0.01),
                 _row(f"dsl-C1-host#r{rep}", 9.5 + rep * 0.01)]
    table, _ = decide.cells(decide._load([_write(tmp_path, rows)]))
    (g1,) = decide.head_to_head(table, "G1")
    assert [r["regime"] for r in g1["regimes"]] == ["dsl"]
    assert g1["regimes"][0]["arms"]["device"].startswith("C1-default")
    assert g1["regimes"][0]["winner"] is None


def test_rule_7_blocks_a_losing_path_whose_removal_raises_peak_memory(tmp_path):
    rows = []
    for rep in range(3):
        rows += [_row(f"r1-C1-default#r{rep}", 3 + rep * 0.01, vram=9000),
                 _row(f"r1-C1-esco-auto#r{rep}", 8 + rep * 0.01, vram=4000)]
    table, _ = decide.cells(decide._load([_write(tmp_path, rows)]))
    (g6,) = decide.head_to_head(table, "G6")
    assert "BLOCKED by rule 7" in g6["decisions"]["esco-auto"] and "VRAM 2.25x" in g6["decisions"]["esco-auto"]
    assert g6["decisions"]["dense"].startswith("wins r1")


def test_a_cpu_path_is_not_guarded_on_vram_and_cpu_son_stays(tmp_path):
    rows = []
    for rep in range(3):
        rows += [_row(f"r1-son-D1-default#r{rep}", 30 + rep * 0.01, vram=4),
                 _row(f"r1-son-D1-ported#r{rep}", 3 + rep * 0.01, vram=1200),
                 _row(f"r1-son-D1-boolean#r{rep}", 20 + rep * 0.01, vram=600)]
    table, _ = decide.cells(decide._load([_write(tmp_path, rows)]))
    (s1,) = decide.head_to_head(table, "S1")
    assert "rule 9" in s1["decisions"]["cpu"]
    assert "BLOCKED by rule 7" in s1["decisions"]["boolean"] and "VRAM 2.00x" in s1["decisions"]["boolean"]
