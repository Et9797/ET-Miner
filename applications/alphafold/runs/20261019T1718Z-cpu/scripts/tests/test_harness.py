"""Fixture tests for the CPU reproduction harness.

Usage:
    run_py.sh -m pytest tests -q -p no:cacheprovider     (from the scripts directory)

Options: none.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import _harness as h  # noqa: E402
import compare  # noqa: E402
import null_model  # noqa: E402

N_MINED = 76_890_945
GPU_NULL5 = (HERE.parent.parent / "20260902T0000Z/phase3/2026_01/exp/experiment_null_model_20260902_070216.json")


def fixture_rows(n=400, n_items=12, seed=7):
    """Random transactions with planted co-occurrence."""
    rnd = random.Random(seed)
    rows = []
    for _ in range(n):
        r = {i for i in range(n_items) if rnd.random() < 0.25}
        if rnd.random() < 0.3:
            r |= {1, 3, 5, 7}
        rows.append(sorted(r) or [0])
    return pl.DataFrame({"items": rows}, schema={"items": pl.List(pl.Int32)})


def table(rows):
    """Canonical tables from [(itemset, count)]."""
    return h.normalize(pl.DataFrame({"itemset": [r[0] for r in rows], "count": [r[1] for r in rows]},
                                    schema={"itemset": pl.List(pl.Int64), "count": pl.Int64}))


def test_campaign_min_counts():
    want = {1e-3: 76_891, 1e-4: 7_690, 1e-5: 769, 1e-6: 77, 2e-7: 16, 1e-7: 8}
    for s, m in want.items():
        assert h.min_support_for(m, N_MINED, s) == s
    assert h.min_support_for(4, N_MINED) == (4 - 0.5) / N_MINED
    assert h.min_support_for(77, N_MINED, 1e-5) == (77 - 0.5) / N_MINED


def test_counts_recovered_from_supports():
    counts = np.array([8, 9, 769, 76_891, 12_345_678, N_MINED], dtype=np.int64)
    assert (h.counts_from_support(counts / N_MINED, N_MINED) == counts).all()
    with pytest.raises(AssertionError):
        h.counts_from_support(np.array([8.4 / N_MINED]), N_MINED)


def test_item_order_normalized():
    t = table([([5, 1, 3], 4), ([3, 1], 6), ([1, 3, 2], 5)])
    assert t[3]["itemset"].to_list() == [[1, 2, 3], [1, 3, 5]]
    assert h.signature(t) == h.signature(table([([1, 3, 5], 4), ([1, 3], 6), ([2, 3, 1], 5)]))
    with pytest.raises(AssertionError):
        table([([1, 1, 2], 3)])


def test_int32_reference_equals_int64_cpu(tmp_path):
    ref = pl.DataFrame({"itemset": [[3, 1], [7]], "support": [5 / 100, 9 / 100]},
                       schema={"itemset": pl.List(pl.Int32), "support": pl.Float64})
    ref.write_parquet(tmp_path / "frequent_k1.parquet")
    cpu = table([([1, 3], 5), ([7], 9)])
    per_k, diffs = compare.compare(cpu, h.load_tables(tmp_path, 100))
    assert not diffs and per_k[2]["both"] == 1


def test_planted_difference_caught_and_adjudicated():
    df = fixture_rows()
    dc = h.DirectCounter(df, range(12))
    truth = {}
    for its in ([1, 3], [1, 3, 5], [1, 3, 5, 7], [2, 4]):
        truth[tuple(its)] = dc.count(its)
    rows = [(list(k), v) for k, v in truth.items()]
    good = table(rows)
    assert compare.compare(good, table(rows))[1] == []
    bad_count = table([(k, v + (1 if k == [1, 3, 5] else 0)) for k, v in rows])
    missing = table([r for r in rows if r[0] != [1, 3, 5, 7]])
    for planted, kind in ((bad_count, "count_mismatch"), (missing, "ref_only")):
        per_k, diffs = compare.compare(planted, good)
        assert [d["kind"] for d in diffs] == [kind]
        assert h.signature(planted)["sha256"] != h.signature(good)["sha256"]
    tmp = Path(pytest.importorskip("tempfile").mkdtemp()) / "fx.parquet"
    df.write_parquet(tmp)
    diffs = compare.adjudicate(compare.compare(bad_count, good)[1], str(tmp), None, 1)
    assert diffs[0]["wrong"] == ["cpu"] and diffs[0]["direct"] == truth[(1, 3, 5)]


def test_direct_counter_matches_brute_force():
    df = fixture_rows(n=300)
    dc = h.DirectCounter(df, range(12))
    rows = [set(r) for r in df["items"].to_list()]
    for its in ([1], [1, 3], [2, 5, 9], [1, 3, 5, 7], [11, 0]):
        assert dc.count(its) == sum(set(its) <= r for r in rows)


def test_null_construction_equals_explode_shuffle_groupby():
    df = fixture_rows(n=500)
    flat, offsets = null_model.flat_and_offsets(df["items"])
    np.random.default_rng(123).shuffle(flat)
    ours = null_model.rebuild(flat, offsets)
    ex = df.with_row_index("row").explode("items")
    ex = ex.with_columns(items=pl.Series(flat))
    ref = ex.group_by("row", maintain_order=True).agg(pl.col("items").unique().sort()).sort("row")
    assert ours.height == df.height
    assert ours["items"].cast(pl.List(pl.Int64)).to_list() == ref["items"].cast(pl.List(pl.Int64)).to_list()
    assert ours["items"].list.len().sum() <= len(flat)


def test_engine_matches_oracle_on_fixture():
    from efficient_apriori import apriori as ea_apriori

    from et_miner import apriori

    df = fixture_rows()
    m = 20
    res = apriori(df, min_support=h.min_support_for(m, df.height), max_length=None, use_gpu=False, n_jobs=2)
    cpu = h.normalize(res, df.height)
    rows = [tuple(r) for r in df["items"].to_list()]
    sets, _ = ea_apriori(rows, min_support=(m - 0.5) / df.height, min_confidence=1.0, max_length=12)
    ref = table([(list(t), c) for lvl in sets.values() for t, c in lvl.items()])
    assert compare.compare(cpu, ref)[1] == []
    assert h.signature(cpu)["itemsets"] > 50


@pytest.mark.skipif(not GPU_NULL5.exists(), reason="GPU null JSON not present")
def test_null_seeds_match_gpu_run():
    gpu = json.loads(GPU_NULL5.read_text())["parameters"]["permutation_seeds"]
    assert null_model.permutation_seeds(42, len(gpu)) == [int(x) for x in gpu]
    assert null_model.permutation_seeds(42, 100)[:5] == null_model.permutation_seeds(42, 5)


def _null_fixture(tmp_path, totals, kmax, real_k6=22, gpu_shift=0):
    """Write a fake CPU null dir, real meta and GPU null JSON."""
    for i, (t, km) in enumerate(zip(totals, kmax, strict=True)):
        h.dump_json({"total_itemsets": t, "k_max": km, "k_distribution": {str(k): 1 for k in range(1, km + 1)}},
                    tmp_path / "null" / f"perm_{i:03d}" / "meta.json")
    h.dump_json({"signature": {"k_distribution": {"4": 10, "5": 10, "6": real_k6}}}, tmp_path / "real.json")
    rnd = random.Random(5)
    runs = [{"total_itemsets": 171306 + gpu_shift + rnd.randint(-100, 100)} for _ in range(100)]
    h.dump_json({"null_runs": runs}, tmp_path / "gpu.json")


def _verdict(tmp_path, mode, monkeypatch):
    import null_verdict

    out = tmp_path / f"v_{mode}.json"
    monkeypatch.setattr(sys, "argv", ["nv", str(tmp_path / "null"), str(tmp_path / "real.json"),
                                      str(tmp_path / "gpu.json"), "--mode", mode, "--out", str(out)])
    null_verdict.main()
    return json.loads(out.read_text())


def test_null_verdict_criterion(tmp_path, monkeypatch):
    _null_fixture(tmp_path, [171300, 171310, 171290, 171320, 171305], [6] * 5)
    assert _verdict(tmp_path, "n5", monkeypatch)["verdict"] == "consistent"
    _null_fixture(tmp_path, [171300, 171310, 171290, 171320, 171305], [6, 6, 7, 6, 6])
    v = _verdict(tmp_path, "n5", monkeypatch)
    assert v["verdict"] == "inconsistent" and not v["criteria"]["a_kmax_all_6"]
    _null_fixture(tmp_path, [172300] * 5, [6] * 5)
    assert not _verdict(tmp_path, "n5", monkeypatch)["criteria"]["b_mean_within_3sigma_sqrt5"]
    _null_fixture(tmp_path, [171300] * 5, [6] * 5, real_k6=1)
    assert not _verdict(tmp_path, "n5", monkeypatch)["criteria"]["d_no_perm_reaches_real_at_k456_and_none_k_ge_7"]


def test_unsorted_engine_output_is_rejected_without_sorting():
    df = pl.DataFrame({"itemset": [[3, 1], [1, 2]], "count": [4, 5]},
                      schema={"itemset": pl.List(pl.Int64), "count": pl.Int64})
    with pytest.raises(AssertionError):
        h.normalize(df, sort_items=False)
    assert h.normalize(df)[2]["itemset"].to_list() == [[1, 2], [1, 3]]
