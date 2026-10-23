"""Fixture tests for compare.py's level-at-a-time mode and the incremental signature.

Usage:
    run_py.sh -m pytest tests -q -p no:cacheprovider     (from the scripts directory)

Options: none.
"""

import sys
from pathlib import Path

import polars as pl
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _harness as h  # noqa: E402
import compare  # noqa: E402

N = 100


def tables(rows):
    """Canonical tables from [(itemset, count)]."""
    return h.normalize(pl.DataFrame({"itemset": [r[0] for r in rows], "count": [r[1] for r in rows]},
                                    schema={"itemset": pl.List(pl.Int64), "count": pl.Int64}))


ROWS = [([1], 40), ([3], 30), ([5], 20), ([1, 3], 12), ([1, 5], 9), ([3, 5], 7), ([1, 3, 5], 6)]


def write_ref(rows, d, id_map=None):
    """Write GPU-style per-level tables (frequent_k{K}.parquet, Int32 itemsets, support only)."""
    d.mkdir(parents=True, exist_ok=True)
    for k, t in tables(rows).items():
        items = [[(id_map or {}).get(i, i) for i in it] for it in t["itemset"].to_list()]
        pl.DataFrame({"itemset": items, "support": (t["count"] / N).to_list()},
                     schema={"itemset": pl.List(pl.Int32), "support": pl.Float64}).write_parquet(d / f"frequent_k{k}.parquet")


def run_levels(cpu_rows, ref_rows, tmp, lut=None, id_map=None, ref_min=None):
    h.write_tables(tables(cpu_rows), tmp / "cpu", N)
    write_ref(ref_rows, tmp / "ref", id_map)
    return compare.compare_levels(compare.level_files(tmp / "cpu"), compare.level_files(tmp / "ref"), N, None,
                                  ref_min, lut)


def test_levels_equal_whole_table_path(tmp_path):
    per_k, diffs, sc, sr = run_levels(ROWS, ROWS, tmp_path)
    whole_k, whole_d = compare.compare(h.load_tables(tmp_path / "cpu", N), h.load_tables(tmp_path / "ref", N))
    assert not diffs and not whole_d and per_k == whole_k
    assert sc == sr == h.signature(tables(ROWS))


def test_incremental_signature_equals_signature():
    t = tables(ROWS)
    sig = h.Signature()
    for k in sorted(t):
        sig.update(k, t[k])
    assert sig.result() == h.signature(t)
    with pytest.raises(AssertionError):
        sig.update(1, t[1])


def test_level_on_one_side_only_is_caught(tmp_path):
    _, diffs, sc, sr = run_levels(ROWS, ROWS[:-1], tmp_path)
    assert [(d["k"], d["kind"]) for d in diffs] == [(3, "cpu_only")] and sc != sr


def test_count_off_by_one_is_caught(tmp_path):
    ref = [(i, c + 1) if i == [1, 5] else (i, c) for i, c in ROWS]
    _, diffs, _, _ = run_levels(ROWS, ref, tmp_path)
    assert [(d["kind"], d["cpu"], d["ref"]) for d in diffs] == [("count_mismatch", 9, 10)]


def test_translation_applied_per_level(tmp_path):
    id_map = {1: 1, 3: 5, 5: 3}
    _, diffs, _, _ = run_levels(ROWS, ROWS, tmp_path, lut=id_map, id_map=id_map)
    assert not diffs
    _, diffs, _, _ = run_levels(ROWS, ROWS, tmp_path / "untranslated", id_map=id_map)
    assert diffs


def test_reference_filter_applies_per_level(tmp_path):
    per_k, diffs, _, sr = run_levels([r for r in ROWS if r[1] >= 9], ROWS, tmp_path, ref_min=9)
    assert not diffs and sr["itemsets"] == 5 and 3 not in per_k


def test_two_files_for_one_level_rejected(tmp_path):
    write_ref(ROWS, tmp_path)
    (tmp_path / "frequent_k2.parquet").rename(tmp_path / "a_k2.parquet")
    write_ref(ROWS, tmp_path)
    with pytest.raises(AssertionError):
        compare.level_files(tmp_path)
