"""#26 -- the row-split miner returned three different `itemset` dtypes.

`_apriori_row_split_multi_gpu` has three return paths. Two gave
`List(Int64)`; the PyArrow fast path -- the one that normally runs -- gave
`List(Int32)`, because `col_to_item_arr` is `np.int32` and Arrow preserves it.
The sibling route `_apriori_from_bitvecs` builds the same lookup as int64, so
the two GPU routes disagreed with each other as well: a consumer that
concatenated or joined frames from both got a schema error or a silent
mismatch depending on the polars operation.

The flushed parquet is deliberately NOT widened. That asymmetry is safe only
because of a property of the current readers, so the property is asserted here
rather than described in a docstring.
"""

from __future__ import annotations

import builtins
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from et_miner.core.result import _empty_result


def _gpu_count() -> int:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


@pytest.fixture(scope="module")
def wide_df() -> pl.DataFrame:
    """Enough frequent items that every K level emits, so all levels defer."""
    rng = np.random.default_rng(11)
    rows = []
    for _ in range(4000):
        r = set(rng.choice(16, size=int(rng.integers(4, 9)), replace=False).tolist())
        if 3 in r:
            r.add(2)
        rows.append(sorted(r))
    return pl.DataFrame({"items": rows})


def test_empty_result_is_list_int64():
    assert _empty_result().schema["itemset"] == pl.List(pl.Int64)


@pytest.mark.gpu
class TestReturnPathsAgree:
    """All three paths out of the row-split miner, pinned to one dtype."""

    @staticmethod
    def _mine(df):
        from et_miner import apriori

        return apriori(df, min_support=0.05, item_col="items", use_gpu=True, prune_equal_support=True)

    def test_pyarrow_path_is_list_int64(self, wide_df):
        """CONTROL: without the .astype(np.int64) at row_split.py the concat
        keeps col_to_item_arr's int32 and this is List(Int32)."""
        got = self._mine(wide_df)
        assert got.height > 0, "fixture must reach the deferred path"
        assert got.schema["itemset"] == pl.List(pl.Int64)

    def test_list_fallback_is_list_int64(self, wide_df, monkeypatch):
        """Force the fallback by making `import pyarrow` fail -- not by
        corrupting data, which would change which branch runs for the wrong
        reason. The narrowed `except ImportError` means only this injection
        reaches the fallback; a broken pyarrow now propagates."""
        real_import = builtins.__import__

        def _no_pyarrow(name, *args, **kwargs):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ImportError("injected: no pyarrow")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_pyarrow)
        got = self._mine(wide_df)
        assert got.height > 0
        assert got.schema["itemset"] == pl.List(pl.Int64)

    def test_both_paths_agree_exactly(self, wide_df, monkeypatch):
        """Same lattice, same dtype, same values -- the fallback is a fallback,
        not a second implementation."""
        fast = self._mine(wide_df)

        real_import = builtins.__import__

        def _no_pyarrow(name, *args, **kwargs):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ImportError("injected: no pyarrow")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_pyarrow)
        slow = self._mine(wide_df)

        assert fast.schema == slow.schema
        assert sorted(map(tuple, fast["itemset"].to_list())) == sorted(map(tuple, slow["itemset"].to_list()))


@pytest.mark.gpu
def test_flushed_parquet_stays_int32(wide_df, tmp_path):
    """The asymmetry is a decision, so it is pinned. If a future change widens
    the on-disk dtype, that is a 2x storage change on every artifact under
    runs/ and it should be a deliberate, visible edit -- not a side effect."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from et_miner import apriori

    out = tmp_path / "flushed"
    apriori(
        wide_df,
        min_support=0.05,
        item_col="items",
        use_gpu=True,
        prune_equal_support=True,
        output_dir=str(out),
    )
    files = sorted(out.glob("frequent_k*.parquet"))
    assert files, "expected at least one flushed level"
    field = pq.read_schema(files[0]).field("itemset")
    assert pa.types.is_int32(field.type.value_type), f"on-disk itemset dtype moved: {field.type}"


# ── The boundary enumeration ────────────────────────────────────────────────
#
# WHAT THIS DOES NOT SEE, stated because it is a proximity scan standing in for
# a semantic property:
#
#   - a reader whose `itemset` column access is more than _WINDOW lines away,
#     or reached through a helper that takes the frame as an argument;
#   - a path built at runtime, so the artifact kind is not visible in source;
#   - anything outside src/ (bench/, applications/, a notebook, a consumer repo);
#   - and, most importantly, the actual hazard. The risk is not "someone adds a
#     reader" -- it is "someone JOINS an in-memory frame against a flushed
#     file", which no scan of read call sites can detect. Today no such join
#     exists: core/rules.py is parquet-to-parquet on both sides, the resume
#     reader is dtype-agnostic, and the mine_two_phase anchor read goes through
#     .to_list().
#
# So this is a tripwire, not coverage. It is worth keeping because it makes an
# unlisted new reader visible at review time; it must not be mistaken for proof
# that the asymmetry is safe.
#
# A first draft matched any file that called a parquet reader AND mentioned the
# word "itemset" anywhere. That flagged cli.py (which reads the *input*
# transactions) and two docstring examples in streaming/. Proximity to an
# actual `"itemset"` column reference is the property that matters.

_WINDOW = 15

_KNOWN_ITEMSET_READERS = {
    # file:line                  # why the on-disk int32 is fine there
    "gpu/row_split.py": "resume reader indexes item_to_col[flat_item_ids]; anchor read uses .to_list()",
    "core/rules.py": (
        "parquet-to-parquet on both join sides (generate_rules_drop1 and "
        "compute_self_sufficiency both scan_parquet + join on the k-1 cols); "
        "widening costs ~84 GB at K=7 -- an estimate resting on a ~3.5e9-row "
        "K=6 frame, which is not recorded anywhere, not a measurement"
    ),
}


def test_no_unlisted_itemset_reader():
    src = Path(__file__).resolve().parents[1] / "src" / "et_miner"
    reader = re.compile(r"(scan_parquet|read_parquet|read_table|ParquetFile)\s*\(")
    # A column reference, not the English word: "itemset" in quotes or as an
    # attribute/key. This is what excludes prose, help strings and log lines.
    column = re.compile(r"""["']itemset["']""")

    found = set()
    for path in src.rglob("*.py"):
        lines = path.read_text().splitlines()
        hits = [i for i, ln in enumerate(lines) if reader.search(ln) and not ln.lstrip().startswith((">>>", "#"))]
        if not hits:
            continue
        for i in hits:
            window = lines[max(0, i - _WINDOW) : i + _WINDOW + 1]
            if any(column.search(w) for w in window):
                found.add(str(path.relative_to(src)))
                break

    unlisted = found - set(_KNOWN_ITEMSET_READERS)
    assert not unlisted, (
        f"new parquet reader(s) touching the `itemset` column: {sorted(unlisted)}. "
        "Decide the inner dtype at that boundary and add it to "
        "_KNOWN_ITEMSET_READERS with the reason. On-disk is large_list<int32>; "
        "in-memory returns are List(Int64)."
    )
