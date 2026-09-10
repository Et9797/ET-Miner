"""Host-RAM peak of the deferred-frame build in `gpu/row_split.py`.

Separate from `test_row_split_dtypes.py`, whose docstring scopes it to #26 (the
three disagreeing `itemset` dtypes). This file pins a different property of the
same code: the peak host allocation of `_build_deferred_frame`, which is what
decides whether a campaign survives its last level on the `output_dir=None`
route.

The identity the comment in that function states is

    peak = 8 * total_items + 8 * (total_rows + 1)

-- `flat_values` plus `offsets`, both int64, and nothing else. It is asserted
here SYMBOLICALLY, recomputed from the fixture, never as a literal byte count:
a literal would have to be re-derived by hand every time the fixture changed,
and the previous version of that comment was wrong precisely because its
figures were bound to one fixture shape.

Two measured facts license the way this is written:

* **The non-NumPy share is a shape-independent CONSTANT, ~2.4 KB.** Measured
  across five shapes (4M-40M items, k = 2/5/10) the all-domain peak exceeds
  the identity by 2,426 B once and 2,378 B in the other four -- it does not
  scale with items or with rows. So `get_traced_memory()` is used with NO
  domain filter: separating `np.lib.tracemalloc_domain` would remove a
  constant two thousandths of a percent and add a NumPy-internal constant to
  the test's API surface. Note it is a constant and not a percentage; quoting
  it as a percentage would make it a property of the fixture again.
* **A 5% band both holds and discriminates.** The real implementation measures
  ratio 1.0000. The three regressions this is here to catch measure 1.250
  (`.astype` on a concatenate), 1.167 (a non-in-place cumsum) and 1.333
  (rebuilding the `widths` pair) -- all far outside the band, and
  `test_band_rejects_the_known_regressions` keeps that discrimination checked
  rather than merely claimed.

pyarrow's FIRST `LargeListArray.from_arrays` call in a process allocates ~26 MB
of one-time initialisation, which tracemalloc attributes to whatever traced
region happens to run first. Every test here warms it before `reset_peak()`.
Without that warmup this test fails at small N and silently passes at large N
as the constant is amortised away -- flaky in the one direction that looks like
success.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import polars as pl
import pytest

pytest.importorskip("pyarrow")

from et_miner.gpu.row_split import _build_deferred_frame  # noqa: E402

K = 5
N_CHUNKS = 8
N_ITEMS = 4_000_000
BAND = 1.05


def _fixture() -> tuple[list[np.ndarray], np.ndarray]:
    """Equal chunks of uniform k. A harness property, not a code property --
    the identity is linear in `total_items` and `total_rows` separately, and
    `test_identity_holds_across_shapes` varies both."""
    rows = N_ITEMS // K
    per = rows // N_CHUNKS
    arrs = [np.arange(per * K, dtype=np.int32).reshape(per, K) for _ in range(N_CHUNKS)]
    return arrs, np.ones(sum(a.shape[0] for a in arrs), dtype=np.int64)


def _identity(arrs: list[np.ndarray]) -> int:
    total_rows = sum(a.shape[0] for a in arrs)
    total_items = sum(a.size for a in arrs)
    return 8 * total_items + 8 * (total_rows + 1)


def _warm() -> None:
    _build_deferred_frame([np.array([[1, 2]], dtype=np.int32)], np.ones(1, dtype=np.int64))


def _peak_of(fn, *args) -> tuple[int, object]:
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        out = fn(*args)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak, out


def test_peak_is_flat_plus_offsets_and_nothing_else():
    _warm()
    arrs, supports = _fixture()
    peak, frame = _peak_of(_build_deferred_frame, arrs, supports)
    expected = _identity(arrs)

    # The frame must also be RIGHT: a build that returns an empty or truncated
    # frame would sail through a memory bound on its own.
    assert frame.height == sum(a.shape[0] for a in arrs)
    assert frame["itemset"].dtype == pl.List(pl.Int64)
    assert frame["itemset"].list.len().sum() == sum(a.size for a in arrs)

    assert peak < BAND * expected, (
        f"deferred-frame peak {peak:,} B exceeds {BAND}x the identity "
        f"{expected:,} B (ratio {peak / expected:.3f}). The build allocated "
        f"something beyond `flat_values` + `offsets`."
    )


@pytest.mark.parametrize(("n_items", "k"), [(4_000_000, 5), (2_000_000, 2), (2_000_000, 10)])
def test_identity_holds_across_shapes(n_items, k):
    """Vary total_items and total_rows independently -- the fixture's equal
    chunks and uniform k are the harness's, and a bound that only held at one
    shape is the defect this file's identity replaced."""
    _warm()
    rows = n_items // k
    per = rows // N_CHUNKS
    arrs = [np.arange(per * k, dtype=np.int32).reshape(per, k) for _ in range(N_CHUNKS)]
    supports = np.ones(sum(a.shape[0] for a in arrs), dtype=np.int64)
    peak, _ = _peak_of(_build_deferred_frame, arrs, supports)
    expected = _identity(arrs)
    assert peak < BAND * expected, f"k={k}: {peak:,} B vs {expected:,} B (ratio {peak / expected:.3f})"


def test_band_rejects_the_known_regressions():
    """The band is only worth asserting if it would fail on the forms it
    replaced. Each of these builds the SAME frame and is rejected on memory
    alone -- so a future edit reintroducing one cannot land green, and the 1.05
    band cannot be widened without this test noticing what it stops catching."""
    import pyarrow as pa

    def regress_astype(arrs, _sup):
        total_rows = sum(a.shape[0] for a in arrs)
        flat = np.concatenate([a.ravel() for a in arrs]).astype(np.int64)
        offsets = np.empty(total_rows + 1, dtype=np.int64)
        offsets[0] = 0
        pos = 1
        for a in arrs:
            nr, k = a.shape
            offsets[pos : pos + nr] = k
            pos += nr
        np.cumsum(offsets[1:], out=offsets[1:])
        return pa.LargeListArray.from_arrays(offsets, flat)

    def regress_cumsum_copy(arrs, _sup):
        flat, offsets = _fill(arrs)
        offsets[1:] = np.cumsum(offsets[1:])  # not in-place: full-size temporary
        return pa.LargeListArray.from_arrays(offsets, flat)

    def regress_widths(arrs, _sup):
        total_items = sum(a.size for a in arrs)
        flat = np.empty(total_items, dtype=np.int64)
        pos = 0
        widths = []
        for a in arrs:
            nr, k = a.shape
            flat[pos : pos + a.size] = a.ravel()
            pos += a.size
            widths.append(np.full(nr, k, dtype=np.int64))
        joined = np.concatenate(widths)  # the per-chunk list AND its concatenation
        offsets = np.empty(joined.size + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(joined, out=offsets[1:])
        return pa.LargeListArray.from_arrays(offsets, flat)

    def _fill(arrs):
        total_rows = sum(a.shape[0] for a in arrs)
        total_items = sum(a.size for a in arrs)
        flat = np.empty(total_items, dtype=np.int64)
        offsets = np.empty(total_rows + 1, dtype=np.int64)
        offsets[0] = 0
        ip, rp = 0, 1
        for a in arrs:
            nr, k = a.shape
            flat[ip : ip + a.size] = a.ravel()
            ip += a.size
            offsets[rp : rp + nr] = k
            rp += nr
        return flat, offsets

    _warm()
    arrs, supports = _fixture()
    expected = _identity(arrs)
    for name, fn in [
        ("astype-on-concatenate", regress_astype),
        ("non-in-place cumsum", regress_cumsum_copy),
        ("widths list + concatenation", regress_widths),
    ]:
        peak, _ = _peak_of(fn, arrs, supports)
        assert peak >= BAND * expected, (
            f"the {name} regression measured {peak / expected:.3f}x the identity, "
            f"inside the {BAND}x band -- the band no longer discriminates and "
            f"test_peak_is_flat_plus_offsets_and_nothing_else is not pinning anything."
        )
