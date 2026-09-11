"""Host-RAM peak of the deferred-frame build in `gpu/row_split.py`.

Separate from `test_row_split_dtypes.py`, whose docstring scopes it to #26 (the
three disagreeing `itemset` dtypes). This file pins a different property of the
same code: the peak host allocation of `_build_deferred_frame`, which is what
decides whether a campaign survives its last level on the `output_dir=None`
route.

The identity the comment in that function states is

    peak = 8 * total_items + 8 * (total_rows + 1)

-- `flat_values` plus `offsets`, both int64, and nothing else. It is asserted
here SYMBOLICALLY, recomputed from the fixture, never as a literal byte count.

THE BOUND IS ABSOLUTE, NOT A RATIO -- and at the production shape, DIFFERENCED.
Measured excess of the real implementation over the identity, one fresh process
per cell, inputs built before the pyarrow warm-up (the order `_excess` uses):

    N = 1,664 items (the real `smoke` lattice)     2,350 B   ratio 1.1245
    N = 100,000                                    2,378 B   ratio 1.0025
    N = 4,000,000                                  2,378 B   ratio 1.0001
    N = 40,000,000                                 2,378 B   ratio 1.0000

Four orders of magnitude in N; the excess does not move, the ratio moves by
12%. It is flat in the chunk count too (2,378 B at 1, 8 and 64 chunks of the
4M shape). So the excess is an additive constant, and a RATIO band measures
the fixture, not the code -- the previous version of this file asserted
`peak < 1.05 * identity`, which the real implementation FAILS at the real
`smoke` lattice shape (1.1245). That band passed only because its fixture was
N = 4M, where a 2.4 KB constant is 0.006%.

`SLACK_B` replaces it, and it is DERIVED, not fitted: the smallest regression
this file exists to catch is one that doubles the identity, and at the
production shape (`smoke`, identity 18,872 B) a doubling's excess IS the
identity, so the slack sits under it at 16,384 B. The real build clears that
by ~7x at every shape above. It is not "an order of magnitude above the
constant" and is not meant to be -- a slack chosen as a multiple of the
constant is fitted to the measurement, and the measurement is what moves.

WHAT THE ABSOLUTE ROW CAN AND CANNOT SEE AT THE SMOKE SHAPE, measured. The
three regressions below cost `max(4N - R, 0)` (the `.astype` on a
concatenate), `R` (a non-in-place cumsum) and `2R` (the `widths` pair) over
the identity, with N = total items and R = 8 * (total_rows + 1) as
`gpu/row_split.py`'s identity comment defines them -- 4N is the byte size of
the int32 concatenation, so these are bytes throughout. At the fixture (4M
items, kbar = 5) they measure 9.6, 6.4 and 12.8 MB: 586x, 390x and 781x the
slack, and `test_slack_rejects_the_known_regressions` holds that. At the
`smoke` lattice (1,664 items, R = 5,560 B) the same three measure 2,724, 8,091
and 14,460 B -- ALL inside 16,384 B, as they were inside the 65,536 B an
earlier version used. Discrimination is a large-N property: at the production
shape no absolute slack that also admits the real build (2,350 B, with ~100 B
of run-to-run jitter) separates an R-scale regression from it by a margin
worth asserting. So the smoke shape is checked by DIFFERENCING instead:
`test_the_excess_does_not_scale_with_n` subtracts the 40M-item excess from the
smoke excess (28 B apart for the real build) and bounds the difference by half
a page, which the constant drops out of and an extra R (5,560 B) does not;
`test_the_smoke_difference_rejects_the_r_scale_regressions` holds that it
discriminates. The absolute smoke row in `test_identity_holds_across_shapes`
is the positive control the ratio band failed; its discrimination lives in the
differenced rows.

THE `.astype` FORM IS BLIND AT LOW KBAR, and at the smoke shape too. With
R = 8N/kbar (to within one element), `4N - R` is at or below zero for
kbar <= 2, so at k = 1 and k = 2 the form allocates nothing beyond the identity
and its peak is the same ~2.4 KB constant as the real build's: measured over
2M items, 2,551 to 2,567 B against the real build's 2,378 B, same process,
either order. At the `smoke` lattice (kbar = 2.3977) the formula predicts
1,096 B of separation, but that peak occurs before pyarrow's constant is
allocated and is smaller than it, so the measured excess is again the
constant: 2,708 to 2,724 B against 2,350 B. No peak bound, absolute or
differenced, separates the astype form at these shapes, and
`test_no_peak_bound_separates_the_astype_form_at_low_kbar` asserts that rather
than hiding it. It is caught at the fixture, where 4N - R is 9.6 MB.

pyarrow's FIRST `LargeListArray.from_arrays` call in a process allocates ~26 MB
of one-time initialisation, which tracemalloc attributes to whatever traced
region happens to run first. Every test here warms it before `reset_peak()`.
Without that warmup this test fails at small N and silently passes at large N
as the constant is amortised away -- flaky in the one direction that looks like
success.

No exact byte count is asserted anywhere below. The constant depends on the
ORDER of that warm-up relative to building the inputs, not on the shape:
warming first and then building measures 2,398 B at the smoke lattice and
2,426 B at the 4M fixture; building first and then warming measures 2,350 B
and 2,378 B -- three fresh processes per cell, 48 B apart at both shapes.
`test_peak_is_flat_plus_offsets_and_nothing_else` warms first; `_excess`
builds first; that is the whole 2,426/2,378 pair an earlier version of this
docstring attributed to call-frame depth. Why the order moves the figure by
48 B is not established here, and nothing below depends on it: the INVARIANCE
in N and in chunk count reproduces under either order, and the bounds are set
by the regressions, not by the constant.
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

SLACK_B = 16_384
"""Absolute headroom over the identity, in bytes.

Derived from the regression class this file exists to catch, not from the
constant: a build that doubles the identity costs the identity itself, which
at the production shape (`smoke`, 18,872 B) is the smallest any in-tree preset
produces, so the slack sits just under it. The real build's ~2.4 KB constant
clears it by ~7x. The three known regressions clear it by 390x-781x at the
fixture shape and are NOT separated by it at the smoke shape (module
docstring), which is why that shape is also checked by differencing.
"""

SMOKE_DIFF_B = 2_048
"""Bound on |excess(smoke lattice) - excess(40M items)|, in bytes.

Half a page: 20x the ~100 B of run-to-run jitter observed in the constant, and
0.37 of the smallest regression the smoke shape can show -- one extra
offsets-sized array, R = 5,560 B. The real build measures 28 B apart.
"""

# The k-histogram of the canonical `smoke` preset's real output: 694 itemsets,
# 1,664 items, kbar = 2.3977. `deferred_itemsets_np` holds ONE ENTRY PER K
# LEVEL, so the production shape is a strongly-peaked lattice with a tiny tail
# -- not the equal-chunks/uniform-k fixture. The old ratio band could not hold
# this case at all; the real build measures 2,350 B over the identity here,
# inside the 16,384 B slack, and the shape is differenced against the 40M
# shape in `test_the_excess_does_not_scale_with_n`.
#
# `test_the_smoke_histogram_is_the_real_preset` re-mines the preset and
# compares. A copied-in "measured" figure with no artifact behind it is the
# defect this whole file is being corrected for; the mine costs ~1.7s, which is
# cheaper than the claim being wrong. `gpu/row_split.py` cites this constant,
# so the check covers that citation too.
SMOKE_K_HIST = {1: 118, 2: 290, 3: 202, 4: 63, 5: 18, 6: 3}


def _lattice(k_hist: dict[int, int]) -> tuple[list[np.ndarray], np.ndarray]:
    arrs = [np.arange(count * k, dtype=np.int32).reshape(count, k) for k, count in sorted(k_hist.items())]
    return arrs, np.ones(sum(a.shape[0] for a in arrs), dtype=np.int64)


def _uniform(n_items: int, k: int, n_chunks: int = N_CHUNKS) -> tuple[list[np.ndarray], np.ndarray]:
    """Equal chunks of uniform k. A harness property, not a code property --
    the identity is linear in `total_items` and `total_rows` separately, and
    `test_identity_holds_across_shapes` varies both, including one real
    lattice."""
    per = (n_items // k) // n_chunks
    arrs = [np.arange(per * k, dtype=np.int32).reshape(per, k) for _ in range(n_chunks)]
    return arrs, np.ones(sum(a.shape[0] for a in arrs), dtype=np.int64)


def _fixture() -> tuple[list[np.ndarray], np.ndarray]:
    return _uniform(N_ITEMS, K)


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


def _excess(fn, arrs, supports) -> int:
    _warm()
    peak, _ = _peak_of(fn, arrs, supports)
    return peak - _identity(arrs)


def test_peak_is_flat_plus_offsets_and_nothing_else():
    """The bound itself. This is the test a reintroduced regression has to get
    past -- `test_slack_rejects_the_known_regressions` measures three local
    reimplementations and never calls `_build_deferred_frame`, so it establishes
    that the slack DISCRIMINATES; it is not what catches a regression landing in
    the real function."""
    _warm()
    arrs, supports = _fixture()
    peak, frame = _peak_of(_build_deferred_frame, arrs, supports)
    expected = _identity(arrs)

    # The frame must also be RIGHT: a build that returns an empty or truncated
    # frame would sail through a memory bound on its own.
    assert frame.height == sum(a.shape[0] for a in arrs)
    assert frame["itemset"].dtype == pl.List(pl.Int64)
    assert frame["itemset"].list.len().sum() == sum(a.size for a in arrs)

    assert peak <= expected + SLACK_B, (
        f"deferred-frame peak {peak:,} B exceeds the identity {expected:,} B by "
        f"{peak - expected:,} B, over the {SLACK_B:,} B slack. The build allocated "
        f"something beyond `flat_values` + `offsets`."
    )


@pytest.mark.parametrize(
    ("label", "build"),
    [
        ("uniform k=5, 4M items", lambda: _uniform(4_000_000, 5)),
        ("uniform k=2, 2M items", lambda: _uniform(2_000_000, 2)),
        ("uniform k=10, 2M items", lambda: _uniform(2_000_000, 10)),
        ("uniform k=1, 400k items", lambda: _uniform(400_000, 1)),
        ("smoke lattice, kbar=2.3977", lambda: _lattice(SMOKE_K_HIST)),
    ],
)
def test_identity_holds_across_shapes(label, build):
    """Vary total_items and total_rows independently, and include one shape the
    code actually produces. The `smoke` lattice case is the one the previous
    ratio band could not hold: the real implementation measures 1.1245x the
    identity there, because 1,664 items make a 2.4 KB constant 12% of the
    total. It clears the absolute slack by ~7x. What that row establishes is
    that the real build PASSES at the production shape; its discrimination
    there is the differenced bound in `test_the_excess_does_not_scale_with_n`,
    since at 1,664 items every known regression is inside `SLACK_B` too."""
    _warm()
    arrs, supports = build()
    peak, _ = _peak_of(_build_deferred_frame, arrs, supports)
    expected = _identity(arrs)
    assert peak <= expected + SLACK_B, (
        f"{label}: peak {peak:,} B over identity {expected:,} B by {peak - expected:,} B "
        f"(slack {SLACK_B:,} B)"
    )


def test_the_excess_does_not_scale_with_n():
    """The invariance that licenses an absolute bound instead of a ratio. A
    400x change in N must not move the excess by more than a page; if it ever
    scales with N, an absolute slack is the wrong mechanism and this file needs
    rewriting rather than a bigger `SLACK_B`.

    Shapes differenced rather than compared to a literal: the constant is
    context-dependent (48 B between the two warm-up orders, module docstring),
    so only a difference is a property of the code.

    The `smoke` lattice is differenced here too, against the 40M shape, and
    that difference is the CHECK on the production shape -- the absolute row
    in `test_identity_holds_across_shapes` only establishes that the real
    build passes there. Real: 2,350 against 2,378, 28 B apart. The bound is
    `SMOKE_DIFF_B`, half a page, because one extra offsets-sized array at this
    shape is R = 5,560 B and must not fit; `_regress_cumsum_copy` measures
    8,091 B here, 5.7 KB over the 40M excess, and
    `test_the_smoke_difference_rejects_the_r_scale_regressions` holds that."""
    small_arrs, small_sup = _uniform(100_000, 5)
    large_arrs, large_sup = _uniform(40_000_000, 5)
    smoke_arrs, smoke_sup = _lattice(SMOKE_K_HIST)

    small = _excess(_build_deferred_frame, small_arrs, small_sup)
    large = _excess(_build_deferred_frame, large_arrs, large_sup)
    smoke = _excess(_build_deferred_frame, smoke_arrs, smoke_sup)

    assert abs(large - small) < 4096, (
        f"the excess moved from {small:,} B to {large:,} B across a 400x change in N "
        f"({abs(large - small):,} B apart). It is supposed to be additive and constant; "
        "if it scales, replace the absolute slack rather than widening it."
    )
    assert abs(large - smoke) < SMOKE_DIFF_B, (
        f"the smoke lattice's excess is {smoke:,} B against {large:,} B at 40M items "
        f"({abs(large - smoke):,} B apart, bound {SMOKE_DIFF_B:,} B). The constant cancels in this "
        "difference; an extra offsets-sized array at this shape is 5,560 B and would not."
    )
    assert large < SLACK_B and small < SLACK_B and smoke < SLACK_B, "premise: all inside the slack"


def test_the_excess_does_not_scale_with_chunk_count():
    """The other free dimension: `deferred_itemsets_np` has one entry per K
    level, so the chunk count is a real variable and not a fixture artifact."""
    one = _excess(_build_deferred_frame, *_uniform(4_000_000, 5, n_chunks=1))
    many = _excess(_build_deferred_frame, *_uniform(4_000_000, 5, n_chunks=64))
    assert abs(many - one) < 4096, f"excess moved {one:,} B -> {many:,} B across 1 vs 64 chunks"


# --------------------------------------------------------------------------
# The three forms this bound replaced, and the one it cannot separate
# --------------------------------------------------------------------------


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


def _regress_astype(arrs, _sup):
    import pyarrow as pa

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


def _regress_cumsum_copy(arrs, _sup):
    import pyarrow as pa

    flat, offsets = _fill(arrs)
    offsets[1:] = np.cumsum(offsets[1:])  # not in-place: full-size temporary
    return pa.LargeListArray.from_arrays(offsets, flat)


def _regress_widths(arrs, _sup):
    import pyarrow as pa

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


def test_slack_rejects_the_known_regressions():
    """The slack is only worth asserting if it would fail on the forms it
    replaced. Each of these builds the SAME frame and is rejected on memory
    alone, so `SLACK_B` cannot be widened without this test noticing what it
    stops catching.

    AT THE FIXTURE SHAPE, which is the scope of the claim: discrimination is a
    large-N property, and the astype form specifically is invisible to any peak
    bound at small N or low kbar -- see
    `test_no_peak_bound_separates_the_astype_form_at_low_kbar`. What this
    establishes is that the slack discriminates, not that a regression in
    `_build_deferred_frame` would be caught here: none of these three call it.
    `test_peak_is_flat_plus_offsets_and_nothing_else` is what does that."""
    arrs, supports = _fixture()
    for name, fn in [
        ("astype-on-concatenate", _regress_astype),
        ("non-in-place cumsum", _regress_cumsum_copy),
        ("widths list + concatenation", _regress_widths),
    ]:
        excess = _excess(fn, arrs, supports)
        assert excess > SLACK_B, (
            f"the {name} regression measured {excess:,} B over the identity, inside the "
            f"{SLACK_B:,} B slack -- the slack no longer discriminates and "
            "test_peak_is_flat_plus_offsets_and_nothing_else is not pinning anything."
        )


def test_the_smoke_difference_rejects_the_r_scale_regressions():
    """The differenced smoke bound is only worth asserting if it fails on the
    forms the absolute slack admits at that shape. The cumsum copy (R) and the
    `widths` pair (2R) measure 8,091 and 14,460 B at the smoke lattice --
    inside `SLACK_B`, asserted as the premise -- and 5.7 and 12 KB over the
    real build's 40M excess, outside `SMOKE_DIFF_B`. The astype form is not in
    this list: it is inside the difference as well (2,724 against 2,350 B),
    the blind spot the module docstring states and
    `test_no_peak_bound_separates_the_astype_form_at_low_kbar` asserts."""
    arrs, supports = _lattice(SMOKE_K_HIST)
    large = _excess(_build_deferred_frame, *_uniform(40_000_000, 5))
    for name, fn in [("non-in-place cumsum", _regress_cumsum_copy), ("widths list + concatenation", _regress_widths)]:
        excess = _excess(fn, arrs, supports)
        assert excess < SLACK_B, f"premise: {name} measures {excess:,} B at the smoke lattice, inside SLACK_B"
        assert excess - large >= SMOKE_DIFF_B, (
            f"{name} at the smoke lattice is {excess:,} B over the identity, only {excess - large:,} B "
            f"over the real build's 40M excess ({large:,} B) -- inside SMOKE_DIFF_B, so the differenced "
            "row no longer discriminates at the production shape."
        )


@pytest.mark.parametrize(
    ("label", "build"),
    [
        ("k=1, 2M items", lambda: _uniform(2_000_000, 1)),
        ("k=2, 2M items", lambda: _uniform(2_000_000, 2)),
        ("smoke lattice, kbar=2.3977", lambda: _lattice(SMOKE_K_HIST)),
    ],
)
def test_no_peak_bound_separates_the_astype_form_at_low_kbar(label, build):
    """The blind spot, asserted so it cannot be quietly reclassified as covered.

    The astype form costs `max(4N - R, 0)` over the identity, which is zero at
    kbar <= 2 and, at the smoke lattice, a predicted 1,096 B that lands under
    the ~2.4 KB constant. At these shapes its peak is the real build's within a
    few hundred bytes -- measured 2,551 to 2,567 B against 2,378 B over 2M
    items and 2,708 to 2,724 B against 2,350 B at the smoke lattice, same
    process, either order. A previous revision justified a bound by claiming
    it "closes the k=2 blind spot"; it does not, no peak-based bound does, and
    this test fails if that ever becomes true so the claim can be revisited on
    evidence."""
    arrs, supports = build()
    real = _excess(_build_deferred_frame, arrs, supports)
    astype = _excess(_regress_astype, arrs, supports)

    assert astype <= real + SMOKE_DIFF_B, (
        f"{label}: the astype form now measures {astype:,} B over the identity against the "
        f"real build's {real:,} B. If that is a real separation, the docstring's "
        "'no peak bound distinguishes these at low kbar' is stale."
    )
    assert astype < SLACK_B, f"{label}: premise -- the astype form is inside the slack here"


@pytest.mark.parametrize("k", [1, 2])
def test_the_other_two_regressions_are_still_caught_at_k2(k):
    """The blind spot above is specific to the astype form. The `widths` pair
    and the cumsum copy scale with R = 8 * rows, not with `4N - R`, so over 2M
    items they are outside the slack at k = 1 and k = 2 alike (16 and 8 MB for
    the copy, twice that for the pair) -- which is why the fix is still worth a
    test at low kbar. The binding variable is N, not kbar: at the smoke
    lattice's 1,664 items the same two forms measure 8,091 and 14,460 B,
    INSIDE the slack, and are caught there only by the differenced bound in
    `test_the_smoke_difference_rejects_the_r_scale_regressions`."""
    arrs, supports = _uniform(2_000_000, k)
    for name, fn in [("non-in-place cumsum", _regress_cumsum_copy), ("widths", _regress_widths)]:
        excess = _excess(fn, arrs, supports)
        assert excess > SLACK_B, f"k={k}: {name} measured only {excess:,} B over the identity"


def test_the_smoke_histogram_is_the_real_preset():
    """`SMOKE_K_HIST` must be what mining the `smoke` preset actually produces.

    CPU tier, at the preset's own `min_support` and with no `max_length`, which
    is how the campaign runs it. If the preset spec ever changes, this fails
    here rather than leaving a stale "measured" histogram in two files -- this
    one and the identity comment in `gpu/row_split.py`, which cites it by name.
    """
    from collections import Counter

    from et_miner.core.apriori import apriori
    from et_miner.synthetic import PRESETS, generate_transactions

    spec = PRESETS["smoke"]
    df, _ = generate_transactions(spec)
    res = apriori(df, min_support=spec.min_support, item_col="items")
    hist = dict(sorted(Counter(len(x) for x in res["itemset"].to_list()).items()))

    assert hist == SMOKE_K_HIST, (
        f"the smoke lattice moved: {hist} != {SMOKE_K_HIST}. Update SMOKE_K_HIST and the "
        "kbar table in gpu/row_split.py's identity comment together -- it cites this constant."
    )
    n_itemsets = sum(SMOKE_K_HIST.values())
    n_items = sum(k * c for k, c in SMOKE_K_HIST.items())
    assert (n_itemsets, n_items) == (694, 1664)
    assert abs(n_items / n_itemsets - 2.3977) < 1e-4, "kbar quoted in row_split.py's table"
