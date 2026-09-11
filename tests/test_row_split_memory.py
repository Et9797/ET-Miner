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
4M shape). So the excess is an additive constant, C, and a RATIO band measures
the fixture, not the code -- the previous version of this file asserted
`peak < 1.05 * identity`, which the real implementation FAILS at the real
`smoke` lattice shape. That band passed only because its fixture was N = 4M,
where a 2.4 KB constant is 0.006%. The two invariances are what the tests hold
(`test_the_excess_does_not_scale_with_n`,
`test_the_excess_does_not_scale_with_chunk_count`); C's value is asserted
nowhere, and the last paragraph below says why it cannot be.

`SLACK_B` replaces the band, and it is DERIVED, not fitted: the smallest
regression this file exists to catch is one that doubles the identity, and at
the production shape (`smoke`, identity 18,872 B) a doubling's excess IS the
identity. The slack sits under it, at 16,384 B, so a doubling is rejected
there for ANY constant C >= 0 -- 18,872 + C > 16,384 needs no measurement --
and `test_the_slack_rejects_a_doubling_at_the_smoke_shape` holds both that
inequality and a doubling form measured through the same apparatus. It is not
"an order of magnitude above the constant" and is not meant to be -- a slack
chosen as a multiple of the constant is fitted to the measurement, and the
measurement is what moves. Why 16,384 and not something tighter: headroom
over the constant, ~7x. A tighter slack would reject a doubling too, and
would also separate the two R-scale forms below from the real build at the
smoke shape; this file separates them by differencing instead, and keeps the
slack a statement about the doubling class alone.

WHAT THE ABSOLUTE ROW CAN AND CANNOT SEE AT THE SMOKE SHAPE. The three
regressions below allocate `max(4N - R, 0)` (the `.astype` on a concatenate),
`R` (a non-in-place cumsum) and `2R` (the `widths` pair) beyond the identity,
with N = total items and R = 8 * (total_rows + 1) as `gpu/row_split.py`'s
identity comment defines them -- 4N is the byte size of the int32
concatenation, so these are bytes throughout, and they follow from the code,
not from a measurement. At the fixture (4M items, kbar = 5) they are 9.6, 6.4
and 12.8 MB: 586x, 390x and 781x the slack, and
`test_slack_rejects_the_known_regressions` holds that. At the `smoke` lattice
(1,664 items, 694 rows, R = 5,560 B) they are 1,096, 5,560 and 11,120 B --
ALL inside 16,384 B, with C on top or not, as they were inside the 65,536 B an
earlier version used. So the smoke shape is checked by DIFFERENCING:
`test_the_excess_does_not_scale_with_n` subtracts the 40M-item excess from the
smoke excess and bounds the difference by `SMOKE_DIFF_B`, half a page, which
C drops out of and an extra R (over twice that bound) does not;
`test_the_smoke_difference_rejects_the_r_scale_regressions` holds that the
cumsum and `widths` forms clear the real build by at least that bound at the
smoke shape. Their margin over the real build is NOT their term: the real
build's peak carries C and the cumsum form's does not (its temporary is freed
before the arrow call), so the margin is at least R - 8 - C there, and
`SMOKE_DIFF_B`'s docstring states the interval of C on which the bound is
valid. The absolute smoke row in `test_identity_holds_across_shapes` is the
positive control the ratio band failed; its discrimination lives in the
differenced rows.

THE `.astype` FORM IS BLIND AT LOW KBAR, and at the smoke shape too. With
R = 8N/kbar (to within one element), `4N - R` is at or below zero for
kbar <= 2, so at k = 1 and k = 2 the form allocates nothing beyond the identity
at its own peak; at the `smoke` lattice (kbar = 2.3977) its term is 1,096 B,
under C. Measured in this file's own run order, the form's excess is BELOW the
real build's at all three shapes -- by an amount that depends on the process
and on the position in the run, which is why no figure is written here: the
revision before this one quoted one, measured one fresh process per cell,
with the wrong sign. `test_no_peak_bound_separates_the_astype_form_at_low_kbar`
asserts the relation `astype <= real` and nothing numeric. The form is caught
at the fixture, where 4N - R is 9.6 MB.

pyarrow's FIRST `LargeListArray.from_arrays` call in a process allocates ~26 MB
of one-time initialisation, which tracemalloc attributes to whatever traced
region happens to run first. Every test here warms it before `reset_peak()`.
Without that warmup this test fails at small N and silently passes at large N
as the constant is amortised away -- flaky in the one direction that looks like
success. The first traced call of any of the forms below carries a smaller
cost of the same kind -- its own, and not its allocation -- so `_excess`
calls the function it is about to measure once on a one-row input first, and
that cost is paid outside the measured region for the real build and the
regression forms alike.

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
produces, so the slack sits just under it and rejects a doubling there for any
constant C >= 0. `test_the_slack_rejects_a_doubling_at_the_smoke_shape` holds
that inequality and a measured doubling. The real build's ~2.4 KB constant
clears the slack by ~7x. The three known regressions clear it by 390x-781x at
the fixture shape and are NOT separated by it at the smoke shape (module
docstring), which is why that shape is also checked by differencing.
"""

SMOKE_DIFF_B = 2_048
"""The differenced bound at the smoke shape, in bytes -- half a page.

Two rows hold it, and it means one thing in both: how far apart two excesses
measured at, or against, the smoke lattice may be.

`test_the_excess_does_not_scale_with_n` holds the real build's smoke excess
within it of the real build's 40M excess. That is the check on the production
shape: the constant C cancels in that difference, and an extra offsets-sized
array there (R = 5,560 B) is over twice the bound and cannot fit.

`test_the_smoke_difference_rejects_the_r_scale_regressions` holds the cumsum
and `widths` forms at least this far ABOVE the real build at the smoke shape,
so the bound cannot be widened past the smallest regression the differencing
exists to catch. Against the real build at the smoke shape, not against a
second 40M build: the first row already holds the two real-build excesses
within this bound of each other, so a form this row puts `SMOKE_DIFF_B` over
the smoke real build is, against the 40M reference the first row uses, over by
that margin shifted by a gap the first row bounds and does not pin. C does NOT
cancel in this second row -- the real build's peak carries it and the cumsum
form's does not (module docstring) -- so the cumsum form clears the real build
by at least R - 8 - C, and the row is valid while C < R - 8 - SMOKE_DIFF_B,
3,504 B at this shape. C is ~2.4 KB in every context measured, and the row's
failure message prints both figures.
"""

# The k-histogram of the canonical `smoke` preset's real output: 694 itemsets,
# 1,664 items, kbar = 2.3977. `deferred_itemsets_np` holds ONE ENTRY PER K
# LEVEL, so the production shape is a strongly-peaked lattice with a tiny tail
# -- not the equal-chunks/uniform-k fixture. The old ratio band could not hold
# this case at all; the real build's constant is 12% of the identity here,
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


def _warm(fn=_build_deferred_frame) -> None:
    """One call of `fn` on a one-row input: pyarrow's one-time initialisation
    and `fn`'s own first-call cost both land here instead of in the measured
    region. The default warms the real build; `_excess` also warms the
    function it is about to measure."""
    fn([np.array([[1, 2]], dtype=np.int32)], np.ones(1, dtype=np.int64))


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
    _warm(fn)
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
    ratio band could not hold: the real implementation measures ~12% over the
    identity there, because 1,664 items make a ~2.4 KB constant 12% of the
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
    build passes there. The bound is `SMOKE_DIFF_B`, half a page, because one
    extra offsets-sized array at this shape is R = 5,560 B and must not fit;
    `test_the_smoke_difference_rejects_the_r_scale_regressions` holds that the
    two forms which add one clear the real build by that bound at this shape."""
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
# The three forms this bound replaced, the one it cannot separate, and the
# doubling it is derived from
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


def _regress_double(arrs, _sup):
    """The regression class `SLACK_B` is derived from: a build that costs the
    identity twice. The two arrays are filled, both are copied, and arrow is
    handed the copies while the originals are still live."""
    import pyarrow as pa

    flat, offsets = _fill(arrs)
    np.cumsum(offsets[1:], out=offsets[1:])
    flat_copy, offsets_copy = flat.copy(), offsets.copy()
    out = pa.LargeListArray.from_arrays(offsets_copy, flat_copy)
    return out, flat, offsets  # the originals outlive the arrow call


def test_slack_rejects_the_known_regressions():
    """The slack is only worth asserting if it would fail on the forms it
    replaced. Each of these builds the same arrow array the real build wraps
    and is rejected on memory alone, so `SLACK_B` cannot be widened without
    this test noticing what it stops catching.

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


def test_the_slack_rejects_a_doubling_at_the_smoke_shape():
    """`SLACK_B` is derived from one regression class -- a build that costs the
    identity twice -- at one shape, the smoke lattice. This is that derivation
    held as two relations with no literal in either. The first pins the
    constant: the slack sits under the smoke identity, so a second identity
    exceeds it whatever the constant C >= 0 is. The second pins the apparatus:
    a doubling form measured through `_excess` is over the slack. Raising
    `SLACK_B` back to the 65,536 B an earlier revision used fails the first;
    the second is what stops the first from being arithmetic with nothing
    measured behind it."""
    arrs, supports = _lattice(SMOKE_K_HIST)
    identity = _identity(arrs)
    assert SLACK_B < identity, (
        f"SLACK_B = {SLACK_B:,} B is not under the smoke identity {identity:,} B: a build "
        "that doubles the peak at the production shape would pass the slack."
    )
    excess = _excess(_regress_double, arrs, supports)
    assert excess > SLACK_B, (
        f"a doubling at the smoke lattice measured {excess:,} B over the identity, inside "
        f"the {SLACK_B:,} B slack -- the apparatus is not seeing the second identity."
    )


def test_the_smoke_difference_rejects_the_r_scale_regressions():
    """The differenced smoke bound is only worth asserting if it fails on the
    forms the absolute slack admits at that shape. The cumsum copy (R) and the
    `widths` pair (2R) are inside `SLACK_B` at the smoke lattice -- asserted as
    the premise -- and each must clear the REAL build at the same shape by
    `SMOKE_DIFF_B`. Against the smoke real build rather than a second 40M one:
    `test_the_excess_does_not_scale_with_n` holds the real build's smoke and
    40M excesses within `SMOKE_DIFF_B` of each other, so the margin here is
    the margin the differenced row would see, shifted by a gap that row bounds
    and does not pin; and the 40M build was this file's costliest allocation,
    in a test not marked slow. The astype form is not in this list: it is not
    above the real build at all here, the blind spot the module docstring
    states and `test_no_peak_bound_separates_the_astype_form_at_low_kbar`
    asserts."""
    arrs, supports = _lattice(SMOKE_K_HIST)
    real = _excess(_build_deferred_frame, arrs, supports)
    for name, fn in [("non-in-place cumsum", _regress_cumsum_copy), ("widths list + concatenation", _regress_widths)]:
        excess = _excess(fn, arrs, supports)
        assert excess < SLACK_B, f"premise: {name} measures {excess:,} B at the smoke lattice, inside SLACK_B"
        assert excess - real >= SMOKE_DIFF_B, (
            f"{name} at the smoke lattice is {excess:,} B over the identity, only {excess - real:,} B "
            f"over the real build's {real:,} B there -- inside SMOKE_DIFF_B, so the differenced "
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

    The astype form's term over the identity is `max(4N - R, 0)`: zero at
    kbar <= 2 and, at the smoke lattice, 1,096 B, under the constant. So at
    these shapes its peak is not above the real build's, and the assertion is
    that relation, `astype <= real`, with no figure in it: how far below it
    lands depends on the process and on where in the run it is measured, and
    the revision before this one quoted a figure with the wrong sign. A
    revision before that justified a bound by claiming it "closes the k=2
    blind spot"; it does not, no peak-based bound does, and this test fails if
    the form ever measures above the real build, so the claim can be revisited
    on evidence."""
    arrs, supports = build()
    real = _excess(_build_deferred_frame, arrs, supports)
    astype = _excess(_regress_astype, arrs, supports)

    assert astype <= real, (
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
    lattice's 694 rows the same two forms add R = 5,560 and 2R = 11,120 B,
    INSIDE the slack with the constant on top, and are caught there only by
    the differenced bound in
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
