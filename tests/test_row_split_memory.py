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

THE BOUND IS ABSOLUTE, NOT A RATIO, and that is the whole design of this file.
Measured excess of the real implementation over the identity:

    N = 1,664 items (the real `smoke` lattice)     2,350 B   ratio 1.1245
    N = 400,000                                    2,378 B   ratio 1.0006
    N = 4,000,000                                  2,378 B   ratio 1.0001
    N = 40,000,000                                 2,378 B   ratio 1.0000

Four orders of magnitude in N; the excess does not move, the ratio moves by
12%. It is flat in the chunk count too (2,378 / 2,394 / 2,378 B at 1, 8 and 64
chunks -- the 16 B spread is measurement jitter, not a trend). So the
excess is an additive constant and a RATIO band measures the fixture, not the
code -- the previous version of this file asserted `peak < 1.05 * identity`,
which the real implementation FAILS at the real `smoke` lattice shape (1.1245).
That band passed only because its fixture was N = 4M, where a 2.4 KB constant
is 0.006%. `SLACK_B` replaces it: the real implementation clears it by 27-28x
at every shape above, and `test_the_excess_does_not_scale_with_n` keeps the
invariance that licenses an absolute bound under test rather than merely
recorded here.

WHERE THIS BOUND IS BLIND, stated because a peak-based test cannot close it and
should not be described as if it had. The `.astype`-on-concatenate regression
costs `max(4N - R, 0)` bytes over the identity, with N = 8 * total_items and
R = 8 * (total_rows + 1). That is IDENTICALLY ZERO at kbar <= 2, so at kbar <= 2
no bound on the peak -- ratio, slack or otherwise -- can tell the fixed code
from that regression. Measured: at k = 2 and k = 1 over 2M items the astype
form's excess is 1,175 B, BELOW the real implementation's own 2,378 B. At the
real `smoke` lattice (kbar = 2.3977, just past the 7/3 crossover) it is
1,332 B, still below. `test_no_peak_bound_separates_the_astype_form_at_k2`
asserts the blind spot rather than hiding it. Discrimination is a large-N
property and `test_slack_rejects_the_known_regressions` establishes it at the
fixture shape, where the three regressions run 98x-195x over the slack.

pyarrow's FIRST `LargeListArray.from_arrays` call in a process allocates ~26 MB
of one-time initialisation, which tracemalloc attributes to whatever traced
region happens to run first. Every test here warms it before `reset_peak()`.
Without that warmup this test fails at small N and silently passes at large N
as the constant is amortised away -- flaky in the one direction that looks like
success.

No exact byte count is asserted anywhere below. The excess is
context-dependent, not shape-dependent: the same build measures 2,426 B inside
one call frame and 2,378 B in another, at identical shapes. The INVARIANCE
reproduces; the constant is an artifact of where it was measured, so `SLACK_B`
is set an order of magnitude above it rather than fitted to it.
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

SLACK_B = 65_536
"""Absolute headroom over the identity, in bytes.

Chosen an order of magnitude above the ~2.4 KB measured excess, not fitted to
it: the constant is a property of the measurement context and moves between
call frames, while its INVARIANCE in N is what this file relies on. The three
known regressions clear it by 98x-195x at the fixture shape, so the gap costs
no discrimination there.
"""

# The k-histogram of the canonical `smoke` preset's real output: 694 itemsets,
# 1,664 items, kbar = 2.3977. `deferred_itemsets_np` holds ONE ENTRY PER K
# LEVEL, so the production shape is a strongly-peaked lattice with a tiny tail
# -- not the equal-chunks/uniform-k fixture. The old ratio band could not hold
# this case at all; an absolute slack clears it by 28x (2,350 B against 65,536).
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
    total. It clears the absolute slack by 28x."""
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

    Two shapes differenced rather than either compared to a literal: the
    constant itself is context-dependent (2,378 B at module level, 2,426 B
    inside a call frame, same shapes), so only the difference is a property of
    the code."""
    small_arrs, small_sup = _uniform(100_000, 5)
    large_arrs, large_sup = _uniform(40_000_000, 5)

    small = _excess(_build_deferred_frame, small_arrs, small_sup)
    large = _excess(_build_deferred_frame, large_arrs, large_sup)

    assert abs(large - small) < 4096, (
        f"the excess moved from {small:,} B to {large:,} B across a 400x change in N "
        f"({abs(large - small):,} B apart). It is supposed to be additive and constant; "
        "if it scales, replace the absolute slack rather than widening it."
    )
    assert large < SLACK_B and small < SLACK_B, "premise: both must be inside the slack"


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
    `test_no_peak_bound_separates_the_astype_form_at_k2`. What this establishes
    is that the slack discriminates, not that a regression in
    `_build_deferred_frame` would be caught here: none of these three call it.
    `test_peak_is_flat_plus_offsets_and_nothing_else` is what does that."""
    arrs, supports = _fixture()
    expected = _identity(arrs)
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


@pytest.mark.parametrize("k", [1, 2])
def test_no_peak_bound_separates_the_astype_form_at_k2(k):
    """The blind spot, asserted so it cannot be quietly reclassified as covered.

    The astype form costs `max(4N - R, 0)` over the identity, which is exactly
    zero at kbar <= 2. At those shapes its peak is indistinguishable from the
    fixed code's -- measurably BELOW it, since both sit under the same ~2.4 KB
    constant. A previous revision justified a bound by claiming it "closes the
    k=2 blind spot"; it does not, no peak-based bound does, and this test fails
    if that ever becomes true so the claim can be revisited on evidence."""
    arrs, supports = _uniform(2_000_000, k)
    real = _excess(_build_deferred_frame, arrs, supports)
    astype = _excess(_regress_astype, arrs, supports)

    assert astype <= real + 4096, (
        f"k={k}: the astype form now measures {astype:,} B over the identity against the "
        f"real build's {real:,} B. If that is a real separation, the docstring's "
        "'no peak bound distinguishes these at kbar <= 2' is stale."
    )
    assert astype < SLACK_B, f"k={k}: premise -- the astype form is inside the slack here"


@pytest.mark.parametrize("k", [1, 2])
def test_the_other_two_regressions_are_still_caught_at_k2(k):
    """The blind spot above is specific to the astype form. The `widths` pair
    and the cumsum copy both scale with R, not with `4N - R`, so they stay
    outside the slack at every kbar -- which is why the fix is still worth a
    test at low kbar."""
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
