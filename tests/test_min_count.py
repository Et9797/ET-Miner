"""The min-count threshold, checked against one shared table.

`count >= ceil(min_support * n_transactions)` is implemented three times:

    src/et_miner/core/result.py      _min_count
    src/et_miner/synthetic.py        SynthSpec.min_count
    rust_ext/src/core/apriori.rs     exact_min_count

Defect #11 existed because all three carried the *same wrong expression*,
`ceil(fl64(s) * N)`, and therefore agreed with each other while all three
disagreed with efficient-apriori -- the oracle CLAUDE.md mandates. Agreement
between implementations is worth nothing when they share a bug, so this file
checks all three against `tests/fixtures/min_count_cases.json`, which the Rust
`#[test]` reads too. One table, not four.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

from et_miner.core.result import _min_count
from et_miner.synthetic import PRESETS, SynthSpec

FIXTURE = Path(__file__).parent / "fixtures" / "min_count_cases.json"
CASES = json.loads(FIXTURE.read_text())["cases"]


def _ids(cases):
    return [f"s={c['min_support']!r},n={c['n_rows']}" for c in cases]


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_min_count_matches_the_shared_fixture(case):
    got = _min_count(case["min_support"], case["n_rows"])
    assert got == case["expected"], case["why"]


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_synthspec_min_count_agrees_with_the_helper(case):
    """The third site. It feeds check_preset_purpose's vacuity guard and the
    tier gate's oracle threshold, so drift here moves what the gate certifies
    rather than failing it."""
    spec = SynthSpec(
        name="fixture", n_rows=case["n_rows"], vocab_size=10, zipf_a=1.0, row_len_mean=3,
        min_support=case["min_support"],
    )
    assert spec.min_count == case["expected"] == _min_count(
        case["min_support"], case["n_rows"]
    )


def test_the_fixture_discriminates_against_the_float_expression():
    """If no case distinguishes the exact ceiling from `ceil(fl64(s)*N)`, the
    tests above would pass against the defect and prove nothing."""
    differing = [
        c for c in CASES
        if math.ceil(c["min_support"] * c["n_rows"]) != c["expected"]
    ]
    assert len(differing) >= 3, f"fixture no longer discriminates: {len(differing)} cases"


def test_the_documented_rule_is_the_exact_decimal_ceiling():
    """State the contract as an assertion: min_support denotes the shortest
    decimal that round-trips to the float, not the float's exact binary value."""
    for c in CASES:
        assert c["expected"] == math.ceil(Fraction(str(c["min_support"])) * c["n_rows"])


def test_smoke_preset_threshold_is_unchanged():
    """PRESETS['smoke'] is the tier-equivalence gate's dataset. If its threshold
    moved, the gate would be certifying a different lattice than before and the
    change would be invisible in the gate's own output."""
    assert PRESETS["smoke"].min_count == 600


@pytest.mark.parametrize("n_rows", [100, 10_000, 100_000])
def test_boundary_itemset_at_exactly_the_threshold_is_kept(n_rows):
    """The case #11 was measured on, as a unit test: an itemset whose count is
    exactly ceil(s*N) must be frequent. The old expression returned one more
    than the exact ceiling at s=0.07, so the boundary itemset -- and the entire
    cone above it -- was dropped."""
    s = 0.07
    exact = math.ceil(Fraction(str(s)) * n_rows)
    assert _min_count(s, n_rows) == exact
    assert exact / n_rows >= s or math.isclose(exact / n_rows, s, rel_tol=1e-12), (
        "an itemset at exactly the threshold must satisfy the oracle's "
        "count/N >= min_support criterion"
    )


def test_rust_apriori_from_csr_keeps_the_boundary_itemset():
    """The Rust site, end to end through the public entry point.

    `apriori_from_csr` is exported from et_miner (__init__.py) and is Tier 2 of
    CLAUDE.md's mandated chain. It carries its own copy of the threshold
    (`exact_min_count`), so the Python fixture above cannot speak for it -- and
    Tier 2 in test_tier_equivalence.py reaches the Rust *counting* path via
    apriori(sparse=True), not this function.

    Before the fix this returned 3 itemsets; the 4 containing item 0 were
    dropped because the threshold came back as 701 rather than 700.
    """
    et_miner_rust = pytest.importorskip("et_miner_rust")
    import numpy as np

    n_rows, s = 10_000, 0.07
    threshold = math.ceil(Fraction(str(s)) * n_rows)  # 700
    rows = [[0, 1, 2]] * threshold + [[1, 2]] * (n_rows - threshold)

    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    flat: list[int] = []
    for i, r in enumerate(rows):
        flat.extend(r)
        indptr[i + 1] = len(flat)

    itemsets, counts = et_miner_rust.apriori_from_csr(
        indptr, np.array(flat, dtype=np.int64), n_rows, 3, s, 8
    )
    got = {tuple(sorted(x)): c for x, c in zip(itemsets, counts)}

    assert got.get((0,)) == threshold, (
        f"item 0 has count exactly {threshold} = ceil({s} * {n_rows}) and must be frequent; "
        f"got {sorted(got)}"
    )
    for cone in [(0, 1), (0, 2), (0, 1, 2)]:
        assert cone in got, f"{cone} sits above the boundary itemset and was lost with it"
