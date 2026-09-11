"""#22 -- anchor_items prunes the level the NEXT level generates from.

The anchor mask is applied to `current_flat`, and that same filtered array then
becomes BOTH downstream populations: `full_flat` (the subset oracle every K+1
apriori test resolves against) and `prev_frequent_flat` (the generation base).

Two independent unsoundnesses at once:

  channel 1 -- the apriori oracle needs the (k-1)-subsets that DROP the anchor,
    and those are unanchored by construction, so a restricted oracle rejects
    valid candidates;
  channel 2 -- the prefix-join needs the family closed under its two
    prefix-parents, and an anchored candidate's parents may both be unanchored,
    so X can never be generated.

This is the same defect class gpu/row_split.py documents as FIXED for
free-sets. The difference in outcome is one property: freeness is
anti-monotone, anchoredness is not.

The expected set is computed from the UNANCHORED lattice mined by the same
engine, so this compares the engine against itself rather than against a
reimplementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _fixture(n_rows: int = 6000, n_items: int = 20, seed: int = 7):
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        r = set(rng.choice(n_items, size=int(rng.integers(4, 9)), replace=False).tolist())
        # nested vocabulary: a child implies its parent, so the lattice is deep
        if 3 in r:
            r.add(2)
        if 5 in r:
            r.add(4)
        rows.append(sorted(r))
    return pl.DataFrame({"items": rows})


def reproduce() -> tuple[bool, str]:
    sys.path.insert(0, str(REPO / "src"))
    try:
        import cupy  # noqa: F401
    except ImportError:
        return False, "no CUDA device; the row-split anchor path is unreachable"

    from et_miner import apriori

    df = _fixture()
    findings = []

    # HIGH-sorting anchors isolate channel 2; LOW-sorting anchors isolate
    # channel 1. Both gate settings, because apriori() ties the two flags and
    # gates=ON is what the public API actually produces.
    for anchors, label in (({16, 17, 18, 19}, "HIGH"), ({0, 1, 2, 3}, "LOW")):
        for gates in (False, True):
            full = apriori(df, min_support=0.05, max_length=5, use_gpu=True,
                           prune_equal_support=gates)
            expected = {
                tuple(sorted(s)) for s in full["itemset"].to_list()
                if len(s) >= 2 and anchors & set(s)
            }
            got = {
                tuple(sorted(s))
                for s in apriori(df, min_support=0.05, max_length=5, use_gpu=True,
                                 prune_equal_support=gates,
                                 anchor_items=anchors)["itemset"].to_list()
                if len(s) >= 2
            }
            missing = expected - got
            if missing:
                findings.append(
                    f"{label} gates={'ON ' if gates else 'OFF'}: expected {len(expected):,} "
                    f"got {len(got):,} MISSING {len(missing):,} "
                    f"({100 * len(missing) / len(expected):.1f}%)"
                )

    return bool(findings), "; ".join(findings) if findings else "every anchored itemset is returned"


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
