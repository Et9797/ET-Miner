#!/usr/bin/env python3
"""Which (min_support, n_rows) pairs change under the #11 min-count fix?

The miner keeps `count >= ceil(s * N)`. Today that is evaluated as
`math.ceil(fl64(s) * N)`, which can be one too high when the binary64 product
lands just above an integer. The fix evaluates the exact decimal:
`math.ceil(Fraction(str(s)) * N)`.

Run this against every (min_support, n_rows) pair a published figure was
produced with. A pair that prints SHIFT had its threshold silently raised by
one, which drops every itemset at exactly that count -- and the whole cone
above it.
"""

from __future__ import annotations

import argparse
import math
from decimal import Decimal
from fractions import Fraction


def current(min_support: float, n_rows: int) -> int:
    """What core/result.py:17 computes today."""
    return math.ceil(min_support * n_rows)


def exact(min_support: float, n_rows: int) -> int:
    """What the fix computes: the exact decimal ceiling."""
    return math.ceil(Fraction(str(min_support)) * n_rows)


def sweep(n_rows: int, supports: list[float]) -> list[tuple[float, int, int]]:
    return [(s, current(s, n_rows), exact(s, n_rows)) for s in supports]


def _exhaustive_supports(max_sig: int = 5) -> list[float]:
    """Every value with <= max_sig significant digits in [1e-6, 1).

    This is what turns "no threshold shifts" from a sample into a clearance for
    a given row count -- 211,104 values at max_sig=5.
    """
    out: set[float] = set()
    for exp in range(-6, 0):
        for sig in range(1, 10 ** max_sig):
            if len(str(sig).rstrip("0")) > max_sig:
                continue
            v = float(Decimal(sig) * (Decimal(10) ** exp))
            if 0.0 < v < 1.0:
                out.add(v)
    return sorted(out)


def _default_supports() -> list[float]:
    out: list[float] = []
    for mag in (-1, -2, -3, -4, -5, -6):
        for lead in range(1, 10):
            out.append(round(lead * 10.0**mag, 10))
        for lead in (15, 25, 35, 45, 5, 55, 65, 75, 85, 95):
            out.append(round(lead * 10.0 ** (mag - 1), 11))
    return sorted(set(s for s in out if 0.0 < s < 1.0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-rows", type=int, action="append", required=True,
                    help="row count of a run (repeatable)")
    ap.add_argument("--min-support", type=float, action="append",
                    help="threshold to check (repeatable); default sweeps a grid")
    ap.add_argument("--exhaustive", action="store_true",
                    help="sweep every value with <=5 significant digits in [1e-6, 1)")
    args = ap.parse_args()

    if args.min_support:
        supports = args.min_support
    elif args.exhaustive:
        supports = _exhaustive_supports()
    else:
        supports = _default_supports()
    any_shift = False

    for n in args.n_rows:
        rows = sweep(n, supports)
        shifted = [r for r in rows if r[1] != r[2]]
        print(f"\nn_rows = {n:,}   checked {len(rows)} thresholds")
        if not shifted:
            print("  no threshold in this set shifts -- lattice unchanged")
            continue
        any_shift = True
        print(f"  {len(shifted)} SHIFT (current threshold is one too high):")
        print(f"    {'min_support':>14}  {'current':>12}  {'exact':>12}")
        for s, cur, ex in shifted:
            print(f"    {s!r:>14}  {cur:>12,}  {ex:>12,}")

    return 1 if any_shift else 0


if __name__ == "__main__":
    raise SystemExit(main())
