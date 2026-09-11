"""#2 -- generate_rules() silently drops rules and reports lift = 0.0.

_build_support_lookup keys the map on the itemset's STORED order
(core/rules.py:43) but both lookups query it SORTED (:81, :89). Given #1 those
differ, producing two distinct silent failures: an lhs miss returns 0.0 and the
rule is discarded by the continue at :82-83; an rhs miss returns 0.0 and :90
emits lift = 0.0 -- a plausible-looking wrong number rather than a dropped row.

The comparison is against the same frame with every tuple sorted, which is
what the fix makes the producer emit.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def reproduce() -> tuple[bool, str]:
    sys.path.insert(0, str(REPO / "src"))
    import numpy as np
    import polars as pl

    from et_miner import apriori
    from et_miner.core.rules import generate_rules

    rng = np.random.default_rng(0)
    rows = [sorted(rng.choice(15, size=int(rng.integers(4, 9)), replace=False).tolist())
            for _ in range(300)]
    df = pl.DataFrame({"items": rows})

    res = apriori(df, min_support=0.05, max_length=3)
    as_emitted = generate_rules(res, min_confidence=0.0)

    canonical = res.with_columns(
        pl.col("itemset").list.sort().alias("itemset")
    )
    as_sorted = generate_rules(canonical, min_confidence=0.0)

    dropped = len(as_sorted) - len(as_emitted)
    zero_lift = sum(1 for r in as_emitted if r.lift == 0.0)
    live = dropped > 0 or zero_lift > 0
    return live, (
        f"as-emitted {len(as_emitted)} rules vs canonical {len(as_sorted)} "
        f"-> {dropped} missing ({100*dropped/max(len(as_sorted),1):.1f}%); "
        f"{zero_lift} survivors carry lift == 0.0"
    )


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
