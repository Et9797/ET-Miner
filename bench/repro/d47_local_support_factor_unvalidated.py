"""#47 -- `local_support_factor` is unvalidated; a value above 1.0 silently
breaks SON's superset guarantee.

`apriori_streaming` (son.py) computed
`local_min_support = min_support * local_support_factor` with no range check.
Fixed: both SON entries raise ValueError outside (0, 1]. SON is only correct when the local threshold is <= the
global one ("frequent globally => frequent in at least one chunk"); a factor
above 1.0 makes pass 1 STRICTER than pass 2, so globally frequent itemsets can
be missed entirely, with no error.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def reproduce() -> tuple[bool, str]:
    sys.path.insert(0, str(REPO / "src"))
    import polars as pl

    from et_miner.streaming.son import apriori_streaming

    # 4 chunks; the pair {1,2} sits just above the global threshold but is
    # thinly spread, so a stricter local threshold loses it.
    rows = ([[1, 2]] * 55 + [[1]] * 45) * 4
    df = pl.DataFrame({"items": rows})

    ok = apriori_streaming(df.lazy(), min_support=0.5, chunk_size=100,
                           local_support_factor=0.9, show_progress=False)
    try:
        bad = apriori_streaming(df.lazy(), min_support=0.5, chunk_size=100,
                                local_support_factor=5.0, show_progress=False)
    except ValueError as e:
        return False, f"factor=5.0 raises ValueError: {e}"

    accepted = True  # it did not raise
    lost = ok.height - bad.height
    live = accepted and lost > 0
    if not live:
        live = accepted  # accepting an out-of-range factor at all is the defect
    return live, (
        f"factor=5.0 accepted without error; itemsets {ok.height} -> {bad.height} "
        f"({lost} lost)"
    )


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
