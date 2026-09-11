"""#48 -- `chunk_size` is unvalidated; a negative value returns an empty frame
with no error.

son.py:197 computes math.ceil(n_total / effective_chunk_size) with no check.
chunk_size=0 raises ZeroDivisionError there; a negative value makes n_chunks
negative, so range(n_chunks) is empty, BOTH passes no-op, and :325-329 returns
a normal empty frame -- indistinguishable from "nothing was frequent".
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def reproduce() -> tuple[bool, str]:
    sys.path.insert(0, str(REPO / "src"))
    import polars as pl

    from et_miner.streaming.son import apriori_streaming

    df = pl.DataFrame({"items": [[1, 2, 3]] * 200 + [[1, 2]] * 200})
    truth = apriori_streaming(df.lazy(), min_support=0.3, chunk_size=100, show_progress=False)

    outcomes = {}
    for cs in (0, -1):
        try:
            r = apriori_streaming(df.lazy(), min_support=0.3, chunk_size=cs, show_progress=False)
            outcomes[cs] = f"{r.height} rows, no error"
        except Exception as e:  # noqa: BLE001
            outcomes[cs] = type(e).__name__

    silent_empty = outcomes.get(-1, "").startswith("0 rows")
    live = silent_empty or outcomes.get(0) == "ZeroDivisionError"
    return live, (
        f"valid chunk_size -> {truth.height} rows; chunk_size=0 -> {outcomes.get(0)}; "
        f"chunk_size=-1 -> {outcomes.get(-1)}"
    )


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
