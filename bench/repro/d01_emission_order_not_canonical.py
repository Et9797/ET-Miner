"""#1 -- Tier 1 emits itemsets in lexicographic column-NAME order.

build_boolean_matrix names its columns i_0 .. i_N with no zero padding
(core/matrix.py:136), assigned positionally over the surviving frequent items
sorted by item id (:121-137). Every ordering decision in the CPU path is then a
STRING comparison over those names -- sorted(...) at core/candidates.py:71 and
pl.col("a") < pl.col("b") at :94 and :224. From 11 frequent items on,
"i_10" < "i_2" while item_ids[10] > item_ids[2], and core/apriori.py:734 maps
that order straight to item ids with no re-sort.

Two independent symptoms, both checked here:
  (a) emitted itemsets are not ascending;
  (b) the order is unstable ACROSS THRESHOLDS -- an item's column index depends
      on how many other items cleared the threshold below it, so the same
      itemset is emitted differently at two min_support values.
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

    rng = np.random.default_rng(0)
    rows = [sorted(rng.choice(15, size=int(rng.integers(4, 9)), replace=False).tolist())
            for _ in range(300)]
    df = pl.DataFrame({"items": rows})

    res = apriori(df, min_support=0.05)
    sets = res["itemset"].to_list()
    unsorted_rows = sum(1 for s in sets if list(s) != sorted(s))

    # (b) cross-threshold instability. The column index of an item depends on
    # how many OTHER items cleared the threshold below it, so this needs the
    # frequent-item COUNT to differ between the two thresholds -- not merely
    # two different thresholds. Items 2 and 10 are near-ubiquitous; the rest
    # clear a low threshold only.
    rows2 = []
    for i in range(300):
        r = {2, 10}
        if i % 10 < 4:
            r |= set(rng.choice(15, size=5, replace=False).tolist())
        rows2.append(sorted(r))
    df2 = pl.DataFrame({"items": rows2})

    def emitted(ms: float) -> dict[tuple, list]:
        r = apriori(df2, min_support=ms)
        return {tuple(sorted(s)): list(s) for s in r["itemset"].to_list()}

    n_freq_hi = len(apriori(df2, min_support=0.8, max_length=1))
    n_freq_lo = len(apriori(df2, min_support=0.1, max_length=1))
    lo, hi = emitted(0.1), emitted(0.8)
    unstable = [k for k in (set(lo) & set(hi)) if lo[k] != hi[k]]

    live = unsorted_rows > 0 or bool(unstable)
    ex = next((list(s) for s in sets if list(s) != sorted(s)), None)
    return live, (
        f"{unsorted_rows}/{len(sets)} emitted itemsets not ascending (e.g. {ex}); "
        f"{len(unstable)} itemsets emitted in a different order at min_support "
        f"0.1 ({n_freq_lo} frequent items) vs 0.8 ({n_freq_hi})"
    )


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
