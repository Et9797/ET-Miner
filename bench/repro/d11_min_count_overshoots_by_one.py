"""#11 -- _min_count overshoots by one, dropping itemsets the mandated oracle keeps.

CLAUDE.md documents the rule as `count >= ceil(s*N)`. What is implemented is
`count >= ceil(fl64(s)*N)`: 0.07 has no exact binary64 representation, so
0.07*10000 = 700.0000000000001 and math.ceil returns 701 where the exact
ceiling is 700. The loss is not one itemset but the whole cone above it.

Three sites carry the same wrong expression and therefore AGREE with each other
while all three disagree with efficient-apriori, whose criterion is
`count / len(transactions) >= min_support` (700/10000 >= 0.07 is True). Under
CLAUDE.md's own correctness policy, which names efficient-apriori as *the*
canonical oracle, that settles the intended contract.

This repro asserts against the oracle, not against the other two sites.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def reproduce() -> tuple[bool, str]:
    sys.path.insert(0, str(REPO / "src"))
    import polars as pl
    from efficient_apriori import apriori as ea_apriori

    from et_miner import apriori
    from et_miner.core.result import _min_count

    N, S = 10_000, 0.07
    # item 0 in exactly 700 rows -> support exactly 0.07, exactly on the boundary
    rows = [[0, 1, 2] for _ in range(700)] + [[1, 2] for _ in range(N - 700)]
    df = pl.DataFrame({"items": rows})

    mined = {tuple(sorted(s)) for s in apriori(df, min_support=S)["itemset"].to_list()}
    oracle_sets, _ = ea_apriori(
        [tuple(r) for r in rows], min_support=S, min_confidence=1.0, max_length=8
    )
    oracle = {tuple(sorted(k)) for by_k in oracle_sets.values() for k in by_k}

    missing = sorted(oracle - mined)
    threshold = _min_count(S, N)
    live = bool(missing)
    return live, (
        f"_min_count({S}, {N}) = {threshold} (exact ceiling 700); "
        f"miner {len(mined)} itemsets vs oracle {len(oracle)}; "
        f"missing from the miner: {missing[:5]}"
    )


if __name__ == "__main__":
    live, ev = reproduce()
    print(("LIVE  " if live else "FIXED ") + ev)
    raise SystemExit(0 if live else 1)
