"""O4's stake per level: the tidset bytes a K≥3 level holds, the share of inferred survivors, and identical tidsets once.

Reads a complete lattice dump (`<name>.lattice.parquet` with its `.json`
sidecar, written by `bench/runner.py --mode waste`) and prints per level k ≥ 3:
the frequent itemsets, the share inferable from a non-free (k−1)-subset (I in
`bench/results/2026-10-05-candidate-waste/FINDINGS.md`), the tidset bytes
(4 B per tid), the share of those bytes held by inferred survivors, and the
bytes left when itemsets with the same closure (the same tidset) store it once.
Exact: counts come from the dump, closures from the equal-count supersets in it.

Usage: uv run python o4_stake.py DUMP_PREFIX [DUMP_PREFIX ...]
"""

import json
import sys
from collections import defaultdict

import polars as pl


def stake(prefix: str) -> None:
    meta = json.loads(open(prefix + ".json").read())
    df = pl.read_parquet(prefix + ".parquet")
    cnt = {tuple(sorted(s)): c for s, c in zip(df["itemset"].to_list(), df["count"].to_list())}
    by_k = defaultdict(list)
    for s in cnt:
        by_k[len(s)].append(s)
    free = {s: len(s) == 1 or all(cnt[s[:i] + s[i + 1 :]] != cnt[s] for i in range(len(s))) for s in cnt}
    closure = defaultdict(set)
    for k in sorted(by_k):
        for z in by_k[k]:
            for i in range(k):
                x = z[:i] + z[i + 1 :]
                if k > 1 and cnt[x] == cnt[z]:
                    closure[x].add(z[i])
    print(f"== {prefix} N={meta['n_rows']:,} min_count={meta['min_count']:,} ({len(cnt):,} itemsets)")
    print(f"{'k':>3} {'n':>8} {'I %':>6} {'bytes MB':>10} {'I bytes %':>9} {'distinct MB':>11} {'distinct %':>10}")
    tot = tot_i = tot_d = 0
    for k in sorted(by_k):
        if k < 3:
            continue
        b = bi = ni = 0
        tidsets = {}
        for s in by_k[k]:
            b += 4 * cnt[s]
            if any(not free[s[:i] + s[i + 1 :]] for i in range(k)):
                ni += 1
                bi += 4 * cnt[s]
            tidsets[tuple(sorted(set(s) | closure[s]))] = cnt[s]
        d = 4 * sum(tidsets.values())
        tot, tot_i, tot_d = tot + b, tot_i + bi, tot_d + d
        print(
            f"{k:>3} {len(by_k[k]):>8,} {100 * ni / len(by_k[k]):>6.1f} {b / 1e6:>10.1f} {100 * bi / b:>9.1f} "
            f"{d / 1e6:>11.1f} {100 * d / b:>10.1f}"
        )
    print(
        f"all K>=3: {tot / 1e6:.1f} MB, inferred {100 * tot_i / max(tot, 1):.1f} %, distinct {100 * tot_d / max(tot, 1):.1f} %"
    )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        stake(p)
