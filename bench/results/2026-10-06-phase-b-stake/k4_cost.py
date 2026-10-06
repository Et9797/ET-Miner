"""K=4 work on a workload mined to K=3: what one more level would add, dense against sparse (ESCO).

Reads a lattice dump (see `o4_stake.py`) holding the frequent triples. The K=4
candidates are the pairs of a prefix group (a, b, c), (a, b, d); the subset
test keeps those with (a, c, d) and (b, c, d) frequent. Exact, per triple
(a, b, c), with set intersections: the generated and the surviving candidates,
dense bytes (survivors × ceil(N/64) words × 8 B, no tile reuse) and sparse merge
bytes (4 B × (|tids(a, b, c)| + |tids(a, b, d)|) per survivor), and the size of
the K=3 tidsets a conversion at K=4 would hold.

Usage: uv run python k4_cost.py DUMP_PREFIX [DUMP_PREFIX ...]
"""

import json
import math
import sys
from collections import defaultdict

import polars as pl


def k4_cost(prefix: str) -> None:
    meta = json.loads(open(prefix + ".json").read())
    n = meta["n_rows"]
    words = math.ceil(n / 64)
    df = pl.read_parquet(prefix + ".parquet").with_columns(pl.col("itemset").list.len().alias("k"))
    f3 = df.filter(pl.col("k") == 3)
    cnt = {tuple(sorted(s)): c for s, c in zip(f3["itemset"].to_list(), f3["count"].to_list())}
    groups = defaultdict(set)
    for t in cnt:
        groups[t[:2]].add(t[2])
    generated = sum(len(g) * (len(g) - 1) // 2 for g in groups.values())
    survivors = merge = 0
    for (a, b, c), count in cnt.items():
        ds = groups[(a, b)] & groups.get((a, c), set()) & groups.get((b, c), set())  # every d > c
        if ds:
            survivors += len(ds)
            merge += count * len(ds) + sum(cnt[(a, b, d)] for d in ds)
    print(
        f"== {prefix} N={n:,} F3={len(cnt):,} K=3 tidsets={4 * sum(cnt.values()) / 1e9:.2f} GB; K=4 generated "
        f"{generated:,}, after subset test {survivors:,}; dense bytes {8 * survivors * words:.2e}, "
        f"sparse merge bytes {4 * merge:.2e}"
    )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        k4_cost(p)
