"""K=3 tile-pair census on the explosion workloads: the tile-pairs the tiled kernel counts, per tile size.

Reads a lattice dump (see `o6_stake.py`). The K=3 groups are the frequent
pairs per prefix item a; a pair (b, c) of a's successors survives the subset
test when (b, c) is a frequent pair. Exact, per a, with a sparse submatrix:
for the groups the tiled kernel takes (≥ `TILED_MIN_GROUP_PAIRS[3]` pairs) it
prints all tile-pairs and the counted ones (≥ 1 surviving pair) at tile sizes
32, 16 and 8, with the share of the counted tile-pairs' slots that hold a
surviving pair; and the candidates of the per-candidate groups.

Usage: uv run python k3_tiles.py DUMP_PREFIX [DUMP_PREFIX ...]
"""

import json
import math
import sys

import numpy as np
import polars as pl
import scipy.sparse as sp

MIN_PAIRS_K3 = 120  # gpu/row_split_chunks.py::TILED_MIN_GROUP_PAIRS[3]
TILES = (32, 16, 8)


def census(prefix: str) -> None:
    meta = json.loads(open(prefix + ".json").read())
    df = pl.read_parquet(prefix + ".parquet").with_columns(pl.col("itemset").list.len().alias("k"))
    words = math.ceil(meta["n_rows"] / 64)
    ab = np.sort(np.array(df.filter(pl.col("k") == 2)["itemset"].to_list(), dtype=np.int64), axis=1)
    m = int(ab.max()) + 1
    upper = sp.csr_matrix((np.ones(len(ab), dtype=np.int8), (ab[:, 0], ab[:, 1])), shape=(m, m))
    all_tp = dict.fromkeys(TILES, 0)
    counted_tp = dict.fromkeys(TILES, 0)
    gen_tiled = cnt_tiled = gen_pc = cnt_pc = n_tiled = 0
    for a in range(m):
        s = int(upper.indptr[a + 1] - upper.indptr[a])
        if s < 2:
            continue
        n_pairs = s * (s - 1) // 2
        succ = upper.indices[upper.indptr[a] : upper.indptr[a + 1]]
        sub = upper[succ][:, succ].tocoo()  # row < col: positions of b < c in succ(a)
        if n_pairs < MIN_PAIRS_K3:
            gen_pc += n_pairs
            cnt_pc += sub.nnz
            continue
        n_tiled += 1
        gen_tiled += n_pairs
        cnt_tiled += sub.nnz
        for t in TILES:
            nt = -(-s // t)
            all_tp[t] += nt * (nt + 1) // 2
            if sub.nnz:
                counted_tp[t] += np.unique((sub.row // t).astype(np.int64) * nt + sub.col // t).size
    print(
        f"== {prefix.split('/')[-1]} N={meta['n_rows']:,} words={words:,}: tiled groups {n_tiled:,}, "
        f"generated {gen_tiled:,}, surviving {cnt_tiled:,}; per-candidate groups generated {gen_pc:,}, "
        f"surviving {cnt_pc:,}"
    )
    for t in TILES:
        print(
            f"   T={t:2d}: tile-pairs {all_tp[t]:,}, counted {counted_tp[t]:,} "
            f"({100 * counted_tp[t] / all_tp[t]:.2f} %), their slots {counted_tp[t] * t * t:,} "
            f"(use {100 * cnt_tiled / (counted_tp[t] * t * t):.1f} %)"
        )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        census(p)
