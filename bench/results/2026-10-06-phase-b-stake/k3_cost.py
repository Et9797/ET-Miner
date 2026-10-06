"""K=3 work on a K=3 workload: the dense and the sparse (ESCO) bytes read for the candidates the subset test keeps.

Reads a lattice dump (see `o4_stake.py`). The K=3 candidates that survive the
subset test are the triangles a < b < c of the frequent-pair graph; per prefix
item a they are the frequent pairs inside a's successors. Exact, per a, with a
sparse submatrix: the surviving candidates, dense bytes (candidates ×
ceil(N/64) words × 8 B, no tile reuse) and sparse merge bytes
(4 B × (|tids(a, b)| + |tids(a, c)|) per candidate). Also prints the
conversion size (4 B × Σ K=2 counts) and the fit-check bound (twice that).

Usage: uv run python k3_cost.py DUMP_PREFIX [DUMP_PREFIX ...]
"""

import json
import math
import sys

import numpy as np
import polars as pl
import scipy.sparse as sp


def k3_cost(prefix: str) -> None:
    meta = json.loads(open(prefix + ".json").read())
    df = pl.read_parquet(prefix + ".parquet").with_columns(pl.col("itemset").list.len().alias("k"))
    n = meta["n_rows"]
    words = math.ceil(n / 64)
    f2 = df.filter(pl.col("k") == 2)
    ab = np.sort(np.array(f2["itemset"].to_list(), dtype=np.int64), axis=1)
    w = f2["count"].to_numpy().astype(np.int64)
    m = int(ab.max()) + 1
    upper = sp.csr_matrix((np.ones(len(w), dtype=np.int64), (ab[:, 0], ab[:, 1])), shape=(m, m))
    weight = sp.csr_matrix((w, (ab[:, 0], ab[:, 1])), shape=(m, m))
    cands = merge = 0
    for a in range(m):
        lo, hi = upper.indptr[a], upper.indptr[a + 1]
        if hi - lo < 2:
            continue
        succ = upper.indices[lo:hi]
        wa = weight.data[lo:hi]  # same order as succ: both built from the same coordinates
        sub = upper[succ][:, succ]  # (b, c) frequent with b, c in succ(a): one candidate (a, b, c) each
        if sub.nnz == 0:
            continue
        cands += sub.nnz
        merge += int(np.asarray(sub.sum(axis=1)).ravel() @ wa) + int(np.asarray(sub.sum(axis=0)).ravel() @ wa)
    s2 = int(w.sum())
    print(
        f"== {prefix} N={n:,} words={words:,} F2={len(w):,} mean K=2 count={s2 / len(w):.0f} "
        f"(N/32={n / 32:,.0f}) conversion={4 * s2 / 1e9:.2f} GB (fit bound {8 * s2 / 1e9:.2f} GB)"
    )
    print(
        f"   K=3 after subset test {cands:,} (F3={df.filter(pl.col('k') == 3).height:,}); dense bytes "
        f"{8 * cands * words:.2e}, sparse merge bytes {4 * merge:.2e}, ratio {4 * merge / (8 * cands * words):.1e}"
    )


if __name__ == "__main__":
    for p in sys.argv[1:]:
        k3_cost(p)
