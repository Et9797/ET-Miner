"""K=3 census of the explosion workloads under several item orders (M10), with the row-wise bound (M2).

Reads a lattice dump. The K=3 groups are, per prefix item a, the frequent
pairs (a, b) with b after a; a pair (b, c) of a's successors is counted when
(b, c) is frequent (the subset test). Exact, per a, with a sparse submatrix of
the frequent-pair graph. For each item order (``orig``: ascending item id, the
miner's column order; ``asc`` / ``desc``: ascending / descending K=1 support;
``deg``: ascending degree in the frequent-pair graph) it reports the generated
candidates, the counted ones, today's kernel dispatch per group
(``row_split_chunks.group_kernels``) and each kernel's work: per-candidate
counted candidates, the group kernel's listed pairs and staged rows, the tiled
kernel's counted 32x32 tile-pairs. Order-independent: ``U`` (sum over the
counted candidates of their smallest pair count) and the summed supports of
the frequent ones.

Usage: uv run python bench/moonshots/k3_census.py LATTICE [--orders orig,asc,desc,deg]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candidate_waste import load_lattice  # noqa: E402

TILE = 32


def _order(name: str, lat, deg: np.ndarray) -> np.ndarray:
    """new label of each column under the order ``name``."""
    n = lat.n_cols
    if name == "orig":
        return np.arange(n)
    key = {"asc": lat.counts[1], "desc": -lat.counts[1], "deg": deg}[name]
    ranked = np.lexsort((np.arange(n), key))
    label = np.empty(n, dtype=np.int64)
    label[ranked] = np.arange(n)
    return label


def census(lat, order: str) -> dict:
    from et_miner.gpu.row_split_chunks import group_kernels

    t0 = time.perf_counter()
    ab = lat.items[2].astype(np.int64)
    cnt = lat.counts[2]
    n = lat.n_cols
    deg = np.bincount(ab.ravel(), minlength=n)
    label = _order(order, lat, deg)
    lo = np.minimum(label[ab[:, 0]], label[ab[:, 1]])
    hi = np.maximum(label[ab[:, 0]], label[ab[:, 1]])
    upper = sp.csr_matrix((cnt.astype(np.int64), (lo, hi)), shape=(n, n))
    upper.sort_indices()
    out_deg = np.diff(upper.indptr)
    sizes = out_deg[out_deg >= 2]
    pairs = sizes * (sizes - 1) // 2
    kern = group_kernels(3, pairs, sizes)
    kind_of = np.full(n, -1)
    kind_of[out_deg >= 2] = np.where(kern.tiled, 2, np.where(kern.group, 1, 0))
    res = {"order": order, "generated": int(pairs.sum()), "groups": int(len(sizes)),
           "max_suffixes": int(sizes.max()) if len(sizes) else 0, "counted": 0, "U": 0,
           "percand_counted": 0, "group_listed_pairs": 0, "group_staged_rows": 0,
           "tiled_counted_tilepairs": 0, "tiled_tilepairs": 0, "tiled_listed_pairs": 0,
           "groups_by_kernel": {name: int((kind_of == v).sum()) for v, name in enumerate(("percand", "group", "tiled"))}}
    for a in np.nonzero(out_deg >= 2)[0]:
        lo_, hi_ = upper.indptr[a], upper.indptr[a + 1]
        succ = upper.indices[lo_:hi_]
        c_a = upper.data[lo_:hi_]
        sub = upper[succ][:, succ].tocoo()
        if sub.nnz == 0:
            if kind_of[a] == 2:
                s = len(succ)
                nt = -(-s // TILE)
                res["tiled_tilepairs"] += nt * (nt + 1) // 2
            continue
        i, j = sub.row.astype(np.int64), sub.col.astype(np.int64)
        res["counted"] += sub.nnz
        res["U"] += int(np.minimum(np.minimum(c_a[i], c_a[j]), sub.data).sum())
        kind = kind_of[a]
        if kind == 0:
            res["percand_counted"] += sub.nnz
        elif kind == 1:
            res["group_listed_pairs"] += sub.nnz
            res["group_staged_rows"] += 1 + len(np.union1d(i, j))
        else:
            s = len(succ)
            nt = -(-s // TILE)
            res["tiled_tilepairs"] += nt * (nt + 1) // 2
            res["tiled_listed_pairs"] += sub.nnz
            res["tiled_counted_tilepairs"] += int(np.unique((i // TILE) * nt + j // TILE).size)
    if 3 in lat.counts:
        res["frequent"] = int(len(lat.counts[3]))
        res["freq_sum"] = int(lat.counts[3].sum())
    res["words"] = math.ceil(lat.n_rows / 64)
    res["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lattice")
    ap.add_argument("--orders", default="orig,asc,desc,deg")
    args = ap.parse_args()
    lat = load_lattice(Path(args.lattice))
    for order in args.orders.split(","):
        print(json.dumps({"lattice": Path(args.lattice).name, **census(lat, order)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
