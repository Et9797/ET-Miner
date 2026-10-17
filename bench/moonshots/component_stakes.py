"""Item components of a lattice per starting level K0 (M4, the zeta-transform finisher).

Before level K0 >= 3, every frequent itemset of size >= K0 lies inside one
connected component of the graph that joins the items of each frequent
(K0-1)-itemset: any two of its items share a frequent (K0-1)-subset. Per K0 the
script reports the components holding at least K0 items, their sizes n_c, the
table a superset-sum zeta transform needs (2^n_c int32 counts), the itemsets
of size >= K0 the finisher would emit (checked to lie inside one component),
and with ``--dataset`` the rows with at least K0 items of some component (the
histogram's input).

The per-family variant: the frequent (K0-1)-itemsets grouped by their
(K0-2)-prefix P, with E(P) the items that extend P. Every frequent itemset of
size >= K0-1 whose first K0-2 items are P lies inside P with E(P), since each
of its items x outside P makes P+x a frequent (K0-1)-subset. Per K0 the
script reports the families, the largest E(P), the largest table 2^|E(P)| and
the sum over families of count(P) x |E(P)| (the rows a conditional histogram
reads).

Usage: uv run python bench/moonshots/component_stakes.py LATTICE [--dataset NAME] > OUT.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candidate_waste import load_lattice  # noqa: E402
from rows import frequent_rows, row_ids  # noqa: E402


def components(lat, k0: int) -> np.ndarray:
    """Component label per column from the frequent (k0-1)-itemsets; -1 for columns in none."""
    sets = lat.items[k0 - 1].astype(np.int64)
    a = np.repeat(sets[:, :1], sets.shape[1] - 1, axis=1).ravel()
    b = sets[:, 1:].ravel()
    g = sp.coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(lat.n_cols, lat.n_cols))
    _, label = connected_components(g, directed=False)
    used = np.zeros(lat.n_cols, dtype=bool)
    used[np.unique(sets)] = True
    return np.where(used, label, -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lattice")
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()
    lat = load_lattice(Path(args.lattice))
    rows = frequent_rows(args.dataset, lat.col_to_item) if args.dataset else None
    rid = row_ids(rows[0]) if rows else None
    top = max(lat.items)
    for k0 in range(3, top + 1):
        if len(lat.items.get(k0 - 1, [])) == 0:
            break
        label = components(lat, k0)
        sizes = np.bincount(label[label >= 0])
        big = np.nonzero(sizes >= k0)[0]
        emitted = 0
        for k in range(k0, top + 1):
            sets = lat.items[k]
            lab = label[sets]
            if (lab != lab[:, :1]).any() or (lab < 0).any():
                raise AssertionError(f"K0={k0}: a frequent {k}-itemset spans two components")
            emitted += len(sets)
        out = {
            "k0": k0,
            "components": int(len(big)),
            "sizes": sorted((int(s) for s in sizes[big]), reverse=True)[:20],
            "max_items": int(sizes[big].max()) if len(big) else 0,
            "table_bytes": int(sum(4 * 2 ** int(s) for s in sizes[big])),
            "zeta_adds": int(sum(int(s) * 2 ** int(s) for s in sizes[big])),
            "itemsets_from_k0": emitted,
        }
        fam = lat.items[k0 - 1]
        if k0 >= 3 and len(fam):
            prefix_rows = lat.lookup(np.ascontiguousarray(fam[:, :-1])) if k0 > 2 else np.zeros(len(fam), np.int64)
            pid, ext = np.unique(prefix_rows, return_counts=True)
            prefix_counts = lat.counts[k0 - 2][pid]
            out["families"] = int(len(pid))
            out["family_max_ext"] = int(ext.max())
            out["family_max_table_bytes"] = int(4 * 2 ** int(ext.max()))
            out["family_tables_bytes"] = int(sum(4 * 2 ** int(e) for e in ext))
            out["family_rows_x_ext"] = int((prefix_counts * ext).sum())
        if rows is not None:
            comp_of_pos = label[rows[1]]
            in_big = np.isin(comp_of_pos, big)
            key = rid[in_big] * (int(label.max()) + 1) + comp_of_pos[in_big]
            _, per = np.unique(key, return_counts=True)
            out["row_component_hits"] = int((per >= k0).sum())
            out["row_component_items"] = int(per[per >= k0].sum())
        print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
