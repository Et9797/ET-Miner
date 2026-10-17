"""All-zero word tiles the counting kernels stage, per row order (M5b, M12).

The tiled and group kernels stage 32-word tiles (2,048 rows) of the prefix AND
and of the suffix rows, and the per-candidate kernel reads the same words one
candidate at a time. A tile whose prefix AND is zero adds nothing to any count
of its group; a suffix row's tile that is zero adds nothing to its pairs.

``deep``: per K>=3 level, the share of a group's 32-word tiles whose prefix
AND is zero, weighted by the group's generated pairs, over a uniform sample of
the level's prefix groups; and the share a kernel finds from the per-item tile
occupancy alone (the AND of the prefix items' occupancy bits, a superset of the
non-zero prefix tiles); for each row order:
    orig     the dataset's order
    lex      rows sorted by their frequent items in descending-support order
             (lexicographic on the item indicator vector)
    len_lex  rows sorted by their number of frequent items (descending), then lex

``k3``: on a K=3 explosion, exactly, for the tiled kernel's counted
tile-pairs: the row tiles staged today ((1 + 64) per word tile), with a skip of
the zero prefix tiles, and with a skip of the zero prefix tiles and of the zero
suffix-row tiles inside the rest; for each row order.

Usage:
    uv run python bench/moonshots/tile_stakes.py deep LATTICE --dataset NAME [--groups 2000]
    uv run python bench/moonshots/tile_stakes.py k3 LATTICE --dataset NAME
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
from rows import frequent_rows, row_ids  # noqa: E402

TILE_W = 32
TILE_ROWS = TILE_W * 64
TILE_T = 32
ORDERS = ("orig", "lex", "len_lex")


def row_order(name: str, indptr: np.ndarray, pos: np.ndarray, support: np.ndarray) -> np.ndarray | None:
    """New position of each row under ``name`` (None for the dataset's order)."""
    if name == "orig":
        return None
    n_rows = len(indptr) - 1
    rank = np.empty(len(support), dtype=np.int64)
    rank[np.lexsort((np.arange(len(support)), -support))] = np.arange(len(support))
    r = rank[pos]
    rid = row_ids(indptr)
    n_keys = -(-len(support) // 63)
    keys = []
    for w in range(n_keys):
        m = (r >= 63 * w) & (r < 63 * (w + 1))
        key = np.zeros(n_rows, dtype=np.int64)
        np.bitwise_or.at(key, rid[m], np.left_shift(np.int64(1), 62 - (r[m] - 63 * w)))
        keys.append(-key)
    sort_keys = keys[::-1]
    if name == "len_lex":
        sort_keys.append(-np.diff(indptr))
    order = np.lexsort(sort_keys)
    new = np.empty(n_rows, dtype=np.int64)
    new[order] = np.arange(n_rows)
    return new


def bitvecs(indptr: np.ndarray, pos: np.ndarray, n_cols: int, new_row: np.ndarray | None) -> np.ndarray:
    """(n_cols, words) uint64 bitvectors of the rows, in the order ``new_row``."""
    n_rows = len(indptr) - 1
    words = math.ceil(n_rows / 64)
    rid = row_ids(indptr)
    if new_row is not None:
        rid = new_row[rid]
    key = pos * words + rid // 64
    order = np.argsort(key, kind="stable")
    key = key[order]
    bit = np.left_shift(np.uint64(1), (rid[order] % 64).astype(np.uint64))
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    out = np.zeros(n_cols * words, dtype=np.uint64)
    out[key[starts]] = np.bitwise_or.reduceat(bit, starts)
    return out.reshape(n_cols, words)


def deep(lat, indptr, pos, n_groups: int, seed: int) -> None:
    from et_miner.gpu.kernels.k3plus import build_k3plus_groups_from_flat

    for order in ORDERS:
        t0 = time.perf_counter()
        bv = bitvecs(indptr, pos, lat.n_cols, row_order(order, indptr, pos, lat.counts[1]))
        words = bv.shape[1]
        n_tiles = -(-words // TILE_W)
        pad = n_tiles * TILE_W - words
        item_tiles = np.pad(bv, ((0, 0), (0, pad))).reshape(lat.n_cols, n_tiles, TILE_W).any(axis=2)
        rng = np.random.default_rng(seed)
        for k in range(3, max(lat.items) + 2):
            if len(lat.items.get(k - 1, [])) < 2:
                break
            groups = build_k3plus_groups_from_flat(np.ascontiguousarray(lat.items[k - 1]))
            if groups is None:
                break
            prefix = np.asarray(groups.prefix_items, dtype=np.int64).reshape(-1, k - 2)
            pairs = np.diff(np.asarray(groups.cumulative_pairs, dtype=np.int64))
            live = np.nonzero(pairs > 0)[0]
            pick = live if len(live) <= n_groups else np.sort(rng.choice(live, n_groups, replace=False))
            zero = np.zeros(len(pick))
            zero_occ = np.zeros(len(pick))
            for x, g in enumerate(pick):
                acc = bv[prefix[g, 0]].copy()
                occ = item_tiles[prefix[g, 0]].copy()
                for c in range(1, k - 2):
                    acc &= bv[prefix[g, c]]
                    occ &= item_tiles[prefix[g, c]]
                tiles = np.pad(acc, (0, pad)).reshape(n_tiles, TILE_W).any(axis=1)
                zero[x] = 1.0 - tiles.mean()
                zero_occ[x] = 1.0 - occ.mean()
            w = pairs[pick].astype(np.float64)
            print(json.dumps({"mode": "deep", "order": order, "k": k, "groups": int(len(live)),
                              "sampled": int(len(pick)), "zero_prefix_tile_share": float((zero * w).sum() / w.sum()),
                              "zero_by_item_occupancy": float((zero_occ * w).sum() / w.sum()),
                              "unweighted": float(zero.mean()), "elapsed_s": round(time.perf_counter() - t0, 1)}),
                  flush=True)
        del bv


def occupancy(indptr, pos, n_cols: int, new_row: np.ndarray | None) -> np.ndarray:
    """(n_cols, n_tiles) bool: column c has a row in 2,048-row tile t."""
    n_rows = len(indptr) - 1
    n_tiles = -(-n_rows // TILE_ROWS)
    rid = row_ids(indptr)
    if new_row is not None:
        rid = new_row[rid]
    occ = np.zeros((n_cols, n_tiles), dtype=bool)
    occ[pos, rid // TILE_ROWS] = True
    return occ


def k3(lat, indptr, pos) -> None:
    from et_miner.gpu.row_split_chunks import group_kernels

    ab = lat.items[2].astype(np.int64)
    n = lat.n_cols
    upper = sp.csr_matrix((np.ones(len(ab), dtype=np.int8), (ab[:, 0], ab[:, 1])), shape=(n, n))
    upper.sort_indices()
    out_deg = np.diff(upper.indptr)
    sizes = out_deg[out_deg >= 2]
    kern = group_kernels(3, sizes * (sizes - 1) // 2, sizes)
    tiled = np.zeros(n, dtype=bool)
    tiled[np.nonzero(out_deg >= 2)[0][kern.tiled]] = True
    for order in ORDERS:
        t0 = time.perf_counter()
        occ = occupancy(indptr, pos, n, row_order(order, indptr, pos, lat.counts[1]))
        n_tiles = occ.shape[1]
        today = prefix_skip = both_skip = 0
        n_tp = 0
        for a in np.nonzero(tiled)[0]:
            succ = upper.indices[upper.indptr[a] : upper.indptr[a + 1]]
            sub = upper[succ][:, succ].tocoo()
            if sub.nnz == 0:
                continue
            s = len(succ)
            nt = -(-s // TILE_T)
            tp = np.unique((sub.row.astype(np.int64) // TILE_T) * nt + sub.col // TILE_T)
            ta, tb = tp // nt, tp % nt
            pa = occ[a]
            per_row = (occ[succ] & pa).sum(axis=1)
            per_tile = np.bincount(np.arange(s) // TILE_T, weights=per_row, minlength=nt)
            n_tp += len(tp)
            today += len(tp) * n_tiles * (1 + 2 * TILE_T)
            prefix_skip += len(tp) * int(pa.sum()) * (1 + 2 * TILE_T)
            both_skip += int(len(tp) * int(pa.sum()) + (per_tile[ta] + per_tile[tb]).sum())
        print(json.dumps({"mode": "k3", "order": order, "counted_tilepairs": n_tp, "tiles": n_tiles,
                          "staged_row_tiles_today": today, "with_prefix_skip": prefix_skip,
                          "with_prefix_and_suffix_skip": both_skip,
                          "elapsed_s": round(time.perf_counter() - t0, 1)}), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("deep", "k3"))
    ap.add_argument("lattice")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--groups", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lat = load_lattice(Path(args.lattice))
    indptr, pos, _ = frequent_rows(args.dataset, lat.col_to_item)
    if args.mode == "deep":
        deep(lat, indptr, pos, args.groups, args.seed)
    else:
        k3(lat, indptr, pos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
