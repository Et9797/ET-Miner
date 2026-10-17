"""Per-level replay of the row-split miner's K>=3 levels for the moonshot stakes (M2, M5a, M7, M8).

Replays each K>=3 level from a complete lattice dump as the GPU route
generates it (prefix groups of the previous level, ``build_k3plus_groups_from_flat``)
and classifies every generated candidate as ``bench/candidate_waste.py`` does:
prunable (a (k-1)-subset is infrequent), inferable (all subsets frequent, one
not free) or counted. Per level it reports:

- the kernel work of today's dispatch (``row_split_chunks.group_kernels`` with
  no knob set): per-candidate row reads, the group kernel's listed pairs and
  staged rows, the tiled kernel's counted tile-pairs, all per word;
- M2: ``U`` = the sum over the counted candidates of their smallest
  (k-1)-subset count, and the exact sum of the frequent ones' supports plus a
  bound for the infrequent ones (the increments a row-wise kernel makes);
- M5a (with ``--dataset``): the rows holding at least k of the level's
  active items (the items of level k-1), and the sum over rows of
  C(active items, k);
- M7: among the counted candidates that count inference does not settle, the
  share the non-derivable-itemset bounds of depth <= 3 settle: an upper bound
  below min_count (infrequent) or equal lower and upper bounds (the count);
  every settled frequent candidate is checked against its true count;
- M8: the sum over the level's prefix groups of C(m, 3), the triples a
  look-ahead would count, against the next level's generated candidates.

Both regimes are reported: complete lattice (every non-prunable candidate
counted) and count inference (the inferable ones taken from their subsets).

Usage: uv run python bench/moonshots/replay_stakes.py LATTICE [--dataset NAME] [--no-ndi]
       [--ndi-sample F] [--max-k K] > OUT.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candidate_waste import _expand, _segments, load_lattice  # noqa: E402
from rows import frequent_rows, row_ids  # noqa: E402

TILE = 32
NDI_BATCH = 1 << 20


def _subset_counts(lat, sets: np.ndarray, drop: tuple[int, ...]) -> np.ndarray:
    """Counts of ``sets`` without the columns ``drop`` (every such subset is frequent), N for the empty set."""
    keep = [c for c in range(sets.shape[1]) if c not in drop]
    if not keep:
        return np.full(len(sets), lat.n_rows, dtype=np.int64)
    sub = np.ascontiguousarray(sets[:, keep])
    idx = lat.lookup(sub)
    if (idx < 0).any():
        raise AssertionError("a subset of a counted candidate is missing from the lattice")
    return lat.counts[len(keep)][idx]


def ndi_bounds(lat, sets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) support bounds from the deduction rules of depth 1 to 3 (Calders & Goethals 2002)."""
    k = sets.shape[1]
    s1 = {(a,): _subset_counts(lat, sets, (a,)) for a in range(k)}
    upper = np.minimum.reduce(list(s1.values()))
    lower = np.zeros(len(sets), dtype=np.int64)
    s2 = {}
    if k >= 2:
        for a, b in combinations(range(k), 2):
            s2[(a, b)] = _subset_counts(lat, sets, (a, b))
            lower = np.maximum(lower, s1[(a,)] + s1[(b,)] - s2[(a, b)])
    if k >= 3:
        for a, b, c in combinations(range(k), 3):
            s3 = _subset_counts(lat, sets, (a, b, c))
            bound = s1[(a,)] + s1[(b,)] + s1[(c,)] - s2[(a, b)] - s2[(a, c)] - s2[(b, c)] + s3
            upper = np.minimum(upper, bound)
    return lower, upper


def replay_level(lat, k: int, *, ndi: bool, ndi_sample: float, rng) -> dict:
    from et_miner.gpu.kernels.k3plus import build_k3plus_groups_from_flat
    from et_miner.gpu.row_split_chunks import group_kernels

    t0 = time.perf_counter()
    groups = build_k3plus_groups_from_flat(np.ascontiguousarray(lat.items[k - 1]), with_src_rows=True)
    if groups is None:
        return {"k": k, "generated": 0}
    prefix = np.asarray(groups.prefix_items, dtype=np.int32).reshape(-1, k - 2)
    so = np.asarray(groups.suffix_offsets, dtype=np.int64)
    suffixes = np.asarray(groups.suffixes, dtype=np.int64)
    slot_row = np.asarray(groups.suffix_src_rows, dtype=np.int64)
    cumulative = np.asarray(groups.cumulative_pairs, dtype=np.int64)
    sizes, pairs = np.diff(so), np.diff(cumulative)
    n_groups = len(sizes)
    kern = group_kernels(k, pairs, sizes)
    kind = np.where(kern.tiled, 2, np.where(kern.group, 1, 0))  # 0 per-candidate, 1 group, 2 tiled
    slot_group = np.repeat(np.arange(n_groups, dtype=np.int64), sizes)
    prev_counts, prev_free = lat.counts[k - 1], lat.free[k - 1]

    drop_ids, drop_ok = [], []
    for d in range(k - 2):
        q = np.delete(prefix, d, axis=1)
        qid = lat.lookup(q) if q.shape[1] else np.zeros(n_groups, dtype=np.int64)
        rid, rok = lat.step(k - 2, np.maximum(qid, 0)[slot_group], suffixes, (qid >= 0)[slot_group])
        drop_ids.append(rid)
        drop_ok.append(rok)

    nt = (sizes + TILE - 1) // TILE
    tp = np.where(sizes >= 2, nt * (nt + 1) // 2, 0)
    tile_base = np.concatenate([[0], np.cumsum(tp)[:-1]]).astype(np.int64)
    n_tp = int(tp.sum())
    regimes = ("full", "infer")
    tile_counted = {r: np.zeros(n_tp, dtype=bool) for r in regimes}
    slot_used = {r: np.zeros(len(suffixes), dtype=bool) for r in regimes}
    listed = {r: np.zeros(n_groups, dtype=np.int64) for r in regimes}
    acc = {r: dict.fromkeys(("counted", "U", "freq_sum", "infreq_bound", "frequent"), 0) for r in regimes}
    agg = dict.fromkeys(("generated", "P", "I", "C", "frequent"), 0)
    ndi_acc = dict.fromkeys(("considered", "skip_infrequent", "exact", "exact_frequent", "checked_frequent"), 0)

    for batch in _segments(np.arange(n_groups), sizes, pairs):
        g, i, j = _expand(batch)
        si, sj = so[g] + i, so[g] + j
        s_i, s_j = suffixes[si], suffixes[sj]
        a, b = slot_row[si], slot_row[sj]
        minc = np.minimum(prev_counts[a], prev_counts[b])
        nonfree = ~prev_free[a] | ~prev_free[b]
        found = np.ones(len(g), dtype=bool)
        for d in range(k - 2):
            pos, hit = lat.step(k - 1, drop_ids[d][si], s_j, drop_ok[d][si])
            found &= hit
            minc = np.where(hit, np.minimum(minc, prev_counts[pos]), minc)
            nonfree |= hit & ~prev_free[pos]
        pos, freq = lat.step(k, a, s_j, np.ones(len(g), dtype=bool))
        true_count = np.where(freq, lat.counts[k][pos], 0) if k in lat.counts else np.zeros(len(g), dtype=np.int64)
        agg["generated"] += len(g)
        agg["P"] += int((~found).sum())
        agg["I"] += int((found & nonfree).sum())
        agg["C"] += int((found & ~nonfree).sum())
        agg["frequent"] += int(freq.sum())
        ta, tb = i // TILE, j // TILE
        tidx = tile_base[g] + ta * nt[g] - ta * (ta - 1) // 2 + (tb - ta)
        for r, counted in (("full", found), ("infer", found & ~nonfree)):
            a_ = acc[r]
            a_["counted"] += int(counted.sum())
            a_["U"] += int(minc[counted].sum())
            a_["freq_sum"] += int(true_count[counted & freq].sum())
            a_["infreq_bound"] += int(np.minimum(minc, lat.min_count - 1)[counted & ~freq].sum())
            a_["frequent"] += int((counted & freq).sum())
            tile_counted[r][tidx[counted]] = True
            slot_used[r][si[counted]] = True
            slot_used[r][sj[counted]] = True
            listed[r] += np.bincount(g[counted], minlength=n_groups)

        if ndi:
            c_mask = found & ~nonfree
            if ndi_sample < 1.0:
                c_mask &= rng.random(len(g)) < ndi_sample
            sel = np.nonzero(c_mask)[0]
            for s in range(0, len(sel), NDI_BATCH):
                part = sel[s : s + NDI_BATCH]
                sets = np.concatenate([prefix[g[part]], s_i[part, None], s_j[part, None]], axis=1).astype(np.int32)
                lo, up = ndi_bounds(lat, sets)
                skip = up < lat.min_count
                exact = ~skip & (lo == up)
                fr = freq[part]
                if (skip & fr).any():
                    raise AssertionError(f"K={k}: an upper bound below min_count on a frequent candidate")
                if (exact & fr & (up != true_count[part])).any():
                    raise AssertionError(f"K={k}: equal bounds disagree with a true count")
                ndi_acc["considered"] += len(part)
                ndi_acc["skip_infrequent"] += int(skip.sum())
                ndi_acc["exact"] += int(exact.sum())
                ndi_acc["exact_frequent"] += int((exact & fr).sum())
                ndi_acc["checked_frequent"] += int((exact & fr).sum())

    words = math.ceil(lat.n_rows / 64)
    row = {"k": k, "groups": n_groups, "words": words, **agg,
           "groups_by_kernel": {name: int((kind == v).sum()) for v, name in enumerate(("percand", "group", "tiled"))},
           "triples_sum": int((sizes * (sizes - 1) * (sizes - 2) // 6).sum())}
    tile_group = np.repeat(np.arange(n_groups), tp)
    for r in regimes:
        used_per_group = np.bincount(slot_group, weights=slot_used[r], minlength=n_groups).astype(np.int64)
        counted_groups = listed[r] > 0
        work = {
            "percand_candidates": int(listed[r][kind == 0].sum()),
            "percand_row_reads": int(listed[r][kind == 0].sum()) * k,
            "group_listed_pairs": int(listed[r][kind == 1].sum()),
            "group_staged_rows": int(((k - 2) + used_per_group)[(kind == 1) & counted_groups].sum()),
            "tiled_counted_tilepairs": int((tile_counted[r] & (kind[tile_group] == 2)).sum()),
            "tiled_listed_pairs": int(listed[r][kind == 2].sum()),
        }
        work["tiled_slots"] = work["tiled_counted_tilepairs"] * TILE * TILE
        work["tiled_staged_rows"] = work["tiled_counted_tilepairs"] * ((k - 2) + 2 * TILE)
        row[r] = {**acc[r], **work}
    if ndi:
        row["ndi"] = {**ndi_acc, "sample": ndi_sample}
    row["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return row


def row_stats(lat, indptr: np.ndarray, pos: np.ndarray, k: int) -> dict:
    """Rows with at least k of level k-1's items, and the sum over rows of C(that count, k)."""
    active = np.zeros(lat.n_cols, dtype=bool)
    active[np.unique(lat.items[k - 1])] = True
    per_row = np.bincount(row_ids(indptr)[active[pos]], minlength=len(indptr) - 1)
    kept = per_row >= k
    combos = sum(math.comb(int(n), k) * int(c) for n, c in zip(*np.unique(per_row[kept], return_counts=True)))
    return {"active_items": int(active.sum()), "kept_rows": int(kept.sum()), "kept_share": float(kept.mean()),
            "row_subsets": int(combos)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lattice")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--no-ndi", action="store_true")
    ap.add_argument("--ndi-sample", type=float, default=1.0)
    ap.add_argument("--max-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lat = load_lattice(Path(args.lattice))
    rng = np.random.default_rng(args.seed)
    rows = None
    if args.dataset:
        rows = frequent_rows(args.dataset, lat.col_to_item)
    top = max(lat.items)
    if args.max_k:
        top = min(top, args.max_k - 1)
    for k in range(3, top + 2):
        if len(lat.items.get(k - 1, [])) < 2:
            break
        out = replay_level(lat, k, ndi=not args.no_ndi, ndi_sample=args.ndi_sample, rng=rng)
        if rows is not None:
            out["rows"] = row_stats(lat, rows[0], rows[1], k)
        print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
