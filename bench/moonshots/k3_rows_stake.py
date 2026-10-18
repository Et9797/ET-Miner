"""K=3 counted from the rows (M2) and the pigeonhole reduce (M9), exactly, on the CPU.

From a dataset and its lattice dump (``consolidation_run._dump_lattice``): the
K=3 candidates the subset test keeps are the triangles of the frequent-pair
graph (a < b < c with all three pairs frequent). Every row lists its frequent
items; a row holds a countable candidate exactly when the candidate's three
pairs are among the row's frequent pairs. This script enumerates, per row, the
triangles among the row's frequent items, and reports:

- ``increments``: the sum over rows of the countable candidates each row holds,
  which equals the sum of the countable candidates' supports (what a row-wise
  K=3 kernel adds);
- ``row_pairs`` / ``row_triples``: the sum over rows of C(L, 2) and C(L, 3) of
  the frequent items per row (the pairs a row-wise kernel looks up and the
  triples a naive enumeration tests);
- per shard of an n-GPU row split (the miner's cuts, ``csr_bitvec._row_split_cuts``):
  each occurring candidate's per-shard counts, the share of written entries a
  pigeonhole reduce flags (some shard reaches ceil(min_count / n)), and a check
  that the candidates whose summed count reaches min_count are exactly the
  lattice's K=3 level.

Usage: uv run python bench/moonshots/k3_rows_stake.py DATASET LATTICE [--gpus 2] [--tmp DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rows import frequent_rows  # noqa: E402

N_BUCKETS = 32
CHUNK_CELLS = 1 << 24


def _shard_of_rows(n_rows: int, n_gpus: int) -> np.ndarray:
    from et_miner.gpu.csr_bitvec import _row_split_cuts

    shard = np.empty(n_rows, dtype=np.int8)
    for i, (s, e) in enumerate(_row_split_cuts(n_rows, n_gpus, balance="rows")):
        shard[s:e] = i
    return shard


def run(dataset: str, lattice: str, n_gpus: int, tmp: str) -> dict:
    from candidate_waste import load_lattice

    t0 = time.perf_counter()
    lat = load_lattice(Path(lattice))
    nf = lat.n_cols
    pairs = lat.items[2].astype(np.int64)
    pair_keys = np.sort(pairs[:, 0] * nf + pairs[:, 1])
    indptr, pos, n_rows = frequent_rows(dataset, lat.col_to_item)
    shard = _shard_of_rows(n_rows, n_gpus)
    lens = np.diff(indptr)
    out = {
        "dataset": dataset,
        "n_rows": n_rows,
        "min_count": lat.min_count,
        "n_frequent_items": nf,
        "n_frequent_pairs": len(pair_keys),
        "n_frequent_k3": int(len(lat.items.get(3, []))),
        "row_pairs": int((lens * (lens - 1) // 2).sum()),
        "row_triples": int((lens * (lens - 1) * (lens - 2) // 6).sum()),
    }
    buckets: list[list[np.ndarray]] = [[] for _ in range(N_BUCKETS)]
    bucket_files = [Path(tmp) / f"b{b}.npy" for b in range(N_BUCKETS)]
    spilled: list[list[Path]] = [[] for _ in range(N_BUCKETS)]
    pending = 0
    increments = 0
    per_row_hist = np.zeros(1, dtype=np.int64)

    def spill():
        nonlocal pending
        for b in range(N_BUCKETS):
            if buckets[b]:
                f = bucket_files[b].with_suffix(f".{len(spilled[b])}.npy")
                np.save(f, np.concatenate(buckets[b]))
                spilled[b].append(f)
                buckets[b] = []
        pending = 0

    for length in range(3, int(lens.max()) + 1):
        rows_l = np.nonzero(lens == length)[0]
        if len(rows_l) == 0:
            continue
        pi, pj = np.array(list(combinations(range(length), 2))).T
        col_of = {(i, j): c for c, (i, j) in enumerate(zip(pi, pj))}
        tri = np.array(list(combinations(range(length), 3)))
        c_ij = np.array([col_of[(i, j)] for i, j, _ in tri])
        c_ik = np.array([col_of[(i, k)] for i, _, k in tri])
        c_jk = np.array([col_of[(j, k)] for _, j, k in tri])
        step = max(1, CHUNK_CELLS // len(tri))
        for s in range(0, len(rows_l), step):
            rr = rows_l[s : s + step]
            items = pos[indptr[rr][:, None] + np.arange(length)]
            keys = items[:, pi] * nf + items[:, pj]
            at = np.searchsorted(pair_keys, keys)
            flags = (at < len(pair_keys)) & (pair_keys[np.minimum(at, len(pair_keys) - 1)] == keys)
            hit = flags[:, c_ij] & flags[:, c_ik] & flags[:, c_jk]
            per_row = hit.sum(axis=1)
            increments += int(per_row.sum())
            if per_row.max(initial=0) >= len(per_row_hist):
                per_row_hist = np.pad(per_row_hist, (0, int(per_row.max()) + 1 - len(per_row_hist)))
            np.add.at(per_row_hist, per_row, 1)
            r_idx, t_idx = np.nonzero(hit)
            if len(r_idx) == 0:
                continue
            it = items[r_idx]
            cand = (it[np.arange(len(r_idx)), tri[t_idx, 0]] * nf + it[np.arange(len(r_idx)), tri[t_idx, 1]]) * nf + it[
                np.arange(len(r_idx)), tri[t_idx, 2]
            ]
            tagged = cand * n_gpus + shard[rr[r_idx]]
            b_of = cand % N_BUCKETS
            order = np.argsort(b_of, kind="stable")
            tagged, b_of = tagged[order], b_of[order]
            bounds = np.searchsorted(b_of, np.arange(N_BUCKETS + 1))
            for b in range(N_BUCKETS):
                if bounds[b + 1] > bounds[b]:
                    buckets[b].append(tagged[bounds[b] : bounds[b + 1]])
            pending += len(tagged)
            if pending > 50_000_000:
                spill()
    spill()
    out["increments"] = increments
    out["rows_by_countable_held"] = {int(i): int(c) for i, c in enumerate(per_row_hist) if c}
    out["enumerate_s"] = round(time.perf_counter() - t0, 1)

    # Aggregate per candidate and shard.
    flag_at = math.ceil(lat.min_count / n_gpus)
    occurring = flagged = frequent = 0
    count_hist = np.zeros(65, dtype=np.int64)
    sum_check = 0
    freq_keys = []
    for b in range(N_BUCKETS):
        if not spilled[b]:
            continue
        tagged = np.concatenate([np.load(f) for f in spilled[b]])
        for f in spilled[b]:
            f.unlink()
        u, c = np.unique(tagged, return_counts=True)
        cand, sh = u // n_gpus, u % n_gpus
        first = np.ones(len(cand), dtype=bool)
        first[1:] = cand[1:] != cand[:-1]
        starts = np.nonzero(first)[0]
        total = np.add.reduceat(c, starts)
        per_shard_max = np.maximum.reduceat(c, starts)
        sum_check += int(total.sum())
        occurring += len(starts)
        flagged += int((per_shard_max >= flag_at).sum())
        is_freq = total >= lat.min_count
        frequent += int(is_freq.sum())
        freq_keys.append(np.stack([cand[starts][is_freq], total[is_freq]], axis=1))
        np.add.at(count_hist, np.minimum(total, 64), 1)
    if sum_check != increments:
        raise AssertionError(f"aggregated {sum_check:,} increments, enumerated {increments:,}")
    want = np.empty((0, 2), dtype=np.int64)
    if 3 in lat.items:
        it = lat.items[3].astype(np.int64)
        want = np.stack([(it[:, 0] * nf + it[:, 1]) * nf + it[:, 2], lat.counts[3]], axis=1)
        want = want[np.argsort(want[:, 0])]
    got = np.concatenate(freq_keys) if freq_keys else np.empty((0, 2), dtype=np.int64)
    got = got[np.argsort(got[:, 0])]
    out["k3_matches_lattice"] = bool(np.array_equal(want, got))
    out["n_gpus"] = n_gpus
    out["flag_threshold"] = flag_at
    out["occurring_candidates"] = occurring
    out["flagged"] = flagged
    out["frequent"] = frequent
    out["count_hist_capped_64"] = {int(i): int(c) for i, c in enumerate(count_hist) if c}
    out["total_s"] = round(time.perf_counter() - t0, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("lattice")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp:
        print(json.dumps(run(args.dataset, args.lattice, args.gpus, tmp)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
