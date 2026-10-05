"""Offline per-level classifier of the candidates the GPU route generates.

Replays the row-split miner's K>=2 candidate generation (the K=2 pair space
over the frequent items, then the prefix groups of the previous level's
generation base, built by the same group builder) from a complete frequent
lattice, and labels every generated candidate:

    P  prunable    at least one (k-1)-subset is infrequent
    I  inferable   every (k-1)-subset is frequent and at least one is not
                   free; the count is the minimum of the subset counts
    C  must count  everything else

Per level it records the generated candidates (the GPU route's
``n_candidates``), the frequent survivors, |P|, |I|, |C|, the share of P a
per-candidate, per-32x32-tile-pair and per-suffix prune removes, the work
model of both kernels (candidates, tile-pairs) and, for every regime, the
candidates with a (k-1)-subset missing from the free level. Every level
checks: P ∪ I ∪ C is the generated set, no P candidate is frequent, every I
candidate is frequent with the inferred count, and the missing-from-free
candidates are exactly P ∪ I. CPU only (numpy and the Rust group builder).

Usage:
    uv run python bench/candidate_waste.py classify LATTICE --workload ID --out JSONL
        [--free] [--gpu-result JSON] [--free-dump PARQUET]
        [--sample-groups F --seed N] [--brute]
    uv run python bench/candidate_waste.py report --raw RAW_JSONL [RAW_JSONL ...] --waste JSONL

Options:
    LATTICE            ``<config>.lattice.parquet`` of a complete-lattice run
                       (``dump_lattice`` in ``bench/consolidation_run.py``;
                       the ``.lattice.json`` sidecar beside it holds n_rows,
                       min_count and max_length)
    --workload         workload id recorded with each row (e.g. dsl)
    --free             replay the free-set route (``prune_equal_support=True``)
    --gpu-result       a GPU config's ``.result.json`` in the same regime;
                       its per-level n_candidates and n_frequent must match
    --free-dump        the free-set run's lattice dump; the replayed free
                       levels must equal it
    --sample-groups F  classify only prefix groups drawn with inclusion
                       probability F (systematic over groups ordered by size,
                       random start) at levels K>=3; K=2 is then counted
                       analytically
    --seed             seed of the sample's random start (default 0)
    --brute            count every classified candidate on the CPU from the
                       dataset's rows and check the labels against the true
                       counts (small datasets only)
    report             markdown tables from the classifier's rows and the
                       ``--mode waste`` runner rows (level time splits)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Suffixes per tile of the tiled kernel (``shared_tiled.TILE_T``).
TILE = 32

#: Candidates per classification batch.
BATCH = 1 << 24

#: K=2 pair spaces above this are counted analytically (no K=2 subset can be infrequent).
K2_ENUMERATE_MAX = 200_000_000


@dataclass
class Lattice:
    """A complete frequent lattice in column space (ascending item ids = columns)."""

    meta: dict
    n_rows: int
    min_count: int
    n_cols: int
    col_to_item: np.ndarray
    items: dict[int, np.ndarray]
    counts: dict[int, np.ndarray]
    keys: dict[int, np.ndarray]
    free: dict[int, np.ndarray]

    def step(self, level: int, ids: np.ndarray, last: np.ndarray, ok: np.ndarray):
        """Extend (level-1)-row ids by one item: (row ids at ``level``, found).

        A level-t itemset's key is ``id(prefix) * n_cols + last item``; keys
        ascend with the lexicographic row order, so one binary search finds a
        row, and a missing prefix or key means the itemset is infrequent.
        """
        keys = self.keys.get(level)
        if keys is None or len(keys) == 0:
            return np.zeros(len(ids), dtype=np.int64), np.zeros(len(ids), dtype=bool)
        q = ids.astype(np.int64) * self.n_cols + last
        pos = np.searchsorted(keys, q)
        np.minimum(pos, len(keys) - 1, out=pos)
        hit = ok & (keys[pos] == q)
        return np.where(hit, pos, 0), hit

    def lookup(self, sets: np.ndarray) -> np.ndarray:
        """Row index of each (n, t) ascending itemset in level t, -1 when infrequent."""
        n, t = sets.shape
        ids = np.zeros(n, dtype=np.int64)
        ok = np.ones(n, dtype=bool)
        for c in range(t):
            ids, ok = self.step(c + 1, ids, sets[:, c], ok)
        return np.where(ok, ids, -1)


def _level_arrays(df, t: int, col_to_item: np.ndarray):
    import polars as pl

    sub = df.filter(pl.col("k") == t)
    flat = sub["itemset"].explode(empty_as_null=True).to_numpy().astype(np.int64).reshape(-1, t)
    cols = np.searchsorted(col_to_item, flat)
    if len(flat) and not np.array_equal(col_to_item[np.minimum(cols, len(col_to_item) - 1)], flat):
        raise ValueError(f"level {t} holds an item missing from level 1")
    order = np.lexsort(cols[:, ::-1].T) if len(cols) else np.empty(0, dtype=np.int64)
    return cols[order].astype(np.int32), sub["count"].to_numpy()[order].astype(np.int64)


def load_lattice(path: Path) -> Lattice:
    """Read a lattice dump; build the per-level keys and free flags."""
    import polars as pl

    meta = json.loads(Path(str(path).removesuffix(".parquet") + ".json").read_text())
    df = pl.read_parquet(path).with_columns(pl.col("itemset").list.len().alias("k"))
    col_to_item = np.sort(df.filter(pl.col("k") == 1)["itemset"].explode(empty_as_null=True).to_numpy().astype(np.int64))
    n_cols = len(col_to_item)
    lat = Lattice(meta, int(meta["n_rows"]), int(meta["min_count"]), n_cols, col_to_item, {}, {}, {}, {})
    for t in range(1, int(df["k"].max()) + 1):
        items, counts = _level_arrays(df, t, col_to_item)
        if len(items) == 0:
            break
        if (counts < lat.min_count).any():
            raise ValueError(f"level {t} holds an itemset below min_count")
        lat.items[t], lat.counts[t] = items, counts
        if t == 1:
            lat.keys[1] = np.arange(n_cols, dtype=np.int64)
            lat.free[1] = counts < lat.n_rows
            continue
        prefix = lat.lookup(items[:, :-1])
        if (prefix < 0).any():
            raise ValueError(f"level {t} holds an itemset whose prefix is not frequent")
        keys = prefix * n_cols + items[:, -1]
        if len(keys) > 1 and not (np.diff(keys) > 0).all():
            raise ValueError(f"level {t} keys are not strictly ascending")
        lat.keys[t] = keys
        nonfree = np.zeros(len(items), dtype=bool)
        for d in range(t):
            idx = lat.lookup(np.delete(items, d, axis=1))
            if (idx < 0).any():
                raise ValueError(f"level {t} holds an itemset with an infrequent subset")
            nonfree |= lat.counts[t - 1][idx] == counts
        lat.free[t] = ~nonfree
    return lat


def _parents(lat: Lattice, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Row ids in level k-1 of each level-k itemset's two prefix-parents."""
    items = lat.items[k]
    a = lat.keys[k] // lat.n_cols
    b = lat.lookup(np.delete(items, k - 2, axis=1))
    return a, b


class BruteCounter:
    """Exact supports of arbitrary itemsets from the dataset's rows (bitsets on the CPU)."""

    def __init__(self, dataset: str, lat: Lattice):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import polars as pl
        from consolidation_run import _load

        df, n_rows, _ = _load(dataset)
        if n_rows != lat.n_rows:
            raise ValueError(f"{dataset} has {n_rows} rows, the lattice {lat.n_rows}")
        ex = df.with_row_index("row").explode("items", empty_as_null=True).drop_nulls("items")
        rows = ex["row"].to_numpy().astype(np.int64)
        item = ex["items"].to_numpy().astype(np.int64)
        col = np.searchsorted(lat.col_to_item, item)
        keep = (col < lat.n_cols) & (lat.col_to_item[np.minimum(col, lat.n_cols - 1)] == item)
        rows, col = rows[keep], col[keep]
        self.n_words = (n_rows + 63) // 64
        bits = np.zeros(lat.n_cols * self.n_words, dtype=np.uint64)
        np.bitwise_or.at(bits, col * self.n_words + rows // 64, np.left_shift(np.uint64(1), (rows % 64).astype(np.uint64)))
        self.bits = bits.reshape(lat.n_cols, self.n_words)
        del pl

    def count(self, sets: np.ndarray) -> np.ndarray:
        out = np.empty(len(sets), dtype=np.int64)
        step = max(1, (1 << 26) // (8 * self.n_words))
        for s in range(0, len(sets), step):
            part = sets[s : s + step]
            acc = self.bits[part[:, 0]].copy()
            for c in range(1, part.shape[1]):
                acc &= self.bits[part[:, c]]
            out[s : s + step] = np.bitwise_count(acc).sum(axis=1, dtype=np.int64)
        return out


def _segments(groups: np.ndarray, sizes: np.ndarray, pairs: np.ndarray):
    """Batches of (group, first pair, end pair) segments, at most BATCH pairs each.

    A group above BATCH is cut at multiples of TILE suffix positions of j, so
    no tile-pair straddles two segments.
    """
    batch: list[tuple[int, int, int]] = []
    filled = 0
    for g in groups:
        g = int(g)
        if pairs[g] <= BATCH:
            segs = [(g, 0, int(pairs[g]))]
        else:
            segs = []
            j0 = 0
            s = int(sizes[g])
            while j0 < s:
                j1 = j0 + TILE
                while j1 < s and (j1 + TILE) * (j1 + TILE - 1) // 2 - j0 * (j0 - 1) // 2 <= BATCH:
                    j1 += TILE
                j1 = min(j1, s)
                segs.append((g, j0 * (j0 - 1) // 2 if j0 else 0, j1 * (j1 - 1) // 2))
                j0 = j1
        for seg in segs:
            n = seg[2] - seg[1]
            if batch and filled + n > BATCH:
                yield batch
                batch, filled = [], 0
            batch.append(seg)
            filled += n
    if batch:
        yield batch


def _expand(batch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(group, i, j) of every pair in the batch's segments, in the kernels' j-major order."""
    seg = np.array(batch, dtype=np.int64)
    lens = seg[:, 2] - seg[:, 1]
    starts = np.zeros(len(seg), dtype=np.int64)
    np.cumsum(lens[:-1], out=starts[1:])
    total = int(lens.sum())
    g = np.repeat(seg[:, 0], lens)
    t = np.arange(total, dtype=np.int64) - np.repeat(starts - seg[:, 1], lens)
    j = np.floor((1.0 + np.sqrt(1.0 + 8.0 * t.astype(np.float64))) / 2.0).astype(np.int64)
    j = np.where(j * (j - 1) // 2 > t, j - 1, j)
    j = np.where((j + 1) * j // 2 <= t, j + 1, j)
    i = t - j * (j - 1) // 2
    return g, i, j


def _systematic_sample(pairs: np.ndarray, fraction: float, rng) -> np.ndarray:
    """Group ids drawn with inclusion probability ``fraction`` each, spread over sizes."""
    order = np.argsort(pairs, kind="stable")
    step = 1.0 / fraction
    picks = np.floor(rng.uniform(0.0, step) + step * np.arange(int(math.ceil(len(order) / step)) + 1))
    picks = picks[picks < len(order)].astype(np.int64)
    return np.sort(order[picks])


def classify_level(lat: Lattice, k: int, base: np.ndarray, in_free_prev: np.ndarray, *,
                   tiled_min: int, sample: float | None, rng, brute: BruteCounter | None) -> dict:
    """Generate level k from ``base`` (row ids into level k-1) as the GPU route does; label it."""
    from et_miner.gpu.kernels.k3plus import build_k3plus_groups_from_flat

    t0 = time.perf_counter()
    prev_counts, prev_free = lat.counts[k - 1], lat.free[k - 1]
    if k == 2:
        n = len(base)
        prefix = np.empty((1, 0), dtype=np.int32)
        so = np.array([0, n], dtype=np.int64)
        suffixes = lat.items[1][base, 0].astype(np.int64)
        slot_row = base.astype(np.int64)
        cumulative = np.array([0, n * (n - 1) // 2], dtype=np.int64)
    else:
        groups = build_k3plus_groups_from_flat(np.ascontiguousarray(lat.items[k - 1][base]), with_src_rows=True)
        if groups is None:
            return {"k": k, "base": int(len(base)), "generated": 0}
        prefix = np.asarray(groups.prefix_items, dtype=np.int32).reshape(-1, k - 2)
        so = np.asarray(groups.suffix_offsets, dtype=np.int64)
        suffixes = np.asarray(groups.suffixes, dtype=np.int64)
        slot_row = base[np.asarray(groups.suffix_src_rows, dtype=np.int64)].astype(np.int64)
        cumulative = np.asarray(groups.cumulative_pairs, dtype=np.int64)
    sizes = np.diff(so)
    pairs = np.diff(cumulative)
    n_groups = len(sizes)
    generated = int(cumulative[-1])
    tiled_g = pairs >= tiled_min
    row = {"k": k, "base": int(len(base)), "prev_level": int(len(lat.items[k - 1])), "generated": generated,
           "groups": n_groups, "groups_tiled": int(tiled_g.sum()), "candidates_tiled": int(pairs[tiled_g].sum())}

    if k == 2 and (sample is not None or generated > K2_ENUMERATE_MAX):
        nonfree_items = int((~prev_free[base]).sum())
        n_i = nonfree_items * (len(base) - nonfree_items) + nonfree_items * (nonfree_items - 1) // 2
        row.update({"classified": generated, "analytic": True, "P": 0, "I": n_i, "C": generated - n_i})
        return row

    chosen = np.arange(n_groups) if sample is None else _systematic_sample(pairs, sample, rng)
    row["groups_classified"] = int(len(chosen))
    slot_group = np.repeat(np.arange(n_groups, dtype=np.int64), sizes)

    # Prefix-drop subsets Y_d = (prefix - p_d) + s_i + s_j: the (prefix - p_d) row id
    # per group, then the (prefix - p_d) + s_i row id per suffix slot.
    drop_ids, drop_ok = [], []
    for d in range(k - 2):
        q = np.delete(prefix, d, axis=1)
        qid = lat.lookup(q) if q.shape[1] else np.zeros(n_groups, dtype=np.int64)
        rid, rok = lat.step(k - 2, np.maximum(qid, 0)[slot_group], suffixes, (qid >= 0)[slot_group])
        drop_ids.append(rid)
        drop_ok.append(rok)

    # Tile-pairs of the classified groups: global index = tile_base[g] + offset(ta, tb).
    nt = (sizes + TILE - 1) // TILE
    tp = np.where(sizes >= 2, nt * (nt + 1) // 2, 0)
    tile_base = np.full(n_groups, -1, dtype=np.int64)
    tp_chosen = tp[chosen]
    tile_base[chosen] = np.concatenate([[0], np.cumsum(tp_chosen)[:-1]]) if len(chosen) else []
    n_tp = int(tp_chosen.sum())
    tile_tot = np.zeros(n_tp)
    tile_p = np.zeros(n_tp)
    tile_pi = np.zeros(n_tp)
    slot_valid = np.zeros(len(suffixes))

    agg = {key: 0 for key in ("classified", "P", "I", "C", "frequent", "I_frequent", "missing_free",
                              "P_frequent", "I_infrequent", "I_count_mismatch", "missing_vs_PI",
                              "brute_freq_mismatch", "brute_count_mismatch", "brute_P_frequent",
                              "brute_I_mismatch")}
    by_kernel = {name: {"candidates": 0, "P": 0, "I": 0, "C": 0, "frequent": 0} for name in ("tiled", "percand")}
    per_group = np.zeros((n_groups, 4))  # classified, P, I, frequent (for the sample's error)

    for batch in _segments(chosen, sizes, pairs):
        g, i, j = _expand(batch)
        si, sj = so[g] + i, so[g] + j
        s_j = suffixes[sj]
        a, b = slot_row[si], slot_row[sj]
        minc = np.minimum(prev_counts[a], prev_counts[b])
        nonfree = ~prev_free[a] | ~prev_free[b]
        in_free = in_free_prev[a] & in_free_prev[b]
        found = np.ones(len(g), dtype=bool)
        for d in range(k - 2):
            pos, hit = lat.step(k - 1, drop_ids[d][si], s_j, drop_ok[d][si])
            found &= hit
            minc = np.where(hit, np.minimum(minc, prev_counts[pos]), minc)
            nonfree |= hit & ~prev_free[pos]
            in_free &= hit & in_free_prev[pos]
        p_mask = ~found
        i_mask = found & nonfree
        pos, freq = lat.step(k, a, s_j, np.ones(len(g), dtype=bool))
        true_count = lat.counts[k][pos] if k in lat.counts else np.zeros(len(g), dtype=np.int64)

        agg["classified"] += len(g)
        agg["P"] += int(p_mask.sum())
        agg["I"] += int(i_mask.sum())
        agg["C"] += int((found & ~nonfree).sum())
        agg["frequent"] += int(freq.sum())
        agg["I_frequent"] += int((i_mask & freq).sum())
        agg["missing_free"] += int((~in_free).sum())
        agg["P_frequent"] += int((p_mask & freq).sum())
        agg["I_infrequent"] += int((i_mask & ~freq).sum())
        agg["I_count_mismatch"] += int((i_mask & freq & (minc != true_count)).sum())
        agg["missing_vs_PI"] += int(((~in_free) != (p_mask | i_mask)).sum())
        tiled_c = tiled_g[g]
        for name, m in (("tiled", tiled_c), ("percand", ~tiled_c)):
            by_kernel[name]["candidates"] += int(m.sum())
            by_kernel[name]["P"] += int((m & p_mask).sum())
            by_kernel[name]["I"] += int((m & i_mask).sum())
            by_kernel[name]["C"] += int((m & found & ~nonfree).sum())
            by_kernel[name]["frequent"] += int((m & freq).sum())

        ta, tb = i // TILE, j // TILE
        ntg = nt[g]
        tidx = tile_base[g] + ta * ntg - ta * (ta - 1) // 2 + (tb - ta)
        tile_tot += np.bincount(tidx, minlength=n_tp)
        tile_p += np.bincount(tidx, weights=p_mask, minlength=n_tp)
        tile_pi += np.bincount(tidx, weights=p_mask | i_mask, minlength=n_tp)
        not_p = (~p_mask).astype(np.float64)
        slot_valid += np.bincount(si, weights=not_p, minlength=len(suffixes))
        slot_valid += np.bincount(sj, weights=not_p, minlength=len(suffixes))
        per_group[:, 0] += np.bincount(g, minlength=n_groups)
        per_group[:, 1] += np.bincount(g, weights=p_mask, minlength=n_groups)
        per_group[:, 2] += np.bincount(g, weights=i_mask, minlength=n_groups)
        per_group[:, 3] += np.bincount(g, weights=freq, minlength=n_groups)

        if brute is not None:
            sets = np.concatenate([prefix[g], suffixes[si][:, None], s_j[:, None]], axis=1)
            exact = brute.count(sets)
            bfreq = exact >= lat.min_count
            agg["brute_freq_mismatch"] += int((bfreq != freq).sum())
            agg["brute_count_mismatch"] += int((freq & (exact != true_count)).sum())
            agg["brute_P_frequent"] += int((p_mask & bfreq).sum())
            agg["brute_I_mismatch"] += int((i_mask & (exact != minc)).sum())

    tile_group = np.repeat(chosen, tp_chosen)
    full_p = (tile_tot > 0) & (tile_p == tile_tot)
    full_pi = (tile_tot > 0) & (tile_pi == tile_tot)
    tiles = {}
    for name, m in (("all", np.ones(n_tp, dtype=bool)), ("tiled", tiled_g[tile_group]),
                    ("percand", ~tiled_g[tile_group])):
        tiles[name] = {
            "tilepairs": int(m.sum()),
            "tilepairs_P": int((m & full_p).sum()),
            "tilepairs_PI": int((m & full_pi).sum()),
            "candidates_in_P_tiles": int(tile_tot[m & full_p].sum()),
            "candidates_in_PI_tiles": int(tile_tot[m & full_pi].sum()),
        }
    valid_g = np.bincount(slot_group, weights=slot_valid > 0, minlength=n_groups).astype(np.int64)
    kept = valid_g[chosen] * (valid_g[chosen] - 1) // 2
    suffix_recoverable = int((pairs[chosen] - kept).sum())

    row.update(agg)
    row["by_kernel"] = by_kernel
    row["tiles"] = tiles
    row["suffix_prune_recoverable_P"] = suffix_recoverable
    row["partition_ok"] = agg["P"] + agg["I"] + agg["C"] == agg["classified"]
    if sample is None:
        row["enumerated_all"] = agg["classified"] == generated
    else:
        row["sample_fraction"] = sample
        pg = per_group[chosen]
        for idx, name in ((1, "P"), (2, "I"), (3, "frequent")):
            share = pg[:, idx].sum() / max(pg[:, 0].sum(), 1)
            resid = pg[:, idx] - share * pg[:, 0]
            se = math.sqrt((1 - sample) * float((resid ** 2).sum())) / max(pg[:, 0].sum(), 1)
            row[f"share_{name}"] = share
            row[f"share_{name}_se"] = se
        if k == 3:
            row["population_not_P"] = _k3_not_prunable(lat, base)
            row["population_P"] = generated - row["population_not_P"]
    row["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return row


def _k3_not_prunable(lat: Lattice, base: np.ndarray) -> int:
    """Exact number of K=3 candidates whose third pair is frequent, over the whole level.

    A K=3 candidate (a; b, c) is not prunable iff (b, c) is a frequent pair, so
    the count is the number of base pairs (a, b), (a, c) under every frequent
    pair (b, c): the popcount of the two lower-neighbour bitsets.
    """
    pairs = lat.items[2]
    words = (lat.n_cols + 63) // 64
    lower = np.zeros(lat.n_cols * words, dtype=np.uint64)
    a, b = pairs[base, 0].astype(np.int64), pairs[base, 1].astype(np.int64)
    np.bitwise_or.at(lower, b * words + a // 64, np.left_shift(np.uint64(1), (a % 64).astype(np.uint64)))
    lower = lower.reshape(lat.n_cols, words)
    total = 0
    for s in range(0, len(pairs), 1 << 15):
        part = pairs[s : s + (1 << 15)]
        total += int(np.bitwise_count(lower[part[:, 0]] & lower[part[:, 1]]).sum(dtype=np.int64))
    return total


def replay(lat: Lattice, *, free: bool, sample: float | None, seed: int, brute: BruteCounter | None,
           free_dump: Lattice | None = None, log=print):
    """Every level the GPU route generates, in order, as classify_level rows."""
    from et_miner.gpu.row_split_chunks import TILED_MIN_GROUP_PAIRS

    rng = np.random.default_rng(seed)
    max_length = lat.meta.get("max_length") or lat.n_cols
    full_prev = np.ones(len(lat.items[1]), dtype=bool)
    base_mask = lat.free[1].copy() if free else np.ones(len(lat.items[1]), dtype=bool)
    emitted_prev = int(base_mask.sum())
    k = 2
    while k <= min(max_length, lat.n_cols) and emitted_prev >= k:
        base = np.nonzero(base_mask)[0]
        in_free_prev = base_mask if free else lat.free[k - 1]
        tiled_min = TILED_MIN_GROUP_PAIRS.get(k, TILED_MIN_GROUP_PAIRS[max(TILED_MIN_GROUP_PAIRS)])
        row = classify_level(lat, k, base, in_free_prev, tiled_min=tiled_min, sample=sample, rng=rng, brute=brute)
        row["regime"] = "free" if free else "full"
        if k in lat.items:
            pa, pb = _parents(lat, k)
            generated_freq = base_mask[pa] & base_mask[pb]
            if free:
                literal_nonfree = np.zeros(len(pa), dtype=bool)
                for d in range(k):
                    idx = lat.lookup(np.delete(lat.items[k], d, axis=1))
                    literal_nonfree |= full_prev[idx] & (lat.counts[k - 1][idx] == lat.counts[k])
                emitted = generated_freq & ~literal_nonfree
                row["free_replay_equals_true_free"] = bool(np.array_equal(emitted, lat.free[k]))
                if free_dump is not None:
                    dumped = free_dump.items.get(k, np.empty((0, k), dtype=np.int32))
                    row["free_replay_equals_gpu_dump"] = bool(np.array_equal(lat.items[k][emitted], dumped))
            else:
                emitted = generated_freq
            row["generated_frequent"] = int(generated_freq.sum())
            row["emitted"] = int(emitted.sum())
            if sample is None and "frequent" in row:
                row["enumerated_frequent_matches"] = row["frequent"] == row["generated_frequent"]
            full_prev, base_mask = generated_freq, emitted
        else:
            row["generated_frequent"] = row["emitted"] = 0
            full_prev = base_mask = np.zeros(0, dtype=bool)
        log(json.dumps({key: row[key] for key in ("k", "generated", "emitted", "P", "I", "C", "elapsed_s")
                        if key in row}))
        yield row
        emitted_prev = row["emitted"]
        if emitted_prev == 0:
            break
        k += 1


COUNT_PHASES = ("count_percand", "count_tiled", "count_fused")


def _med(values: list[float]) -> str:
    import statistics

    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.median(values):.3f} [{min(values):.3f}, {max(values):.3f}]"


def _pct(part: float, whole: float) -> str:
    return f"{100 * part / whole:.1f}" if whole else "—"


def _savings(levels: dict[int, dict], reps: list[dict]) -> dict:
    """Σ K>=3 counting seconds (medians over reps) and the share each prune granularity skips.

    Model: per-candidate kernel time is proportional to its candidates, tiled
    (and fused) kernel time to its tile-pairs; a sampled level uses its sample's
    shares. Skipping at candidate granularity in tiled groups is an upper bound
    (the tiled kernel stages whole tiles).
    """
    import statistics

    est = {key: 0.0 for key in ("count", "P_candidate", "P_tile", "P_suffix", "PI_candidate", "PI_tile")}
    for k, w in levels.items():
        if k < 3 or "by_kernel" not in w or not reps:
            continue
        med = {ph: statistics.median([r["timings"]["level_split"].get(str(k), {}).get(ph, 0.0) for r in reps])
               for ph in COUNT_PHASES}
        pc, tl, tiles = w["by_kernel"]["percand"], w["by_kernel"]["tiled"], w["tiles"]["tiled"]
        t_pc, t_tl = med["count_percand"], med["count_tiled"] + med["count_fused"]

        def share(a, b):
            return a / b if b else 0.0

        est["count"] += t_pc + t_tl
        est["P_candidate"] += t_pc * share(pc["P"], pc["candidates"]) + t_tl * share(tl["P"], tl["candidates"])
        est["P_tile"] += t_pc * share(pc["P"], pc["candidates"]) + t_tl * share(tiles["tilepairs_P"], tiles["tilepairs"])
        est["P_suffix"] += (t_pc + t_tl) * share(w["suffix_prune_recoverable_P"], w["classified"])
        est["PI_candidate"] += (t_pc * share(pc["P"] + pc["I"], pc["candidates"])
                                + t_tl * share(tl["P"] + tl["I"], tl["candidates"]))
        est["PI_tile"] += (t_pc * share(pc["P"] + pc["I"], pc["candidates"])
                           + t_tl * share(tiles["tilepairs_PI"], tiles["tilepairs"]))
    return est


def report(raws: list[Path], waste: Path) -> str:
    """Markdown tables: a summary per regime, then labels and time split per level."""
    runs: dict[tuple[str, str], list[dict]] = {}
    for raw in raws:
        for line in raw.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") != "ok" or "level_split" not in r.get("timings", {}):
                continue
            cfg = r["config"]
            regime = "free" if cfg.get("prune_equal_support") else "full"
            runs.setdefault((cfg["base_id"].split("-", 1)[0], regime), []).append(r)
    rows: dict[tuple[str, str], dict[int, dict]] = {}
    for line in waste.read_text().splitlines():
        w = json.loads(line)
        rows.setdefault((w["workload"], w["regime"]), {})[w["k"]] = w

    out = ["| workload | regime | reps | wall s | Σ K≥3 levels s | Σ K≥3 counting s | K≥3 generated | P % | I % "
           "| P, candidate | P, tile | P, suffix | P∪I, candidate | P∪I, tile |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(rows):
        levels, reps = rows[key], runs.get(key, [])
        gen = sum(w["generated"] for k, w in levels.items() if k >= 3)
        n = sum(w.get("classified", 0) for k, w in levels.items() if k >= 3)
        p = sum(w.get("P", 0) for k, w in levels.items() if k >= 3)
        i = sum(w.get("I", 0) for k, w in levels.items() if k >= 3)
        sums = [sum(v["level"] for kk, v in r["timings"]["level_split"].items() if int(kk) >= 3) for r in reps]
        est = _savings(levels, reps)
        out.append(
            f"| {key[0]} | {key[1]} | {len(reps)} | {_med([r['wall_s'] for r in reps])} | {_med(sums)} "
            f"| {est['count']:.3f} | {gen:,} | {_pct(p, n)} | {_pct(i, n)} | {est['P_candidate']:.3f} "
            f"| {est['P_tile']:.3f} | {est['P_suffix']:.3f} | {est['PI_candidate']:.3f} | {est['PI_tile']:.3f} |"
        )
    out.append("")

    for key in sorted(rows):
        workload, regime = key
        levels = rows[key]
        reps = runs.get(key, [])
        sampled = any("sample_fraction" in w for w in levels.values())
        out.append(f"### {workload}, {regime}{' (K=3 sampled)' if sampled else ''}\n")
        out.append("| K | generated | frequent | emitted | P | I | C | P % | I % | P in fully-P tiles % (tiled groups) "
                   "| P per suffix % | level s | counting s | rest s |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k in sorted(levels):
            w = levels[k]
            split = [r["timings"]["level_split"].get(str(k)) for r in reps]
            split = [x for x in split if x]
            n = w.get("classified", 0)
            p, i, c = w.get("P", 0), w.get("I", 0), w.get("C", 0)
            tiles = w.get("tiles", {}).get("tiled", {})
            tiled_p = w.get("by_kernel", {}).get("tiled", {}).get("P", 0)
            out.append(
                f"| {k} | {w['generated']:,} | {w.get('generated_frequent', 0):,} | {w.get('emitted', 0):,} "
                f"| {p:,} | {i:,} | {c:,} | {_pct(p, n)} | {_pct(i, n)} "
                f"| {_pct(tiles.get('candidates_in_P_tiles', 0), tiled_p)} "
                f"| {_pct(w.get('suffix_prune_recoverable_P', 0), p)} | {_med([x['level'] for x in split])} "
                f"| {_med([sum(x[ph] for ph in COUNT_PHASES) for x in split])} "
                f"| {_med([x['level'] - sum(x[ph] for ph in COUNT_PHASES) for x in split])} |"
            )
        out.append("")
        if reps:
            phases = ("group_build", "group_upload", "budget", *COUNT_PHASES, "reduce", "filter", "decode", "sort",
                      "free_prune", "other")
            out.append("Σ K≥3 time split, s, median [min, max] over reps:\n")
            out.append("| " + " | ".join(phases) + " |")
            out.append("|" + "---|" * len(phases))
            cells = [_med([sum(v[ph] for kk, v in r["timings"]["level_split"].items() if int(kk) >= 3) for r in reps])
                     for ph in phases]
            out.append("| " + " | ".join(cells) + " |")
            out.append("")
    return "\n".join(out)


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--raw", type=Path, nargs="+", required=True)
    r.add_argument("--waste", type=Path, required=True)
    c = sub.add_parser("classify")
    c.add_argument("lattice", type=Path)
    c.add_argument("--workload", required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--free", action="store_true")
    c.add_argument("--gpu-result", type=Path)
    c.add_argument("--free-dump", type=Path)
    c.add_argument("--sample-groups", type=float)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--brute", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "report":
        print(report(args.raw, args.waste))
        return 0

    t0 = time.perf_counter()
    lat = load_lattice(args.lattice)
    free_dump = None
    if args.free_dump is not None:
        free_dump = Lattice(lat.meta, lat.n_rows, lat.min_count, lat.n_cols, lat.col_to_item, {}, {}, {}, {})
        import polars as pl

        df = pl.read_parquet(args.free_dump).with_columns(pl.col("itemset").list.len().alias("k"))
        for t in range(1, int(df["k"].max()) + 1):
            free_dump.items[t], free_dump.counts[t] = _level_arrays(df, t, lat.col_to_item)
    gpu_levels = {}
    if args.gpu_result is not None:
        gpu_levels = {lv["k"]: lv for lv in json.loads(args.gpu_result.read_text())["levels"]}
    brute = BruteCounter(lat.meta["dataset"], lat) if args.brute else None
    print(f"lattice: {sum(len(v) for v in lat.items.values()):,} itemsets to K={max(lat.items)}, "
          f"{lat.n_cols:,} items, min_count {lat.min_count:,} ({time.perf_counter() - t0:.1f}s)", file=sys.stderr)

    failures = []
    replayed = set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        for row in replay(lat, free=args.free, sample=args.sample_groups, seed=args.seed, brute=brute,
                          free_dump=free_dump, log=lambda s: print(s, file=sys.stderr)):
            row.update({"workload": args.workload, "dataset": lat.meta["dataset"], "n_rows": lat.n_rows,
                        "min_count": lat.min_count, "brute": bool(args.brute)})
            gpu = gpu_levels.get(row["k"])
            if gpu is not None:
                row["gpu_n_candidates"] = gpu["n_candidates"]
                row["gpu_n_frequent"] = gpu["n_frequent"]
                if gpu["n_candidates"] != row["generated"] or gpu["n_frequent"] != row["emitted"]:
                    failures.append(f"K={row['k']}: GPU {gpu['n_candidates']}/{gpu['n_frequent']} vs "
                                    f"replay {row['generated']}/{row['emitted']}")
            for key in ("P_frequent", "I_infrequent", "I_count_mismatch", "missing_vs_PI", "brute_freq_mismatch",
                        "brute_count_mismatch", "brute_P_frequent", "brute_I_mismatch"):
                if row.get(key):
                    failures.append(f"K={row['k']}: {key}={row[key]}")
            for key in ("partition_ok", "enumerated_all", "enumerated_frequent_matches",
                        "free_replay_equals_true_free", "free_replay_equals_gpu_dump"):
                if row.get(key) is False:
                    failures.append(f"K={row['k']}: {key} is False")
            f.write(json.dumps(row) + "\n")
            f.flush()
            replayed.add(row["k"])
    missing = sorted(k for k in gpu_levels if k >= 2 and k not in replayed)
    if missing:
        failures.append(f"GPU levels not replayed: {missing}")
    for msg in failures:
        print(f"CHECK FAILED {args.workload}: {msg}", file=sys.stderr)
    print(f"done in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
