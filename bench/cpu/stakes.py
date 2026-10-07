"""Offline stakes for the CPU-tier levers L1-L5: exact level-by-level mining with the new representations.

Mines one workload level by level with the structures the levers would
introduce, times every variant of every step, and checks exactness twice: all
variants of a step must agree, and the assembled lattice must carry the same
signature (itemset hash, count sum) as the campaign's rows for the workload.
Prints one JSON line per measured step and one summary line per workload.

Levers and variants:
    L2  CSR build      explode_join (_build_csr_from_transactions), numpy_map (Polars
                       explode + np.searchsorted onto the frequent items)
    L1  K=2            gram_sparse (scipy M.T @ M, upper triangle >= min_count),
                       gram_dense (the same product densified), polars_pairs
                       (self-join of the exploded table on the row, group_by pair)
    L4  K>=3 gen       array (lexsorted int32 (n, k) arrays, vectorised prefix-group
                       join, subset test by binary search on packed keys),
                       pyref (core.candidates._generate_candidates on the column names)
    L3  K>=3 count     rust_simd (count_itemsets_simd, as the sparse route calls it),
                       bitvec (per-candidate AND of k uint64 columns + popcount),
                       proj (per prefix group: rows = prefix tidset, one scipy Gram of
                       the suffix columns), gbitvec (per group: prefix AND once, then
                       pair popcounts on the prefix's non-zero words)
    L5  overheads      csr_len_filter, int64 casts, threshold mask on a count array
    ref rust_full      et_miner_rust.apriori_from_csr (the existing all-Rust miner)

Usage:
    uv run python bench/cpu/stakes.py --workload or003 --threads 1 \
        --baseline bench/results/2026-10-07-cpu-baseline/raw.jsonl --out stakes.jsonl

Options:
    --workload   an id of bench/cpu/matrix.py WORKLOADS
    --threads    thread budget (env pools, Rust n_threads); numpy and scipy run single-threaded
    --baseline   campaign raw.jsonl; the stake lattice must match its signature
    --out        jsonl file to append the rows to
    --skip       comma-separated variants to leave out (e.g. pyref,polars_pairs)
"""

from __future__ import annotations

import os
import sys


def _argv_threads() -> str:
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--threads="):
            return a.split("=", 1)[1]
    return "1"


# Thread pools read their env at import, so it is set before numpy/polars load.
for _k in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = _argv_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import scipy.sparse as sp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from child_run import result_signatures  # noqa: E402
from consolidation_run import _load  # noqa: E402
from cpu.matrix import WORKLOADS  # noqa: E402

from et_miner.core.result import _min_count  # noqa: E402

LUT16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)
#: Bytes of AND results materialised at once by the bitvector counters.
CHUNK_BYTES = 64 << 20
#: The Polars self-join materialises one row per co-occurring pair occurrence (≈ 16 B each).
POLARS_PAIRS_MAX = 150_000_000


class Peak(threading.Thread):
    """Samples RSS every 5 ms; ``delta_mb`` is the peak above the starting RSS."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        import psutil

        self._p = psutil.Process()
        self.start_mb = self.peak_mb = self._p.memory_info().rss / 2**20
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            self.peak_mb = max(self.peak_mb, self._p.memory_info().rss / 2**20)
            self._halt.wait(0.005)

    def stop(self) -> float:
        self._halt.set()
        self.join()
        self.peak_mb = max(self.peak_mb, self._p.memory_info().rss / 2**20)
        return round(self.peak_mb - self.start_mb, 1)


class Recorder:
    def __init__(self, workload: str, threads: int, out: Path | None, skip: set[str]) -> None:
        self.base = {"workload": workload, "threads": threads}
        self.out = out
        self.skip = skip
        self.rows: list[dict] = []

    def run(self, lever: str, variant: str, fn, *args, k: int | None = None, **extra):
        """Times ``fn(*args)``; records seconds and the RSS peak above the start. None when skipped."""
        if variant in self.skip:
            return None
        peak = Peak()
        peak.start()
        t0 = time.perf_counter()
        out = fn(*args)
        dt = time.perf_counter() - t0
        rss = peak.stop()
        self.emit({"lever": lever, "variant": variant, "k": k, "s": round(dt, 6), "rss_delta_mb": rss, **extra})
        return out

    def emit(self, row: dict) -> None:
        row = {**self.base, **row}
        self.rows.append(row)
        line = json.dumps(row)
        print(line, flush=True)
        if self.out is not None:
            with self.out.open("a") as f:
                f.write(line + "\n")


# ── L2: CSR build ───────────────────────────────────────────────────────────


def csr_explode_join(df: pl.DataFrame, min_support: float):
    """The direct CSR path the GPU route uses (explode + inner join onto the frequent items)."""
    from et_miner.core.matrix import _build_csr_from_transactions

    csr, col_to_item, _ = _build_csr_from_transactions(df.lazy(), min_support, "items")
    return csr, np.array([col_to_item[i] for i in range(len(col_to_item))], dtype=np.int64)


def csr_numpy_map(df: pl.DataFrame, min_count: int):
    """Polars explode for the flat values, then np.searchsorted onto the sorted frequent item ids.

    Steps: list lengths and the flat item column from Polars; K=1 counts by
    np.unique; keep items with count >= min_count; map every flat value to its
    column by binary search; row ids by repeating the row index; indptr from a
    bincount of the kept rows. Rows hold sorted unique items in every workload
    here, so the per-row column order is already ascending (asserted).
    """
    s = df.get_column("items")
    lens = s.list.len().fill_null(0).to_numpy()
    flat = s.explode().drop_nulls().to_numpy()
    ids, counts = np.unique(flat, return_counts=True)
    freq = ids[counts >= min_count]
    col = np.searchsorted(freq, flat)
    col_c = np.minimum(col, len(freq) - 1)
    keep = freq[col_c] == flat
    rows = np.repeat(np.arange(len(lens), dtype=np.int64), lens)[keep]
    indices = col[keep].astype(np.int32)
    indptr = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=len(lens)), out=indptr[1:])
    csr = sp.csr_matrix((np.ones(len(indices), dtype=np.int32), indices, indptr), shape=(len(lens), len(freq)))
    return csr, freq.astype(np.int64)


# ── L1: K=2 ─────────────────────────────────────────────────────────────────


def _pairs_sorted(i, j, c):
    o = np.lexsort((j, i))
    return np.stack([i[o], j[o]], axis=1).astype(np.int32), c[o].astype(np.int64)


def k2_gram_sparse(csr, min_count: int):
    g = sp.triu(csr.T @ csr, k=1, format="coo")
    m = g.data >= min_count
    return _pairs_sorted(g.row[m], g.col[m], g.data[m])


def k2_gram_dense(csr, min_count: int):
    g = (csr.T @ csr).toarray()
    i, j = np.nonzero(g >= min_count)
    m = i < j
    i, j = i[m], j[m]
    return _pairs_sorted(i, j, g[i, j])


def k2_polars_pairs(csr, min_count: int):
    rows = np.repeat(np.arange(csr.shape[0], dtype=np.int64), np.diff(csr.indptr))
    t = pl.DataFrame({"r": rows, "a": csr.indices.astype(np.int32)}).lazy()
    out = (
        t.join(t.select("r", pl.col("a").alias("b")), on="r")
        .filter(pl.col("a") < pl.col("b"))
        .group_by("a", "b")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= min_count)
        .collect()
    )
    return _pairs_sorted(out["a"].to_numpy(), out["b"].to_numpy(), out["n"].to_numpy())


# ── L4: K>=3 candidate generation ───────────────────────────────────────────


def _pack(a: np.ndarray, base: int) -> np.ndarray | None:
    """Mixed-radix int64 keys preserving lexicographic order, or None when they would overflow."""
    if a.shape[1] * np.log2(max(base, 2)) >= 62:
        return None
    key = np.zeros(len(a), dtype=np.int64)
    for c in range(a.shape[1]):
        key = key * base + a[:, c]
    return key


def _rows_in(sub: np.ndarray, prev: np.ndarray, prev_key: np.ndarray | None, base: int) -> np.ndarray:
    """Boolean mask: which rows of ``sub`` occur in the lexsorted ``prev``."""
    if prev_key is not None:
        key = _pack(sub, base)
        pos = np.searchsorted(prev_key, key)
        pos_c = np.minimum(pos, len(prev_key) - 1)
        return prev_key[pos_c] == key
    v = np.dtype((np.void, prev.dtype.itemsize * prev.shape[1]))
    return np.isin(np.ascontiguousarray(sub).view(v).ravel(), np.ascontiguousarray(prev).view(v).ravel())


def gen_array(prev: np.ndarray, k: int, base: int) -> np.ndarray:
    """K-candidates from the lexsorted frequent (k-1)-itemsets, with the full subset test.

    Steps: prefix groups are runs of equal first k-2 columns; every member pairs
    with each later member of its run (vectorised with repeat/cumsum); a
    candidate keeps prefix + both last items. The two subsets that drop one of
    the last two items are the parents themselves; the k-2 others are looked up
    in ``prev`` by binary search on packed keys (np.isin on row bytes when the
    keys would overflow int64). The output is lexsorted.
    """
    n = len(prev)
    if n < 2:
        return np.empty((0, k), dtype=np.int32)
    if k == 3:
        change = np.r_[True, prev[1:, 0] != prev[:-1, 0]]
    else:
        change = np.r_[True, np.any(prev[1:, : k - 2] != prev[:-1, : k - 2], axis=1)]
    starts = np.flatnonzero(change)
    sizes = np.diff(np.r_[starts, n])
    pos = np.arange(n) - np.repeat(starts, sizes)
    partners = np.repeat(sizes, sizes) - 1 - pos
    total = int(partners.sum())
    if total == 0:
        return np.empty((0, k), dtype=np.int32)
    left = np.repeat(np.arange(n), partners)
    off = np.arange(total) - np.repeat(np.cumsum(partners) - partners, partners)
    right = left + 1 + off
    cands = np.empty((total, k), dtype=np.int32)
    cands[:, : k - 1] = prev[left]
    cands[:, k - 1] = prev[right, k - 2]
    prev_key = _pack(prev, base)
    keep = np.ones(total, dtype=bool)
    for drop in range(k - 2):
        cols = [c for c in range(k) if c != drop]
        idx = np.flatnonzero(keep)
        keep[idx] = _rows_in(cands[idx][:, cols], prev, prev_key, base)
    return cands[keep]


def gen_pyref(prev: np.ndarray, k: int, names: list[str], name_idx: dict[str, int]) -> np.ndarray:
    from et_miner.core.candidates import _generate_candidates

    tuples = [tuple(names[c] for c in row) for row in prev.tolist()]
    t0 = time.perf_counter()
    out = _generate_candidates(tuples, k)
    dt = time.perf_counter() - t0
    arr = np.array([[name_idx[c] for c in t] for t in out], dtype=np.int32).reshape(-1, k)
    if len(arr):
        arr = arr[np.lexsort(arr.T[::-1])]
    return arr, dt


# ── L3: K>=3 counting ───────────────────────────────────────────────────────


def popcount_rows(a: np.ndarray, use_lut: bool) -> np.ndarray:
    if use_lut or not hasattr(np, "bitwise_count"):
        return LUT16[a.view(np.uint16)].sum(axis=1, dtype=np.int64)
    return np.bitwise_count(a).sum(axis=1, dtype=np.int64)


def build_bitvecs(csc, n_rows: int) -> np.ndarray:
    """(n_cols, ceil(n_rows/64)) uint64 column bitvectors from CSC."""
    n_cols = csc.shape[1]
    w = (n_rows + 63) // 64
    flat = np.zeros(n_cols * w, dtype=np.uint64)
    cols = np.repeat(np.arange(n_cols, dtype=np.int64), np.diff(csc.indptr))
    rows = csc.indices.astype(np.int64)
    np.bitwise_or.at(flat, cols * w + (rows >> 6), np.left_shift(np.uint64(1), (rows & 63).astype(np.uint64)))
    return flat.reshape(n_cols, w)


def count_rust(csr, cands: np.ndarray, n_threads: int):
    from et_miner_rust import count_itemsets_simd

    indptr = csr.indptr.astype(np.int64)
    indices = csr.indices.astype(np.int64)
    lists = cands.tolist()
    t0 = time.perf_counter()
    out = count_itemsets_simd(indptr, indices, csr.shape[0], csr.shape[1], lists, n_threads)
    return np.asarray(out, dtype=np.int64), time.perf_counter() - t0


def count_bitvec(bv: np.ndarray, cands: np.ndarray, use_lut: bool = False) -> np.ndarray:
    w = bv.shape[1]
    step = max(1, CHUNK_BYTES // (8 * w))
    out = np.empty(len(cands), dtype=np.int64)
    for s in range(0, len(cands), step):
        c = cands[s : s + step]
        acc = bv[c[:, 0]] & bv[c[:, 1]]
        for j in range(2, c.shape[1]):
            acc &= bv[c[:, j]]
        out[s : s + step] = popcount_rows(acc, use_lut)
    return out


def _groups(cands: np.ndarray, k: int):
    """(start, end) of each run of equal first k-2 columns of the lexsorted candidates."""
    n = len(cands)
    if k == 3:
        change = np.r_[True, cands[1:, 0] != cands[:-1, 0]]
    else:
        change = np.r_[True, np.any(cands[1:, : k - 2] != cands[:-1, : k - 2], axis=1)]
    starts = np.flatnonzero(change)
    return starts, np.r_[starts[1:], n]


def count_proj(csr, csc, cands: np.ndarray, k: int):
    """Per prefix group: restrict rows to the prefix tidset, one scipy Gram of the suffix columns.

    Returns (counts, stats): stats holds the group count and the time spent in
    groups by suffix-count bucket, for the dispatch crossover.
    """
    out = np.empty(len(cands), dtype=np.int64)
    starts, ends = _groups(cands, k)
    cache: dict[tuple, np.ndarray] = {}
    buckets: dict[str, list[float]] = {}
    for s, e in zip(starts.tolist(), ends.tolist()):
        t0 = time.perf_counter()
        prefix = tuple(cands[s, : k - 2].tolist())
        rows = cache.get(prefix)
        if rows is None:
            parent = cache.get(prefix[:-1]) if len(prefix) > 1 else None
            last = prefix[-1]
            col = csc.indices[csc.indptr[last] : csc.indptr[last + 1]]
            if len(prefix) == 1:
                rows = col
            else:
                if parent is None:
                    parent = col_tidset(csc, prefix[:-1])
                    cache[prefix[:-1]] = parent
                rows = np.intersect1d(parent, col, assume_unique=True)
            cache[prefix] = rows
        a = cands[s:e, k - 2]
        b = cands[s:e, k - 1]
        suffix = np.union1d(a, b)
        x = csr[rows][:, suffix]
        g = (x.T @ x).toarray()
        out[s:e] = g[np.searchsorted(suffix, a), np.searchsorted(suffix, b)]
        key = _bucket(len(suffix))
        buckets.setdefault(key, [0, 0.0, 0])
        buckets[key][0] += 1
        buckets[key][1] += time.perf_counter() - t0
        buckets[key][2] += e - s
    return out, {"groups": len(starts), "buckets": {b: [n, round(t, 4), c] for b, (n, t, c) in buckets.items()}}


def col_tidset(csc, items: tuple) -> np.ndarray:
    rows = csc.indices[csc.indptr[items[0]] : csc.indptr[items[0] + 1]]
    for it in items[1:]:
        rows = np.intersect1d(rows, csc.indices[csc.indptr[it] : csc.indptr[it + 1]], assume_unique=True)
    return rows


def count_gbitvec(bv: np.ndarray, cands: np.ndarray, k: int):
    """Per prefix group: AND the prefix once, keep its non-zero words, then popcount each pair."""
    out = np.empty(len(cands), dtype=np.int64)
    starts, ends = _groups(cands, k)
    buckets: dict[str, list] = {}
    for s, e in zip(starts.tolist(), ends.tolist()):
        t0 = time.perf_counter()
        pre = bv[cands[s, 0]].copy()
        for j in range(1, k - 2):
            pre &= bv[cands[s, j]]
        nz = np.flatnonzero(pre)
        a = cands[s:e, k - 2]
        b = cands[s:e, k - 1]
        suffix = np.union1d(a, b)
        sub = bv[suffix][:, nz] & pre[nz]
        ia, ib = np.searchsorted(suffix, a), np.searchsorted(suffix, b)
        step = max(1, CHUNK_BYTES // (8 * max(1, len(nz))))
        for c0 in range(0, e - s, step):
            c1 = min(c0 + step, e - s)
            out[s + c0 : s + c1] = popcount_rows(sub[ia[c0:c1]] & sub[ib[c0:c1]], False)
        key = _bucket(len(suffix))
        buckets.setdefault(key, [0, 0.0, 0])
        buckets[key][0] += 1
        buckets[key][1] += time.perf_counter() - t0
        buckets[key][2] += e - s
    return out, {"groups": len(starts), "buckets": {b: [n, round(t, 4), c] for b, (n, t, c) in buckets.items()}}


def _bucket(m: int) -> str:
    for hi in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        if m <= hi:
            return f"<={hi}"
    return ">512"


# ── driver ──────────────────────────────────────────────────────────────────


def _campaign_signature(raw: Path | None, workload: str) -> dict | None:
    if raw is None or not raw.exists():
        return None
    for line in raw.read_text().splitlines():
        r = json.loads(line)
        c = r.get("config", {})
        if r.get("status") == "ok" and c.get("route") == "F" and str(c.get("base_id", "")).startswith(workload + "-"):
            levels = {lv["k"]: lv.get("n_frequent") for lv in r.get("levels", []) if lv.get("n_frequent") is not None}
            return {"n_itemsets": r["n_itemsets"], "sum_counts": r["sum_counts"], "itemset_hash": r["itemset_hash"],
                    "levels": levels, "from": r["id"]}
    return None


def mine(workload: str, threads: int, rec: Recorder, raw: Path | None) -> dict:
    dataset, min_support, max_length = WORKLOADS[workload]
    df, n_rows, _ = _load(dataset)
    min_count = _min_count(min_support, n_rows)
    max_tx = int(df.select(pl.col("items").list.len().max()).item() or 0)

    # L2
    ref = rec.run("L2", "explode_join", csr_explode_join, df, min_support)
    csr, item_ids = rec.run("L2", "numpy_map", csr_numpy_map, df, min_count)
    assert csr.has_sorted_indices and np.all(np.diff(csr.indptr) >= 0)
    if ref is not None:
        rcsr, rids = ref
        rcsr.sort_indices()
        assert np.array_equal(rids, item_ids), "L2: frequent items differ"
        assert np.array_equal(rcsr.indptr, csr.indptr) and np.array_equal(rcsr.indices, csr.indices), "L2: CSR differs"
        assert rcsr.data.max(initial=1) == 1, "duplicate items in a transaction"
    n_cols = csr.shape[1]
    max_k = min(max_length or 10**9, max_tx, n_cols)
    csc = rec.run("L2", "tocsc", csr.tocsc)
    counts1 = rec.run("K1", "bincount", lambda: np.bincount(csr.indices, minlength=n_cols))
    lattice: list[tuple[np.ndarray, np.ndarray]] = [(np.arange(n_cols, dtype=np.int32)[:, None], counts1)]
    rec.emit({"lever": "K1", "variant": "result", "k": 1, "n_frequent": n_cols, "n_rows": n_rows,
              "min_count": min_count})

    # L1: K=2
    prev = None
    if max_k >= 2 and n_cols >= 2:
        lens = np.diff(csr.indptr)
        pair_rows = int((lens * (lens - 1) // 2).sum())
        rec.emit({"lever": "L1", "variant": "shape", "k": 2, "pair_rows": pair_rows,
                  "sum_len_sq": int((lens.astype(np.int64) ** 2).sum()), "nnz": int(csr.nnz)})
        if pair_rows > POLARS_PAIRS_MAX:
            rec.skip.add("polars_pairs")
            rec.emit({"lever": "L1", "variant": "polars_pairs", "k": 2, "skipped": f"pair_rows {pair_rows} > {POLARS_PAIRS_MAX}"})
        outs = {}
        for name, fn in (("gram_sparse", k2_gram_sparse), ("gram_dense", k2_gram_dense),
                         ("polars_pairs", k2_polars_pairs)):
            r = rec.run("L1", name, fn, csr, min_count, k=2, n_candidates=n_cols * (n_cols - 1) // 2)
            if r is not None:
                outs[name] = r
        base_name, (pairs, cnt) = next(iter(outs.items()))
        for name, (p2, c2) in outs.items():
            assert np.array_equal(p2, pairs) and np.array_equal(c2, cnt), f"L1: {name} differs from {base_name}"
        rec.emit({"lever": "L1", "variant": "result", "k": 2, "n_frequent": len(pairs),
                  "dense_bytes": n_cols * n_cols * 4})
        lattice.append((pairs, cnt))
        prev = pairs

    # Bitvectors for the K>=3 counters (built once).
    bv = None
    if max_k >= 3 and prev is not None and len(prev) >= 3:
        bv = rec.run("L3", "bitvec_build", build_bitvecs, csc, n_rows, words=(n_rows + 63) // 64,
                     bytes=n_cols * ((n_rows + 63) // 64) * 8)

    width = len(str(n_cols))
    names = [f"i_{i:0{width}d}" for i in range(n_cols)]
    name_idx = {nm: i for i, nm in enumerate(names)}
    k = 3
    while prev is not None and k <= max_k and len(prev) >= k - 1:
        cands = rec.run("L4", "array", gen_array, prev, k, n_cols, k=k, n_prev=len(prev))
        if "pyref" not in rec.skip:
            ref_c, dt = gen_pyref(prev, k, names, name_idx)
            rec.emit({"lever": "L4", "variant": "pyref", "k": k, "s": round(dt, 6), "n_prev": len(prev)})
            assert np.array_equal(ref_c, cands), f"L4: array candidates differ from _generate_candidates at K={k}"
        if len(cands) == 0:
            break
        counts = {}
        r = rec.run("L3", "rust_simd", count_rust, csr, cands, threads, k=k, n_candidates=len(cands))
        if r is not None:
            counts["rust_simd"] = r[0]
            rec.emit({"lever": "L3", "variant": "rust_call", "k": k, "s": round(r[1], 6)})
        for name, fn, args in (("bitvec", count_bitvec, (bv, cands)), ("bitvec_lut", count_bitvec, (bv, cands, True))):
            r = rec.run("L3", name, fn, *args, k=k, n_candidates=len(cands))
            if r is not None:
                counts[name] = r
        for name, fn, args in (("proj", count_proj, (csr, csc, cands, k)), ("gbitvec", count_gbitvec, (bv, cands, k))):
            r = rec.run("L3", name, fn, *args, k=k, n_candidates=len(cands))
            if r is not None:
                counts[name] = r[0]
                rec.emit({"lever": "L3", "variant": f"{name}_groups", "k": k, **r[1]})
        base_name, c0 = next(iter(counts.items()))
        for name, c in counts.items():
            assert np.array_equal(c, c0), f"L3: {name} differs from {base_name} at K={k}"
        keep = rec.run("L5", "threshold_mask", lambda: np.flatnonzero(c0 >= min_count), k=k)
        rec.run("L5", "csr_len_filter", lambda: csr[np.flatnonzero(np.diff(csr.indptr) >= k)], k=k)
        rec.run("L5", "int64_casts", lambda: (csr.indptr.astype(np.int64), csr.indices.astype(np.int64)), k=k)
        prev = cands[keep]
        rec.emit({"lever": "L3", "variant": "result", "k": k, "n_candidates": len(cands), "n_frequent": len(prev)})
        if len(prev) == 0:
            break
        lattice.append((prev, c0[keep]))
        k += 1

    # Reference: the all-Rust miner that already exists.
    def rust_full():
        from et_miner_rust import apriori_from_csr

        return apriori_from_csr(csr.indptr.astype(np.int64), csr.indices.astype(np.int64), n_rows, n_cols,
                                float(min_support), int(max_length or 0))

    rf = rec.run("ref", "rust_full", rust_full)

    # Signature of the stake lattice, in the campaign's form.
    rows = [(item_ids[s].tolist(), int(c) / n_rows) for sets, cnts in lattice for s, c in zip(sets, cnts)]
    frame = pl.DataFrame(rows, schema={"itemset": pl.List(pl.Int64), "support": pl.Float64}, orient="row")
    sig = result_signatures(frame, n_rows)
    camp = _campaign_signature(raw, workload)
    summary = {"lever": "summary", "variant": "lattice", **sig, "levels": {i + 1: len(s) for i, (s, _) in enumerate(lattice)}}
    if rf is not None:
        rsets, rcnts = rf
        rframe = pl.DataFrame([(item_ids[list(s)].tolist(), int(c) / n_rows) for s, c in zip(rsets, rcnts)],
                              schema={"itemset": pl.List(pl.Int64), "support": pl.Float64}, orient="row")
        summary["rust_full_matches"] = result_signatures(rframe, n_rows) == sig
    if camp is not None:
        summary["campaign_matches"] = all(sig[x] == camp[x] for x in ("n_itemsets", "sum_counts", "itemset_hash"))
        summary["campaign_row"] = camp["from"]
    rec.emit(summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, choices=list(WORKLOADS))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--skip", default="")
    args = ap.parse_args()
    rec = Recorder(args.workload, args.threads, args.out, {s for s in args.skip.split(",") if s})
    summary = mine(args.workload, args.threads, rec, args.baseline)
    ok = summary.get("campaign_matches", True) and summary.get("rust_full_matches", True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
