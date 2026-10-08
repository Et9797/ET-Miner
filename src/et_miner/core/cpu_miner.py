"""The CPU route of apriori(): mining from one CSR of the frequent items.

Polars reads the transactions once: the row count, the K=1 counts (explode +
group_by), and a CSR of each row's frequent items, built in row chunks. Every
level is then mined from that CSR; nothing goes back through a boolean
DataFrame. With n_jobs > 1 the K=2 Gram blocks, the bitvector build and the
K>=3 prefix groups run on a thread pool (numpy and scipy release the GIL in
these kernels); n_jobs=1 runs everything on the calling thread.

Usage:
    from et_miner.core.cpu_miner import build_transaction_csr, mine_cpu

    tc = build_transaction_csr(lf, min_support=0.01, item_col="items")
    result = mine_cpu(lf, min_support=0.01, max_length=None, item_col="items")

Options (module constants, patched by tests):
    CSR_CHUNK_NNZ      list entries mapped to columns per chunk while the CSR
                       is built; bounds the build's temporary memory
    GRAM_BUDGET_BYTES  bytes a Gram block may take (12 B per entry of a
                       dense block), K=2 and the K>=3 projections alike;
                       above it the Gram is produced in blocks. Pooled K>=3
                       counting splits it over the blocks in flight
    CAND_CHUNK         K>=3 candidates generated (before the subset test) per
                       chunk; bounds the generation's temporary memory
    PAIR_MASK_BYTES    largest n_items**2 for which the K=3 subset test reads
                       a boolean pair mask instead of binary-searching keys
    BITVEC_BUDGET_BYTES  largest column-bitvector array (n_items x rows/8 B);
                       above it every K>=3 group is counted by projection
    BITVEC_CHUNK       CSR entries turned into bitvector words, or gathered
                       for one projection Gram, per step
    INDPTR32_LIMIT     CSR entries from which the indptr is int64 (int32 below)
    PROJ_MIN_SUFFIXES  (max words, min suffixes) steps: a prefix group with at
                       least that many suffixes is counted by projection when
                       the bitvectors have at most that many words
    COMPACT_RATIO      K>=3 counting moves to the rows holding >= k items once
                       they are at most this share of the current rows
    AND_CHUNK_BYTES    bytes of pair ANDs materialised at once
    HEAVY_GROUP_WORK   pairs x bitvector words from which a prefix group goes
                       to the thread pool (n_jobs > 1); lighter groups stay on
                       the calling thread
    PARALLEL_GRAM_WORK pair occurrences (sum of squared row lengths) from which
                       the K=2 Gram is split into blocks for the thread pool
    GRAM_BITVEC_RATIO  K=2 runs on bitvectors instead of the scipy Gram when
                       candidate pairs x words <= this x pair occurrences
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import polars as pl
from loguru import logger
from scipy.sparse import csr_matrix

from .profiling import ProfilingSession
from .result import _empty_result, _min_count

CSR_CHUNK_NNZ = 500_000
GRAM_BUDGET_BYTES = 256 << 20
CAND_CHUNK = 2_000_000
PAIR_MASK_BYTES = 64 << 20
BITVEC_BUDGET_BYTES = 512 << 20
BITVEC_CHUNK = 125_000
INDPTR32_LIMIT = 2**31
#: Measured by bench/cpu/l3_crossover.py on real prefix groups (K=3 and K=4): projection becomes
#: faster at about 30-40 suffixes on 570-1,563 words and about 66-100 on 3,907-15,625 words.
PROJ_MIN_SUFFIXES: tuple[tuple[float, int], ...] = ((2048, 40), (float("inf"), 80))
COMPACT_RATIO = 0.5
AND_CHUNK_BYTES = 4 << 20
HEAVY_GROUP_WORK = 2_000_000
PARALLEL_GRAM_WORK = 5_000_000
#: Measured on the eight campaign workloads (bitvector K=2 incl. its build, over the scipy Gram): 0.61x and
#: 0.69x at work ratios 1.7 and 2.7 (skewed_rows, deep_k), 6.3-38.6x at 7.1-175 (smoke, Online Retail, wide).
GRAM_BITVEC_RATIO = 3.0

_LUT16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)
_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")


@dataclass(frozen=True)
class TransactionCSR:
    """The frequent items of every transaction, one CSR row per transaction.

    Column j holds ``items[j]``; columns ascend in item order, so a row's
    column ids ascend with its item ids. ``counts[j]`` is the number of rows
    holding column j (each item counted once per row). ``n_scanned`` is the
    number of items whose explode count reached min_count, i.e. the K=1
    candidates (an item repeated inside a row counts once per occurrence
    there, then once per row in ``counts``).
    """

    indptr: np.ndarray
    indices: np.ndarray
    n_rows: int
    items: pl.Series
    counts: np.ndarray
    n_scanned: int

    @property
    def n_cols(self) -> int:
        return len(self.items)

    def to_scipy(self) -> csr_matrix:
        """The CSR as a scipy matrix with int32 ones as data (products accumulate in int32)."""
        data = np.ones(len(self.indices), dtype=np.int32)
        return csr_matrix((data, self.indices, self.indptr), shape=(self.n_rows, self.n_cols))


def _chunk_bounds(lens: np.ndarray, chunk_nnz: int) -> list[tuple[int, int]]:
    """Row ranges holding at most ``chunk_nnz`` list entries each (at least one row)."""
    n = len(lens)
    cum = np.cumsum(lens)
    bounds, start = [], 0
    while start < n:
        before = int(cum[start - 1]) if start else 0
        end = int(np.searchsorted(cum, before + chunk_nnz, side="right"))
        end = min(max(end, start + 1), n)
        bounds.append((start, end))
        start = end
    return bounds


def _count_items(column: pl.Series, lens: np.ndarray) -> pl.DataFrame:
    """(item, count Int64) over the list column, nulls dropped, counted per row chunk and summed.

    Chunking keeps Polars' hash tables to one chunk's distinct items, where one
    explode + group_by over the whole column held a table per thread.
    """
    parts = [
        column.slice(r0, r1 - r0)
        .explode()
        .drop_nulls()
        .to_frame("item")
        .group_by("item")
        .agg(pl.len().cast(pl.Int64).alias("count"))
        for r0, r1 in _chunk_bounds(lens, CSR_CHUNK_NNZ)
    ]
    if not parts:
        return pl.DataFrame(schema={"item": column.dtype.inner, "count": pl.Int64})
    return pl.concat(parts).group_by("item").agg(pl.col("count").sum())


def _map_rows(column: pl.Series, items: pl.Series, bound: int) -> tuple[np.ndarray, np.ndarray]:
    """(indptr, indices int32) of each row's frequent items; indptr is int32 below ``INDPTR32_LIMIT`` entries.

    Steps per row chunk: explode the chunk's lists (a null list explodes to
    one null, so its length counts as 1 for row alignment), map each value to
    its column id with ``replace_strict`` (values that are not frequent, and
    nulls, map to -1), take row ids from the list lengths, keep the mapped
    entries. A chunk whose column ids are not strictly increasing within a row
    (an unsorted row, or an item repeated in it) is sorted per row and
    deduplicated, so every row holds each column at most once. The output is
    written into one array of ``bound`` entries (the frequent items' explode
    counts, an upper bound) and shrunk in place. A 32-bit indptr keeps scipy
    from widening the indices to int64 when it wraps them.
    """
    lens = column.list.len().fill_null(1).to_numpy().astype(np.int64)
    n_rows = len(lens)
    col_ids = pl.Series(np.arange(len(items), dtype=np.int32))
    indptr = np.zeros(n_rows + 1, dtype=np.int32 if bound < INDPTR32_LIMIT else np.int64)
    out = np.empty(bound, dtype=np.int32)
    filled = 0
    for r0, r1 in _chunk_bounds(lens, CSR_CHUNK_NNZ):
        c = (
            column.slice(r0, r1 - r0)
            .explode()
            .replace_strict(items, col_ids, default=None, return_dtype=pl.Int32)
            .fill_null(-1)
            .to_numpy()
        )
        r = np.repeat(np.arange(r1 - r0, dtype=np.int32), lens[r0:r1])
        valid = c >= 0
        r, c = r[valid], c[valid]
        if len(c) > 1 and np.any((r[1:] == r[:-1]) & (c[1:] <= c[:-1])):
            order = np.lexsort((c, r))
            r, c = r[order], c[order]
            keep = np.r_[True, (r[1:] != r[:-1]) | (c[1:] != c[:-1])]
            r, c = r[keep], c[keep]
        indptr[r0 + 1 : r1 + 1] = np.bincount(r, minlength=r1 - r0)
        out[filled : filled + len(c)] = c
        filled += len(c)
    np.cumsum(indptr, out=indptr)
    out.resize(filled, refcheck=False)
    return indptr, out


def build_transaction_csr(lf: pl.LazyFrame, min_support: float, item_col: str = "items") -> TransactionCSR | None:
    """CSR of the frequent items of every transaction, or None when no item reaches min_count.

    Steps: collect the list column; count every item with explode + group_by
    per row chunk (``_count_items``, nulls dropped) and keep those at or above
    ``_min_count``, sorted by item; map the column to column ids in row chunks
    (``_map_rows``); recount each column once per row. An item that reached
    min_count only through repeats inside rows falls below it here and its
    column is removed.
    """
    column = lf.select(pl.col(item_col)).collect(engine="in-memory").get_column(item_col)
    n_rows = len(column)
    min_count = _min_count(min_support, n_rows)
    lens = column.list.len().fill_null(1).to_numpy().astype(np.int64)
    freq = _count_items(column, lens).filter(pl.col("count") >= min_count).sort("item")
    if freq.height == 0:
        return None
    items = freq.get_column("item")
    indptr, indices = _map_rows(column, items, int(freq.get_column("count").sum()))
    counts = np.bincount(indices, minlength=len(items)).astype(np.int64)
    keep = counts >= min_count
    if not keep.all():
        remap = (np.cumsum(keep) - 1).astype(np.int32)
        kept = keep[indices]
        rows = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(indptr))[kept]
        indices = remap[indices[kept]]
        indptr = np.zeros(n_rows + 1, dtype=indptr.dtype)
        np.cumsum(np.bincount(rows, minlength=n_rows), out=indptr[1:])
        items = items.filter(pl.Series(keep))
        counts = counts[keep]
    logger.debug("CPU CSR: {} rows, {} frequent items, {} entries", n_rows, len(items), len(indices))
    return TransactionCSR(indptr, indices, n_rows, items, counts, freq.height)


def _workers(n_jobs: int) -> int:
    """Threads for n_jobs: -1 = every CPU, otherwise at least 1."""
    return (os.cpu_count() or 1) if n_jobs == -1 else max(1, n_jobs)


def _split_by_weight(weights: np.ndarray, parts: int, max_width: int) -> list[tuple[int, int]]:
    """Contiguous index ranges with about equal total weight, each at most ``max_width`` long."""
    n = len(weights)
    cum = np.cumsum(weights, dtype=np.float64)
    total = cum[-1] if n else 0.0
    cuts = [0]
    if total > 0 and parts > 1:
        marks = np.searchsorted(cum, total * np.arange(1, parts) / parts, side="right")
        cuts += sorted({int(x) for x in marks} - {0, n})
    cuts.append(n)
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        out += [(x, min(b, x + max_width)) for x in range(a, b, max_width)]
    return out


def count_pairs(
    m: csr_matrix, gen: np.ndarray, min_count: int, pool: ThreadPoolExecutor | None = None, n_workers: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Every pair of generating columns with count >= min_count, from one Gram matrix.

    Returns (pairs int32 (n, 2) with pairs[:, 0] < pairs[:, 1], lexsorted;
    counts int64). ``gen`` is a boolean mask of the columns the pairs may use.

    Steps: G = M.T @ M holds every pair count (scipy sparse product,
    accumulated in int32). When a dense block of G (12 B per entry) fits
    ``GRAM_BUDGET_BYTES`` (shared by the blocks in flight) and no pool is
    given, the product is taken whole; otherwise the rows of G are produced in
    column blocks, (M[:, block]).T @ M, from one transposed copy of M, split
    by column count so the blocks carry similar work, several per worker
    (only when the rows hold at least ``PARALLEL_GRAM_WORK`` pair occurrences;
    below that the pool costs more than it saves).
    Per block keep entries above the diagonal, between generating columns, at
    or above min_count; then lexsort the survivors.
    """
    n = m.shape[1]
    if n < 2:
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.int64)
    lens = np.diff(m.indptr).astype(np.int64)
    if pool is not None and int((lens * lens).sum()) < PARALLEL_GRAM_WORK:
        pool, n_workers = None, 1
    width = max(1, int(GRAM_BUDGET_BYTES // (12 * n * max(1, n_workers))))
    parts = 4 * n_workers if pool is not None else 1
    if width >= n and parts == 1:
        blocks, mt = [(0, n)], None
    else:
        blocks = _split_by_weight(np.bincount(m.indices, minlength=n), parts, width)
        mt = m.T.tocsr()

    def block(rng: tuple[int, int]):
        j0, j1 = rng
        g = (m.T @ m).tocsr() if mt is None else (mt[j0:j1] @ m).tocsr()
        rows = np.repeat(np.arange(j0, j1, dtype=np.int32), np.diff(g.indptr))
        cols = g.indices
        keep = (cols > rows) & (g.data >= min_count) & gen[rows] & gen[cols]
        return rows[keep], cols[keep].astype(np.int32), g.data[keep].astype(np.int64)

    results = list(pool.map(block, blocks)) if pool is not None else [block(b) for b in blocks]
    i = np.concatenate([r[0] for r in results])
    j = np.concatenate([r[1] for r in results])
    c = np.concatenate([r[2] for r in results])
    order = np.lexsort((j, i))
    return np.stack([i[order], j[order]], axis=1), c[order]


def count_pairs_bitvec(
    bv: np.ndarray, gen: np.ndarray, min_count: int, pool: ThreadPoolExecutor | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """count_pairs on column bitvectors: popcount(bv[a] & bv[b]) for every pair of generating columns.

    ``gen`` holds the generating column ids, ascending. Returns the same
    (pairs, counts) as ``count_pairs``. Steps per left member a (position i in
    ``gen``): AND its bitvector with the contiguous block of the later members'
    bitvectors (a view, no gather), in blocks whose AND stays within
    ``AND_CHUNK_BYTES``, popcount, keep counts >= min_count. Left members are
    independent, so a pool runs them concurrently.
    """
    m, w = len(gen), bv.shape[1]
    if m < 2:
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.int64)
    sub = bv if (m == bv.shape[0] and np.array_equal(gen, np.arange(m))) else bv[gen]
    block = max(1, AND_CHUNK_BYTES // (8 * w))

    def left(i: int):
        out_j, out_c = [], []
        for j0 in range(i + 1, m, block):
            j1 = min(m, j0 + block)
            c = popcount_rows(sub[j0:j1] & sub[i])
            keep = np.flatnonzero(c >= min_count)
            out_j.append(keep + j0)
            out_c.append(c[keep])
        j = np.concatenate(out_j)
        return np.full(len(j), i, dtype=np.int64), j, np.concatenate(out_c)

    results = list(pool.map(left, range(m - 1))) if pool is not None else [left(i) for i in range(m - 1)]
    i = np.concatenate([r[0] for r in results])
    j = np.concatenate([r[1] for r in results])
    pairs = np.stack([gen[i], gen[j]], axis=1).astype(np.int32)
    return pairs, np.concatenate([r[2] for r in results]).astype(np.int64)


def _pack(rows: np.ndarray, base: int) -> np.ndarray | None:
    """Mixed-radix int64 keys of int32 rows (lexicographic order kept), or None if they could overflow."""
    if rows.shape[1] * math.log2(max(base, 2)) >= 62:
        return None
    key = np.zeros(len(rows), dtype=np.int64)
    for c in range(rows.shape[1]):
        key = key * base + rows[:, c]
    return key


class _Membership:
    """Which rows of an int32 array occur in a lexsorted level of the same width.

    Packed int64 keys and binary search when the keys fit in 62 bits; else a
    byte-wise row comparison (np.isin on a void view). At width 2 with
    n_items**2 <= PAIR_MASK_BYTES, a boolean pair mask.
    """

    def __init__(self, level: np.ndarray, base: int) -> None:
        self.base = base
        self.mask = None
        self.keys = None
        if level.shape[1] == 2 and base * base <= PAIR_MASK_BYTES:
            self.mask = np.zeros((base, base), dtype=bool)
            self.mask[level[:, 0], level[:, 1]] = True
            return
        self.keys = _pack(level, base)
        if self.keys is None:
            self.void = np.dtype((np.void, 4 * level.shape[1]))
            self.rows = np.ascontiguousarray(level, dtype=np.int32).view(self.void).ravel()

    def contains(self, sub: np.ndarray) -> np.ndarray:
        if self.mask is not None:
            return self.mask[sub[:, 0], sub[:, 1]]
        if self.keys is not None:
            key = _pack(sub, self.base)
            pos = np.minimum(np.searchsorted(self.keys, key), len(self.keys) - 1)
            return self.keys[pos] == key
        return np.isin(np.ascontiguousarray(sub, dtype=np.int32).view(self.void).ravel(), self.rows)


def generate_candidates(prev: np.ndarray, k: int, n_cols: int) -> Iterator[np.ndarray]:
    """K-candidates from the lexsorted (k-1)-itemsets ``prev``, in lexsorted chunks.

    A candidate joins two itemsets that share their first k-2 items (a prefix
    group) and survives when every (k-1)-subset is in ``prev`` (Apriori's
    subset test). Yields int32 (m, k) arrays; concatenated they are the full
    lexsorted candidate list of ``core.candidates._generate_candidates``.

    Steps: find the prefix runs; each member pairs with every later member of
    its run, so the number of partners per member is known; walk the members
    in chunks whose partner total stays near ``CAND_CHUNK`` (a large run is
    split by its left member); per chunk expand the pairs with repeat/cumsum,
    then test the k-2 subsets that drop a prefix item (the two that drop a
    suffix item are the joined itemsets themselves) with ``_Membership``.
    """
    n = len(prev)
    if n < 2:
        return
    if k == 3:
        change = prev[1:, 0] != prev[:-1, 0]
    else:
        change = np.any(prev[1:, : k - 2] != prev[:-1, : k - 2], axis=1)
    starts = np.flatnonzero(np.r_[True, change])
    sizes = np.diff(np.r_[starts, n])
    pos = np.arange(n) - np.repeat(starts, sizes)
    partners = np.repeat(sizes, sizes) - 1 - pos
    cum = np.cumsum(partners)
    member = _Membership(prev, n_cols)
    e0 = 0
    while e0 < n:
        before = int(cum[e0 - 1]) if e0 else 0
        e1 = min(max(int(np.searchsorted(cum, before + CAND_CHUNK, side="right")), e0 + 1), n)
        p = partners[e0:e1]
        total = int(p.sum())
        if total:
            left = np.repeat(np.arange(e0, e1, dtype=np.int64), p)
            right = left + 1 + (np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(p) - p, p))
            cands = np.empty((total, k), dtype=np.int32)
            cands[:, : k - 1] = prev[left]
            cands[:, k - 1] = prev[right, k - 2]
            del left, right
            keep = np.ones(total, dtype=bool)
            for drop in range(k - 2):
                idx = np.flatnonzero(keep)
                if len(idx) == 0:
                    break
                keep[idx] = member.contains(cands[idx][:, [c for c in range(k) if c != drop]])
            out = cands[keep]
            if len(out):
                yield out
        e0 = e1


def popcount_rows(a: np.ndarray) -> np.ndarray:
    """Set bits per row of a 2-D uint64 array (np.bitwise_count on NumPy >= 2, a 16-bit table below)."""
    if _HAS_BITWISE_COUNT:
        return np.bitwise_count(a).sum(axis=1, dtype=np.int64)
    return _LUT16[np.ascontiguousarray(a).view(np.uint16)].sum(axis=1, dtype=np.int64)


def _row_entries(indptr: np.ndarray, rows: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Positions in ``indices`` of every entry of ``rows`` (whose lengths are ``lens``), row by row."""
    total = int(lens.sum())
    starts = indptr[rows].astype(np.int64)
    return np.repeat(starts - (np.cumsum(lens) - lens), lens) + np.arange(total, dtype=np.int64)


def _aligned_bounds(lens: np.ndarray, chunk: int, align: int = 64) -> list[tuple[int, int]]:
    """Row ranges of about ``chunk`` entries whose inner boundaries are multiples of ``align``."""
    out, start, n = [], 0, len(lens)
    for _, e in _chunk_bounds(lens, chunk):
        e = n if e == n else max(((e // align) * align), start + align)
        if e > start:
            out.append((start, min(e, n)))
            start = min(e, n)
    if start < n:
        out.append((start, n))
    return out


def build_bitvecs(
    indptr: np.ndarray,
    indices: np.ndarray,
    n_cols: int,
    rows: np.ndarray | None = None,
    pool: ThreadPoolExecutor | None = None,
) -> np.ndarray:
    """(n_cols, ceil(n / 64)) uint64 column bitvectors over ``rows`` of a CSR (None = every row).

    Bit i of column c is set when the i-th selected row holds column c.

    Steps per range of selected rows holding about ``BITVEC_CHUNK`` entries
    (inner boundaries on multiples of 64 rows, so ranges own disjoint words):
    gather the rows' entries from the CSR, compute each entry's word
    (column * words + i // 64) and bit position (i % 64), group the entries by
    bit position (a stable sort on a uint8 key), and OR each group into its
    words. Within one bit position a word gets at most one entry, so each OR
    is a plain fancy-index update. No transposed copy of the CSR is made.
    """
    lens_all = np.diff(indptr).astype(np.int64)
    sel_lens = lens_all if rows is None else lens_all[rows]
    n = len(sel_lens)
    w = (n + 63) // 64
    out = np.zeros(n_cols * w, dtype=np.uint64)
    if n == 0 or len(indices) == 0:
        return out.reshape(n_cols, w)

    def fill(rng: tuple[int, int]) -> None:
        r0, r1 = rng
        lens = sel_lens[r0:r1]
        if lens.sum() == 0:
            return
        old = np.arange(r0, r1) if rows is None else rows[r0:r1]
        cols = indices[_row_entries(indptr, old, lens)].astype(np.int64)
        local = np.repeat(np.arange(r0, r1, dtype=np.int64), lens)
        bit = (local & 63).astype(np.uint8)
        order = np.argsort(bit, kind="stable")
        word = (cols * w + (local >> 6))[order]
        cuts = np.searchsorted(bit[order], np.arange(65))
        for v in range(64):
            if cuts[v + 1] > cuts[v]:
                idx = word[cuts[v] : cuts[v + 1]]
                out[idx] |= np.uint64(1) << np.uint64(v)

    ranges = _aligned_bounds(sel_lens, BITVEC_CHUNK)
    if pool is not None and len(ranges) > 1:
        list(pool.map(fill, ranges))
    else:
        for r in ranges:
            fill(r)
    return out.reshape(n_cols, w)


def proj_min_suffixes(words: int) -> int:
    """Suffixes from which a prefix group is counted by projection, at this bitvector width."""
    for max_words, min_suffixes in PROJ_MIN_SUFFIXES:
        if words <= max_words:
            return min_suffixes
    return PROJ_MIN_SUFFIXES[-1][1]


class RowSpace:
    """The rows K>=3 counting runs on: a selection of the transaction CSR's rows and their bitvectors.

    ``rows`` (ascending ids into the transaction CSR) restricts counting to the
    transactions long enough to hold the level's candidates; None keeps every
    row. The CSR itself is shared, never copied. Bitvectors are built when
    they fit ``BITVEC_BUDGET_BYTES``; without them every group is projected
    over its prefix's tidset, from a CSC built on first use.
    """

    def __init__(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        n_cols: int,
        rows: np.ndarray | None = None,
        pool: ThreadPoolExecutor | None = None,
    ):
        self.indptr, self.indices, self.n_cols, self.rows = indptr, indices, n_cols, rows
        self.n_rows = (len(indptr) - 1) if rows is None else len(rows)
        self.words = (self.n_rows + 63) // 64
        fits = n_cols * self.words * 8 <= BITVEC_BUDGET_BYTES
        self.bitvecs = build_bitvecs(indptr, indices, n_cols, rows, pool) if fits else None
        self.proj_min = proj_min_suffixes(self.words) if fits else 0
        self._csc: csr_matrix | None = None
        self._lock = threading.Lock()

    def full_rows(self, local: np.ndarray) -> np.ndarray:
        """Transaction-CSR ids of rows given as positions in this space."""
        return local if self.rows is None else self.rows[local]

    def tidset(self, items: np.ndarray) -> np.ndarray:
        """Transaction-CSR ids of the rows holding every item (CSC column intersection)."""
        with self._lock:
            if self._csc is None:
                data = np.ones(len(self.indices), dtype=bool)
                self._csc = csr_matrix(
                    (data, self.indices, self.indptr), shape=(len(self.indptr) - 1, self.n_cols)
                ).tocsc()
        ptr, idx = self._csc.indptr, self._csc.indices
        rows = idx[ptr[items[0]] : ptr[items[0] + 1]]
        for it in items[1:]:
            rows = np.intersect1d(rows, idx[ptr[it] : ptr[it + 1]], assume_unique=True)
        return rows

    def gram_pairs(
        self, rows: np.ndarray, suffix: np.ndarray, ia: np.ndarray, ib: np.ndarray, budget: int
    ) -> np.ndarray:
        """Counts (int64) of the suffix-position pairs (ia, ib), ia < ib, over transaction rows ``rows``.

        The counts are entries of the Gram matrix G = X.T @ X of the ``suffix``
        columns over ``rows``. Only G's rows ia.min() .. ia.max() are needed,
        and of each row block [a0, a1) only the columns from a0 on. The blocks
        are as tall as their dense form (12 B per entry: the int64 sum and one
        int32 product) allows within ``budget``. Per block and per chunk of
        rows holding about ``BITVEC_CHUNK`` entries: gather the rows' entries
        from the CSR, keep those in a suffix column at or after a0 (a lookup
        table maps column -> suffix position), build that small CSR X and add
        X[:, :a1 - a0].T @ X; then read the block's pairs.
        """
        out = np.zeros(len(ia), dtype=np.int64)
        if len(ia) == 0:
            return out
        s = len(suffix)
        lut = np.full(self.n_cols, -1, dtype=np.int32)
        lut[suffix] = np.arange(s, dtype=np.int32)
        lens_all = np.diff(self.indptr).astype(np.int64)[rows]
        chunks = _chunk_bounds(lens_all, BITVEC_CHUNK)
        lo, hi = int(ia.min()), int(ia.max()) + 1
        height = max(1, int(budget // (12 * (s - lo))))
        for a0 in range(lo, hi, height):
            a1 = min(hi, a0 + height)
            sel = np.flatnonzero((ia >= a0) & (ia < a1))
            if len(sel) == 0:
                continue
            g = np.zeros((a1 - a0, s - a0), dtype=np.int64)
            for r0, r1 in chunks:
                lens = lens_all[r0:r1]
                if lens.sum() == 0:
                    continue
                pos = lut[self.indices[_row_entries(self.indptr, rows[r0:r1], lens)]]
                local = np.repeat(np.arange(r1 - r0, dtype=np.int64), lens)
                keep = pos >= a0
                local, pos = local[keep], pos[keep] - a0
                ptr = np.zeros(r1 - r0 + 1, dtype=np.int64)
                np.cumsum(np.bincount(local, minlength=r1 - r0), out=ptr[1:])
                x = csr_matrix((np.ones(len(pos), dtype=np.int32), pos, ptr), shape=(r1 - r0, s - a0))
                g += (x.T.tocsr()[: a1 - a0] @ x).toarray()
            out[sel] = g[ia[sel] - a0, ib[sel] - a0]
        return out


def _group_runs(cands: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(starts, ends) of the runs of equal first k-2 columns in lexsorted candidates."""
    n = len(cands)
    if k == 3:
        change = cands[1:, 0] != cands[:-1, 0]
    else:
        change = np.any(cands[1:, : k - 2] != cands[:-1, : k - 2], axis=1)
    starts = np.flatnonzero(np.r_[True, change])
    return starts, np.r_[starts[1:], n]


def _count_group_bitvec(bv: np.ndarray, pre: np.ndarray, ia: np.ndarray, ib: np.ndarray, suffix: np.ndarray):
    """Pair counts of one group: the prefix's non-zero words of each suffix, ANDed per pair, popcounted."""
    nz = np.flatnonzero(pre)
    out = np.zeros(len(ia), dtype=np.int64)
    if len(nz) == 0:
        return out
    sub = bv[suffix[:, None], nz] & pre[nz]
    step = max(1, AND_CHUNK_BYTES // (8 * len(nz)))
    for c0 in range(0, len(ia), step):
        c1 = min(c0 + step, len(ia))
        out[c0:c1] = popcount_rows(sub[ia[c0:c1]] & sub[ib[c0:c1]])
    return out


def _count_group_proj(
    space: RowSpace, rows: np.ndarray, ia: np.ndarray, ib: np.ndarray, suffix: np.ndarray, budget: int | None = None
):
    """Pair counts of one group: the Gram matrix of the suffix columns over the prefix's transaction rows."""
    return space.gram_pairs(rows, suffix, ia, ib, GRAM_BUDGET_BYTES if budget is None else budget)


def count_candidates(
    cands: np.ndarray, k: int, space: RowSpace, pool: ThreadPoolExecutor | None = None, n_workers: int = 1
) -> np.ndarray:
    """Counts (int64) of lexsorted K-candidates sharing prefix runs, one prefix group at a time.

    Steps per run of equal first k-2 items: the suffix columns are the union
    of the candidates' last two items. With bitvectors, AND the prefix's
    columns (the AND of all but its last item is reused across consecutive
    groups); a group with at least ``space.proj_min`` suffixes is counted by
    projection (its rows are the prefix AND's set bits), a smaller one on the
    bitvectors. Without bitvectors every group is projected over the
    prefix's CSC tidset. With a pool, the groups whose pairs x words reach
    ``HEAVY_GROUP_WORK`` are cut into slices of about equal work, several per
    worker, and counted on the pool while the lighter groups are counted on
    the calling thread (many small numpy calls from several threads contend
    for the GIL and run slower than one thread). Every group writes its own
    range of the output. A projection's Gram blocks fit ``GRAM_BUDGET_BYTES``,
    shared by the workers and the calling thread when a pool runs.
    """
    out = np.empty(len(cands), dtype=np.int64)
    starts, ends = _group_runs(cands, k)
    if pool is None or len(starts) < 2:
        _count_groups(cands, k, space, starts, ends, out, GRAM_BUDGET_BYTES)
        return out
    budget = GRAM_BUDGET_BYTES // (n_workers + 1)
    work = (ends - starts).astype(np.float64) * max(1, space.words)
    heavy = work >= HEAVY_GROUP_WORK
    hs, he = starts[heavy], ends[heavy]
    slices = _split_by_weight(work[heavy], 4 * n_workers, len(hs)) if len(hs) else []
    futures = [pool.submit(_count_groups, cands, k, space, hs[a:b], he[a:b], out, budget) for a, b in slices]
    _count_groups(cands, k, space, starts[~heavy], ends[~heavy], out, budget)
    for f in futures:
        f.result()
    return out


def _count_groups(
    cands: np.ndarray, k: int, space: RowSpace, starts: np.ndarray, ends: np.ndarray, out: np.ndarray, budget: int
):
    """count_candidates' per-group work over the runs [starts[i], ends[i]), written into ``out``."""
    bv = space.bitvecs
    parent_key, parent_and = None, None
    for s, e in zip(starts.tolist(), ends.tolist()):
        prefix = cands[s, : k - 2]
        a, b = cands[s:e, k - 2], cands[s:e, k - 1]
        suffix = np.union1d(a, b)
        ia, ib = np.searchsorted(suffix, a), np.searchsorted(suffix, b)
        if bv is None:
            out[s:e] = _count_group_proj(space, space.tidset(prefix), ia, ib, suffix, budget)
            continue
        if k == 3:
            pre = bv[prefix[0]]
        else:
            key = prefix[:-1].tobytes()
            if key != parent_key:
                parent_key = key
                parent_and = bv[prefix[0]].copy()
                for it in prefix[1:-1]:
                    parent_and &= bv[it]
            pre = parent_and & bv[prefix[-1]]
        if len(suffix) >= space.proj_min:
            local = np.flatnonzero(np.unpackbits(pre.view(np.uint8), bitorder="little"))
            out[s:e] = _count_group_proj(space, space.full_rows(local), ia, ib, suffix, budget)
        else:
            out[s:e] = _count_group_bitvec(bv, pre, ia, ib, suffix)


class _LevelIndex:
    """Row positions in a lexsorted int32 level: packed keys and binary search, byte-wise rows if keys overflow."""

    def __init__(self, level: np.ndarray, base: int) -> None:
        self.base = base
        self.keys = _pack(level, base)
        if self.keys is None:
            self.void = np.dtype((np.void, 4 * level.shape[1]))
            rows = np.ascontiguousarray(level, dtype=np.int32).view(self.void).ravel()
            self.order = np.argsort(rows, kind="stable")
            self.sorted = rows[self.order]

    def find(self, sub: np.ndarray) -> np.ndarray:
        """Positions of ``sub``'s rows in the level, -1 where absent."""
        if self.keys is not None:
            key = _pack(sub, self.base)
            pos = np.minimum(np.searchsorted(self.keys, key), len(self.keys) - 1)
            return np.where(self.keys[pos] == key, pos, -1)
        q = np.ascontiguousarray(sub, dtype=np.int32).view(self.void).ravel()
        pos = np.minimum(np.searchsorted(self.sorted, q), len(self.sorted) - 1)
        return np.where(self.sorted[pos] == q, self.order[pos], -1)


@dataclass
class _Level:
    """One complete frequent level: lexsorted itemsets, their counts, and (when tracked) which are free."""

    sets: np.ndarray
    counts: np.ndarray
    free: np.ndarray | None
    _index: _LevelIndex | None = None

    def subset_positions(self, cands: np.ndarray, base: int) -> np.ndarray:
        """(n, k) positions in this level of each candidate's (k-1)-subsets (column j drops item j)."""
        if self._index is None:
            self._index = _LevelIndex(self.sets, base)
        index = self._index
        k = cands.shape[1]
        out = np.empty((len(cands), k), dtype=np.int64)
        for j in range(k):
            out[:, j] = index.find(cands[:, [c for c in range(k) if c != j]])
        return out


def _non_free(counts: np.ndarray, subset_counts: np.ndarray) -> np.ndarray:
    """An itemset is not free when a (k-1)-subset has its count (Bastide et al. 2000)."""
    return np.any(subset_counts == counts[:, None], axis=1)


def _emit(levels: list[tuple[np.ndarray, np.ndarray]], items: pl.Series, n_rows: int) -> pl.DataFrame:
    """The result frame: every emitted level's itemsets as item-id lists, support = count / n_rows.

    Item ids keep the input's kind as the old per-row emission did: integers as
    Int64 (UInt64 kept), floats as Float64, categoricals as String.
    """
    levels = [(s, c) for s, c in levels if len(s)]
    if not levels:
        return _empty_result()
    dtype = items.dtype
    if dtype.is_integer() and dtype != pl.UInt64:
        items = items.cast(pl.Int64)
    elif dtype.is_float():
        items = items.cast(pl.Float64)
    elif dtype.base_type() in (pl.Categorical, pl.Enum):
        items = items.cast(pl.String)
    frames = []
    for sets, counts in levels:
        n, k = sets.shape
        values = items.gather(pl.Series(sets.ravel().astype(np.int64)))
        frames.append(pl.DataFrame({"itemset": values.reshape((n, k)).arr.to_list(), "support": counts / n_rows}))
    return pl.concat(frames)


def mine_cpu(
    lf: pl.LazyFrame,
    min_support: float,
    max_length: int | None,
    item_col: str = "items",
    *,
    prune_equal_support: bool = False,
    use_generator_pruning: bool = False,
    enable_length_filter: bool = True,
    n_jobs: int = 1,
    level_callback: Callable[[int, int, int, float], None] | None = None,
    profile: bool = False,
    warn_complexity: bool = True,
) -> pl.DataFrame | tuple[pl.DataFrame, ProfilingSession]:
    """Mine the frequent itemsets (or free-sets) of ``lf`` on the CPU.

    Steps:

    1. Build the CSR (profile phase ``matrix_build``) and emit K=1 from its
       column counts. Items in every transaction are not free.
    2. K=2: ``count_pairs`` over the generating items.
    3. K>=3: stream ``generate_candidates`` chunks from the previous level's
       generating itemsets; where Pascal licenses it (a non-free subset) take
       the minimum subset count, count the rest with ``count_candidates`` on
       the current row space, keep counts >= min_count.
    4. Free-set test where tracked: an itemset whose (k-1)-subset in the
       complete previous level has its count is not free. A free-set run
       emits and generates from the free itemsets; Pascal needs the flags.

    Every level is a lexsorted int32 array with an int64 count array; the
    result frame is built once, at the end. The parameters mean what they
    mean on ``apriori()``.
    """
    from .apriori import _warn_complexity

    session = ProfilingSession() if profile else None

    def done(df: pl.DataFrame):
        return (df, session) if profile else df

    if session:
        session.start_phase("matrix_build")
    tc = build_transaction_csr(lf, min_support, item_col)
    if session:
        session.end_phase(
            n_frequent_items=tc.n_cols if tc else 0,
            n_transactions=tc.n_rows if tc else lf.select(pl.len()).collect(engine="streaming").item(),
        )
    if tc is None:
        return done(_empty_result())

    n_trans, n_cols = tc.n_rows, tc.n_cols
    # min_support=0 gives 0; at 0 the Gram (which only sees co-occurring pairs) and the bitvectors would disagree.
    min_count = max(1, _min_count(min_support, n_trans))
    max_tx_length = lf.select(pl.col(item_col).list.len().max()).collect(engine="streaming").item() or 0
    effective_max_length = min(max_length if max_length else float("inf"), max_tx_length, n_cols)

    t_level = time.perf_counter()
    if session:
        session.start_phase("k1_support")
    ones = np.arange(n_cols, dtype=np.int32)[:, None]
    free = tc.counts < n_trans
    prev = _Level(ones, tc.counts, free)
    gen = ones[free] if prune_equal_support else ones
    emitted = [(gen, tc.counts[free] if prune_equal_support else tc.counts)]
    if session:
        session.end_phase(n_frequent=len(gen))
    if level_callback:
        level_callback(1, tc.n_scanned, len(gen), (time.perf_counter() - t_level) * 1000)
    if len(gen) == 0:
        return done(_empty_result())
    if warn_complexity:
        _warn_complexity(len(gen), min_support)

    workers = _workers(n_jobs)
    pool = ThreadPoolExecutor(workers) if workers > 1 else None
    try:
        _mine_levels(
            tc, gen, prev, emitted, min_count, effective_max_length, prune_equal_support, use_generator_pruning,
            enable_length_filter, level_callback, session, pool, workers,
        )
    finally:
        if pool is not None:
            pool.shutdown()
    return done(_emit(emitted, tc.items, n_trans))


def _mine_levels(
    tc: TransactionCSR,
    gen: np.ndarray,
    prev: _Level,
    emitted: list[tuple[np.ndarray, np.ndarray]],
    min_count: int,
    effective_max_length: float,
    prune_equal_support: bool,
    use_generator_pruning: bool,
    enable_length_filter: bool,
    level_callback: Callable[[int, int, int, float], None] | None,
    session: ProfilingSession | None,
    pool: ThreadPoolExecutor | None,
    workers: int,
) -> None:
    """mine_cpu's levels K>=2: appends each emitted level (itemsets, counts) to ``emitted``."""
    n_trans, n_cols = tc.n_rows, tc.n_cols
    track_free = prune_equal_support or use_generator_pruning
    space: RowSpace | None = None
    k = 2
    while k <= effective_max_length and len(gen) >= k:
        t_level = time.perf_counter()
        if k == 2:
            if session:
                session.record_phase("k2_candidate_gen", 0.0, n_candidates=len(gen) * (len(gen) - 1) // 2)
                session.start_phase("k2_support_count")
            lens = np.diff(tc.indptr).astype(np.int64)
            occurrences = int((lens * (lens - 1) // 2).sum())
            words = (n_trans + 63) // 64
            pairs_work = len(gen) * (len(gen) - 1) // 2 * words
            if n_cols * words * 8 <= BITVEC_BUDGET_BYTES and pairs_work <= GRAM_BITVEC_RATIO * occurrences:
                space = RowSpace(tc.indptr, tc.indices, n_cols, None, pool)
                sets, counts = count_pairs_bitvec(space.bitvecs, gen[:, 0], min_count, pool)
            else:
                gen_mask = np.zeros(n_cols, dtype=bool)
                gen_mask[gen[:, 0]] = True
                sets, counts = count_pairs(tc.to_scipy(), gen_mask, min_count, pool, workers)
            n_candidates = len(gen) * (len(gen) - 1) // 2
            # Pascal would infer the pairs holding a non-free item; the Gram counts them anyway.
            n_nonfree = int((~prev.free[gen[:, 0]]).sum()) if use_generator_pruning else 0
            n_inferred = n_candidates - (len(gen) - n_nonfree) * (len(gen) - n_nonfree - 1) // 2
        else:
            if enable_length_filter:
                rows_k = np.flatnonzero(np.diff(tc.indptr) >= k)
                if space is None or len(rows_k) <= COMPACT_RATIO * space.n_rows:
                    space = RowSpace(tc.indptr, tc.indices, n_cols, rows_k if len(rows_k) < n_trans else None, pool)
            elif space is None:
                space = RowSpace(tc.indptr, tc.indices, n_cols, pool=pool)
            infer = use_generator_pruning and not prune_equal_support
            parts_s, parts_c = [], []
            n_candidates = n_inferred = 0
            t_gen = t_count = 0.0
            chunks = generate_candidates(gen, k, n_cols)
            while True:
                t0 = time.perf_counter()
                cands = next(chunks, None)
                t_gen += time.perf_counter() - t0
                if cands is None:
                    break
                t0 = time.perf_counter()
                n_candidates += len(cands)
                counts = np.empty(len(cands), dtype=np.int64)
                needs = np.ones(len(cands), dtype=bool)
                if infer:
                    pos = prev.subset_positions(cands, n_cols)
                    licensed = np.any(~prev.free[pos], axis=1)
                    counts[licensed] = prev.counts[pos[licensed]].min(axis=1)
                    needs = ~licensed
                    n_inferred += int(licensed.sum())
                if needs.any():
                    counts[needs] = count_candidates(cands[needs], k, space, pool, workers)
                keep = counts >= min_count
                parts_s.append(cands[keep])
                parts_c.append(counts[keep])
                t_count += time.perf_counter() - t0
            if session:
                session.record_phase(f"k{k}_candidate_gen", t_gen * 1000, n_candidates=n_candidates)
            if n_candidates == 0:
                break
            if session:
                session.record_phase(f"k{k}_support_count", t_count * 1000)
            sets = np.concatenate(parts_s) if parts_s else np.empty((0, k), dtype=np.int32)
            counts = np.concatenate(parts_c) if parts_c else np.empty(0, dtype=np.int64)
        level_free = None
        if track_free and len(sets):
            pos = prev.subset_positions(sets, n_cols)
            level_free = ~_non_free(counts, prev.counts[pos])
        elif track_free:
            level_free = np.zeros(0, dtype=bool)
        if prune_equal_support:
            gen = sets[level_free]
            emitted.append((gen, counts[level_free]))
        else:
            gen = sets
            emitted.append((sets, counts))
        prev = _Level(sets, counts, level_free)
        if session:
            if k == 2:
                session.end_phase(n_frequent=len(gen), n_inferred=n_inferred)
            else:
                session.phases[-1].extra.update(n_frequent=len(gen), n_inferred=n_inferred)
        if level_callback:
            level_callback(k, n_candidates, len(gen), (time.perf_counter() - t_level) * 1000)
        if len(gen) == 0:
            break
        k += 1

