"""The CPU route of apriori(): mining from one CSR of the frequent items.

Polars reads the transactions once: the row count, the K=1 counts (explode +
group_by), and a CSR of each row's frequent items, built in row chunks. Every
level is then mined from that CSR; nothing goes back through a boolean
DataFrame.

Usage:
    from et_miner.core.cpu_miner import build_transaction_csr, mine_cpu

    tc = build_transaction_csr(lf, min_support=0.01, item_col="items")
    result = mine_cpu(lf, min_support=0.01, max_length=None, item_col="items")

Options (module constants, patched by tests):
    CSR_CHUNK_NNZ      list entries mapped to columns per chunk while the CSR
                       is built; bounds the build's temporary memory
    GRAM_BUDGET_BYTES  bytes one K=2 Gram block may take (12 B per entry of
                       a dense block); above it the columns are split into
                       blocks
    CAND_CHUNK         K>=3 candidates generated (before the subset test) per
                       chunk; bounds the generation's temporary memory
    PAIR_MASK_BYTES    largest n_items**2 for which the K=3 subset test reads
                       a boolean pair mask instead of binary-searching keys
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import polars as pl
from loguru import logger
from scipy.sparse import csr_matrix

from .profiling import ProfilingSession
from .result import _build_result_df, _empty_result, _min_count

CSR_CHUNK_NNZ = 500_000
GRAM_BUDGET_BYTES = 256 << 20
CAND_CHUNK = 2_000_000
PAIR_MASK_BYTES = 64 << 20


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


def _map_rows(column: pl.Series, items: pl.Series, bound: int) -> tuple[np.ndarray, np.ndarray]:
    """(indptr, indices int32) of each row's frequent items; indptr is int32 below 2**31 entries.

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
    indptr = np.zeros(n_rows + 1, dtype=np.int32 if bound < 2**31 else np.int64)
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

    Steps: count the rows; count every item with explode + group_by (streaming,
    nulls dropped) and keep those at or above ``_min_count``, sorted by item;
    collect the list column and map it to column ids in row chunks
    (``_map_rows``); recount each column once per row. An item that reached
    min_count only through repeats inside rows falls below it here and its
    column is removed.
    """
    n_rows = lf.select(pl.len()).collect(engine="streaming").item()
    min_count = _min_count(min_support, n_rows)
    freq = (
        lf.select(pl.col(item_col).explode().alias("item"))
        .drop_nulls()
        .group_by("item")
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") >= min_count)
        .sort("item")
        .collect(engine="streaming")
    )
    if freq.height == 0:
        return None
    items = freq.get_column("item")
    column = lf.select(pl.col(item_col)).collect(engine="in-memory").get_column(item_col)
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


def count_pairs(m: csr_matrix, gen: np.ndarray, min_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Every pair of generating columns with count >= min_count, from one Gram matrix.

    Returns (pairs int32 (n, 2) with pairs[:, 0] < pairs[:, 1], lexsorted;
    counts int64). ``gen`` is a boolean mask of the columns the pairs may use.

    Steps: G = M.T @ M holds every pair count (scipy sparse product,
    accumulated in int32). When a dense block of G (12 B per entry) fits
    ``GRAM_BUDGET_BYTES`` the product is taken whole; otherwise the rows of G
    are produced in column blocks, (M[:, block]).T @ M, from one transposed
    copy of M. Per block keep entries above the diagonal, between generating
    columns, at or above min_count; then lexsort the survivors.
    """
    n = m.shape[1]
    if n < 2:
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.int64)
    width = max(1, int(GRAM_BUDGET_BYTES // (12 * n)))
    blocks = [(0, n)] if width >= n else [(j0, min(n, j0 + width)) for j0 in range(0, n, width)]
    mt = m.T.tocsr() if len(blocks) > 1 else None
    out_i, out_j, out_c = [], [], []
    for j0, j1 in blocks:
        g = (m.T @ m).tocsr() if mt is None else (mt[j0:j1] @ m).tocsr()
        rows = np.repeat(np.arange(j0, j1, dtype=np.int32), np.diff(g.indptr))
        cols = g.indices
        keep = (cols > rows) & (g.data >= min_count) & gen[rows] & gen[cols]
        out_i.append(rows[keep])
        out_j.append(cols[keep].astype(np.int32))
        out_c.append(g.data[keep].astype(np.int64))
        del g, rows, cols, keep
    i, j, c = np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_c)
    order = np.lexsort((j, i))
    return np.stack([i[order], j[order]], axis=1), c[order]


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


def _count_on_csr(
    csr: csr_matrix,
    col_to_idx: dict[str, int],
    itemsets: list[tuple[str, ...]],
    k: int,
    n_jobs: int,
    length_filter: bool,
) -> dict[tuple[str, ...], int]:
    """Counts of same-length K>=3 itemsets from the CSR, by the CSR counters of core.sparse."""
    from . import sparse

    if length_filter:
        rows = np.flatnonzero(np.diff(csr.indptr) >= k)
        if len(rows) < csr.shape[0]:
            csr = csr[rows]
    return sparse._count_support_sparse_k_gt_2(csr, col_to_idx, itemsets, False, n_jobs)


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

    Steps: build the CSR (profile phase ``matrix_build``); emit K=1 from its
    column counts; then per level generate candidates from the previous
    level's generating itemsets (the free-sets in a free-set run), infer counts
    where Pascal licenses it, count the rest from the CSR, keep counts at or
    above min_count, and in a free-set run drop itemsets with a (k-1)-subset of
    equal count, tested against the complete previous level. The parameters
    mean what they mean on ``apriori()``.
    """
    from .apriori import _infer_count_from_subsets, _prune_equal_support, _warn_complexity

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

    n_trans = tc.n_rows
    max_tx_length = lf.select(pl.col(item_col).list.len().max()).collect(engine="streaming").item() or 0
    effective_max_length = min(max_length if max_length else float("inf"), max_tx_length, tc.n_cols)

    width = len(str(tc.n_cols))
    item_cols = [f"i_{idx:0{width}d}" for idx in range(tc.n_cols)]
    col_to_idx = {c: i for i, c in enumerate(item_cols)}
    item_values = tc.items.to_list()
    col_to_item = dict(zip(item_cols, item_values))
    csr = tc.to_scipy()

    _k1_start = time.perf_counter()
    if session:
        session.start_phase("k1_support")
    min_count_threshold = _min_count(min_support, n_trans)
    results: list[tuple[list, float]] = []
    prev_frequent: list[tuple[str, ...]] = []
    for col, count in zip(item_cols, tc.counts.tolist()):
        # Free-set semantics start at K=1: an item in every transaction has the
        # empty set's support, so it is not a generator.
        if prune_equal_support and count == n_trans:
            continue
        results.append(([col_to_item[col]], count / n_trans))
        prev_frequent.append((col,))
    if session:
        session.end_phase(n_frequent=len(prev_frequent))
    if level_callback:
        level_callback(1, tc.n_scanned, len(prev_frequent), (time.perf_counter() - _k1_start) * 1000)
    if not prev_frequent:
        return done(_empty_result())
    if warn_complexity:
        _warn_complexity(len(prev_frequent), min_support)

    # The complete previous level (every frequent itemset) feeds the subset
    # tests of the free-set prune; the free level feeds Pascal inference.
    prev_counts: dict[tuple[str, ...], int] = {(c,): n for c, n in zip(item_cols, tc.counts.tolist())}
    prev_free: set[tuple[str, ...]] | None = {s for s, n in prev_counts.items() if n < n_trans}

    k = 2
    while k <= effective_max_length and len(prev_frequent) >= k:
        _k_start = time.perf_counter()
        if k == 2:
            if session:
                session.start_phase("k2_candidate_gen")
                session.end_phase(n_candidates=len(prev_frequent) * (len(prev_frequent) - 1) // 2)
                session.start_phase("k2_support_count")
            gen = np.zeros(tc.n_cols, dtype=bool)
            gen[[col_to_idx[s[0]] for s in prev_frequent]] = True
            pairs, pair_counts = count_pairs(csr, gen, min_count_threshold)
            n_gen = int(gen.sum())
            n_candidates = n_gen * (n_gen - 1) // 2
            # Pascal would infer the pairs holding a non-free item; the Gram counts them anyway.
            n_nonfree = int((tc.counts[gen] == n_trans).sum()) if use_generator_pruning else 0
            n_inferred = n_candidates - (n_gen - n_nonfree) * (n_gen - n_nonfree - 1) // 2
            current_frequent = [(item_cols[a], item_cols[b]) for a, b in pairs.tolist()]
            current_counts = dict(zip(current_frequent, pair_counts.tolist()))
        else:
            if session:
                session.start_phase(f"k{k}_candidate_gen")
            prev_arr = np.array([[col_to_idx[c] for c in s] for s in prev_frequent], dtype=np.int32)
            prev_arr = prev_arr[np.lexsort(prev_arr.T[::-1])]
            chunks = list(generate_candidates(prev_arr, k, tc.n_cols))
            cand_arr = np.concatenate(chunks) if chunks else np.empty((0, k), dtype=np.int32)
            candidates = [tuple(item_cols[c] for c in row) for row in cand_arr.tolist()]
            if session:
                session.end_phase(n_candidates=len(candidates))
            if not candidates:
                break
            if session:
                session.start_phase(f"k{k}_support_count")
            n_candidates = len(candidates)

            inferred: dict[tuple[str, ...], int] = {}
            needs_counting: list[tuple[str, ...]] = []
            if use_generator_pruning:
                for candidate in candidates:
                    c = _infer_count_from_subsets(candidate, prev_counts, prev_free)
                    if c is not None:
                        inferred[candidate] = c
                    else:
                        needs_counting.append(candidate)
            else:
                needs_counting = candidates
            counted = (
                _count_on_csr(csr, col_to_idx, needs_counting, k, n_jobs, enable_length_filter)
                if needs_counting
                else {}
            )
            all_counts = {**inferred, **counted}
            n_inferred = len(inferred)

            current_frequent = []
            current_counts = {}
            for itemset in candidates:
                count = all_counts[itemset]
                if count >= min_count_threshold:
                    current_frequent.append(itemset)
                    current_counts[itemset] = count

        if prune_equal_support:
            current_frequent = _prune_equal_support(current_frequent, current_counts, prev_counts)
            current_free = set(current_frequent)
        elif use_generator_pruning:
            current_free = set(_prune_equal_support(current_frequent, current_counts, prev_counts))
        else:
            current_free = None

        for itemset in current_frequent:
            results.append(([col_to_item[c] for c in itemset], current_counts[itemset] / n_trans))
        if session:
            session.end_phase(n_frequent=len(current_frequent), n_inferred=n_inferred)
        if level_callback:
            level_callback(k, n_candidates, len(current_frequent), (time.perf_counter() - _k_start) * 1000)
        if not current_frequent:
            break
        prev_counts = current_counts
        prev_free = current_free
        prev_frequent = current_frequent
        k += 1

    return done(_build_result_df(results))
