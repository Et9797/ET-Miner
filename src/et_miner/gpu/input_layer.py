"""The row-split route's device input layer: transactions to row-split bitvectors on the GPUs.

Serves integer item columns (Int8-Int64, UInt8-UInt32) whose ids span at most
``core.cpu_miner.INT_SPAN_LIMIT`` values; ``core.apriori`` sends any other
column through the host CSR (``core.matrix._build_csr_from_transactions``).
The bitvectors hold the host path's bits: the same row shards, the same
columns (the ids with at least min_count occurrences, ascending) and one bit
per row and item however often the item repeats in the row.

Steps:
1. Each Polars chunk of the list column is read through its Arrow buffers in
   place: the row offsets and the item values. A chunk with null items, or
   with a null list holding entries, is exploded by Polars instead, its nulls
   marked; so is every chunk when pyarrow cannot be imported. The items'
   smallest and largest values (nulls aside) give the span.
2. Per row shard (``csr_bitvec._row_split_cuts``), the ids minus the smallest
   are written as int32 into two pinned buffers in turn, a null as the null
   id ``span``, and copied asynchronously to the shard's device, after the
   shard's row offsets.
3. K=1: an occurrence bincount over the id range on every device, summed on
   the host; the ids reaching min_count are the columns.
4. A lookup table maps ids to columns and ``ids_to_bitvec`` sets the bits,
   each row at its own position or, with the lex row order, at its rank in
   its shard's rows sorted by their most frequent items (``_row_order``).

With ``keep_rows`` each shard's entries stay on its device (``DeviceRows``)
when they fit, and the row-wise K=2 and K=3 kernels' rows are cut from them
there.

Usage:
    from et_miner.gpu.input_layer import NOT_SERVED, build_device_input

    built = build_device_input(column, min_support, n_gpus=2, keep_rows=True)
    if built is NOT_SERVED: ...  # the host CSR path
    elif built is None: ...      # no item reaches min_count

Options:
    ET_MINER_INPUT_LAYER  "host" | "device" pins the input layer (``_env``)
    ET_MINER_ROW_ORDER    "keep" | "lex" pins the row order (``_env``)
    CHUNK_ENTRIES         entries per pinned buffer
    KEEP_ROWS_BYTES       device bytes per entry that cutting the row-wise
                          kernels' rows may take; the entries are kept only
                          when that fits the available VRAM once the
                          bitvectors are built
    CANON_BLOCK_ENTRIES   entries ``_canonicalize`` sorts at once
    CANON_ENTRY_BYTES     device bytes per entry of such a block, which a
                          shard with unsorted rows or repeated items adds to
                          the keep decision
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from et_miner.core.cpu_miner import INT_SPAN_LIMIT, _IntIds
from et_miner.core.result import _INT64_ITEM_DTYPES, _min_count

from .csr_bitvec import _row_split_cuts
from .row_split_chunks import _device_available_bytes

CHUNK_ENTRIES = 1 << 24
KEEP_ROWS_BYTES = 28
CANON_BLOCK_ENTRIES = 1 << 22
CANON_ENTRY_BYTES = 48

#: ``build_device_input``'s answer for a column it does not serve.
NOT_SERVED = object()


@dataclass
class _Shard:
    """One row shard's entries on its device: row offsets and an int32 key per entry.

    The keys are the item ids minus the smallest (``key_to_col`` maps them to
    columns, -1 for none), or, once ``_canonicalize`` ran, the columns
    themselves, sorted and unique within each row. ``canonical`` holds when
    every row's columns strictly ascend.
    """

    device: int
    n_rows: int
    row_ptr: Any
    keys: Any
    key_to_col: Any = None
    canonical: bool = True


class DeviceRows:
    """The shards' entries kept on their devices for the row-wise K=2 and K=3 kernels.

    ``pairs`` gives the K=2 dispatch's row pairs and ``upload`` each shard's
    rows as positions among a column set (``K2Rows``); both read the same
    positions, computed once per column set. With ``keep`` the entries stay
    for another column set (K=3's active columns); otherwise they are
    released once the positions exist. When a device runs out of memory for
    them, both return None and the rows are released: the row-wise kernels
    only speed K=2 and K=3 up, and the bitvectors count those levels too.
    """

    def __init__(self, shards: list[_Shard], n_cols: int):
        self._shards = shards
        self._n_cols = n_cols
        self._cols: np.ndarray | None = None
        self._positions: list[tuple[Any, Any]] = []

    def _position(self, cols: np.ndarray, keep: bool) -> None:
        import cupy as cp

        if self._cols is not None and np.array_equal(self._cols, cols):
            return
        pos_of_col = np.full(self._n_cols + 1, -1, dtype=np.int32)
        pos_of_col[cols + 1] = np.arange(len(cols), dtype=np.int32)
        self._cols, self._positions = cols, []
        for sh in self._shards:
            with cp.cuda.Device(sh.device):
                if not sh.canonical:
                    _canonicalize(sh, self._n_cols)
                pos = cp.asarray(pos_of_col)[sh.key_to_col + 1][sh.keys]
                if not keep:
                    sh.keys = sh.key_to_col = None
                kept = cp.zeros(len(pos) + 1, dtype=cp.int64)
                cp.cumsum(pos >= 0, dtype=cp.int64, out=kept[1:])
                self._positions.append((kept[sh.row_ptr], pos))
                del kept

    def pairs(self, freq_cols, keep: bool = False) -> int | None:
        """Σ C(len, 2) over the rows, ``len`` being a row's distinct columns among ``freq_cols``; None out of
        device memory."""
        sums = self.pairs_triples(freq_cols, keep)
        return None if sums is None else sums[0]

    def pairs_triples(self, cols, keep: bool = False) -> tuple[int, float] | None:
        """(Σ C(len, 2), Σ C(len, 3)) over the rows as in ``pairs``, the triples in float64; None out of device
        memory."""
        import cupy as cp

        try:
            self._position(np.asarray(cols, dtype=np.int64), keep)
            pairs, triples = 0, 0.0
            for sh, (row_ptr, _) in zip(self._shards, self._positions):
                with cp.cuda.Device(sh.device):
                    lens = cp.diff(row_ptr)
                    pairs += int((lens * (lens - 1) // 2).sum())
                    n = lens.astype(cp.float64)
                    triples += float((n * (n - 1) * (n - 2)).sum() / 6)
            sums = pairs, triples
        except cp.cuda.memory.OutOfMemoryError:
            sums = None
        if sums is None:
            self._out_of_memory()
        return sums

    def upload(self, cols, keep: bool = False) -> dict | None:
        """Each shard's rows as positions in ``cols``, keyed by device; without ``keep`` the entries are released.
        None out of device memory."""
        import cupy as cp

        from .kernels import K2Rows

        try:
            self._position(np.asarray(cols, dtype=np.int64), keep)
            out = {}
            for sh, (row_ptr, pos) in zip(self._shards, self._positions):
                with cp.cuda.Device(sh.device):
                    out[sh.device] = K2Rows(row_ptr, pos[pos >= 0], sh.n_rows)
        except cp.cuda.memory.OutOfMemoryError:
            out = None
        if out is None:
            self._out_of_memory()
            return None
        self.drop_positions()
        if not keep:
            self.release()
        return out

    def _out_of_memory(self) -> None:
        """Releases the rows once the OutOfMemoryError (and the arrays its traceback held) is gone."""
        import cupy as cp

        devices = sorted({sh.device for sh in self._shards})
        self.release()
        for device in devices:
            with cp.cuda.Device(device):
                cp.get_default_memory_pool().free_all_blocks()
        logger.info("Device rows: out of device memory; K=2 and K=3 count from the bitvectors")

    def drop_positions(self) -> None:
        """Drop the last column set's positions; the entries stay if ``keep`` kept them."""
        self._cols, self._positions = None, []

    def release(self) -> None:
        """Drop every shard's entries."""
        self._shards, self._cols, self._positions = [], None, []


@dataclass
class DeviceInput:
    """Row-split bitvectors built on the devices: ``build_bitvecs_row_split``'s list, and its mapping."""

    bitvecs_list: list
    col_to_item: dict[int, int]
    n_transactions: int
    rows: DeviceRows | None
    row_order: str = "keep"


def _canonicalize(sh: _Shard, n_cols: int) -> None:
    """Sort a shard's columns within each row and drop repeats (an unsorted row or a repeated item).

    Rows go in blocks of about ``CANON_BLOCK_ENTRIES`` entries, which bounds
    the sort's temporaries. Each block's columns are written back into
    ``keys`` in place: a block never yields more entries than it read, so its
    output ends before the next block's input starts.
    """
    import cupy as cp

    n_entries = len(sh.keys)
    starts = cp.arange(0, n_entries, CANON_BLOCK_ENTRIES, dtype=cp.int64)
    cuts = sorted({0, sh.n_rows} | set((cp.searchsorted(sh.row_ptr, starts, side="right") - 1).get().tolist()))
    bounds = sh.row_ptr[cp.asarray(cuts)].get().tolist()
    row_ptr = cp.empty(sh.n_rows + 1, dtype=cp.int64)
    out = 0
    for r0, r1, e0, e1 in zip(cuts, cuts[1:], bounds, bounds[1:]):
        rows = cp.searchsorted(sh.row_ptr[r0 + 1 : r1 + 1], cp.arange(e0, e1, dtype=cp.int64), side="right")
        cols = sh.key_to_col[sh.keys[e0:e1]]
        valid = cols >= 0
        key = cp.unique(rows[valid] * n_cols + cols[valid])
        del rows, cols, valid
        row_ptr[r0:r1] = cp.searchsorted(key // n_cols, cp.arange(r1 - r0, dtype=cp.int64), side="left") + out
        sh.keys[out : out + len(key)] = key % n_cols
        out += len(key)
    row_ptr[sh.n_rows] = out
    sh.row_ptr, sh.keys = row_ptr, sh.keys[:out]
    sh.key_to_col = cp.arange(n_cols, dtype=cp.int32)
    sh.canonical = True


def _keep_bytes(sh: _Shard) -> int:
    """Device bytes that cutting the row-wise kernels' rows from a shard may take, ``_canonicalize`` included.

    A block starts at the row holding its first entry and ends at the row
    holding the next block's, so it holds at most ``CANON_BLOCK_ENTRIES`` plus
    the longest row less one entries.
    """
    import cupy as cp

    n_entries = len(sh.keys)
    if sh.canonical:
        return KEEP_ROWS_BYTES * n_entries
    with cp.cuda.Device(sh.device):
        longest = int(cp.diff(sh.row_ptr).max()) if sh.n_rows else 0
    block = min(n_entries, CANON_BLOCK_ENTRIES + max(longest - 1, 0))
    return KEEP_ROWS_BYTES * n_entries + CANON_ENTRY_BYTES * block + 8 * (sh.n_rows + 1)


@dataclass
class _Chunk:
    """One Polars chunk of the list column: row offsets from 0, its entries as they are, and their nulls.

    ``nulls`` marks the entries that are no item (a null item, or the one
    entry a null list explodes to); None when there are none. ``lo`` / ``hi``
    are the smallest and largest item, None for a chunk without one.
    """

    offsets: np.ndarray
    entries: np.ndarray
    nulls: np.ndarray | None
    lo: int | None
    hi: int | None


def _chunk_entries(s: pl.Series, in_place: bool) -> _Chunk:
    """One Polars chunk's entries: the Arrow values in place, or for a chunk with nulls Polars' explode.

    With ``in_place`` (pyarrow importable) and no null item, and no null list
    holding entries, the entries are a view of the Arrow values. Otherwise
    Polars explodes the chunk: an empty list explodes to nothing, a null list
    to one null, and the nulls are marked.
    """
    if in_place:
        arr = s.to_arrow()
        offsets = arr.offsets.to_numpy().astype(np.int64, copy=False)
        values = arr.values
        null_lists_hold_entries = arr.null_count > 0 and bool(
            np.diff(offsets)[arr.is_null().to_numpy(zero_copy_only=False)].any()
        )
        if values.null_count == 0 and not null_lists_hold_entries:
            entries = values.to_numpy(zero_copy_only=True)[offsets[0] : offsets[-1]]
            if len(entries):
                return _Chunk(offsets - offsets[0], entries, None, int(entries.min()), int(entries.max()))
            return _Chunk(offsets - offsets[0], entries, None, None, None)
    lens = s.list.len().fill_null(1).to_numpy().astype(np.int64)
    exploded = s.explode(empty_as_null=False, keep_nulls=True)
    nulls = exploded.is_null().to_numpy()
    entries = exploded.fill_null(0).to_numpy()
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    if offsets[-1] != len(entries):
        raise RuntimeError(f"a chunk explodes to {len(entries)} entries, its lengths sum to {offsets[-1]}")
    items = entries[~nulls]
    if len(items):
        return _Chunk(offsets, entries, nulls, int(items.min()), int(items.max()))
    return _Chunk(offsets, entries, nulls, None, None)


def _read_chunks(column: pl.Series) -> tuple[list[tuple[int, _Chunk]], int, int]:
    """(first row, chunk) of every non-empty Polars chunk, and the smallest and largest item (0, 0 for none)."""
    try:
        import pyarrow  # noqa: F401

        in_place = True
    except ImportError:
        in_place = False
    pieces, r0 = [], 0
    for s in column.get_chunks():
        if len(s):
            pieces.append((r0, _chunk_entries(s, in_place)))
            r0 += len(s)
    los = [c.lo for _, c in pieces if c.lo is not None]
    his = [c.hi for _, c in pieces if c.hi is not None]
    return pieces, min(los, default=0), max(his, default=0)


class _Uploader:
    """Copies entries to the devices through two pinned int32 buffers, in turn, on one stream per device."""

    def __init__(self):
        import cupy as cp

        self._bufs = [
            np.frombuffer(cp.cuda.alloc_pinned_memory(CHUNK_ENTRIES * 4), dtype=np.int32, count=CHUNK_ENTRIES)
            for _ in range(2)
        ]
        self._events: list[Any] = [None, None]
        self._streams: dict[int, Any] = {}
        self._turn = 0

    def copy(self, dst, entries: np.ndarray, nulls: np.ndarray | None, ids: _IntIds) -> None:
        """``dst[i] = entries[i] - ids.lo`` as int32, and the null id ``ids.span`` where ``nulls`` is set."""
        import cupy as cp

        device = dst.device.id
        with cp.cuda.Device(device):
            stream = self._streams.setdefault(device, cp.cuda.Stream(non_blocking=True))
            for i in range(0, len(entries), CHUNK_ENTRIES):
                part = entries[i : i + CHUNK_ENTRIES]
                b = self._turn
                self._turn ^= 1
                if self._events[b] is not None:
                    self._events[b].synchronize()
                buf = self._bufs[b][: len(part)]
                if ids.lo == 0 and part.dtype == np.int32:
                    np.copyto(buf, part)
                else:
                    np.subtract(part, ids.lo, out=buf, dtype=np.int64, casting="unsafe")
                if nulls is not None:
                    buf[nulls[i : i + CHUNK_ENTRIES]] = ids.span
                dst[i : i + len(part)].set(buf, stream=stream)
                self._events[b] = stream.record()

    def finish(self) -> None:
        for stream in self._streams.values():
            stream.synchronize()


def _upload_shards(pieces: list[tuple[int, _Chunk]], ids: _IntIds, cuts: list[tuple[int, int]]) -> list[_Shard]:
    """Each shard's row offsets and int32 ids on its device (shard ``i`` on device ``i``)."""
    import cupy as cp

    shards, uploader = [], _Uploader()
    for device, (s0, s1) in enumerate(cuts):
        parts, row_ptr, base = [], [np.zeros(1, dtype=np.int64)], 0
        for c0, chunk in pieces:
            offsets = chunk.offsets
            a, b = max(s0, c0), min(s1, c0 + len(offsets) - 1)
            if a >= b:
                continue
            e0, e1 = int(offsets[a - c0]), int(offsets[b - c0])
            row_ptr.append(offsets[a - c0 + 1 : b - c0 + 1] - e0 + base)
            parts.append((base, chunk.entries[e0:e1], None if chunk.nulls is None else chunk.nulls[e0:e1]))
            base += e1 - e0
        with cp.cuda.Device(device):
            keys = cp.empty(base, dtype=cp.int32)
            for off, entries, nulls in parts:
                uploader.copy(keys[off : off + len(entries)], entries, nulls, ids)
            shard_ptr = cp.asarray(np.concatenate(row_ptr))
        shards.append(_Shard(device, s1 - s0, shard_ptr, keys))
    uploader.finish()
    return shards


def _occurrences(shards: list[_Shard], span: int) -> np.ndarray:
    """K=1: each id's occurrences (a repeated item counts each time, nulls not at all), summed over the shards."""
    import cupy as cp

    counts = np.zeros(span, dtype=np.int64)
    for sh in shards:
        if len(sh.keys):
            with cp.cuda.Device(sh.device):
                counts += cp.bincount(sh.keys, minlength=span + 1)[:span].get()
    return counts


def _row_order(shards: list[_Shard], lut: np.ndarray, rank_of_col: np.ndarray) -> list:
    """Each shard's lex row order: the bit position of every row (CuPy int32), rows sorted by ``row_order_keys``."""
    import cupy as cp

    from .kernels import get_cuda_kernel

    orders = []
    for sh in shards:
        with cp.cuda.Device(sh.device):
            key_hi = cp.empty(sh.n_rows, dtype=cp.uint64)
            key_lo = cp.empty(sh.n_rows, dtype=cp.uint64)
            get_cuda_kernel("row_order_keys")(
                ((sh.n_rows + 255) // 256,),
                (256,),
                (sh.row_ptr, sh.keys, cp.asarray(lut), cp.asarray(rank_of_col), key_hi, key_lo, np.int64(sh.n_rows)),
            )
            # (key_hi, key_lo) order by two stable radix sorts: by key_lo, then by key_hi through it.
            order = cp.argsort(key_lo)
            order = order[cp.argsort(key_hi[order])]
            del key_hi, key_lo
            new_pos = cp.empty(sh.n_rows, dtype=cp.int32)
            new_pos[order] = cp.arange(sh.n_rows, dtype=cp.int32)
            orders.append(new_pos)
    return orders


def _scatter_bitvecs(shards: list[_Shard], lut: np.ndarray, n_cols: int, new_pos: list | None = None) -> list:
    """Each shard's bitvectors from its ids through ``lut`` (each row at ``new_pos`` when given); records whether
    its rows are canonical."""
    import cupy as cp

    from .kernels import get_cuda_kernel

    bitvecs_list = []
    for s, sh in enumerate(shards):
        with cp.cuda.Device(sh.device):
            sh.key_to_col = cp.asarray(lut)
            n_u64s = (sh.n_rows + 63) // 64
            bitvecs = cp.zeros((n_cols, n_u64s), dtype=cp.uint64)
            noncanonical = cp.zeros(1, dtype=cp.int32)
            positions = new_pos[s] if new_pos is not None else cp.zeros(1, dtype=cp.int32)
            get_cuda_kernel("ids_to_bitvec")(
                ((sh.n_rows + 255) // 256,),
                (256,),
                (
                    sh.row_ptr,
                    sh.keys,
                    sh.key_to_col,
                    bitvecs,
                    noncanonical,
                    np.int64(sh.n_rows),
                    np.int64(n_u64s),
                    positions,
                    np.int32(new_pos is not None),
                ),
            )
            sh.canonical = not int(noncanonical.get()[0])
            bitvecs_list.append((bitvecs, sh.device, sh.n_rows))
    return bitvecs_list


def build_device_input(column: pl.Series, min_support: float, n_gpus: int, *, keep_rows: bool, row_order: str = "keep"):
    """Row-split bitvectors of ``column`` built on the GPUs: a ``DeviceInput``, None or ``NOT_SERVED``.

    None when no item reaches min_count. ``NOT_SERVED`` for a column the
    layer does not serve (see the module docstring) and for more than 2³¹ - 1
    rows, which the row-split miner rejects. ``keep_rows`` keeps the entries
    for the row-wise kernels when they fit (``KEEP_ROWS_BYTES``). With
    ``row_order="lex"`` each shard's rows take the lex row order
    (``_row_order``) in the bitvectors; counts do not depend on it.
    """
    import cupy as cp

    dtype = column.dtype
    n_rows = len(column)
    if not isinstance(dtype, pl.List) or dtype.inner not in _INT64_ITEM_DTYPES or n_rows > np.iinfo(np.int32).max:
        return NOT_SERVED
    pieces, lo, hi = _read_chunks(column)
    if hi - lo + 1 > INT_SPAN_LIMIT:
        return NOT_SERVED
    if n_rows == 0:
        return None
    ids = _IntIds(lo, hi - lo + 1, dtype.inner)
    min_count = _min_count(min_support, n_rows)
    cuts = _row_split_cuts(n_rows, min(n_gpus, cp.cuda.runtime.getDeviceCount()))
    shards = _upload_shards(pieces, ids, cuts)
    del pieces
    occurrences = _occurrences(shards, ids.span)
    freq = np.flatnonzero((occurrences >= min_count) & (occurrences > 0))
    logger.info(
        f"Device input: {n_rows:,} transactions, min_count={min_count}, {len(freq)} frequent items, "
        f"{sum(len(sh.keys) for sh in shards):,} entries on {len(shards)} GPU(s)"
    )
    if not len(freq):
        return None

    lut = np.full(ids.span + 1, -1, dtype=np.int32)
    lut[freq] = np.arange(len(freq), dtype=np.int32)
    new_pos = None
    if row_order == "lex":
        rank_of_col = np.empty(len(freq), dtype=np.int32)
        rank_of_col[np.argsort(-occurrences[freq], kind="stable")] = np.arange(len(freq), dtype=np.int32)
        new_pos = _row_order(shards, lut, rank_of_col)
    bitvecs_list = _scatter_bitvecs(shards, lut, len(freq), new_pos)
    del new_pos

    rows = None
    if keep_rows:
        if all(_keep_bytes(sh) <= _device_available_bytes(sh.device)[0] for sh in shards):
            rows = DeviceRows(shards, len(freq))
        else:
            logger.info("Device input: the entries do not fit beside the bitvectors; K=2 counts from the bitvectors")
    if rows is None:
        for sh in shards:
            with cp.cuda.Device(sh.device):
                sh.keys = sh.key_to_col = sh.row_ptr = None
                cp.get_default_memory_pool().free_all_blocks()
    return DeviceInput(bitvecs_list, dict(enumerate((freq + ids.lo).tolist())), n_rows, rows, row_order)
