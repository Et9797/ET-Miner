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
   with a null list holding entries, is exploded by Polars instead, each null
   becoming the null id ``span``; so is every chunk when pyarrow cannot be
   imported.
2. Per row shard (``csr_bitvec._row_split_cuts``), the ids minus the smallest
   are written as int32 into two pinned buffers in turn and copied
   asynchronously to the shard's device, after the shard's row offsets.
3. K=1: an occurrence bincount over the id range on every device, summed on
   the host; the ids reaching min_count are the columns.
4. A lookup table maps ids to columns and ``ids_to_bitvec`` sets the bits.

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
    CHUNK_ENTRIES         entries per pinned buffer
    KEEP_ROWS_BYTES       device bytes per entry that cutting the row-wise
                          kernels' rows may take; the entries are kept only
                          when that fits the available VRAM once the
                          bitvectors are built
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from et_miner.core.cpu_miner import _int_ids, _IntIds
from et_miner.core.result import _min_count

from .csr_bitvec import _row_split_cuts
from .row_split_chunks import _device_available_bytes

CHUNK_ENTRIES = 1 << 24
KEEP_ROWS_BYTES = 28

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
    released once the positions exist.
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

    def pairs(self, freq_cols, keep: bool = False) -> int:
        """Σ C(len, 2) over the rows, ``len`` being a row's distinct columns among ``freq_cols``."""
        import cupy as cp

        self._position(np.asarray(freq_cols, dtype=np.int64), keep)
        total = 0
        for sh, (row_ptr, _) in zip(self._shards, self._positions):
            with cp.cuda.Device(sh.device):
                lens = cp.diff(row_ptr)
                total += int((lens * (lens - 1) // 2).sum())
        return total

    def upload(self, cols, keep: bool = False) -> dict:
        """Each shard's rows as positions in ``cols``, keyed by device; without ``keep`` the entries are released."""
        import cupy as cp

        from .kernels import K2Rows

        self._position(np.asarray(cols, dtype=np.int64), keep)
        out = {}
        for sh, (row_ptr, pos) in zip(self._shards, self._positions):
            with cp.cuda.Device(sh.device):
                out[sh.device] = K2Rows(row_ptr, pos[pos >= 0], sh.n_rows)
        self._cols, self._positions = None, []
        if not keep:
            self.release()
        return out

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


def _canonicalize(sh: _Shard, n_cols: int) -> None:
    """Sort a shard's columns within each row and drop repeats (an unsorted row or a repeated item)."""
    import cupy as cp

    rows = cp.searchsorted(sh.row_ptr[1:], cp.arange(len(sh.keys), dtype=cp.int64), side="right")
    cols = sh.key_to_col[sh.keys]
    valid = cols >= 0
    key = cp.unique(rows[valid] * n_cols + cols[valid])
    del rows, cols, valid
    sh.row_ptr = cp.searchsorted(key // n_cols, cp.arange(sh.n_rows + 1, dtype=cp.int64), side="left")
    sh.keys = (key % n_cols).astype(cp.int32)
    sh.key_to_col = cp.arange(n_cols, dtype=cp.int32)
    sh.canonical = True


def _chunk_entries(s: pl.Series, ids: _IntIds, in_place: bool) -> tuple[np.ndarray, np.ndarray, bool]:
    """(row offsets int64 from 0, entries, shifted) of one Polars chunk of the list column.

    With ``in_place`` (pyarrow importable) the entries are the Arrow values in
    place, not yet minus the smallest id (``shifted`` False). Otherwise, and
    for a chunk with null items or with a null list holding entries, Polars
    explodes the chunk: an empty list explodes to nothing, a null list to one
    null, and every null becomes the null id ``span`` (``shifted`` True, int32).
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
            return offsets - offsets[0], entries, False
    lens = s.list.len().fill_null(1).to_numpy().astype(np.int64)
    exploded = s.explode(empty_as_null=False, keep_nulls=True)
    entries = np.full(len(exploded), ids.span, dtype=np.int32)
    entries[exploded.is_not_null().to_numpy()] = ids.shift(exploded.drop_nulls().to_numpy())
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    if offsets[-1] != len(entries):
        raise RuntimeError(f"a chunk explodes to {len(entries)} entries, its lengths sum to {offsets[-1]}")
    return offsets, entries, True


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

    def copy(self, dst, entries: np.ndarray, lo: int, shifted: bool) -> None:
        """``dst[i] = entries[i] - lo`` as int32 (``entries`` as they are when ``shifted``)."""
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
                if shifted or (lo == 0 and part.dtype == np.int32):
                    np.copyto(buf, part, casting="unsafe")
                else:
                    np.subtract(part, lo, out=buf, dtype=np.int64, casting="unsafe")
                dst[i : i + len(part)].set(buf, stream=stream)
                self._events[b] = stream.record()

    def finish(self) -> None:
        for stream in self._streams.values():
            stream.synchronize()


def _upload_shards(column: pl.Series, ids: _IntIds, cuts: list[tuple[int, int]]) -> list[_Shard]:
    """Each shard's row offsets and int32 ids on its device (shard ``i`` on device ``i``)."""
    import cupy as cp

    try:
        import pyarrow  # noqa: F401

        in_place = True
    except ImportError:
        in_place = False
    pieces, r0 = [], 0
    for s in column.get_chunks():
        if len(s):
            pieces.append((r0, *_chunk_entries(s, ids, in_place)))
            r0 += len(s)
    shards, uploader = [], _Uploader()
    for device, (s0, s1) in enumerate(cuts):
        parts, row_ptr, base = [], [np.zeros(1, dtype=np.int64)], 0
        for c0, offsets, entries, shifted in pieces:
            a, b = max(s0, c0), min(s1, c0 + len(offsets) - 1)
            if a >= b:
                continue
            e0, e1 = int(offsets[a - c0]), int(offsets[b - c0])
            row_ptr.append(offsets[a - c0 + 1 : b - c0 + 1] - e0 + base)
            parts.append((base, entries[e0:e1], shifted))
            base += e1 - e0
        with cp.cuda.Device(device):
            keys = cp.empty(base, dtype=cp.int32)
            for off, entries, shifted in parts:
                uploader.copy(keys[off : off + len(entries)], entries, ids.lo, shifted)
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


def _scatter_bitvecs(shards: list[_Shard], lut: np.ndarray, n_cols: int) -> list:
    """Each shard's bitvectors from its ids through ``lut``; records whether its rows are canonical."""
    import cupy as cp

    from .kernels import get_cuda_kernel

    bitvecs_list = []
    for sh in shards:
        with cp.cuda.Device(sh.device):
            sh.key_to_col = cp.asarray(lut)
            n_u64s = (sh.n_rows + 63) // 64
            bitvecs = cp.zeros((n_cols, n_u64s), dtype=cp.uint64)
            noncanonical = cp.zeros(1, dtype=cp.int32)
            get_cuda_kernel("ids_to_bitvec")(
                ((sh.n_rows + 255) // 256,),
                (256,),
                (sh.row_ptr, sh.keys, sh.key_to_col, bitvecs, noncanonical, np.int64(sh.n_rows), np.int64(n_u64s)),
            )
            sh.canonical = not int(noncanonical.get()[0])
            bitvecs_list.append((bitvecs, sh.device, sh.n_rows))
    return bitvecs_list


def build_device_input(column: pl.Series, min_support: float, n_gpus: int, *, keep_rows: bool):
    """Row-split bitvectors of ``column`` built on the GPUs: a ``DeviceInput``, None or ``NOT_SERVED``.

    None when no item reaches min_count. ``NOT_SERVED`` for a column the
    layer does not serve (see the module docstring) and for more than 2³¹ - 1
    rows, which the row-split miner rejects. ``keep_rows`` keeps the entries
    for the row-wise kernels when they fit (``KEEP_ROWS_BYTES``).
    """
    import cupy as cp

    ids = _int_ids(column)
    n_rows = len(column)
    if ids is None or n_rows > np.iinfo(np.int32).max:
        return NOT_SERVED
    if n_rows == 0:
        return None
    min_count = _min_count(min_support, n_rows)
    cuts = _row_split_cuts(n_rows, min(n_gpus, cp.cuda.runtime.getDeviceCount()))
    shards = _upload_shards(column, ids, cuts)
    freq = np.flatnonzero(_occurrences(shards, ids.span) >= min_count)
    logger.info(
        f"Device input: {n_rows:,} transactions, min_count={min_count}, {len(freq)} frequent items, "
        f"{sum(len(sh.keys) for sh in shards):,} entries on {len(shards)} GPU(s)"
    )
    if not len(freq):
        return None

    lut = np.full(ids.span + 1, -1, dtype=np.int32)
    lut[freq] = np.arange(len(freq), dtype=np.int32)
    bitvecs_list = _scatter_bitvecs(shards, lut, len(freq))

    rows = None
    if keep_rows:
        if all(KEEP_ROWS_BYTES * len(sh.keys) <= _device_available_bytes(sh.device)[0] for sh in shards):
            rows = DeviceRows(shards, len(freq))
        else:
            logger.info("Device input: the entries do not fit beside the bitvectors; K=2 counts from the bitvectors")
    if rows is None:
        for sh in shards:
            with cp.cuda.Device(sh.device):
                sh.keys = sh.key_to_col = sh.row_ptr = None
                cp.get_default_memory_pool().free_all_blocks()
    return DeviceInput(bitvecs_list, dict(enumerate((freq + ids.lo).tolist())), n_rows, rows)
