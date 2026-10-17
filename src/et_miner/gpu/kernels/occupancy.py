"""Tile occupancy of bitvectors: per column, one bit per tile of ``OCC_TILE_WORDS`` words (2,048 rows).

A bit is set when the column holds a row in that tile. The per-candidate and
group kernels skip a tile when one of a candidate's (or a group prefix's)
items has no row there, since the AND of the items is then zero on the whole
tile. Under the dataset's row order nearly every tile is occupied; under the
lex row order of ``gpu.input_layer`` the deep itemsets' rows gather, and most
tiles are empty for them.

Usage:
    occ = tile_occupancy(bitvecs_gpu)
    count_k3plus_per_candidate(..., occ=occ)
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

#: Words per occupancy tile: one warp's words in the per-candidate kernel, ``GP_TILE_W`` in the group kernel.
OCC_TILE_WORDS = 32
#: Columns whose occupancy is computed at once (bounds the boolean temporary).
OCC_BLOCK_COLS = 1024

_NONE: dict[int, tuple] = {}


class Occupancy(NamedTuple):
    bits: Any  # CuPy uint64 (n_cols, words), C-contiguous
    words: int
    device_id: int


def tile_occupancy(bitvecs_gpu) -> Occupancy:
    """The tile occupancy of (n_cols, n_u64s) bitvectors, on their device."""
    import cupy as cp

    n_cols, n_u64s = bitvecs_gpu.shape
    n_tiles = -(-n_u64s // OCC_TILE_WORDS)
    words = max(1, -(-n_tiles // 64))
    with cp.cuda.Device(bitvecs_gpu.device.id):
        bits = cp.zeros((n_cols, words), dtype=cp.uint64)
        shifts = cp.arange(64, dtype=cp.uint64)
        for c0 in range(0, n_cols, OCC_BLOCK_COLS):
            c1 = min(n_cols, c0 + OCC_BLOCK_COLS)
            nonzero = cp.zeros((c1 - c0, words * 64 * OCC_TILE_WORDS), dtype=cp.bool_)
            nonzero[:, :n_u64s] = bitvecs_gpu[c0:c1] != 0
            tiles = nonzero.reshape(c1 - c0, words * 64, OCC_TILE_WORDS).any(axis=2)
            bits[c0:c1] = (tiles.reshape(c1 - c0, words, 64).astype(cp.uint64) << shifts).sum(axis=2, dtype=cp.uint64)
        return Occupancy(bits, words, int(bitvecs_gpu.device.id))


def occupancy_args(occ: Occupancy | None, device_id: int) -> tuple:
    """The kernels' trailing (occ, occ_words) arguments; occ_words 0 (count every tile) when ``occ`` is None."""
    if occ is None:
        if device_id not in _NONE:
            import cupy as cp

            with cp.cuda.Device(device_id):
                _NONE[device_id] = (cp.zeros(1, dtype=cp.uint64), np.int64(0))
        return _NONE[device_id]
    if occ.device_id != device_id:
        raise ValueError(f"tile occupancy lives on device {occ.device_id}, the launch on device {device_id}")
    return occ.bits, np.int64(occ.words)
