"""Row-wise K=3 counting: each row adds 1 to the counted candidates it holds, over a slot layout by pair.

The counted candidates of K=3 are the triangles (u, v, w) of the frequent-pair
graph: all three pairs frequent, the candidates the subset test admits. Pair p
= (u, v), the p-th pair of the lexsorted K=2 level, owns one slot per frequent
partner w > v of v, at ``slot_off[p]`` + the rank of w among those partners; a
slot whose (u, w) is not frequent is never counted and stays 0.
``k3_layout`` lays the graph out on the host, ``build_k3_graph`` copies it to
a device with a bitmap of the pairs, ``count_k3_rows`` counts a slot range from
one shard's rows, and ``decode_k3_slots`` maps surviving slots back to column
triples.

Usage:
    layout = k3_layout(prev_frequent_flat)
    graph = build_k3_graph(layout, device_id)
    rows = upload_k2_rows(row_ptr, pos, device_id)  # positions in layout.active_cols
    counts = count_k3_rows(rows, graph, slot_start, slot_size)
    triples = decode_k3_slots(survivor_slots, layout)
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from .k2 import K2Rows
from .loader import _assert_dtype, _assert_home, get_cuda_kernel

#: Active positions a row may hold (``K3_ROWS_MAX_LEN`` in ``_src/k3_rows.cu``: one 64-bit neighbour mask each).
K3_ROWS_MAX_LEN = 64
#: Largest frequent-pair bitmap and its word ranks (12 B per 64 pairs of active items) the kernel reads; above
#: it, it searches the neighbour lists.
PAIR_BITS_MAX_BYTES = 512 << 20


class K3Layout(NamedTuple):
    """The frequent-pair graph of a lexsorted K=2 level and its slot layout, on the host."""

    active_cols: np.ndarray  # int32 (m,): position -> column, ascending
    pair_cols: np.ndarray  # int32 (n_pairs, 2): the K=2 level, lexsorted
    u: np.ndarray  # int32 (n_pairs,): each pair as positions, u < v
    v: np.ndarray  # int32 (n_pairs,): ascending per u; also the adjacency
    adj_off: np.ndarray  # int64 (m + 1,): u's partners are v[adj_off[u]:adj_off[u + 1]]
    slot_off: np.ndarray  # int64 (n_pairs + 1,): pair p's slots, one per partner of v

    @property
    def n_slots(self) -> int:
        return int(self.slot_off[-1])

    @property
    def n_generated(self) -> int:
        """Σ_u C(|N+(u)|, 2): the candidates the prefix groups generate."""
        degree = np.diff(self.adj_off)
        return int((degree * (degree - 1) // 2).sum())

    @property
    def device_bytes(self) -> int:
        """The device arrays ``build_k3_graph`` uploads, the pair bitmap included when it is built."""
        m = len(self.active_cols)
        words = -(-m // 64)
        bits = m * words * 12 if m * words * 12 <= PAIR_BITS_MAX_BYTES else 0
        return int(self.adj_off.nbytes + self.v.nbytes + self.slot_off.nbytes + bits)


class K3Graph(NamedTuple):
    """A ``K3Layout`` with its arrays on one device."""

    layout: K3Layout
    adj_off: Any  # CuPy copies of the layout's arrays
    adj: Any
    slot_off: Any
    pair_bits: Any  # CuPy uint64 (m, bits_words): bit v of row u set for a frequent pair; empty when too large
    pair_rank: Any  # CuPy int32 (m, bits_words): the set bits of row u before each word
    bits_words: int


def k3_layout(prev_frequent_flat: np.ndarray) -> K3Layout:
    """The slot layout of a lexsorted K=2 level (columns): its pairs as positions among the active columns."""
    flat = np.asarray(prev_frequent_flat, dtype=np.int32)
    present = np.zeros(int(flat.max()) + 1 if flat.size else 0, dtype=bool)
    present[flat.ravel()] = True
    active = np.flatnonzero(present).astype(np.int32)
    pos_of_col = np.cumsum(present, dtype=np.int32) - 1
    u, v = pos_of_col[flat[:, 0]], pos_of_col[flat[:, 1]]
    adj_off = np.zeros(len(active) + 1, dtype=np.int64)
    np.cumsum(np.bincount(u, minlength=len(active)), out=adj_off[1:])
    slot_off = np.zeros(len(u) + 1, dtype=np.int64)
    np.cumsum(np.diff(adj_off)[v], out=slot_off[1:])
    return K3Layout(active, flat, u, v, adj_off, slot_off)


def k3_pair_bits(u, v, m: int) -> tuple[Any, Any, int]:
    """(bitmap uint64, rank int32, words per row) of the lexsorted pairs (u, v), on the current device.

    Bit v of row u is set for each pair; ``rank[u, w]`` is the number of u's
    partners in the words before ``w``. Empty maps above ``PAIR_BITS_MAX_BYTES``.
    """
    import cupy as cp

    from .loader import get_popcount_kernel

    words = -(-m // 64)
    if m * words * 12 > PAIR_BITS_MAX_BYTES:
        return cp.zeros(0, dtype=cp.uint64), cp.zeros(0, dtype=cp.int32), 0
    bits = cp.zeros(m * words, dtype=cp.uint64)
    if len(u):
        # The pairs are distinct, so each word's bits add without carries: the sum is their OR.
        word = u.astype(cp.int64) * words + v // 64
        cp.add.at(bits, word, cp.left_shift(cp.uint64(1), (v % 64).astype(cp.uint64)))
    per_word = get_popcount_kernel()(bits).astype(cp.int32).reshape(m, words)
    rank = cp.cumsum(per_word, axis=1, dtype=cp.int32) - per_word
    return bits, rank.ravel(), words


def build_k3_graph(layout: K3Layout, device_id: int) -> K3Graph:
    """``layout``'s arrays, its pair bitmap and word ranks on ``device_id``."""
    import cupy as cp

    with cp.cuda.Device(device_id):
        v_gpu = cp.asarray(layout.v)
        bits, rank, words = k3_pair_bits(cp.asarray(layout.u), v_gpu, len(layout.active_cols))
        return K3Graph(layout, cp.asarray(layout.adj_off), v_gpu, cp.asarray(layout.slot_off), bits, rank, words)


def count_k3_rows(rows: K2Rows, graph: K3Graph, slot_start: int, slot_size: int):
    """Counts of the slots [slot_start, slot_start + slot_size) from one shard's rows (int32, on its device).

    ``rows`` holds the shard's positions among ``graph.active_cols``, at most
    ``K3_ROWS_MAX_LEN`` per row. The per-shard arrays sum to the global counts.
    """
    import cupy as cp

    _assert_home("count_k3_rows", row_ptr=rows.row_ptr, pos=rows.pos, adj=graph.adj, slot_off=graph.slot_off)
    _assert_dtype("count_k3_rows", row_ptr=(rows.row_ptr, "int64"), pos=(rows.pos, "int32"))
    with cp.cuda.Device(rows.row_ptr.device.id):
        out = cp.zeros(slot_size, dtype=cp.int32)
        if slot_size <= 0 or rows.n_rows == 0:
            return out
        slot_end = slot_start + slot_size
        pair_lo = int(np.searchsorted(graph.layout.slot_off, slot_start, side="right")) - 1
        pair_hi = int(np.searchsorted(graph.layout.slot_off, slot_end, side="left"))
        n_sm = cp.cuda.Device().attributes["MultiProcessorCount"]
        blocks = max(1, min(-(-rows.n_rows // 8), n_sm * 32))
        get_cuda_kernel("count_k3_rows")(
            (blocks,),
            (256,),
            (
                rows.row_ptr,
                rows.pos,
                np.int64(rows.n_rows),
                graph.adj_off,
                graph.adj,
                graph.slot_off,
                graph.pair_bits,
                graph.pair_rank,
                np.int64(graph.bits_words),
                np.int64(pair_lo),
                np.int64(pair_hi),
                np.int64(slot_start),
                np.int64(slot_end),
                out,
            ),
        )
        cp.cuda.Stream.null.synchronize()
    return out


def decode_k3_slots(slots: np.ndarray, layout: K3Layout) -> np.ndarray:
    """(n, 3) int32 column triples of slot indices (ascending within each triple)."""
    slots = np.asarray(slots, dtype=np.int64)
    pairs = np.searchsorted(layout.slot_off, slots, side="right") - 1
    v = layout.v[pairs]
    w = layout.v[layout.adj_off[v] + (slots - layout.slot_off[pairs])]
    out = np.empty((len(slots), 3), dtype=np.int32)
    out[:, :2] = layout.pair_cols[pairs]
    out[:, 2] = layout.active_cols[w]
    return out
