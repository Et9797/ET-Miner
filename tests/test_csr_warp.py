"""GPU tests for the warp-cooperative CSR intersection kernels (_src/csr_warp.cu).

Reference is numpy ``np.intersect1d`` over sorted-unique rows. Candidates are
enumerated in-kernel from real group arrays whose suffix slots map to shard
rows through ``suffix_src_rows`` (never the identity here), so the tests pin
the kernel-side decode to the candidate order ``decode_k3plus_flat`` produces
on the host.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy", reason="cupy not installed")

pytestmark = pytest.mark.gpu

from et_miner.gpu.kernels.csr_warp import CANDS_PER_BLOCK, count_csr_gather, count_csr_range, write_csr_gather
from et_miner.gpu.kernels.decode import decode_k3plus_flat
from et_miner.gpu.kernels.k3plus import K3PlusGroups, build_k3plus_groups_from_flat, upload_k3plus_groups

UNIVERSE = 20_000
#: Row sizes straddling the 32-lane chunk and both sides > 32 chunks (3000, 5000).
SIZES = [0, 1, 2, 31, 32, 33, 100, 1000, 3000, 5000]


def _sorted_unique(rng, size, universe=UNIVERSE):
    return np.sort(rng.choice(universe, size=size, replace=False)).astype(np.int32)


def _overlapping(rng, base, size, overlap):
    """A sorted-unique row of `size` tids sharing about `overlap` of them with `base`."""
    take = min(int(round(size * overlap)), len(base), size)
    shared = rng.choice(base, size=take, replace=False) if take else np.array([], np.int32)
    pool = np.setdiff1d(np.arange(UNIVERSE, dtype=np.int32), base)
    rest = rng.choice(pool, size=size - take, replace=False) if size - take else np.array([], np.int32)
    return np.sort(np.concatenate([shared, rest]).astype(np.int32))


def _csr(rows):
    lengths = np.array([len(r) for r in rows], dtype=np.int64)
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    indices = np.concatenate([np.asarray(r, np.int32) for r in rows]).astype(np.int32)
    return cp.asarray(offsets), cp.asarray(indices)


def _groups(slot_rows):
    """One prefix group per list; slot s of a group is suffix value s and maps
    to the given shard row. Returns the groups and the candidate-ordered
    (row_a, row_b) pairs (j-major within a group, like the decode)."""
    gpi, gpo, gs, gso, cpairs, gsr = [], [0], [], [0], [0], []
    pairs = []
    for g, rows_of_group in enumerate(slot_rows):
        size = len(rows_of_group)
        gpi.append(g)
        gpo.append(len(gpi))
        gs.extend(range(size))
        gso.append(len(gs))
        gsr.extend(int(r) for r in rows_of_group)
        cpairs.append(cpairs[-1] + size * (size - 1) // 2)
        for j in range(size):
            for i in range(j):
                pairs.append((int(rows_of_group[i]), int(rows_of_group[j])))
    groups = K3PlusGroups(
        prefix_items=np.array(gpi, np.int32),
        prefix_offsets=np.array(gpo, np.int64),
        suffixes=np.array(gs, np.int32),
        suffix_offsets=np.array(gso, np.int64),
        cumulative_pairs=np.array(cpairs, np.int64),
        total_candidates=int(cpairs[-1]),
        groups=None,
        suffix_src_rows=np.array(gsr, np.int64),
    )
    return groups, pairs


def _reference(rows, pairs):
    return [np.intersect1d(rows[a], rows[b], assume_unique=True).astype(np.int32) for a, b in pairs]


@pytest.fixture(scope="module")
def fx():
    rng = np.random.default_rng(2026)
    rows: list[np.ndarray] = []
    base = _sorted_unique(rng, 5000)
    size_rows = list(range(len(SIZES)))
    rows += [_overlapping(rng, base, s, 0.5) for s in SIZES]
    x = _sorted_unique(rng, 300)
    special = list(range(len(rows), len(rows) + 8))
    rows += [
        x,
        x.copy(),  # identical content in a different row
        _overlapping(rng, x, 300, 0.0),  # disjoint from x
        np.array([7], np.int32),
        np.array([7], np.int32),
        np.array([9], np.int32),
        np.array([], np.int32),
        np.array([], np.int32),
    ]
    extra = list(range(len(rows), len(rows) + 27))
    rows += [_overlapping(rng, base, int(rng.integers(0, 400)), float(rng.random())) for _ in extra]
    rng.shuffle(size_rows)  # slot order != size order -> both probe/search branches
    groups, pairs = _groups([size_rows, special, extra[:2], extra[2:5], extra[5:10], extra[10:27]])
    assert groups.total_candidates % CANDS_PER_BLOCK != 0  # partial last block
    assert groups.total_candidates > 8 * CANDS_PER_BLOCK  # many blocks
    offsets_gpu, indices_gpu = _csr(rows)
    groups_gpu = upload_k3plus_groups(groups, 0, with_src_rows=True)
    return dict(
        rows=rows,
        groups=groups,
        pairs=pairs,
        ref=_reference(rows, pairs),
        off=offsets_gpu,
        idx=indices_gpu,
        ggpu=groups_gpu,
        tc=groups.total_candidates,
    )


def _ref_counts(fx):
    return np.array([len(r) for r in fx["ref"]], dtype=np.int32)


def test_fixture_covers_both_branches_and_long_rows(fx):
    lens = [(len(fx["rows"][a]), len(fx["rows"][b])) for a, b in fx["pairs"]]
    assert any(la < lb for la, lb in lens) and any(la > lb for la, lb in lens)
    assert any(la > 1024 and lb > 1024 for la, lb in lens), "both sides must span > 32 chunks somewhere"
    assert any(la == 0 and lb > 0 for la, lb in lens) and any(la > 0 and lb == 0 for la, lb in lens)
    assert any(la == 0 and lb == 0 for la, lb in lens)


def test_count_range_full(fx):
    got = count_csr_range(fx["off"], fx["idx"], fx["ggpu"], 0, fx["tc"]).get()
    np.testing.assert_array_equal(got, _ref_counts(fx))


@pytest.mark.parametrize("start,size", [(0, 5), (5, 16), (21, 60), (100, 23), (223 - 3, 3)])
def test_count_range_chunk_offsets(fx, start, size):
    size = min(size, fx["tc"] - start)
    got = count_csr_range(fx["off"], fx["idx"], fx["ggpu"], start, size).get()
    np.testing.assert_array_equal(got, _ref_counts(fx)[start : start + size])


def test_count_gather_random_ids(fx):
    rng = np.random.default_rng(1)
    ids = rng.integers(0, fx["tc"], size=300)  # unsorted, with duplicates
    got = count_csr_gather(fx["off"], fx["idx"], fx["ggpu"], cp.asarray(ids)).get()
    np.testing.assert_array_equal(got, _ref_counts(fx)[ids])


def _materialize(fx, ids):
    ids_gpu = cp.asarray(np.asarray(ids, dtype=np.int64))
    cnt = count_csr_gather(fx["off"], fx["idx"], fx["ggpu"], ids_gpu)
    out_off = cp.zeros(len(ids) + 1, dtype=cp.int64)
    if len(ids):
        cp.cumsum(cnt.astype(cp.int64), out=out_off[1:])
    out_idx = cp.empty(int(out_off[-1]), dtype=cp.int32)
    write_csr_gather(fx["off"], fx["idx"], fx["ggpu"], ids_gpu, out_off, out_idx)
    return out_off.get(), out_idx.get()


@pytest.mark.parametrize("subset", ["all", "random"])
def test_write_gather_exact(fx, subset):
    ids = np.arange(fx["tc"]) if subset == "all" else np.random.default_rng(3).integers(0, fx["tc"], size=77)
    out_off, out_idx = _materialize(fx, ids)
    expected = [fx["ref"][i] for i in ids]
    exp_off = np.zeros(len(ids) + 1, np.int64)
    np.cumsum([len(e) for e in expected], out=exp_off[1:])
    np.testing.assert_array_equal(out_off, exp_off)
    np.testing.assert_array_equal(out_idx, np.concatenate(expected) if expected else np.array([], np.int32))
    for k in range(len(ids)):  # strictly increasing within every written row
        seg = out_idx[out_off[k] : out_off[k + 1]]
        assert np.all(np.diff(seg) > 0)


def test_empty_id_list(fx):
    out_off, out_idx = _materialize(fx, np.array([], np.int64))
    assert out_off.tolist() == [0] and out_idx.shape == (0,)
    assert count_csr_gather(fx["off"], fx["idx"], fx["ggpu"], cp.asarray(np.array([], np.int64))).shape == (0,)
    assert count_csr_range(fx["off"], fx["idx"], fx["ggpu"], 0, 0).shape == (0,)


def test_out_of_range_rejected(fx):
    with pytest.raises(ValueError):
        count_csr_gather(fx["off"], fx["idx"], fx["ggpu"], cp.asarray(np.array([fx["tc"]], np.int64)))
    with pytest.raises(ValueError):
        count_csr_range(fx["off"], fx["idx"], fx["ggpu"], fx["tc"] - 1, 2)
    with pytest.raises(ValueError):
        count_csr_range(fx["off"], fx["idx"], {k: v for k, v in fx["ggpu"].items() if k != "gsr"}, 0, 1)


def test_end_to_end_with_real_builder():
    """Real K=2 prev itemsets -> Rust/numpy groups with suffix_src_rows ->
    kernel counts must equal host decode_k3plus_flat + np.intersect1d."""
    rng = np.random.default_rng(7)
    prev = set()
    for a in range(12):
        for b in rng.choice(np.arange(a + 1, 40), size=int(rng.integers(2, 7)), replace=False):
            prev.add((a, int(b)))
    flat = np.array(sorted(prev), np.int32)
    rng.shuffle(flat, axis=0)
    groups = build_k3plus_groups_from_flat(flat, with_src_rows=True)
    assert groups is not None and groups.suffix_src_rows is not None
    rows = [_sorted_unique(rng, int(rng.integers(0, 600)), universe=2000) for _ in range(len(flat))]
    off, idx = _csr(rows)
    ggpu = upload_k3plus_groups(groups, 0, with_src_rows=True)
    tc = groups.total_candidates
    got = count_csr_range(off, idx, ggpu, 0, tc).get()
    decoded = decode_k3plus_flat(np.arange(tc), groups, 3)
    row_of = {tuple(r): i for i, r in enumerate(flat.tolist())}
    expected = np.array(
        [
            len(np.intersect1d(rows[row_of[(p, si)]], rows[row_of[(p, sj)]], assume_unique=True))
            for p, si, sj in decoded.tolist()
        ],
        np.int32,
    )
    np.testing.assert_array_equal(got, expected)
    out_off, out_idx = None, None
    ids = cp.asarray(np.arange(tc, dtype=np.int64))
    cnt = count_csr_gather(off, idx, ggpu, ids)
    out_off = cp.zeros(tc + 1, dtype=cp.int64)
    cp.cumsum(cnt.astype(cp.int64), out=out_off[1:])
    out_idx = cp.empty(int(out_off[-1]), dtype=cp.int32)
    write_csr_gather(off, idx, ggpu, ids, out_off, out_idx)
    np.testing.assert_array_equal(
        out_idx.get(),
        np.concatenate(
            [np.intersect1d(rows[row_of[(p, si)]], rows[row_of[(p, sj)]], assume_unique=True) for p, si, sj in decoded.tolist()]
        ).astype(np.int32),
    )
