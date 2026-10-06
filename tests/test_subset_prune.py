"""The counting kernels' (k-1)-subset test, against counts computed with NumPy.

Each K>=3 counting kernel (per-candidate, group, tiled dense, tiled fused, sparse CSR)
counts one level of a small dataset in which items imply their parents, with an
index of the complete previous level. With SUBSET_PRUNE a candidate with an
infrequent (k-1)-subset is not counted; with SUBSET_INFER a candidate with a
non-free subset gets the minimum of its subset counts, written only where
``write_inferred`` is set. Survivors and their counts never change.
"""

from __future__ import annotations

import numpy as np
import pytest

N_ROWS = 3000
N_ITEMS = 70
MIN_COUNT = 30


def _dataset(seed: int = 3):
    """Boolean (rows, items): items >= 6 imply item % 6, so many pairs are not
    free; 70 items give prefix groups wider than one 32-suffix tile, whose
    rare-by-rare tile-pairs hold no frequent pair."""
    rng = np.random.default_rng(seed)
    weights = 1.0 / np.arange(1, N_ITEMS + 1) ** 0.9
    weights /= weights.sum()
    m = np.zeros((N_ROWS, N_ITEMS), dtype=bool)
    for r in range(N_ROWS):
        items = rng.choice(N_ITEMS, size=int(rng.integers(2, 9)), replace=False, p=weights)
        m[r, items] = True
        m[r, [i % 6 for i in items if i >= 6]] = True
    return m


def _count(m, itemset) -> int:
    return int(m[:, list(itemset)].all(axis=1).sum())


def _level(m, k):
    """Frequent k-itemsets (lexicographic) and their counts, grown level by level."""
    level = [(i,) for i in range(N_ITEMS) if _count(m, (i,)) >= MIN_COUNT]
    for _ in range(k - 1):
        level = [s + (x,) for s in level for x in range(s[-1] + 1, N_ITEMS) if _count(m, s + (x,)) >= MIN_COUNT]
    counts = [_count(m, s) for s in level]
    return np.array(level, dtype=np.int32).reshape(-1, k), np.array(counts, dtype=np.int64)


def _free(m, rows, counts):
    out = []
    for row, c in zip(rows, counts):
        subs = [_count(m, [x for x in row if x != y]) if len(row) > 1 else N_ROWS for y in row]
        out.append(all(s != c for s in subs))
    return np.array(out, dtype=bool)


def _bitvecs(m):
    words = (N_ROWS + 63) // 64
    bits = np.zeros((N_ITEMS, words), dtype=np.uint64)
    for col in range(N_ITEMS):
        for r in np.nonzero(m[:, col])[0]:
            bits[col, r // 64] |= np.uint64(1) << np.uint64(r % 64)
    return bits


@pytest.fixture(scope="module", params=[3, 4])
def level(request):
    """Level k's candidates with their true counts and labels (P, I) against level k-1."""
    from et_miner.gpu.kernels import build_k3plus_groups_from_flat, decode_k3plus_flat

    k = request.param
    m = _dataset()
    prev, prev_counts = _level(m, k - 1)
    prev_free = _free(m, prev, prev_counts)
    groups = build_k3plus_groups_from_flat(prev, with_src_rows=True)
    cands = decode_k3plus_flat(np.arange(groups.total_candidates), groups, k)
    lookup = {tuple(r): i for i, r in enumerate(prev.tolist())}
    true, prunable, inferable, inferred = [], [], [], []
    for cand in cands.tolist():
        true.append(_count(m, cand))
        subs = [tuple(x for x in cand if x != y) for y in cand]
        rows = [lookup.get(s) for s in subs]
        prunable.append(any(r is None for r in rows[:-2]))
        found = [r for r in rows if r is not None]
        inferable.append(not prunable[-1] and not all(prev_free[r] for r in found))
        inferred.append(min(prev_counts[r] for r in found))
    labels = {
        "true": np.array(true),
        "P": np.array(prunable),
        "I": np.array(inferable),
        "inferred": np.array(inferred),
    }
    assert labels["P"].any() and labels["I"].any(), "the fixture must exercise both labels"
    assert not (labels["true"][labels["P"]] >= MIN_COUNT).any()
    assert (labels["inferred"][labels["I"]] == labels["true"][labels["I"]]).all()
    return k, m, prev, prev_counts, prev_free, groups, labels


def _index(prev, prev_counts, prev_free, mode, write_inferred):
    from et_miner.gpu.kernels import upload_subset_index

    return upload_subset_index(prev, prev_counts, prev_free, mode=mode, device_id=0, write_inferred=write_inferred)


def _expected_exact(labels, mode, write_inferred):
    """Per-candidate kernels: P -> 0; I -> inferred or 0 (with SUBSET_INFER); else the count."""
    from et_miner.gpu.kernels import SUBSET_INFER

    out = labels["true"].copy()
    out[labels["P"]] = 0
    if mode & SUBSET_INFER:
        out[labels["I"]] = labels["inferred"][labels["I"]] if write_inferred else 0
    return out


MODES = [("prune", False), ("infer", True), ("infer", False)]


@pytest.mark.gpu
@pytest.mark.parametrize("mode_name,write_inferred", MODES)
def test_per_candidate_kernel(level, mode_name, write_inferred):
    import cupy as cp

    from et_miner.gpu.kernels import SUBSET_INFER, SUBSET_PRUNE, count_k3plus_per_candidate

    k, m, prev, prev_counts, prev_free, groups, labels = level
    mode = SUBSET_PRUNE | (SUBSET_INFER if mode_name == "infer" else 0)
    bv = cp.asarray(_bitvecs(m))
    got = count_k3plus_per_candidate(
        bv, groups, bv.shape[1], index=_index(prev, prev_counts, prev_free, mode, write_inferred)
    ).get()
    np.testing.assert_array_equal(got, _expected_exact(labels, mode, write_inferred))
    plain = count_k3plus_per_candidate(bv, groups, bv.shape[1]).get()
    np.testing.assert_array_equal(plain, labels["true"])


@pytest.mark.gpu
@pytest.mark.parametrize("mode_name,write_inferred", MODES)
def test_sparse_csr_kernel(level, mode_name, write_inferred):
    import cupy as cp

    from et_miner.gpu.kernels import SUBSET_INFER, SUBSET_PRUNE, count_csr_range, upload_k3plus_groups
    from et_miner.gpu.sparse_csr import convert_shards_to_csr

    k, m, prev, prev_counts, prev_free, groups, labels = level
    mode = SUBSET_PRUNE | (SUBSET_INFER if mode_name == "infer" else 0)
    bv = cp.asarray(_bitvecs(m))
    (shard,) = convert_shards_to_csr([(bv, 0, N_ROWS)], prev, prev_counts)
    groups_gpu = upload_k3plus_groups(groups, 0, with_src_rows=True)
    got = count_csr_range(
        shard.offsets, shard.indices, groups_gpu, 0, groups.total_candidates,
        index=_index(prev, prev_counts, prev_free, mode, write_inferred),
    ).get()
    np.testing.assert_array_equal(got, _expected_exact(labels, mode, write_inferred))


@pytest.mark.gpu
@pytest.mark.parametrize("mode_name,write_inferred", MODES)
def test_tiled_dense_kernel(level, mode_name, write_inferred):
    """A tile-pair is skipped only when none of its pairs needs counting; a counted
    tile-pair writes true counts for every pair, so each slot is either the true
    count or what the per-candidate kernel writes, and some tile-pair is skipped."""
    import cupy as cp

    from et_miner.gpu.kernels import SUBSET_INFER, SUBSET_PRUNE, count_shared_tiled_allcounts

    k, m, prev, prev_counts, prev_free, groups, labels = level
    mode = SUBSET_PRUNE | (SUBSET_INFER if mode_name == "infer" else 0)
    bv = cp.asarray(_bitvecs(m))
    got = count_shared_tiled_allcounts(
        bv, groups, bv.shape[1], index=_index(prev, prev_counts, prev_free, mode, write_inferred)
    ).get()
    exact = _expected_exact(labels, mode, write_inferred)
    assert ((got == labels["true"]) | (got == exact)).all()
    counted = ~labels["P"] & ~(labels["I"] if mode & SUBSET_INFER else False)
    np.testing.assert_array_equal(got[counted], labels["true"][counted])
    assert (got != labels["true"]).any(), "no tile-pair was skipped"


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", ["per-candidate", "tiled"])
@pytest.mark.parametrize("mode_name", ["prune", "infer"])
def test_untouched_marks_the_same_entries_on_every_device(level, kernel, mode_name):
    """With ``untouched=UNTOUCHED`` the writer flag changes values, never which
    entries are written (the compacted reduce relies on it); the written values
    are the zero-filled output's, and only prunable candidates stay unwritten
    (all of them on the per-candidate kernel, those of skipped tile-pairs on the
    tiled one)."""
    import cupy as cp

    from et_miner.gpu.kernels import (
        SUBSET_INFER,
        SUBSET_PRUNE,
        UNTOUCHED,
        count_k3plus_per_candidate,
        count_shared_tiled_allcounts,
    )

    k, m, prev, prev_counts, prev_free, groups, labels = level
    mode = SUBSET_PRUNE | (SUBSET_INFER if mode_name == "infer" else 0)
    count = count_k3plus_per_candidate if kernel == "per-candidate" else count_shared_tiled_allcounts
    bv = cp.asarray(_bitvecs(m))
    written = None
    for writer in (True, False):
        index = _index(prev, prev_counts, prev_free, mode, writer)
        marked = count(bv, groups, bv.shape[1], index=index, untouched=UNTOUCHED).get()
        zero_filled = count(bv, groups, bv.shape[1], index=index).get()
        if written is None:
            written = marked != UNTOUCHED
        np.testing.assert_array_equal(marked != UNTOUCHED, written)
        np.testing.assert_array_equal(np.where(written, marked, 0), zero_filled)
    assert (~written).any()
    if kernel == "per-candidate":
        np.testing.assert_array_equal(~written, labels["P"])
    else:
        assert labels["P"][~written].all()


@pytest.mark.gpu
@pytest.mark.parametrize("mode_name,write_inferred", MODES)
@pytest.mark.parametrize("untouched", [0, -1])
def test_group_kernel_writes_the_per_candidate_kernels_entries(level, mode_name, write_inferred, untouched):
    """On the groups within its cap, the group kernel writes exactly what the per-candidate
    kernel writes: the same values and, with an ``untouched`` marker, the same entries."""
    import cupy as cp

    from et_miner.gpu.kernels import (
        GROUP_MAX_SUFFIXES,
        SUBSET_INFER,
        SUBSET_PRUNE,
        count_group_pairs,
        count_k3plus_per_candidate,
        select_k3plus_groups,
    )

    k, m, prev, prev_counts, prev_free, groups, labels = level
    mode = SUBSET_PRUNE | (SUBSET_INFER if mode_name == "infer" else 0)
    capped = select_k3plus_groups(groups, np.diff(groups.suffix_offsets) <= GROUP_MAX_SUFFIXES)
    assert 0 < capped.total_candidates
    bv = cp.asarray(_bitvecs(m))
    index = _index(prev, prev_counts, prev_free, mode, write_inferred)
    want = count_k3plus_per_candidate(bv, capped, bv.shape[1], index=index, untouched=untouched).get()
    got = count_group_pairs(bv, capped, bv.shape[1], index=index, untouched=untouched).get()
    np.testing.assert_array_equal(got, want)
    if untouched:
        assert (got == untouched).any(), "no candidate was skipped"


@pytest.mark.gpu
@pytest.mark.parametrize("mode_name", ["prune", "infer"])
def test_tiled_fused_kernel(level, mode_name):
    import cupy as cp

    from et_miner.gpu.kernels import SUBSET_INFER, SUBSET_PRUNE, count_tiled_fused

    k, m, prev, prev_counts, prev_free, groups, labels = level
    mode = SUBSET_PRUNE | (SUBSET_INFER if mode_name == "infer" else 0)
    bv = cp.asarray(_bitvecs(m))
    idx, cnt = count_tiled_fused(bv, groups, bv.shape[1], MIN_COUNT,
                                 index=_index(prev, prev_counts, prev_free, mode, True))
    frequent = np.nonzero(labels["true"] >= MIN_COUNT)[0]
    np.testing.assert_array_equal(idx, frequent)
    np.testing.assert_array_equal(cnt, labels["true"][frequent])


def test_index_rejects_unsorted_rows():
    pytest.importorskip("cupy")
    from et_miner.gpu.kernels import SUBSET_PRUNE, upload_subset_index

    with pytest.raises(ValueError, match="ascending"):
        upload_subset_index(np.array([[1, 2], [0, 3]]), np.array([5, 5]), mode=SUBSET_PRUNE, device_id=0,
                            write_inferred=True)


def test_all_subsets_in_keeps_rows_whose_subsets_are_in_the_level():
    from et_miner.gpu.mining import _all_subsets_in

    level = np.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=np.int32)
    current = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3]], dtype=np.int32)
    assert _all_subsets_in(current, level).tolist() == [True, False, False]
