"""GPU equivalence tests: shared/tiled and group kernels vs the per-candidate kernels.

The tiled dense kernel must be BIT-IDENTICAL to the per-candidate one (same
candidate indexing, same int32 array), and the fused tiled kernel must return
exactly the dense survivors (same indices, same counts) — across group sizes
straddling the T=32 tile, empty/one-suffix groups, odd word-tile tails, the
all-zero-prefix early-exit, and the fused overflow-retry protocol.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy", reason="cupy not installed")

pytestmark = pytest.mark.gpu

from et_miner.gpu.kernels.group_pairs import GROUP_MAX_SUFFIXES, count_group_pairs
from et_miner.gpu.kernels.k2 import count_pairs_k2_per_candidate
from et_miner.gpu.kernels.k3plus import K3PlusGroups, count_k3plus_per_candidate
from et_miner.gpu.kernels.shared_tiled import (
    TILE_T,
    compute_cumulative_tilepairs,
    count_pairs_k2_shared,
    count_shared_tiled_allcounts,
    count_tiled_fused,
    k2_groups,
)


def _random_bitvecs(n_cols: int, n_u64s: int, seed: int = 0, density: float = 0.3):
    rng = np.random.default_rng(seed)
    host = rng.random((n_cols, n_u64s * 64)) < density
    packed = np.zeros((n_cols, n_u64s), dtype=np.uint64)
    for w in range(64):
        packed |= host[:, w::64][:, :n_u64s].astype(np.uint64) << np.uint64(w)
    return cp.asarray(packed)


def _groups_from_sizes(sizes, prefix_len: int, n_cols: int, seed: int = 1) -> K3PlusGroups:
    """Synthetic prefix groups with given suffix sizes over random columns."""
    rng = np.random.default_rng(seed)
    gpi, gpo, gs, gso, cpairs = [], [0], [], [0], [0]
    for s in sizes:
        cols = rng.choice(n_cols, size=prefix_len + s, replace=False)
        gpi.extend(int(c) for c in cols[:prefix_len])
        gpo.append(len(gpi))
        gs.extend(int(c) for c in sorted(cols[prefix_len:]))
        gso.append(len(gs))
        cpairs.append(cpairs[-1] + s * (s - 1) // 2)
    return K3PlusGroups(
        prefix_items=np.array(gpi, dtype=np.int32),
        prefix_offsets=np.array(gpo, dtype=np.int64),
        suffixes=np.array(gs, dtype=np.int32),
        suffix_offsets=np.array(gso, dtype=np.int64),
        cumulative_pairs=np.array(cpairs, dtype=np.int64),
        total_candidates=cpairs[-1],
        groups=None,
    )


STRADDLE_SIZES = [2, 3, TILE_T - 1, TILE_T, TILE_T + 1, 2 * TILE_T + 5, 200]


class TestDenseEquivalence:
    @pytest.mark.parametrize("n_u64s", [1, 31, 32, 33, 65])
    def test_bit_equal_across_word_tails(self, n_u64s):
        bv = _random_bitvecs(300, n_u64s, seed=n_u64s)
        groups = _groups_from_sizes(STRADDLE_SIZES, prefix_len=2, n_cols=300)
        legacy = count_k3plus_per_candidate(bv, groups, n_u64s).get()
        shared = count_shared_tiled_allcounts(bv, groups, n_u64s).get()
        np.testing.assert_array_equal(shared, legacy)

    def test_one_suffix_groups_contribute_nothing(self):
        bv = _random_bitvecs(100, 8)
        groups = _groups_from_sizes([1, 5, 1, 40, 1], prefix_len=1, n_cols=100)
        legacy = count_k3plus_per_candidate(bv, groups, 8).get()
        shared = count_shared_tiled_allcounts(bv, groups, 8).get()
        np.testing.assert_array_equal(shared, legacy)
        ctp = compute_cumulative_tilepairs(groups.suffix_offsets)
        assert int(np.diff(ctp)[0]) == 0  # pairless group owns no tile-pairs

    def test_empty_prefix_group_is_k2(self):
        """prefix_len=0 (the K=2 synthetic group shape): prefix AND = ~0."""
        bv = _random_bitvecs(120, 16, seed=3)
        cols = sorted(int(c) for c in np.random.default_rng(4).choice(120, 60, replace=False))
        legacy = count_pairs_k2_per_candidate(bv, cols, 16).get()
        shared = count_pairs_k2_shared(bv, cols, 16).get()
        np.testing.assert_array_equal(shared, legacy)

    def test_all_zero_prefix_early_exit(self):
        """A prefix column with an all-zero bitvec must yield all-zero counts
        (and exercises the staged-tile skip path)."""
        bv = _random_bitvecs(50, 8, seed=5)
        bv[7] = 0  # the prefix column
        gpi = np.array([7], dtype=np.int32)
        groups = K3PlusGroups(
            prefix_items=gpi,
            prefix_offsets=np.array([0, 1], dtype=np.int64),
            suffixes=np.arange(10, 50, dtype=np.int32),
            suffix_offsets=np.array([0, 40], dtype=np.int64),
            cumulative_pairs=np.array([0, 40 * 39 // 2], dtype=np.int64),
            total_candidates=40 * 39 // 2,
            groups=None,
        )
        shared = count_shared_tiled_allcounts(bv, groups, 8).get()
        assert not shared.any()
        legacy = count_k3plus_per_candidate(bv, groups, 8).get()
        np.testing.assert_array_equal(shared, legacy)

    def test_group_aligned_chunks_bit_equal(self):
        bv = _random_bitvecs(300, 12, seed=6)
        groups = _groups_from_sizes([10, 33, 64, 5, 90], prefix_len=2, n_cols=300, seed=7)
        cp_arr = np.asarray(groups.cumulative_pairs)
        full = count_k3plus_per_candidate(bv, groups, 12).get()
        # chunk at every group boundary pairing
        for a in range(len(cp_arr) - 1):
            for b in range(a + 1, len(cp_arr)):
                start, end = int(cp_arr[a]), int(cp_arr[b])
                if end <= start:
                    continue
                got = count_shared_tiled_allcounts(bv, groups, 12, chunk_start=start, chunk_size=end - start).get()
                np.testing.assert_array_equal(got, full[start:end])

    def test_unaligned_chunk_rejected(self):
        bv = _random_bitvecs(100, 4)
        groups = _groups_from_sizes([10, 10], prefix_len=1, n_cols=100)
        with pytest.raises(ValueError, match="group-aligned"):
            count_shared_tiled_allcounts(bv, groups, 4, chunk_start=3, chunk_size=10)


def _dense_survivors(counts, min_count):
    counts = np.asarray(counts)
    idx = np.nonzero(counts >= min_count)[0].astype(np.int64)
    return idx, counts[idx].astype(np.int64)


class TestFusedEquivalence:
    @pytest.mark.parametrize("prefix_len", [1, 3])
    def test_fused_returns_exactly_the_dense_survivors(self, prefix_len):
        bv = _random_bitvecs(300, 10, seed=8, density=0.5)
        groups = _groups_from_sizes(STRADDLE_SIZES, prefix_len=prefix_len, n_cols=300)
        dense = count_k3plus_per_candidate(bv, groups, 10).get()
        min_count = int(np.median(dense))
        want_idx, want_cnt = _dense_survivors(dense, min_count)
        got_idx, got_cnt = count_tiled_fused(bv, groups, 10, min_count)
        assert len(want_idx) > 0
        np.testing.assert_array_equal(got_idx, want_idx)
        np.testing.assert_array_equal(got_cnt, want_cnt)

    def test_k2_fused_returns_exactly_the_dense_survivors(self):
        bv = _random_bitvecs(150, 6, seed=10)
        cols = sorted(int(c) for c in np.random.default_rng(11).choice(150, 50, replace=False))
        want_idx, want_cnt = _dense_survivors(count_pairs_k2_per_candidate(bv, cols, 6).get(), 20)
        got_idx, got_cnt = count_tiled_fused(bv, k2_groups(cols), 6, 20)
        np.testing.assert_array_equal(got_idx, want_idx)
        np.testing.assert_array_equal(got_cnt, want_cnt)

    def test_overflow_retry_returns_everything(self):
        """Force capacity overflow: the kernel keeps counting, the wrapper
        re-allocates to the exact reported size and re-runs — never truncates."""
        bv = _random_bitvecs(80, 4, seed=12, density=0.6)
        groups = k2_groups(list(range(60)))
        base_idx, base_cnt = count_tiled_fused(bv, groups, 4, 1)
        assert len(base_idx) > 10
        tiny_idx, tiny_cnt = count_tiled_fused(bv, groups, 4, 1, initial_capacity=3)
        np.testing.assert_array_equal(tiny_idx, base_idx)
        np.testing.assert_array_equal(tiny_cnt, base_cnt)


#: Group sizes around the group kernel's layouts (<= 16 suffixes: lanes are words; 17+: threads
#: are pairs) and its 64-suffix cap.
GROUP_SIZES = [0, 1, 2, 3, 5, 16, 17, 31, 32, 33, 48, 63, GROUP_MAX_SUFFIXES]


class TestGroupKernelEquivalence:
    """The group kernel writes exactly the per-candidate kernel's array."""

    @pytest.mark.parametrize("n_u64s", [1, 31, 32, 33, 65])
    @pytest.mark.parametrize("prefix_len", [0, 1, 3])
    def test_bit_equal_across_word_tails_and_layouts(self, n_u64s, prefix_len):
        bv = _random_bitvecs(300, n_u64s, seed=n_u64s)
        groups = _groups_from_sizes(GROUP_SIZES, prefix_len=prefix_len, n_cols=300, seed=prefix_len + 2)
        legacy = count_k3plus_per_candidate(bv, groups, n_u64s).get()
        np.testing.assert_array_equal(count_group_pairs(bv, groups, n_u64s).get(), legacy)

    def test_all_zero_prefix_tiles_count_zero(self):
        bv = _random_bitvecs(80, 70, seed=5)
        bv[3, :40] = 0  # the prefix column: the first word tile is all zero, the second partly
        groups = _groups_from_sizes([6, 40], prefix_len=0, n_cols=80, seed=9)
        groups = groups._replace(
            prefix_items=np.array([3, 3], dtype=np.int32), prefix_offsets=np.array([0, 1, 2], dtype=np.int64)
        )
        legacy = count_k3plus_per_candidate(bv, groups, 70).get()
        np.testing.assert_array_equal(count_group_pairs(bv, groups, 70).get(), legacy)

    def test_group_aligned_chunks_bit_equal(self):
        bv = _random_bitvecs(300, 12, seed=6)
        groups = _groups_from_sizes([10, 33, 64, 1, 5, 20], prefix_len=2, n_cols=300, seed=7)
        cp_arr = np.asarray(groups.cumulative_pairs)
        full = count_k3plus_per_candidate(bv, groups, 12).get()
        for a in range(len(cp_arr) - 1):
            for b in range(a + 1, len(cp_arr)):
                start, end = int(cp_arr[a]), int(cp_arr[b])
                if end <= start:
                    continue
                got = count_group_pairs(bv, groups, 12, chunk_start=start, chunk_size=end - start).get()
                np.testing.assert_array_equal(got, full[start:end])

    def test_unaligned_chunk_rejected(self):
        bv = _random_bitvecs(100, 4)
        groups = _groups_from_sizes([10, 10], prefix_len=1, n_cols=100)
        with pytest.raises(ValueError, match="group-aligned"):
            count_group_pairs(bv, groups, 4, chunk_start=3, chunk_size=10)

    def test_a_group_over_the_cap_is_rejected(self):
        bv = _random_bitvecs(200, 4)
        groups = _groups_from_sizes([5, GROUP_MAX_SUFFIXES + 1], prefix_len=1, n_cols=200)
        with pytest.raises(ValueError, match=f"at most {GROUP_MAX_SUFFIXES} suffixes"):
            count_group_pairs(bv, groups, 4)
        # a chunk of the groups within the cap is served
        first = int(groups.cumulative_pairs[1])
        np.testing.assert_array_equal(
            count_group_pairs(bv, groups, 4, chunk_start=0, chunk_size=first).get(),
            count_k3plus_per_candidate(bv, groups, 4, chunk_start=0, chunk_size=first).get(),
        )
