"""GPU equivalence tests: shared/tiled kernels vs the legacy kernels.

The shared variant must be BIT-IDENTICAL to legacy on the dense path
(same candidate indexing, same int32 array) and set-identical on the fused
path — across group sizes straddling the T=32 tile, empty/one-suffix
groups, odd word-tile tails, the all-zero-prefix early-exit, mega-group
legacy routing, and the fused overflow-retry protocol.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy", reason="cupy not installed")

pytestmark = pytest.mark.gpu

from et_miner.gpu.kernels.k2 import count_pairs_fused_k2, count_pairs_k2_allcounts
from et_miner.gpu.kernels.k3plus import K3PlusGroups, count_k3plus_allcounts, count_k3plus_fully_fused
from et_miner.gpu.kernels.shared_tiled import (
    TILE_T,
    compute_cumulative_tilepairs,
    count_k3plus_shared_fused,
    count_pairs_k2_shared,
    count_pairs_k2_shared_fused,
    count_shared_tiled_allcounts,
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
        legacy = count_k3plus_allcounts(bv, groups, n_u64s, variant="legacy").get()
        shared = count_shared_tiled_allcounts(bv, groups, n_u64s).get()
        np.testing.assert_array_equal(shared, legacy)

    def test_one_suffix_groups_contribute_nothing(self):
        bv = _random_bitvecs(100, 8)
        groups = _groups_from_sizes([1, 5, 1, 40, 1], prefix_len=1, n_cols=100)
        legacy = count_k3plus_allcounts(bv, groups, 8, variant="legacy").get()
        shared = count_shared_tiled_allcounts(bv, groups, 8).get()
        np.testing.assert_array_equal(shared, legacy)
        ctp = compute_cumulative_tilepairs(groups.suffix_offsets)
        assert int(np.diff(ctp)[0]) == 0  # pairless group owns no tile-pairs

    def test_empty_prefix_group_is_k2(self):
        """prefix_len=0 (the K=2 synthetic group shape): prefix AND = ~0."""
        bv = _random_bitvecs(120, 16, seed=3)
        cols = sorted(int(c) for c in np.random.default_rng(4).choice(120, 60, replace=False))
        legacy = count_pairs_k2_allcounts(bv, cols, 16, variant="legacy").get()
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
        legacy = count_k3plus_allcounts(bv, groups, 8, variant="legacy").get()
        np.testing.assert_array_equal(shared, legacy)

    def test_group_aligned_chunks_bit_equal(self):
        bv = _random_bitvecs(300, 12, seed=6)
        groups = _groups_from_sizes([10, 33, 64, 5, 90], prefix_len=2, n_cols=300, seed=7)
        cp_arr = np.asarray(groups.cumulative_pairs)
        full = count_k3plus_allcounts(bv, groups, 12, variant="legacy").get()
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


class TestFusedEquivalence:
    def test_k3_set_equal_with_boundary_counts(self):
        # Small item pool -> heavy prefix sharing -> varied group sizes.
        bv = _random_bitvecs(40, 10, seed=8, density=0.4)
        rng = np.random.default_rng(9)
        prev = sorted({tuple(sorted(int(c) for c in rng.choice(40, 3, replace=False))) for _ in range(400)})
        legacy_c, legacy_n = count_k3plus_fully_fused(bv, prev, 4, 10, min_count=5)
        shared_c, shared_n = count_k3plus_shared_fused(bv, prev, 4, 10, min_count=5)
        assert len(shared_c) > 0
        assert set(zip(map(tuple, legacy_c), map(int, legacy_n))) == set(
            zip(map(tuple, shared_c), map(int, shared_n))
        )

    def test_k2_fused_set_equal(self):
        bv = _random_bitvecs(150, 6, seed=10)
        cols = sorted(int(c) for c in np.random.default_rng(11).choice(150, 50, replace=False))
        legacy_p, legacy_n = count_pairs_fused_k2(bv, cols, 6, min_count=20)
        shared_p, shared_n = count_pairs_k2_shared_fused(bv, cols, 6, min_count=20)
        assert set(zip(map(tuple, legacy_p), map(int, legacy_n))) == set(
            zip(map(tuple, shared_p), map(int, shared_n))
        )

    def test_overflow_retry_returns_everything(self):
        """Force capacity overflow: the kernel keeps counting, the wrapper
        re-allocates to the exact reported size and re-runs — never truncates."""
        bv = _random_bitvecs(80, 4, seed=12, density=0.6)
        cols = list(range(60))
        baseline_p, baseline_n = count_pairs_k2_shared_fused(bv, cols, 4, min_count=1)
        assert len(baseline_p) > 10
        tiny_p, tiny_n = count_pairs_k2_shared_fused(bv, cols, 4, min_count=1, max_results=3)
        assert set(zip(map(tuple, baseline_p), map(int, baseline_n))) == set(
            zip(map(tuple, tiny_p), map(int, tiny_n))
        )


class TestVariantWiring:
    def test_allcounts_env_routing(self, monkeypatch):
        bv = _random_bitvecs(100, 8, seed=13)
        groups = _groups_from_sizes([20, 40], prefix_len=1, n_cols=100)
        results = {}
        for variant in ("legacy", "shared"):
            monkeypatch.setenv("ET_MINER_KERNEL_VARIANT", variant)
            results[variant] = count_k3plus_allcounts(bv, groups, 8).get()
        np.testing.assert_array_equal(results["legacy"], results["shared"])
