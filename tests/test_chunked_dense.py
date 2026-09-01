"""Tests for the dense-path chunk planning and byte-model budget.

The planner and budget-formula tests here are pure CPU (no marker) — they
validate the math the GPU chunk loop relies on, including at AlphaFold-scale
numbers. GPU equivalence tests for the chunked paths live in the gpu-marked
classes appended for the box campaign.
"""

import numpy as np
import pytest

from et_miner.gpu.kernels.filter import HOST_SORT_BYTES_PER_SURVIVOR
from et_miner.gpu.row_split_chunks import (
    CHUNK_BYTES_PER_CANDIDATE,
    ChunkPlan,
    chunk_budget_from_bytes,
    plan_candidate_chunks,
    plan_group_chunks,
)

GIB = 1 << 30


class TestChunkBudgetFormula:
    """The byte model, exercised at the scales that matter."""

    def test_h200_alphafold_scale(self):
        """8×H200 shape: 141 GiB cards, ~40 GiB resident group data."""
        avail = 90 * GIB  # measured free after bitmaps
        total = 141 * GIB
        group = 40 * GIB
        mc = chunk_budget_from_bytes(avail, total, group_data_bytes=group, use_nccl=True)
        margin = max(1 * GIB, int(total * 0.04))
        usable = avail - group - margin
        # The chunk's dense counts must fit with slack to spare.
        assert mc >= 1
        assert mc * CHUNK_BYTES_PER_CANDIDATE <= usable
        # And the budget is not absurdly conservative (>50% utilization).
        assert mc * CHUNK_BYTES_PER_CANDIDATE >= usable * 0.5

    def test_rtx3090_scale(self):
        """24 GiB card: the old hardcoded 6 GiB margin was 25% of the device."""
        avail = 20 * GIB
        total = 24 * GIB
        mc = chunk_budget_from_bytes(avail, total, group_data_bytes=2 * GIB, use_nccl=True)
        # 4% of 24 GiB < 1 GiB, so the floor applies — usable = 17 GiB.
        assert mc * CHUNK_BYTES_PER_CANDIDATE <= 17 * GIB
        assert mc > 1_000_000_000  # still billions of candidates per chunk

    def test_non_nccl_needs_more_headroom(self):
        """Today's fallback materializes a peer copy on GPU 0 — budget halves."""
        kw = dict(avail_bytes=20 * GIB, total_vram_bytes=24 * GIB, group_data_bytes=0)
        assert chunk_budget_from_bytes(**kw, use_nccl=False) < chunk_budget_from_bytes(**kw, use_nccl=True)

    def test_staging_replaces_peer_copy_term(self):
        """With a fixed staging buffer the per-candidate term drops back."""
        kw = dict(avail_bytes=20 * GIB, total_vram_bytes=24 * GIB, group_data_bytes=0, use_nccl=False)
        with_staging = chunk_budget_from_bytes(**kw, staging_bytes=512 * (1 << 20))
        without = chunk_budget_from_bytes(**kw)
        assert with_staging > without

    def test_env_cap_only_lowers(self):
        kw = dict(avail_bytes=20 * GIB, total_vram_bytes=24 * GIB)
        uncapped = chunk_budget_from_bytes(**kw)
        assert chunk_budget_from_bytes(**kw, env_cap=12_345) == 12_345
        assert chunk_budget_from_bytes(**kw, env_cap=uncapped * 100) == uncapped

    def test_exhausted_budget_still_progresses(self):
        """Negative usable never deadlocks the loop — floor is 1 candidate."""
        assert chunk_budget_from_bytes(1 * GIB, 24 * GIB, group_data_bytes=5 * GIB) == 1

    def test_small_pool_limit_keeps_usable_budget(self):
        """A tight per-device pool limit (the OOM-regression scenario) must
        shrink chunks, not collapse them: the margin is capped at a quarter
        of available, so ~512 MB of headroom still yields millions of
        candidates per chunk instead of a 1-candidate de-facto hang."""
        mc = chunk_budget_from_bytes(512 * (1 << 20), 24 * GIB)
        assert mc >= 10_000_000
        assert mc * CHUNK_BYTES_PER_CANDIDATE <= 512 * (1 << 20)

    def test_all_survivors_worst_case_host_model(self):
        """100%-survivor chunks: sort workspace is host-side, per the model.

        The budget deliberately does NOT reserve GPU/host worst-case survivor
        space per candidate — the filter checks feasibility per call and
        valves. This test pins the documented model numbers so a drive-by
        change to either constant breaks loudly.
        """
        assert CHUNK_BYTES_PER_CANDIDATE == 4  # int32 dense counts
        assert HOST_SORT_BYTES_PER_SURVIVOR == 48  # 12B pair + 8B argsort + 12B gather + slack

        # H200 box (2 TB host): even a full 8B-candidate chunk surviving 100%
        # sorts within host RAM.
        mc = chunk_budget_from_bytes(90 * GIB, 141 * GIB, group_data_bytes=40 * GIB)
        assert mc * HOST_SORT_BYTES_PER_SURVIVOR < 0.8 * (2 << 40)

        # 3090 box (64 GB host): a 100%-survivor chunk exceeds host workspace,
        # which is exactly when the filter must route to the sliced valve.
        mc3090 = chunk_budget_from_bytes(20 * GIB, 24 * GIB)
        assert mc3090 * HOST_SORT_BYTES_PER_SURVIVOR > 0.8 * (64 * GIB)


class TestPlanCandidateChunks:
    def test_empty(self):
        assert plan_candidate_chunks(0, 100) == []

    def test_single_chunk(self):
        assert plan_candidate_chunks(50, 100) == [ChunkPlan(0, 50)]

    def test_exact_multiple(self):
        plans = plan_candidate_chunks(200, 100)
        assert plans == [ChunkPlan(0, 100), ChunkPlan(100, 100)]

    def test_remainder(self):
        plans = plan_candidate_chunks(250, 100)
        assert plans[-1] == ChunkPlan(200, 50)

    def test_coverage_no_overlap(self):
        plans = plan_candidate_chunks(1_000_003, 4096)
        assert plans[0].start == 0
        assert all(a.start + a.size == b.start for a, b in zip(plans, plans[1:]))
        assert sum(p.size for p in plans) == 1_000_003

    def test_min_one(self):
        assert plan_candidate_chunks(3, 0) == [ChunkPlan(0, 1), ChunkPlan(1, 1), ChunkPlan(2, 1)]


def _cum(sizes):
    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)


class TestPlanGroupChunks:
    def test_empty(self):
        assert plan_group_chunks(_cum([]), 100) == []
        assert plan_group_chunks(_cum([0, 0]), 100) == []

    def test_all_fit_one_chunk(self):
        plans = plan_group_chunks(_cum([10, 20, 30]), 100, tiled_min_group_pairs=0)
        assert plans == [ChunkPlan(0, 60, use_legacy=False)]

    def test_boundaries_land_on_groups(self):
        sizes = [7, 11, 13, 17, 19, 23]
        cum = _cum(sizes)
        plans = plan_group_chunks(cum, 30, tiled_min_group_pairs=0)
        boundaries = set(int(x) for x in cum)
        for p in plans:
            assert p.start in boundaries
            assert p.start + p.size in boundaries
            assert p.size <= 30
            assert not p.use_legacy

    def test_exact_fit_boundary(self):
        plans = plan_group_chunks(_cum([5, 5, 5]), 10, tiled_min_group_pairs=0)
        assert plans == [ChunkPlan(0, 10, False), ChunkPlan(10, 5, False)]

    def test_mega_group_goes_legacy(self):
        # middle group (250 pairs) exceeds the 100-candidate budget
        plans = plan_group_chunks(_cum([40, 250, 40]), 100, tiled_min_group_pairs=0)
        legacy = [p for p in plans if p.use_legacy]
        assert [(p.start, p.size) for p in legacy] == [(40, 100), (140, 100), (240, 50)]
        aligned = [p for p in plans if not p.use_legacy]
        assert aligned == [ChunkPlan(0, 40, False), ChunkPlan(290, 40, False)]

    def test_group_just_over_budget(self):
        plans = plan_group_chunks(_cum([101]), 100)
        assert plans == [ChunkPlan(0, 100, True), ChunkPlan(100, 1, True)]

    def test_group_exactly_budget_not_legacy(self):
        assert plan_group_chunks(_cum([100]), 100) == [ChunkPlan(0, 100, False)]

    def test_coverage_property(self):
        rng = np.random.default_rng(7)
        sizes = rng.integers(1, 500, size=200)
        cum = _cum(sizes)
        plans = plan_group_chunks(cum, 777)  # default tiny-threshold: mixed classes
        assert plans[0].start == 0
        assert all(a.start + a.size == b.start for a, b in zip(plans, plans[1:]))
        assert sum(p.size for p in plans) == int(cum[-1])

    def test_deterministic(self):
        sizes = [3, 9, 1000, 4, 4, 4, 900, 2]
        a = plan_group_chunks(_cum(sizes), 50)
        b = plan_group_chunks(_cum(sizes), 50)
        assert a == b

    def test_budget_smaller_than_every_group(self):
        """max_cands=1: every multi-pair group becomes legacy sub-chunks."""
        plans = plan_group_chunks(_cum([2, 3]), 1)
        assert all(p.use_legacy for p in plans)
        assert sum(p.size for p in plans) == 5

    def test_tiny_groups_routed_legacy_by_default(self):
        """Groups under ET_MINER_TILED_MIN_GROUP_PAIRS (64) waste a
        256-thread tile-pair block — contiguous runs go legacy wholesale."""
        plans = plan_group_chunks(_cum([5, 5, 5]), 100)
        assert plans == [ChunkPlan(0, 15, use_legacy=True)]

    def test_mixed_tiny_and_mid_runs(self):
        plans = plan_group_chunks(_cum([5, 5, 200, 200, 5]), 500, tiled_min_group_pairs=64)
        assert plans == [
            ChunkPlan(0, 10, use_legacy=True),
            ChunkPlan(10, 400, use_legacy=False),
            ChunkPlan(410, 5, use_legacy=True),
        ]

    def test_k2_synthetic_single_group(self):
        """The K=2 pair space is planned as one synthetic group: tiled when
        whole-in-one-chunk, legacy sub-chunks when it exceeds the budget,
        legacy when trivially small."""
        assert plan_group_chunks(_cum([1000]), 10_000) == [ChunkPlan(0, 1000, False)]
        assert plan_group_chunks(_cum([30]), 10_000) == [ChunkPlan(0, 30, True)]
        mega = plan_group_chunks(_cum([25_000]), 10_000)
        assert all(p.use_legacy for p in mega) and sum(p.size for p in mega) == 25_000


class TestEnvCapAccessor:
    def test_unset_is_none(self, monkeypatch):
        monkeypatch.delenv("ET_MINER_MAX_CHUNK_CANDS", raising=False)
        from et_miner import _env

        assert _env.max_chunk_candidates() is None

    def test_set_parses(self, monkeypatch):
        monkeypatch.setenv("ET_MINER_MAX_CHUNK_CANDS", "100000")
        from et_miner import _env

        assert _env.max_chunk_candidates() == 100_000

    def test_filter_impl_default(self, monkeypatch):
        monkeypatch.delenv("ET_MINER_FILTER_IMPL", raising=False)
        from et_miner import _env

        assert _env.filter_impl() == "compact"


@pytest.mark.gpu
class TestChunkedEquivalenceGPU:
    """Forced multi-chunk runs must produce exactly the single-chunk result.

    ET_MINER_MAX_CHUNK_CANDS caps the measured budget, so tiny caps force
    many chunks on small data — including K>=3 chunk boundaries landing
    between prefix groups (group-aligned planner) and, with a cap smaller
    than a group, the legacy mega-group sub-chunk path.
    """

    @pytest.fixture(scope="class")
    def smoke_run(self):
        cp = pytest.importorskip("cupy")  # noqa: F841
        from et_miner.core.apriori import apriori
        from et_miner.synthetic import PRESETS, generate_transactions

        spec = PRESETS["smoke"]
        df, _ = generate_transactions(spec)

        def run():
            res = apriori(df, min_support=spec.min_support, item_col="items", use_gpu=True, n_gpus=2)
            return {
                (tuple(sorted(int(i) for i in s)), round(sup * spec.n_rows))
                for s, sup in zip(res["itemset"].to_list(), res["support"].to_list())
            }

        return run

    def test_unforced_baseline(self, smoke_run, monkeypatch):
        monkeypatch.delenv("ET_MINER_MAX_CHUNK_CANDS", raising=False)
        assert len(smoke_run()) > 0

    @pytest.mark.parametrize("cap", [50_000, 5_000, 700])
    def test_forced_chunks_equal_unforced(self, smoke_run, monkeypatch, cap):
        monkeypatch.delenv("ET_MINER_MAX_CHUNK_CANDS", raising=False)
        baseline = smoke_run()
        monkeypatch.setenv("ET_MINER_MAX_CHUNK_CANDS", str(cap))
        assert smoke_run() == baseline

    def test_forced_chunks_all_filter_impls_agree(self, smoke_run, monkeypatch):
        monkeypatch.setenv("ET_MINER_MAX_CHUNK_CANDS", "5000")
        results = []
        for impl in ("compact", "cupy", "cpu"):
            monkeypatch.setenv("ET_MINER_FILTER_IMPL", impl)
            results.append(smoke_run())
        assert results[0] == results[1] == results[2]
