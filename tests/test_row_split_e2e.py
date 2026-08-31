"""End-to-end GPU tests for the multi-GPU row-split path.

First test coverage for _apriori_row_split_multi_gpu and mine_two_phase.
All tests run on 1 GPU (row-split degenerates to one shard) and scale to 2
where marked; the CPU tier is the correctness reference (itself gated
against efficient-apriori by test_tier_equivalence.py).
"""

import polars as pl
import pytest

cp = pytest.importorskip("cupy", reason="cupy not installed")

pytestmark = pytest.mark.gpu

from et_miner.core.apriori import apriori
from et_miner.synthetic import PRESETS, SynthSpec, estimate_level_sizes, generate_transactions

SPEC = PRESETS["smoke"]


def _gpu_count() -> int:
    try:
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def _counted(df: pl.DataFrame, n_rows: int) -> set:
    return {
        (tuple(sorted(int(i) for i in itemset)), round(sup * n_rows))
        for itemset, sup in zip(df["itemset"].to_list(), df["support"].to_list())
    }


def _counted_from_parquet_dir(directory, n_rows: int) -> set:
    # Branch on is_file/is_dir rather than a trailing-slash glob: Python
    # 3.10's pathlib has no directory-only glob matching, so "frequent_k*/"
    # would re-match the plain .parquet files (and then read zero parts).
    out = set()
    for p in sorted(directory.glob("frequent_k*")):
        if p.is_file() and p.suffix == ".parquet":
            out |= _counted(pl.read_parquet(p), n_rows)
        elif p.is_dir():
            parts = sorted(p.glob("part_*.parquet"))
            if parts:
                out |= _counted(pl.read_parquet(parts), n_rows)
    return out


@pytest.fixture(scope="module")
def smoke_df():
    df, data = generate_transactions(SPEC)
    return df, data


@pytest.fixture(scope="module")
def cpu_reference(smoke_df) -> set:
    df, _ = smoke_df
    return _counted(apriori(df, min_support=SPEC.min_support, item_col="items"), SPEC.n_rows)


def _mine_row_split(df, **kwargs):
    """n_gpus=2 routes apriori() to the row-split path; the builder clamps
    to the devices actually present, so this exercises row-split even on a
    single-GPU box."""
    return apriori(df, min_support=SPEC.min_support, item_col="items", use_gpu=True, n_gpus=2, **kwargs)


class TestRowSplitEquivalence:
    def test_matches_cpu_reference(self, smoke_df, cpu_reference):
        df, _ = smoke_df
        assert _counted(_mine_row_split(df), SPEC.n_rows) == cpu_reference

    def test_two_gpu_matches_single_gpu_path(self, smoke_df):
        if _gpu_count() < 2:
            pytest.skip("needs 2 CUDA devices")
        df, _ = smoke_df
        single = apriori(df, min_support=SPEC.min_support, item_col="items", use_gpu=True)
        assert _counted(_mine_row_split(df), SPEC.n_rows) == _counted(single, SPEC.n_rows)

    def test_nccl_fallback_forced(self, smoke_df, cpu_reference, monkeypatch):
        """ET_MINER_DISABLE_NCCL=1 exercises the staged D2D reduce."""
        monkeypatch.setenv("ET_MINER_DISABLE_NCCL", "1")
        df, _ = smoke_df
        assert _counted(_mine_row_split(df), SPEC.n_rows) == cpu_reference

    def test_output_dir_parquet_equivalence(self, smoke_df, cpu_reference, tmp_path):
        df, _ = smoke_df
        _mine_row_split(df, output_dir=str(tmp_path))
        assert _counted_from_parquet_dir(tmp_path, SPEC.n_rows) == cpu_reference

    def test_planted_motifs_recovered_on_gpu(self, smoke_df):
        df, data = smoke_df
        counts = dict(_counted(_mine_row_split(df), SPEC.n_rows))
        for motif, planted in data.planted:
            got = counts.get(motif)
            assert got is not None and got >= planted, f"motif {motif}: {got} < planted {planted}"


class TestMineTwoPhase:
    def test_equal_supports_match_plain_row_split(self, smoke_df, tmp_path):
        """With phase2_support == phase1_support every frequent item is an
        anchor, so phase 2 must equal a plain row-split run with the same
        pruning flags (mine_two_phase forces prune_equal_support=True)."""
        from et_miner.gpu.row_split import mine_two_phase

        df, _ = smoke_df
        two_phase_dir = tmp_path / "two_phase"
        plain_dir = tmp_path / "plain"
        two_phase_dir.mkdir()
        plain_dir.mkdir()

        mine_two_phase(
            df,
            phase1_support=SPEC.min_support,
            phase2_support=SPEC.min_support,
            item_col="items",
            n_gpus=2,
            output_dir=str(two_phase_dir),
        )
        apriori(
            df,
            min_support=SPEC.min_support,
            item_col="items",
            use_gpu=True,
            n_gpus=2,
            prune_equal_support=True,
            output_dir=str(plain_dir),
        )
        got = _counted_from_parquet_dir(two_phase_dir / "phase2", SPEC.n_rows)
        expected = _counted_from_parquet_dir(plain_dir, SPEC.n_rows)
        assert got == expected


# Bespoke spec for the OOM regression: big enough that its dense K=2 counts
# dwarf the pool headroom we grant, small enough to finish in minutes.
OOM_SPEC = SynthSpec(
    name="oom_pytest",
    n_rows=200_000,
    vocab_size=12_000,
    zipf_a=0.75,
    row_len_mean=16,
    row_len_max=48,
    min_support=0.00005,
    seed=17,
)


@pytest.mark.slow
class TestOOMRegression:
    def test_completes_under_pool_limit(self, monkeypatch):
        """Budget-driven chunking must keep every allocation inside a CuPy
        pool limit set on EVERY participating device (pools are per-device).
        A violation raises cupy OutOfMemoryError — completion IS the proof —
        and the result must equal the unlimited run exactly."""
        est = estimate_level_sizes(OOM_SPEC)
        assert est["expected_k2_candidates"] > 10_000_000, "spec drifted: too small to stress chunking"

        df, _ = generate_transactions(OOM_SPEC)
        # max_length=2: the stress target is the chunked K=2 dense path
        # (72M candidates) under a tight pool limit; unbounded depth at
        # min_count=10 would burn box time without adding coverage (K>=3
        # chunk equivalence is covered by TestChunkedEquivalenceGPU).
        kwargs = dict(
            min_support=OOM_SPEC.min_support, item_col="items", use_gpu=True, n_gpus=2, max_length=2
        )

        unlimited = _counted(apriori(df, **kwargs), OOM_SPEC.n_rows)

        # Grant each device only ~512 MB of pool headroom beyond what the
        # bitvec build already used, and force small chunks so several
        # count/reduce/filter rounds must fit inside it.
        monkeypatch.setenv("ET_MINER_MAX_CHUNK_CANDS", "8000000")
        n_dev = min(2, _gpu_count())
        limits = []
        try:
            for d in range(n_dev):
                with cp.cuda.Device(d):
                    pool = cp.get_default_memory_pool()
                    pool.free_all_blocks()
                    limit = pool.used_bytes() + (512 << 20)
                    pool.set_limit(size=limit)
                    limits.append(limit)
            limited = _counted(apriori(df, **kwargs), OOM_SPEC.n_rows)
        finally:
            for d in range(n_dev):
                with cp.cuda.Device(d):
                    cp.get_default_memory_pool().set_limit(size=0)  # restore unlimited

        assert limited == unlimited
        assert len(limited) > 0
