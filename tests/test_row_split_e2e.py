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


# ── Closed-itemset pruning on the sparse path (regression) ─────────────────
# Two datasets, each asserting sparse+prune == dense+prune == CPU+prune on
# exact itemsets AND counts. Both were run against the unfixed tree
# (main @ 3d94c52) first: Spec A's sparse leg fails (the CSR tidset rows
# are built before prune_closed filters current_flat, so rows misalign at
# the next level whenever a pruned row precedes a surviving one); Spec B's
# dense leg fails (the Rust closed-prune binary-searches a K=3 table that
# is emitted in j-major group order, not lex order, under-prunes, and then
# emits K=5 itemsets the CPU tier never generates).

PRUNE_BG_SPEC = SynthSpec(
    name="prune_bg",
    n_rows=50_000,
    vocab_size=1_500,
    zipf_a=1.3,
    row_len_mean=10,
    row_len_max=40,
    motif_count=4,
    motif_size=6,
    motif_penetration=0.015,
    min_support=0.004,
    seed=23,
)


def _spec_a_df() -> tuple[pl.DataFrame, float]:
    """Sparse-shaped background with item IDs shifted by +10 (so the block
    owns the lowest columns), plus a 1,000-row block [0..5] and 500
    single-item rows per block item. Block pairs are closed (1,000 vs 1,500
    singles) but every block triple equals its pairs (1,000), so closed
    pruning removes the 20 triples at K=3 — a sparse level under
    sparse_from_k=3 — from row positions 0..19, ahead of every survivor."""
    df, _ = generate_transactions(PRUNE_BG_SPEC)
    dtype = df.schema["items"]
    background = df.select(pl.col("items").list.eval(pl.element() + 10).cast(dtype))
    block = pl.DataFrame({"items": [[0, 1, 2, 3, 4, 5]] * 1000}).with_columns(pl.col("items").cast(dtype))
    singles = pl.DataFrame({"items": [[i] for i in range(6) for _ in range(500)]}).with_columns(
        pl.col("items").cast(dtype)
    )
    return pl.concat([background, block, singles]), PRUNE_BG_SPEC.min_support


def _spec_b_df(m: int = 12) -> tuple[pl.DataFrame, float]:
    """1,000 rows [0..m), 100 rows per pair and 100 rows per item: singles
    2,200, pairs 1,100, triples and deeper 1,000. Pairs and triples are
    closed, so the K=3 table is emitted in j-major group order (the K=2
    group (0,) has 11 suffixes → 55 triangular-order candidates) and
    pruning first fires at K=4, where every quad equals its triples. Block
    sizes 6 and 8 do not reproduce the under-pruning (lucky lookups still
    prune all or all-but-one quads); 12 does."""
    import itertools

    rows = [list(range(m))] * 1000
    rows += [[i, j] for i, j in itertools.combinations(range(m), 2) for _ in range(100)]
    rows += [[i] for i in range(m) for _ in range(100)]
    df = pl.DataFrame({"items": rows}, schema={"items": pl.List(pl.Int64)})
    return df, 900 / df.height


def _prune_legs(df: pl.DataFrame, min_support: float, *, sparse_from_k=3) -> tuple[set, set, set]:
    n = df.height
    kw = dict(min_support=min_support, item_col="items", prune_equal_support=True)
    cpu = _counted(apriori(df, **kw), n)
    dense = _counted(apriori(df, use_gpu=True, n_gpus=2, **kw), n)
    sparse = _counted(apriori(df, use_gpu=True, n_gpus=2, sparse_from_k=sparse_from_k, **kw), n)
    return cpu, dense, sparse


class TestClosedPruning:
    def test_spec_a_sparse_misalignment(self):
        df, min_support = _spec_a_df()
        cpu, dense, sparse = _prune_legs(df, min_support)
        assert len(cpu) > 0
        assert dense == cpu, "dense+prune diverged from CPU+prune"
        assert sparse == cpu, "sparse+prune diverged from CPU+prune (CSR rows misaligned after prune_closed)"

    def test_spec_b_dense_under_pruning(self):
        df, min_support = _spec_b_df()
        cpu, dense, sparse = _prune_legs(df, min_support)
        assert len(cpu) > 0
        assert dense == cpu, "dense+prune emitted itemsets CPU+prune never generates (closed-prune under-pruning)"
        assert sparse == cpu, "sparse+prune diverged from CPU+prune"


class TestTwoPhaseSparse:
    """The exact reported combination: mine_two_phase defaults to
    sparse_from_k="auto" and forces prune_equal_support=True."""

    def test_auto_prune_matches_cpu(self, tmp_path):
        from et_miner.gpu.row_split import mine_two_phase

        df, min_support = _spec_a_df()
        n = df.height
        out = tmp_path / "two_phase"
        out.mkdir()
        mine_two_phase(
            df,
            phase1_support=min_support,
            phase2_support=min_support,
            item_col="items",
            n_gpus=2,
            output_dir=str(out),
        )
        got = _counted_from_parquet_dir(out / "phase2", n)
        expected = _counted(apriori(df, min_support=min_support, item_col="items", prune_equal_support=True), n)
        assert got == expected
