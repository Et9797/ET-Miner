"""What each apriori() route promises, and what it refuses.

apriori() takes a wide parameter set and dispatches to one of six routes, not
all of which implement all of it. Every mismatch here used to be silent: the
caller passed the parameter, the route dropped it, and nothing in the return
value or the logs said so. The negative tests assert the refusal; the positive
tests assert we did not over-restrict and break a combination that works.

`resume_from_k` and a direct `anchor_items` call had no coverage at all before
this file.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from et_miner import apriori


def _gpu_count() -> int:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = [
        sorted(rng.choice(12, size=int(rng.integers(3, 7)), replace=False).tolist())
        for _ in range(400)
    ]
    # nested vocabulary, so the free-set prune actually removes something
    return pl.DataFrame({"items": [r + [100] if 0 in r else r for r in rows]})


class TestUnsupportedCombinationsRaise:
    def test_streaming_with_pruning(self, df):
        """Measured before the fix: 214 rows (the complete lattice) where the
        gated answer is 176. The caller asked for free-sets and got everything."""
        with pytest.raises(ValueError, match="prune_equal_support"):
            apriori(df, min_support=0.05, streaming=True, chunk_size=100,
                    prune_equal_support=True)

    def test_streaming_with_generator_pruning(self, df):
        with pytest.raises(ValueError, match="use_generator_pruning"):
            apriori(df, min_support=0.05, streaming=True, chunk_size=100,
                    use_generator_pruning=True)

    def test_anchor_items_on_the_cpu_route(self, df):
        """It appeared in the signature, the routing test and the forwarded
        kwarg -- and nowhere in the CPU loop, so the caller got the complete,
        differently-shaped lattice."""
        with pytest.raises(ValueError, match="anchor_items"):
            apriori(df, min_support=0.05, anchor_items={0, 1})

    def test_anchor_items_on_the_streaming_route(self, df):
        with pytest.raises(ValueError, match="anchor_items"):
            apriori(df, min_support=0.05, streaming=True, chunk_size=100,
                    anchor_items={0, 1})

    def test_output_dir_on_the_cpu_route(self, df, tmp_path):
        """Documented as the per-K flush that "prevents CPU RAM OOM at K=7+
        scale". On this route it wrote nothing: an empty directory, the OOM it
        was meant to prevent, and no diagnostic."""
        with pytest.raises(ValueError, match="output_dir"):
            apriori(df, min_support=0.05, output_dir=str(tmp_path))

    def test_resume_from_k_on_the_cpu_route(self, df):
        """Silently re-mined from K=1 -- on a multi-day campaign."""
        with pytest.raises(ValueError, match="resume_from_k"):
            apriori(df, min_support=0.05, resume_from_k=3)

    def test_memory_budget_without_streaming(self, df):
        with pytest.raises(ValueError, match="memory_budget_gb"):
            apriori(df, min_support=0.05, memory_budget_gb=1.0)

    def test_profile_on_multi_gpu_streaming(self, df):
        """The worst of the group: a result frame has exactly two columns, so
        `result, session = apriori(...)` SUCCEEDED and handed back a column of
        itemsets and a column of floats, with `session.summary` absent."""
        with pytest.raises(ValueError, match="profile"):
            apriori(df, min_support=0.05, streaming=True, chunk_size=100,
                    n_gpus=2, profile=True)

    def test_profile_on_a_row_split_run(self, df):
        with pytest.raises(ValueError, match="profile"):
            apriori(df, min_support=0.05, use_gpu=True, n_gpus=2, profile=True)

    @pytest.mark.gpu
    def test_gpu_resident_with_pruning(self, df):
        """N1, not in the reviewed 62. _route_for_pruning is evaluated before
        the `if gpu_resident:` branch, so the call landed on row-split and
        gpu_resident was silently ignored -- measured 176 rows (the row-split
        free-sets) rather than the gpu-resident lattice."""
        with pytest.raises(ValueError, match="gpu_resident"):
            apriori(df, min_support=0.05, use_gpu=True, prune_equal_support=True,
                    gpu_resident=True)


class TestSupportedCombinationsStillWork:
    """The guard must reject only what a route genuinely cannot do."""

    def test_profile_on_the_cpu_route(self, df):
        from et_miner.core.profiling import ProfilingSession

        result, session = apriori(df, min_support=0.05, profile=True)
        assert isinstance(result, pl.DataFrame) and isinstance(session, ProfilingSession)

    def test_profile_on_single_gpu_streaming(self, df):
        result, session = apriori(df, min_support=0.05, streaming=True, chunk_size=100,
                                  profile=True)
        assert isinstance(result, pl.DataFrame)
        assert session is not None

    def test_memory_budget_reaches_the_multi_gpu_streaming_route(self, df):
        """#10 -- forwarded, not rejected. The callee had no such parameter and
        apriori() did not pass it, so a run sized its chunks against a budget the
        caller never chose, on the one path whose reason to exist is not
        exceeding memory."""
        from et_miner.streaming import multi_gpu as mg

        assert "memory_budget_gb" in inspect.signature(mg.apriori_streaming_multi_gpu).parameters

        seen: dict = {}
        real = mg.apriori_streaming_multi_gpu

        def spy(*a, **kw):
            seen.update(kw)
            raise RuntimeError("captured")

        mg.apriori_streaming_multi_gpu = spy
        try:
            with pytest.raises(RuntimeError, match="captured"):
                apriori(df, min_support=0.05, streaming=True, chunk_size=100,
                        n_gpus=2, memory_budget_gb=0.25)
        finally:
            mg.apriori_streaming_multi_gpu = real
        assert seen.get("memory_budget_gb") == 0.25

    @pytest.mark.gpu
    def test_anchor_items_on_the_gpu_route(self, df):
        got = apriori(df, min_support=0.05, use_gpu=True, anchor_items={0, 1})
        assert isinstance(got, pl.DataFrame)

    @pytest.mark.gpu
    def test_output_dir_on_the_row_split_route(self, df, tmp_path):
        apriori(df, min_support=0.05, use_gpu=True, prune_equal_support=True,
                output_dir=str(tmp_path))
        written = sorted(p.name for p in Path(tmp_path).iterdir())
        assert written, "row-split must flush per-K parquet"
        assert any(n.startswith("frequent_k") for n in written), written


@pytest.mark.gpu
class TestResumeFromK:
    """resume_from_k had ZERO coverage: the whole resume block -- parquet
    reload, the flat reshape, the free-set under-prune warning -- was untested."""

    def test_resume_reproduces_the_uninterrupted_run(self, df, tmp_path):
        full_dir = tmp_path / "full"
        resumed_dir = tmp_path / "resumed"
        full_dir.mkdir()
        resumed_dir.mkdir()

        apriori(df, min_support=0.05, use_gpu=True, prune_equal_support=True,
                output_dir=str(full_dir))

        # seed the resumed run with the levels the first run wrote, then resume
        for p in sorted(full_dir.glob("frequent_k*.parquet")):
            k = int(p.stem.split("frequent_k")[1])
            if k <= 2:
                (resumed_dir / p.name).write_bytes(p.read_bytes())

        apriori(df, min_support=0.05, use_gpu=True, prune_equal_support=True,
                output_dir=str(resumed_dir), resume_from_k=2)

        def levels(d: Path) -> dict[int, set]:
            out: dict[int, set] = {}
            for p in sorted(d.glob("frequent_k*.parquet")):
                k = int(p.stem.split("frequent_k")[1])
                out[k] = {
                    tuple(sorted(x)) for x in pl.read_parquet(p)["itemset"].to_list()
                }
            return out

        a, b = levels(full_dir), levels(resumed_dir)
        # Guard against a vacuous pass: two empty runs also compare equal.
        assert len(a) >= 3, f"fixture must reach K>=3 for resume to mean anything, got {sorted(a)}"
        assert sum(len(v) for v in a.values()) > 0
        assert set(a) == set(b), f"resumed run produced levels {sorted(b)}, expected {sorted(a)}"
        for k in sorted(a):
            assert a[k] == b[k], f"K={k} differs between the full and resumed runs"
