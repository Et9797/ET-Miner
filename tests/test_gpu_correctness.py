"""GPU-path correctness: anchoring, result truncation, the K cap, memory guards, routing.

Every defect here ends the same way — an incomplete lattice handed back through
a value-returning API, with at most a log line. A miner whose contract is
exactness must raise, or restrict only what it *reports*; never return a bare
truncated result.
"""

from __future__ import annotations

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
def nested_df() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(6000):
        r = set(rng.choice(20, size=int(rng.integers(4, 9)), replace=False).tolist())
        if 3 in r:
            r.add(2)  # nested vocabulary: a child implies its parent
        if 5 in r:
            r.add(4)
        rows.append(sorted(r))
    return pl.DataFrame({"items": rows})


@pytest.mark.gpu
class TestAnchorIsAnOutputSelector:
    """#22 -- the anchor mask used to filter `current_flat`, and that one array
    became BOTH the subset oracle (full_flat) and the generation base
    (prev_frequent_flat). Unsound twice over: the apriori oracle must test the
    (k-1)-subsets that DROP the anchor, which are unanchored by construction;
    and the prefix-join needs the family closed under its two prefix-parents,
    which an anchored candidate's parents need not be.

    Anchoredness is not anti-monotone. That single property is why the same code
    shape is sound for the free-set prune and catastrophic here.
    """

    @pytest.mark.parametrize("anchors", [{16, 17, 18, 19}, {0, 1, 2, 3}, {9, 10}],
                             ids=["high-sorting", "low-sorting", "middle"])
    @pytest.mark.parametrize("gates", [False, True], ids=["gates-off", "gates-on"])
    def test_every_anchored_itemset_is_returned(self, nested_df, anchors, gates):
        """Anchor sort position determines WHICH channel dominates, never
        whether the feature works -- so all three positions are checked. Both
        gate settings, because apriori() ties the two flags and gates=ON is what
        the public API actually produces."""
        full = apriori(nested_df, min_support=0.05, max_length=5, use_gpu=True,
                       prune_equal_support=gates)
        expected = {
            tuple(sorted(s)) for s in full["itemset"].to_list() if anchors & set(s)
        }
        got = {
            tuple(sorted(s))
            for s in apriori(nested_df, min_support=0.05, max_length=5, use_gpu=True,
                             prune_equal_support=gates,
                             anchor_items=anchors)["itemset"].to_list()
        }
        assert expected, "fixture produced no anchored itemsets"
        assert not (expected - got), f"{len(expected - got)} anchored itemsets missing"
        assert not (got - expected), f"{len(got - expected)} unexpected itemsets emitted"

    def test_no_unanchored_itemset_is_emitted(self, nested_df):
        """Including at K=1, which used to be exempt: _anchor_keep_mask returned
        None for k<2, so a two-phase run emitted every frequent item at K=1
        while filtering every deeper level."""
        anchors = {16, 17, 18, 19}
        got = apriori(nested_df, min_support=0.05, max_length=5, use_gpu=True,
                      anchor_items=anchors)["itemset"].to_list()
        leaked = [list(s) for s in got if not (anchors & set(s))]
        assert not leaked, f"{len(leaked)} unanchored itemsets emitted, e.g. {leaked[:5]}"

    @pytest.mark.parametrize("n_gpus", [1, 2])
    def test_sparse_csr_branch(self, nested_df, n_gpus):
        """The sparse branch is separate code: the deleted filter there also
        filtered the survivor index array in lockstep, and that array feeds
        materialize_survivors, whose per-shard length check RAISES."""
        if n_gpus > _gpu_count():
            pytest.skip(f"needs {n_gpus} CUDA devices")
        anchors = {16, 17, 18, 19}
        kw = dict(min_support=0.05, max_length=5, use_gpu=True, n_gpus=n_gpus,
                  prune_equal_support=True, sparse_from_k=3)
        full = apriori(nested_df, **kw)
        expected = {tuple(sorted(s)) for s in full["itemset"].to_list() if anchors & set(s)}
        got = {tuple(sorted(s))
               for s in apriori(nested_df, anchor_items=anchors, **kw)["itemset"].to_list()}
        assert got == expected

    def test_the_anchored_run_reaches_every_level_the_full_lattice_anchors(self, nested_df):
        """The generation base must stay unrestricted, so the anchored run must
        report an itemset at every K where the FULL lattice has one containing
        an anchor.

        Note this is not "the same deepest level as the unanchored run": the
        deepest reported level can legitimately be shallower, because there may
        simply be no anchored itemset that deep. Asserting equality with the
        unanchored depth would encode an assumption about the fixture rather
        than the property -- and it fails on this one for exactly that reason.
        """
        anchors = {16, 17, 18, 19}
        full = apriori(nested_df, min_support=0.05, max_length=5, use_gpu=True)
        anchored = apriori(nested_df, min_support=0.05, max_length=5, use_gpu=True,
                           anchor_items=anchors)

        want = {len(s) for s in full["itemset"].to_list() if anchors & set(s)}
        got = {len(s) for s in anchored["itemset"].to_list()}
        assert max(want) >= 3, f"fixture must anchor at K>=3 to be meaningful, got {sorted(want)}"
        assert got == want, f"levels reported {sorted(got)}, expected {sorted(want)}"


class TestResultTruncationRaises:
    """#23/#24 -- four multi-GPU kernels clamped with a bare min(n, gpu_max)
    where their single-GPU siblings called the helper, and the helper itself
    tolerated 5% loss with a logger.warning. Neither fix works alone: calling
    the helper at four more sites while it still tolerated 5% would merely have
    extended the silent-loss window to all eight."""

    def test_any_overflow_raises(self):
        from et_miner.gpu.kernels.loader import _warn_result_truncation

        assert _warn_result_truncation(100, 100, "exact fit") == 100
        with pytest.raises(RuntimeError, match="Result truncation"):
            _warn_result_truncation(9880, 9583, "3% overflow")  # was tolerated
        with pytest.raises(RuntimeError, match="Result truncation"):
            _warn_result_truncation(10_000_001, 10_000_000, "one over")

    def test_the_message_names_only_reachable_remedies(self):
        """It lands hours into a run, so it has to be actionable ON THIS ROUTE.

        The first version advised `resume_from_k` and raising `max_results`.
        Neither is reachable here: these kernels run under
        `_apriori_from_bitvecs`, where `max_results` is not a public apriori()
        parameter and `_validate_route_support` refuses both `resume_from_k` and
        `output_dir`. So the message told a user hours into a campaign to do two
        things that raise ValueError.
        """
        from et_miner.gpu.kernels.loader import _warn_result_truncation

        with pytest.raises(RuntimeError) as exc:
            _warn_result_truncation(5000, 4000, "ctx", k=6)
        msg = str(exc.value)
        assert "K=6" in msg, msg
        # the reachable remedies
        assert "n_gpus>1" in msg and "prune_equal_support=True" in msg, msg
        assert "max_length" in msg, msg
        # and not the unreachable ones
        assert "resume_from_k" not in msg, msg
        assert "Raise max_results" not in msg, msg

    def test_no_bare_clamp_survives(self):
        """The four sites were found by exactly this grep."""
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "src" / "et_miner" / "gpu" / "kernels"
        offenders = [
            f"{p.name}:{i}"
            for p in root.glob("*.py")
            for i, line in enumerate(p.read_text().splitlines(), 1)
            if "min(n, gpu_max)" in line or "min(n_actual, max_results)" in line
        ]
        assert not offenders, f"bare truncation clamps: {offenders}"


class TestKernelKCap:
    """#25 -- the K>=3 kernels cache the candidate in a fixed 64-slot shared
    array. Beyond the cap they read an uninitialised slot AS A COLUMN INDEX and
    write one past the array into the block reduction. Only the shared/tiled
    wrapper enforced anything; the legacy, fully-fused and gpu-resident
    wrappers passed the shape straight through."""

    def test_the_cap_is_a_single_constant(self):
        from et_miner.gpu.kernels.loader import MAX_SUPPORTED_K, _assert_k_supported

        assert MAX_SUPPORTED_K == 62
        _assert_k_supported(62)  # must not raise
        with pytest.raises(ValueError, match="exceeds the kernel cap"):
            _assert_k_supported(63)

    @pytest.mark.gpu
    def test_an_oversized_candidate_raises_instead_of_miscounting(self):
        """Before this, K=63 returned 640 where the true count is 0."""
        import cupy as cp

        from et_miner.gpu.kernels import count_itemsets_fused_k3plus

        n_rows, n_cols, n_u64s = 640, 80, 10
        bv = cp.asarray(np.full((n_cols, n_u64s), 0xFFFFFFFFFFFFFFFF, dtype=np.uint64))
        count_itemsets_fused_k3plus(bv, [tuple(range(62))], n_u64s, 1)  # at the cap: fine
        with pytest.raises(ValueError, match="exceeds the kernel cap"):
            count_itemsets_fused_k3plus(bv, [tuple(range(63))], n_u64s, 1)


@pytest.mark.gpu
class TestMemoryGuardRaises:
    """#28 -- the guard returned True, the caller broke out of the level loop
    with a logger.warning, and control fell through to _build_result_df. The
    caller received a DataFrame that stopped at some K with nothing on it to say
    so: measured 249 itemsets against a complete 31,160, a 99.2% loss."""

    def test_a_tripped_guard_raises_rather_than_truncating(self, nested_df):
        with pytest.raises(MemoryError, match="Memory guard tripped"):
            apriori(nested_df, min_support=0.05, max_length=4, use_gpu=True, max_ram_gb=0.0)

    def test_the_message_says_the_lattice_is_incomplete(self, nested_df):
        with pytest.raises(MemoryError) as exc:
            apriori(nested_df, min_support=0.05, max_length=4, use_gpu=True, max_ram_gb=0.0)
        assert "INCOMPLETE" in str(exc.value)
        assert "max_ram_gb" in str(exc.value)

    def test_the_default_guards_do_not_fire(self, nested_df):
        got = apriori(nested_df, min_support=0.05, max_length=4, use_gpu=True)
        assert got.height > 0


class TestRouteResolution:
    """#32 -- the route was chosen from the ambient device count and the
    caller's n_gpus was ignored, so n_gpus=1 on a two-GPU box still landed on
    the fan-out kernels. And gpu/dispatch.py had NO logging at all, so a
    campaign that lost itemsets to a downstream clamp produced a log
    indistinguishable from the 1-GPU run that would have raised."""

    def test_caller_budget_caps_the_device_count(self, monkeypatch):
        from et_miner.gpu import dispatch

        monkeypatch.setattr(dispatch, "get_gpu_count", lambda: 8)
        big = dispatch.CANDIDATE_COUNT_THRESHOLD_K3 * 10

        assert dispatch._resolve_gpus(1, big, dispatch.CANDIDATE_COUNT_THRESHOLD_K3, "t") == 1
        assert dispatch._resolve_gpus(2, big, dispatch.CANDIDATE_COUNT_THRESHOLD_K3, "t") == 2
        # a budget above what exists is clamped, not honoured
        assert dispatch._resolve_gpus(99, big, dispatch.CANDIDATE_COUNT_THRESHOLD_K3, "t") == 8
        # None keeps the previous ambient behaviour
        assert dispatch._resolve_gpus(None, big, dispatch.CANDIDATE_COUNT_THRESHOLD_K3, "t") == 8
        # below the work threshold, one device regardless
        assert dispatch._resolve_gpus(8, 1, dispatch.CANDIDATE_COUNT_THRESHOLD_K3, "t") == 1

    def test_the_resolved_route_is_logged(self, monkeypatch):
        from loguru import logger

        from et_miner.gpu import dispatch

        monkeypatch.setattr(dispatch, "get_gpu_count", lambda: 4)
        lines: list[str] = []
        sink = logger.add(lines.append, level="DEBUG", format="{message}")
        try:
            dispatch._resolve_gpus(2, 10**9, dispatch.CANDIDATE_COUNT_THRESHOLD_K3, "k=5")
        finally:
            logger.remove(sink)
        out = "".join(lines)
        assert "caller n_gpus=2" in out and "available=4" in out and "running on 2" in out


@pytest.mark.gpu
class TestMineTwoPhase:
    """The only existing anchor test sets phase2_support == phase1_support, so
    every frequent item is an anchor and the mask is all-True -- a no-op. These
    use DISTINCT supports, which is the entire point of two-phase mining and the
    configuration in which #22 lost 95-99% of the answer."""

    def test_phase2_returns_exactly_the_anchored_itemsets(self, nested_df, tmp_path):
        from et_miner.gpu.row_split import mine_two_phase

        p1, p2 = 0.20, 0.05
        _, _, anchors = mine_two_phase(
            nested_df, phase1_support=p1, phase2_support=p2, max_length=4,
            n_gpus=1, output_dir=str(tmp_path), sparse_from_k=None,
        )
        assert anchors, "phase 1 found no anchors"

        got = set()
        for f in sorted((tmp_path / "phase2").glob("frequent_k*.parquet")):
            got |= {tuple(sorted(s)) for s in pl.read_parquet(f)["itemset"].to_list()}

        full = apriori(nested_df, min_support=p2, max_length=4, use_gpu=True,
                       n_gpus=1, prune_equal_support=True)
        expected = {
            tuple(sorted(s)) for s in full["itemset"].to_list() if anchors & set(s)
        }
        assert expected, "fixture produced no anchored itemsets at phase2_support"
        assert got == expected, (
            f"missing {len(expected - got)}, unexpected {len(got - expected)}"
        )

    def test_the_default_phase2_support_is_tractable(self):
        """The default was 0.00001, chosen when the anchor filter was believed to
        cut the search space. It does not -- Phase 2 mines the full lattice and
        post-filters -- so a default that deep is a trap."""
        import inspect

        from et_miner.gpu.row_split import mine_two_phase

        default = inspect.signature(mine_two_phase).parameters["phase2_support"].default
        assert default >= 1e-4, f"phase2_support default {default} is a full-lattice run"


class TestLevelStateCleanupIsolation:
    """#27 -- one bare `except Exception: pass` wrapped two unrelated releases.

    `free_groups(...)` and `sparse_state.release()` free different allocations:
    the current level's group arrays (tens of GB on a dense K>=3 level) and the
    resident CSR shards. Sharing one `try` meant a failure in the first silently
    skipped the second, and the `pass` meant nothing recorded it. Both run from
    a `finally`, so neither may raise -- but neither may cancel the other.

    No GPU needed: the point is the control flow, so both collaborators are
    injected.
    """

    @staticmethod
    def _patch_free_groups(monkeypatch, fn):
        import et_miner.gpu.row_split as rs

        monkeypatch.setattr(rs, "free_groups", fn)

    def test_shards_are_released_when_free_groups_raises(self, monkeypatch, caplog):
        """CONTROL: pre-fix, `release()` is never reached."""
        from et_miner.gpu.row_split import _release_level_state

        released = []

        def _boom(_groups):
            raise RuntimeError("group free exploded")

        self._patch_free_groups(monkeypatch, _boom)

        class _Sparse:
            def release(self):
                released.append(True)

        _release_level_state(object(), _Sparse())
        assert released == [True], "a failing group free must not skip the shard release"

    def test_group_free_runs_when_release_raises(self, monkeypatch):
        from et_miner.gpu.row_split import _release_level_state

        freed = []
        self._patch_free_groups(monkeypatch, lambda g: freed.append(g))

        class _Sparse:
            def release(self):
                raise RuntimeError("release exploded")

        sentinel = object()
        _release_level_state(sentinel, _Sparse())
        assert freed == [sentinel]

    def test_never_raises_and_logs_both_failures(self, monkeypatch):
        """It runs from a `finally`, often with an exception already in flight:
        a cleanup failure must never replace the original error. But it must be
        visible, which the bare `pass` made impossible."""
        from loguru import logger

        from et_miner.gpu.row_split import _release_level_state

        messages: list[str] = []
        sink = logger.add(lambda m: messages.append(m), level="WARNING")
        try:

            def _boom(_groups):
                raise RuntimeError("group free exploded")

            self._patch_free_groups(monkeypatch, _boom)

            class _Sparse:
                def release(self):
                    raise RuntimeError("release exploded")

            _release_level_state(object(), _Sparse())  # must not raise
        finally:
            logger.remove(sink)

        joined = "".join(messages)
        assert "group arrays" in joined and "CSR shards" in joined
        assert "group free exploded" in joined and "release exploded" in joined


class TestApr1oriPruneSetIsLazy:
    """#29 -- the prune set was built by both callers and read by neither.

    `_prune_groups_apriori`'s Rust fast path takes `prev_flat_np` and returns
    without touching `prev_frequent_set`; only the Python fallback reads it. So
    on every build with the extension present, `set(map(tuple, ...))` at the two
    row_split call sites was pure cost -- ~371 B/itemset and ~35 s per level at
    10M itemsets -- for an argument that was discarded.

    The equivalence assertions matter more than the timing: making the argument
    optional is only safe if the fallback derives *the same* set.
    """

    @staticmethod
    def _groups(prev_flat):
        from et_miner.gpu.kernels import build_k3plus_groups_from_flat

        return build_k3plus_groups_from_flat(prev_flat, with_src_rows=True)

    @staticmethod
    def _prev_flat():
        # A K=3 level with enough structure that pruning actually removes
        # suffixes -- otherwise both branches trivially agree.
        return np.array(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 4], [1, 4], [4, 5]],
            dtype=np.int32,
        )

    def test_none_matches_an_explicit_set_on_the_python_fallback(self):
        """The fallback must derive exactly what the callers used to pass."""
        from et_miner.gpu.mining import _prune_groups_apriori

        prev_flat = self._prev_flat()
        explicit = set(map(tuple, prev_flat.tolist()))

        with_set = _prune_groups_apriori(self._groups(prev_flat), explicit, 3, prev_flat_np=None)
        derived = _prune_groups_apriori(self._groups(prev_flat), None, 3, prev_flat_np=prev_flat)

        assert (with_set is None) == (derived is None)
        if with_set is not None:
            assert with_set.total_candidates == derived.total_candidates
            np.testing.assert_array_equal(with_set.suffixes, derived.suffixes)
            np.testing.assert_array_equal(with_set.prefix_items, derived.prefix_items)

    def test_both_none_raises_rather_than_pruning_nothing(self):
        """Silently pruning nothing is indistinguishable from a level with no
        invalid candidates, which is why this is an error and not a default."""
        from et_miner.gpu.mining import _prune_groups_apriori

        with pytest.raises(ValueError, match="needs a previous level"):
            _prune_groups_apriori(self._groups(self._prev_flat()), None, 3, prev_flat_np=None)

    @pytest.mark.parametrize("force_python", [False, True])
    def test_prev_flat_np_wins_over_a_disagreeing_set(self, force_python, monkeypatch):
        """Documented precedence, asserted on BOTH branches.

        The Rust path ignores the set structurally, so testing only that proves
        nothing about the contract. Parametrised over a forced Python fallback
        because the precedence must not depend on whether the wheel is present:
        an answer that changes with the build is the two-paths defect this
        change exists to remove.

        No cross-check is performed when the two disagree -- comparing them
        would cost the very set this change removes.
        """
        import et_miner.backends as backends

        from et_miner.gpu.mining import _prune_groups_apriori

        if force_python:
            monkeypatch.setattr(backends, "get_rust_ext", lambda: None)

        prev_flat = self._prev_flat()
        authoritative = _prune_groups_apriori(self._groups(prev_flat), None, 3, prev_flat_np=prev_flat)
        with_wrong_set = _prune_groups_apriori(self._groups(prev_flat), {(99, 98)}, 3, prev_flat_np=prev_flat)

        assert (authoritative is None) == (with_wrong_set is None)
        assert authoritative is not None, "fixture must survive pruning"
        assert authoritative.total_candidates == with_wrong_set.total_candidates
        np.testing.assert_array_equal(authoritative.suffixes, with_wrong_set.suffixes)
