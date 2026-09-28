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


class TestKernelKCap:
    """#25 -- the K>=3 kernels cache the candidate in a fixed 64-slot shared
    array. Beyond the cap they read an uninitialised slot AS A COLUMN INDEX and
    write one past the array into the block reduction, so every K>=3 wrapper
    checks the one host-side cap before launching."""

    def test_the_cap_is_a_single_constant(self):
        from et_miner.gpu.kernels.loader import MAX_SUPPORTED_K, _assert_k_supported

        assert MAX_SUPPORTED_K == 62
        _assert_k_supported(62)  # must not raise
        with pytest.raises(ValueError, match="exceeds the kernel cap"):
            _assert_k_supported(63)

    @pytest.mark.gpu
    @pytest.mark.parametrize("kernel", ["per-candidate", "tiled-dense", "tiled-fused"])
    def test_an_oversized_candidate_raises_instead_of_miscounting(self, kernel):
        """At the cap every K>=3 kernel counts exactly; one past it raises."""
        import cupy as cp

        from et_miner.gpu.kernels import K3PlusGroups, count_k3plus_allcounts, count_tiled_fused

        n_rows, n_cols, n_u64s = 640, 80, 10
        bv = cp.asarray(np.full((n_cols, n_u64s), 0xFFFFFFFFFFFFFFFF, dtype=np.uint64))

        def one_candidate(prefix_len):
            return K3PlusGroups(
                prefix_items=np.arange(prefix_len, dtype=np.int32),
                prefix_offsets=np.array([0, prefix_len], dtype=np.int64),
                suffixes=np.array([prefix_len, prefix_len + 1], dtype=np.int32),
                suffix_offsets=np.array([0, 2], dtype=np.int64),
                cumulative_pairs=np.array([0, 1], dtype=np.int64),
                total_candidates=1,
                groups=None,
            )

        def count(groups):
            if kernel == "tiled-fused":
                return count_tiled_fused(bv, groups, n_u64s, 1)[1].tolist()
            variant = "legacy" if kernel == "per-candidate" else "shared"
            return count_k3plus_allcounts(bv, groups, n_u64s, variant=variant).get().tolist()

        assert count(one_candidate(60)) == [n_rows]  # K = 62
        with pytest.raises(ValueError, match="exceeds the kernel cap"):
            count(one_candidate(61))  # K = 63


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


@pytest.mark.gpu
class TestGroupArraysAreReleasedOnAnAbortedLevel:
    """#31 -- the dense K>=3 group arrays had no `finally`.

    `upload_k3plus_groups` puts group data on every device and it stays
    resident across all chunks -- tens of GB on a wide level. The free ran
    after `run_chunked_dense_level` returned, so any raise inside skipped it
    entirely, most obviously the result-truncation RuntimeError PR 5 added,
    which is precisely the case where the process carries on afterwards.

    Measured INSIDE the `except` block, deliberately. Once the traceback is
    released the miner's frames die and the dict is collected anyway, so a
    check after the handler would pass with or without the fix. While the
    traceback is alive, the pre-fix code still holds the arrays and the
    post-fix code does not -- that is the only window in which they differ.
    """

    def test_pool_releases_the_group_arrays_before_the_traceback_dies(self, nested_df, monkeypatch):
        import cupy as cp

        import et_miner.gpu.row_split as rs
        from et_miner.core.matrix import _build_csr_from_transactions

        built = _build_csr_from_transactions(nested_df.lazy(), 0.05, "items")
        assert built is not None, "fixture produced no frequent items"
        csr, idx_to_item, n_trans = built

        pool = cp.get_default_memory_pool()
        during: dict[str, int] = {}

        # K=2 goes through run_chunked_dense_level too, before any group data
        # exists. Raising there would measure nothing, so the injection is
        # gated on an actual K>=3 upload having happened.
        # upload_k3plus_groups is imported inside the miner's body, so it has
        # to be patched at its source module, not on rs.
        import et_miner.gpu.kernels as kernels

        real_upload = kernels.upload_k3plus_groups
        real_level = rs.run_chunked_dense_level
        uploaded = {"yes": False}

        def _spy_upload(*args, **kwargs):
            uploaded["yes"] = True
            return real_upload(*args, **kwargs)

        def _raise_after_upload(*args, **kwargs):
            if not uploaded["yes"]:
                return real_level(*args, **kwargs)
            during["used"] = pool.used_bytes()
            raise RuntimeError("injected: chunked dense level failed")

        monkeypatch.setattr(kernels, "upload_k3plus_groups", _spy_upload)
        monkeypatch.setattr(rs, "run_chunked_dense_level", _raise_after_upload)

        used_after = None
        try:
            rs._apriori_row_split_multi_gpu(
                csr,
                idx_to_item,
                n_trans,
                0.05,
                None,
                1,
                prune_non_free=True,
            )
        except RuntimeError as exc:
            assert "injected" in str(exc)
            used_after = pool.used_bytes()
        else:  # pragma: no cover - the injection must fire
            pytest.fail("the injected failure did not propagate")

        assert "used" in during, "fixture never reached the dense K>=3 upload"
        assert used_after < during["used"], (
            "group arrays were still resident while the traceback held the frame: "
            f"{during['used']:,} -> {used_after:,} bytes"
        )
