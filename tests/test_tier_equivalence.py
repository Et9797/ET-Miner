"""The mandated cross-tier correctness gate (see CLAUDE.md).

Every smoke/validation run asserts, on the ``smoke`` synthetic preset:

    Tier 1 Polars == Tier 2 Rust (sparse=True)
        == row-split 1 GPU (measured kernel dispatch)
        == row-split 1 GPU, tiled kernel pinned
        == row-split 1 GPU, per-candidate kernel pinned
        == row-split 1 GPU, forced chunks (the fused tiled kernel)
        == row-split 1 GPU, no subset test
        == row-split 1 GPU, count inference (measured dispatch, tiled, per-candidate,
           forced chunks)
        == row-split 1 GPU, row-wise K=2 (shared-memory / global atomics, plain / forced chunks)
        == ESCO 1 GPU, auto and fixed K=3, including forced chunks and count inference
        == SON 1 GPU, forced chunks (the batched itemset kernel)
        == row-split 2 GPUs == row-split 2 GPUs, forced chunks (per-candidate sub-chunks)
        == row-split 2 GPUs, count inference (plain and forced chunks)
        == row-split 2 GPUs, row-wise K=2 (plain and forced chunks)
        == row-split 2 GPUs, compacted reduce (plain and forced chunks, with and
           without count inference)
        == ESCO 2 GPUs, including count inference
        == SON 2 GPUs, forced chunks
        == efficient-apriori (the canonical oracle)

Every surviving counting kernel is pinned by the leg's own environment, not by
a default: ET_MINER_TILED_MIN_GROUP_PAIRS pins the tiled (0) or the
per-candidate (a pair count no group reaches) kernel for every prefix group,
ET_MINER_MAX_CHUNK_CANDS forces chunking (on one GPU the pair space and every
larger group then go to the fused tiled kernel; on two, to per-candidate
sub-chunks; a row-wise K=2 counts every chunk from the rows),
ET_MINER_K2_KERNEL=rows pins the row-wise K=2 kernel, ET_MINER_REDUCE=compact pins the
compacted multi-GPU reduce, and a chunk_size below the row count forces SON to four chunks.
Every row-split leg also asserts which previous-level index its K>=3 levels
were given: the subset test by default, none with ``prune_apriori=False``, and
the inferring index, written by exactly one device per level, with
``use_generator_pruning=True``.

Comparisons are exact on itemsets AND absolute counts — never weakened to
count-only or tolerance checks. Oracle boundary handling: the miner keeps
``count >= ceil(min_support * N)`` while efficient-apriori keeps
``support >= min_support`` in float, so the oracle is called with
``min_support = (min_count - 0.5) / N`` (exact for integer counts) and an
explicit ``max_length`` (efficient-apriori silently defaults to 8).

CPU-tier assertions run everywhere (they gate CI); GPU-tier assertions are
gpu-marked and self-skip below the needed device count.
"""

import math
from itertools import chain, combinations

import numpy as np
import polars as pl
import pytest

from et_miner.core.apriori import apriori
from et_miner.core.result import _min_count
from et_miner.synthetic import PRESETS, generate_transactions

ea_apriori = pytest.importorskip("efficient_apriori", reason="efficient_apriori not installed").apriori

SPEC = PRESETS["smoke"]
#: Explicit — efficient-apriori defaults to max_length=8 and would silently
#: truncate deeper itemsets from the oracle.
ORACLE_MAX_LENGTH = 32

CountedSet = set[tuple[tuple[int, ...], int]]


def _result_to_counted_set(result_df) -> CountedSet:
    """et-miner result DataFrame → {(sorted itemset, absolute count)}."""
    out: CountedSet = set()
    for itemset, sup in zip(result_df["itemset"].to_list(), result_df["support"].to_list()):
        out.add((tuple(sorted(int(i) for i in itemset)), round(sup * SPEC.n_rows)))
    return out


def _assert_ascending(result_df, label: str) -> None:
    """Every emitted itemset must be an ASCENDING tuple of item ids.

    This is the property _result_to_counted_set launders away: it canonicalises
    with sorted() before comparing, so a tier emitting [10, 2] compares equal to
    one emitting [2, 10] and the gate passes 9/9 either way. The equivalence
    assertions above are deliberately left alone (CLAUDE.md forbids weakening
    them, and the canonicalisation also insulates the gate from efficient-apriori
    emitting in an undocumented order); this is the ADDITIVE half that makes the
    producer's own contract checkable.

    It is also the only in-code guard on the row-split route, which is the one
    route that already got the order right -- and therefore the one most worth
    protecting from regression.
    """
    bad = [list(x) for x in result_df["itemset"].to_list() if list(x) != sorted(x)]
    assert not bad, f"{label}: {len(bad)} itemsets not in ascending item-id order, e.g. {bad[:5]}"


def _counted(result_df, label: str) -> CountedSet:
    """Assert the emission-order contract, then reduce to the counted set."""
    _assert_ascending(result_df, label)
    return _result_to_counted_set(result_df)


def _assert_counted_sets_equal(got: CountedSet, expected: CountedSet, label: str) -> None:
    got_keys = {k for k, _ in got}
    exp_keys = {k for k, _ in expected}
    missing = exp_keys - got_keys
    extra = got_keys - exp_keys
    assert not missing, f"{label}: missing itemsets: {sorted(missing)[:10]}"
    assert not extra, f"{label}: extra itemsets: {sorted(extra)[:10]}"
    got_counts = dict(got)
    exp_counts = dict(expected)
    mismatches = [(k, exp_counts[k], got_counts[k]) for k in exp_keys if got_counts[k] != exp_counts[k]]
    assert not mismatches, f"{label}: count mismatches (itemset, expected, got): {mismatches[:10]}"


@pytest.fixture(scope="session")
def smoke_dataset():
    df, data = generate_transactions(SPEC)
    return df, data


@pytest.fixture(scope="session")
def oracle_set(smoke_dataset) -> CountedSet:
    """efficient-apriori mined once per session, boundary-safe."""
    _, data = smoke_dataset
    rows = np.split(data.indices, data.indptr[1:-1])
    transactions = [tuple(int(x) for x in r) for r in rows]
    ea_support = (SPEC.min_count - 0.5) / SPEC.n_rows
    itemsets, _ = ea_apriori(
        transactions, min_support=ea_support, min_confidence=1.0, max_length=ORACLE_MAX_LENGTH
    )
    out: CountedSet = set()
    for _k, sets_of_k in itemsets.items():
        for fset, count in sets_of_k.items():
            out.add((tuple(sorted(int(i) for i in fset)), int(count)))
    return out


@pytest.fixture(scope="session")
def tier1_set(smoke_dataset) -> CountedSet:
    df, _ = smoke_dataset
    return _counted(apriori(df, min_support=SPEC.min_support, item_col="items"), "Tier 1 Polars")


def _gpu_count() -> int:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


# ── CPU half of the chain (runs everywhere, gates CI) ──────────────────────


def test_tier1_polars_matches_oracle(tier1_set, oracle_set):
    _assert_counted_sets_equal(tier1_set, oracle_set, "Tier 1 Polars vs efficient-apriori")


def test_tier2_rust_matches_oracle(smoke_dataset, oracle_set):
    df, _ = smoke_dataset
    got = _counted(apriori(df, min_support=SPEC.min_support, item_col="items", sparse=True), "Tier 2 Rust/sparse")
    _assert_counted_sets_equal(got, oracle_set, "Tier 2 Rust/sparse vs efficient-apriori")


def test_boundary_count_is_kept_at_a_threshold_computed_independently():
    """#14 -- the gate must not source its expected value from the code under test.

    Every other test here calls the miners with SPEC.min_support and the oracle
    with (SPEC.min_count - 0.5)/N, where SPEC.min_count is the *same expression*
    the miners use, character for character. Any error in that expression moves
    both thresholds by the same amount and the comparison still passes: the gate
    cannot detect a min-count error by construction, which is what let #11 live.

    The -0.5 convention is right for what it does and stays. What it cannot do
    is probe the boundary itself, so this case does that directly, with the
    threshold derived from the DECIMAL rather than from _min_count: an itemset
    whose count is exactly ceil(Fraction(str(s)) * N) must be returned.

    s = 0.07 is chosen because 0.07 has no exact binary64 form, so the naive
    expression returns one too many and drops the boundary itemset -- and with
    it the whole cone above it.
    """
    from fractions import Fraction

    s, n_rows = 0.07, 10_000
    threshold = math.ceil(Fraction(str(s)) * n_rows)  # 700, independently derived
    assert threshold == 700

    rows = [[0, 1, 2]] * threshold + [[1, 2]] * (n_rows - threshold)
    got = {tuple(sorted(x)) for x in apriori(pl.DataFrame({"items": rows}), min_support=s)["itemset"].to_list()}

    assert (0,) in got, (
        f"an item with count exactly {threshold} = ceil({s} * {n_rows}) must be frequent; "
        f"the miner's threshold is {_min_count(s, n_rows)}"
    )
    for cone in [(0, 1), (0, 2), (0, 1, 2)]:
        assert cone in got, f"{cone} sits above the boundary itemset and was lost with it"


def test_planted_motifs_recovered(tier1_set, smoke_dataset):
    """Oracle-independent ground truth: every planted motif and every subset
    must be mined with count >= the planted count (>=, never == — background
    noise can only add occurrences)."""
    _, data = smoke_dataset
    counts = dict(tier1_set)
    for motif, planted_count in data.planted:
        subsets = chain.from_iterable(combinations(motif, r) for r in range(1, len(motif) + 1))
        for sub in subsets:
            got = counts.get(tuple(sub))
            assert got is not None, f"planted subset {sub} of motif {motif} not mined"
            assert got >= planted_count, f"subset {sub}: mined {got} < planted {planted_count}"


def test_fpgrowth_second_oracle_agrees(smoke_dataset):
    """Optional independent cross-check of the oracle itself: mlxtend
    fpgrowth on a subsample must agree with efficient-apriori exactly."""
    fpgrowth_mod = pytest.importorskip("mlxtend.frequent_patterns", reason="mlxtend not installed")
    import pandas as pd

    _, data = smoke_dataset
    n_sub = 10_000
    sub_indptr = data.indptr[: n_sub + 1]
    sub_items = data.indices[: sub_indptr[-1]]

    onehot = np.zeros((n_sub, data.n_cols), dtype=bool)
    onehot[np.repeat(np.arange(n_sub), np.diff(sub_indptr)), sub_items] = True
    min_count = SPEC.min_count  # same absolute count on the subsample
    fp = fpgrowth_mod.fpgrowth(
        pd.DataFrame(onehot), min_support=(min_count - 0.5) / n_sub, use_colnames=True
    )
    fp_set = {
        (tuple(sorted(int(i) for i in row)), round(sup * n_sub))
        for row, sup in zip(fp["itemsets"], fp["support"])
    }

    rows = np.split(sub_items, sub_indptr[1:-1])
    transactions = [tuple(int(x) for x in r) for r in rows]
    itemsets, _ = ea_apriori(
        transactions, min_support=(min_count - 0.5) / n_sub, min_confidence=1.0, max_length=ORACLE_MAX_LENGTH
    )
    ea_set = {
        (tuple(sorted(int(i) for i in fs)), int(c)) for d in itemsets.values() for fs, c in d.items()
    }
    _assert_counted_sets_equal(fp_set, ea_set, "fpgrowth vs efficient-apriori (subsample)")


# ── GPU half of the chain (box campaign; auto-skipped without devices) ─────

#: No prefix group reaches this many pairs, so every group runs per-candidate.
_NO_GROUP = str(10**12)
#: A chunk budget far below the smoke preset's pair space and largest groups.
_TINY_CHUNK = "40"


_WRAPPERS = (
    "count_pairs_k2_shared",
    "count_pairs_k2_per_candidate",
    "count_pairs_k2_rows",
    "count_shared_tiled_allcounts",
    "count_k3plus_per_candidate",
    "count_tiled_fused",
    "count_itemsets_cuda",
)


def _gpu_leg(smoke_dataset, oracle_set, monkeypatch, label, env, *, runs=(), never=(), index_mode="prune",
             **kwargs):
    """One GPU leg; `runs` / `never` name the kernel wrappers the pin must reach / exclude.

    `index_mode` is the previous-level index every K>=3 level must get: "prune",
    "infer" or None (no index uploaded).
    """
    from et_miner.gpu import kernels

    calls = dict.fromkeys(_WRAPPERS, 0)
    for name in _WRAPPERS:
        def spy(*a, _real=getattr(kernels, name), _name=name, **k):
            calls[_name] += 1
            return _real(*a, **k)

        monkeypatch.setattr(kernels, name, spy)
    uploads = []

    def index_spy(*a, _real=kernels.upload_subset_index, **k):
        uploads.append((k["mode"], k["device_id"], k["write_inferred"], len(a[0])))
        return _real(*a, **k)

    monkeypatch.setattr(kernels, "upload_subset_index", index_spy)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    df, _ = smoke_dataset
    got = _counted(apriori(df, min_support=SPEC.min_support, item_col="items", use_gpu=True, **kwargs), label)
    _assert_counted_sets_equal(got, oracle_set, f"{label} vs efficient-apriori")
    assert all(calls[n] for n in runs), f"{label}: the pinned kernel did not run: {calls}"
    assert not any(calls[n] for n in never), f"{label}: a kernel the pin excludes ran: {calls}"
    if index_mode is None:
        assert not uploads, f"{label}: a subset index was uploaded: {uploads}"
        return
    want = kernels.SUBSET_PRUNE | (kernels.SUBSET_INFER if index_mode == "infer" else 0)
    assert uploads, f"{label}: no level got a subset index"
    assert {u[0] for u in uploads} == {want}, f"{label}: index modes {uploads}"
    if not kwargs.get("streaming"):
        n_dev = kwargs.get("n_gpus", 1)
        writers = [u for u in uploads if u[2]]
        assert len(uploads) == n_dev * len(writers), f"{label}: one writing device per level expected: {uploads}"


def _needs_two_gpus():
    if _gpu_count() < 2:
        pytest.skip("needs 2 CUDA devices")


@pytest.mark.gpu
def test_row_split_one_gpu_matches_oracle(smoke_dataset, oracle_set, monkeypatch):
    _gpu_leg(smoke_dataset, oracle_set, monkeypatch, "row-split 1 GPU", {})


@pytest.mark.gpu
def test_row_split_one_gpu_tiled_matches_oracle(smoke_dataset, oracle_set, monkeypatch):
    _gpu_leg(smoke_dataset, oracle_set, monkeypatch, "row-split 1 GPU, tiled", {"ET_MINER_TILED_MIN_GROUP_PAIRS": "0"},
             runs=("count_pairs_k2_shared", "count_shared_tiled_allcounts"),
             never=("count_pairs_k2_per_candidate", "count_k3plus_per_candidate", "count_tiled_fused"))


@pytest.mark.gpu
def test_row_split_one_gpu_per_candidate_matches_oracle(smoke_dataset, oracle_set, monkeypatch):
    _gpu_leg(smoke_dataset, oracle_set, monkeypatch, "row-split 1 GPU, per-candidate",
             {"ET_MINER_TILED_MIN_GROUP_PAIRS": _NO_GROUP},
             runs=("count_pairs_k2_per_candidate", "count_k3plus_per_candidate"),
             never=("count_pairs_k2_shared", "count_shared_tiled_allcounts", "count_tiled_fused"))


@pytest.mark.gpu
def test_row_split_one_gpu_forced_chunks_matches_oracle(smoke_dataset, oracle_set, monkeypatch):
    """On one GPU a level beyond one dense chunk is counted by the fused tiled kernel."""
    _gpu_leg(smoke_dataset, oracle_set, monkeypatch, "row-split 1 GPU, forced chunks",
             {"ET_MINER_MAX_CHUNK_CANDS": _TINY_CHUNK, "ET_MINER_TILED_MIN_GROUP_PAIRS": "0"},
             runs=("count_tiled_fused",), never=("count_pairs_k2_per_candidate", "count_k3plus_per_candidate"))


@pytest.mark.gpu
def test_row_split_one_gpu_without_subset_test_matches_oracle(smoke_dataset, oracle_set, monkeypatch):
    _gpu_leg(smoke_dataset, oracle_set, monkeypatch, "row-split 1 GPU, no subset test", {}, index_mode=None,
             prune_apriori=False)


@pytest.mark.gpu
@pytest.mark.parametrize("pin", ["dispatch", "tiled", "per-candidate", "forced chunks"])
def test_row_split_one_gpu_count_inference_matches_oracle(smoke_dataset, oracle_set, monkeypatch, pin):
    env, runs = {
        "dispatch": ({}, ()),
        "tiled": ({"ET_MINER_TILED_MIN_GROUP_PAIRS": "0"}, ("count_shared_tiled_allcounts",)),
        "per-candidate": ({"ET_MINER_TILED_MIN_GROUP_PAIRS": _NO_GROUP}, ("count_k3plus_per_candidate",)),
        "forced chunks": ({"ET_MINER_MAX_CHUNK_CANDS": _TINY_CHUNK, "ET_MINER_TILED_MIN_GROUP_PAIRS": "0"},
                          ("count_tiled_fused",)),
    }[pin]
    _gpu_leg(smoke_dataset, oracle_set, monkeypatch, f"row-split 1 GPU, count inference ({pin})", env, runs=runs,
             index_mode="infer", use_generator_pruning=True)


#: The bitvec K=2 wrappers, which a row-wise K=2 pin excludes.
_K2_DENSE = ("count_pairs_k2_shared", "count_pairs_k2_per_candidate")


def _row_wise_k2_leg(smoke_dataset, oracle_set, monkeypatch, label, *, atomics, chunked, **kwargs):
    """A leg with K=2 pinned to the row-wise kernel, on shared-memory or global atomics."""
    from et_miner.gpu.kernels import k2

    if atomics == "global":
        monkeypatch.setattr(k2, "K2_ROWS_SHARED_PAIRS", 0)
    launched = []

    def kernel_spy(name, _real=k2.get_cuda_kernel):
        launched.append(name)
        return _real(name)

    monkeypatch.setattr(k2, "get_cuda_kernel", kernel_spy)
    env = {"ET_MINER_K2_KERNEL": "rows"}
    if chunked:
        env["ET_MINER_MAX_CHUNK_CANDS"] = _TINY_CHUNK
    _gpu_leg(smoke_dataset, oracle_set, monkeypatch, label, env, runs=("count_pairs_k2_rows",), never=_K2_DENSE,
             **kwargs)
    want = "count_pairs_k2_rows_shared" if atomics == "shared" else "count_pairs_k2_rows"
    assert set(launched) == {want}, f"{label}: row-wise kernels launched: {sorted(set(launched))}"


@pytest.mark.gpu
@pytest.mark.parametrize("atomics", ["shared", "global"])
@pytest.mark.parametrize("chunked", [False, True])
def test_row_split_one_gpu_row_wise_k2_matches_oracle(smoke_dataset, oracle_set, monkeypatch, atomics, chunked):
    _row_wise_k2_leg(smoke_dataset, oracle_set, monkeypatch,
                     f"row-split 1 GPU, row-wise K=2 ({atomics} atomics, chunks={chunked})",
                     atomics=atomics, chunked=chunked)


@pytest.mark.gpu
def test_son_one_gpu_forced_chunks_matches_oracle(smoke_dataset, oracle_set, monkeypatch):
    _gpu_leg(smoke_dataset, oracle_set, monkeypatch, "SON 1 GPU, four chunks", {},
             runs=("count_itemsets_cuda",), streaming=True, chunk_size=SPEC.n_rows // 4 + 1, show_progress=False)


@pytest.mark.gpu
@pytest.mark.parametrize("sparse_from_k", ["auto", 3])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("infer", [False, True])
def test_esco_one_gpu_matches_oracle(smoke_dataset, oracle_set, monkeypatch, sparse_from_k, chunked, infer):
    from et_miner.gpu import sparse_csr

    calls = []
    real = sparse_csr.count_csr_range

    def spy(*a, **kw):
        calls.append(kw.get("index"))
        return real(*a, **kw)

    monkeypatch.setattr(sparse_csr, "count_csr_range", spy)
    env = {"ET_MINER_MAX_CHUNK_CANDS": _TINY_CHUNK} if chunked else {}
    _gpu_leg(smoke_dataset, oracle_set, monkeypatch,
             f"ESCO 1 GPU ({sparse_from_k}, chunks={chunked}, inference={infer})", env,
             index_mode="infer" if infer else "prune", sparse_from_k=sparse_from_k, n_gpus=1,
             use_generator_pruning=infer)
    assert calls, "ESCO leg must actually count sparse candidates"
    assert all(index is not None for index in calls), "every sparse level must get the subset index"


@pytest.mark.gpu
@pytest.mark.multigpu
@pytest.mark.parametrize("sparse_from_k", ["auto", 3])
@pytest.mark.parametrize("infer", [False, True])
def test_esco_two_gpus_matches_oracle(smoke_dataset, oracle_set, monkeypatch, sparse_from_k, infer):
    _needs_two_gpus()
    _gpu_leg(smoke_dataset, oracle_set, monkeypatch, f"ESCO 2 GPUs ({sparse_from_k}, inference={infer})",
             {"ET_MINER_MAX_CHUNK_CANDS": _TINY_CHUNK}, index_mode="infer" if infer else "prune",
             sparse_from_k=sparse_from_k, n_gpus=2, use_generator_pruning=infer)


@pytest.mark.gpu
@pytest.mark.multigpu
def test_row_split_two_gpus_matches_oracle(smoke_dataset, oracle_set, monkeypatch):
    _needs_two_gpus()
    _gpu_leg(smoke_dataset, oracle_set, monkeypatch, "row-split 2 GPUs", {}, n_gpus=2)


@pytest.mark.gpu
@pytest.mark.multigpu
def test_row_split_two_gpus_forced_chunks_matches_oracle(smoke_dataset, oracle_set, monkeypatch):
    """Across GPUs a level beyond one chunk runs per-candidate sub-chunks, reduced per chunk."""
    _needs_two_gpus()
    _gpu_leg(smoke_dataset, oracle_set, monkeypatch, "row-split 2 GPUs, forced chunks",
             {"ET_MINER_MAX_CHUNK_CANDS": _TINY_CHUNK, "ET_MINER_TILED_MIN_GROUP_PAIRS": "0"},
             runs=("count_pairs_k2_per_candidate", "count_k3plus_per_candidate"), never=("count_tiled_fused",),
             n_gpus=2)


@pytest.mark.gpu
@pytest.mark.multigpu
@pytest.mark.parametrize("chunked", [False, True])
def test_row_split_two_gpus_row_wise_k2_matches_oracle(smoke_dataset, oracle_set, monkeypatch, chunked):
    """Each GPU counts its own rows; the per-chunk reduce sums the pair arrays."""
    _needs_two_gpus()
    _row_wise_k2_leg(smoke_dataset, oracle_set, monkeypatch, f"row-split 2 GPUs, row-wise K=2 (chunks={chunked})",
                     atomics="shared", chunked=chunked, n_gpus=2)


@pytest.mark.gpu
@pytest.mark.multigpu
@pytest.mark.parametrize("chunked", [False, True])
def test_row_split_two_gpus_count_inference_matches_oracle(smoke_dataset, oracle_set, monkeypatch, chunked):
    """Inferred counts are written by one device, so the cross-GPU sum stays exact."""
    _needs_two_gpus()
    env = {"ET_MINER_MAX_CHUNK_CANDS": _TINY_CHUNK, "ET_MINER_TILED_MIN_GROUP_PAIRS": "0"} if chunked else {}
    _gpu_leg(smoke_dataset, oracle_set, monkeypatch, f"row-split 2 GPUs, count inference (chunks={chunked})", env,
             index_mode="infer", n_gpus=2, use_generator_pruning=True)


@pytest.mark.gpu
@pytest.mark.multigpu
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("infer", [False, True])
def test_row_split_two_gpus_compacted_reduce_matches_oracle(smoke_dataset, oracle_set, monkeypatch, chunked, infer):
    """The K>=3 levels reduce only the entries the kernels wrote, compacted on both GPUs."""
    _needs_two_gpus()
    from et_miner.gpu.kernels import filter as filter_mod

    compacted = []

    def spy(counts, *, keep_mask, _real=filter_mod.compact_written):
        compacted.append((int(counts.device.id), keep_mask))
        return _real(counts, keep_mask=keep_mask)

    monkeypatch.setattr(filter_mod, "compact_written", spy)
    env = {"ET_MINER_REDUCE": "compact"}
    pins = {}
    if chunked:
        env |= {"ET_MINER_MAX_CHUNK_CANDS": _TINY_CHUNK, "ET_MINER_TILED_MIN_GROUP_PAIRS": "0"}
        pins = {"runs": ("count_k3plus_per_candidate",), "never": ("count_tiled_fused",)}
    _gpu_leg(smoke_dataset, oracle_set, monkeypatch,
             f"row-split 2 GPUs, compacted reduce (chunks={chunked}, inference={infer})", env,
             index_mode="infer" if infer else "prune", n_gpus=2, use_generator_pruning=infer, **pins)
    assert compacted, "no chunk was reduced compacted"
    per_device = {d: sum(1 for c, _ in compacted if c == d) for d in (0, 1)}
    assert per_device[0] == per_device[1] == len(compacted) // 2, f"compacted on {compacted}"
    assert all(keep == (d == 0) for d, keep in compacted), f"the mask belongs on GPU 0: {compacted}"


@pytest.mark.gpu
@pytest.mark.multigpu
def test_son_two_gpus_forced_chunks_matches_oracle(smoke_dataset, oracle_set, monkeypatch):
    _needs_two_gpus()
    _gpu_leg(smoke_dataset, oracle_set, monkeypatch, "SON 2 GPUs, four chunks", {},
             runs=("count_itemsets_cuda",), streaming=True, n_gpus=2, chunk_size=SPEC.n_rows // 4 + 1,
             show_progress=False)
