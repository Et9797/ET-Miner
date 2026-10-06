"""What ``prune_equal_support`` means, asserted directly.

``prune_equal_support=False`` returns the complete frequent lattice.
``prune_equal_support=True`` returns the frequent **free-sets** (generators):
the itemsets with no equal-support proper subset [Bastide et al. 2000]. Free-sets
are anti-monotone, so a level generated from the previous level's free-sets loses
no free-set — but only if the level that is emitted is the same level the next
one is generated from, and only if the equal-support test resolves against the
COMPLETE previous level.

Both halves used to be wrong on the GPU path: the level was flushed before the
prune and the next level was generated from the survivors, so the output
advertised itemsets the run would never extend, and frequent, apriori-valid
itemsets went missing from K=5 on — layout-sensitively, because which parent
survived depended on unrelated items sharing a prefix group.

The CPU guards here need no GPU and run in about a second.
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
import pytest

from et_miner.core.apriori import apriori

MIN_SUPPORT = 0.03
MAX_K = 5


def _fixture(n_rows: int = 4000, n_base: int = 14, extra_items: int = 0, seed: int = 7):
    """Nested vocabulary: a child item implies its parent, so support(parent, child)
    equals support(child) and the free-set prune actually engages. One item is
    present in every transaction (not free at K=1). ``extra_items`` appends ids at
    the end of the layout without touching any existing row's other items.
    """
    rng = np.random.default_rng(seed)
    parent = {i: i % 4 for i in range(4, n_base)}
    themes = [rng.choice(range(4, n_base), size=5, replace=False) for _ in range(8)]
    n_cols = n_base + extra_items
    matrix = np.zeros((n_rows, n_cols), dtype=bool)
    rows = []
    for i in range(n_rows):
        items = {int(x) for x in themes[i % 8][rng.random(5) < 0.9]}
        items |= {parent[x] for x in items}
        items |= {int(x) for x in rng.choice(n_base, size=int(rng.integers(0, 3)), replace=False)}
        items.add(1)  # present in every transaction: support == 1.0, so not free
        if extra_items and i % 3 == 0:
            items |= {n_base + j for j in range(extra_items)}
        rows.append(sorted(items))
        matrix[i, list(items)] = True
    df = pl.DataFrame({"items": rows}, schema={"items": pl.List(pl.Int64)})
    return df, matrix


def _brute_force(matrix, min_support=MIN_SUPPORT, max_k=MAX_K):
    """(complete frequent lattice, free-sets) as {itemset: count} / {itemset}."""
    n_rows, n_cols = matrix.shape
    min_count = int(np.ceil(min_support * n_rows))
    counts = {}
    for k in range(1, max_k + 1):
        for c in itertools.combinations(range(n_cols), k):
            count = int(np.logical_and.reduce(matrix[:, list(c)], axis=1).sum()) if k > 1 else int(matrix[:, c[0]].sum())
            if count >= min_count:
                counts[c] = count
    free = {
        c
        for c, v in counts.items()
        # the empty set has count n_rows, so an itemset with that count is not free either
        if v != n_rows
        and not any(counts.get(s) == v for r in range(1, len(c)) for s in itertools.combinations(c, r))
    }
    return counts, free


def _mined(df: pl.DataFrame, **kwargs) -> dict[tuple[int, ...], float]:
    result = apriori(df, min_support=MIN_SUPPORT, max_length=MAX_K, item_col="items", **kwargs)
    return {
        tuple(sorted(int(i) for i in row["itemset"])): float(row["support"])
        for row in result.iter_rows(named=True)
    }


def _assert_emit_equals_generate(mined) -> None:
    """Every emitted itemset's (k-1)-subsets are emitted too — i.e. the level a
    run publishes is the level it generates the next one from."""
    orphans = [
        m for m in mined if len(m) > 1 and not all(tuple(sorted(set(m) - {x})) in mined for x in m)
    ]
    assert not orphans, f"{len(orphans)} emitted itemsets the run would never extend, e.g. {orphans[:3]}"


# ── CPU tier ────────────────────────────────────────────────────────────────


def test_flag_off_returns_the_complete_lattice():
    df, matrix = _fixture()
    counts, _ = _brute_force(matrix)
    mined = _mined(df)
    assert set(mined) == set(counts)
    assert all(mined[c] == pytest.approx(counts[c] / len(matrix), abs=1e-12) for c in mined)
    _assert_emit_equals_generate(mined)


def test_flag_on_returns_exactly_the_free_sets():
    df, matrix = _fixture()
    counts, free = _brute_force(matrix)
    mined = _mined(df, prune_equal_support=True)
    assert set(mined) == free, (
        f"extra: {sorted(set(mined) - free)[:3]} missing: {sorted(free - set(mined))[:3]}"
    )
    assert all(mined[c] == pytest.approx(counts[c] / len(matrix), abs=1e-12) for c in mined)
    _assert_emit_equals_generate(mined)
    assert len(free) < len(counts), "fixture must actually exercise the prune"


def test_free_set_output_is_layout_insensitive():
    """Appending ids that the itemsets in question do not contain must not change
    which of them are emitted. It used to: a suffix survived the apriori prune if
    it was in >= 1 valid pair, so a new neighbour in the same prefix group could
    rescue itemsets that were otherwise dropped ("no longer dropped: 240, newly
    dropped: 0" in the field report)."""
    narrow, _ = _fixture()
    wide, _ = _fixture(extra_items=3)
    for kwargs in ({}, {"prune_equal_support": True}):
        a = set(_mined(narrow, **kwargs))
        b = {m for m in _mined(wide, **kwargs) if all(i < 14 for i in m)}
        assert a == b, f"{kwargs}: gained {sorted(b - a)[:3]}, lost {sorted(a - b)[:3]}"


def test_generator_pruning_does_not_change_results():
    """Pascal inference is licensed by a non-free subset and is exact. The old
    rule inferred whenever all subsets shared a count, which overcounts."""
    df, _ = _fixture()
    assert _mined(df, use_generator_pruning=True) == _mined(df)
    assert _mined(df, use_generator_pruning=True, prune_equal_support=True) == _mined(
        df, prune_equal_support=True
    )


def test_infer_count_rejects_its_own_counterexample():
    from et_miner.core.apriori import _infer_count_from_subsets

    # support({A,B}) == support({A,C}) == support({B,C}) == 30 while
    # support({A,B,C}) == 15: all three pairs are free, so nothing is inferable.
    prev_counts = {("A", "B"): 30, ("A", "C"): 30, ("B", "C"): 30}
    assert _infer_count_from_subsets(("A", "B", "C"), prev_counts, set(prev_counts)) is None
    # with a non-free subset the count is exactly the minimum over the subsets
    assert _infer_count_from_subsets(("A", "B", "C"), prev_counts, {("A", "B"), ("A", "C")}) == 30
    assert _infer_count_from_subsets(("A", "B", "C"), prev_counts, None) is None


def test_row_split_helpers_keep_only_free_sets_without_a_gpu():
    """The row-split level loop, driven on the CPU with the engine's own helpers:
    groups from the kept level, count, free-set prune against the COMPLETE
    level. Guards the helpers on machines with no CUDA device."""
    from et_miner.gpu.kernels.k3plus import build_k3plus_groups_from_flat
    from et_miner.gpu.mining import _prune_non_free_mask

    _, matrix = _fixture()
    counts, free = _brute_force(matrix)
    n_rows, n_cols = matrix.shape

    kept_flat = np.array(sorted(c[0] for c in counts if len(c) == 1 and counts[c] != n_rows), dtype=np.int32)
    kept = (kept_flat.reshape(-1, 1), np.array([counts[(int(c),)] for c in kept_flat], dtype=np.int64))
    full = kept
    emitted = {(int(c),) for c in kept_flat}

    for k in range(2, MAX_K + 1):
        prev_flat, _ = kept
        full_flat, full_counts = full
        if len(prev_flat) < k:
            break
        if k == 2:
            candidates = list(itertools.combinations([int(x[0]) for x in prev_flat], 2))
        else:
            groups = build_k3plus_groups_from_flat(np.ascontiguousarray(prev_flat, dtype=np.int32))
            if groups is None:
                break
            pi, po = np.asarray(groups.prefix_items), np.asarray(groups.prefix_offsets)
            sf, so = np.asarray(groups.suffixes), np.asarray(groups.suffix_offsets)
            candidates = []
            for g in range(len(so) - 1):
                prefix = tuple(int(x) for x in pi[po[g] : po[g + 1]])
                suffixes = [int(x) for x in sf[so[g] : so[g + 1]]]
                candidates += [prefix + (a, b) for a, b in itertools.combinations(suffixes, 2)]

        frequent = {c: counts[c] for c in candidates if c in counts}
        if not frequent:
            break
        cur_flat = np.array(sorted(frequent), dtype=np.int32)
        cur_counts = np.array([frequent[tuple(r)] for r in cur_flat.tolist()], dtype=np.int64)
        full = (cur_flat, cur_counts)
        mask = _prune_non_free_mask(cur_flat, cur_counts, full_flat, full_counts)
        kept = (np.ascontiguousarray(cur_flat[mask]), cur_counts[mask])
        emitted |= {tuple(int(x) for x in row) for row in kept[0].tolist()}

    assert emitted == free


# ── GPU paths ───────────────────────────────────────────────────────────────

# NOT a module-level importorskip: the CPU guards above are the ones that must
# still run on a machine with no CUDA device.


def _gpu_count() -> int:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


#: (prune_apriori, use_generator_pruning): no subset test, the subset test, and count inference.
SUBSET_TEST = [(False, False), (True, False), (True, True)]


@pytest.mark.gpu
@pytest.mark.parametrize("n_gpus", [1, 2])
@pytest.mark.parametrize("prune", [False, True])
@pytest.mark.parametrize("sparse_from_k", [None, 3, "auto"])
@pytest.mark.parametrize("subset_test", SUBSET_TEST)
def test_row_split_matches_the_reference(n_gpus, prune, sparse_from_k, subset_test):
    if n_gpus > _gpu_count():
        pytest.skip(f"needs {n_gpus} GPUs")
    df, matrix = _fixture()
    counts, free = _brute_force(matrix)
    mined = _mined(df, use_gpu=True, n_gpus=n_gpus, prune_equal_support=prune, sparse_from_k=sparse_from_k,
                   prune_apriori=subset_test[0], use_generator_pruning=subset_test[1])
    assert set(mined) == (free if prune else set(counts))
    assert all(mined[c] == pytest.approx(counts[c] / len(matrix), abs=1e-12) for c in mined)
    _assert_emit_equals_generate(mined)


def _efficient_apriori_free_sets(df: pl.DataFrame, n_rows: int):
    """(lattice counts, free-sets) derived from efficient-apriori's lattice (boundary-safe, explicit depth)."""
    ea_apriori = pytest.importorskip("efficient_apriori").apriori
    min_count = int(np.ceil(MIN_SUPPORT * n_rows))
    itemsets, _ = ea_apriori(
        [tuple(r) for r in df["items"].to_list()], min_support=(min_count - 0.5) / n_rows,
        min_confidence=1.0, max_length=MAX_K,
    )
    counts = {tuple(sorted(s)): int(c) for level in itemsets.values() for s, c in level.items()}
    free = {
        c
        for c, v in counts.items()
        if v != n_rows and not any(counts.get(s) == v for r in range(1, len(c)) for s in itertools.combinations(c, r))
    }
    return counts, free


@pytest.mark.gpu
@pytest.mark.parametrize("n_gpus", [1, 2])
@pytest.mark.parametrize("sparse_from_k", [None, 3])
@pytest.mark.parametrize("subset_test", SUBSET_TEST[:2])
def test_free_sets_match_the_efficient_apriori_lattice(n_gpus, sparse_from_k, subset_test):
    """The free-sets of a GPU run equal those derived from the oracle's complete lattice."""
    if n_gpus > _gpu_count():
        pytest.skip(f"needs {n_gpus} GPUs")
    df, matrix = _fixture()
    counts, free = _efficient_apriori_free_sets(df, len(matrix))
    assert len(free) < len(counts), "fixture must actually exercise the prune"
    mined = _mined(df, use_gpu=True, n_gpus=n_gpus, prune_equal_support=True, sparse_from_k=sparse_from_k,
                   prune_apriori=subset_test[0])
    assert set(mined) == free, f"extra: {sorted(set(mined) - free)[:3]} missing: {sorted(free - set(mined))[:3]}"
    assert all(round(mined[c] * len(matrix)) == counts[c] for c in mined)


@pytest.mark.gpu
@pytest.mark.parametrize("prune", [False, True])
@pytest.mark.parametrize("subset_test", SUBSET_TEST)
def test_row_split_matches_the_reference_across_chunks(monkeypatch, prune, subset_test):
    """Several candidate chunks per level must not change the answer."""
    monkeypatch.setenv("ET_MINER_MAX_CHUNK_CANDS", "37")
    df, matrix = _fixture()
    counts, free = _brute_force(matrix)
    mined = _mined(df, use_gpu=True, n_gpus=min(2, _gpu_count()), prune_equal_support=prune,
                   prune_apriori=subset_test[0], use_generator_pruning=subset_test[1])
    assert set(mined) == (free if prune else set(counts))


@pytest.mark.gpu
@pytest.mark.parametrize("n_gpus", [1, 2])
@pytest.mark.parametrize("tiles", ["0", None, "group"])
def test_free_sets_on_wide_groups_match_the_efficient_apriori_lattice(monkeypatch, n_gpus, tiles):
    """Prefix groups wider than one 32-suffix tile, with implications: a counted
    tile-pair also counts its pairs that have a subset outside the free level, and
    those must still not be emitted. `tiles="0"` pins every group on the tiled kernel,
    `"group"` every group of at most 64 suffixes on the group kernel."""
    if n_gpus > _gpu_count():
        pytest.skip(f"needs {n_gpus} GPUs")
    if tiles == "group":
        monkeypatch.setenv("ET_MINER_SMALL_GROUP_KERNEL", "group")
        monkeypatch.setenv("ET_MINER_TILED_MIN_GROUP_PAIRS", str(10**12))
    elif tiles is not None:
        monkeypatch.setenv("ET_MINER_TILED_MIN_GROUP_PAIRS", tiles)
    rng = np.random.default_rng(3)
    weights = 1.0 / np.arange(1, 71) ** 0.9
    rows = []
    for _ in range(3000):
        items = {int(x) for x in rng.choice(70, size=int(rng.integers(2, 9)), replace=False, p=weights / weights.sum())}
        rows.append(sorted(items | {i % 6 for i in items if i >= 6}))
    df = pl.DataFrame({"items": rows}, schema={"items": pl.List(pl.Int64)})
    counts, free = _efficient_apriori_free_sets(df, len(rows))
    mined = _mined(df, use_gpu=True, n_gpus=n_gpus, prune_equal_support=True)
    assert set(mined) == free, f"extra: {sorted(set(mined) - free)[:3]} missing: {sorted(free - set(mined))[:3]}"


def _hidden_witness_fixture():
    """{1, 2, 3, 4} is generated from free parents {1,2,3} and {1,2,4} and is not
    free, but both of its equal-count (k-1)-subsets, {1,3,4} and {2,3,4}, are not
    free either ({3, 4} occurs only with 1 and 2), so neither is in the free level.
    Item 0 adds two more free 3-sets, so the free level is large enough for K=4;
    item 5 puts a free candidate, {1, 2, 3, 5}, in the same prefix group and tile."""
    blocks = {(1, 2, 3, 4): 200, (1, 2, 3): 100, (1, 2, 4): 100, (3,): 100, (4,): 100, (1, 3): 100,
              (2, 4): 100, (1, 4): 100, (2, 3): 100, (1, 2): 100, (1,): 100, (2,): 100,
              (0, 1, 2): 100, (0, 1, 3): 100, (0, 1): 100, (0, 2): 100, (0, 3): 100, (0,): 100,
              (1, 2, 3, 5): 100, (1, 3, 5): 50, (2, 3, 5): 50, (1, 2, 5): 50, (1, 5): 50, (2, 5): 50,
              (3, 5): 50, (5,): 50}
    rows = [list(b) for b, n in blocks.items() for _ in range(n)]
    matrix = np.zeros((len(rows), 6), dtype=bool)
    for i, r in enumerate(rows):
        matrix[i, r] = True
    return pl.DataFrame({"items": rows}, schema={"items": pl.List(pl.Int64)}), matrix


@pytest.mark.gpu
@pytest.mark.parametrize("n_gpus", [1, 2])
@pytest.mark.parametrize("path", ["dispatch", "tiled", "per-candidate", "group", "no index", "no subset test", "esco"])
def test_free_sets_with_hidden_witnesses(monkeypatch, n_gpus, path):
    """{1, 2, 3, 4} must not be emitted on any path: the tiled kernels count it
    whenever its tile-pair holds a pair to count ({1, 2, 3, 5}), a level whose
    index does not fit counts it outright, and the free-set test, which resolves
    against the free level only, cannot see its witnesses. Such survivors are
    dropped on the host before the test."""
    if n_gpus > _gpu_count():
        pytest.skip(f"needs {n_gpus} GPUs")
    from et_miner.gpu import row_split

    df, matrix = _hidden_witness_fixture()
    counts, free = _brute_force(matrix, max_k=4)
    assert (1, 2, 3, 4) in counts and (1, 2, 3, 4) not in free and (1, 2, 3, 5) in free
    assert {(1, 2, 3), (1, 2, 4)} <= free and not {(1, 3, 4), (2, 3, 4)} & free
    kwargs = {}
    if path == "tiled":
        monkeypatch.setenv("ET_MINER_TILED_MIN_GROUP_PAIRS", "0")
    elif path == "per-candidate":
        monkeypatch.setenv("ET_MINER_TILED_MIN_GROUP_PAIRS", str(10**12))
        monkeypatch.setenv("ET_MINER_SMALL_GROUP_KERNEL", "percand")
    elif path == "group":
        monkeypatch.setenv("ET_MINER_TILED_MIN_GROUP_PAIRS", str(10**12))
        monkeypatch.setenv("ET_MINER_SMALL_GROUP_KERNEL", "group")
    elif path == "no index":
        monkeypatch.setattr(row_split, "_device_available_bytes", lambda did: (0, 1 << 34))
    elif path == "no subset test":
        kwargs["prune_apriori"] = False
    elif path == "esco":
        kwargs["sparse_from_k"] = 3
    assert set(_mined(df, use_gpu=True, n_gpus=n_gpus, prune_equal_support=True, **kwargs)) == free
    assert set(_mined(df, use_gpu=True, n_gpus=n_gpus, **kwargs)) == set(counts)
    assert set(_mined(df, use_gpu=True, n_gpus=n_gpus, use_generator_pruning=True)) == set(counts)


@pytest.mark.gpu
@pytest.mark.parametrize("prune", [False, True])
def test_bitvecs_route_honours_the_flag_and_never_writes_to_the_caller(prune):
    """Defect B: the flag reached no pruning implementation on this route, and
    the engine zeroed dead columns in the caller's own array."""
    from et_miner.core.matrix import _build_csr_from_transactions
    from et_miner.gpu.bitvec import _build_gpu_bitvec_matrix

    df, matrix = _fixture()
    counts, free = _brute_force(matrix)
    csr, col_to_item, n_trans = _build_csr_from_transactions(df.lazy(), MIN_SUPPORT, "items")
    bitvecs = _build_gpu_bitvec_matrix(csr)
    before = bitvecs.copy()

    result = apriori(
        bitvecs=(bitvecs, col_to_item, n_trans),
        min_support=MIN_SUPPORT,
        max_length=MAX_K,
        prune_equal_support=prune,
    )
    mined = {
        tuple(sorted(int(i) for i in row["itemset"])): float(row["support"])
        for row in result.iter_rows(named=True)
    }
    assert set(mined) == (free if prune else set(counts))
    assert bool((before == bitvecs).all()), "the engine wrote to an array it was handed"


@pytest.mark.gpu
@pytest.mark.gpu
def test_profile_with_pruning_returns_the_session_and_the_same_itemsets():
    from et_miner.core.profiling import ProfilingSession

    df, _ = _fixture()
    plain = apriori(df, min_support=MIN_SUPPORT, use_gpu=True, prune_equal_support=True)
    result, session = apriori(df, min_support=MIN_SUPPORT, use_gpu=True, prune_equal_support=True, profile=True)
    assert isinstance(session, ProfilingSession) and session.phases
    assert sorted(map(tuple, result["itemset"].to_list())) == sorted(map(tuple, plain["itemset"].to_list()))


@pytest.mark.gpu
@pytest.mark.parametrize("resume_from_k", [2, 3])
def test_resumed_free_set_run_is_exact(tmp_path, resume_from_k):
    """The flushed level of a free-set run holds only the free-sets; with the
    subset test that is all the next level needs, so a resumed run emits exactly
    the free-sets an uninterrupted run emits."""
    if _gpu_count() < 1:
        pytest.skip("needs a CUDA device")
    import glob
    import os

    df, matrix = _hidden_witness_fixture()
    _, free = _brute_force(matrix, max_k=4)
    out = str(tmp_path / "run")
    apriori(df, min_support=MIN_SUPPORT, max_length=4, use_gpu=True, prune_equal_support=True, output_dir=out)
    for k in range(resume_from_k + 1, 5):
        os.remove(f"{out}/frequent_k{k}.parquet")
    apriori(df, min_support=MIN_SUPPORT, max_length=4, use_gpu=True, prune_equal_support=True, output_dir=out,
            resume_from_k=resume_from_k)
    flushed = pl.concat([pl.read_parquet(p) for p in sorted(glob.glob(f"{out}/frequent_k*.parquet"))])
    assert {tuple(sorted(x)) for x in flushed["itemset"].to_list()} == free
