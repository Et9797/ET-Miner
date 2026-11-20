"""The deduction bounds of the K>=3 subset test (``_src/_subset_index.cu``, SUBSET_BOUNDS) against a host reference.

A candidate that count inference leaves gets
  u = min(min_a s(X-a), min_{a<b<c} [s(X-a)+s(X-b)+s(X-c) - s(X-ab)-s(X-ac)-s(X-bc) + s(X-abc)]),
  l = max(0, max_{a<b} [s(X-a) + s(X-b) - s(X-ab)]),
and is skipped when u < min_count, given u when l = u, counted otherwise. The
kernels run on zeroed bitvectors here, so a written count can only be a settled
one. End to end, mining with the bounds matches the oracle with every kernel
pinned, on one GPU and two.
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import polars as pl
import pytest

from et_miner.core.result import _min_count
from tests._oracle import ea_counts, mined_counts


def _gpu_count() -> int:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


def _lattice(rows, min_count: int, max_k: int) -> dict[tuple, int]:
    counts: Counter = Counter()
    for r in rows:
        for k in range(1, max_k + 1):
            counts.update(itertools.combinations(r, k))
    return {s: c for s, c in counts.items() if c >= min_count}


def _reference(x: tuple, lat: dict, n_total: int, min_count: int) -> tuple[str, int]:
    """('pascal' | 'skip' | 'infer' | 'count', value) of candidate x whose (k-1)-subsets are all frequent.

    'pascal' is today's inference (a non-free subset); 'skip' and 'infer' are
    the bounds' outcomes.
    """
    k = len(x)
    s = {sub: (lat[sub] if sub else n_total) for d in (1, 2, 3) for sub in itertools.combinations(x, k - d)}
    c1 = [s[tuple(v for v in x if v != a)] for a in x]
    free = [all(s[sub] != s[tuple(v for v in sub if v != b)] for b in sub) for sub in itertools.combinations(x, k - 1)]
    if not all(free):
        return "pascal", min(c1)
    upper = min(c1)
    lower = 0
    for a, b in itertools.combinations(range(k), 2):
        rest = tuple(v for i, v in enumerate(x) if i not in (a, b))
        lower = max(lower, c1[a] + c1[b] - s[rest])
    for a, b, c in itertools.combinations(range(k), 3):
        ab = tuple(v for i, v in enumerate(x) if i not in (a, b))
        ac = tuple(v for i, v in enumerate(x) if i not in (a, c))
        bc = tuple(v for i, v in enumerate(x) if i not in (b, c))
        abc = tuple(v for i, v in enumerate(x) if i not in (a, b, c))
        upper = min(upper, c1[a] + c1[b] + c1[c] - s[ab] - s[ac] - s[bc] + s[abc])
    if upper < min_count:
        return "skip", 0
    if lower == upper:
        return "infer", upper
    return "count", 0


def _dense_rows(seed: int, n_rows: int) -> list[tuple]:
    """Seven items present independently (40-80 %), and a row holding item 0 also holds item 1 or item 2.

    The dense lattice gives upper bounds below min_count, and the cover of
    item 0 by items 1 and 2 gives equal bounds: both outcomes Pascal inference
    does not reach.
    """
    rng = np.random.default_rng(seed)
    p = np.array([0.8, 0.5, 0.5, 0.6, 0.6, 0.4, 0.4])
    rows = []
    for _ in range(n_rows):
        r = set(np.flatnonzero(rng.random(len(p)) < p).tolist())
        if 0 in r and not {1, 2} & r:
            r.add(1 if rng.random() < 0.5 else 2)
        rows.append(tuple(sorted(r)))
    return rows


#: (k, seed) cases where the bounds settle candidates; (4, 17) and (4, 53) include equal bounds.
_BOUND_CASES = [(3, 5), (3, 10), (3, 53), (4, 10), (4, 17), (4, 53)]


@pytest.mark.gpu
@pytest.mark.parametrize("k, seed", _BOUND_CASES)
def test_classifier_matches_the_host_reference(k, seed):
    """Per-candidate kernel on zeroed bitvecs: settled candidates carry their count, the rest stay 0."""
    import cupy as cp

    from et_miner.gpu.kernels import (
        SUBSET_BOUNDS,
        SUBSET_INFER,
        SUBSET_PRUNE,
        build_k3plus_groups_from_flat,
        count_k3plus_per_candidate,
        upload_subset_index,
    )

    rows = _dense_rows(seed, 300)
    n_total, min_count = len(rows), 40
    lat = _lattice(rows, min_count, k)

    def level(m):
        items = sorted(s for s in lat if len(s) == m)
        return np.array(items, dtype=np.int32).reshape(-1, m), np.array([lat[s] for s in items], dtype=np.int64)

    prev, prev_counts = level(k - 1)
    free = np.array(
        [all(lat[tuple(r)] != lat.get(tuple(v for v in r if v != b), n_total) for b in r) for r in prev.tolist()]
    )
    deep = [level(k - 2), level(k - 3) if k > 3 else (np.zeros((0, 0), np.int32), np.zeros(0, np.int64))]
    index = upload_subset_index(
        prev,
        prev_counts,
        free,
        mode=SUBSET_PRUNE | SUBSET_INFER | SUBSET_BOUNDS,
        device_id=0,
        write_inferred=True,
        deep=deep,
        n_total=n_total,
        min_count=min_count,
    )
    groups = build_k3plus_groups_from_flat(prev)
    zeros = cp.zeros((7, (n_total + 63) // 64), dtype=cp.uint64)
    got = count_k3plus_per_candidate(zeros, groups, zeros.shape[1], index=index).get()

    prev_set = {tuple(r) for r in prev.tolist()}
    outcomes = Counter()
    cand = 0
    for g in range(len(groups.prefix_offsets) - 1):
        pref = tuple(groups.prefix_items[groups.prefix_offsets[g] : groups.prefix_offsets[g + 1]].tolist())
        suf = groups.suffixes[groups.suffix_offsets[g] : groups.suffix_offsets[g + 1]].tolist()
        for j in range(len(suf)):
            for i in range(j):
                x = pref + (suf[i], suf[j])
                if all(tuple(v for v in x if v != a) in prev_set for a in x):
                    kind, value = _reference(x, lat, n_total, min_count)
                else:
                    kind, value = "subset", 0
                outcomes[kind] += 1
                assert got[cand] == value, (x, kind, value, got[cand])
                cand += 1
    assert cand == len(got)
    assert outcomes["skip"] + outcomes["infer"] > 0, f"the bounds settled nothing: {dict(outcomes)}"


@pytest.mark.gpu
@pytest.mark.parametrize(
    "pin",
    [
        pytest.param({}, id="dispatch"),
        pytest.param({"ET_MINER_TILED_MIN_GROUP_PAIRS": "0"}, id="tiled"),
        pytest.param(
            {"ET_MINER_TILED_MIN_GROUP_PAIRS": str(10**12), "ET_MINER_SMALL_GROUP_KERNEL": "percand"},
            id="per-candidate",
        ),
        pytest.param(
            {"ET_MINER_TILED_MIN_GROUP_PAIRS": str(10**12), "ET_MINER_SMALL_GROUP_KERNEL": "group"}, id="group"
        ),
        pytest.param({"ET_MINER_MAX_CHUNK_CANDS": "37", "ET_MINER_TILED_MIN_GROUP_PAIRS": "0"}, id="forced-chunks"),
    ],
)
@pytest.mark.parametrize("n_gpus", [1, pytest.param(2, marks=pytest.mark.multigpu)])
@pytest.mark.parametrize("seed", [10, 53])
def test_mining_with_the_bounds_matches_the_oracle(monkeypatch, pin, n_gpus, seed):
    """Ties at min_count, empty rows and an item in every row (all-ones words)."""
    if _gpu_count() < n_gpus:
        pytest.skip(f"needs {n_gpus} CUDA devices")
    from et_miner import apriori

    rows = [list(r) + [99] for r in _dense_rows(seed, 1_000 + 37 * seed)]
    for i in range(0, len(rows), 9):
        rows[i] = []
    counts = sorted(Counter(p for r in rows for p in itertools.combinations(sorted(r), 3)).values())
    c = counts[len(counts) // 2]
    s = c / len(rows)
    while _min_count(s, len(rows)) > c:
        s = np.nextafter(s, 0.0)
    for name, value in pin.items():
        monkeypatch.setenv(name, value)
    df = pl.DataFrame({"items": pl.Series(rows, dtype=pl.List(pl.Int64))})
    results = {}
    for bounds in ("0", "1"):
        monkeypatch.setenv("ET_MINER_DEDUCTION", bounds)
        res = apriori(df, min_support=float(s), use_gpu=True, n_gpus=n_gpus, use_generator_pruning=True)
        results[bounds] = mined_counts(res, len(rows))
    assert results["1"] == results["0"] == ea_counts(rows, float(s))


@pytest.mark.gpu
def test_without_room_for_the_older_levels_only_the_bounds_go(monkeypatch):
    """Room for every K-1 index but not for the older levels' beside it: the subset test and inference stay."""
    from et_miner import apriori
    from et_miner.gpu import kernels, row_split

    rows = [list(r) + [99] for r in _dense_rows(10, 1_370)]
    counts = sorted(Counter(p for r in rows for p in itertools.combinations(sorted(r), 3)).values())
    c = counts[len(counts) // 2]
    s = c / len(rows)
    while _min_count(s, len(rows)) > c:
        s = np.nextafter(s, 0.0)
    df = pl.DataFrame({"items": pl.Series(rows, dtype=pl.List(pl.Int64))})
    uploads = {}
    real_upload = kernels.upload_subset_index

    def spy(index_rows, *a, **k):
        uploads[index_rows.shape[1] + 1] = (k["mode"], k.get("deep") is not None, kernels.index_nbytes(*index_rows.shape))
        return real_upload(index_rows, *a, **k)

    monkeypatch.setattr(kernels, "upload_subset_index", spy)
    monkeypatch.setenv("ET_MINER_DEDUCTION", "1")
    roomy = apriori(df, min_support=float(s), use_gpu=True, use_generator_pruning=True)
    every = dict(uploads)
    assert all(mode & kernels.SUBSET_BOUNDS for mode, _, _ in every.values())

    widest = max(nbytes for _, _, nbytes in every.values())
    real_free = row_split._device_available_bytes
    monkeypatch.setattr(row_split, "_device_available_bytes", lambda did: (4 * widest, real_free(did)[1]))
    uploads.clear()
    tight = apriori(df, min_support=float(s), use_gpu=True, use_generator_pruning=True)
    assert set(uploads) == set(every), "a level lost its subset test"
    dropped = [k for k, (mode, deep, _) in uploads.items() if not mode & kernels.SUBSET_BOUNDS]
    assert dropped, "every level kept its bounds: the test is vacuous"
    for k in dropped:
        mode, deep, _ = uploads[k]
        assert mode == kernels.SUBSET_PRUNE | kernels.SUBSET_INFER and not deep, k
    want = ea_counts(rows, float(s))
    assert mined_counts(tight, len(rows)) == mined_counts(roomy, len(rows)) == want


def _bounds_frame(seed: int = 10):
    rows = [list(r) + [99] for r in _dense_rows(seed, 1_000 + 37 * seed)]
    counts = sorted(Counter(p for r in rows for p in itertools.combinations(sorted(r), 3)).values())
    c = counts[len(counts) // 2]
    s = c / len(rows)
    while _min_count(s, len(rows)) > c:
        s = np.nextafter(s, 0.0)
    return rows, float(s), pl.DataFrame({"items": pl.Series(rows, dtype=pl.List(pl.Int64))})


@pytest.mark.gpu
@pytest.mark.parametrize("bounds", ["0", "1"])
def test_older_levels_are_kept_only_where_the_bounds_may_run(monkeypatch, bounds):
    """The miner's complete levels, read at each level's callback: none without the bounds, at most three with."""
    import sys

    from et_miner import apriori

    rows, s, df = _bounds_frame()
    held = []
    monkeypatch.setenv("ET_MINER_DEDUCTION", bounds)
    res = apriori(
        df,
        min_support=s,
        use_gpu=True,
        use_generator_pruning=True,
        level_callback=lambda k, *_: held.append(sorted(sys._getframe(1).f_locals["_complete_levels"])),
    )
    assert len(held) >= 3
    if bounds == "0":
        assert all(not levels for levels in held), held
    else:
        assert all(1 <= len(levels) <= 3 for levels in held), held
    assert mined_counts(res, len(rows)) == ea_counts(rows, s)


@pytest.mark.gpu
def test_the_memory_guard_drops_the_older_levels_before_it_trips(monkeypatch):
    """Over max_ram_gb once, after K=1: the guard drops the bounds' levels, measures again and mines on."""
    import sys

    import psutil

    from et_miner import apriori
    from et_miner.gpu import kernels

    rows, s, df = _bounds_frame()
    uploads = {}
    real_upload = kernels.upload_subset_index

    def spy(index_rows, *a, **k):
        uploads[index_rows.shape[1] + 1] = k.get("deep") is not None
        return real_upload(index_rows, *a, **k)

    monkeypatch.setattr(kernels, "upload_subset_index", spy)
    monkeypatch.setenv("ET_MINER_DEDUCTION", "1")
    roomy = apriori(df, min_support=s, use_gpu=True, use_generator_pruning=True, max_ram_gb=1e6)
    assert uploads and all(uploads.values())

    real_info = psutil.Process.memory_info
    over = []

    def memory_info(self):
        info = real_info(self)
        if not over and sys._getframe(1).f_code.co_name == "_memory_gb":
            over.append(1)
            return info._replace(rss=1 << 60)
        return info

    monkeypatch.setattr(psutil.Process, "memory_info", memory_info)
    uploads.clear()
    tight = apriori(df, min_support=s, use_gpu=True, use_generator_pruning=True, max_ram_gb=1e6)
    assert over, "the guard never measured"
    assert uploads and not any(uploads.values()), "a level kept the bounds the guard dropped"
    assert mined_counts(tight, len(rows)) == mined_counts(roomy, len(rows)) == ea_counts(rows, s)


def test_deduction_knob_is_validated(monkeypatch):
    from et_miner import _env

    monkeypatch.setenv("ET_MINER_DEDUCTION", "yes")
    with pytest.raises(ValueError, match="ET_MINER_DEDUCTION"):
        _env.deduction()
    monkeypatch.setenv("ET_MINER_DEDUCTION", "0")
    assert _env.deduction() == "0"
