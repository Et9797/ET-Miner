"""Row-split: the Apriori subset test was welded to the free-set gate.

`core/apriori.py` dispatched to `_apriori_row_split_multi_gpu` with
`prune_apriori=prune_equal_support`. A complete-lattice run on that route
(`n_gpus>1` or `anchor_items`, `prune_equal_support=False`) therefore counted
every suffix extension of every frequent group with no downward closure at all.
Exact either way -- the subset test only removes candidates that cannot be
frequent -- so the output never showed it; the K>=3 candidate count did.

`64` is a filename slot, not a BUGS_FOUND number: this came out of a downstream
pipeline's recon (its plan calls the fix D2), not the 62-defect review.

LIVE when `apriori()` has no `prune_apriori` parameter, or when a complete-lattice
row-split run with the default never runs the subset test. The run is observed
directly by wrapping `gpu.row_split._prune_groups_apriori` (the row-split miner's
GPU workers are threads of this process), and the mined lattice is checked
against a host-side brute-force count so exactness is asserted at the same time.
Needs two CUDA devices; about two seconds.
"""

from __future__ import annotations

import inspect
import itertools
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

N_ROWS = 4000
N_ITEMS = 14
MIN_SUPPORT = 0.03
MAX_LENGTH = 5
SEED = 7


def _fixture():
    """Nested vocabulary plus themed noise: frequent pairs are plentiful,
    frequent triples are not, so K>=3 groups hold prunable candidates."""
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(SEED)
    parent = {i: i % 4 for i in range(4, N_ITEMS)}
    themes = [rng.choice(range(4, N_ITEMS), size=5, replace=False) for _ in range(8)]
    matrix = np.zeros((N_ROWS, N_ITEMS), dtype=bool)
    rows = []
    for i in range(N_ROWS):
        items = {int(x) for x in themes[i % 8][rng.random(5) < 0.9]}
        items |= {parent[x] for x in items}
        items |= {int(x) for x in rng.choice(N_ITEMS, size=int(rng.integers(0, 3)), replace=False)}
        items.add(1)
        rows.append(sorted(items))
        matrix[i, list(items)] = True
    return pl.DataFrame({"items": rows}, schema={"items": pl.List(pl.Int64)}), matrix


def _brute_force(matrix):
    import numpy as np

    min_count = int(np.ceil(MIN_SUPPORT * N_ROWS))
    counts = {}
    for k in range(1, MAX_LENGTH + 1):
        for c in itertools.combinations(range(N_ITEMS), k):
            count = int(np.logical_and.reduce(matrix[:, list(c)], axis=1).sum())
            if count >= min_count:
                counts[c] = count
    return counts


def reproduce():
    """Returns (is_live, evidence) -- the bench/repro contract."""
    sys.path.insert(0, str(REPO / "src"))
    try:
        import cupy as cp
    except ImportError:
        return False, "SKIPPED: no cupy"
    n_devices = cp.cuda.runtime.getDeviceCount()
    if n_devices < 2:
        return False, f"SKIPPED: needs 2 CUDA devices, found {n_devices}"

    from et_miner.core.apriori import apriori
    from et_miner.gpu import row_split

    if "prune_apriori" not in inspect.signature(apriori).parameters:
        return True, "apriori() has no prune_apriori parameter: the subset test follows prune_equal_support"

    real = row_split._prune_groups_apriori
    calls: list[tuple[int, int, int]] = []

    def spy(groups_info, prev_frequent_set, k, prev_flat_np=None):
        before = int(groups_info.total_candidates)
        out = real(groups_info, prev_frequent_set, k, prev_flat_np=prev_flat_np)
        calls.append((k, before, int(out.total_candidates) if out is not None else 0))
        return out

    row_split._prune_groups_apriori = spy
    try:
        df, matrix = _fixture()
        result = apriori(
            df, min_support=MIN_SUPPORT, max_length=MAX_LENGTH, use_gpu=True, n_gpus=2, prune_equal_support=False
        )
    finally:
        row_split._prune_groups_apriori = real

    mined = {
        tuple(sorted(int(i) for i in row["itemset"])): round(float(row["support"]) * N_ROWS)
        for row in result.iter_rows(named=True)
    }
    want = _brute_force(matrix)
    exact = mined == want
    pruned = any(after < before for _, before, after in calls)
    is_live = (not exact) or (not calls) or (not pruned)
    evidence = f"devices={n_devices} lattice={len(want):,} mined={len(mined):,} exact={exact} subset_test_calls={calls}"
    return is_live, evidence


if __name__ == "__main__":
    live, ev = reproduce()
    print(f"{'LIVE ' if live else 'FIXED'}  {ev}")
    sys.exit(1 if live else 0)
