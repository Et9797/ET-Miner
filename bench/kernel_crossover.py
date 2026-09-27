"""Time the dense per-candidate and tiled K>=3 kernels over group shape and row count.

Builds synthetic prefix groups (a prefix of K-2 items and `m` suffixes per
group, the number of groups chosen so every point counts about the same number
of candidates) over correlated random bitvecs of `n_u64s` words, and times
count_k3plus_allcounts with variant="legacy" and variant="shared" on one GPU,
the median of --reps launches after a warm-up. Prints one JSON line per point.

Usage: CUDA_VISIBLE_DEVICES=0 python bench/kernel_crossover.py [--reps 3] > crossover.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

WORDS = (570, 7_813, 31_250, 312_500)
SUFFIXES = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512)
KS = (3, 6)
TARGET_CANDIDATES = 200_000
MAX_WORDS_X_CANDS = 7e10


def _random_word(cp, shape):
    lo = cp.random.randint(0, 2**32, size=shape, dtype=cp.int64).astype(cp.uint64)
    hi = cp.random.randint(0, 2**32, size=shape, dtype=cp.int64).astype(cp.uint64)
    return (hi << cp.uint64(32)) | lo


def _random_bitvecs(cp, n_cols: int, n_u64s: int):
    """Correlated columns: a shared base (1 bit in 4) AND per-column noise (1 in 2).

    One column is 1/8 dense and a k-itemset 1/4 * 2**-k, so supports stay in
    the 1-10% range of real deep levels instead of vanishing like independent
    columns would.
    """
    base = _random_word(cp, (1, n_u64s)) & _random_word(cp, (1, n_u64s))
    return _random_word(cp, (n_cols, n_u64s)) & base


def _groups(k: int, m: int, n_groups: int, n_cols: int, rng):
    from et_miner.gpu.kernels import K3PlusGroups

    p = k - 2
    prefix = rng.integers(0, n_cols, size=n_groups * p).astype(np.int32)
    suffixes = np.concatenate([np.sort(rng.choice(n_cols, size=m, replace=False)) for _ in range(n_groups)])
    pairs = m * (m - 1) // 2
    return K3PlusGroups(
        prefix_items=prefix,
        prefix_offsets=np.arange(0, (n_groups + 1) * p, p, dtype=np.int64),
        suffixes=suffixes.astype(np.int32),
        suffix_offsets=np.arange(0, (n_groups + 1) * m, m, dtype=np.int64),
        cumulative_pairs=np.arange(0, (n_groups + 1) * pairs, pairs, dtype=np.int64),
        total_candidates=n_groups * pairs,
        groups=None,
    )


def main() -> int:
    import cupy as cp

    from et_miner.gpu.kernels import count_k3plus_allcounts, upload_k3plus_groups
    from et_miner.gpu.kernels.shared_tiled import compute_cumulative_tilepairs

    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    n_cols = 600
    for n_u64s in WORDS:
        bitvecs = _random_bitvecs(cp, n_cols, n_u64s)
        for k, m in [(k, m) for k in KS for m in SUFFIXES]:
            pairs = m * (m - 1) // 2
            n_groups = max(1, TARGET_CANDIDATES // pairs)
            while n_groups > 1 and n_groups * pairs * n_u64s > MAX_WORDS_X_CANDS:
                n_groups //= 2
            groups = _groups(k, m, n_groups, n_cols, rng)
            gpu = upload_k3plus_groups(groups, 0)
            point = {
                "k": k,
                "n_u64s": n_u64s,
                "suffixes": m,
                "groups": n_groups,
                "candidates": groups.total_candidates,
                "tilepairs": int(compute_cumulative_tilepairs(groups.suffix_offsets)[-1]),
            }
            results = {}
            for variant in ("legacy", "shared"):
                count_k3plus_allcounts(bitvecs, groups, n_u64s, groups_gpu=gpu, variant=variant)
                times = []
                for _ in range(args.reps):
                    cp.cuda.Device(0).synchronize()
                    t0 = time.perf_counter()
                    out = count_k3plus_allcounts(bitvecs, groups, n_u64s, groups_gpu=gpu, variant=variant)
                    cp.cuda.Device(0).synchronize()
                    times.append(time.perf_counter() - t0)
                results[variant] = out.get()
                point[f"{variant}_s"] = statistics.median(times)
            point["equal"] = bool(np.array_equal(results["legacy"], results["shared"]))
            point["shared_over_legacy"] = point["shared_s"] / point["legacy_s"]
            print(json.dumps(point), flush=True)
            del gpu
            cp.get_default_memory_pool().free_all_blocks()
        del bitvecs
        cp.get_default_memory_pool().free_all_blocks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
