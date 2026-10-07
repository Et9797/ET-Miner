"""Crossover of the two K>=3 group counters of the CPU miner: bitvector pairs vs projection.

Takes real prefix groups from a workload (K=3: one-item prefixes and their
frequent partners; K=4: frequent pairs and their frequent third items),
shrinks each group's suffix set to a range of sizes (evenly spaced
suffixes), and times both counters on the same group: ``gbitvec`` (prefix AND,
pair popcounts on the prefix's non-zero words) and ``proj`` (Gram matrix of
the suffix columns over the prefix's rows, rows taken from the prefix AND as
``count_candidates`` does). Both must return the same counts. Prints one JSON
line per (workload, k, suffixes) with the median time per group of each
counter and their ratio.

Usage:
    uv run python bench/cpu/l3_crossover.py --workloads or002,wide,deepk,skew --out FILE

Options:
    --workloads  ids of bench/cpu/matrix.py WORKLOADS
    --sizes      suffix counts to measure
    --groups     groups measured per (workload, k, size), the largest first
    --rows       keep only the first N rows of each workload (an intermediate width)
    --out        jsonl to append to
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from consolidation_run import _load  # noqa: E402
from cpu.matrix import WORKLOADS  # noqa: E402

from et_miner.core import cpu_miner  # noqa: E402
from et_miner.core.result import _min_count  # noqa: E402


def _time(fn, reps: int = 3) -> tuple[float, np.ndarray]:
    best, out = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def _groups(level: np.ndarray, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(prefix, suffixes) of the (k-1)-level's runs of equal first k-2 items."""
    starts = np.flatnonzero(np.r_[True, np.any(level[1:, : k - 2] != level[:-1, : k - 2], axis=1)])
    ends = np.r_[starts[1:], len(level)]
    return [(level[s, : k - 2], level[s:e, k - 2]) for s, e in zip(starts, ends)]


def measure(workload: str, sizes: list[int], n_groups: int, rows: int | None = None) -> list[dict]:
    """Time both counters on shrunk real groups at K=3 and K=4 of one workload."""
    import polars as pl

    dataset, min_support, _ = WORKLOADS[workload]
    df, n_rows, _ = _load(dataset)
    if rows:
        df, n_rows = df.head(rows), min(rows, n_rows)
    tc = cpu_miner.build_transaction_csr(df.lazy() if isinstance(df, pl.DataFrame) else df, min_support, "items")
    min_count = _min_count(min_support, n_rows)
    space = cpu_miner.RowSpace(tc.indptr, tc.indices, tc.n_cols)
    pairs, _ = cpu_miner.count_pairs(tc.to_scipy(), np.ones(tc.n_cols, dtype=bool), min_count)
    cands3 = np.concatenate(list(cpu_miner.generate_candidates(pairs, 3, tc.n_cols)))
    triples = cands3[cpu_miner.count_candidates(cands3, 3, space) >= min_count]
    bv = space.bitvecs
    out = []
    for k, level in ((3, pairs), (4, triples)):
        groups = sorted(_groups(level, k), key=lambda g: -len(g[1]))
        for size in sizes:
            ratios, tg_all, tp_all = [], [], []
            for prefix, suffixes in [g for g in groups if len(g[1]) >= size][:n_groups]:
                suffix = suffixes[np.linspace(0, len(suffixes) - 1, size).astype(int)]
                ia, ib = (np.array(x, dtype=np.int64) for x in zip(*[(i, j) for i in range(size) for j in range(i + 1, size)]))
                pre = bv[prefix[0]].copy()
                for it in prefix[1:]:
                    pre &= bv[it]
                tg, cg = _time(lambda: cpu_miner._count_group_bitvec(bv, pre, ia, ib, suffix))
                tp, cp = _time(lambda: cpu_miner._count_group_proj(
                    space.csr, np.flatnonzero(np.unpackbits(pre.view(np.uint8), bitorder="little")), ia, ib, suffix))
                assert np.array_equal(cg, cp), "counters disagree"
                tg_all.append(tg)
                tp_all.append(tp)
                ratios.append(tp / tg)
            if ratios:
                row = {"workload": workload, "rows": n_rows, "k": k, "words": space.words, "suffixes": size, "groups": len(ratios),
                       "gbitvec_ms": round(1000 * statistics.median(tg_all), 4),
                       "proj_ms": round(1000 * statistics.median(tp_all), 4),
                       "proj_over_gbitvec": round(statistics.median(ratios), 3)}
                print(json.dumps(row), flush=True)
                out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", default="or002,wide,deepk,skew")
    ap.add_argument("--sizes", default="4,8,16,24,32,48,64,96,128,192,256,384,512")
    ap.add_argument("--groups", type=int, default=20)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    sizes = [int(x) for x in args.sizes.split(",")]
    for w in args.workloads.split(","):
        for row in measure(w, sizes, args.groups, args.rows):
            if args.out:
                with args.out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
