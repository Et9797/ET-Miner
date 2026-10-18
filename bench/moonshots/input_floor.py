"""The host work a GPU-native input layer keeps (M1's floor), timed on the CPU.

For a dataset already in memory (the ``apriori()`` call's input), times the
steps M1 still runs on the host: the row lengths, and the exploded item ids as
an int32 NumPy array, whole and in row chunks. For reference it also times the
host K=1 of the CPU route's integer path (a bincount over the id span) and the
two Polars passes the GPU route runs today (the K=1 group_by and the
explode-and-join). Median of ``--reps`` runs per step, seconds.

Usage: uv run python bench/moonshots/input_floor.py DATASET [--reps 3] [--chunk-rows 5000000]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _timed(fn, reps: int) -> float:
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> int:
    import polars as pl
    from consolidation_run import _load

    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--chunk-rows", type=int, default=5_000_000)
    args = ap.parse_args()
    df, n_rows, _ = _load(args.dataset)
    items = df["items"]
    lf = df.lazy()
    out = {"dataset": args.dataset, "n_rows": n_rows, "nnz": int(items.list.len().sum())}
    out["lengths"] = _timed(lambda: items.list.len().fill_null(0).cast(pl.Int32).to_numpy(), args.reps)
    out["explode_int32"] = _timed(lambda: items.explode().drop_nulls().cast(pl.Int32).to_numpy(), args.reps)

    def chunked():
        for s in range(0, n_rows, args.chunk_rows):
            items.slice(s, args.chunk_rows).explode().drop_nulls().cast(pl.Int32).to_numpy()

    out["explode_int32_chunked"] = _timed(chunked, args.reps)
    flat = items.explode().drop_nulls().to_numpy().astype(np.int64)
    lo = int(flat.min())
    out["bincount_k1"] = _timed(lambda: np.bincount(flat - lo), args.reps)
    out["polars_k1_group_by"] = _timed(
        lambda: lf.select(pl.col("items").explode().drop_nulls()).group_by("items").agg(pl.len()).collect(
            engine="streaming"
        ),
        args.reps,
    )
    freq = lf.select(pl.col("items").explode().drop_nulls()).group_by("items").agg(pl.len()).collect()["items"]
    mapping = pl.DataFrame({"items": freq, "__col": np.arange(len(freq), dtype=np.int64)}).lazy()
    out["polars_explode_join"] = _timed(
        lambda: lf.with_row_index("__row").explode("items").join(mapping, on="items", how="inner").select(
            pl.col("__row").cast(pl.Int64), "__col"
        ).collect(engine="streaming"),
        1,
    )
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
