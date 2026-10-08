"""Stages of the CPU route's input layer: the CSR build and the bitvector builds (diagnostic).

Per workload and thread setting, in a fresh process with the five pools pinned:
load the workload, warm up on a 20,000-row smoke slice (min_support 0.02,
max_length 3), then

1. one ``mine_cpu(profile=True)`` call, recording every ``build_bitvecs`` call
   (selected rows, columns, words, seconds, pooled or not);
2. ``build_transaction_csr``'s stages on an instrumented copy whose output is
   checked against the real function, median of 3;
3. every recorded ``build_bitvecs`` call replayed sequentially and, at T4, on a
   4-worker pool (median of 3), and split sequentially into gather, sort and OR.

Appends one JSON row per (workload, threads). Not used by any decision rule.

Usage:
    uv run python bench/cpu/input_stages.py rows.jsonl [workload ...]
"""

import os
import sys

if len(sys.argv) > 1 and sys.argv[1] == "--one":
    for _k in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_k] = sys.argv[3]

import json  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
WORKLOADS = ["smoke", "deepk", "skew", "wide", "or005", "or0001k2"]
REPS = 3


def _median_time(fn) -> float:
    out = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return round(statistics.median(out), 4)


def _csr_stages(lf, min_support: float) -> dict:
    """build_transaction_csr with a timer per stage; returns seconds per stage and the CSR."""
    import numpy as np
    import polars as pl

    from et_miner.core import cpu_miner as cm
    from et_miner.core.result import _min_count

    t: dict[str, float] = {}

    def lap(name: str, t0: float) -> float:
        now = time.perf_counter()
        t[name] = t.get(name, 0.0) + now - t0
        return now

    t0 = time.perf_counter()
    column = lf.select(pl.col("items")).collect(engine="in-memory").get_column("items")
    t0 = lap("collect", t0)
    n_rows = len(column)
    min_count = _min_count(min_support, n_rows)
    lens = column.list.len().fill_null(1).to_numpy().astype(np.int64)
    t0 = lap("lens", t0)
    parts = []
    for r0, r1 in cm._chunk_bounds(lens, cm.CSR_CHUNK_NNZ):
        flat = column.slice(r0, r1 - r0).explode().drop_nulls()
        t0 = lap("count_explode", t0)
        parts.append(flat.to_frame("item").group_by("item").agg(pl.len().cast(pl.Int64).alias("count")))
        t0 = lap("count_group_by", t0)
    counted = pl.concat(parts).group_by("item").agg(pl.col("count").sum())
    t0 = lap("count_merge", t0)
    freq = counted.filter(pl.col("count") >= min_count).sort("item")
    t0 = lap("filter_sort", t0)
    items = freq.get_column("item")
    bound = int(freq.get_column("count").sum())

    # _map_rows, step by step
    lens = column.list.len().fill_null(1).to_numpy().astype(np.int64)
    t0 = lap("map_lens", t0)
    col_ids = pl.Series(np.arange(len(items), dtype=np.int32))
    indptr = np.zeros(n_rows + 1, dtype=np.int32 if bound < cm.INDPTR32_LIMIT else np.int64)
    out = np.empty(bound, dtype=np.int32)
    filled = 0
    t0 = lap("map_alloc", t0)
    for r0, r1 in cm._chunk_bounds(lens, cm.CSR_CHUNK_NNZ):
        e = column.slice(r0, r1 - r0).explode()
        t0 = lap("map_explode", t0)
        e = e.replace_strict(items, col_ids, default=None, return_dtype=pl.Int32)
        t0 = lap("map_replace_strict", t0)
        c = e.fill_null(-1).to_numpy()
        t0 = lap("map_to_numpy", t0)
        r = np.repeat(np.arange(r1 - r0, dtype=np.int32), lens[r0:r1])
        valid = c >= 0
        r, c = r[valid], c[valid]
        t0 = lap("map_rows_filter", t0)
        if len(c) > 1 and np.any((r[1:] == r[:-1]) & (c[1:] <= c[:-1])):
            order = np.lexsort((c, r))
            r, c = r[order], c[order]
            keep = np.r_[True, (r[1:] != r[:-1]) | (c[1:] != c[:-1])]
            r, c = r[keep], c[keep]
        t0 = lap("map_order_check", t0)
        indptr[r0 + 1 : r1 + 1] = np.bincount(r, minlength=r1 - r0)
        out[filled : filled + len(c)] = c
        filled += len(c)
        t0 = lap("map_write", t0)
    np.cumsum(indptr, out=indptr)
    out.resize(filled, refcheck=False)
    t0 = lap("map_finish", t0)
    counts = np.bincount(out, minlength=len(items)).astype(np.int64)
    keep = counts >= min_count
    t0 = lap("recount", t0)
    return {k: round(v, 4) for k, v in t.items()}, (indptr, out, keep)


def _bitvec_parts(indptr, indices, n_cols: int, rows) -> dict:
    """build_bitvecs' fill split into gather, sort and OR, summed over its ranges (sequential)."""
    import numpy as np

    from et_miner.core import cpu_miner as cm

    lens_all = np.diff(indptr).astype(np.int64)
    sel_lens = lens_all if rows is None else lens_all[rows]
    n = len(sel_lens)
    w = (n + 63) // 64
    out = np.zeros(n_cols * w, dtype=np.uint64)
    t = {"gather": 0.0, "sort": 0.0, "or": 0.0}
    for r0, r1 in cm._aligned_bounds(sel_lens, cm.BITVEC_CHUNK):
        lens = sel_lens[r0:r1]
        if lens.sum() == 0:
            continue
        t0 = time.perf_counter()
        old = np.arange(r0, r1) if rows is None else rows[r0:r1]
        cols = indices[cm._row_entries(indptr, old, lens)].astype(np.int64)
        local = np.repeat(np.arange(r0, r1, dtype=np.int64), lens)
        t1 = time.perf_counter()
        bit = (local & 63).astype(np.uint8)
        order = np.argsort(bit, kind="stable")
        word = (cols * w + (local >> 6))[order]
        cuts = np.searchsorted(bit[order], np.arange(65))
        t2 = time.perf_counter()
        for v in range(64):
            if cuts[v + 1] > cuts[v]:
                out[word[cuts[v] : cuts[v + 1]]] |= np.uint64(1) << np.uint64(v)
        t3 = time.perf_counter()
        t["gather"] += t1 - t0
        t["sort"] += t2 - t1
        t["or"] += t3 - t2
    return {k: round(v, 4) for k, v in t.items()}, out.reshape(n_cols, w)


def one(w: str, n_jobs: int, out_path: str) -> None:
    sys.path.insert(0, str(REPO / "bench"))
    sys.path.insert(0, str(REPO / "bench" / "cpu"))
    from loguru import logger

    logger.remove()
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import polars as pl
    from consolidation_run import _load
    from matrix import WORKLOADS as TABLE
    from son_stakes import _rev

    from et_miner.core import cpu_miner as cm

    ds, ms, ml = TABLE[w]
    df, n, _ = _load(ds)
    warm = pl.read_parquet(REPO / "datasets" / "synth" / "smoke.parquet").head(20_000)
    cm.mine_cpu(warm.lazy(), 0.02, 3, n_jobs=n_jobs)

    calls = []
    real_build = cm.build_bitvecs

    def recorded(indptr, indices, n_cols, rows=None, pool=None):
        t0 = time.perf_counter()
        bv = real_build(indptr, indices, n_cols, rows, pool)
        calls.append({"rows": rows, "n_cols": n_cols, "pool": pool is not None, "s": time.perf_counter() - t0,
                      "indptr": indptr, "indices": indices})
        return bv

    cm.build_bitvecs = recorded
    t0 = time.perf_counter()
    _, sess = cm.mine_cpu(df.lazy(), ms, ml, n_jobs=n_jobs, profile=True)
    wall = time.perf_counter() - t0
    cm.build_bitvecs = real_build
    phases = {p.name: round(p.duration_ms / 1000, 4) for p in sess.phases}

    real = cm.build_transaction_csr(df.lazy(), ms)
    stage_runs = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        stages, (indptr, indices, keep) = _csr_stages(df.lazy(), ms)
        stage_runs.append((time.perf_counter() - t0, stages))
        assert keep.all(), "instrumented copy does not cover the column-removal path"
        assert np.array_equal(indptr, real.indptr) and np.array_equal(indices, real.indices)
    stage_runs.sort(key=lambda x: x[0])
    csr_total, csr_stages = stage_runs[len(stage_runs) // 2]
    csr_real = _median_time(lambda: cm.build_transaction_csr(df.lazy(), ms))
    max_len_s = _median_time(lambda: df.lazy().select(pl.col("items").list.len().max()).collect(engine="streaming").item())

    pool = ThreadPoolExecutor(n_jobs) if n_jobs > 1 else None
    bitvecs = []
    for c in calls:
        rows = c["rows"]
        ref = real_build(c["indptr"], c["indices"], c["n_cols"], rows)
        parts, bv = _bitvec_parts(c["indptr"], c["indices"], c["n_cols"], rows)
        assert np.array_equal(bv, ref)
        n_sel = (len(c["indptr"]) - 1) if rows is None else len(rows)
        lens = np.diff(c["indptr"]) if rows is None else np.diff(c["indptr"])[rows]
        entry = {
            "rows": n_sel, "words": (n_sel + 63) // 64, "n_cols": c["n_cols"], "entries": int(lens.sum()),
            "in_run_s": round(c["s"], 4), "in_run_pool": c["pool"],
            "seq_s": _median_time(lambda: real_build(c["indptr"], c["indices"], c["n_cols"], rows)),
            "parts_s": parts,
        }
        if pool is not None:
            entry["pool_s"] = _median_time(lambda: real_build(c["indptr"], c["indices"], c["n_cols"], rows, pool))
        bitvecs.append(entry)
    if pool is not None:
        pool.shutdown()

    row = {
        "workload": w, "threads": f"T{n_jobs}", "n_rows": n, "entries": int(len(real.indices)),
        "n_cols": real.n_cols, "wall_s": round(wall, 4), "phases_s": phases,
        "csr_real_s": csr_real, "csr_instrumented_s": round(csr_total, 4), "csr_stages_s": csr_stages,
        "max_len_s": max_len_s, "bitvec_in_run_s": round(sum(c["s"] for c in calls), 4), "bitvecs": bitvecs,
        "rev": _rev(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(json.dumps(row), flush=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        for w in sys.argv[2:] or WORKLOADS:
            for t in (1, 4):
                subprocess.run([sys.executable, __file__, "--one", w, str(t), sys.argv[1]], check=True)
