"""C4: the public counting API's three CPU engines on the same batches.

Two steps, each in a fresh process with the thread pools pinned before polars is imported:

    dump   save batches of itemsets with the boolean matrix they are counted over to <dir>/<name>/:
           the in-core levels of a workload (the CPU route's candidates of each level K, generated from the
           mined level K-1 by cpu_miner.generate_candidates, and every pair of frequent items at K=2) over
           the workload's boolean matrix at its min_support, and SON's pass-2 candidate list (every length)
           over the first chunk's matrix of the candidate items. A batch holds at most --cap itemsets, an
           even stride through the level.
    time   per batch and arm, in a fresh process each (an arm the kernel kills, out of memory, is recorded and the
           run goes on), the wall time of one call (median of --reps), the counts checked equal across the arms:
             polars   count_support_batched(sparse=False, batch_size=10_000)  (count_support_vectorized)
             sparse   count_support_batched(sparse=True, n_jobs=THREADS)      (count_support_sparse)
             arrays   the same length filter, then the matrix's CSR counted by cpu_miner.count_itemsets
                      (a thread pool of THREADS)

Usage:
    POLARS_MAX_THREADS=17 ... python bench/routes/count_api.py dump WORKLOAD DIR [--son-chunk ROWS] [--cap N]
    POLARS_MAX_THREADS=17 ... python bench/routes/count_api.py time DIR [--reps 3] [--timeout S] [--threads 17]

`time` prints one JSON line per (batch, arm); an arm still running after --timeout seconds is recorded as a
lower bound.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "bench"))

WORKLOADS = {
    "or002": ("online_retail", 0.002),
    "or005": ("online_retail", 0.005),
    "wide": ("wide_vocab", 0.004),
    "dsl": ("deep_sparse_large", 0.015),
    "deepk": ("deep_k", 0.02),
}


def _load(dataset: str):
    import polars as pl

    if dataset == "online_retail":
        return pl.read_parquet(REPO / "datasets" / "online_retail_ii" / "transactions.parquet").select(
            pl.col("items").cast(pl.List(pl.Int64)))
    return pl.read_parquet(REPO / "datasets" / "synth" / f"{dataset}.parquet")


def _stride(rows: list, cap: int) -> list:
    if len(rows) <= cap:
        return rows
    step = len(rows) / cap
    return [rows[int(i * step)] for i in range(cap)]


def _save(out: Path, name: str, matrix, itemsets: list[tuple[str, ...]], n_rows: int, meta: dict) -> None:
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    matrix.write_parquet(d / "matrix.parquet")
    (d / "batch.json").write_text(json.dumps({"n_rows": n_rows, "itemsets": itemsets, **meta}))


def dump(workload: str, out: Path, son_chunk: int | None, cap: int) -> None:
    import polars as pl

    from et_miner import apriori
    from et_miner.core import cpu_miner
    from et_miner.core.matrix import build_boolean_matrix

    dataset, support = WORKLOADS[workload]
    df = _load(dataset)
    if son_chunk:
        from et_miner.streaming import son

        captured = {}
        real = son._count_chunk

        def spy(chunk_lf, item_col, items, cands, pool, workers):
            if not captured:
                captured.update(items=items, cands=cands, chunk=chunk_lf.collect())
            return real(chunk_lf, item_col, items, cands, pool, workers)

        son._count_chunk = spy
        apriori(df, min_support=support, streaming=True, chunk_size=son_chunk)
        if not captured:
            print(json.dumps({"dumped": [], "why": "pass 2 counted nothing (the partition bound settled every candidate)"}))
            return
        items, cands, chunk = captured["items"], captured["cands"], captured["chunk"]
        values = items.to_list()
        width = len(str(len(values)))
        names = [f"i_{i:0{width}d}" for i in range(len(values))]
        # A boolean column per candidate item (SON's former _build_matrix_for_items, gone with its boolean GPU passes).
        matrix = chunk.select([pl.col("items").list.contains(v).alias(n) for v, n in zip(values, names)])
        rows = [tuple(names[c] for c in row) for k in sorted(cands) for row in cands[k].tolist()]
        _save(out, f"{workload}-son-pass2", matrix, _stride(rows, cap), chunk.height,
              {"source": "son pass 2, chunk 0", "n_total": len(rows)})
        return
    res = apriori(df, min_support=support)
    matrix, col_to_item, n_rows = build_boolean_matrix(df.lazy(), support, "items")
    to_col = {v: k for k, v in col_to_item.items()}
    cols = sorted(to_col, key=lambda c: c)
    col_of = {item: i for i, item in enumerate(cols)}
    names = [to_col[i] for i in cols]
    levels = {}
    for s in res["itemset"].to_list():
        levels.setdefault(len(s), []).append(sorted(col_of[i] for i in s))
    for k in range(2, max(levels) + 2):
        if k == 2:
            n = len(names)
            rows = [(names[a], names[b]) for a in range(n) for b in range(a + 1, n)]
        else:
            prev = levels.get(k - 1)
            if not prev:
                break
            prev = np.array(sorted(prev), dtype=np.int32)
            cands = list(cpu_miner.generate_candidates(prev, k, len(names)))
            if not cands:
                break
            rows = [tuple(names[c] for c in row) for row in np.concatenate(cands).tolist()]
        _save(out, f"{workload}-k{k}", matrix, _stride(rows, cap), n_rows,
              {"source": f"in-core level K={k}", "n_total": len(rows)})
    del pl


def _arrays(matrix, itemsets, n_rows: int, threads: int) -> dict:
    """count_support_batched's length filter, then the CSR counted by cpu_miner.count_itemsets."""
    from concurrent.futures import ThreadPoolExecutor

    from et_miner.core.cpu_miner import count_itemsets
    from et_miner.core.matrix import _filter_transactions_by_length, _polars_to_sparse_csr

    k_min = min(len(set(s)) for s in itemsets)
    filtered, _ = _filter_transactions_by_length(matrix, k_min)
    csr, col_idx = _polars_to_sparse_csr(filtered)
    by_k: dict[int, list] = {}
    for s in itemsets:
        by_k.setdefault(len(s), []).append(sorted(col_idx[c] for c in s))
    cands = {}
    for k, rows in by_k.items():
        arr = np.array(rows, dtype=np.int32).reshape(-1, k)
        order = np.lexsort(arr.T[::-1])
        cands[k] = (arr[order], order)
    with ThreadPoolExecutor(threads) as pool:
        counts = count_itemsets(csr.indptr, csr.indices, csr.shape[1], {k: c for k, (c, _) in cands.items()}, pool,
                                threads)
    out = {}
    for k, (_, order) in cands.items():
        got = np.empty(len(order), dtype=np.int64)
        got[order] = counts[k]
        for row, c in zip((s for s in itemsets if len(s) == k), got.tolist()):
            out[row] = c
    return out


def _timed(fn, timeout: float):
    def _expire(signum, frame):
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    t0 = time.perf_counter()
    try:
        return fn(), time.perf_counter() - t0, False
    except TimeoutError:
        return None, timeout, True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


ARMS = ("polars", "sparse", "arrays")


def time_arm(d: Path, arm: str, reps: int, timeout: float, threads: int) -> None:
    """One arm on one batch, in this process: one JSON line with its times and a digest of its counts."""
    import hashlib

    import polars as pl

    from et_miner.core.matrix import count_support_batched

    meta = json.loads((d / "batch.json").read_text())
    matrix = pl.read_parquet(d / "matrix.parquet")
    itemsets = [tuple(s) for s in meta["itemsets"]]
    n = meta["n_rows"]
    fn = {
        "polars": lambda: count_support_batched(matrix, itemsets, n, batch_size=10_000, sparse=False),
        "sparse": lambda: count_support_batched(matrix, itemsets, n, sparse=True, n_jobs=threads),
        "arrays": lambda: _arrays(matrix, itemsets, n, threads),
    }[arm]
    times, cut, got = [], False, None
    for _ in range(reps):
        got, dt, cut = _timed(fn, timeout)
        times.append(dt)
        if cut:
            break
    digest = None
    if got is not None:
        counts = sorted((tuple(k), int(v)) for k, v in got.items())
        digest = hashlib.sha256(json.dumps(counts).encode()).hexdigest()[:16]
    print(json.dumps({
        "batch": d.name, "source": meta["source"], "n_itemsets": len(itemsets), "n_total": meta["n_total"],
        "n_rows": n, "n_cols": matrix.width, "arm": arm, "s": [round(t, 4) for t in times],
        "median_s": round(statistics.median(times), 4), "lower_bound": cut, "counts_digest": digest,
        "threads": threads,
    }), flush=True)


def _oom_kills() -> int | None:
    try:
        for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines():
            if line.startswith("oom_kill "):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def time_batches(root: Path, reps: int, timeout: float, threads: int, only: str | None) -> None:
    import subprocess

    for d in sorted(p for p in root.iterdir() if (p / "batch.json").exists()):
        if only and only not in d.name:
            continue
        reference = None
        for arm in ARMS:
            kills = _oom_kills()
            cmd = [sys.executable, __file__, "arm", str(d), arm, "--reps", str(reps), "--timeout", str(timeout),
                   "--threads", str(threads)]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
            lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
            if proc.returncode != 0 or not lines:
                after = _oom_kills()
                oom = kills is not None and after is not None and after > kills
                print(json.dumps({"batch": d.name, "arm": arm, "status": f"exit {proc.returncode}"
                                  + (" (cgroup oom_kill)" if oom else ""), "lower_bound": True, "threads": threads}),
                      flush=True)
                continue
            row = json.loads(lines[-1])
            if row["counts_digest"] is not None:
                reference = reference or row["counts_digest"]
                row["same_counts"] = row["counts_digest"] == reference
            print(json.dumps({**row, "status": "ok"}), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("workload", choices=sorted(WORKLOADS))
    d.add_argument("dir")
    d.add_argument("--son-chunk", type=int, default=None)
    d.add_argument("--cap", type=int, default=10_000)
    t = sub.add_parser("time")
    t.add_argument("dir")
    t.add_argument("--reps", type=int, default=3)
    t.add_argument("--timeout", type=float, default=300.0)
    t.add_argument("--threads", type=int, default=17)
    t.add_argument("--only", default=None)
    a = sub.add_parser("arm", help="one arm on one batch (the child of `time`)")
    a.add_argument("batch")
    a.add_argument("arm", choices=ARMS)
    a.add_argument("--reps", type=int, default=3)
    a.add_argument("--timeout", type=float, default=300.0)
    a.add_argument("--threads", type=int, default=17)
    args = ap.parse_args()
    if args.cmd == "dump":
        dump(args.workload, Path(args.dir), args.son_chunk, args.cap)
    elif args.cmd == "arm":
        time_arm(Path(args.batch), args.arm, args.reps, args.timeout, args.threads)
    else:
        time_batches(Path(args.dir), args.reps, args.timeout if math.isfinite(args.timeout) else 1e9, args.threads,
                     args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
