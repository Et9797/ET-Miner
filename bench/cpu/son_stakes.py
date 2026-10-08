"""Offline stakes for SON (streaming=True) on the CPU: phase times of the current SON and of an array-miner port.

Runs one arm on one workload in this process and appends one JSON row: wall
time, per-phase seconds, the result signature, pass-1 candidates per K and the
process high-water RSS. Every arm of a workload must carry one signature; the
driver (``--check``) compares them.

Arms:
    current  ``son.apriori_streaming`` before the port (trees up to
             508cdb0): per chunk a Polars boolean matrix
             (``build_boolean_matrix``), ``_generate_candidates`` and
             ``count_support_batched``; pass 2 rebuilds a boolean matrix of the
             candidate items per chunk (``_build_matrix_for_items``) and counts
             every candidate with ``count_support_batched``. Timed by wrapping
             those functions in place; on a later tree it stops with an error,
             and S1 reads it from S0's rows.
    array    the port as S0 measured it (kept as it ran, so S0 can be
             repeated; the product's versions live in son.py and
             cpu_miner.py): pass 1 builds each chunk's CSR
             (``build_transaction_csr`` at the local threshold) and mines it
             with ``cpu_miner._mine_levels``; the local levels are mapped to one
             item order and deduplicated per K; pass 2 maps each chunk onto the
             candidate items (``_map_rows``), counts K=1 with a bincount, K=2
             on bitvectors or from the Gram (the in-core dispatch rule), and
             K>=3 with ``count_candidates`` on a row space of the rows holding
             at least k items.
    array-pc ``array`` with pass 2's K>=3 counted per candidate (AND of the
             k column bitvectors, popcount) instead of per prefix group: the
             union of local lattices splits into many small groups, where the
             per-group cost of ``count_candidates`` weighs most.
    incore   ``apriori()`` without streaming (the CPU route), for reference.
    built    ``apriori(streaming=True)`` on the tree as it is, untimed inside:
             the current SON before the port, the port after it.

Phases (seconds): count_transactions; pass 1 p1_matrix, p1_mine (p1_gen and
p1_count inside it) for ``current``, p1_csr and p1_mine for the array arms;
union; pass 2 p2_matrix and p2_count for ``current``, p2_csr, p2_k1,
p2_k2_bitvec or p2_k2_gram, p2_space (row spaces and bitvectors) and
p2_count_k<k> per K for the array arms; filter_emit.

Usage:
    uv run python bench/cpu/son_stakes.py --workload or005 --threads 1 --arm current --out rows.jsonl
    uv run python bench/cpu/son_stakes.py --check rows.jsonl
    uv run python bench/cpu/son_stakes.py --matrix --arms built,incore --reps 3 --out rows.jsonl

Options:
    --workload   an id of bench/cpu/matrix.py WORKLOADS, or of INPUT_WORKLOADS
    --threads    thread budget: the env pools and n_jobs
    --arm        current | array | array-pc | incore | built
    --chunks     SON chunks (chunk_size = ceil(N / chunks)); default 4
    --out        jsonl file the row is appended to
    --check      compare the signatures per workload in a jsonl file and exit
    --workloads  comma-separated workload ids for --matrix (default SON_WORKLOADS)
    --matrix     run --workloads x T1/T4 x --arms x --reps, one fresh process
                 per config, rep-major, a config past --cap seconds recorded as
                 a timeout and not repeated, no new config after --max-hours
    --arms       comma-separated arms for --matrix (default built,incore)
    --reps       repetitions for --matrix (default 1)
    --cap        seconds per config for --matrix (default 600)
    --max-hours  no new config after this many hours (default 1.5)
"""

from __future__ import annotations

import os
import sys


def _argv_threads() -> str:
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--threads="):
            return a.split("=", 1)[1]
    return "1"


# Thread pools read their env at import, so it is set before numpy/polars load.
for _k in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = _argv_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(REPO / "bench" / "cpu"))

from child_run import result_signatures  # noqa: E402
from consolidation_run import _load  # noqa: E402
from matrix import WORKLOADS  # noqa: E402

LOCAL_SUPPORT_FACTOR = 0.9
WARMUP_ROWS = 20_000
#: PROTOCOL.md Amendment 3: or003 and or002 are left out (a chunk's local lattice is out of reach for any engine).
SON_WORKLOADS = ("smoke", "deepk", "skew", "wide", "or005", "or0001k2")
#: PROTOCOL.md Amendment 6: the input layer at scale (deep_sparse_large, 20M rows), mined to K=2 as or0001k2 is.
INPUT_WORKLOADS = {"dslk2": ("deep_sparse_large", 0.015, 2)}


class Phases:
    """Seconds per phase name, accumulated."""

    def __init__(self) -> None:
        self.s: dict[str, float] = defaultdict(float)

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.s[name] += time.perf_counter() - t0


def run_current(lf: pl.LazyFrame, min_support: float, max_length, chunk_size: int, n_jobs: int, ph: Phases) -> tuple:
    """son.apriori_streaming with its building blocks timed in place (trees before the port only)."""
    import et_miner.core.candidates as cand_mod
    from et_miner.streaming import son

    if not hasattr(son, "_mine_chunk_frequent"):
        raise RuntimeError("arm `current` measures SON before the port (trees up to 508cdb0); S1 reads it from S0")

    state = {"pass1": False}
    real = {
        "build": son.build_boolean_matrix,
        "mine": son._mine_chunk_frequent,
        "items": son._build_matrix_for_items,
        "count": son.count_support_batched,
        "gen": cand_mod._generate_candidates,
    }

    def build(*a, **k):
        with ph("p1_matrix"):
            return real["build"](*a, **k)

    def mine(*a, **k):
        state["pass1"] = True
        try:
            with ph("p1_mine"):
                return real["mine"](*a, **k)
        finally:
            state["pass1"] = False

    def items(*a, **k):
        with ph("p2_matrix"):
            return real["items"](*a, **k)

    def count(*a, **k):
        with ph("p1_count" if state["pass1"] else "p2_count"):
            return real["count"](*a, **k)

    def gen(*a, **k):
        with ph("p1_gen"):
            return real["gen"](*a, **k)

    son.build_boolean_matrix, son._mine_chunk_frequent = build, mine
    son._build_matrix_for_items, son.count_support_batched = items, count
    cand_mod._generate_candidates = gen
    try:
        res, session = son.apriori_streaming(
            lf, min_support=min_support, max_length=max_length, chunk_size=chunk_size,
            show_progress=False, n_jobs=n_jobs, profile=True,
        )
    finally:
        son.build_boolean_matrix, son._mine_chunk_frequent = real["build"], real["mine"]
        son._build_matrix_for_items, son.count_support_batched = real["items"], real["count"]
        cand_mod._generate_candidates = real["gen"]
    prof = {p.name: p.duration_ms / 1000 for p in session.phases}
    ph.s["p1_other"] = prof.get("pass1_local_mining", 0.0) - ph.s["p1_matrix"] - ph.s["p1_mine"]
    ph.s["p2_other"] = prof.get("pass2_global_counting", 0.0) - ph.s["p2_matrix"] - ph.s["p2_count"]
    ph.s["count_transactions"] = prof.get("count_transactions", 0.0)
    extra = {p.name: p.extra for p in session.phases}
    return res, {"n_candidates": extra.get("pass1_local_mining", {}).get("n_candidates")}


def _count_pairs_of(indptr, indices, n_cols, pairs, n_rows, pool, ph: Phases) -> np.ndarray:
    """Counts of the given column pairs over every row: bitvectors or the Gram, by the in-core K=2 rule."""
    from et_miner.core import cpu_miner as cm

    lens = np.diff(indptr).astype(np.int64)
    occurrences = int((lens * (lens - 1) // 2).sum())
    words = (n_rows + 63) // 64
    if n_cols * words * 8 <= cm.BITVEC_BUDGET_BYTES and len(pairs) * words <= cm.GRAM_BITVEC_RATIO * occurrences:
        with ph("p2_k2_bitvec"):
            bv = cm.build_bitvecs(indptr, indices, n_cols, None, pool)
            out = np.empty(len(pairs), dtype=np.int64)
            step = max(1, cm.AND_CHUNK_BYTES // (8 * max(1, words)))
            for c0 in range(0, len(pairs), step):
                c1 = min(c0 + step, len(pairs))
                out[c0:c1] = cm.popcount_rows(bv[pairs[c0:c1, 0]] & bv[pairs[c0:c1, 1]])
            return out
    with ph("p2_k2_gram"):
        space = cm.RowSpace.__new__(cm.RowSpace)
        space.indptr, space.indices, space.n_cols, space.rows = indptr, indices, n_cols, None
        return space.gram_pairs(
            np.arange(n_rows, dtype=np.int64), np.arange(n_cols, dtype=np.int32),
            pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64), cm.GRAM_BUDGET_BYTES,
        )


def _unique_rows(sets: np.ndarray, base: int) -> np.ndarray:
    """Lexsorted distinct rows of an int32 (n, k) array: np.unique on packed int64 keys when they fit."""
    from et_miner.core import cpu_miner as cm

    keys = cm._pack(sets, base)
    if keys is None:
        return np.unique(sets, axis=0)
    keys = np.unique(keys)
    out = np.empty((len(keys), sets.shape[1]), dtype=np.int32)
    for c in range(sets.shape[1] - 1, -1, -1):
        keys, out[:, c] = np.divmod(keys, base)
    return out


def _count_per_candidate(cands: np.ndarray, space, pool, workers: int) -> np.ndarray:
    """Counts of K-candidates on the row space's bitvectors: AND of the k columns, popcount, per candidate."""
    from et_miner.core import cpu_miner as cm

    if space.bitvecs is None:
        return cm.count_candidates(cands, cands.shape[1], space, pool, workers)
    bv = space.bitvecs
    out = np.empty(len(cands), dtype=np.int64)
    step = max(1, cm.AND_CHUNK_BYTES // (8 * max(1, space.words)))
    for c0 in range(0, len(cands), step):
        c = cands[c0 : c0 + step]
        acc = bv[c[:, 0]] & bv[c[:, 1]]
        for j in range(2, c.shape[1]):
            acc &= bv[c[:, j]]
        out[c0 : c0 + len(c)] = cm.popcount_rows(acc)
    return out


def _group_stats(cands: dict[int, np.ndarray]) -> dict:
    """Prefix groups per K of the pass-2 candidates: count and mean candidates per group."""
    from et_miner.core import cpu_miner as cm

    out = {}
    for k, c in cands.items():
        if k >= 3 and len(c):
            starts, ends = cm._group_runs(c, k)
            out[int(k)] = {"groups": len(starts), "mean_size": round(float((ends - starts).mean()), 2)}
    return out


def run_array_pc(lf: pl.LazyFrame, min_support: float, max_length, chunk_size: int, n_jobs: int, ph: Phases) -> tuple:
    """run_array with pass 2's K>=3 counted per candidate instead of per prefix group."""
    return run_array(lf, min_support, max_length, chunk_size, n_jobs, ph, per_candidate=True)


def run_array(
    lf: pl.LazyFrame, min_support: float, max_length, chunk_size: int, n_jobs: int, ph: Phases,
    per_candidate: bool = False,
) -> tuple:
    """The port: pass 1 on each chunk's CSR with the array miner, pass 2 with count_candidates."""
    from et_miner.core import cpu_miner as cm
    from et_miner.core.result import _empty_result, _min_count

    with ph("count_transactions"):
        n_total = lf.select(pl.len()).collect(engine="streaming").item()
    bounds = [(o, min(chunk_size, n_total - o)) for o in range(0, n_total, chunk_size)]
    local_s = min_support * LOCAL_SUPPORT_FACTOR
    workers = cm._workers(n_jobs)
    pool = ThreadPoolExecutor(workers) if workers > 1 else None
    try:
        chunk_items: list[pl.Series] = []
        chunk_levels: list[list[np.ndarray]] = []
        for off, n in bounds:
            with ph("p1_csr"):
                tc = cm.build_transaction_csr(lf.slice(off, n), local_s, "items")
            if tc is None:
                continue
            with ph("p1_mine"):
                ones = np.arange(tc.n_cols, dtype=np.int32)[:, None]
                prev = cm._Level(ones, tc.counts, tc.counts < tc.n_rows)
                emitted = [(ones, tc.counts)]
                eff = min(max_length or math.inf, int(np.diff(tc.indptr).max()), tc.n_cols)
                cm._mine_levels(
                    tc, ones, prev, emitted, max(1, _min_count(local_s, tc.n_rows)), eff,
                    False, False, True, None, None, pool, workers,
                )
            chunk_items.append(tc.items)
            chunk_levels.append([s for s, _ in emitted])
        if not chunk_items:
            return _empty_result(), {"n_candidates": 0}

        with ph("union"):
            items = pl.concat(chunk_items).unique().sort()
            per_k: dict[int, list[np.ndarray]] = defaultdict(list)
            for its, levels in zip(chunk_items, chunk_levels):
                remap = items.search_sorted(its).to_numpy().astype(np.int32)
                for sets in levels:
                    if len(sets):
                        per_k[sets.shape[1]].append(remap[sets])
            cands = {k: _unique_rows(np.concatenate(v), len(items)) for k, v in sorted(per_k.items())}
            del chunk_levels, per_k
        n_cols = len(items)
        totals = {k: np.zeros(len(c), dtype=np.int64) for k, c in cands.items()}

        for off, n in bounds:
            with ph("p2_csr"):
                column = lf.slice(off, n).select(pl.col("items")).collect(engine="in-memory").get_column("items")
                bound = int(column.list.len().fill_null(1).cast(pl.Int64).sum())
                # ids=None: S0's Polars mapping, not the integer path SON's pass 2 ships with.
                indptr, indices = cm._map_rows(column, items, bound, None)
                del column
            with ph("p2_k1"):
                totals[1] += np.bincount(indices, minlength=n_cols)
            if 2 in cands:
                totals[2] += _count_pairs_of(indptr, indices, n_cols, cands[2], n, pool, ph)
            space = None
            for k in sorted(c for c in cands if c >= 3):
                with ph("p2_space"):
                    rows_k = np.flatnonzero(np.diff(indptr) >= k)
                    if space is None or len(rows_k) <= cm.COMPACT_RATIO * space.n_rows:
                        space = cm.RowSpace(indptr, indices, n_cols, rows_k if len(rows_k) < n else None, pool)
                with ph(f"p2_count_k{k}"):
                    if per_candidate:
                        totals[k] += _count_per_candidate(cands[k], space, pool, workers)
                    else:
                        totals[k] += cm.count_candidates(cands[k], k, space, pool, workers)

        with ph("filter_emit"):
            min_count = _min_count(min_support, n_total)
            levels = []
            for k, c in cands.items():
                keep = totals[k] >= min_count
                levels.append((c[keep], totals[k][keep]))
            res = cm._emit(levels, items, n_total)
    finally:
        if pool is not None:
            pool.shutdown()
    return res, {"n_candidates": int(sum(len(c) for c in cands.values())),
                 "candidates_per_k": {int(k): len(c) for k, c in cands.items()},
                 "groups_per_k": _group_stats(cands)}


def run_incore(lf: pl.LazyFrame, min_support: float, max_length, chunk_size: int, n_jobs: int, ph: Phases) -> tuple:
    from et_miner import apriori

    with ph("incore"):
        return apriori(lf, min_support=min_support, max_length=max_length, n_jobs=n_jobs), {}


def run_built(lf: pl.LazyFrame, min_support: float, max_length, chunk_size: int, n_jobs: int, ph: Phases) -> tuple:
    from et_miner import apriori

    with ph("son"):
        res = apriori(
            lf, min_support=min_support, max_length=max_length, streaming=True, chunk_size=chunk_size,
            n_jobs=n_jobs, show_progress=False,
        )
    return res, {}


ARMS = {
    "current": run_current, "array": run_array, "array-pc": run_array_pc, "incore": run_incore, "built": run_built,
}


def check(path: Path) -> int:
    """Every row of a workload must carry one signature; prints a table and returns 1 on a divergence."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    bad = 0
    by_w: dict[str, set] = defaultdict(set)
    for r in rows:
        if r.get("status") == "ok":
            by_w[r["workload"]].add((r["n_itemsets"], r["sum_counts"], r["itemset_hash"]))
    for w, sigs in by_w.items():
        print(f"{w}: {'ok' if len(sigs) == 1 else 'DIVERGES'} ({len(sigs)} signature(s))")
        bad += len(sigs) != 1
    return 1 if bad else 0


def _rev() -> str:
    """HEAD, suffixed +dirty when a tracked file differs from it."""
    import subprocess

    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()

    return git("rev-parse", "--short", "HEAD") + ("+dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def _append(path: str, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def run_matrix(args) -> int:
    """SON_WORKLOADS x T1/T4 x arms x reps, rep-major, each config in a fresh process under the cap."""
    import platform
    import subprocess

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    versions = subprocess.run(
        [sys.executable, "-c", "import polars, numpy, scipy; print(polars.__version__, numpy.__version__, scipy.__version__)"],
        capture_output=True, text=True,
    ).stdout.strip()
    (out.parent / "env.txt").write_text(
        f"rev {_rev()}\npython {platform.python_version()}\npolars numpy scipy {versions}\n\n{lscpu}"
    )
    t_end = time.time() + args.max_hours * 3600
    timed_out: set[tuple] = set()
    for rep in range(args.reps):
        for w in args.workloads.split(","):
            for t in (1, 4):
                for arm in args.arms.split(","):
                    base = {"workload": w, "arm": arm, "threads": t, "chunks": args.chunks, "rep": rep}
                    if (w, t, arm) in timed_out:
                        _append(args.out, {**base, "status": "skipped: timed out in an earlier rep"})
                        continue
                    if time.time() > t_end:
                        _append(args.out, {**base, "status": f"skipped: past --max-hours {args.max_hours}"})
                        continue
                    cmd = [sys.executable, __file__, "--workload", w, "--threads", str(t), "--arm", arm,
                           "--chunks", str(args.chunks), "--rep", str(rep), "--out", args.out]
                    print(f"rep {rep} {w} T{t} {arm}", flush=True)
                    try:
                        p = subprocess.run(cmd, timeout=args.cap, capture_output=True, text=True)
                    except subprocess.TimeoutExpired:
                        timed_out.add((w, t, arm))
                        _append(args.out, {**base, "status": "timeout", "cap_s": args.cap, "rev": _rev()})
                        continue
                    if p.returncode != 0:
                        _append(args.out, {**base, "status": f"error: exit {p.returncode}: {p.stderr[-500:]}",
                                           "rev": _rev()})
    return check(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--check")
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--arms", default="built,incore")
    ap.add_argument("--workloads", default=",".join(SON_WORKLOADS))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--cap", type=float, default=600)
    ap.add_argument("--max-hours", type=float, default=1.5)
    args = ap.parse_args()
    if args.check:
        return check(Path(args.check))
    if args.matrix:
        return run_matrix(args)

    from loguru import logger

    logger.remove()
    dataset, min_support, max_length = {**WORKLOADS, **INPUT_WORKLOADS}[args.workload]
    df, n_rows, _ = _load(dataset)
    chunk_size = math.ceil(n_rows / args.chunks)
    fn = ARMS[args.arm]

    warm = pl.read_parquet(REPO / "datasets" / "synth" / "smoke.parquet").head(WARMUP_ROWS)
    fn(warm.lazy(), 0.02, 3, WARMUP_ROWS // 2, args.threads, Phases())

    ph = Phases()
    t0 = time.perf_counter()
    c0 = time.process_time()
    res, extra = fn(df.lazy(), min_support, max_length, chunk_size, args.threads, ph)
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    row = {
        "workload": args.workload, "arm": args.arm, "threads": args.threads, "chunks": args.chunks,
        "rep": args.rep, "chunk_size": chunk_size, "status": "ok", "wall_s": round(wall, 4),
        "cpu_s": round(cpu, 4), "phases_s": {k: round(v, 4) for k, v in sorted(ph.s.items())},
        "ru_maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        **extra, **result_signatures(res, n_rows),
        "rev": _rev(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(json.dumps(row), flush=True)
    if args.out:
        _append(args.out, row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
