"""M2's K=3 crossover sweep (bench/moonshots/PROTOCOL-M2.md, "Crossover sweep"), and the dispatch
statistic's sweep (``--grid dispatch``, bench/moonshots/PROTOCOL-K3-DISPATCH.md).

Synthetic problems on one GPU: N rows of L distinct items each, drawn uniformly
from F items, mined with ``max_length=3`` at ``min_count`` = a third of the
mean pair count, so (nearly) every pair is frequent and the K=3 level is the
whole triple space: ``r3 = Σ_rows C(L_active, 2) / (generated × ceil(N / 64))``
with ``generated = C(F, 3)``. One process per point, pinned to device 0 with
NCCL off and the thread pools at 16: an untimed warm-up run of each kernel,
then 3 interleaved reps (dense, rows, dense, rows, ...). The K=3 level time
(``level_callback``) of each run is recorded; the point's value is the
median. Every run must mine the same itemsets with the same counts, the miner
must log the row-wise K=3 exactly on the ``rows`` runs, and no run may log a
fallback (``consolidation_run.FALLBACK_PATTERNS``, NCCL being off aside).

``r3*`` follows k2_crossover's rule: the geometric mean of the r3 values of the
adjacent pair (sorted by r3) at the lowest crossing from rows-faster below to
rows-slower above; ``monotone`` is false when the points cross more than once
or the other way.

Rows append to <out> (default: ``k3_crossover.jsonl`` in the runner's
per-revision campaign directory, stamped with the revision); a point with an ok
row at this revision is skipped, so the sweep resumes.

``--grid dispatch`` runs PROTOCOL-K3-DISPATCH.md's 43 points instead: M2's 18,
long rows, short rows over 4M rows, uniform rows with ``min_count`` at the mean
pair count (rho about 1/2) and Zipf rows. Its points differ from M2's procedure
in three ways:
- ``min_count`` is ``c`` times the mean pair count (``c`` = 1/3 unless the id
  ends in ``-c<c>``);
- the warm-up runs each kernel on the point's first ``WARMUP_ROWS`` rows;
- a point whose rho is below 1 adds an untimed rows run that sums the kernel's
  slot counts: the increments, which equal the row triples when rho is 1.

Each row records the miner's K=3 statistics (generated candidates, slots, row
pairs and triples, rho, r3 and s). ``--analyze`` then prints each statistic's
regret: the K=3 milliseconds lost against the faster kernel when it picks rows
below a threshold (``regret``).

Usage: uv run python bench/moonshots/k3_crossover.py [--grid m2|dispatch] [--out FILE] [--max-gpu-hours H] [--analyze]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from k2_crossover import generate_rows  # noqa: E402

N_ROWS = 1_000_000
ITEMS = (50, 100, 200, 400, 800, 1_600)
ROW_LENS = (8, 16, 32)
REPS = 3
KERNELS = ("dense", "rows")
TIMEOUT_S = 900
ENV = {
    "CUDA_VISIBLE_DEVICES": "0",
    "ET_MINER_DISABLE_NCCL": "1",
    "POLARS_MAX_THREADS": "16",
    "RAYON_NUM_THREADS": "16",
    "MKL_NUM_THREADS": "16",
    "OMP_NUM_THREADS": "16",
}
ROWS_LOG = "row-wise over"
WARMUP_ROWS = 20_000
#: The miner's row-wise K=3 log line (``row_split._count_k3_rows``).
STATS_RE = re.compile(
    r"K=3: ([\d,]+) total candidates, row-wise over ([\d,]+) row pairs and ([\d.e+]+) row triples "
    r"\(r3 = ([\d.e+-]+|inf), s = ([\d.e+-]+|inf), rho = ([\d.e+-]+)\) into ([\d,]+) slots"
)


def grid() -> list[dict]:
    return [
        {"id": f"uniform-N1M-F{f}-L{n}", "n_rows": N_ROWS, "n_items": f, "row_len": n}
        for n in ROW_LENS
        for f in ITEMS
        if n <= f
    ]


def _point(dist: str, n_rows: int, f: int, length: int, c: float | None = None) -> dict:
    pid = f"{dist}-N{n_rows // 1_000_000}M-F{f}-L{length}" + ("" if c is None else f"-c{c:g}")
    return {"id": pid, "dist": dist, "n_rows": n_rows, "n_items": f, "row_len": length, "c": c,
            "warmup_rows": WARMUP_ROWS}


def grid_dispatch() -> list[dict]:
    """PROTOCOL-K3-DISPATCH.md's points: M2's, then long rows, short rows, rho about 1/2, Zipf."""
    pts = [{**p, "dist": "uniform", "c": None, "warmup_rows": WARMUP_ROWS} for p in grid()]
    pts += [_point("uniform", N_ROWS, f, n) for f, n in ((800, 24), (1_200, 32), (1_200, 40), (1_200, 48),
                                                         (1_600, 40), (1_600, 48), (1_600, 56), (1_600, 64))]
    pts += [_point("uniform", 4 * N_ROWS, f, n) for f, n in ((200, 8), (250, 8), (300, 8), (400, 8),
                                                             (450, 12), (600, 12))]
    pts += [_point("uniform", N_ROWS, f, n, 1.0) for f, n in ((800, 32), (1_600, 32), (1_600, 48), (1_600, 64))]
    pts += [_point("zipf", N_ROWS, f, n) for f, n in ((800, 16), (800, 32), (1_600, 16), (1_600, 24),
                                                      (1_600, 32), (1_600, 48), (2_400, 32))]
    return pts


GRIDS = {"m2": grid, "dispatch": grid_dispatch}


def regret(rows: list[dict], key: str, at: float | None = None) -> dict:
    """The K=3 milliseconds lost against the faster kernel when rows run below a threshold of ``key``.

    With ``at`` the threshold is given; otherwise it is the one of least regret
    among the geometric means of adjacent points sorted by ``key`` (plus none
    and all), the lowest on a tie.
    """
    pts = sorted((r for r in rows if r.get("status") == "ok"), key=lambda r: r[key])
    best = sum(min(r["median_dense_ms"], r["median_rows_ms"]) for r in pts)

    def lost(theta: float) -> float:
        return sum(r["median_rows_ms"] if r[key] < theta else r["median_dense_ms"] for r in pts) - best

    if at is not None:
        return {"key": key, "threshold": at, "regret_ms": lost(at)}
    cuts = [0.0] + [math.sqrt(a[key] * b[key]) for a, b in zip(pts, pts[1:]) if a[key] < b[key]] + [math.inf]
    scored = [(lost(t), t) for t in cuts]
    least = min(scored)
    near = [t for v, t in scored if v <= least[0] * 1.05 + 1e-9]
    return {"key": key, "threshold": least[1], "regret_ms": least[0], "within_5pct": [min(near), max(near)]}


def crossover(rows: list[dict]) -> dict:
    """``r3*`` from the ok points (see the module docstring)."""
    pts = sorted((r for r in rows if r.get("status") == "ok"), key=lambda r: r["r3"])
    up, down = [], []
    for a, b in zip(pts, pts[1:]):
        if a["ratio"] < 1 <= b["ratio"]:
            up.append((a, b))
        elif b["ratio"] < 1 <= a["ratio"]:
            down.append((a, b))
    r_star = min((math.sqrt(a["r3"] * b["r3"]) for a, b in up), default=None)
    return {
        "r3_star": r_star,
        "up": [(a["id"], b["id"]) for a, b in up],
        "down": [(a["id"], b["id"]) for a, b in down],
        "monotone": len(up) == 1 and not down,
    }


def _child(pt: dict) -> dict:
    import polars as pl
    import pyarrow as pa
    from child_run import result_signatures
    from consolidation_run import _FALLBACK_RE
    from loguru import logger

    from et_miner.core.apriori import apriori
    from et_miner.core.result import _min_count

    from et_miner.gpu import kernels

    t0 = time.perf_counter()
    n_rows, f, length = pt["n_rows"], pt["n_items"], pt["row_len"]
    rows = generate_rows(pt.get("dist", "uniform"), n_rows, f, length, zlib.crc32(pt["id"].encode()))
    offsets = np.arange(0, rows.size + 1, length, dtype=np.int64)
    df = pl.DataFrame(
        {"items": pl.Series("items", pa.LargeListArray.from_arrays(offsets, rows.ravel().astype(np.int64)))}
    )
    del rows
    mean_pair = n_rows * math.comb(length, 2) / math.comb(f, 2)
    c = pt.get("c")
    min_count = max(1, int(mean_pair / 3 if c is None else mean_pair * c))
    min_support = (min_count - 0.5) / n_rows
    if _min_count(min_support, n_rows) != min_count:
        raise RuntimeError(f"min_support {min_support!r} does not give min_count {min_count}")
    gen_s = time.perf_counter() - t0

    logged: list[str] = []
    logger.add(lambda m: logged.append(m.record["message"]), level="DEBUG")

    def run(kernel: str, frame=df) -> tuple[float, tuple, str]:
        os.environ["ET_MINER_K3_KERNEL"] = kernel
        logged.clear()
        levels: dict[int, float] = {}
        res = apriori(
            frame,
            use_gpu=True,
            n_gpus=1,
            min_support=min_support,
            max_length=3,
            level_callback=lambda k, n_cand, n_freq, ms: levels.__setitem__(k, ms),
        )
        fallbacks = [m for m in logged if _FALLBACK_RE.search(m) and "NCCL unavailable" not in m]
        if fallbacks:
            raise RuntimeError(f"K3 kernel {kernel!r}: fallback logged: {fallbacks[0][:300]}")
        k3_log = [m for m in logged if m.startswith("  K=3:") and ROWS_LOG in m]
        if (kernel == "rows") != bool(k3_log):
            raise RuntimeError(f"K3 kernel {kernel!r} pinned, but the miner's K=3 log says otherwise")
        return levels.get(3, 0.0), tuple(result_signatures(res, frame.height).values()), (k3_log or [""])[0]

    warmup = df if pt.get("warmup_rows") is None else df.head(pt["warmup_rows"])
    for k in KERNELS:
        run(k, warmup)
    times: dict[str, list[float]] = {k: [] for k in KERNELS}
    sigs, rows_log = set(), ""
    for _ in range(REPS):
        for k in KERNELS:
            ms, sig, log = run(k)
            times[k].append(round(ms, 3))
            sigs.add(sig)
            rows_log = log or rows_log
    if len(sigs) != 1:
        raise RuntimeError(f"the kernels disagree on the result: {sorted(sigs)}")
    n_itemsets, sum_counts, itemset_hash = sigs.pop()
    med = {k: statistics.median(v) for k, v in times.items()}
    stats = {}
    if pt.get("warmup_rows") is None:
        # M2's rows: the miner's own r3 (row pairs over the active items), read from its log line.
        stats["r3"] = float(rows_log.split("(r3 = ", 1)[1].split(",", 1)[0].split(")", 1)[0])
    else:
        m = STATS_RE.search(rows_log)
        if m is None:
            raise RuntimeError(f"no K=3 statistics in the rows log: {rows_log[:300]}")
        gen, r2, t3, r3, s, rho, slots = m.groups()
        stats = {"generated": int(gen.replace(",", "")), "row_pairs": int(r2.replace(",", "")),
                 "row_triples": float(t3), "r3": float(r3), "s": float(s), "rho": float(rho),
                 "slots": int(slots.replace(",", ""))}
        words = -(-n_rows // 64)
        stats["s_slots"] = stats["s"] * stats["generated"] / stats["slots"] if stats["slots"] else math.inf
        if stats["rho"] >= 0.99995:
            stats["increments"] = stats["row_triples"]
        else:
            real, total = kernels.count_k3_rows, [0]

            def spy(*a, **k):
                out = real(*a, **k)
                total[0] += int(out.sum(dtype="int64"))
                return out

            kernels.count_k3_rows = spy
            try:
                run("rows")
            finally:
                kernels.count_k3_rows = real
            stats["increments"] = total[0]
        stats["s_true"] = (stats["row_pairs"] + stats["increments"]) / (stats["generated"] * words)
    return {
        **pt,
        "status": "ok",
        **stats,
        "min_count": min_count,
        "n_itemsets": n_itemsets,
        "sum_counts": sum_counts,
        "itemset_hash": itemset_hash,
        "k3_ms": times,
        "median_dense_ms": med["dense"],
        "median_rows_ms": med["rows"],
        "ratio": med["rows"] / med["dense"] if med["dense"] else math.inf,
        "rows_log": rows_log,
        "gen_s": round(gen_s, 1),
    }


def _run_point(pt: dict, rev: str) -> dict:
    print(f"→ {pt['id']}", flush=True)
    t = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, __file__, "--child", json.dumps(pt)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            env={**os.environ, **ENV},
        )
        last = proc.stdout.strip().splitlines()[-1:] or [""]
        row = (
            json.loads(last[0])
            if proc.returncode == 0 and last[0].startswith("{")
            else {**pt, "status": f"error (rc={proc.returncode}): {proc.stderr.strip().splitlines()[-1:]}"}
        )
    except subprocess.TimeoutExpired:
        row = {**pt, "status": "timeout"}
    return {**row, "rev": rev, "proc_s": round(time.time() - t, 1)}


def _report(rows: list[dict]) -> None:
    key = "s" if any("s" in r for r in rows) else "r3"
    for r in sorted(rows, key=lambda r: r.get(key, 0)):
        if r.get("status") == "ok":
            extra = "" if key == "r3" else f" s={r['s']:<10.3g} rho={r['rho']:<6.3g}"
            print(
                f"  {r['id']:<28} r3={r['r3']:<10.3g}{extra} dense={r['median_dense_ms']:>10.1f} ms "
                f"rows={r['median_rows_ms']:>10.1f} ms  ratio={r['ratio']:.3f}"
            )
        else:
            print(f"  {r['id']:<28} {r.get('status')}")
    print(json.dumps(crossover(rows), indent=1))
    if key == "s":
        from et_miner.gpu.row_split_chunks import K3_ROWS_MAX_R

        ok = [r for r in rows if r.get("status") == "ok"]
        table = [regret(ok, "r3", K3_ROWS_MAX_R), regret(ok, "r3"), regret(ok, "s"), regret(ok, "s_slots"),
                 regret(ok, "s_true")]
        print(json.dumps({"points": len(ok), "regret": table}, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=sorted(GRIDS), default="m2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-gpu-hours", type=float, default=None)
    ap.add_argument("--analyze", action="store_true", help="only print the analysis of the rows in --out")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        print(json.dumps(_child(json.loads(args.child))))
        return 0

    from runner import _campaign_out, _git_rev

    here = _git_rev()
    if here == "unknown":
        print("cannot determine the revision; fix git and re-run")
        return 2
    out = Path(args.out) if args.out else _campaign_out(here) / "k3_crossover.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    if not args.analyze:
        gpu_s = 0.0
        for pt in GRIDS[args.grid]():
            if any(r["id"] == pt["id"] and r.get("status") == "ok" and r.get("rev") == here for r in rows):
                continue
            if args.max_gpu_hours is not None and gpu_s > args.max_gpu_hours * 3600:
                print(f"max-gpu-hours reached ({gpu_s / 3600:.2f}) — stopping (resume with the same command)")
                return 1
            row = _run_point(pt, here)
            gpu_s += row["proc_s"]
            rows.append(row)
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"   {row['status']} ratio={row.get('ratio')}", flush=True)
        print(f"GPU-hours this invocation: {gpu_s / 3600:.3f}")
    _report([r for r in rows if r.get("rev") == here])
    return 0


if __name__ == "__main__":
    sys.exit(main())
