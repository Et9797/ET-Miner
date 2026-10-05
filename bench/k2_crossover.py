"""The K=2 crossover sweep of phase A (bench/optimizations/PROTOCOL.md, "K=2 crossover sweep").

Synthetic K=2-only problems on one GPU: N rows of L distinct items each, drawn
uniformly from F items (or with Zipf(1.0) popularity), mined with
`max_length=2` at `min_count` = the rarest item's count, so every item is
frequent and `r = Σ_rows C(L, 2) / (C(F, 2) × ceil(N / 64))`. One process per
point, pinned to device 0 with NCCL off, the other two phase A knobs at
`dense`/`recount` and the thread pools at 6: an untimed warm-up run of each
kernel, then 3 interleaved reps (dense, rows, dense, rows, ...). The K=2 level
time (`level_callback`) of each run is recorded; the point's value is the
median. Every run must mine the same pairs with the same counts, the miner
must log the row-wise K=2 exactly on the `rows` runs, and no run may log a
fallback (`consolidation_run.FALLBACK_PATTERNS`, NCCL being off aside).

The N = 1M grid runs first; `r*` follows the protocol (`crossover`). Then the
two uniform points whose r values bracket the crossover (without one, the two
whose ratio is closest to 1) run again at N = 4M: their ratio must stay on the
same side of 1 as at 1M if r, not N, predicts.

Rows append to <out> (default: `k2_crossover.jsonl` in the runner's
per-revision campaign directory, stamped with the revision); a point with an ok
row at this revision is skipped, so the sweep resumes.

Usage: uv run python bench/k2_crossover.py [--out FILE] [--max-gpu-hours H] [--analyze]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

N_ROWS = 1_000_000
N_ROWS_CHECK = 4_000_000
ITEMS = (100, 300, 1_000, 3_000, 10_000, 30_000)
ROW_LENS = (5, 10, 20, 40)
ZIPF_ROW_LEN = 20
REPS = 3
KERNELS = ("dense", "rows")
TIMEOUT_S = 900
ENV = {
    "CUDA_VISIBLE_DEVICES": "0",
    "ET_MINER_DISABLE_NCCL": "1",
    "ET_MINER_REDUCE": "dense",
    "ET_MINER_ESCO_MATERIALIZE": "recount",
    "POLARS_MAX_THREADS": "6",
    "RAYON_NUM_THREADS": "6",
    "MKL_NUM_THREADS": "6",
    "OMP_NUM_THREADS": "6",
}
ROWS_LOG = "row-wise over"


def point(dist: str, n_rows: int, n_items: int, row_len: int) -> dict:
    return {
        "id": f"{dist}-N{n_rows // 1_000_000}M-F{n_items}-L{row_len}",
        "dist": dist,
        "n_rows": n_rows,
        "n_items": n_items,
        "row_len": row_len,
    }


def grid() -> list[dict]:
    pts = [point("uniform", N_ROWS, f, n) for n in ROW_LENS for f in ITEMS]
    return pts + [point("zipf", N_ROWS, f, ZIPF_ROW_LEN) for f in ITEMS]


def r_of(n_rows: int, n_items: int, row_len: int) -> float:
    return n_rows * math.comb(row_len, 2) / (math.comb(n_items, 2) * -(-n_rows // 64))


def generate_rows(dist: str, n_rows: int, n_items: int, row_len: int, seed: int) -> np.ndarray:
    """(n_rows, row_len) int32, each row `row_len` distinct items in ascending order.

    A drawn item already in its row is drawn again (rejection), so a uniform
    row is a uniform `row_len`-subset; a Zipf row is successive sampling with
    weights 1/rank.
    """
    if row_len > n_items:
        raise ValueError(f"row length {row_len} exceeds {n_items} items")
    rng = np.random.default_rng(seed)
    if dist == "zipf":
        cdf = np.cumsum(1.0 / np.arange(1, n_items + 1))
        cdf /= cdf[-1]

        def draw(size):
            return np.minimum(np.searchsorted(cdf, rng.random(size), side="right"), n_items - 1).astype(np.int32)
    elif dist == "uniform":

        def draw(size):
            return rng.integers(0, n_items, size=size, dtype=np.int32)
    else:
        raise ValueError(f"unknown distribution {dist!r}")
    out = draw((n_rows, row_len))
    out.sort(axis=1)
    todo = np.arange(n_rows)
    while len(todo):
        sub = out[todo]
        dup = np.zeros(sub.shape, dtype=bool)
        dup[:, 1:] = sub[:, 1:] == sub[:, :-1]
        hit = dup.any(axis=1)
        todo, sub, dup = todo[hit], sub[hit], dup[hit]
        sub[dup] = draw(int(dup.sum()))
        sub.sort(axis=1)
        out[todo] = sub
    return out


def crossover(rows: list[dict]) -> dict:
    """`r*` from the ok N = 1M points (PROTOCOL.md).

    The uniform points sorted by r: a crossing is an adjacent pair on either
    side of ratio 1 (ratio = rows / dense K=2 median). `r*` is the geometric
    mean of the pair's r values at the lowest crossing from rows-faster below
    to rows-slower above; if the Zipf line crosses that way lower, its value.
    `monotone` is false when the uniform points cross more than once or the
    other way: then r alone does not order the kernels.
    """

    def crossings(pts):
        pts = sorted(pts, key=lambda p: p["r"])
        up, down = [], []
        for a, b in zip(pts, pts[1:]):
            if a["ratio"] < 1 <= b["ratio"]:
                up.append((a, b))
            elif b["ratio"] < 1 <= a["ratio"]:
                down.append((a, b))
        return up, down

    ok = [r for r in rows if r.get("status") == "ok" and r["n_rows"] == N_ROWS]
    up, down = crossings([r for r in ok if r["dist"] == "uniform"])
    zup, _ = crossings([r for r in ok if r["dist"] == "zipf"])

    def gm(pair):
        return math.sqrt(pair[0]["r"] * pair[1]["r"])

    def ids(pairs):
        return [(a["id"], b["id"]) for a, b in pairs]

    r_star = min(map(gm, up), default=None)
    if r_star is not None and zup and min(map(gm, zup)) < r_star:
        r_star = min(map(gm, zup))
    return {
        "r_star": r_star,
        "uniform_up": ids(up),
        "uniform_down": ids(down),
        "zipf_up": ids(zup),
        "monotone": len(up) == 1 and not down,
    }


def check_points(rows: list[dict]) -> list[dict]:
    """The two uniform 1M points re-run at N = 4M."""
    ok = [r for r in rows if r.get("status") == "ok" and r["n_rows"] == N_ROWS and r["dist"] == "uniform"]
    up = crossover(rows)["uniform_up"]
    if up:
        chosen = list(up[0])
    else:
        chosen = [r["id"] for r in sorted(ok, key=lambda r: abs(math.log(r["ratio"])))[:2]]
    by_id = {r["id"]: r for r in ok}
    return [point("uniform", N_ROWS_CHECK, by_id[i]["n_items"], by_id[i]["row_len"]) for i in chosen]


def _child(pt: dict) -> dict:
    import polars as pl
    import pyarrow as pa
    from child_run import result_signatures
    from consolidation_run import _FALLBACK_RE
    from loguru import logger

    from et_miner.core.apriori import apriori
    from et_miner.core.result import _min_count

    t0 = time.perf_counter()
    rows = generate_rows(pt["dist"], pt["n_rows"], pt["n_items"], pt["row_len"], zlib.crc32(pt["id"].encode()))
    counts = np.bincount(rows.ravel(), minlength=pt["n_items"])
    min_count = int(counts.min())
    if min_count < 1:
        raise RuntimeError(f"{int((counts == 0).sum())} items never drawn")
    offsets = np.arange(0, rows.size + 1, pt["row_len"], dtype=np.int64)
    df = pl.DataFrame(
        {"items": pl.Series("items", pa.LargeListArray.from_arrays(offsets, rows.ravel().astype(np.int64)))}
    )
    del rows
    min_support = (min_count - 0.5) / pt["n_rows"]
    if _min_count(min_support, pt["n_rows"]) != min_count:
        raise RuntimeError(f"min_support {min_support!r} does not give min_count {min_count}")
    gen_s = time.perf_counter() - t0

    logged: list[str] = []
    logger.add(lambda m: logged.append(m.record["message"]), level="DEBUG")

    def run(kernel: str) -> tuple[float, float, tuple]:
        os.environ["ET_MINER_K2_KERNEL"] = kernel
        logged.clear()
        levels: dict[int, float] = {}
        t = time.perf_counter()
        res = apriori(
            df,
            use_gpu=True,
            n_gpus=1,
            min_support=min_support,
            max_length=2,
            level_callback=lambda k, n_cand, n_freq, ms: levels.__setitem__(k, ms),
        )
        wall = time.perf_counter() - t
        fallbacks = [m for m in logged if _FALLBACK_RE.search(m) and "NCCL unavailable" not in m]
        if fallbacks:
            raise RuntimeError(f"K2 kernel {kernel!r}: fallback logged: {fallbacks[0][:300]}")
        if (kernel == "rows") != any(ROWS_LOG in m for m in logged):
            raise RuntimeError(f"K2 kernel {kernel!r} pinned, but the miner's K=2 log says otherwise")
        n_pairs = res.filter(pl.col("itemset").list.len() == 2).height
        return levels[2], wall, (n_pairs, *result_signatures(res, pt["n_rows"]).values())

    for k in KERNELS:
        run(k)
    times: dict[str, list[float]] = {k: [] for k in KERNELS}
    walls: dict[str, list[float]] = {k: [] for k in KERNELS}
    sigs = set()
    for _ in range(REPS):
        for k in KERNELS:
            ms, wall, sig = run(k)
            times[k].append(round(ms, 3))
            walls[k].append(round(wall, 3))
            sigs.add(sig)
    if len(sigs) != 1:
        raise RuntimeError(f"the kernels disagree on the result: {sorted(sigs)}")
    n_pairs_frequent, n_itemsets, sum_counts, itemset_hash = sigs.pop()
    med = {k: statistics.median(v) for k, v in times.items()}
    return {
        **pt,
        "status": "ok",
        "r": r_of(pt["n_rows"], pt["n_items"], pt["row_len"]),
        "min_count": min_count,
        "n_frequent_pairs": n_pairs_frequent,
        "n_itemsets": n_itemsets,
        "sum_counts": sum_counts,
        "itemset_hash": itemset_hash,
        "k2_ms": times,
        "wall_s": walls,
        "median_dense_ms": med["dense"],
        "median_rows_ms": med["rows"],
        "ratio": med["rows"] / med["dense"],
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
    for r in sorted(rows, key=lambda r: (r["n_rows"], r["dist"], r.get("r", 0))):
        if r.get("status") == "ok":
            print(
                f"  {r['id']:<26} r={r['r']:<10.3g} dense={r['median_dense_ms']:>10.1f} ms "
                f"rows={r['median_rows_ms']:>10.1f} ms  ratio={r['ratio']:.3f}"
            )
        else:
            print(f"  {r['id']:<26} {r.get('status')}")
    print(json.dumps(crossover(rows), indent=1))
    twins = {
        (r["dist"], r["n_items"], r["row_len"]): r for r in rows if r.get("status") == "ok" and r["n_rows"] == N_ROWS
    }
    for r in rows:
        if r["n_rows"] == N_ROWS_CHECK and r.get("status") == "ok":
            twin = twins[(r["dist"], r["n_items"], r["row_len"])]
            same = (r["ratio"] < 1) == (twin["ratio"] < 1)
            print(
                f"  N=4M {r['id']}: ratio {r['ratio']:.3f} vs {twin['ratio']:.3f} at 1M — "
                + ("same side of 1" if same else "OTHER SIDE of 1: N, not only r, moves the crossover")
            )


def main() -> int:
    ap = argparse.ArgumentParser()
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
    out = Path(args.out) if args.out else _campaign_out(here) / "k2_crossover.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    if not args.analyze:
        gpu_s = 0.0
        for stage in (grid, lambda: check_points([r for r in rows if r.get("rev") == here])):
            for pt in stage():
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
