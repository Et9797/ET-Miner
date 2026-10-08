"""SON's profile phases at 1 thread: one run per workload, each in a fresh process (diagnostic, Amendment 5).

Loads the workload, warms up on a 20,000-row smoke slice as son_stakes.py
does, runs ``apriori_streaming(profile=True)`` with 4 chunks and appends one
JSON row: wall time, seconds per profile phase, the phases' extras and the
tree digest. Not used by any decision rule.

Usage:
    uv run python bench/cpu/son_phases.py rows.jsonl
"""

import os

for _k in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"

import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def one(w: str, out: str) -> None:
    sys.path.insert(0, str(REPO / "bench"))
    sys.path.insert(0, str(REPO / "bench" / "cpu"))
    from loguru import logger

    logger.remove()
    import polars as pl
    from consolidation_run import _load
    from matrix import WORKLOADS
    from son_stakes import WARMUP_ROWS, _rev

    from et_miner.streaming.son import apriori_streaming

    ds, ms, ml = WORKLOADS[w]
    df, n, _ = _load(ds)
    warm = pl.read_parquet(REPO / "datasets" / "synth" / "smoke.parquet").head(WARMUP_ROWS)
    apriori_streaming(warm.lazy(), min_support=0.02, max_length=3, chunk_size=WARMUP_ROWS // 2, show_progress=False, n_jobs=1)
    t0 = time.perf_counter()
    _, sess = apriori_streaming(
        df.lazy(), min_support=ms, max_length=ml, chunk_size=math.ceil(n / 4), show_progress=False, profile=True, n_jobs=1
    )
    wall = time.perf_counter() - t0
    row = {
        "workload": w, "threads": 1, "wall_s": round(wall, 4),
        "phases_s": {p.name: round(p.duration_ms / 1000, 4) for p in sess.phases},
        "extra": {p.name: p.extra for p in sess.phases}, "rev": _rev(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(json.dumps(row), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(sys.argv[2], sys.argv[3])
    else:
        for w in ["smoke", "deepk", "skew", "wide", "or005", "or0001k2"]:
            subprocess.run([sys.executable, __file__, "--one", w, sys.argv[1]], check=True)
