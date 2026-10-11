"""Untimed: SON with use_gpu=True vs efficient-apriori and the in-core CPU route, exact itemsets and counts.

Usage: uv run python bench/son_gpu_completeness.py [workloads] [factors]   (comma-separated; default
son_stakes.SON_WORKLOADS + dslk2 at the tree's LOCAL_SUPPORT_FACTOR). One JSON row per pair, FAILS=<n> last.
"""

import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "bench"), str(REPO / "bench" / "cpu")]

from loguru import logger  # noqa: E402

from consolidation_run import _load  # noqa: E402
from et_miner import apriori  # noqa: E402
from et_miner.streaming.son import LOCAL_SUPPORT_FACTOR, apriori_streaming  # noqa: E402
from matrix import WORKLOADS  # noqa: E402
from son_stakes import INPUT_WORKLOADS, SON_WORKLOADS, _rev  # noqa: E402

logger.remove()


#: Workloads efficient-apriori is not run on, and why; their GPU rows compare against the CPU route only (which the
#: tier-equivalence gate ties to efficient-apriori on the smoke preset).
EA_SKIP = {
    "dslk2": "20M rows: efficient-apriori holds every transaction and every item's tid set as Python objects, "
    "estimated well past this box's 31 GB (not run)",
}


def as_counts(res, n_rows: int) -> dict:
    return {
        tuple(sorted(int(i) for i in s)): round(sup * n_rows)
        for s, sup in zip(res["itemset"].to_list(), res["support"].to_list())
    }


def ea_counts(df, min_support: float, max_length) -> dict:
    """efficient-apriori under the CLAUDE.md convention: (min_count - 0.5) / N and an explicit max_length."""
    import polars as pl
    from efficient_apriori import itemsets_from_transactions

    from et_miner.core.result import _min_count

    n = df.height
    ml = max_length or int(df.select(pl.col("items").list.len().max()).item() or 1)
    tx = [tuple(r) for r in df["items"].to_list()]
    itemsets, _ = itemsets_from_transactions(tx, (_min_count(min_support, n) - 0.5) / n, max_length=ml)
    return {tuple(sorted(int(i) for i in k)): v for level in itemsets.values() for k, v in level.items()}


def diff(want: dict, got: dict) -> dict:
    return {
        "missing": len(want.keys() - got.keys()),
        "extra": len(got.keys() - want.keys()),
        "wrong_counts": sum(1 for k in want.keys() & got.keys() if want[k] != got[k]),
    }


def main() -> int:
    workloads = sys.argv[1].split(",") if len(sys.argv) > 1 else [*SON_WORKLOADS, *INPUT_WORKLOADS]
    factors = [float(f) for f in sys.argv[2].split(",")] if len(sys.argv) > 2 else [LOCAL_SUPPORT_FACTOR]
    rev = _rev()
    fails = 0
    for w in workloads:
        dataset, min_support, max_length = {**WORKLOADS, **INPUT_WORKLOADS}[w]
        df, n_rows, _ = _load(dataset)
        chunk_size = math.ceil(n_rows / 4)
        cpu = as_counts(apriori(df.lazy(), min_support=min_support, max_length=max_length), n_rows)
        ea = None if w in EA_SKIP else ea_counts(df, min_support, max_length)
        cpu_vs_ea = None if ea is None else diff(ea, cpu)
        for f in factors:
            t0 = time.perf_counter()
            res = apriori_streaming(
                df.lazy(), min_support=min_support, max_length=max_length, chunk_size=chunk_size,
                local_support_factor=f, use_gpu=True, show_progress=False,
            )
            wall = round(time.perf_counter() - t0, 2)
            got = as_counts(res, n_rows)
            gpu_vs_cpu = diff(cpu, got)
            gpu_vs_ea = None if ea is None else diff(ea, got)
            ok = not any(d and any(d.values()) for d in (gpu_vs_cpu, gpu_vs_ea, cpu_vs_ea))
            fails += not ok
            print(json.dumps({
                "workload": w, "factor": f, "n_rows": n_rows, "chunk_size": chunk_size,
                "ea_itemsets": None if ea is None else len(ea), "ea_skipped": EA_SKIP.get(w),
                "cpu_itemsets": len(cpu), "gpu_itemsets": len(got),
                "gpu_vs_ea": gpu_vs_ea, "gpu_vs_cpu": gpu_vs_cpu, "cpu_vs_ea": cpu_vs_ea,
                "ok": ok, "untimed_wall_s": wall, "rev": rev,
            }), flush=True)
    print(f"FAILS={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
