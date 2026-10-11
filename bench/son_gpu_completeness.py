"""Untimed: SON with use_gpu=True vs the in-core CPU route, exact itemsets and counts, per workload x factor.

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
from son_stakes import INPUT_WORKLOADS, SON_WORKLOADS  # noqa: E402

logger.remove()


def as_counts(res, n_rows: int) -> dict:
    return {
        tuple(sorted(int(i) for i in s)): round(sup * n_rows)
        for s, sup in zip(res["itemset"].to_list(), res["support"].to_list())
    }


def main() -> int:
    workloads = sys.argv[1].split(",") if len(sys.argv) > 1 else [*SON_WORKLOADS, *INPUT_WORKLOADS]
    factors = [float(f) for f in sys.argv[2].split(",")] if len(sys.argv) > 2 else [LOCAL_SUPPORT_FACTOR]
    fails = 0
    for w in workloads:
        dataset, min_support, max_length = {**WORKLOADS, **INPUT_WORKLOADS}[w]
        df, n_rows, _ = _load(dataset)
        chunk_size = math.ceil(n_rows / 4)
        ref = as_counts(apriori(df.lazy(), min_support=min_support, max_length=max_length), n_rows)
        for f in factors:
            t0 = time.perf_counter()
            res = apriori_streaming(
                df.lazy(), min_support=min_support, max_length=max_length, chunk_size=chunk_size,
                local_support_factor=f, use_gpu=True, show_progress=False,
            )
            got = as_counts(res, n_rows)
            missing = ref.keys() - got.keys()
            extra = got.keys() - ref.keys()
            wrong = sum(1 for k in ref.keys() & got.keys() if ref[k] != got[k])
            ok = not (missing or extra or wrong)
            fails += not ok
            print(json.dumps({
                "workload": w, "factor": f, "n_rows": n_rows, "chunk_size": chunk_size, "ref_itemsets": len(ref),
                "gpu_itemsets": len(got), "missing": len(missing), "extra": len(extra), "wrong_counts": wrong,
                "ok": ok, "untimed_wall_s": round(time.perf_counter() - t0, 2),
            }), flush=True)
    print(f"FAILS={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
