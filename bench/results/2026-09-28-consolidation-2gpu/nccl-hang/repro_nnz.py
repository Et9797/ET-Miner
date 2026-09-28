"""Warm-up of the campaign child (smoke head(20000), min_support 0.02, max_length 3) on 2 GPUs.
A hang dumps every thread's stack after 45 s and exits 1 (faulthandler)."""
import faulthandler
import sys
import time
faulthandler.dump_traceback_later(45, exit=True)
import polars as pl  # noqa: E402  (armed the hang dump first)
from et_miner import apriori  # noqa: E402  (armed the hang dump first)
repo = sys.argv[1]
df = pl.read_parquet(f"{repo}/datasets/synth/smoke.parquet").head(20000)
t0 = time.perf_counter()
res = apriori(df, use_gpu=True, n_gpus=2, min_support=0.02, max_length=3)
n = getattr(res, "height", None)
if n is None:
    try:
        n = len(res)
    except Exception:
        n = "?"
print(f"OK {time.perf_counter() - t0:.2f}s itemsets={n}", flush=True)
