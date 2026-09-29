"""Time the Rust host roles R2 and R4 against their fallbacks on real level arrays.

Two steps, each in a fresh process so the thread budget is fixed before any
pool starts:

    dump   mine a dataset on the row-split miner (one GPU) and save the inputs
           of every R2 (prefix-group build) and R4 (free-set prune) call to
           <dir>/<dataset>/<role>_k<K>.npz
    time   load every dumped input and time Rust against the fallback, per
           call, R4 together with the fancy-index that follows it

Usage:
    RAYON_NUM_THREADS=6 python bench/microbench_rust.py dump DATASET DIR [--free-sets]
        [--max-length K] [--then-k3] [--n-gpus N]
    RAYON_NUM_THREADS=6 python bench/microbench_rust.py time DIR [--reps 3] [--fallback-timeout S]

`time` prints one JSON line per call; a fallback still running after
--fallback-timeout seconds is interrupted and recorded as a lower bound. `--then-k3` with
`--max-length 2` dumps the K=3 group build the miner would run next, without
counting K=3 (stress_k2's K=3 level holds ~12B candidates).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def dump(dataset: str, out: Path, free_sets: bool, max_length: int | None, then_k3: bool = False,
         n_gpus: int = 1) -> None:
    import polars as pl

    from et_miner.core.matrix import _build_csr_from_transactions
    from et_miner.gpu import kernels, row_split
    from et_miner.synthetic import PRESETS

    out.mkdir(parents=True, exist_ok=True)
    build, prune_free = (
        kernels.build_k3plus_groups_from_flat,
        row_split._prune_non_free_mask,
    )
    seen: dict[str, int] = {}

    def _name(role: str, k: int) -> Path:
        key = f"{role}_k{k}"
        seen[key] = seen.get(key, 0) + 1
        return out / (f"{key}.npz" if seen[key] == 1 else f"{key}_{seen[key]}.npz")

    def _build(flat):
        np.savez(_name("r2", flat.shape[1] + 1), flat=flat)
        return build(flat)

    def _free(cur, cc, prev, pc):
        np.savez(_name("r4", cur.shape[1]), cur=cur, cc=cc, prev=prev, pc=pc)
        return prune_free(cur, cc, prev, pc)

    kernels.build_k3plus_groups_from_flat = _build  # row_split imports it at call time
    row_split._prune_non_free_mask = _free

    spec = PRESETS[dataset]
    df = pl.read_parquet(REPO / "datasets" / "synth" / f"{dataset}.parquet")
    csr, idx_to_item, n = _build_csr_from_transactions(df.lazy(), spec.min_support, "items")
    res = row_split._apriori_row_split_multi_gpu(
        csr, idx_to_item, n, spec.min_support, max_length, n_gpus, None,
        prune_non_free=free_sets,
    )
    if then_k3:
        item_to_col = {item: col for col, item in idx_to_item.items()}
        pairs = [sorted(item_to_col[i] for i in s) for s in res["itemset"].to_list() if len(s) == 2]
        flat = np.array(pairs, dtype=np.int32)
        flat = flat[np.lexsort(flat[:, ::-1].T)]
        kernels.build_k3plus_groups_from_flat(flat)
    print(json.dumps({"dumped": sorted(p.name for p in out.glob("*.npz"))}))


def _time(fn, reps: int, timeout: float) -> tuple[list[float], bool]:
    """Wall time per rep; a rep interrupted at `timeout` s ends the series as a lower bound."""
    import signal

    def _expire(signum, frame):
        raise TimeoutError

    times = []
    previous = signal.signal(signal.SIGALRM, _expire)
    try:
        for _ in range(reps):
            t0 = time.perf_counter()
            if math.isfinite(timeout):
                signal.setitimer(signal.ITIMER_REAL, timeout)
            try:
                fn()
            except TimeoutError:
                times.append(timeout)
                return times, True
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            times.append(time.perf_counter() - t0)
    finally:
        signal.signal(signal.SIGALRM, previous)
    return times, False


def time_calls(root: Path, reps: int, timeout: float) -> None:
    from et_miner.backends import get_rust_ext
    from et_miner.gpu import mining
    from et_miner.gpu.kernels.k3plus import _build_k3plus_groups_numpy, build_k3plus_groups_from_flat

    rust = get_rust_ext()
    if rust is None:
        raise SystemExit("et_miner_rust is not available; nothing to compare")

    for path in sorted(root.rglob("*.npz")):
        z = np.load(path, allow_pickle=False)
        role = path.name.split("_")[0]
        if role == "r2":
            flat = z["flat"]
            rust_fn = lambda: build_k3plus_groups_from_flat(flat)  # noqa: E731
            fb_fn = lambda: _build_k3plus_groups_numpy(flat)  # noqa: E731
            size = {"rows": int(flat.shape[0]), "k": int(flat.shape[1]) + 1}
        elif role == "r4":
            cur, cc, prev, pc = z["cur"], z["cc"], z["prev"], z["pc"]

            def rust_fn():
                keep = mining._prune_non_free_mask(cur, cc, prev, pc)
                return cur[keep], cc[keep]

            def fb_fn():
                keep = mining._prune_non_free_mask_python(cur, cc, prev, pc)
                return cur[keep], cc[keep]

            size = {"rows": int(cur.shape[0]), "prev_rows": int(prev.shape[0]), "k": int(cur.shape[1])}
        else:
            continue
        rust_t, _ = _time(rust_fn, reps, float("inf"))
        fb_t, cut = _time(fb_fn, reps, timeout)
        print(json.dumps({
            "file": str(path.relative_to(root)), "role": role, **size,
            "rust_s": [round(t, 6) for t in rust_t], "fallback_s": [round(t, 6) for t in fb_t],
            "fallback_lower_bound": cut,
            "rayon_threads": os.environ.get("RAYON_NUM_THREADS"),
        }), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("dataset")
    d.add_argument("dir")
    d.add_argument("--free-sets", action="store_true")
    d.add_argument("--max-length", type=int, default=None)
    d.add_argument("--then-k3", action="store_true")
    d.add_argument("--n-gpus", type=int, default=1)
    t = sub.add_parser("time")
    t.add_argument("dir")
    t.add_argument("--reps", type=int, default=3)
    t.add_argument("--fallback-timeout", type=float, default=300.0)
    args = ap.parse_args()
    if not os.environ.get("RAYON_NUM_THREADS"):
        print("set RAYON_NUM_THREADS before running", file=sys.stderr)
        return 2
    if args.cmd == "dump":
        sub_dir = Path(args.dir) / (args.dataset + ("-free" if args.free_sets else ""))
        dump(args.dataset, sub_dir, args.free_sets, args.max_length, args.then_k3, args.n_gpus)
    else:
        time_calls(Path(args.dir), args.reps, args.fallback_timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
