"""Permutation null model on the CPU route (in-core `apriori` per permutation).

Seeds follow experiment_null_model.py: a master `np.random.default_rng(SEED)`
draws one seed per permutation, and each permutation's own generator draws
the shuffle seed. The flat item vector of all rows is shuffled with NumPy and
cut at the original row offsets (row lengths and item totals are kept),
duplicates are removed within each row, and the result is mined at MIN_COUNT.
Each permutation runs in its own process and writes PERM_DIR/meta.json last,
which is its checkpoint; finished permutations are skipped.

Usage:
    run_py.sh null_model.py DATA.parquet OUT_DIR --runs R [options]

Options:
    --min-count M     null threshold (default 769)
    --seed S          master seed (default 42)
    --perms A-B       permutation indices to run, 0-based inclusive (default all R)
    --check-seeds J   GPU null JSON whose parameters.permutation_seeds must match ours
    --one I           internal: run permutation I in this process
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as h  # noqa: E402

THREADS = h.check_env()


def permutation_seeds(seed, runs):
    """Return the per-permutation seeds of experiment_null_model.py."""
    import numpy as np

    master = np.random.default_rng(seed)
    return [int(master.integers(0, 2**63)) for _ in range(runs)]


def shuffle_seed(perm_seed):
    """Return the shuffle seed a permutation draws from its own generator."""
    import numpy as np

    return int(np.random.default_rng(perm_seed).integers(0, 2**63))


def flat_and_offsets(items):
    """Return (flat item vector, row offsets) of a polars List series."""
    import numpy as np

    lengths = items.list.len().to_numpy().astype(np.int64)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    flat = items.explode().to_numpy().copy()
    assert len(flat) == offsets[-1]
    return flat, offsets


def rebuild(shuffled, offsets):
    """Cut a flat item vector at row offsets and drop duplicates within each row."""
    import polars as pl
    import pyarrow as pa

    arr = pa.LargeListArray.from_arrays(pa.array(offsets, type=pa.int64()), pa.array(shuffled))
    return pl.DataFrame({"items": pl.Series("items", arr)}).with_columns(pl.col("items").list.unique().list.sort())


def run_one(a, i, seeds):
    """Shuffle, rebuild, mine and record permutation `i`."""
    import numpy as np

    from et_miner import apriori

    pdir = Path(a.out) / f"perm_{i:03d}"
    stats = h.RunStats()
    df = h.load_items(a.data)
    n_rows = df.height
    s = h.min_support_for(a.min_count, n_rows, a.min_count / n_rows)
    flat, offsets = flat_and_offsets(df["items"])
    del df
    sseed = shuffle_seed(seeds[i])
    np.random.default_rng(sseed).shuffle(flat)
    null_df = rebuild(flat, offsets)
    del flat, offsets
    build_s = stats.snapshot()["wall_s"]
    levels = []

    def on_level(k, n_candidates, n_frequent, duration_ms):
        levels.append({"k": k, "candidates": int(n_candidates), "frequent": int(n_frequent),
                       "ms": float(duration_ms)})

    t0 = time.perf_counter()
    res = apriori(null_df, min_support=s, max_length=None, use_gpu=False, n_jobs=THREADS,
                  level_callback=on_level, show_progress=False)
    mine_s = time.perf_counter() - t0
    run = stats.snapshot()
    del null_df
    tables = h.normalize(res, n_rows)
    h.write_tables(tables, pdir / "parquet", n_rows)
    sig = h.signature(tables)
    meta = {"perm": i, "perm_seed": seeds[i], "shuffle_seed": sseed,
            "shuffle": "numpy default_rng(shuffle_seed).shuffle",
            "n_rows": n_rows, "min_count": a.min_count, "min_support": s, "build_s": build_s,
            "mine_wall_s": mine_s, "process": run, "levels": levels, "total_itemsets": sig["itemsets"],
            "k_distribution": sig["k_distribution"], "k_max": max(tables), "signature": sig,
            "env": h.env_record(THREADS)}
    h.dump_json(meta, pdir / "meta.json")
    with open(Path(a.out) / "checkpoint.jsonl", "a") as f:
        f.write(json.dumps({k: meta[k] for k in ("perm", "perm_seed", "shuffle_seed", "total_itemsets",
                                                  "k_max", "mine_wall_s")}) + "\n")
    print(json.dumps({"perm": i, "total": sig["itemsets"], "k_max": meta["k_max"], "mine_wall_s": round(mine_s, 1),
                      "peak_rss_mb": round(run["peak_rss_mb"])}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--runs", type=int, required=True)
    ap.add_argument("--min-count", type=int, default=769)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perms")
    ap.add_argument("--check-seeds")
    ap.add_argument("--one", type=int)
    a = ap.parse_args()

    seeds = permutation_seeds(a.seed, a.runs)
    if a.check_seeds:
        gpu = json.loads(Path(a.check_seeds).read_text())["parameters"]["permutation_seeds"]
        n = min(len(gpu), len(seeds))
        assert [int(x) for x in gpu[:n]] == seeds[:n], "permutation seeds differ from the GPU run"
    if a.one is not None:
        run_one(a, a.one, seeds)
        return
    lo, hi = (int(x) for x in a.perms.split("-")) if a.perms else (0, a.runs - 1)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    h.dump_json({"seed": a.seed, "runs": a.runs, "permutation_seeds": seeds}, Path(a.out) / "seeds.json")
    for i in range(lo, hi + 1):
        if (Path(a.out) / f"perm_{i:03d}" / "meta.json").exists():
            print(f"perm {i}: done, skipped", flush=True)
            continue
        cmd = [sys.executable, __file__, a.data, a.out, "--runs", str(a.runs), "--min-count", str(a.min_count),
               "--seed", str(a.seed), "--one", str(i)]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
