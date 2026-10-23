"""Shared helpers for the CPU reproduction harness.

Thread and device pinning checks, min_count/min_support conversion, canonical
per-K itemset tables (write, load, signature), GPU-reference loading and
direct counting over a transaction set.

Usage:
    import _harness as h   # after `h.check_env()` succeeds, import polars/numpy

Options: none (library module). `THREADS` comes from the environment.
"""

import hashlib
import json
import math
import os
import resource
import time
from fractions import Fraction
from pathlib import Path

PINNED_VARS = ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "OMP_NUM_THREADS",
               "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def check_env():
    """Assert no CUDA device is visible and every pool is pinned; return THREADS.

    Must run before polars or numpy is imported, so the pins are in effect
    when their pools start.
    """
    threads = int(os.environ["THREADS"])
    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "CUDA_VISIBLE_DEVICES must be set and empty"
    for var in PINNED_VARS:
        assert os.environ.get(var) == str(threads), f"{var} must equal THREADS={threads}"
    assert not any(k.startswith("ET_MINER_") for k in os.environ), "no ET_MINER_* knobs allowed"
    assert "ET_UPLOAD_GCS" not in os.environ
    return threads


def min_support_for(min_count, n_rows, support=None):
    """Return the min_support that makes ceil(Fraction(str(s)) * N) == min_count.

    A given round `support` is used when it hits the target; otherwise
    (min_count - 0.5) / N, which is exact for integer counts.
    """
    for s in ([support] if support is not None else []) + [(min_count - 0.5) / n_rows]:
        if math.ceil(Fraction(str(s)) * n_rows) == min_count:
            return s
    raise AssertionError(f"no support reaches min_count={min_count} at N={n_rows}")


def counts_from_support(support, n_rows):
    """Recover absolute counts from supports (count / N), asserting a residual below 1e-6."""
    import numpy as np

    scaled = np.asarray(support, dtype=np.float64) * n_rows
    counts = np.rint(scaled)
    resid = float(np.max(np.abs(scaled - counts))) if len(counts) else 0.0
    assert resid < 1e-6, f"count recovery residual {resid}"
    return counts.astype(np.int64)


def normalize(df, n_rows=None, sort_items=True):
    """Return {K: DataFrame[itemset List(Int64) ascending, count Int64]} in canonical row order.

    `df` holds `itemset` and either `count` or `support` (then `n_rows` is required).
    With `sort_items=False` the items are not sorted but must already be strictly
    ascending (asserted per level), which avoids a copy of a large engine result.
    """
    import polars as pl

    if "count" not in df.columns:
        df = df.with_columns(count=pl.Series(counts_from_support(df["support"].to_numpy(), n_rows)))
    items = pl.col("itemset").cast(pl.List(pl.Int64))
    df = df.select(items.list.sort() if sort_items else items, pl.col("count").cast(pl.Int64))
    df = df.with_columns(k=pl.col("itemset").list.len())
    out = {}
    for (k,), part in df.partition_by("k", as_dict=True, maintain_order=False).items():
        out[int(k)] = canonical(part.drop("k"), int(k))
    return dict(sorted(out.items()))


def as_matrix(part, k):
    """Return the itemsets of one K level as an (n, k) int64 array."""
    import numpy as np

    flat = part["itemset"].explode().to_numpy()
    assert len(flat) == part.height * k
    return np.ascontiguousarray(flat.reshape(part.height, k).astype(np.int64))


def canonical(part, k):
    """Sort one K level lexicographically by itemset and check items are strictly ascending."""
    import numpy as np

    m = as_matrix(part, k)
    if k > 1:
        assert (np.diff(m, axis=1) > 0).all(), f"K={k}: itemset not strictly ascending / duplicate item"
    order = np.lexsort(m.T[::-1]) if k > 0 else np.arange(part.height)
    return part[order]


class Signature:
    """Incremental form of `signature`: feed canonical per-K tables in ascending K."""

    def __init__(self):
        self.h, self.n, self.s, self.dist = hashlib.sha256(), 0, 0, {}

    def update(self, k, part):
        assert not self.dist or k > max(self.dist), "levels must arrive in ascending K"
        m = as_matrix(part, k)
        c = part["count"].to_numpy().astype("<i8")
        self.h.update(f"K{k}:{part.height}\n".encode())
        self.h.update(m.astype("<i8").tobytes())
        self.h.update(c.tobytes())
        self.n += part.height
        self.s += int(c.sum())
        self.dist[k] = part.height

    def result(self):
        return {"itemsets": self.n, "sum_counts": self.s, "sha256": self.h.hexdigest(), "k_distribution": self.dist}


def signature(tables):
    """Return {itemsets, sum_counts, sha256, k_distribution} of canonical per-K tables."""
    sig = Signature()
    for k, part in sorted(tables.items()):
        sig.update(k, part)
    return sig.result()


def write_tables(tables, out_dir, n_rows):
    """Write canonical per-K tables as out_dir/k{K}.parquet (itemset, count, support)."""
    import polars as pl

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, part in tables.items():
        part.with_columns(support=pl.col("count").cast(pl.Float64) / n_rows).write_parquet(
            out_dir / f"k{k}.parquet", compression="zstd")


def load_tables(path, n_rows=None, min_count=None, translate=None):
    """Load itemset tables from a CPU dir (k*.parquet with count), a GPU per-K dir
    (frequent_k*.parquet with support) or one parquet file, as canonical per-K tables.

    `min_count` filters rows with count >= min_count. `translate` maps item ids
    ({old_id: new_id}, every id present) before normalizing. A support-only table
    is first cut at support * n_rows >= min_count - 0.75 per file (a superset of the
    exact count filter, since recovered counts have a residual below 1e-6).
    """
    import polars as pl

    path = Path(path)
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    assert files, f"no parquet under {path}"

    def read(f):
        part = pl.read_parquet(f)
        part = part.select([c for c in ("itemset", "count", "support") if c in part.columns])
        if min_count is not None and n_rows is not None and "count" not in part.columns:
            part = part.filter(pl.col("support") * n_rows >= min_count - 0.75)
        return part

    df = pl.concat([read(f) for f in files], how="vertical_relaxed")
    if translate is not None:
        df = df.with_columns(pl.col("itemset").list.eval(
            pl.element().cast(pl.Int64).replace_strict(translate, return_dtype=pl.Int64)))
    tables = normalize(df, n_rows)
    if min_count is not None:
        tables = {k: t.filter(pl.col("count") >= min_count) for k, t in tables.items()}
        tables = {k: t for k, t in tables.items() if t.height}
    return tables


def load_items(parquet, head=None):
    """Load the `items` list column (optionally the first `head` rows, as validate_row_split.py)."""
    import polars as pl

    lf = pl.scan_parquet(parquet).select("items")
    if head is not None:
        lf = lf.head(head)
    return lf.collect()


class DirectCounter:
    """Counts the rows that contain every item of an itemset, by tidset intersection.

    Only the tidsets of `needed_items` are built; a row holding an item twice
    counts once.
    """

    def __init__(self, items_df, needed_items):
        import numpy as np
        import polars as pl

        needed = pl.Series(sorted({int(i) for i in needed_items}), dtype=pl.Int64)
        ex = (items_df.with_row_index("row").explode("items")
              .filter(pl.col("items").cast(pl.Int64).is_in(needed.implode()))
              .sort(["items", "row"]))
        rows = ex["row"].to_numpy().astype(np.int64)
        item = ex["items"].to_numpy().astype(np.int64)
        keep = np.ones(len(rows), dtype=bool)
        keep[1:] = (item[1:] != item[:-1]) | (rows[1:] != rows[:-1])
        self.rows, item = rows[keep], item[keep]
        self.n_rows = items_df.height
        uniq, start = np.unique(item, return_index=True)
        end = np.append(start[1:], len(item))
        self.span = {int(u): (int(a), int(b)) for u, a, b in zip(uniq, start, end, strict=True)}

    def count(self, itemset):
        """Return the number of rows containing all items of `itemset`."""
        import numpy as np

        sets = []
        for it in itemset:
            if int(it) not in self.span:
                return 0
            a, b = self.span[int(it)]
            sets.append(self.rows[a:b])
        sets.sort(key=len)
        acc = sets[0]
        for s in sets[1:]:
            acc = np.intersect1d(acc, s, assume_unique=True)
            if not len(acc):
                break
        return int(len(acc))


class RunStats:
    """Wall time, CPU seconds and peak RSS of the current process from construction."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.ru0 = resource.getrusage(resource.RUSAGE_SELF)

    def snapshot(self):
        """Return {wall_s, cpu_s, peak_rss_mb} since construction (peak RSS is per process)."""
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "wall_s": time.perf_counter() - self.t0,
            "cpu_s": (ru.ru_utime - self.ru0.ru_utime) + (ru.ru_stime - self.ru0.ru_stime),
            "peak_rss_mb": ru.ru_maxrss / 1024,
        }


def env_record(threads):
    """Return the engine identity and device/thread facts recorded with every run."""
    import et_miner

    return {
        "et_miner_file": et_miner.__file__,
        "has_rust_extension": bool(et_miner.has_rust_extension()),
        "THREADS": threads,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dev_nvidia": sorted(str(p) for p in Path("/dev").glob("nvidia*")),
        "pins": {v: os.environ.get(v) for v in PINNED_VARS},
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def dump_json(obj, path):
    """Write `obj` as indented JSON (int keys become strings)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + "\n")
