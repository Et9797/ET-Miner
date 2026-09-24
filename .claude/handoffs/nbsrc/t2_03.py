# %% [markdown]
# # Tier 2.3: Same results, and what the speed looks like
#
# Tier 1 and tier 2 must return the same itemsets with the same counts.
# This notebook checks that on the shared sample for three routes and an independent reference miner,
# then measures the speed of each route at three data sizes.
#
# The three routes:
#
# | Route | Call | Where the loop runs | Where counting runs |
# |---|---|---|---|
# | Tier 1 | `apriori(df, sparse=False)` | Python | Polars |
# | Tier 2, per level | `apriori(df, sparse=True)` | Python | SciPy for K=2, Rust `count_itemsets_simd` for K>=3 |
# | Tier 2, whole loop | `apriori_from_csr(indptr, indices, ...)` | Rust | Rust |
#
# **What you will learn**
#
# - how to assert that two routes agree on itemsets and counts;
# - how to call `apriori_from_csr` on your own data;
# - which route is faster on this machine and why;
# - which limit of the CPU tiers motivates the GPU tier.
#
# **Prerequisites**: [02-inside-the-rust-core](02-inside-the-rust-core.ipynb).

# %% [markdown]
# ## 0. Setup
#
# `to_csr()` turns a list column into the `int64` CSR arrays that Rust expects, with item values mapped to column
# indices 0..n-1. `mine_rust()` calls `apriori_from_csr` and maps the columns back to item values.
# `counted()` turns any result into a set of `(itemset, count)` pairs, the form every equality check below uses.

# %%
import os
import warnings

os.environ.setdefault("LOGURU_LEVEL", "WARNING")
warnings.filterwarnings("ignore", message="IProgress not found")

import dataclasses
import datetime
import math
import platform
import time
from pathlib import Path

import numpy as np
import polars as pl
from efficient_apriori import apriori as ea_apriori

import et_miner
from et_miner import apriori, apriori_from_csr
from et_miner.core.result import _min_count
from et_miner.synthetic import PRESETS, generate_transactions

assert et_miner.HAS_RUST, "build the extension first: see 01-build-and-install.md"
DATA = Path("../data")


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    return f"{cpu}, {os.cpu_count()} logical CPUs, Python {platform.python_version()}, {datetime.date.today()}"


def to_csr(df: pl.DataFrame, item_col: str = "items"):
    lengths = df[item_col].list.len().to_numpy()
    values = df[item_col].explode(empty_as_null=True).drop_nulls().to_numpy()
    ordered = np.sort(values, kind="stable")
    vocab = ordered[np.r_[True, ordered[1:] != ordered[:-1]]]  # distinct item values, ascending
    cols = np.searchsorted(vocab, values)
    key = np.repeat(np.arange(df.height, dtype=np.int64), lengths) * len(vocab) + cols
    key = np.sort(key, kind="stable")  # sort items within each row
    key = key[np.r_[True, key[1:] != key[:-1]]]  # drop duplicates within a row
    row_id, cols = np.divmod(key, len(vocab))
    indptr = np.searchsorted(row_id, np.arange(df.height + 1))
    return indptr.astype(np.int64), cols.astype(np.int64), df.height, len(vocab), vocab.tolist()


def mine_rust(df: pl.DataFrame, min_support: float, max_length: int = 0):
    indptr, indices, n_rows, n_cols, vocab = to_csr(df)
    itemsets, counts = apriori_from_csr(indptr, indices, n_rows, n_cols, min_support, max_length)
    return {(tuple(vocab[c] for c in s), int(n)) for s, n in zip(itemsets, counts)}


def counted(result: pl.DataFrame, n_rows: int):
    return {(tuple(s), round(v * n_rows)) for s, v in zip(result["itemset"].to_list(), result["support"].to_list())}


smoke = pl.read_parquet(DATA / "smoke.parquet")
N, S = smoke.height, 0.01
print(hardware(), "| et_miner_rust", et_miner.get_rust_version())

# %% [markdown]
# ## 1. Same itemsets, same counts
#
# The next cell mines the shared sample with the three routes and with
# [efficient-apriori](https://github.com/tommyod/Efficient-Apriori), an independent pure-Python Apriori
# that the repository uses as its reference in `tests/test_tier_equivalence.py`.
# Two details of that call are required for an exact comparison:
#
# - `min_support = (min_count - 0.5) / N`: efficient-apriori compares a float support, the library an integer count;
#   the half-count offset makes both keep exactly the itemsets with `count >= min_count`;
# - an explicit `max_length`: efficient-apriori stops at length 8 unless told otherwise.

# %%
min_count = _min_count(S, N)
tier1 = counted(apriori(smoke, min_support=S, sparse=False), N)
tier2_levels = counted(apriori(smoke, min_support=S, sparse=True), N)
tier2_rust = mine_rust(smoke, S)

transactions = [tuple(r) for r in smoke["items"].to_list()]
ea_itemsets, _ = ea_apriori(transactions, min_support=(min_count - 0.5) / N, min_confidence=1.0,
                            max_length=max(len(t) for t in transactions))
oracle = {(tuple(sorted(s)), int(c)) for level in ea_itemsets.values() for s, c in level.items()}

assert tier1 == tier2_levels == tier2_rust == oracle
print(f"min_count={min_count}: all four agree on {len(tier1)} (itemset, count) pairs")

# %% [markdown]
# All four answers are the same set of itemsets with the same counts.
# The comparison uses integer counts, not float supports, so there is no rounding tolerance in it.

# %% [markdown]
# ## 2. Speed at three data sizes
#
# The next cell generates the sample's preset at 60,000, 240,000 and 960,000 rows (same seed and distribution)
# and times each route once per size. The CSR conversion for `apriori_from_csr` is timed separately,
# because the other two routes include their own input preparation.

# %%
apriori(smoke, min_support=S, sparse=True, n_jobs=-1)  # warm-up: thread pools and libraries load on first use
rows = []
for n_rows in (60_000, 240_000, 960_000):
    df, _ = generate_transactions(dataclasses.replace(PRESETS["smoke"], n_rows=n_rows))
    t = time.perf_counter(); r1 = apriori(df, min_support=S, sparse=False); t1 = time.perf_counter() - t
    t = time.perf_counter(); r2 = apriori(df, min_support=S, sparse=True, n_jobs=-1); t2 = time.perf_counter() - t
    t = time.perf_counter(); csr = to_csr(df); tc = time.perf_counter() - t
    t = time.perf_counter(); its, cnt = apriori_from_csr(*csr[:4], S, 0); t3 = time.perf_counter() - t
    assert counted(r1, n_rows) == counted(r2, n_rows) == {(tuple(csr[4][c] for c in s), int(n)) for s, n in zip(its, cnt)}
    rows.append(dict(rows=n_rows, itemsets=r1.height, polars_s=round(t1, 3), sparse_true_s=round(t2, 3),
                     to_csr_s=round(tc, 3), apriori_from_csr_s=round(t3, 3)))
timing = pl.DataFrame(rows).with_columns(
    (pl.col("polars_s") / pl.col("apriori_from_csr_s")).round(1).alias("polars / rust loop"),
    (pl.col("polars_s") / pl.col("sparse_true_s")).round(2).alias("polars / sparse_true"),
)
print(hardware())
timing

# %% [markdown]
# On this machine the whole-loop Rust call is the fastest at every size, by more than an order of magnitude
# (the `polars / rust loop` column leaves the conversion out).
# The CSR conversion (`to_csr_s`, done in NumPy in this notebook) is not free: at the largest size it costs far more
# than the mining call. Convert once and reuse the arrays when you mine the same data at several thresholds.
# The per-level route `sparse=True` is slower than the Polars route here (a ratio below 1 in the last column).
# All three return identical results at every size.

# %% [markdown]
# ## 3. Why `sparse=True` is not faster here
#
# `profile=True` returns the result and a `ProfilingSession`, whose `phases` hold the time of every phase.
# The next cell compares the support-counting phases of the two `apriori` routes on the sample.

# %%
phases = {}
for sparse in (False, True):
    _, session = apriori(smoke, min_support=S, sparse=sparse, profile=True)
    phases[f"sparse={sparse} ms"] = {p.name: round(p.duration_ms, 1) for p in session.phases
                                     if p.name.endswith("support_count")}
names = list(phases["sparse=False ms"])
pl.DataFrame([{"phase": p, **{route: phases[route][p] for route in phases}} for p in names])

# %% [markdown]
# The sparse route costs more per level, and the gap stays even when a level has only a handful of candidates.
# That points at fixed work per level rather than per candidate. The source shows three such costs:
#
# 1. every level converts the whole Polars boolean matrix to a SciPy CSR matrix again
#    (`src/et_miner/core/sparse.py:count_support_sparse` calls `src/et_miner/core/matrix.py:_polars_to_sparse_csr`);
# 2. every Rust call converts CSR to CSC and rebuilds one bit vector per item before counting
#    (`rust_ext/src/core/counting.rs:count_itemsets_simd_raw`);
# 3. the Python side copies the index arrays to `int64` before each call (notebook 2, section 2).
#
# `apriori_from_csr` pays for the conversion and the bit vectors once and keeps them for every level.

# %% [markdown]
# ## 4. When the library picks the sparse route by itself
#
# With the default `sparse=None`, `src/et_miner/core/sparse.py:_choose_counting_strategy` picks the sparse route when
# there are more than 100,000 candidate pairs, or more than 500 items at under 10% density, or when the dense matrix
# would exceed 1 GB. The next cell asks it about the shared sample and about three other shapes.

# %%
from et_miner.core.sparse import _choose_counting_strategy

shapes = [("smoke at 0.01", 118, 60_000, 0.04), ("500 items", 500, 60_000, 0.04),
          ("600 items, 5% dense", 600, 60_000, 0.05), ("100 items, 100M rows", 100, 100_000_000, 0.2)]
pl.DataFrame([dict(shape=name, frequent_items=n, rows=r, density=d, strategy=_choose_counting_strategy(n, r, d))
              for name, n, r, d in shapes])

# %% [markdown]
# The rule looks at size and memory, not at measured speed.
# So on large inputs `apriori` can switch to the sparse route without being asked; pass `sparse=False`
# to keep tier 1, or call `apriori_from_csr` for the whole-loop Rust route.

# %% [markdown]
# ## 5. Using `apriori_from_csr` on your own data
#
# `apriori_from_csr` takes only `min_support` and `max_length` (0 means no limit).
# It has no pruning options, progress reporting, profiling or DataFrame output.
# The `to_csr` and `mine_rust` helpers in the setup cell are enough to use it on any list column;
# the next cell turns their output into the same frame layout `apriori` returns.

# %%
pairs = mine_rust(smoke, S, max_length=2)
frame = pl.DataFrame({"itemset": [list(s) for s, _ in pairs], "support": [c / N for _, c in pairs]})
frame.sort("support", descending=True).head(5)

# %% [markdown]
# The frame has the `itemset` and `support` columns of tier 1, so the rest of the tutorials apply unchanged.

# %% [markdown]
# ## 6. The limit that motivates tier 3
#
# `apriori_from_csr` keeps one bit vector per item for the whole run, and every count walks all
# `ceil(n_rows / 64)` words of each item in the candidate. So memory grows with items times rows, and the
# work per level grows with candidates times rows, spread over the CPU's cores.
# The next cell computes the bit-vector memory for the sample and for the repository's larger synthetic presets,
# using `n_items * ceil(n_rows / 64) * 8` bytes with every vocabulary item, which is what `apriori_from_csr` builds.

# %%
memory = [dict(preset=name, rows=spec.n_rows, items=spec.vocab_size,
               bitvec_gb=round(spec.vocab_size * math.ceil(spec.n_rows / 64) * 8 / 1e9, 3))
          for name, spec in PRESETS.items()]
print("CPU threads available to rayon:", __import__("et_miner_rust").get_num_threads())
pl.DataFrame(memory)

# %% [markdown]
# The largest preset needs several gigabytes of bit vectors, and each level multiplies the candidate count by that
# row width. The timing table in section 2 grows with rows at a fixed thread count.
# Tier 3 keeps the same bit layout but puts it in GPU memory and counts candidates on thousands of GPU threads,
# and it can split the rows over several GPUs.

# %% [markdown]
# ## Summary and next step
#
# - Tier 1, both tier 2 routes and efficient-apriori return identical itemsets and counts on the shared sample.
# - `apriori_from_csr` is the fast CPU route on this machine; `apriori(sparse=True)` is slower here because of
#   fixed conversions every level.
# - The automatic sparse choice is based on size and memory, not speed.
# - Bit-vector memory and per-level work both scale with rows, which is what the GPU tier addresses.
#
# Next: [tier 3, notebook 1](../tier3-gpu/01-gpu-setup-and-first-run.ipynb).
