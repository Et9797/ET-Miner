# %% [markdown]
# # Tier 1.3: Mining a realistic sample
#
# This notebook mines the shared sample with the Polars tier and varies the parameters that change the answer or the cost.
# It then times the miner at two data sizes and at falling support thresholds, to show where the Polars tier slows down.
#
# **What you will learn**
#
# - what `min_support`, `max_length`, `prune_equal_support` and `use_generator_pruning` do to the result;
# - how the cost splits across levels;
# - how run time grows with rows and with a lower threshold;
# - why that motivates the compiled tier.
#
# **Prerequisites**: [02-apriori-step-by-step](02-apriori-step-by-step.ipynb), [concepts.md](../concepts.md#7-parameter-reference).

# %% [markdown]
# ## 0. Setup
#
# `hardware()` prints the label that every timing in this notebook carries.

# %%
import os
import warnings

os.environ.setdefault("LOGURU_LEVEL", "WARNING")
warnings.filterwarnings("ignore", message="IProgress not found")

import dataclasses
import datetime
import platform
import time
from pathlib import Path

import polars as pl
from loguru import logger

from et_miner import apriori
from et_miner.core.matrix import build_boolean_matrix
from et_miner.core.result import _min_count
from et_miner.synthetic import PRESETS, generate_transactions

DATA = Path("../data")
smoke = pl.read_parquet(DATA / "smoke.parquet")
N = smoke.height


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    return f"{cpu}, {os.cpu_count()} logical CPUs, Python {platform.python_version()}, {datetime.date.today()}"


def by_length(df: pl.DataFrame) -> dict:
    return dict(df.group_by(pl.col("itemset").list.len().alias("k")).len().sort("k").iter_rows())


print(hardware())
print(N, "rows")

# %% [markdown]
# ## 1. The baseline run
#
# The next cell mines the sample at `min_support=0.01` (a count of at least 600) with the Polars tier.
# It prints how many itemsets there are of each length, and the three planted motifs of the preset for comparison.

# %%
base = apriori(smoke, min_support=0.01, sparse=False)
print("itemsets:", base.height, "| by length:", by_length(base))
_, gen = generate_transactions(PRESETS["smoke"])
for motif, planted_rows in gen.planted:
    row = base.filter(pl.col("itemset") == list(motif))
    print(f"planted motif {motif}: planted in {planted_rows} rows, mined count {round(row['support'][0] * N)}")

# %% [markdown]
# Each planted 5-item motif is found, with a mined count at least as large as the number of rows it was planted in.
# Background rows can only add occurrences, never remove them.

# %% [markdown]
# ## 2. `min_support`: the main dial
#
# A lower threshold admits more single items, and the number of candidate pairs grows with the square of that.
# The next cell mines at four thresholds and records, per run, the frequent items, the level-2 candidates
# (from `level_callback`), the number of itemsets and the deepest level.

# %%
rows = []
for s in [0.02, 0.01, 0.005, 0.003]:
    levels = []
    res = apriori(smoke, min_support=s, sparse=False, level_callback=lambda *a: levels.append(a))
    lv = {k: (c, f) for k, c, f, _ in levels}
    rows.append(
        dict(min_support=s, min_count=_min_count(s, N), frequent_items=lv[1][1], k2_candidates=lv.get(2, (0, 0))[0],
             itemsets=res.height, max_k=max(by_length(res)))
    )
pl.DataFrame(rows)

# %% [markdown]
# Each step down in `min_support` raises the number of frequent items, candidates and itemsets.
# The level-2 candidate count is `n*(n-1)/2` for `n` frequent items, so it grows fastest.
# The deepest level grows more slowly and can stay the same.

# %% [markdown]
# ## 3. `max_length`: stop early
#
# `max_length=K` stops the loop after level K.
# The next cell checks that the result is exactly the baseline restricted to itemsets of length at most 2.

# %%
short = apriori(smoke, min_support=0.01, sparse=False, max_length=2)
expected = base.filter(pl.col("itemset").list.len() <= 2)
assert set(map(tuple, short["itemset"].to_list())) == set(map(tuple, expected["itemset"].to_list()))
print("max_length=2:", short.height, "itemsets, equal to the baseline cut at length 2")

# %% [markdown]
# `max_length` never changes the support of the itemsets it keeps; it only drops the longer ones.

# %% [markdown]
# ## 4. `prune_equal_support`: free-sets instead of the full lattice
#
# With `prune_equal_support=True` the miner returns only **free-sets** (also called generators):
# itemsets with no proper subset of the same support.
# Every omitted frequent itemset has the same support as one of its subsets, so the free-sets are a smaller summary.
# This is not closed-itemset mining; ET-Miner has no closed-itemset option.
# The next cell compares the two results and shows one omitted itemset next to the subset that shares its support.

# %%
free = apriori(smoke, min_support=0.01, sparse=False, prune_equal_support=True)
print("full lattice:", base.height, by_length(base))
print("free-sets:   ", free.height, by_length(free))

support_of = {tuple(s): v for s, v in zip(base["itemset"].to_list(), base["support"].to_list())}
free_keys = set(map(tuple, free["itemset"].to_list()))
omitted = sorted((s for s in support_of if s not in free_keys), key=len)
example = omitted[0]
twin = next(
    sub for j in range(len(example)) if (sub := example[:j] + example[j + 1 :]) and support_of.get(sub) == support_of[example]
)
print(f"omitted {example} support={support_of[example]:.5f}; its subset {twin} has the same support")

# %% [markdown]
# The free-set result is smaller and stops at a lower level, because the supersets of the planted motifs
# add no support information beyond their subsets.

# %% [markdown]
# ## 5. `use_generator_pruning`: same answer, fewer counts
#
# `use_generator_pruning=True` infers the support of some candidates from their subsets instead of counting them
# (the Pascal rule, CPU path only). It never changes the result.
# The library logs how many candidates it skipped at debug level; the next cell captures those lines
# with a temporary log sink and asserts the result equals the baseline.

# %%
captured = []
sink = logger.add(lambda m: captured.append(m.record["message"]), level="DEBUG",
                  filter=lambda r: "Generator pruning" in r["message"])
pascal = apriori(smoke, min_support=0.01, sparse=False, use_generator_pruning=True)
logger.remove(sink)
assert set(map(tuple, pascal["itemset"].to_list())) == set(support_of)
print("identical result:", pascal.height, "itemsets")
print("\n".join(captured) if captured else "no candidate could be inferred on this sample")

# %% [markdown]
# The result is identical. The log lines, if any, show how many candidates per level were inferred instead of counted.

# %% [markdown]
# ## 6. Where the time goes
#
# `level_callback` also reports the wall time of each level.
# The next cell prints the per-level time for the baseline run.

# %%
levels = []
t0 = time.perf_counter()
apriori(smoke, min_support=0.01, sparse=False, level_callback=lambda *a: levels.append(a))
total = time.perf_counter() - t0
print(hardware(), f"| {N:,} rows | min_support=0.01")
print(pl.DataFrame(levels, schema=["k", "candidates", "frequent", "ms"], orient="row").with_columns(pl.col("ms").round(1)))
print(f"total wall time {total:.2f} s")

# %% [markdown]
# Level 2 has by far the most candidates and takes the largest share of the level time.
# The rest of the wall time is spent outside the level loop: scanning the input, building the boolean matrix and
# assembling the result frame.

# %% [markdown]
# ## 7. Two data sizes
#
# The next cell generates the same preset with ten times as many rows (same seed and item distribution),
# and times the Polars tier on both sizes. It also prints the in-memory size of the boolean matrix.

# %%
big_spec = dataclasses.replace(PRESETS["smoke"], n_rows=10 * N, name="smoke_x10")
big, _ = generate_transactions(big_spec)
timings = []
for label, df in [("smoke", smoke), ("smoke x10", big)]:
    matrix, cols, n = build_boolean_matrix(df.lazy(), 0.01)
    t0 = time.perf_counter()
    res = apriori(df, min_support=0.01, sparse=False)
    timings.append(dict(data=label, rows=n, frequent_items=len(cols), matrix_mb=round(matrix.estimated_size() / 1e6, 1),
                        itemsets=res.height, seconds=round(time.perf_counter() - t0, 2)))
    del matrix
print(hardware())
pl.DataFrame(timings)

# %% [markdown]
# The boolean matrix grows in proportion to the rows, and the number of itemsets stays similar.
# The wall time grows too, by less than the row factor on this data. This notebook does not measure how the time
# splits between per-row and per-candidate work.

# %% [markdown]
# ## 8. Where Polars runs out of steam
#
# The next cell lowers `min_support` on the 60,000-row sample and times each run.
# Watch the level-2 candidate count and the wall time grow together.

# %%
steam = []
for s in [0.01, 0.005, 0.003, 0.002]:
    levels = []
    t0 = time.perf_counter()
    res = apriori(smoke, min_support=s, sparse=False, level_callback=lambda *a: levels.append(a))
    steam.append(dict(min_support=s, k2_candidates=levels[1][1], itemsets=res.height,
                      seconds=round(time.perf_counter() - t0, 2)))
print(hardware(), f"| {N:,} rows")
pl.DataFrame(steam)

# %% [markdown]
# The Polars tier builds one expression per candidate and scans one boolean column per item in it.
# Its cost grows with rows times candidates, and the level-2 candidate count grows quadratically as the threshold falls.
# The table shows the wall time rising with the level-2 candidate count.
# Tier 2 keeps the same algorithm and moves the counting, and optionally the whole loop, into compiled code.

# %% [markdown]
# ## Summary and next step
#
# - `min_support` sets the size of the answer; `max_length` cuts it; `prune_equal_support` summarizes it as free-sets;
#   `use_generator_pruning` saves counting without changing it.
# - Level 2 dominates the cost, and its candidate count grows quadratically as the threshold falls.
# - Polars cost grows with rows times candidates.
#
# Next: tier 2, starting with [01-build-and-install](../tier2-rust-pyo3/01-build-and-install.md).
