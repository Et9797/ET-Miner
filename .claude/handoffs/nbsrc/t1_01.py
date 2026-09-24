# %% [markdown]
# # Tier 1.1: The problem and the data
#
# ET-Miner finds sets of items that occur together in many rows of a table.
# This notebook defines that problem on a table small enough to check by hand,
# shows the input format the library expects, and shows how the library turns
# that input into the representation it counts on.
#
# **What you will learn**
#
# - what a transaction, an item, an itemset and its support are;
# - the input format of `et_miner.apriori` (one list column per row);
# - how `src/et_miner/core/matrix.py:build_boolean_matrix` turns the rows into a boolean matrix;
# - how a support fraction becomes an integer minimum count;
# - the shared sample that tiers 1, 2 and 3 all mine.
#
# **Prerequisites**
#
# - Python and basic Polars.
# - The environment from [the docs README](../README.md#3-environment-setup).
# - Terms are defined in [concepts.md](../concepts.md).

# %% [markdown]
# ## 0. Setup
#
# The cell below imports the library and prints the versions this notebook ran with.
# `LOGURU_LEVEL` hides the library's debug log lines so the outputs stay readable.

# %%
import os
import warnings

os.environ.setdefault("LOGURU_LEVEL", "WARNING")
warnings.filterwarnings("ignore", message="IProgress not found")

import json
from pathlib import Path

import polars as pl

import et_miner
from et_miner import apriori

DATA = Path("../data")
print("et_miner", et_miner.__version__, "| polars", pl.__version__)

# %% [markdown]
# ## 1. A co-occurrence pattern
#
# A **transaction** (or row) is a set of **items**, for example the products in one shopping basket.
# An **itemset** is any set of items.
# Its **support** is the fraction of transactions that contain every item of the set.
# An itemset is **frequent** when its support reaches a threshold called `min_support`.
# The task is to list every frequent itemset and its support.
#
# The next cell builds an 8-row table with 6 items (`a` to `f`) by hand.
# Look at the rows: you will count items in them in the sections below.

# %%
rows = [
    ["a", "b", "c"],
    ["a", "b"],
    ["a", "c", "d"],
    ["b", "c"],
    ["a", "b", "c", "e"],
    ["a", "b", "d"],
    ["c", "e", "f"],
    ["a", "b", "c", "f"],
]
toy = pl.DataFrame({"items": rows})
toy

# %% [markdown]
# The table has one column, `items`, and each cell holds a list.
# That is the whole input format.

# %% [markdown]
# ## 2. The input format
#
# `apriori` takes a Polars `DataFrame` or `LazyFrame` with one list column.
# The column is called `items` by default; `item_col=` selects another name.
# Items can be integers or strings on the CPU routes; the GPU routes need integer items (tier 3, notebook 1).
# The same table is stored in `docs/data/toy_8x6.parquet`, written by `docs/data/make_samples.py`.
# The next cell reads it back and checks that it equals the table above.

# %%
toy_file = pl.read_parquet(DATA / "toy_8x6.parquet")
print(toy_file.schema)
assert toy_file.equals(toy)
print("file matches the hand-built table")

# %% [markdown]
# The file holds the same eight rows, typed `list[str]`.

# %% [markdown]
# ## 3. From a support fraction to a minimum count
#
# The library compares integer counts, not fractions.
# It converts `min_support` to a minimum count with
# `src/et_miner/core/result.py:_min_count`, which computes `ceil(min_support * n_rows)` exactly
# (it reads the float as the decimal it prints as, so `0.07` means exactly 7/100).
# The next cell prints the minimum count for a few thresholds on 8 rows.
# Look at how coarse the steps are.

# %%
from et_miner.core.result import _min_count

for s in [0.1, 0.125, 0.2, 0.25, 0.3, 0.375, 0.5]:
    print(f"min_support={s:<6} -> min_count={_min_count(s, len(toy))}")

# %% [markdown]
# On 8 rows every threshold in (0, 0.125] means "appears at least once", and 0.2 and 0.25 both mean 2.
# On small inputs a small fraction therefore keeps every item and every subset of every row.
# This notebook series uses `min_support=0.25`, which means a count of at least 2.

# %% [markdown]
# ## 4. From rows to a boolean matrix
#
# The Polars tier does not count on the lists directly.
# `build_boolean_matrix` first counts single items, keeps the frequent ones,
# and builds one boolean column per frequent item (one row per transaction).
# Column names are `i_0`, `i_1`, ... and `col_to_item` maps them back to item values.
# The next cell runs it at `min_support=0.25`.

# %%
from et_miner.core.matrix import build_boolean_matrix

matrix, col_to_item, n_rows = build_boolean_matrix(toy.lazy(), 0.25)
print("n_rows =", n_rows)
print("col_to_item =", col_to_item)
matrix

# %% [markdown]
# Every one of the six items is frequent at a count of 2, so the matrix has six columns.
# Each cell answers one question: does row *r* contain item *i*.
# Support counting is then a column AND followed by a sum.

# %% [markdown]
# ## 5. Counting one itemset by hand and with the matrix
#
# The next cell counts the itemset `{a, b}` in two ways:
# directly on the lists, and as `(col_a & col_b).sum()` on the matrix.
# Both numbers must agree.

# %%
item_to_col = {v: k for k, v in col_to_item.items()}
by_lists = sum(1 for r in rows if {"a", "b"} <= set(r))
by_matrix = matrix.select((pl.col(item_to_col["a"]) & pl.col(item_to_col["b"])).sum()).item()
print("rows containing {a, b}:", by_lists, "(lists) |", by_matrix, "(matrix)")
assert by_lists == by_matrix

# %% [markdown]
# Both methods count the same rows.
# The library's `src/et_miner/core/matrix.py:count_support_vectorized` builds this expression for many itemsets at once.

# %% [markdown]
# ## 6. The full answer on the toy table
#
# The next cell calls the real miner.
# `sparse=False` forces the Polars counting path (tier 1); see
# [concepts.md, section 6](../concepts.md#6-routes-and-tiers) for why this matters when the Rust extension is installed.
# The result has two columns, `itemset` and `support`.

# %%
result = apriori(toy, min_support=0.25, sparse=False)
result.with_columns(count=(pl.col("support") * n_rows).round().cast(pl.Int64)).sort(
    pl.col("itemset").list.len(), "itemset"
)

# %% [markdown]
# There are 13 frequent itemsets: six single items, six pairs and one triple.
# You can check each count against the table in section 1; notebook 2 derives them level by level.

# %% [markdown]
# ## 7. The shared sample
#
# The tutorials use one larger sample across all three tiers: the `smoke` preset of
# `src/et_miner/synthetic.py`, saved to `docs/data/smoke.parquet`.
# It has Zipf-distributed item popularity and three planted 5-item motifs.
# It is the same dataset that `tests/test_tier_equivalence.py` mines.
# The next cell loads it and prints its shape and parameters.

# %%
smoke = pl.read_parquet(DATA / "smoke.parquet")
meta = json.loads((DATA / "smoke.json").read_text())
print(smoke.shape, smoke.schema)
print({k: meta[k] for k in ["n_rows", "vocab_size", "row_len_mean", "motif_count", "motif_size", "min_support", "min_count"]})
lengths = smoke.select(pl.col("items").list.len().alias("row_length")).describe()
lengths

# %% [markdown]
# The sample has 60,000 rows over a vocabulary of 2,000 items, and the tutorials mine it at
# `min_support=0.01`, a minimum count of 600.

# %% [markdown]
# ## Summary and next step
#
# - The input is one list column; each row is a transaction.
# - `min_support` becomes an integer count `ceil(min_support * n_rows)`; on small inputs that count is coarse.
# - Tier 1 counts on a boolean matrix with one column per frequent item.
#
# Next: [02-apriori-step-by-step](02-apriori-step-by-step.ipynb) runs the algorithm one level at a time.
