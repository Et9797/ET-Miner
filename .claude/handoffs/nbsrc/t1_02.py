# %% [markdown]
# # Tier 1.2: Apriori step by step
#
# Apriori mines itemsets one size at a time.
# Level K builds candidate K-itemsets from the frequent (K-1)-itemsets, counts them, and keeps the frequent ones.
# This notebook runs that loop on the hand-made table so every number can be checked by counting.
#
# `et_miner.apriori` does not expose a per-level stepping API.
# It reports per-level totals through `level_callback`, but not the itemsets themselves.
# So section 3 reimplements the level loop in about 20 lines of Polars that mirror
# `src/et_miner/core/apriori.py:apriori` (CPU path), and section 5 checks it against the library.
#
# **What you will learn**
#
# - candidate generation: the prefix join and the subset prune;
# - support counting on the boolean matrix;
# - why the frontier shrinks and where the loop stops;
# - how to read, sort and filter the result frame.
#
# **Prerequisites**: [01-the-problem-and-the-data](01-the-problem-and-the-data.ipynb) and [concepts.md](../concepts.md).

# %% [markdown]
# ## 0. Setup

# %%
import os
import warnings

os.environ.setdefault("LOGURU_LEVEL", "WARNING")
warnings.filterwarnings("ignore", message="IProgress not found")

from pathlib import Path

import polars as pl

from et_miner import apriori
from et_miner.core.candidates import _generate_candidates
from et_miner.core.matrix import build_boolean_matrix
from et_miner.core.result import _min_count

DATA = Path("../data")
toy = pl.read_parquet(DATA / "toy_8x6.parquet")
MIN_SUPPORT = 0.25
print(toy.height, "rows | min_support", MIN_SUPPORT, "-> min_count", _min_count(MIN_SUPPORT, toy.height))

# %% [markdown]
# At `min_support=0.25` on 8 rows an itemset must occur in at least 2 rows.

# %% [markdown]
# ## 1. Level 1: count single items
#
# `build_boolean_matrix` already filters single items by the minimum count.
# The next cell shows the count of every item, frequent or not, next to the verdict.

# %%
item_counts = (
    toy.select(pl.col("items").explode(empty_as_null=True).alias("item"))
    .group_by("item")
    .len("count")
    .sort("item")
    .with_columns(frequent=pl.col("count") >= _min_count(MIN_SUPPORT, toy.height))
)
item_counts

# %% [markdown]
# `a`, `b` and `c` occur 6 times each; `d`, `e` and `f` occur exactly twice, which is the minimum.
# All six items are frequent, so the level-1 frontier has six entries.

# %% [markdown]
# ## 2. Candidate generation, shown once by hand
#
# Level K joins two frequent (K-1)-itemsets that share their first K-2 items (the **prefix join**).
# It then drops any candidate with an infrequent (K-1)-subset (the **subset prune**).
# The prune is safe because support can only fall as an itemset grows (downward closure).
# The library does this in `src/et_miner/core/candidates.py:_generate_candidates_simple`.
#
# The next cell applies both steps to the six frequent pairs that level 2 will find,
# written out here as data so the step can be read on its own.

# %%
frequent_pairs = [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("c", "e"), ("c", "f")]
joined = [
    p + (q[-1],) for i, p in enumerate(frequent_pairs) for q in frequent_pairs[i + 1 :] if p[:-1] == q[:-1]
]
pair_set = set(frequent_pairs)
for cand in joined:
    missing = [cand[:j] + cand[j + 1 :] for j in range(3) if cand[:j] + cand[j + 1 :] not in pair_set]
    print(cand, "-> keep" if not missing else f"-> prune, infrequent subset {missing}")

# %% [markdown]
# The join proposes four triples.
# Three of them have a pair that is not frequent, so only `(a, b, c)` has to be counted.
# The pruned triples are never counted at all; that is the saving Apriori is built on.

# %% [markdown]
# ## 3. The level loop in Polars
#
# The next cell is the whole algorithm in about 20 lines.
# It uses the library's boolean matrix and minimum count, and mirrors the library's loop:
# join, prune, count with `pl.all_horizontal(...).sum()`, filter on the integer count.
# It records one row of statistics per level and prints the frontier after every level.

# %%
def mine_levels(lf: pl.LazyFrame, min_support: float, verbose: bool = True):
    matrix, col_to_item, n = build_boolean_matrix(lf, min_support)
    min_count = _min_count(min_support, n)
    counts = matrix.sum().row(0, named=True)
    frontier = [(c,) for c in matrix.columns if counts[c] >= min_count]
    found = {f: counts[f[0]] for f in frontier}
    stats = [dict(k=1, joined=len(matrix.columns), counted=len(matrix.columns), frequent=len(frontier))]
    k = 2
    while len(frontier) >= k:
        prev = set(frontier)
        joined = [a + (b[-1],) for i, a in enumerate(frontier) for b in frontier[i + 1 :] if a[:-1] == b[:-1]]
        cands = [c for c in joined if all(c[:j] + c[j + 1 :] in prev for j in range(k))]
        if not cands:
            break
        row = matrix.select([pl.all_horizontal(list(c)).sum().alias("|".join(c)) for c in cands]).row(0)
        frontier = sorted(c for c, cnt in zip(cands, row) if cnt >= min_count)
        found.update({c: cnt for c, cnt in zip(cands, row) if cnt >= min_count})
        stats.append(dict(k=k, joined=len(joined), counted=len(cands), frequent=len(frontier)))
        if verbose:
            print(f"K={k} frontier:", [tuple(col_to_item[c] for c in f) for f in frontier])
        k += 1
    itemsets = {tuple(col_to_item[c] for c in f): cnt for f, cnt in found.items()}
    return itemsets, pl.DataFrame(stats), n


toy_itemsets, toy_stats, n_toy = mine_levels(toy.lazy(), MIN_SUPPORT)
toy_stats

# %% [markdown]
# Level 2 joins all 15 pairs of the six items and finds 6 frequent pairs.
# Level 3 joins four triples, counts only one after the prune, and it is frequent.
# A 4-itemset has four 3-subsets that must all be frequent, so level 4 needs at least four frequent triples; with one, the loop stops.

# %% [markdown]
# ## 4. The same candidates as the library
#
# The next cell feeds each level's frontier to the library's own
# `src/et_miner/core/candidates.py:_generate_candidates` and compares the candidate lists.
# Column names stand in for items, exactly as inside `apriori`.

# %%
matrix, col_to_item, _ = build_boolean_matrix(toy.lazy(), MIN_SUPPORT)
item_to_col = {v: k for k, v in col_to_item.items()}
by_level: dict[int, list] = {}
for itemset in toy_itemsets:
    by_level.setdefault(len(itemset), []).append(tuple(item_to_col[i] for i in itemset))
for k in sorted(by_level)[:-1]:
    lib = sorted(_generate_candidates(sorted(by_level[k]), k + 1))
    prev = set(by_level[k])
    ours = sorted(
        a + (b[-1],)
        for i, a in enumerate(sorted(by_level[k]))
        for b in sorted(by_level[k])[i + 1 :]
        if a[:-1] == b[:-1] and all((a + (b[-1],))[:j] + (a + (b[-1],))[j + 1 :] in prev for j in range(k + 1))
    )
    print(f"K={k + 1}: library {len(lib)} candidates, notebook {len(ours)} candidates, equal={lib == ours}")
    assert lib == ours

# %% [markdown]
# The notebook's join and prune produce the same candidate lists as the library at every level.

# %% [markdown]
# ## 5. The same result as the library
#
# The next cell runs `apriori(..., sparse=False)` with a `level_callback`.
# The callback receives `(k, n_candidates, n_frequent, duration_ms)` for every level.
# It then asserts that the itemsets and counts equal the notebook's.

# %%
levels = []
lib_result = apriori(toy, min_support=MIN_SUPPORT, sparse=False, level_callback=lambda *a: levels.append(a[:3]))
print("level_callback (k, n_candidates, n_frequent):", levels)
lib_itemsets = {
    tuple(s): round(sup * n_toy) for s, sup in zip(lib_result["itemset"].to_list(), lib_result["support"].to_list())
}
assert lib_itemsets == toy_itemsets
print(len(lib_itemsets), "itemsets, identical to the notebook loop")

# %% [markdown]
# The library reports the same candidate and frequent counts per level, and the same 13 itemsets.
# At K=1 its `n_candidates` is the number of frequent single items, because
# `build_boolean_matrix` drops infrequent items before the loop starts.

# %% [markdown]
# ## 6. The same check on the shared sample
#
# The hand-made table is tiny, so the next cell repeats the comparison on `smoke.parquet`
# (60,000 rows) at `min_support=0.01`.
# `verbose=False` hides the frontiers, which run to hundreds of itemsets.

# %%
smoke = pl.read_parquet(DATA / "smoke.parquet")
smoke_itemsets, smoke_stats, n_smoke = mine_levels(smoke.lazy(), 0.01, verbose=False)
smoke_lib = apriori(smoke, min_support=0.01, sparse=False)
smoke_lib_set = {
    tuple(s): round(sup * n_smoke) for s, sup in zip(smoke_lib["itemset"].to_list(), smoke_lib["support"].to_list())
}
assert smoke_lib_set == smoke_itemsets
print(len(smoke_itemsets), "itemsets, identical")
smoke_stats

# %% [markdown]
# The loop and the library agree on the larger sample too.
# The subset prune matters more here: compare the `joined` and `counted` columns per level.

# %% [markdown]
# ## 7. Reading the result
#
# `apriori` returns a `DataFrame` with two columns:
#
# - `itemset`: a list of items, always in ascending item order;
# - `support`: a float, the fraction of rows that contain the itemset.
#
# The order of rows is not part of the contract, so sort before you read.
# The next cell adds the length and the integer count, sorts, and applies two common filters.

# %%
readable = lib_result.with_columns(
    length=pl.col("itemset").list.len(),
    count=(pl.col("support") * n_toy).round().cast(pl.Int64),
).sort(["length", "support"], descending=[False, True])
print(readable)
print("itemsets of length >= 2:")
print(readable.filter(pl.col("length") >= 2))
print("itemsets that contain 'c':")
print(readable.filter(pl.col("itemset").list.contains("c")))

# %% [markdown]
# The count column is `support * n_rows`, rounded.
# Filtering by length and by membership are plain Polars expressions on the result.

# %% [markdown]
# ## Summary and next step
#
# - Each level joins frequent (K-1)-itemsets on a shared prefix, prunes by the subset test, counts, and filters.
# - The frontier is the list of frequent itemsets of the current level; it feeds the next level.
# - The notebook loop, the library's candidate generator and `apriori(sparse=False)` agree exactly.
#
# Next: [03-mining-a-realistic-sample](03-mining-a-realistic-sample.ipynb) looks at parameters and cost on the shared sample.
