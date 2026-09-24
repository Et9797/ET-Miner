# %% [markdown]
# # Tier 3.4: The sparse-dense crossover
#
# A dense bit vector costs the same whatever the support of the itemset; a sparse list of row ids costs in
# proportion to the support. Early levels have high support and suit bit vectors; deeper levels and lower thresholds
# have low support and suit row-id lists. The GPU miners can switch from one to the other once per run.
# This notebook shows the two layouts, the exact switch rule in the code, how to force or automate it,
# and how to see the switch happen.
#
# **What you will learn**
#
# - the dense bit-vector and sparse CSR tidset layouts, and their byte costs;
# - the exact crossover rule in `src/et_miner/gpu/density.py`;
# - what `sparse_from_k` accepts and what each value does;
# - which log lines show the switch;
# - what the fused kernels on each side do.
#
# **Prerequisites**: [03-shared-memory-and-tiled-routes](03-shared-memory-and-tiled-routes.ipynb).
# Cells marked *GPU* print a skip line without a CUDA device; the setup cell prints the hardware.
#
# A note on names: `sparse_from_k` (this notebook, GPU only) is unrelated to the CPU parameter `sparse=`,
# which chooses between Polars and SciPy counting in tiers 1 and 2. See [concepts.md](../concepts.md#6-routes-and-tiers).

# %% [markdown]
# ## 0. Setup and device check

@@GPU_SETUP@@

# %% [markdown]
# ## 1. Two layouts for the rows of an itemset
#
# The GPU miners hold, for every frequent itemset of the current level, the set of rows that contain it:
#
# - **dense bit vector**: `ceil(n_rows / 64)` 64-bit words, one bit per row; `n_rows / 8` bytes whatever the count;
# - **sparse CSR tidset**: the sorted row ids (transaction ids) as int32; `4 * count` bytes.
#
# The two cost the same when `4 * count = n_rows / 8`, that is at `count = n_rows / 32`, a support of 1/32 (about 3.1%).
# The module docstring states the same rule:

# %%
show("src/et_miner/gpu/density.py", r"^The GPU miners hold", 22)

# %% [markdown]
# The rule is about bytes, and the docstring notes that kernel work scales the same way: dense kernels sweep every
# word, sparse kernels touch only the stored ids.

# %% [markdown]
# ## 2. The bytes per level on the shared sample
#
# The next cell mines the sample on the CPU and computes, per level, the bytes each layout would need for that
# level's frequent itemsets, and the mean count that the crossover rule compares with `n_rows / 32`.

# %%
smoke = pl.read_parquet(DATA / "smoke.parquet")
N, S = smoke.height, 0.01
base = apriori(smoke, min_support=S, sparse=False)
per_level = (
    base.with_columns(k=pl.col("itemset").list.len(), count=(pl.col("support") * N).round().cast(pl.Int64))
    .group_by("k")
    .agg(frequent=pl.len(), mean_count=pl.col("count").mean().round(1), total_count=pl.col("count").sum())
    .sort("k")
    .with_columns(dense_bytes=pl.col("frequent") * (N // 8), sparse_bytes=pl.col("total_count") * 4)
)
print(f"n_rows / 32 = {N / 32}")
per_level

# %% [markdown]
# Level 1 has a mean count above `n_rows / 32`, so bit vectors are smaller there. From level 2 on the mean count is
# below it, and the row-id lists are the smaller layout.

# %% [markdown]
# ## 3. The exact rule and the `sparse_from_k` parameter
#
# `apriori(..., use_gpu=True, sparse_from_k=...)` accepts three kinds of value (`validate_sparse_from_k`):
#
# - `None` (default): never switch;
# - an int K: switch at level K, but never before level 3 (`MIN_SPARSE_K`);
# - `"auto"`: switch at the first level K >= 3 whose previous level has a mean count below `n_rows / 32`.
#
# The switch is one-way: the miners free the bit vectors when they convert. The decision function:

# %%
show("src/et_miner/gpu/density.py", r"^def should_transition_to_sparse", 36)

# %% [markdown]
# The function is plain Python, so the next cell asks it, level by level, what it would decide on the sample for four
# settings, with the mean counts from section 2. `sticky` mimics the callers: once a level switches, later levels stay sparse.

# %%
from et_miner.gpu.density import should_transition_to_sparse, validate_sparse_from_k

mean_counts = dict(zip(per_level["k"].to_list(), per_level["mean_count"].to_list()))
decisions = []
for setting in (None, 3, 5, "auto"):
    sticky, row = False, {"sparse_from_k": str(setting)}
    for k in range(2, max(mean_counts) + 1):
        sticky = sticky or should_transition_to_sparse(setting, k, n_transactions=N, mean_count=mean_counts[k - 1])
        row[f"K={k}"] = "sparse" if sticky else "dense"
    decisions.append(row)
pl.DataFrame(decisions)

# %% [markdown]
# With `"auto"` the switch happens at level 3, because level 2's mean count is below `n_rows / 32`.
# Level 2 is never sparse: the code floors every setting to level 3.
#
# The next cell shows the validation of the parameter, which also runs before any GPU work.

# %%
for value in (None, 4, "auto", "sometimes", True):
    try:
        print(f"{value!r:12} -> accepted as {validate_sparse_from_k(value)!r}")
    except (TypeError, ValueError) as err:
        print(f"{value!r:12} -> {type(err).__name__}: {err}")

# %% [markdown]
# `True` is refused with a `TypeError`, so a boolean cannot be mistaken for the level 1.

# %% [markdown]
# ## 4. What is fused on each side
#
# On the dense side, the single-GPU default route counts K >= 3 with a **fully fused** kernel: candidate generation,
# counting and filtering happen in one kernel, from prefix groups built on the host
# (`src/et_miner/gpu/dispatch.py:dispatch_k3plus_fused`). On the sparse side, candidates are also enumerated inside the
# kernel from the resident group arrays, one warp per candidate, and a second pass writes the survivors' row ids
# (`src/et_miner/gpu/kernels/csr_warp.py`). The next cell prints both descriptions from the source.

# %%
show("src/et_miner/gpu/dispatch.py", r"^def dispatch_k3plus_fused", 7)
show("src/et_miner/gpu/kernels/csr_warp.py", r"Warp-cooperative CSR tidset", 14)

# %% [markdown]
# Neither side builds a list of candidates in memory: the kernels derive each candidate from its index.

# %% [markdown]
# ## 5. Observing the switch
#
# The miners log the switch at INFO level as `DENSITY TRANSITION at K=...`, and each sparse level as
# `K=...: ... candidates (CSR sparse mode)`. The next cell (*GPU*) runs the sample with three settings,
# captures those lines with a log sink, asserts every result equals tier 1, and times each run after a warm-up.

# %%
import time

from loguru import logger


def counted(result):
    return {(tuple(s), round(v * N)) for s, v in zip(result["itemset"].to_list(), result["support"].to_list())}


tier1 = counted(base)
if SKIP_GPU:
    print("skipped: no CUDA device")
else:
    apriori(smoke, min_support=S, use_gpu=True)  # warm-up: kernels compile on first use
    print(hardware())
    for setting in (None, 3, "auto"):
        lines = []
        sink = logger.add(lambda m: lines.append(m.record["message"].strip()), level="INFO",
                          filter=lambda r: "DENSITY TRANSITION" in r["message"] or "CSR sparse mode" in r["message"])
        t0 = time.perf_counter()
        result = apriori(smoke, min_support=S, use_gpu=True, sparse_from_k=setting)
        seconds = time.perf_counter() - t0
        logger.remove(sink)
        assert counted(result) == tier1
        print(f"sparse_from_k={setting!r}: {seconds:.4f} s, result == tier 1")
        for line in lines:
            print("   ", line)

# %% [markdown]
# On a GPU the `None` run prints no transition line, and the other two print the switch at level 3 followed by one
# line per sparse level. On a sample this small the timings mostly show fixed costs.

# %% [markdown]
# ## Summary and next step
#
# - Dense bit vectors cost `n_rows / 8` bytes per itemset; CSR tidsets cost `4 * count`.
# - They cross at a mean count of `n_rows / 32`; `sparse_from_k="auto"` switches at the first level K >= 3 whose
#   previous level falls below it, an int forces the level, `None` never switches.
# - The switch is one-way and logged as `DENSITY TRANSITION at K=...`.
#
# **Checklist when you run this notebook on a GPU**
#
# - [ ] Section 5 prints three runs, each ending `result == tier 1`.
# - [ ] The `3` and `"auto"` runs log `DENSITY TRANSITION at K=3`; the `None` run logs none.
#
# Next: [05-son-streaming](05-son-streaming.ipynb).
