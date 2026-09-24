# %% [markdown]
# # Tier 3.5: SON streaming
#
# The SON algorithm (Savasere, Omiecinski and Navathe, 1995) mines data larger than memory in two passes over row
# chunks: mine each chunk locally, take the union of the local results as candidates, then count every candidate
# over all chunks. ET-Miner implements it in `src/et_miner/streaming/son.py:apriori_streaming`, reached through
# `apriori(..., streaming=True)`. Both passes run on the CPU unless you ask for the GPU, so this notebook's
# worked examples execute everywhere.
#
# **What you will learn**
#
# - the two passes, on the hand-made table, step by step;
# - why the result is exact, with no false negatives and no false positives;
# - the local threshold ET-Miner uses, and why it is lower than the global one;
# - how the chunk size is chosen, and what that means for memory;
# - where the GPU fits in.
#
# **Prerequisites**: [tier 1, notebook 2](../tier1-polars/02-apriori-step-by-step.ipynb) for the level loop,
# and [04-sparse-dense-crossover](04-sparse-dense-crossover.ipynb). Cells marked *GPU* print a skip line without a CUDA device.

# %% [markdown]
# ## 0. Setup and device check

@@GPU_SETUP@@

# %% [markdown]
# ## 1. The two passes, by hand
#
# ET-Miner's pass 1 mines each chunk at a **local** support of `local_support_factor * min_support`
# (default factor 0.9). Pass 2 counts every candidate from pass 1 over all chunks and keeps those whose total
# reaches the global minimum count.
#
# The next cell splits the 8-row table into two chunks of 4 rows and runs pass 1 with the library's own miner on
# each chunk, at `min_support=0.25`.

# %%
from et_miner.core.result import _min_count

toy = pl.read_parquet(DATA / "toy_8x6.parquet")
MIN_SUPPORT, FACTOR, CHUNK = 0.25, 0.9, 4
local_support = MIN_SUPPORT * FACTOR
candidates = set()
for start in range(0, toy.height, CHUNK):
    chunk = toy.slice(start, CHUNK)
    local = apriori(chunk, min_support=local_support, sparse=False)
    found = {tuple(s) for s in local["itemset"].to_list()}
    candidates |= found
    print(f"chunk rows {start}-{start + chunk.height - 1}: local min_count {_min_count(local_support, chunk.height)}, "
          f"{len(found)} locally frequent")
print("union of local results:", len(candidates), "candidates")

# %% [markdown]
# Each chunk of 4 rows needs a count of 1 at the local support (`ceil(0.225 * 4) = 1`), so every itemset seen in
# a chunk becomes a candidate. Small chunks make generous candidate sets; that costs time, not correctness.

# %% [markdown]
# Pass 2 counts each candidate over the whole table and applies the global minimum count.

# %%
rows = [set(r) for r in toy["items"].to_list()]
global_min = _min_count(MIN_SUPPORT, toy.height)
counts = {c: sum(1 for r in rows if set(c) <= r) for c in candidates}
son = {c: n for c, n in counts.items() if n >= global_min}
print(f"global min_count {global_min}: {len(son)} of {len(candidates)} candidates survive")
direct = apriori(toy, min_support=MIN_SUPPORT, sparse=False)
direct_set = {tuple(s): round(v * toy.height) for s, v in zip(direct["itemset"].to_list(), direct["support"].to_list())}
assert son == direct_set
print("two-pass result == one-pass result:", sorted(son.items(), key=lambda kv: (len(kv[0]), kv[0])))

# %% [markdown]
# The two passes return exactly the 13 itemsets and counts of the one-pass miner.

# %% [markdown]
# ## 2. Why it is exact
#
# **No false positives.** Pass 2 counts every candidate over all rows and applies the global threshold, so every
# returned itemset is truly frequent with its true count.
#
# **No false negatives.** Take an itemset with global count `c >= ceil(s * N)`, and chunks of `n_1, ..., n_m` rows
# with local counts `c_1, ..., c_m`. Suppose it is locally infrequent everywhere at local support `f * s` (`f <= 1`):
# then `c_i < ceil(f * s * n_i)`, so `c_i <= ceil(f * s * n_i) - 1 < f * s * n_i` for every chunk. Summing gives
# `c < f * s * N <= s * N <= ceil(s * N)`, a contradiction. So every frequent itemset is locally frequent in at least
# one chunk, and pass 1 proposes it.
#
# The argument needs the local support to be at most `s`. The code computes it as the float product
# `min_support * local_support_factor` (for example `0.01 * 0.9` is `0.009000000000000001`) and then applies the same
# exact `ceil` rule to each chunk; that float is still at most `s` for any factor `<= 1`, so the argument holds.
# It also assumes every chunk is mined: `apriori_streaming` skips a chunk whose matrix cannot be built and logs a
# warning, so check the log for `Chunk ... failed` on real data.
# The next cell checks the same claim on the shared sample, with the library's streaming entry point.

# %%
smoke = pl.read_parquet(DATA / "smoke.parquet")
N, S = smoke.height, 0.01


def counted(result):
    return {(tuple(s), round(v * N)) for s, v in zip(result["itemset"].to_list(), result["support"].to_list())}


tier1 = counted(apriori(smoke, min_support=S, sparse=False))
events = []
streamed = apriori(smoke, min_support=S, streaming=True, chunk_size=15_000, sparse=False,
                   progress_callback=lambda phase, i, n, m: events.append((phase, i, n, m.get("candidates"))))
assert counted(streamed) == tier1
print("streaming, 4 chunks of 15,000 rows == tier 1:", len(tier1), "itemsets")
for phase, i, n, cands in events:
    print(f"  {phase} chunk {i + 1}/{n}" + (f": {cands} candidates so far" if cands is not None else ""))

# %% [markdown]
# The streamed result equals the one-pass result on the sample. The progress callback shows the candidate set
# growing chunk by chunk in pass 1, then one pass-2 event per chunk.

# %% [markdown]
# ## 3. How many candidates the lower local threshold costs
#
# `local_support_factor` is a parameter of `apriori_streaming`. The next cell runs pass 1 by hand on the four
# chunks of the sample with three factors and counts the union of local results each time.

# %%
for factor in (1.0, 0.9, 0.7):
    union = set()
    for start in range(0, N, 15_000):
        local = apriori(smoke.slice(start, 15_000), min_support=S * factor, sparse=False)
        union |= {tuple(x) for x in local["itemset"].to_list()}
    assert {itemset for itemset, _ in tier1} <= union  # no false negatives
    print(f"local_support_factor={factor}: {len(union):5d} candidates for pass 2 (final answer: {len(tier1)} itemsets)")

# %% [markdown]
# A lower factor makes a larger candidate set for pass 2. The assertion checks that every setting contains all
# final itemsets, as the proof requires. Section 2 shows that a factor of 1.0 is already exact, so the default 0.9
# costs extra candidates without being needed for exactness.

# %% [markdown]
# ## 4. Chunk size, memory, and what "larger than memory" means here
#
# Pass 1 builds one chunk's boolean matrix at a time (`n_chunk_rows x n_local_frequent_items` booleans), and pass 2
# does the same for the candidate items. So peak memory follows the chunk size, not the total row count.
# The chunk size is set in one of two ways:
#
# - `chunk_size=` rows per chunk (default 10,000,000 through `apriori`, 40,000,000 when calling `apriori_streaming` directly);
# - `memory_budget_gb=`, which overrides it through `src/et_miner/streaming/son.py:_estimate_chunk_size_from_memory`.
#
# If the whole input fits in one chunk, `apriori_streaming` skips SON and runs the ordinary miner.
# The next cell shows the budget rule and the numbers it gives.

# %%
from et_miner.streaming.son import _estimate_chunk_size_from_memory

show("src/et_miner/streaming/son.py", r"# Boolean matrix: n_transactions", 9)
for budget in (0.5, 4, 32, 256):
    print(f"memory_budget_gb={budget:>5}: chunk_size = {_estimate_chunk_size_from_memory(budget):,} rows")

# %% [markdown]
# The rule assumes 1,000 frequent items at one bit each, doubles it for working space, and clamps the result to
# 100,000 to 100,000,000 rows. It does not look at your data, so check the real item count before relying on it.
#
# Chunks are row slices of a `LazyFrame`, so the input can be a `pl.scan_parquet(...)` over files larger than RAM;
# only one chunk is materialized at a time in each pass. The input is scanned again in each pass.

# %% [markdown]
# ## 5. Where the GPU fits
#
# `apriori(..., streaming=True, use_gpu=True)` counts inside each chunk on the GPU, and `gpu_resident=True` mines each
# chunk with the GPU-resident miner of notebook 2 and counts pass 2 with a GPU kernel. Only one chunk's data is on the
# device at a time, which is how the input can exceed GPU memory. With `n_gpus > 1`, `streaming=True` calls
# `src/et_miner/streaming/multi_gpu.py:apriori_streaming_multi_gpu`, which processes chunks in waves of one chunk per GPU.
# The next cell (*GPU*) runs the single-GPU resident variant on the sample.

# %%
if SKIP_GPU:
    print("skipped: no CUDA device")
else:
    gpu_streamed = apriori(smoke, min_support=S, streaming=True, chunk_size=15_000, gpu_resident=True, use_gpu=True)
    assert counted(gpu_streamed) == tier1
    print("GPU-resident SON, 4 chunks == tier 1:", len(tier1), "itemsets")

# %% [markdown]
# The streaming paths do not implement `prune_equal_support`; `apriori` refuses that combination with a `ValueError`
# rather than return the full lattice. The next cell shows it.

# %%
try:
    apriori(smoke, min_support=S, streaming=True, chunk_size=15_000, prune_equal_support=True)
except ValueError as err:
    print("ValueError:", str(err)[:160], "...")

# %% [markdown]
# ## Summary and next step
#
# - Pass 1 mines every chunk at `local_support_factor * min_support`; pass 2 counts the union over all rows.
# - The result is exact for any factor `<= 1`; a lower factor only adds candidates.
# - Memory follows the chunk size; `memory_budget_gb` derives a chunk size from a fixed rule.
# - The GPU can do the per-chunk work, one chunk at a time.
#
# **Checklist when you run this notebook on a GPU**
#
# - [ ] Section 5 prints `GPU-resident SON, 4 chunks == tier 1`.
#
# Reference: A. Savasere, E. Omiecinski, S. Navathe, "An Efficient Algorithm for Mining Association Rules in Large
# Databases", Proceedings of the 21st VLDB Conference, 1995, pp. 432-444.
#
# Next: [06-multi-gpu](06-multi-gpu.ipynb).
