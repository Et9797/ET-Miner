# %% [markdown]
# # Tier 3.2: GPU-resident mining
#
# In GPU-resident mode the data goes to the GPU once, every level is mined there, and only the final itemsets and
# counts come back. This notebook shows what lives where in that mode, how to estimate its memory footprint,
# and how it differs from the default GPU route, which returns each level to the host.
#
# **What you will learn**
#
# - what `gpu_resident=True` changes in the code;
# - which arrays live in GPU memory, and how big they are for a given input;
# - how to inspect device memory from Python;
# - how much data the default route moves per level, compared with the resident route.
#
# **Prerequisites**: [01-gpu-setup-and-first-run](01-gpu-setup-and-first-run.ipynb).
# Cells marked *GPU* print a skip line without a CUDA device; the setup cell prints the hardware.

# %% [markdown]
# ## 0. Setup and device check

@@GPU_SETUP@@

# %% [markdown]
# ## 1. Two single-GPU routes
#
# With `use_gpu=True` and one GPU, `src/et_miner/core/apriori.py:apriori` builds the bit vectors on the device and
# then calls one of two miners in `src/et_miner/gpu/mining.py`:
#
# | Call | Miner | Between levels |
# |---|---|---|
# | `apriori(df, use_gpu=True)` | `_apriori_from_bitvecs` | each level's frequent itemsets come back to the host; the next level's candidate groups are built on the host and uploaded |
# | `apriori(df, use_gpu=True, gpu_resident=True)` | `_apriori_from_bitvecs_gpu_resident` | frequent itemsets stay in GPU memory and feed the next level there; only their number crosses to the host |
#
# The next cell prints the resident miner's own description and its final transfer.

# %%
show("src/et_miner/gpu/mining.py", r"^def _apriori_from_bitvecs_gpu_resident", 1)
show("src/et_miner/gpu/mining.py", r"Fully GPU-resident Apriori", 4)
show("src/et_miner/gpu/mining.py", r"=== END: single bulk transfer", 3)

# %% [markdown]
# At the end, `_build_results_from_gpu` copies each level's `(n, k)` int32 itemset array and its int64 count array
# to the host in one transfer per level, and maps column indices back to item ids.
# The same flag exists for single-GPU SON streaming (notebook 5). It does not exist on the multi-GPU row-split miner.

# %% [markdown]
# ## 2. What goes to the GPU: the frequent-item bit vectors
#
# Before any kernel runs, the host builds a CSR matrix of the **frequent** items only
# (`src/et_miner/core/matrix.py:_build_csr_from_transactions`), so infrequent items never reach the GPU.
# The GPU then turns that CSR into one bit vector per item (`src/et_miner/gpu/bitvec.py:_build_gpu_bitvec_matrix`).
# The first step runs on the CPU, so the next cell runs it on the shared sample and reports the size of the
# bit-vector matrix it implies.

# %%
import math

from et_miner.core.matrix import _build_csr_from_transactions

smoke = pl.read_parquet(DATA / "smoke.parquet")
N, S = smoke.height, 0.01
csr, col_to_item, n_rows = _build_csr_from_transactions(smoke.lazy(), S, "items")
n_words = math.ceil(n_rows / 64)
print(f"CSR on the host: {csr.shape[0]:,} rows x {csr.shape[1]} frequent items, {csr.nnz:,} non-zeros")
print(f"bit vectors on the GPU: ({csr.shape[1]}, {n_words}) uint64 = {csr.shape[1] * n_words * 8:,} bytes")

# %% [markdown]
# Only the 118 frequent items get a bit vector. The matrix costs `n_frequent_items * ceil(n_rows / 64) * 8` bytes and
# stays on the GPU for the whole run in both routes.

# %% [markdown]
# ## 3. Estimating the resident footprint
#
# The resident route keeps, besides the bit vectors, every level's frequent itemsets `(n_k, k)` as int32 and their
# counts as int64 (the dtypes named in `_build_results_from_gpu`). Level sizes are not known before mining, but any
# CPU tier gives them for a sample. The next cell uses tier 1's `level_callback` on the shared sample and adds up the
# resident bytes. It is an estimate of the arrays the code keeps, not a measurement: kernels also use temporary buffers.

# %%
levels = []
apriori(smoke, min_support=S, sparse=False, level_callback=lambda *a: levels.append(a))
est = pl.DataFrame(
    [dict(k=k, candidates=c, frequent=f, itemset_bytes=f * k * 4, count_bytes=f * 8) for k, c, f, _ in levels]
)
bitvec_bytes = csr.shape[1] * n_words * 8
resident = est["itemset_bytes"].sum() + est["count_bytes"].sum()
print(est)
print(f"bit vectors {bitvec_bytes:,} B + resident results {resident:,} B = {bitvec_bytes + resident:,} B")

# %% [markdown]
# On this sample the bit vectors dominate; the resident results are a few kilobytes.
# At scale the balance can flip: every frequent K-itemset holds `12 + 4k` bytes, and all levels are kept until the end.
# Each resident level also has a result capacity (`max_results`, 10,000,000 itemsets by default in
# `src/et_miner/gpu/kernels/gpu_resident.py`); a level that exceeds it raises an error instead of dropping itemsets.

# %% [markdown]
# ## 4. What each route moves between host and GPU per level
#
# In the default route, each level's survivors go to the host, and the host builds the next level's prefix groups
# and uploads them. In the resident route, nothing but loop control crosses per level. The next cell models the
# per-level traffic of the default route from the level sizes above, counting survivors down (itemsets and counts)
# and the prefix-group arrays up (about one int32 per frequent itemset); it is a model built from the sizes, not a
# measurement.

# %%
model = est.with_columns(
    down_bytes=pl.col("itemset_bytes") + pl.col("count_bytes"),
    up_bytes=pl.col("frequent") * 4,
).select("k", "frequent", "down_bytes", "up_bytes")
print(model)
print("default route, per-level traffic:", model["down_bytes"].sum() + model["up_bytes"].sum(), "bytes in",
      model.height, "round trips")
print("resident route: the same survivors cross once, at the end, plus a few bytes of loop control per level")

# %% [markdown]
# The bytes are small on this sample; what the resident route saves here is the synchronization of one host round
# trip per level and the host-side candidate grouping. The saving grows with the number of levels and with the
# size of each level.

# %% [markdown]
# ## 5. Running both routes and inspecting device memory
#
# The next cell (*GPU*) runs both routes on the shared sample, asserts they return the same itemsets and counts as
# tier 1, times them, and prints device memory from CuPy:
# `cupy.get_default_memory_pool().used_bytes()` (what the library holds through CuPy)
# and `cupy.cuda.Device().mem_info` (free and total bytes on the device).

# %%
import time


def counted(result):
    return {(tuple(s), round(v * N)) for s, v in zip(result["itemset"].to_list(), result["support"].to_list())}


tier1 = counted(apriori(smoke, min_support=S, sparse=False))
if SKIP_GPU:
    print("skipped: no CUDA device")
else:
    import cupy as cp

    pool = cp.get_default_memory_pool()
    apriori(smoke, min_support=S, use_gpu=True)  # warm-up: kernels compile on first use
    rows = []
    for resident in (False, True):
        pool.free_all_blocks()
        t0 = time.perf_counter()
        res = apriori(smoke, min_support=S, use_gpu=True, gpu_resident=resident)
        cp.cuda.Device().synchronize()
        seconds = time.perf_counter() - t0
        assert counted(res) == tier1
        free, total = cp.cuda.Device().mem_info
        rows.append(dict(gpu_resident=resident, seconds=round(seconds, 4), pool_used_bytes=pool.used_bytes(),
                         pool_total_bytes=pool.total_bytes(), device_free_gb=round(free / 1e9, 2)))
    print(hardware())
    print(pl.DataFrame(rows))

# %% [markdown]
# Both routes give the tier-1 answer. The timings and pool sizes printed above are for this sample and device only.

# %% [markdown]
# ## 6. When the flag is refused
#
# `gpu_resident=True` is honored by the bit-vector miner and by single-GPU SON streaming (there only when the input
# spans more than one chunk; a single-chunk input falls back to `apriori` without the flag).
# `src/et_miner/core/apriori.py:_validate_route_support` rejects it everywhere else instead of ignoring it.
# These checks run before any GPU work, so the next cell shows them on the CPU.

# %%
toy = pl.read_parquet(DATA / "toy_8x6.parquet")
for kwargs in (dict(gpu_resident=True), dict(use_gpu=True, n_gpus=2, gpu_resident=True),
               dict(use_gpu=True, prune_equal_support=True, gpu_resident=True)):
    try:
        apriori(toy, min_support=0.25, **kwargs)
    except ValueError as err:
        print(f"{kwargs}\n  -> ValueError: {str(err)[:150]}...")

# %% [markdown]
# Without `use_gpu` the flag has no GPU route to act on; with several GPUs or free-set pruning the call goes to the
# row-split miner, which has no resident mode.

# %% [markdown]
# ## Summary and next step
#
# - `gpu_resident=True` keeps every level's itemsets and counts in GPU memory and copies them back once at the end.
# - The footprint is the frequent-item bit vectors plus `12 + 4k` bytes per frequent K-itemset, kept for all levels.
# - The default route returns each level to the host and uploads the next level's groups.
# - The flag is refused, with an error, on the row-split miner and on multi-GPU streaming.
#
# **Checklist when you run this notebook on a GPU**
#
# - [ ] Section 5 prints a table with two rows and no assertion error.
# - [ ] `pool_used_bytes` is of the order of the estimate in section 3 (kernel buffers come on top).
#
# Next: [03-shared-memory-and-tiled-routes](03-shared-memory-and-tiled-routes.ipynb).
