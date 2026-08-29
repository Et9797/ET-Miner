
// NOTE: Uses long long for index calculation to avoid overflow with large arrays (>2B elements)
extern "C" __global__
void count_itemset_fused(
    const unsigned long long* __restrict__ bitvecs,  // [n_cols, n_u64s]
    const int* __restrict__ items,                    // items in this itemset
    const int n_items,
    const long long n_u64s,                           // int64: bitvec index overflows int32 at >536M columns (item * n_u64s > 2^31)
    const int n_cols,
    unsigned long long* __restrict__ count            // single output
) {
    // Each thread handles multiple u64s for better occupancy
    unsigned long long local_count = 0;

    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;  // int64: bitvec index overflows int32 at >536M columns (item * n_u64s > 2^31)
         idx < n_u64s;
         idx += blockDim.x * gridDim.x) {

        // Start with all 1s
        unsigned long long result = ~0ULL;

        // AND all bitvecs for items in itemset
        for (int i = 0; i < n_items; i++) {
            int item = items[i];
            // Use long long for index calculation to prevent overflow
            long long bitvec_idx = (long long)item * n_u64s + idx;
            result &= bitvecs[bitvec_idx];
        }

        // Popcount using GPU intrinsic
        local_count += __popcll(result);
    }

    // Atomic add to global count
    if (local_count > 0) {
        atomicAdd(count, local_count);
    }
}

// Batch version: count multiple itemsets in one kernel launch
// NOTE: Uses long long for index calculation to avoid overflow with large arrays (>2B elements)
extern "C" __global__
void count_itemsets_batch(
    const unsigned long long* __restrict__ bitvecs,   // [n_cols, n_u64s]
    const int* __restrict__ all_items,                // flattened items
    const long long* __restrict__ offsets,            // [n_itemsets + 1] — FIXED: int* -> long long* for >2B elements
    const long long n_u64s,                           // int64: bitvec index overflows int32 at >536M columns (item * n_u64s > 2^31) for large arrays
    const int n_cols,
    const long long n_itemsets,
    unsigned long long* __restrict__ counts           // [n_itemsets]
) {
    long long itemset_idx = blockIdx.y;
    if (itemset_idx >= n_itemsets) return;

    long long start = offsets[itemset_idx];    // int64: group offsets exceed 2^31 at 536M+ groups
    long long end = offsets[itemset_idx + 1];  // int64: group offsets exceed 2^31 at 536M+ groups

    unsigned long long local_count = 0;

    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;  // int64: bitvec index overflows int32 at >536M columns (item * n_u64s > 2^31)
         idx < n_u64s;
         idx += blockDim.x * gridDim.x) {

        unsigned long long result = ~0ULL;

        for (long long i = start; i < end; i++) {
            int item = all_items[i];
            // Use long long for index calculation to prevent overflow
            long long bitvec_idx = (long long)item * n_u64s + idx;
            result &= bitvecs[bitvec_idx];
        }

        local_count += __popcll(result);
    }

    if (local_count > 0) {
        atomicAdd(&counts[itemset_idx], local_count);
    }
}
