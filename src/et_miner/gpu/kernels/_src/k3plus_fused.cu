
extern "C" __global__
void count_itemsets_fused_k3plus(
    const unsigned long long* __restrict__ bitvecs,    // [n_cols, n_u64s]
    const int* __restrict__ all_items,                 // flattened item indices
    const long long* __restrict__ offsets,             // [n_candidates + 1] — FIXED: int* -> long long* for >2B elements
    const long long n_u64s,
    const int n_cols,
    const long long n_candidates,                      // int64: candidate count exceeds 2^31 at K>=7
    const long long min_count,
    long long* __restrict__ result_indices,            // int64: candidate indices exceed 2^31 at K>=7
    long long* __restrict__ result_counts,             // output: support counts
    long long* __restrict__ n_results,                  // atomic counter (int64 for >2B results)
    const long long max_results,
    const long long candidate_offset                   // voor multi-GPU candidate splitting
) {
    // 2D grid -> lineaire candidate index (zelfde als k=2 pair index)
    long long cand_idx = (long long)blockIdx.y * (long long)gridDim.x
                       + (long long)blockIdx.x + candidate_offset;
    if (cand_idx >= n_candidates) return;

    long long start = offsets[cand_idx];      // int64: group offsets exceed 2^31 at 536M+ groups
    long long end = offsets[cand_idx + 1];    // int64: group offsets exceed 2^31 at 536M+ groups
    int k = end - start;

    // Cache item indices in shared memory (supports up to K=62)
    __shared__ int s_items[64];
    __shared__ unsigned long long warp_sums[8];

    if (threadIdx.x < k && threadIdx.x < 62) {
        s_items[threadIdx.x] = all_items[start + threadIdx.x];
    }
    __syncthreads();

    // AND + popcount over alle u64 words (grid-stride loop)
    unsigned long long local_count = 0;
    for (long long w = threadIdx.x; w < n_u64s; w += blockDim.x) {
        unsigned long long result = ~0ULL;
        for (int i = 0; i < k; i++) {
            result &= bitvecs[(long long)s_items[i] * n_u64s + w];
            if (__ballot_sync(__activemask(), result != 0ULL) == 0) { result = 0ULL; break; }
        }
        local_count += __popcll(result);
    }

    // Warp-level reduction via shuffle
    for (int offset = 16; offset > 0; offset >>= 1)
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, offset);

    // Block-level reduction via shared memory
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) warp_sums[warp_id] = local_count;
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned long long total = 0;
        int n_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < n_warps; w++) total += warp_sums[w];

        // In-kernel filtering -- alleen frequent candidates gaan naar output
        if (total >= (unsigned long long)min_count) {
            long long pos = (long long)atomicAdd((unsigned long long*)n_results, 1ULL);
            if (pos < max_results) {
                result_indices[pos] = cand_idx;  // int64: cast removed, indices exceed 2^31 at 536M+ columns
                result_counts[pos] = (long long)total;
            }
        }
    }
}
