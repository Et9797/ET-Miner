
extern "C" __global__
void count_k3plus_indirect(
    const unsigned long long* __restrict__ bitvecs,
    const int* __restrict__ group_prefix_items,
    const long long* __restrict__ group_prefix_offsets,
    const int* __restrict__ group_suffixes,
    const long long* __restrict__ group_suffix_offsets,
    const long long* __restrict__ cumulative_pairs,
    const long long n_u64s,
    const long long n_groups,
    const long long min_count,
    const long long* __restrict__ input_indices,   // candidate indices to recount
    const long long n_input,                       // number of candidates to recount
    long long* __restrict__ result_indices,
    long long* __restrict__ result_counts,
    long long* __restrict__ n_results,
    const long long max_results
) {
    long long block_idx = (long long)blockIdx.y * (long long)gridDim.x
                        + (long long)blockIdx.x;
    if (block_idx >= n_input) return;

    // Read candidate index from input array (indirection)
    long long cand_idx = input_indices[block_idx];

    // Binary search: find group g
    long long lo = 0, hi = n_groups - 1;
    while (lo < hi) {
        long long mid = (lo + hi + 1) / 2;
        if (cumulative_pairs[mid] <= cand_idx) lo = mid;
        else hi = mid - 1;
    }
    long long g = lo;
    long long pair_idx = cand_idx - cumulative_pairs[g];

    long long suf_start = group_suffix_offsets[g];
    long long group_size = group_suffix_offsets[g + 1] - suf_start;

    long long j_val = (long long)floor(0.5 + sqrt(0.25 + 2.0 * (double)pair_idx));
    long long i_val = pair_idx - j_val * (j_val - 1) / 2;
    if (j_val >= group_size || i_val >= j_val || i_val < 0) return;

    int suffix_i = group_suffixes[suf_start + i_val];
    int suffix_j = group_suffixes[suf_start + j_val];

    long long pref_start = group_prefix_offsets[g];
    int prefix_len = (int)(group_prefix_offsets[g + 1] - pref_start);
    int k = prefix_len + 2;

    __shared__ int s_items[64];
    __shared__ unsigned long long warp_sums[8];

    if (threadIdx.x < prefix_len && threadIdx.x < 62) {
        s_items[threadIdx.x] = group_prefix_items[pref_start + threadIdx.x];
    }
    if (threadIdx.x == 0) {
        s_items[prefix_len] = suffix_i;
        s_items[prefix_len + 1] = suffix_j;
    }
    __syncthreads();

    unsigned long long local_count = 0;
    for (long long w = threadIdx.x; w < n_u64s; w += blockDim.x) {
        unsigned long long result = ~0ULL;
        for (int i = 0; i < k; i++) {
            result &= bitvecs[(long long)s_items[i] * n_u64s + w];
            if (__ballot_sync(__activemask(), result != 0ULL) == 0) {
                result = 0ULL;
                break;
            }
        }
        local_count += __popcll(result);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, offset);

    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) warp_sums[warp_id] = local_count;
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned long long total = 0;
        int n_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < n_warps; w++) total += warp_sums[w];

        if (total >= (unsigned long long)min_count) {
            long long pos = (long long)atomicAdd((unsigned long long*)n_results, 1ULL);
            if (pos < max_results) {
                result_indices[pos] = cand_idx;
                result_counts[pos] = (long long)total;
            }
        }
    }
}
