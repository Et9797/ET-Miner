
extern "C" __global__
void count_k3plus_dense(
    const unsigned long long* __restrict__ bitvecs,
    const int* __restrict__ group_prefix_items,
    const long long* __restrict__ group_prefix_offsets,
    const int* __restrict__ group_suffixes,
    const long long* __restrict__ group_suffix_offsets,
    const long long* __restrict__ cumulative_pairs,
    const long long n_u64s,
    const long long n_groups,  // int64: >2.1B groups at K>=9
    const long long total_candidates,
    int* __restrict__ result_counts,  // int32: counts <= n_transactions < 2^31 (guarded host-side)
    const long long candidate_offset
) {
    long long cand_idx = (long long)blockIdx.y * (long long)gridDim.x
                       + (long long)blockIdx.x + candidate_offset;
    if (cand_idx >= total_candidates) return;

    // Shared decode (_decode_common.cu, prepended by the loader): identical
    // arithmetic to the host-side decode_k3plus_flat.
    long long g, i_val, j_val;
    if (!_decode_candidate(cumulative_pairs, group_suffix_offsets, n_groups, cand_idx, &g, &i_val, &j_val)) return;
    long long suf_start = group_suffix_offsets[g];

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
            if (__ballot_sync(__activemask(), result != 0ULL) == 0) { result = 0ULL; break; }
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
        result_counts[cand_idx - candidate_offset] = (int)total;
    }
}
