
extern "C" __global__
void count_k3plus_gpu_resident(
    const unsigned long long* __restrict__ bitvecs,
    const int* __restrict__ freq_itemsets,      // [n_freq × k_prev] row-major, VRAM
    const int k_prev,                           // columns per row (= k-1)
    const long long* __restrict__ group_starts,  // [n_groups] VRAM (int64 since build_prefix_groups_gpu returns int64)
    const long long* __restrict__ group_sizes,   // [n_groups] VRAM
    const long long* __restrict__ cumulative_pairs,  // [n_groups + 1] VRAM
    const long long n_u64s,
    const long long n_groups,  // int64: >2.1B groups at K>=9
    const long long total_candidates,
    const long long min_count,
    long long* __restrict__ result_indices,     // int64: candidate indices exceed 2^31 at K>=7
    long long* __restrict__ result_counts,
    long long* __restrict__ n_results,         // int64: atomic counter for >2B results
    const long long max_results,
    const long long candidate_offset
) {
    long long cand_idx = (long long)blockIdx.y * (long long)gridDim.x
                       + (long long)blockIdx.x + candidate_offset;
    if (cand_idx >= total_candidates) return;

    // Shared decode (_decode_common.cu, prepended by the loader), sized variant:
    // groups are runs of rows of the sorted prev-level table.
    long long g, i_val, j_val;
    if (!_decode_candidate_sized(cumulative_pairs, group_sizes, n_groups, cand_idx, &g, &i_val, &j_val)) return;
    long long gs = group_starts[g];

    // Build candidate in shared memory:
    // prefix = first k_prev-1 cols of row gs (shared by all rows in group)
    // suffix_i = last col of row gs+i_val
    // suffix_j = last col of row gs+j_val
    __shared__ int s_items[64];
    __shared__ unsigned long long warp_sums[8];

    int prefix_len = k_prev - 1;
    int k = k_prev + 1;

    if (threadIdx.x < prefix_len && threadIdx.x < 62) {
        s_items[threadIdx.x] = freq_itemsets[gs * k_prev + threadIdx.x];
    }
    if (threadIdx.x == 0) {
        s_items[prefix_len] = freq_itemsets[(gs + i_val) * k_prev + (k_prev - 1)];
        s_items[prefix_len + 1] = freq_itemsets[(gs + j_val) * k_prev + (k_prev - 1)];
    }
    __syncthreads();

    // AND + popcount
    unsigned long long local_count = 0;
    for (long long w = threadIdx.x; w < n_u64s; w += blockDim.x) {
        unsigned long long result = ~0ULL;
        for (int i = 0; i < k; i++) {
            result &= bitvecs[(long long)s_items[i] * n_u64s + w];
            if (__ballot_sync(__activemask(), result != 0ULL) == 0) { result = 0ULL; break; }
        }
        local_count += __popcll(result);
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, offset);

    // Block-level reduction
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
                result_indices[pos] = cand_idx;  // int64: cast removed, indices exceed 2^31 at 536M+ columns
                result_counts[pos] = (long long)total;
            }
        }
    }
}
