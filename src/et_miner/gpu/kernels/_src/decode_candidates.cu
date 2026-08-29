
extern "C" __global__
void decode_candidates_gpu(
    const long long* __restrict__ result_indices,  // int64: candidate indices exceed 2^31 at K>=7
    const long long* __restrict__ result_counts,
    const int* __restrict__ freq_itemsets,       // [n_freq × k_prev] row-major
    const int k_prev,
    const long long* __restrict__ group_starts,
    const long long* __restrict__ group_sizes,
    const long long* __restrict__ cumulative_pairs,
    const long long n_groups,  // int64: >2.1B groups at K>=9
    const long long n_results,
    int* __restrict__ output_itemsets,           // [n_results × k] row-major
    long long* __restrict__ output_counts
) {
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_results) return;

    long long cand_idx = result_indices[tid];  // int64: index exceeds 2^31 at K>=7
    int k = k_prev + 1;

    // Binary search for group
    long long lo = 0, hi = n_groups - 1;
    while (lo < hi) {
        long long mid = (lo + hi + 1) / 2;
        if (cumulative_pairs[mid] <= cand_idx) lo = mid;
        else hi = mid - 1;
    }
    long long g = lo;
    long long pair_idx = (long long)cand_idx - cumulative_pairs[g];

    long long gs = group_starts[g];

    // Triangular inverse
    long long j_val = (long long)floor(0.5 + sqrt(0.25 + 2.0 * (double)pair_idx));
    long long i_val = pair_idx - j_val * (j_val - 1) / 2;

    // Write prefix (first k_prev-1 items from any row in group)
    long long out_base = tid * k;
    int prefix_len = k_prev - 1;
    for (int i = 0; i < prefix_len; i++) {
        output_itemsets[out_base + i] = freq_itemsets[gs * k_prev + i];
    }
    // Write suffix_i and suffix_j (last column of rows gs+i_val and gs+j_val)
    output_itemsets[out_base + prefix_len] = freq_itemsets[(gs + i_val) * k_prev + (k_prev - 1)];
    output_itemsets[out_base + prefix_len + 1] = freq_itemsets[(gs + j_val) * k_prev + (k_prev - 1)];

    output_counts[tid] = result_counts[tid];
}
