
extern "C" __global__
void csr_intersect_count(
    const int* __restrict__ tidsets,          // all tid-sets concatenated (i32)
    const long long* __restrict__ offsets,     // CSR offsets per itemset (n_itemsets + 1)
    const long long* __restrict__ pair_a,      // index of first itemset in each candidate pair
    const long long* __restrict__ pair_b,      // index of second itemset in each candidate pair
    long long n_pairs,                         // total candidate pairs to process
    long long* __restrict__ result_counts      // output: intersection size per pair
) {
    long long pair_idx = (long long)blockIdx.y * (long long)gridDim.x + (long long)blockIdx.x;
    if (pair_idx >= n_pairs) return;

    long long a_idx = pair_a[pair_idx];
    long long b_idx = pair_b[pair_idx];

    long long a_start = offsets[a_idx];
    long long a_end = offsets[a_idx + 1];
    long long b_start = offsets[b_idx];
    long long b_end = offsets[b_idx + 1];
    long long a_len = a_end - a_start;
    long long b_len = b_end - b_start;

    // Two-pointer merge: thread 0 does sequential merge
    if (threadIdx.x == 0) {
        long long count = 0;
        long long i = 0, j = 0;
        while (i < a_len && j < b_len) {
            int a_val = tidsets[a_start + i];
            int b_val = tidsets[b_start + j];
            if (a_val == b_val) {
                count++;
                i++;
                j++;
            } else if (a_val < b_val) {
                i++;
            } else {
                j++;
            }
        }
        result_counts[pair_idx] = count;
    }
}
