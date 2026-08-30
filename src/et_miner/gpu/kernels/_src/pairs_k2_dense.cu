
extern "C" __global__
void count_pairs_k2_dense(
    const unsigned long long* __restrict__ bitvecs,
    const int* __restrict__ freq_items,
    const long long n_u64s,
    const int n_freq,
    int* __restrict__ result_counts,  // int32: counts <= n_transactions < 2^31 (guarded host-side)
    const long long pair_offset
) {
    long long pair_idx = (long long)blockIdx.y * (long long)gridDim.x
                       + (long long)blockIdx.x + pair_offset;
    long long total_pairs = (long long)n_freq * ((long long)n_freq - 1) / 2;
    if (pair_idx >= total_pairs) return;

    long long j_val = (long long)floor(0.5 + sqrt(0.25 + 2.0 * (double)pair_idx));
    long long i_val = pair_idx - j_val * (j_val - 1) / 2;
    if (j_val >= n_freq || i_val >= j_val || i_val < 0) return;

    int item_i = freq_items[i_val];
    int item_j = freq_items[j_val];

    unsigned long long local_count = 0;
    for (long long w = threadIdx.x; w < n_u64s; w += blockDim.x) {
        unsigned long long a = bitvecs[(long long)item_i * n_u64s + w];
        unsigned long long b = bitvecs[(long long)item_j * n_u64s + w];
        local_count += __popcll(a & b);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, offset);

    __shared__ unsigned long long warp_sums[8];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) warp_sums[warp_id] = local_count;
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned long long total = 0;
        int n_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < n_warps; w++) total += warp_sums[w];
        result_counts[pair_idx] = (int)total;
    }
}
