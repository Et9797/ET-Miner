
extern "C" __global__
void count_pairs_fused_k2(
    const unsigned long long* __restrict__ bitvecs,  // [n_cols, n_u64s]
    const int* __restrict__ freq_items,              // [n_freq] column indices
    const long long n_u64s,
    const int n_freq,
    const long long min_count,
    long long* __restrict__ result_i,  // output: freq_items index of item i (int64 for >2B items)
    long long* __restrict__ result_j,  // output: freq_items index of item j
    long long* __restrict__ result_count,  // output: support count
    long long* __restrict__ n_results, // atomic counter (int64 for >2B results)
    const long long max_results,            // output buffer capacity
    const long long pair_offset       // offset for multi-GPU pair splitting
) {
    // Decode linear pair index -> (i, j) via triangular number inverse
    // Mapping: pair_idx=0->(0,1), 1->(0,2), 2->(1,2), 3->(0,3), ...
    // 2D grid for >2.15B pairs: gridDim.y * gridDim.x covers up to ~141 trillion
    long long pair_idx = (long long)blockIdx.y * (long long)gridDim.x + (long long)blockIdx.x + pair_offset;
    long long total_pairs = (long long)n_freq * ((long long)n_freq - 1) / 2;
    if (pair_idx >= total_pairs) return;

    // j = floor(0.5 + sqrt(0.25 + 2*pair_idx))
    // i = pair_idx - j*(j-1)/2
    long long j_val = (long long)floor(0.5 + sqrt(0.25 + 2.0 * (double)pair_idx));
    long long i_val = pair_idx - j_val * (j_val - 1) / 2;

    // Safety bounds check
    if (j_val >= n_freq || i_val >= j_val || i_val < 0) return;

    int item_i = freq_items[i_val];
    int item_j = freq_items[j_val];

    // AND + popcount across all u64 words (threads parallelize over words)
    unsigned long long local_count = 0;
    for (long long w = threadIdx.x; w < n_u64s; w += blockDim.x) {
        unsigned long long a = bitvecs[(long long)item_i * n_u64s + w];
        unsigned long long b = bitvecs[(long long)item_j * n_u64s + w];
        local_count += __popcll(a & b);
    }

    // Warp-level reduction via shuffle (faster than shared memory)
    for (int offset = 16; offset > 0; offset >>= 1)
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, offset);

    // Block reduction via shared memory (for blocks > 1 warp)
    __shared__ unsigned long long warp_sums[8];  // max 256 threads = 8 warps
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
                result_i[pos] = i_val;
                result_j[pos] = j_val;
                result_count[pos] = (long long)total;
            }
        }
    }
}
