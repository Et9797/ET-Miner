
extern "C" __global__
void bitvec_extract_tids(
    const unsigned long long* __restrict__ bitvecs,  // [n_rows, n_u64s]
    const long long* __restrict__ offsets,            // [n_rows] write offset per row (from prefix sum)
    int* __restrict__ out_indices,                     // pre-allocated output array
    long long n_rows, long long n_u64s, long long tid_offset
) {
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    long long write_pos = offsets[row];
    long long row_start = row * n_u64s;

    for (long long w = 0; w < n_u64s; w++) {
        unsigned long long word = bitvecs[row_start + w];
        long long base = w * 64 + tid_offset;
        while (word != 0ULL) {
            int bit = __ffsll(word) - 1;
            out_indices[write_pos++] = base + bit;
            word &= word - 1;
        }
    }
}
