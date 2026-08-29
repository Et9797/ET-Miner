
// Convert CSR format to column-oriented bitvectors on GPU
// Each thread handles one row, setting bits for all items in that row
//
// NOTE: Uses long long for ALL index calculations to support >2B elements
// This is critical for large-scale mining (10B+ transactions)

extern "C" __global__
void csr_to_bitvec(
    const long long* __restrict__ csr_indptr,   // [n_rows + 1] row pointers
    const long long* __restrict__ csr_indices,  // [nnz] column indices
    unsigned long long* __restrict__ bitvecs,   // [n_cols, n_u64s] output
    const long long n_rows,
    const long long n_cols,
    const long long n_u64s
) {
    // Each thread processes one row
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    // Get column indices for this row from CSR format
    long long start = csr_indptr[row];
    long long end = csr_indptr[row + 1];

    // Calculate which u64 word and which bit within that word
    // This row maps to bit (row % 64) in word (row / 64)
    long long word_idx = row / 64;
    unsigned long long bit_mask = 1ULL << (row % 64);

    // For each column in this row, set the corresponding bit
    for (long long i = start; i < end; i++) {
        long long col = csr_indices[i];
        // bitvecs layout: [col, word_idx] = bitvecs[col * n_u64s + word_idx]
        // Use atomicOr for thread-safe bit setting (multiple rows may write same word)
        atomicOr(&bitvecs[col * n_u64s + word_idx], bit_mask);
    }
}
