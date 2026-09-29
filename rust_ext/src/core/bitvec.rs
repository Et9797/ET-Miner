//! Bitvector operations for GPU acceleration.
//!
//! This module provides functions to build packed bitvector representations
//! of sparse matrices, optimized for GPU transfer and CUDA kernel operations.

use rayon::prelude::*;
use super::matrix::CscMatrix;

/// Build column bitvecs as u64 arrays for GPU transfer.
///
/// This function exports the column bitvecs in a format ready for GPU consumption.
/// Each column gets a bitvec where bit i is set if transaction i contains that item.
/// The bitvecs are packed into u64s (64 transactions per u64) for efficient GPU operations.
///
/// # Arguments
/// * `csr_indptr` - CSR row pointers array (n_rows + 1)
/// * `csr_indices` - CSR column indices array
/// * `n_rows` - Number of transactions
/// * `n_cols` - Number of items
///
/// # Returns
/// Flattened 2D array of shape [n_cols, ceil(n_rows/64)] containing packed bitvecs.
/// Each "row" represents a column's bitvec.
pub fn build_column_bitvecs_u64_raw(
    csr_indptr: &[i64],
    csr_indices: &[i64],
    n_rows: usize,
    n_cols: usize,
) -> Vec<u64> {
    // Convert CSR to CSC once (enables efficient column iteration)
    let csc = CscMatrix::from_csr(csr_indptr, csr_indices, n_rows, n_cols);

    // Number of u64s needed per column (64 bits per u64)
    let n_u64s = n_rows.div_ceil(64);

    // Build all column bitvecs in parallel using direct word manipulation
    // This is 4x faster than bitvec crate's per-bit operations!
    let column_u64s: Vec<Vec<u64>> = (0..n_cols)
        .into_par_iter()
        .map(|col| {
            // Pre-allocate with zeros
            let mut bitvec = vec![0u64; n_u64s];

            // Set bits directly using word operations
            // bit position = row_idx, word index = row_idx / 64, bit offset = row_idx % 64
            for &row_idx in csc.get_column_rows(col) {
                let row = row_idx as usize;
                bitvec[row >> 6] |= 1u64 << (row & 63);  // row/64 and row%64 using bit ops
            }

            bitvec
        })
        .collect();

    // Flatten to 1D array (row-major: [col0_words..., col1_words..., ...])
    column_u64s.into_iter().flatten().collect()
}

/// Get the number of u64 words needed to represent n_rows bits.
#[inline]
pub fn words_for_rows(n_rows: usize) -> usize {
    n_rows.div_ceil(64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_column_bitvecs() {
        // 3 transactions, 2 items:
        // T0: {0}
        // T1: {0, 1}
        // T2: {1}
        let indptr = vec![0i64, 1, 3, 4];
        let indices = vec![0i64, 0, 1, 1];

        let bitvecs = build_column_bitvecs_u64_raw(&indptr, &indices, 3, 2);

        // n_u64s = 1 (ceil(3/64) = 1)
        // Result should be [col0_bitvec, col1_bitvec]
        assert_eq!(bitvecs.len(), 2);

        // Column 0: transactions 0, 1 → bits 0, 1 set → 0b011 = 3
        assert_eq!(bitvecs[0], 3);

        // Column 1: transactions 1, 2 → bits 1, 2 set → 0b110 = 6
        assert_eq!(bitvecs[1], 6);
    }

    #[test]
    fn test_words_for_rows() {
        assert_eq!(words_for_rows(0), 0);
        assert_eq!(words_for_rows(1), 1);
        assert_eq!(words_for_rows(64), 1);
        assert_eq!(words_for_rows(65), 2);
        assert_eq!(words_for_rows(128), 2);
        assert_eq!(words_for_rows(129), 3);
    }
}
