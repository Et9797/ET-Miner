//! Itemset support counting algorithms.
//!
//! This module provides efficient parallel counting of itemset support
//! using various strategies (sparse, SIMD, dense).

use rayon::prelude::*;
use bitvec::prelude::*;
use super::matrix::CscMatrix;

/// Count support for multiple itemsets in parallel using SPARSE CSR format.
///
/// This is the preferred function for sparse data, avoiding the massive
/// overhead of CSR → Dense conversion.
///
/// # Arguments
/// * `indptr` - CSR row pointers array (n_rows + 1)
/// * `indices` - CSR column indices array
/// * `n_rows` - Number of rows in the matrix
/// * `itemsets` - List of itemsets, each itemset is a list of column indices
///
/// # Returns
/// Support count for each itemset
pub fn count_itemsets_sparse_raw(
    indptr: &[i64],
    indices: &[i64],
    n_rows: usize,
    itemsets: &[Vec<usize>],
) -> Vec<u32> {
    itemsets
        .par_iter()
        .map(|cols| {
            // For each itemset, count rows where ALL columns are present
            let mut count: u32 = 0;

            for row_idx in 0..n_rows {
                // Get the range of non-zero column indices for this row
                let start = indptr[row_idx] as usize;
                let end = indptr[row_idx + 1] as usize;

                // Fast path: if row has fewer non-zeros than itemset size, skip
                if end - start < cols.len() {
                    continue;
                }

                // Check if ALL columns in the itemset are present in this row
                // The indices are sorted, so we can use binary search
                let all_present = cols.iter().all(|&col| {
                    indices[start..end]
                        .binary_search(&(col as i64))
                        .is_ok()
                });

                if all_present {
                    count += 1;
                }
            }
            count
        })
        .collect()
}

/// Count itemset support using column-based SIMD intersection.
///
/// This is a faster alternative to `count_itemsets_sparse_raw` that uses:
/// 1. CSC format for direct column access (no binary search)
/// 2. Bitvec row masks for efficient set intersection
/// 3. SIMD-friendly popcount for counting
///
/// # Arguments
/// * `csr_indptr` - CSR row pointers (converted to CSC internally)
/// * `csr_indices` - CSR column indices
/// * `n_rows` - Number of transactions
/// * `n_cols` - Number of items
/// * `itemsets` - Vec of item sets to count
///
/// # Returns
/// Support count for each itemset
pub fn count_itemsets_simd_raw(
    csr_indptr: &[i64],
    csr_indices: &[i64],
    n_rows: usize,
    n_cols: usize,
    itemsets: &[Vec<usize>],
) -> Vec<u32> {
    // Convert CSR to CSC once (amortized over all itemsets)
    let csc = CscMatrix::from_csr(csr_indptr, csr_indices, n_rows, n_cols);

    // Pre-build column bitvecs (one bit per row)
    // This is the key optimization: O(n_cols) preprocessing for O(1) column access
    let column_bitvecs: Vec<BitVec> = (0..n_cols)
        .map(|col| {
            let mut bv = bitvec![0; n_rows];
            for &row_idx in csc.get_column_rows(col) {
                bv.set(row_idx as usize, true);
            }
            bv
        })
        .collect();

    // Process itemsets in parallel
    itemsets
        .par_iter()
        .map(|cols| {
            if cols.is_empty() {
                return n_rows as u32;  // Empty itemset matches all rows
            }

            // Start with the first column's bitvec
            let first_col = cols[0];
            if first_col >= n_cols {
                return 0;
            }

            let mut result = column_bitvecs[first_col].clone();

            // AND with remaining columns
            for &col in &cols[1..] {
                if col >= n_cols {
                    return 0;
                }
                // Bitwise AND - this is where the magic happens!
                // bitvec optimizes this to use SIMD-width operations
                result &= &column_bitvecs[col];
            }

            // Count ones - bitvec uses popcount instruction
            result.count_ones() as u32
        })
        .collect()
}

/// Count support for multiple itemsets in parallel (DENSE matrix version).
///
/// This is the original function that works on dense matrices.
/// Use `count_itemsets_sparse_raw` for CSR format to avoid conversion overhead.
///
/// # Arguments
/// * `matrix` - Flattened boolean array (row-major, n_rows × n_cols)
/// * `n_rows` - Number of rows
/// * `n_cols` - Number of columns
/// * `itemsets` - List of itemsets, each itemset is a list of column indices
///
/// # Returns
/// Support count for each itemset
pub fn count_itemsets_parallel_raw(
    matrix: &[u8],
    n_rows: usize,
    n_cols: usize,
    itemsets: &[Vec<usize>],
) -> Vec<u32> {
    itemsets
        .par_iter()
        .map(|cols| {
            let mut count: u32 = 0;
            for row_idx in 0..n_rows {
                let row_start = row_idx * n_cols;
                let all_present = cols.iter().all(|&col| {
                    matrix.get(row_start + col).is_some_and(|&v| v != 0)
                });
                if all_present {
                    count += 1;
                }
            }
            count
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_csr() -> (Vec<i64>, Vec<i64>, usize, usize) {
        // 4 transactions, 5 items:
        // T0: {0, 1, 2}
        // T1: {1, 2, 3}
        // T2: {0, 2, 4}
        // T3: {1, 2}
        let indptr = vec![0i64, 3, 6, 9, 11];
        let indices = vec![0i64, 1, 2, 1, 2, 3, 0, 2, 4, 1, 2];
        (indptr, indices, 4, 5)
    }

    #[test]
    fn test_count_itemsets_sparse() {
        let (indptr, indices, n_rows, _) = create_test_csr();

        let itemsets = vec![
            vec![1, 2],     // Should match T0, T1, T3 = 3
            vec![0, 2],     // Should match T0, T2 = 2
            vec![0, 1, 2],  // Should match T0 = 1
            vec![3, 4],     // Should match none = 0
        ];

        let counts = count_itemsets_sparse_raw(&indptr, &indices, n_rows, &itemsets);
        assert_eq!(counts, vec![3, 2, 1, 0]);
    }

    #[test]
    fn test_count_itemsets_simd() {
        let (indptr, indices, n_rows, n_cols) = create_test_csr();

        let itemsets = vec![
            vec![1, 2],     // Should match T0, T1, T3 = 3
            vec![0, 2],     // Should match T0, T2 = 2
            vec![0, 1, 2],  // Should match T0 = 1
            vec![3, 4],     // Should match none = 0
        ];

        let counts = count_itemsets_simd_raw(&indptr, &indices, n_rows, n_cols, &itemsets);
        assert_eq!(counts, vec![3, 2, 1, 0]);
    }

    #[test]
    fn test_sparse_vs_simd_consistency() {
        let (indptr, indices, n_rows, n_cols) = create_test_csr();

        let itemsets: Vec<Vec<usize>> = vec![
            vec![0], vec![1], vec![2], vec![3], vec![4],
            vec![0, 1], vec![1, 2], vec![2, 3],
            vec![0, 1, 2], vec![1, 2, 3],
        ];

        let sparse_counts = count_itemsets_sparse_raw(&indptr, &indices, n_rows, &itemsets);
        let simd_counts = count_itemsets_simd_raw(&indptr, &indices, n_rows, n_cols, &itemsets);

        assert_eq!(sparse_counts, simd_counts);
    }
}
