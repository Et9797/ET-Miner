//! Co-occurrence matrix computation for k=2 itemsets.
//!
//! These functions compute all pairwise item co-occurrences efficiently,
//! which is much faster than computing k=2 itemsets one by one.

use rayon::prelude::*;

/// Count support for k=2 itemsets using matrix multiplication approach.
///
/// For k=2, we can compute all pair supports efficiently.
/// Returns a flattened n_cols × n_cols matrix (row-major).
///
/// # Arguments
/// * `matrix` - Flattened boolean array (row-major, n_rows × n_cols)
/// * `n_rows` - Number of rows
/// * `n_cols` - Number of columns
///
/// # Returns
/// Flattened co-occurrence matrix (n_cols × n_cols, row-major)
pub fn compute_cooccurrence_matrix_raw(
    matrix: &[u8],
    n_rows: usize,
    n_cols: usize,
) -> Vec<u32> {
    let partial_cooccurs: Vec<Vec<u32>> = (0..n_rows)
        .into_par_iter()
        .map(|row_idx| {
            let mut local_cooccur = vec![0u32; n_cols * n_cols];
            let row_start = row_idx * n_cols;

            let present_items: Vec<usize> = (0..n_cols)
                .filter(|&col| matrix.get(row_start + col).is_some_and(|&v| v != 0))
                .collect();

            for (idx_i, &i) in present_items.iter().enumerate() {
                for &j in &present_items[idx_i..] {
                    local_cooccur[i * n_cols + j] += 1;
                }
            }

            local_cooccur
        })
        .collect();

    // Merge partial results
    let mut cooccur = vec![0u32; n_cols * n_cols];
    for partial in partial_cooccurs {
        for (i, &v) in partial.iter().enumerate() {
            cooccur[i] += v;
        }
    }

    cooccur
}

/// Compute k=2 co-occurrence directly from SPARSE CSR format.
///
/// This is MUCH more efficient for sparse data since we only iterate
/// over non-zero entries.
///
/// # Arguments
/// * `indptr` - CSR row pointers
/// * `indices` - CSR column indices
/// * `n_rows` - Number of rows
/// * `n_cols` - Number of columns
///
/// # Returns
/// Flattened co-occurrence matrix (n_cols × n_cols, row-major)
pub fn compute_cooccurrence_sparse_raw(
    indptr: &[i64],
    indices: &[i64],
    n_rows: usize,
    n_cols: usize,
) -> Vec<u32> {
    // Parallelize over rows
    let partial_cooccurs: Vec<Vec<u32>> = (0..n_rows)
        .into_par_iter()
        .map(|row_idx| {
            let mut local_cooccur = vec![0u32; n_cols * n_cols];

            let start = indptr[row_idx] as usize;
            let end = indptr[row_idx + 1] as usize;

            // Get all present items for this row (already sorted!)
            let present_items: Vec<usize> = indices[start..end]
                .iter()
                .map(|&col| col as usize)
                .collect();

            // Update co-occurrence for all pairs
            for (idx_i, &i) in present_items.iter().enumerate() {
                for &j in &present_items[idx_i..] {
                    local_cooccur[i * n_cols + j] += 1;
                }
            }

            local_cooccur
        })
        .collect();

    // Merge partial results
    let mut cooccur = vec![0u32; n_cols * n_cols];
    for partial in partial_cooccurs {
        for (i, &v) in partial.iter().enumerate() {
            cooccur[i] += v;
        }
    }

    cooccur
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cooccurrence_sparse() {
        // 3 transactions, 3 items:
        // T0: {0, 1}
        // T1: {1, 2}
        // T2: {0, 2}
        let indptr = vec![0i64, 2, 4, 6];
        let indices = vec![0i64, 1, 1, 2, 0, 2];

        let cooccur = compute_cooccurrence_sparse_raw(&indptr, &indices, 3, 3);

        // Expected (upper triangular, including diagonal):
        // (0,0): 2 (T0, T2)
        // (0,1): 1 (T0)
        // (0,2): 1 (T2)
        // (1,1): 2 (T0, T1)
        // (1,2): 1 (T1)
        // (2,2): 2 (T1, T2)
        assert_eq!(cooccur[0], 2); // (0,0) = 0*3+0
        assert_eq!(cooccur[1], 1); // (0,1) = 0*3+1
        assert_eq!(cooccur[2], 1); // (0,2) = 0*3+2
        assert_eq!(cooccur[4], 2); // (1,1) = 1*3+1
        assert_eq!(cooccur[5], 1); // (1,2) = 1*3+2
        assert_eq!(cooccur[8], 2); // (2,2) = 2*3+2
    }
}
