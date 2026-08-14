//! Matrix data structures and construction algorithms.
//!
//! This module provides CSC/CSR matrix representations and conversion utilities.
//! All functions are pure Rust with no Python dependencies.

use rayon::prelude::*;
use rand::prelude::*;

/// CSC (Compressed Sparse Column) matrix representation.
///
/// This format enables fast column access for row intersection operations.
/// Each column stores the row indices where it has non-zero values.
pub struct CscMatrix {
    /// Column pointers: indptr[col] to indptr[col+1] gives the range in `indices`
    pub indptr: Vec<usize>,
    /// Row indices for each column (sorted within each column)
    /// NOTE: Using i64 to support trillion-scale datasets (>2.1B indices)
    pub indices: Vec<i64>,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl CscMatrix {
    /// Convert CSR matrix to CSC format (transpose).
    ///
    /// This is a one-time conversion cost that enables efficient column access.
    pub fn from_csr(
        csr_indptr: &[i64],
        csr_indices: &[i64],
        n_rows: usize,
        n_cols: usize,
    ) -> Self {
        // Count entries per column
        let mut col_counts = vec![0usize; n_cols];
        for &col_idx in csr_indices {
            col_counts[col_idx as usize] += 1;
        }

        // Build CSC indptr (cumulative sum)
        let mut csc_indptr = vec![0usize; n_cols + 1];
        for (i, &count) in col_counts.iter().enumerate() {
            csc_indptr[i + 1] = csc_indptr[i] + count;
        }

        // Fill CSC indices
        let nnz = csr_indices.len();
        let mut csc_indices = vec![0i64; nnz];
        let mut col_positions = vec![0usize; n_cols]; // Current position per column

        for row_idx in 0..n_rows {
            let start = csr_indptr[row_idx] as usize;
            let end = csr_indptr[row_idx + 1] as usize;

            for &col_idx in &csr_indices[start..end] {
                let col = col_idx as usize;
                let pos = csc_indptr[col] + col_positions[col];
                csc_indices[pos] = row_idx as i64;
                col_positions[col] += 1;
            }
        }

        CscMatrix {
            indptr: csc_indptr,
            indices: csc_indices,
            n_rows,
            n_cols,
        }
    }

    /// Get all row indices for a given column.
    #[inline]
    pub fn get_column_rows(&self, col: usize) -> &[i64] {
        let start = self.indptr[col];
        let end = self.indptr[col + 1];
        &self.indices[start..end]
    }
}

/// Build CSR matrix from COO format using parallel sorting.
///
/// This is a HIGH-PERFORMANCE replacement for scipy's csr_matrix constructor!
/// Uses rayon for parallel sorting which scales to all CPU cores.
///
/// # Arguments
/// * `rows` - COO row indices (can have duplicates)
/// * `cols` - COO column indices (can have duplicates)
/// * `n_rows` - Number of rows in the matrix
///
/// # Returns
/// Tuple of (indptr, indices) arrays in CSR format.
///
/// # Performance
/// Expected 10-20x faster than scipy for large matrices due to parallel sort!
pub fn build_csr_from_coo_raw(
    rows: &[i64],
    cols: &[i64],
    n_rows: usize,
) -> (Vec<i64>, Vec<i64>) {
    // Create (row, col) tuples for stable sort
    let mut entries: Vec<(i64, i64)> = rows
        .iter()
        .zip(cols.iter())
        .map(|(&r, &c)| (r, c))
        .collect();

    // Parallel sort by (row, col) - this is the KEY speedup!
    entries.par_sort_unstable_by_key(|&(r, c)| (r, c));

    // Build indptr by counting entries per row
    let mut indptr = vec![0i64; n_rows + 1];
    for &(row, _) in &entries {
        indptr[row as usize + 1] += 1;
    }

    // Convert counts to cumulative sum (prefix sum)
    for i in 1..=n_rows {
        indptr[i] += indptr[i - 1];
    }

    // Extract sorted column indices
    let indices: Vec<i64> = entries.into_iter().map(|(_, col)| col).collect();

    (indptr, indices)
}

/// Generate random CSR matrix directly in Rust.
///
/// Combines data generation + COO→CSR conversion in one parallel operation.
/// This is much faster than generating random data in Python and then converting.
///
/// # Arguments
/// * `n_rows` - Number of rows in the matrix
/// * `n_cols` - Number of columns in the matrix
/// * `avg_items_per_row` - Average number of non-zero items per row
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Tuple of (indptr, indices) arrays in CSR format.
pub fn generate_random_csr_raw(
    n_rows: usize,
    n_cols: usize,
    avg_items_per_row: usize,
    seed: u64,
) -> (Vec<i64>, Vec<i64>) {
    let nnz = n_rows * avg_items_per_row;

    // Parallel random generation with thread-local RNGs
    let entries: Vec<(i64, i64)> = (0..nnz)
        .into_par_iter()
        .map_init(
            || {
                let thread_id = rayon::current_thread_index().unwrap_or(0);
                rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(thread_id as u64))
            },
            |rng, _| {
                let row = rng.gen_range(0..n_rows as i64);
                let col = rng.gen_range(0..n_cols as i64);
                (row, col)
            }
        )
        .collect();

    // Parallel sort by (row, col)
    let mut entries = entries;
    entries.par_sort_unstable_by_key(|&(r, c)| (r, c));

    // Build CSR indptr
    let mut indptr = vec![0i64; n_rows + 1];
    for &(row, _) in &entries {
        indptr[row as usize + 1] += 1;
    }
    for i in 1..=n_rows {
        indptr[i] += indptr[i - 1];
    }

    let indices: Vec<i64> = entries.into_iter().map(|(_, col)| col).collect();

    (indptr, indices)
}

/// Generate bootstrapped CSR matrix by sampling source transactions with replacement.
///
/// This function generates a bootstrap sample for statistical analysis.
/// It samples transaction indices uniformly with replacement from the source data
/// and builds CSR format directly.
///
/// # Arguments
/// * `source_data` - List of transactions, each transaction is a list of item indices
/// * `n_rows` - Number of rows to generate in the bootstrap sample
/// * `n_cols` - Number of columns (items) in the matrix
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Tuple of (indptr, indices) arrays in CSR format.
pub fn generate_bootstrap_csr_raw(
    source_data: &[Vec<i64>],
    n_rows: usize,
    n_cols: usize,
    seed: u64,
) -> (Vec<i64>, Vec<i64>) {
    let n_source = source_data.len();

    if n_source == 0 {
        // Edge case: empty source data
        let indptr = vec![0i64; n_rows + 1];
        return (indptr, Vec::new());
    }

    // Define chunk size for parallel processing
    const CHUNK_SIZE: usize = 10000;
    let n_chunks = n_rows.div_ceil(CHUNK_SIZE);

    // Parallel generation: each chunk produces (row_nnzs, sorted_indices)
    let chunk_results: Vec<(Vec<usize>, Vec<i64>)> = (0..n_chunks)
        .into_par_iter()
        .map(|chunk_idx| {
            // Calculate row range for this chunk
            let start_row = chunk_idx * CHUNK_SIZE;
            let end_row = std::cmp::min(start_row + CHUNK_SIZE, n_rows);
            let chunk_rows = end_row - start_row;

            // Thread-local RNG seeded by chunk index for reproducibility
            let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(
                seed.wrapping_add(chunk_idx as u64)
            );

            let mut row_nnzs = Vec::with_capacity(chunk_rows);
            let mut chunk_indices = Vec::new();

            for _ in 0..chunk_rows {
                // Sample a source transaction index with replacement
                let src_idx = rng.gen_range(0..n_source);
                let transaction = &source_data[src_idx];

                // Filter to valid column indices and deduplicate
                let mut items: Vec<i64> = transaction
                    .iter()
                    .filter(|&&col| (col as usize) < n_cols && col >= 0)
                    .copied()
                    .collect();

                // Sort and deduplicate for proper CSR format
                items.sort_unstable();
                items.dedup();

                row_nnzs.push(items.len());
                chunk_indices.extend(items);
            }

            (row_nnzs, chunk_indices)
        })
        .collect();

    // Build final CSR arrays from chunk results
    let mut indptr = Vec::with_capacity(n_rows + 1);
    indptr.push(0i64);

    let mut total_nnz = 0usize;
    for (row_nnzs, _) in &chunk_results {
        for &nnz in row_nnzs {
            total_nnz += nnz;
            indptr.push(total_nnz as i64);
        }
    }

    // Concatenate all indices
    let mut indices = Vec::with_capacity(total_nnz);
    for (_, chunk_indices) in chunk_results {
        indices.extend(chunk_indices);
    }

    (indptr, indices)
}

/// Convert boolean predicate matrix to transaction item lists.
///
/// For each row in the boolean matrix, collects the column indices where the
/// value is `true`. This converts a dense boolean matrix representation to the
/// sparse transaction list format used by apriori algorithms.
///
/// # Arguments
/// * `matrix` - Flattened boolean matrix (row-major)
/// * `n_rows` - Number of rows
/// * `n_cols` - Number of columns
///
/// # Returns
/// List of transactions, where each transaction is a list of column indices.
pub fn prepare_transactions_raw(
    matrix: &[bool],
    n_rows: usize,
    n_cols: usize,
) -> Vec<Vec<usize>> {
    (0..n_rows)
        .into_par_iter()
        .map(|row_idx| {
            let row_start = row_idx * n_cols;
            (0..n_cols)
                .filter(|&col_idx| matrix[row_start + col_idx])
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csc_from_csr() {
        // Simple 3x3 CSR matrix:
        // Row 0: cols [0, 1]
        // Row 1: cols [1, 2]
        // Row 2: cols [0, 2]
        let csr_indptr = vec![0i64, 2, 4, 6];
        let csr_indices = vec![0i64, 1, 1, 2, 0, 2];

        let csc = CscMatrix::from_csr(&csr_indptr, &csr_indices, 3, 3);

        // Column 0: rows [0, 2]
        assert_eq!(csc.get_column_rows(0), &[0i64, 2]);
        // Column 1: rows [0, 1]
        assert_eq!(csc.get_column_rows(1), &[0i64, 1]);
        // Column 2: rows [1, 2]
        assert_eq!(csc.get_column_rows(2), &[1i64, 2]);
    }

    #[test]
    fn test_build_csr_from_coo() {
        let rows = vec![0i64, 0, 1, 1, 2, 2];
        let cols = vec![0i64, 1, 1, 2, 0, 2];

        let (indptr, indices) = build_csr_from_coo_raw(&rows, &cols, 3);

        assert_eq!(indptr, vec![0i64, 2, 4, 6]);
        // Indices should be sorted within each row
        assert_eq!(indices, vec![0i64, 1, 1, 2, 0, 2]);
    }

    #[test]
    fn test_generate_random_csr() {
        let (indptr, indices) = generate_random_csr_raw(100, 50, 5, 42);

        // Should have 101 elements in indptr (n_rows + 1)
        assert_eq!(indptr.len(), 101);
        // First element should be 0
        assert_eq!(indptr[0], 0);
        // Should have approximately 500 non-zero entries
        assert!((indices.len() as i64 - 500).abs() < 100);
    }
}
