//! High-performance Rust extension for et-miner association rule mining.
//!
//! This module provides PyO3 bindings to the pure Rust core algorithms.
//! The core algorithms are in the `core` module and have no Python dependencies.
//!
//! # Architecture
//! - `core/`: Pure Rust algorithms (no PyO3, framework-agnostic)
//! - `lib.rs`: Thin PyO3 wrappers for Python interop
//!
//! # Performance
//! Expected 10-50x speedup over Python/scipy implementation due to:
//! - No GIL contention
//! - Rayon work-stealing parallelism
//! - Cache-friendly memory access patterns
//! - SIMD-friendly loops (auto-vectorized by LLVM)
//! - **ZERO-COPY sparse CSR support**

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

// Core modules containing pure Rust algorithms
pub mod core;

// Re-export core types for direct Rust usage
pub use core::CscMatrix;

// =============================================================================
// PyO3 Wrappers - Thin bindings that call into core functions
// =============================================================================

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
/// * `Vec<u32>` - Support count for each itemset
#[pyfunction]
fn count_itemsets_sparse<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    n_rows: usize,
    itemsets: Vec<Vec<usize>>,
) -> Bound<'py, PyArray1<u32>> {
    let counts = core::counting::count_itemsets_sparse_raw(
        indptr.as_slice().unwrap(),
        indices.as_slice().unwrap(),
        n_rows,
        &itemsets,
    );
    PyArray1::from_vec(py, counts)
}

/// Count itemset support using column-based SIMD intersection.
///
/// This is a faster alternative to `count_itemsets_sparse` that uses:
/// 1. CSC format for direct column access (no binary search)
/// 2. Bitvec row masks for efficient set intersection
/// 3. SIMD-friendly popcount for counting
#[pyfunction]
fn count_itemsets_simd<'py>(
    py: Python<'py>,
    csr_indptr: PyReadonlyArray1<'py, i64>,
    csr_indices: PyReadonlyArray1<'py, i64>,
    n_rows: usize,
    n_cols: usize,
    itemsets: Vec<Vec<usize>>,
) -> Bound<'py, PyArray1<u32>> {
    let counts = core::counting::count_itemsets_simd_raw(
        csr_indptr.as_slice().unwrap(),
        csr_indices.as_slice().unwrap(),
        n_rows,
        n_cols,
        &itemsets,
    );
    PyArray1::from_vec(py, counts)
}

/// Count support for multiple itemsets in parallel (DENSE matrix version).
///
/// Use `count_itemsets_sparse` for CSR format to avoid conversion overhead.
#[pyfunction]
fn count_itemsets_parallel<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, u8>,
    itemsets: Vec<Vec<usize>>,
) -> Bound<'py, PyArray1<u32>> {
    let matrix_array = matrix.as_array();
    let n_rows = matrix_array.nrows();
    let n_cols = matrix_array.ncols();

    // Convert to contiguous slice
    let matrix_slice: Vec<u8> = matrix_array.iter().copied().collect();

    let counts = core::counting::count_itemsets_parallel_raw(
        &matrix_slice,
        n_rows,
        n_cols,
        &itemsets,
    );
    PyArray1::from_vec(py, counts)
}

/// Count support for k=2 itemsets using matrix multiplication approach.
#[pyfunction]
fn compute_cooccurrence_matrix<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, u8>,
) -> Bound<'py, PyArray2<u32>> {
    let matrix_array = matrix.as_array();
    let n_rows = matrix_array.nrows();
    let n_cols = matrix_array.ncols();

    // Convert to contiguous slice
    let matrix_slice: Vec<u8> = matrix_array.iter().copied().collect();

    let cooccur = core::cooccurrence::compute_cooccurrence_matrix_raw(
        &matrix_slice,
        n_rows,
        n_cols,
    );

    // Convert to 2D array
    let cooccur_2d: Vec<Vec<u32>> = cooccur
        .chunks(n_cols)
        .map(|chunk| chunk.to_vec())
        .collect();

    PyArray2::from_vec2(py, &cooccur_2d).unwrap()
}

/// Compute k=2 co-occurrence directly from SPARSE CSR format.
#[pyfunction]
fn compute_cooccurrence_sparse<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    n_rows: usize,
    n_cols: usize,
) -> Bound<'py, PyArray2<u32>> {
    let cooccur = core::cooccurrence::compute_cooccurrence_sparse_raw(
        indptr.as_slice().unwrap(),
        indices.as_slice().unwrap(),
        n_rows,
        n_cols,
    );

    // Convert to 2D array
    let cooccur_2d: Vec<Vec<u32>> = cooccur
        .chunks(n_cols)
        .map(|chunk| chunk.to_vec())
        .collect();

    PyArray2::from_vec2(py, &cooccur_2d).unwrap()
}

/// Build column bitvecs as u64 arrays for GPU transfer.
#[pyfunction]
fn build_column_bitvecs_u64<'py>(
    py: Python<'py>,
    csr_indptr: PyReadonlyArray1<'py, i64>,
    csr_indices: PyReadonlyArray1<'py, i64>,
    n_rows: usize,
    n_cols: usize,
) -> Bound<'py, PyArray2<u64>> {
    let bitvecs = core::bitvec::build_column_bitvecs_u64_raw(
        csr_indptr.as_slice().unwrap(),
        csr_indices.as_slice().unwrap(),
        n_rows,
        n_cols,
    );

    let n_u64s = core::bitvec::words_for_rows(n_rows);

    // Convert to 2D array [n_cols, n_u64s]
    let bitvecs_2d: Vec<Vec<u64>> = bitvecs
        .chunks(n_u64s)
        .map(|chunk| chunk.to_vec())
        .collect();

    PyArray2::from_vec2(py, &bitvecs_2d).unwrap()
}

/// Convert bitvector AND results to CSR tid-sets (reverse direction).
/// Takes flat 2D bitvec array and returns (offsets, indices) CSR arrays.
#[pyfunction]
fn bitvec_to_tidsets<'py>(
    py: Python<'py>,
    bitvecs: PyReadonlyArray2<'py, u64>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i32>>) {
    let shape = bitvecs.shape();
    let n_itemsets = shape[0];
    let n_u64s = shape[1];
    let flat = bitvecs.as_slice().unwrap();

    let (offsets, indices) = core::bitvec::bitvec_to_tidsets_raw(flat, n_itemsets, n_u64s);

    (
        PyArray1::from_vec(py, offsets),
        PyArray1::from_vec(py, indices),
    )
}

/// Convert boolean predicate matrix to transaction item lists.
#[pyfunction]
fn prepare_transactions(
    _py: Python,
    predicate_matrix: PyReadonlyArray2<bool>,
) -> PyResult<Vec<Vec<usize>>> {
    let matrix = predicate_matrix.as_array();
    let n_rows = matrix.nrows();
    let n_cols = matrix.ncols();

    // Convert to contiguous slice
    let matrix_slice: Vec<bool> = matrix.iter().copied().collect();

    let transactions = core::matrix::prepare_transactions_raw(
        &matrix_slice,
        n_rows,
        n_cols,
    );

    Ok(transactions)
}

/// Build CSR matrix from COO format using parallel sorting.
#[pyfunction]
fn build_csr_from_coo<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<'py, i64>,
    cols: PyReadonlyArray1<'py, i64>,
    n_rows: usize,
    _n_cols: usize,  // Not used but kept for API consistency
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let (indptr, indices) = core::matrix::build_csr_from_coo_raw(
        rows.as_slice().unwrap(),
        cols.as_slice().unwrap(),
        n_rows,
    );

    (
        PyArray1::from_vec(py, indptr),
        PyArray1::from_vec(py, indices),
    )
}

/// Generate random CSR matrix directly in Rust.
#[pyfunction]
fn generate_random_csr<'py>(
    py: Python<'py>,
    n_rows: usize,
    n_cols: usize,
    avg_items_per_row: usize,
    seed: u64,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let (indptr, indices) = core::matrix::generate_random_csr_raw(
        n_rows,
        n_cols,
        avg_items_per_row,
        seed,
    );

    (
        PyArray1::from_vec(py, indptr),
        PyArray1::from_vec(py, indices),
    )
}

/// Generate bootstrapped CSR matrix by sampling source transactions with replacement.
#[pyfunction]
fn generate_bootstrap_csr<'py>(
    py: Python<'py>,
    source_data: Vec<Vec<i64>>,
    n_rows: usize,
    n_cols: usize,
    seed: u64,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let (indptr, indices) = core::matrix::generate_bootstrap_csr_raw(
        &source_data,
        n_rows,
        n_cols,
        seed,
    );

    (
        PyArray1::from_vec(py, indptr),
        PyArray1::from_vec(py, indices),
    )
}

/// Get the number of threads Rayon will use for parallel operations.
#[pyfunction]
fn get_num_threads() -> usize {
    core::utils::get_num_threads()
}

/// Set the number of threads Rayon will use for parallel operations.
#[pyfunction]
fn set_num_threads(n_threads: usize) {
    core::utils::set_num_threads(n_threads);
}

// =============================================================================
// Phase 2: New Functions for CSR-to-CSR Workflow
// =============================================================================

/// Generate k=2 candidate pairs from frequent 1-itemsets.
///
/// Given a sorted list of frequent items, generates all pairs (i, j) where i < j.
#[pyfunction]
fn generate_candidates_k2(frequent_items: Vec<usize>) -> Vec<(usize, usize)> {
    core::candidates::generate_candidates_k2(&frequent_items)
}

/// Generate k+1 candidates from frequent k-itemsets using the Apriori principle.
#[pyfunction]
fn generate_candidates_kplus1(frequent_k: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    core::candidates::generate_candidates_kplus1(&frequent_k)
}

/// Run complete Apriori algorithm on CSR matrix.
///
/// This runs the entire Apriori algorithm in Rust, returning all frequent itemsets.
///
/// # Arguments
/// * `csr_indptr` - CSR row pointers (n_rows + 1 elements)
/// * `csr_indices` - CSR column indices
/// * `n_rows` - Number of transactions
/// * `n_cols` - Number of items
/// * `min_support` - Minimum support threshold (0.0 to 1.0)
/// * `max_length` - Maximum itemset length (0 = unlimited)
///
/// # Returns
/// Tuple of (itemsets, counts) where itemsets is a list of item index lists
#[pyfunction]
fn apriori_from_csr<'py>(
    _py: Python<'py>,
    csr_indptr: PyReadonlyArray1<'py, i64>,
    csr_indices: PyReadonlyArray1<'py, i64>,
    n_rows: usize,
    n_cols: usize,
    min_support: f64,
    max_length: usize,
) -> (Vec<Vec<usize>>, Vec<u32>) {
    let result = core::apriori::apriori_from_csr(
        csr_indptr.as_slice().unwrap(),
        csr_indices.as_slice().unwrap(),
        n_rows,
        n_cols,
        min_support,
        max_length,
    );
    result.flatten()
}

/// Build CSR matrix from list of transactions.
///
/// Creates a CSR matrix directly from transaction lists, avoiding Python-side conversion.
#[pyfunction]
fn build_csr_from_transactions<'py>(
    py: Python<'py>,
    transactions: Vec<Vec<i64>>,
    n_items: usize,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let (indptr, indices) = core::apriori::build_csr_from_transactions(&transactions, n_items);
    (
        PyArray1::from_vec(py, indptr),
        PyArray1::from_vec(py, indices),
    )
}

// =============================================================================
// Phase 3: K>=3 Group Building (GPU bottleneck elimination)
// =============================================================================

/// Build K>=3 prefix groups from flat frequent itemset array.
///
/// Takes a (n_freq, k) int32 array and returns group arrays for GPU upload.
/// Uses Rayon parallel sort — replaces numpy lexsort with ~9x less memory
/// and GIL-free execution.
///
/// Returns tuple of (prefix_items, prefix_offsets, suffixes, suffix_offsets,
///                    cumulative_pairs, total_candidates) or None.
#[pyfunction]
fn build_k3plus_groups_from_flat<'py>(
    py: Python<'py>,
    freq_flat: PyReadonlyArray2<'py, i32>,
) -> Option<(
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    i64,
)> {
    let n_freq = freq_flat.shape()[0];
    let k = freq_flat.shape()[1];

    // Zero-copy access for C-contiguous arrays; copy fallback for non-contiguous
    let data_owned: Vec<i32>;
    let data: &[i32] = match freq_flat.as_slice() {
        Ok(s) => s,
        Err(_) => {
            let array = freq_flat.as_array();
            data_owned = array.iter().copied().collect();
            &data_owned
        }
    };

    // Release GIL during heavy computation (parallel sort + boundary scan)
    #[allow(deprecated)]  // allow_threads → detach in PyO3 0.28
    let result = py.allow_threads(|| {
        core::groups::build_k3plus_groups_from_flat_raw(data, n_freq, k)
    })?;

    Some((
        PyArray1::from_vec(py, result.prefix_items),
        PyArray1::from_vec(py, result.prefix_offsets),
        PyArray1::from_vec(py, result.suffixes),
        PyArray1::from_vec(py, result.suffix_offsets),
        PyArray1::from_vec(py, result.cumulative_pairs),
        result.total_candidates,
    ))
}

// =============================================================================
// Phase 4: Pruning Functions (K=4 regression elimination)
// =============================================================================

/// Prune closed itemsets: remove itemsets whose count equals a (k-1)-subset's count.
///
/// Returns boolean mask (true = keep, false = prune). Uses Rayon parallel iteration
/// over current itemsets with binary-search lookup into prev-level counts (R2).
#[pyfunction]
fn prune_closed_flat<'py>(
    py: Python<'py>,
    current_flat: PyReadonlyArray2<'py, i32>,
    current_counts: PyReadonlyArray1<'py, i64>,
    prev_flat: PyReadonlyArray2<'py, i32>,
    prev_counts: PyReadonlyArray1<'py, i64>,
) -> Bound<'py, PyArray1<bool>> {
    let n_current = current_flat.shape()[0];
    let k = current_flat.shape()[1];
    let n_prev = prev_flat.shape()[0];

    // Get contiguous slices (zero-copy for C-contiguous, copy fallback)
    let cur_flat_owned: Vec<i32>;
    let cur_flat: &[i32] = match current_flat.as_slice() {
        Ok(s) => s,
        Err(_) => {
            cur_flat_owned = current_flat.as_array().iter().copied().collect();
            &cur_flat_owned
        }
    };

    let cur_counts_owned: Vec<i64>;
    let cur_counts: &[i64] = match current_counts.as_slice() {
        Ok(s) => s,
        Err(_) => {
            cur_counts_owned = current_counts.as_array().iter().copied().collect();
            &cur_counts_owned
        }
    };

    let prev_flat_owned: Vec<i32>;
    let prev_flat_slice: &[i32] = match prev_flat.as_slice() {
        Ok(s) => s,
        Err(_) => {
            prev_flat_owned = prev_flat.as_array().iter().copied().collect();
            &prev_flat_owned
        }
    };

    let prev_counts_owned: Vec<i64>;
    let prev_counts_slice: &[i64] = match prev_counts.as_slice() {
        Ok(s) => s,
        Err(_) => {
            prev_counts_owned = prev_counts.as_array().iter().copied().collect();
            &prev_counts_owned
        }
    };

    #[allow(deprecated)]
    let mask = py.allow_threads(|| {
        core::groups::prune_closed_flat_raw(
            cur_flat,
            cur_counts,
            prev_flat_slice,
            prev_counts_slice,
            n_current,
            n_prev,
            k,
        )
    });

    PyArray1::from_vec(py, mask)
}

/// Compact variant: prune non-closed itemsets and return pruned (flat, counts)
/// arrays directly — eliminates the Python-side numpy fancy-index bottleneck.
///
/// Replaces the bool-mask roundtrip + Python
/// `current_flat[mask]` with sequential Rust `extend_from_slice` compaction.
/// Verified ~10-13× speedup on the materialization step (Auditor microbench).
///
/// Returns `(pruned_flat_1d, pruned_counts, n_kept)` where `pruned_flat_1d`
/// is shape `(n_kept * k,)` int32 and the caller reshapes to `(n_kept, k)`.
/// `n_kept` is returned explicitly so the Python wrapper avoids a redundant
/// `mask.sum()` for logging.
#[pyfunction]
fn prune_closed_flat_compact<'py>(
    py: Python<'py>,
    current_flat: PyReadonlyArray2<'py, i32>,
    current_counts: PyReadonlyArray1<'py, i64>,
    prev_flat: PyReadonlyArray2<'py, i32>,
    prev_counts: PyReadonlyArray1<'py, i64>,
) -> (Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i64>>, usize) {
    let n_current = current_flat.shape()[0];
    let k = current_flat.shape()[1];
    let n_prev = prev_flat.shape()[0];

    // Get contiguous slices (zero-copy for C-contiguous, copy fallback)
    let cur_flat_owned: Vec<i32>;
    let cur_flat: &[i32] = match current_flat.as_slice() {
        Ok(s) => s,
        Err(_) => {
            cur_flat_owned = current_flat.as_array().iter().copied().collect();
            &cur_flat_owned
        }
    };

    let cur_counts_owned: Vec<i64>;
    let cur_counts: &[i64] = match current_counts.as_slice() {
        Ok(s) => s,
        Err(_) => {
            cur_counts_owned = current_counts.as_array().iter().copied().collect();
            &cur_counts_owned
        }
    };

    let prev_flat_owned: Vec<i32>;
    let prev_flat_slice: &[i32] = match prev_flat.as_slice() {
        Ok(s) => s,
        Err(_) => {
            prev_flat_owned = prev_flat.as_array().iter().copied().collect();
            &prev_flat_owned
        }
    };

    let prev_counts_owned: Vec<i64>;
    let prev_counts_slice: &[i64] = match prev_counts.as_slice() {
        Ok(s) => s,
        Err(_) => {
            prev_counts_owned = prev_counts.as_array().iter().copied().collect();
            &prev_counts_owned
        }
    };

    #[allow(deprecated)]
    let (out_flat, out_counts) = py.allow_threads(|| {
        core::groups::prune_closed_flat_compact_raw(
            cur_flat,
            cur_counts,
            prev_flat_slice,
            prev_counts_slice,
            n_current,
            n_prev,
            k,
        )
    });

    let n_kept = out_counts.len();
    let flat_arr = PyArray1::from_vec(py, out_flat);
    let counts_arr = PyArray1::from_vec(py, out_counts);
    (flat_arr, counts_arr, n_kept)
}

/// Parallel unique-column extraction from a flat
/// int32 array. Replaces single-threaded `np.unique(current_flat)` which on
/// (430M, 6) int32 = 2.58B elements was ~30-90s sequential sort+dedup. Used
/// by apriori.py's `current_live_mgpu = set(...)` step between K-transitions.
///
/// Sub-second op 2.6B elements via Rayon atomic-bitset parallel mark.
/// Returns sorted ascending Vec<i32> of unique column indices.
#[pyfunction]
fn unique_columns_from_flat<'py>(
    py: Python<'py>,
    flat: PyReadonlyArray1<'py, i32>,
) -> Bound<'py, PyArray1<i32>> {
    let flat_owned: Vec<i32>;
    let flat_slice: &[i32] = match flat.as_slice() {
        Ok(s) => s,
        Err(_) => {
            flat_owned = flat.as_array().iter().copied().collect();
            &flat_owned
        }
    };

    #[allow(deprecated)]
    let unique = py.allow_threads(|| core::groups::unique_columns_from_flat_raw(flat_slice));

    PyArray1::from_vec(py, unique)
}

/// Apriori-prune prefix groups: remove suffix pairs whose candidates have
/// non-frequent (k-1)-subsets.
///
/// Takes group arrays + prev_flat, returns pruned group arrays or None.
/// Uses Rayon parallel iteration over groups with HashSet membership tests.
#[pyfunction]
fn prune_groups_apriori<'py>(
    py: Python<'py>,
    prefix_items: PyReadonlyArray1<'py, i32>,
    prefix_offsets: PyReadonlyArray1<'py, i64>,
    suffixes: PyReadonlyArray1<'py, i32>,
    suffix_offsets: PyReadonlyArray1<'py, i64>,
    cumulative_pairs: PyReadonlyArray1<'py, i64>,
    total_candidates: i64,
    prev_flat: PyReadonlyArray2<'py, i32>,
) -> Option<(
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    i64,
)> {
    let n_prev = prev_flat.shape()[0];
    let k = prev_flat.shape()[1] + 1; // prev is (k-1), candidates are k-length

    // Build the K3PlusGroupsResult from arrays (safe fallback for non-contiguous)
    fn slice_or_copy_i32(arr: &PyReadonlyArray1<'_, i32>) -> Vec<i32> {
        match arr.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => arr.as_array().iter().copied().collect(),
        }
    }
    fn slice_or_copy_i64(arr: &PyReadonlyArray1<'_, i64>) -> Vec<i64> {
        match arr.as_slice() {
            Ok(s) => s.to_vec(),
            Err(_) => arr.as_array().iter().copied().collect(),
        }
    }
    let groups = core::groups::K3PlusGroupsResult {
        prefix_items: slice_or_copy_i32(&prefix_items),
        prefix_offsets: slice_or_copy_i64(&prefix_offsets),
        suffixes: slice_or_copy_i32(&suffixes),
        suffix_offsets: slice_or_copy_i64(&suffix_offsets),
        cumulative_pairs: slice_or_copy_i64(&cumulative_pairs),
        total_candidates,
    };

    let prev_flat_owned: Vec<i32>;
    let prev_flat_slice: &[i32] = match prev_flat.as_slice() {
        Ok(s) => s,
        Err(_) => {
            prev_flat_owned = prev_flat.as_array().iter().copied().collect();
            &prev_flat_owned
        }
    };

    #[allow(deprecated)]
    let result = py.allow_threads(|| {
        core::groups::prune_groups_apriori_raw(&groups, prev_flat_slice, n_prev, k)
    })?;

    Some((
        PyArray1::from_vec(py, result.prefix_items),
        PyArray1::from_vec(py, result.prefix_offsets),
        PyArray1::from_vec(py, result.suffixes),
        PyArray1::from_vec(py, result.suffix_offsets),
        PyArray1::from_vec(py, result.cumulative_pairs),
        result.total_candidates,
    ))
}

// =============================================================================
// Python Module Definition
// =============================================================================

/// Python module definition.
#[pymodule]
fn et_miner_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core counting functions
    m.add_function(wrap_pyfunction!(count_itemsets_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(count_itemsets_simd, m)?)?;
    m.add_function(wrap_pyfunction!(count_itemsets_parallel, m)?)?;

    // Co-occurrence computation
    m.add_function(wrap_pyfunction!(compute_cooccurrence_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(compute_cooccurrence_matrix, m)?)?;

    // GPU acceleration support
    m.add_function(wrap_pyfunction!(build_column_bitvecs_u64, m)?)?;
    m.add_function(wrap_pyfunction!(bitvec_to_tidsets, m)?)?;  // V3: reverse direction

    // Fast CSR construction
    m.add_function(wrap_pyfunction!(build_csr_from_coo, m)?)?;
    m.add_function(wrap_pyfunction!(generate_random_csr, m)?)?;
    m.add_function(wrap_pyfunction!(generate_bootstrap_csr, m)?)?;

    // Transaction preparation
    m.add_function(wrap_pyfunction!(prepare_transactions, m)?)?;

    // Thread control
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;

    // Phase 2: CSR-to-CSR Apriori workflow
    m.add_function(wrap_pyfunction!(generate_candidates_k2, m)?)?;
    m.add_function(wrap_pyfunction!(generate_candidates_kplus1, m)?)?;
    m.add_function(wrap_pyfunction!(apriori_from_csr, m)?)?;
    m.add_function(wrap_pyfunction!(build_csr_from_transactions, m)?)?;

    // Phase 3: K>=3 group building (GPU bottleneck elimination)
    m.add_function(wrap_pyfunction!(build_k3plus_groups_from_flat, m)?)?;

    // Phase 4: Pruning functions (K=4 regression elimination)
    m.add_function(wrap_pyfunction!(prune_closed_flat, m)?)?;
    m.add_function(wrap_pyfunction!(prune_closed_flat_compact, m)?)?;
    m.add_function(wrap_pyfunction!(prune_groups_apriori, m)?)?;

    // Parallel unique-column extraction
    m.add_function(wrap_pyfunction!(unique_columns_from_flat, m)?)?;

    // Version info
    m.add("__version__", "0.1.0")?;
    m.add("__author__",  "E. Ahmic")?;

    Ok(())
}
