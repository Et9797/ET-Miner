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

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// Core modules containing pure Rust algorithms
pub mod core;

// Re-export core types for direct Rust usage
pub use core::CscMatrix;

// =============================================================================
// PyO3 Wrappers - Thin bindings that call into core functions
// =============================================================================

// =============================================================================
// FFI boundary helpers
//
// Half this file's entry points already hardened against a non-contiguous
// numpy view (an `arr[::2]`, an F-order array, a strided slice) by copying;
// the other half called `as_slice().unwrap()` and panicked. Under the old
// `panic = "abort"` that was a SIGABRT with no traceback. Both halves now go
// through the same two helpers, so the contract is one rule rather than a
// coin flip over which entry point you happened to call.
// =============================================================================

/// Borrow a 1-D numpy array as a slice, copying only if it is not contiguous.
fn contiguous<'a, T>(arr: &'a PyReadonlyArray1<'a, T>) -> std::borrow::Cow<'a, [T]>
where
    T: numpy::Element + Copy,
{
    match arr.as_slice() {
        Ok(s) => std::borrow::Cow::Borrowed(s),
        Err(_) => std::borrow::Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

/// Validate a CSR shape at the boundary, so a bad shape is a Python exception
/// naming the offending value rather than an index-out-of-bounds panic from
/// somewhere inside the kernel.
fn validate_csr(indptr: &[i64], indices: &[i64], n_rows: usize, n_cols: usize) -> PyResult<()> {
    if indptr.is_empty() {
        return Err(PyValueError::new_err("csr_indptr must have at least one element"));
    }

    // Compared WITHOUT the add. `n_rows + 1` wraps at usize::MAX -- the release
    // profile sets no overflow-checks -- so at n_rows = 2**64-1 the sum was 0,
    // `0 > indptr.len()` was false, the guard was skipped entirely, and the
    // call panicked at the index below. That is the exact in-kernel panic this
    // function exists to convert into a named Python exception.
    if n_rows >= indptr.len() {
        return Err(PyValueError::new_err(format!(
            "n_rows={} requires csr_indptr of length >= {}, got {}",
            n_rows,
            n_rows.saturating_add(1),
            indptr.len()
        )));
    }
    if indptr[0] < 0 {
        return Err(PyValueError::new_err(format!(
            "csr_indptr[0]={} is negative",
            indptr[0]
        )));
    }

    // The kernels slice indices[indptr[i]..indptr[i+1]] for EVERY row, so the
    // whole prefix must be non-decreasing. Bounds-checking only indptr[n_rows]
    // left the in-kernel slice panic reachable on all four entry points this
    // function guards: indptr=[0,5,2] gave "range end index 5 out of range",
    // indptr=[0,2,1] gave "slice index starts at 2 but ends at 1", and a
    // negative interior entry wrapped to 18446744073709551615 because only
    // indptr[n_rows] was ever cast to usize.
    for i in 0..n_rows {
        if indptr[i + 1] < indptr[i] {
            return Err(PyValueError::new_err(format!(
                "csr_indptr must be non-decreasing: indptr[{}]={} > indptr[{}]={}",
                i,
                indptr[i],
                i + 1,
                indptr[i + 1]
            )));
        }
    }

    let nnz = indptr[n_rows] as usize; // non-negative and maximal by the above
    if nnz > indices.len() {
        return Err(PyValueError::new_err(format!(
            "csr_indptr[{}]={} exceeds len(csr_indices)={}",
            n_rows, nnz, indices.len()
        )));
    }

    // Per-row STRICTLY INCREASING, not a global min/max range test. Three
    // separate defects live in this one clause:
    //
    //   * SORTEDNESS: the binary-searching sparse counter that first needed it
    //     (count_itemsets_sparse, removed in 0.4.0) silently UNDERCOUNTED a
    //     row stored descending. No remaining entry point searches a row, but
    //     ascending rows are canonical CSR (scipy's canonical format), and
    //     keeping the rule means a future searching consumer cannot quietly
    //     reintroduce that undercount.
    //   * STRICTNESS, which is a SEPARATE reason and the one most likely to be
    //     relaxed by mistake. Duplicates are idempotent in the bitvec path, so
    //     "non-decreasing" looks sufficient.
    //     It is not: `find_frequent_1` (core/apriori.rs:217 and :230) counts
    //     CSC column-list ENTRIES rather than distinct rows, so one row {0,1}
    //     with column 0 stored twice yields (0,) -> count 2 over n_rows = 1,
    //     i.e. SUPPORT 2.0. A count above n_rows is impossible by construction,
    //     and this is the K=1 level every deeper level's min_count filtering
    //     proceeds from.
    //
    //     Note an anti-monotonicity assertion would NOT catch it: inflating K=1
    //     only moves a superset further below its subsets, so the ordering check
    //     a reader would reach for stays green. The only visible symptom is a
    //     support exceeding 1.0 -- and the predicate for that is `count >
    //     n_rows`, not `>=`, since an item present in every transaction
    //     legitimately counts n_rows.
    //   * checking each row's FIRST element catches a negative sitting beside a
    //     positive. The previous test was `max_idx < 0` over the whole array's
    //     maximum, which only rejects when the LARGEST index is negative, so
    //     indices=[-1, 1] with n_cols=2 cleared the guard and panicked in-kernel.
    //   * checking each row's LAST element bounds the column range, which the
    //     max test did do.
    for i in 0..n_rows {
        let (s, e) = (indptr[i] as usize, indptr[i + 1] as usize);
        if s == e {
            continue;
        }
        let row = &indices[s..e];
        if row[0] < 0 {
            return Err(PyValueError::new_err(format!(
                "csr_indices[{}]={} is negative (row {})",
                s, row[0], i
            )));
        }
        for w in row.windows(2) {
            if w[1] <= w[0] {
                return Err(PyValueError::new_err(format!(
                    "csr_indices must be strictly increasing within each row; \
                     row {} has {} followed by {}",
                    i, w[0], w[1]
                )));
            }
        }
        let last = row[row.len() - 1];
        if last as usize >= n_cols {
            return Err(PyValueError::new_err(format!(
                "column index {} is out of range for n_cols={} (row {})",
                last, n_cols, i
            )));
        }
    }
    Ok(())
}

// =============================================================================
// Full-Rust Apriori over CSR
// =============================================================================

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
) -> PyResult<(Vec<Vec<usize>>, Vec<u32>)> {
    let indptr_c = contiguous(&csr_indptr);
    let indices_c = contiguous(&csr_indices);
    validate_csr(indptr_c.as_ref(), indices_c.as_ref(), n_rows, n_cols)?;
    let result = core::apriori::apriori_from_csr(
        indptr_c.as_ref(),
        indices_c.as_ref(),
        n_rows,
        n_cols,
        min_support,
        max_length,
    );
    Ok(result.flatten())
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
///                    cumulative_pairs, suffix_src_rows, total_candidates) or None.
/// `suffix_src_rows` (int64, parallel to `suffixes`) maps each suffix slot
/// back to its row in `freq_flat`; it is empty unless `with_src_rows=True`
/// (only the sparse-CSR GPU path needs it — 8 B per row otherwise wasted).
#[pyfunction]
#[pyo3(signature = (freq_flat, with_src_rows = false))]
fn build_k3plus_groups_from_flat<'py>(
    py: Python<'py>,
    freq_flat: PyReadonlyArray2<'py, i32>,
    with_src_rows: bool,
) -> Option<(
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
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
        core::groups::build_k3plus_groups_from_flat_raw(data, n_freq, k, with_src_rows)
    })?;

    Some((
        PyArray1::from_vec(py, result.prefix_items),
        PyArray1::from_vec(py, result.prefix_offsets),
        PyArray1::from_vec(py, result.suffixes),
        PyArray1::from_vec(py, result.suffix_offsets),
        PyArray1::from_vec(py, result.cumulative_pairs),
        PyArray1::from_vec(py, result.suffix_src_rows),
        result.total_candidates,
    ))
}

// =============================================================================
// Phase 4: Pruning Functions (K=4 regression elimination)
// =============================================================================

/// Keep only free-sets: remove itemsets whose count equals a (k-1)-subset's count.
///
/// Returns boolean mask (true = keep, false = prune). Uses Rayon parallel iteration
/// over current itemsets with binary-search lookup into prev-level counts (R2).
#[pyfunction]
fn prune_non_free_flat<'py>(
    py: Python<'py>,
    current_flat: PyReadonlyArray2<'py, i32>,
    current_counts: PyReadonlyArray1<'py, i64>,
    prev_flat: PyReadonlyArray2<'py, i32>,
    prev_counts: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
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

    if k >= 2 && !core::groups::is_sorted_by_row(prev_flat_slice, n_prev, k - 1) {
        return Err(PyValueError::new_err(
            "prev_flat must be sorted lexicographically by row for the free-set-prune binary search \
             (K>=3 decode order is j-major within a group, not lex order — lexsort current_flat at level end)",
        ));
    }

    #[allow(deprecated)]
    let mask = py.allow_threads(|| {
        core::groups::prune_non_free_flat_raw(
            cur_flat,
            cur_counts,
            prev_flat_slice,
            prev_counts_slice,
            n_current,
            n_prev,
            k,
        )
    });

    Ok(PyArray1::from_vec(py, mask))
}

// =============================================================================
// Python Module Definition
// =============================================================================

/// Python module definition.
#[pymodule]
fn et_miner_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // GPU acceleration support

    // Thread control

    // Full-Rust Apriori over a CSR matrix
    m.add_function(wrap_pyfunction!(apriori_from_csr, m)?)?;

    // Phase 3: K>=3 group building (GPU bottleneck elimination)
    m.add_function(wrap_pyfunction!(build_k3plus_groups_from_flat, m)?)?;

    // Phase 4: free-set prune
    m.add_function(wrap_pyfunction!(prune_non_free_flat, m)?)?;

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__",  "E. Ahmic")?;

    Ok(())
}
