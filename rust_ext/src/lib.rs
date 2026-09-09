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
use pyo3::exceptions::PyValueError;
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
fn validate_csr(indptr: &[i64], indices: &[i64], n_rows: usize, n_cols: Option<usize>) -> PyResult<()> {
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
    //   * SORTEDNESS: `count_itemsets_sparse_raw` binary-searches each row --
    //     its own comment says "the indices are sorted, so we can use binary
    //     search" -- and nothing validated it. A row stored descending silently
    //     UNDERCOUNTS there (measured 2 against a truth of 3, a 33% loss) while
    //     the SIMD path returns the right answer, so which number a caller got
    //     was decided by `hasattr(rust, "count_itemsets_simd")`, an
    //     optional-feature probe. That is Tier 2 of CLAUDE.md's mandated chain
    //     disagreeing with itself depending on how the wheel was built.
    //   * STRICTNESS, which is a SEPARATE reason and the one most likely to be
    //     relaxed by mistake. Duplicates are harmless to the binary search and
    //     idempotent in the bitvec path, so "non-decreasing" looks sufficient.
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
        if let Some(n_cols) = n_cols {
            let last = row[row.len() - 1];
            if last as usize >= n_cols {
                return Err(PyValueError::new_err(format!(
                    "column index {} is out of range for n_cols={} (row {})",
                    last, n_cols, i
                )));
            }
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (indptr, indices, n_rows, itemsets, n_threads = 0))]
fn count_itemsets_sparse<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    n_rows: usize,
    itemsets: Vec<Vec<usize>>,
    n_threads: usize,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    // The slices are hoisted out of the closure deliberately: a
    // PyReadonlyArray1 is tied to the 'py lifetime and is not Send, so it
    // cannot cross into a rayon pool. &[i64] can.
    let indptr_c = contiguous(&indptr);
    let indices_c = contiguous(&indices);
    let (indptr_s, indices_s) = (indptr_c.as_ref(), indices_c.as_ref());
    validate_csr(indptr_s, indices_s, n_rows, None)?;
    // n_threads == 0 keeps the previous behaviour (the global rayon pool, i.e.
    // every core). Anything else is the caller's n_jobs, honoured for real.
    let counts = core::utils::with_thread_budget(n_threads, || {
        core::counting::count_itemsets_sparse_raw(indptr_s, indices_s, n_rows, &itemsets)
    });
    Ok(PyArray1::from_vec(py, counts))
}

/// Count itemset support using column-based SIMD intersection.
///
/// This is a faster alternative to `count_itemsets_sparse` that uses:
/// 1. CSC format for direct column access (no binary search)
/// 2. Bitvec row masks for efficient set intersection
/// 3. SIMD-friendly popcount for counting
#[pyfunction]
#[pyo3(signature = (csr_indptr, csr_indices, n_rows, n_cols, itemsets, n_threads = 0))]
fn count_itemsets_simd<'py>(
    py: Python<'py>,
    csr_indptr: PyReadonlyArray1<'py, i64>,
    csr_indices: PyReadonlyArray1<'py, i64>,
    n_rows: usize,
    n_cols: usize,
    itemsets: Vec<Vec<usize>>,
    n_threads: usize,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    // Slices hoisted out of the closure: PyReadonlyArray1 is tied to 'py and
    // is not Send, so it cannot cross into a rayon pool. &[i64] can.
    let indptr_c = contiguous(&csr_indptr);
    let indices_c = contiguous(&csr_indices);
    let (indptr_s, indices_s) = (indptr_c.as_ref(), indices_c.as_ref());
    validate_csr(indptr_s, indices_s, n_rows, Some(n_cols))?;
    // n_threads == 0 keeps the previous behaviour (the global rayon pool, i.e.
    // every core). Anything else is the caller's n_jobs, honoured for real.
    let counts = core::utils::with_thread_budget(n_threads, || {
        core::counting::count_itemsets_simd_raw(indptr_s, indices_s, n_rows, n_cols, &itemsets)
    });
    Ok(PyArray1::from_vec(py, counts))
}

/// Build column bitvecs as u64 arrays for GPU transfer.
#[pyfunction]
fn build_column_bitvecs_u64<'py>(
    py: Python<'py>,
    csr_indptr: PyReadonlyArray1<'py, i64>,
    csr_indices: PyReadonlyArray1<'py, i64>,
    n_rows: usize,
    n_cols: usize,
) -> PyResult<Bound<'py, PyArray2<u64>>> {
    let indptr_c = contiguous(&csr_indptr);
    let indices_c = contiguous(&csr_indices);
    validate_csr(indptr_c.as_ref(), indices_c.as_ref(), n_rows, Some(n_cols))?;
    let bitvecs = core::bitvec::build_column_bitvecs_u64_raw(
        indptr_c.as_ref(),
        indices_c.as_ref(),
        n_rows,
        n_cols,
    );

    let n_u64s = core::bitvec::words_for_rows(n_rows);

    // n_rows == 0 gives n_u64s == 0, and `chunks(0)` panics ("chunk size must
    // be non-zero"). An empty transaction set is a legitimate degenerate input
    // -- build_boolean_matrix can produce it -- so return the correctly shaped
    // empty array rather than failing.
    if n_u64s == 0 {
        let empty: Vec<Vec<u64>> = vec![Vec::new(); n_cols];
        return PyArray2::from_vec2(py, &empty)
            .map_err(|e| PyValueError::new_err(format!("could not build empty bitvec array: {}", e)));
    }

    // Convert to 2D array [n_cols, n_u64s]
    let bitvecs_2d: Vec<Vec<u64>> = bitvecs
        .chunks(n_u64s)
        .map(|chunk| chunk.to_vec())
        .collect();

    PyArray2::from_vec2(py, &bitvecs_2d)
        .map_err(|e| PyValueError::new_err(format!("could not build bitvec array: {}", e)))
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
    let flat_owned: Vec<u64>;
    let flat = match bitvecs.as_slice() {
        Ok(sl) => sl,
        Err(_) => {
            flat_owned = bitvecs.as_array().iter().copied().collect();
            &flat_owned
        }
    };

    let (offsets, indices) = core::bitvec::bitvec_to_tidsets_raw(flat, n_itemsets, n_u64s);

    (
        PyArray1::from_vec(py, offsets),
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

/// Get the number of threads Rayon will use for parallel operations.
#[pyfunction]
fn get_num_threads() -> usize {
    core::utils::get_num_threads()
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
    validate_csr(indptr_c.as_ref(), indices_c.as_ref(), n_rows, Some(n_cols))?;
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

/// Compact variant: prune non-free itemsets and return pruned (flat, counts)
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
fn prune_non_free_flat_compact<'py>(
    py: Python<'py>,
    current_flat: PyReadonlyArray2<'py, i32>,
    current_counts: PyReadonlyArray1<'py, i64>,
    prev_flat: PyReadonlyArray2<'py, i32>,
    prev_counts: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i64>>, usize)> {
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
    let (out_flat, out_counts) = py.allow_threads(|| {
        core::groups::prune_non_free_flat_compact_raw(
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
    Ok((flat_arr, counts_arr, n_kept))
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
/// The optional `suffix_src_rows` (int64, parallel to `suffixes`) is carried
/// through the prune slot-for-slot; the result tuple always has it as its
/// sixth element (empty when it was not supplied).
#[pyfunction]
#[pyo3(signature = (prefix_items, prefix_offsets, suffixes, suffix_offsets, cumulative_pairs, total_candidates, prev_flat, suffix_src_rows = None))]
fn prune_groups_apriori<'py>(
    py: Python<'py>,
    prefix_items: PyReadonlyArray1<'py, i32>,
    prefix_offsets: PyReadonlyArray1<'py, i64>,
    suffixes: PyReadonlyArray1<'py, i32>,
    suffix_offsets: PyReadonlyArray1<'py, i64>,
    cumulative_pairs: PyReadonlyArray1<'py, i64>,
    total_candidates: i64,
    prev_flat: PyReadonlyArray2<'py, i32>,
    suffix_src_rows: Option<PyReadonlyArray1<'py, i64>>,
) -> Option<(
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
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
        suffix_src_rows: suffix_src_rows.as_ref().map(slice_or_copy_i64).unwrap_or_default(),
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
        PyArray1::from_vec(py, result.suffix_src_rows),
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

    // GPU acceleration support
    m.add_function(wrap_pyfunction!(build_column_bitvecs_u64, m)?)?;
    m.add_function(wrap_pyfunction!(bitvec_to_tidsets, m)?)?;  // V3: reverse direction

    // Fast CSR construction (synthetic-data generation for benchmarks)
    m.add_function(wrap_pyfunction!(generate_random_csr, m)?)?;

    // Thread control
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;

    // Full-Rust Apriori over a CSR matrix
    m.add_function(wrap_pyfunction!(apriori_from_csr, m)?)?;

    // Phase 3: K>=3 group building (GPU bottleneck elimination)
    m.add_function(wrap_pyfunction!(build_k3plus_groups_from_flat, m)?)?;

    // Phase 4: Pruning functions (K=4 regression elimination)
    m.add_function(wrap_pyfunction!(prune_non_free_flat, m)?)?;
    m.add_function(wrap_pyfunction!(prune_non_free_flat_compact, m)?)?;
    m.add_function(wrap_pyfunction!(prune_groups_apriori, m)?)?;

    // Parallel unique-column extraction
    m.add_function(wrap_pyfunction!(unique_columns_from_flat, m)?)?;

    // Version info
    m.add("__version__", "0.3.0")?;
    m.add("__author__",  "E. Ahmic")?;

    Ok(())
}
