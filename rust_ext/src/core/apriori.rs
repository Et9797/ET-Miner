//! Complete Apriori algorithm implementation in Rust.
//!
//! This module provides a full CSR-to-results Apriori implementation,
//! eliminating all Python loops from the hot path.

use rayon::prelude::*;
use super::matrix::CscMatrix;
use super::candidates::{generate_candidates_k2, generate_candidates_kplus1};

/// Exact `ceil(min_support * n_rows)` under the shortest-round-tripping-decimal
/// contract, mirroring Python's `math.ceil(Fraction(str(s)) * n)`.
///
/// `min_support` denotes the shortest decimal that round-trips to the given
/// `f64`, **not** the exact binary value of that `f64`. That is a contract, not
/// a law, and it is shared with `core.result._min_count` and
/// `synthetic.SynthSpec.min_count`; all three are checked against the single
/// table in `tests/fixtures/min_count_cases.json`, because two hand-maintained
/// tables drift -- which is precisely how defect #11 arose.
///
/// The naive `(min_support * n_rows as f64).ceil()` is wrong on the boundary:
/// 0.07 has no exact binary64 form, so `0.07 * 10000.0` is 700.0000000000001
/// and ceils to 701 where the exact ceiling is 700, dropping every itemset at
/// exactly that count and the whole cone above it.
///
/// Rust's `Display` for `f64` emits the shortest round-tripping decimal and
/// never uses exponent form, so a tiny value prints as a long run of zeros
/// followed by at most 17 significant digits. Splitting the *significand* from
/// the *scale* -- rather than building a `10^len` denominator from the whole
/// fractional part -- is what keeps this inside `u128`: `digits <= 10^17` and
/// `n_rows <= 2^63` give `digits * n_rows <= 9.2e35`, comfortably under
/// `u128::MAX` (~3.4e38).
pub fn exact_min_count(min_support: f64, n_rows: usize) -> u64 {
    if !min_support.is_finite() || min_support <= 0.0 {
        return 0;
    }
    let s = format!("{}", min_support);
    let (int_part, frac_part) = match s.split_once('.') {
        Some((a, b)) => (a, b),
        None => (s.as_str(), ""),
    };
    let joined = format!("{}{}", int_part, frac_part);
    let sig = joined.trim_start_matches('0');
    if sig.is_empty() {
        return 0;
    }
    let digits: u128 = sig
        .parse()
        .expect("shortest round-tripping decimal has <= 17 significant digits");
    let d = frac_part.len() as u32;
    let n = n_rows as u128;
    if d > 38 {
        // 10^d > 1e38 > digits * n for any representable n_rows, so the exact
        // quotient lies in (0, 1) and the ceiling of a positive value is 1.
        return 1;
    }
    let den = 10u128.pow(d);
    let num = digits
        .checked_mul(n)
        .expect("digits * n_rows overflows u128");
    ((num + den - 1) / den) as u64
}

/// Result of Apriori algorithm execution.
#[derive(Debug, Clone)]
pub struct AprioriResult {
    /// All frequent itemsets found, grouped by size k
    pub itemsets: Vec<Vec<Vec<usize>>>,
    /// Support counts for each itemset, matching itemsets structure
    pub counts: Vec<Vec<u32>>,
    /// Total number of transactions
    pub n_transactions: usize,
}

impl AprioriResult {
    /// Get all itemsets flattened into a single list with their counts.
    pub fn flatten(&self) -> (Vec<Vec<usize>>, Vec<u32>) {
        let mut all_itemsets = Vec::new();
        let mut all_counts = Vec::new();

        for (itemsets_k, counts_k) in self.itemsets.iter().zip(&self.counts) {
            all_itemsets.extend(itemsets_k.iter().cloned());
            all_counts.extend(counts_k.iter().copied());
        }

        (all_itemsets, all_counts)
    }

    /// Get total number of frequent itemsets found.
    pub fn total_itemsets(&self) -> usize {
        self.itemsets.iter().map(|v| v.len()).sum()
    }
}

/// Run complete Apriori algorithm on CSR matrix.
///
/// This is the main entry point for running Apriori entirely in Rust,
/// eliminating Python overhead for candidate generation and counting.
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
/// AprioriResult containing all frequent itemsets and their counts
///
/// # Example
/// ```
/// // CSR matrix with 4 transactions, 5 items
/// let indptr = vec![0i64, 3, 6, 9, 11];
/// let indices = vec![0i64, 1, 2, 1, 2, 3, 0, 2, 4, 1, 2];
///
/// let result = apriori_from_csr(&indptr, &indices, 4, 5, 0.5, 0);
/// ```
pub fn apriori_from_csr(
    csr_indptr: &[i64],
    csr_indices: &[i64],
    n_rows: usize,
    n_cols: usize,
    min_support: f64,
    max_length: usize,
) -> AprioriResult {
    let min_count = exact_min_count(min_support, n_rows) as u32;
    let max_k = if max_length == 0 { n_cols } else { max_length };

    // Convert CSR to CSC for efficient column access
    let csc = CscMatrix::from_csr(csr_indptr, csr_indices, n_rows, n_cols);

    // Pre-build column bitvecs for SIMD counting
    let column_bitvecs: Vec<Vec<u64>> = build_column_bitvecs(&csc, n_rows);

    let mut result = AprioriResult {
        itemsets: Vec::new(),
        counts: Vec::new(),
        n_transactions: n_rows,
    };

    // Phase 1: Find frequent 1-itemsets
    let (frequent_1, counts_1) = find_frequent_1(&csc, n_rows, min_count);

    if frequent_1.is_empty() {
        return result;
    }

    result.itemsets.push(frequent_1.iter().map(|&i| vec![i]).collect());
    result.counts.push(counts_1);

    if max_k == 1 {
        return result;
    }

    // Phase 2: Find frequent 2-itemsets
    let candidates_2 = generate_candidates_k2(&frequent_1);
    let (frequent_2, counts_2) = count_and_filter_k2(&column_bitvecs, n_rows, &candidates_2, min_count);

    if frequent_2.is_empty() {
        return result;
    }

    result.itemsets.push(frequent_2.clone());
    result.counts.push(counts_2);

    if max_k == 2 {
        return result;
    }

    // Phase 3+: Find frequent k-itemsets for k >= 3
    let mut current_frequent = frequent_2;
    let mut k = 3;

    while !current_frequent.is_empty() && k <= max_k {
        let candidates_k = generate_candidates_kplus1(&current_frequent);

        if candidates_k.is_empty() {
            break;
        }

        let (frequent_k, counts_k) = count_and_filter_kplus(&column_bitvecs, n_rows, &candidates_k, min_count);

        if frequent_k.is_empty() {
            break;
        }

        result.itemsets.push(frequent_k.clone());
        result.counts.push(counts_k);

        current_frequent = frequent_k;
        k += 1;
    }

    result
}

/// Build u64 bitvecs for each column (item).
fn build_column_bitvecs(csc: &CscMatrix, n_rows: usize) -> Vec<Vec<u64>> {
    let n_u64s = n_rows.div_ceil(64);

    (0..csc.n_cols)
        .into_par_iter()
        .map(|col| {
            let mut bitvec = vec![0u64; n_u64s];
            for &row_idx in csc.get_column_rows(col) {
                let row = row_idx as usize;
                bitvec[row >> 6] |= 1u64 << (row & 63);
            }
            bitvec
        })
        .collect()
}

/// Find frequent 1-itemsets by counting column densities.
fn find_frequent_1(csc: &CscMatrix, _n_rows: usize, min_count: u32) -> (Vec<usize>, Vec<u32>) {
    let counts: Vec<(usize, u32)> = (0..csc.n_cols)
        .into_par_iter()
        .map(|col| {
            let count = csc.get_column_rows(col).len() as u32;
            (col, count)
        })
        .filter(|&(_, count)| count >= min_count)
        .collect();

    let mut frequent: Vec<usize> = counts.iter().map(|&(col, _)| col).collect();
    frequent.sort_unstable();

    let support_counts: Vec<u32> = frequent
        .iter()
        .map(|&col| csc.get_column_rows(col).len() as u32)
        .collect();

    (frequent, support_counts)
}

/// Count and filter k=2 candidates using bitvec intersection.
fn count_and_filter_k2(
    bitvecs: &[Vec<u64>],
    n_rows: usize,
    candidates: &[(usize, usize)],
    min_count: u32,
) -> (Vec<Vec<usize>>, Vec<u32>) {
    let results: Vec<(Vec<usize>, u32)> = candidates
        .par_iter()
        .filter_map(|&(i, j)| {
            let count = count_bitvec_intersection_2(&bitvecs[i], &bitvecs[j], n_rows);
            if count >= min_count {
                Some((vec![i, j], count))
            } else {
                None
            }
        })
        .collect();

    let itemsets: Vec<Vec<usize>> = results.iter().map(|(is, _)| is.clone()).collect();
    let counts: Vec<u32> = results.iter().map(|&(_, c)| c).collect();

    (itemsets, counts)
}

/// Count and filter k>2 candidates using bitvec intersection.
fn count_and_filter_kplus(
    bitvecs: &[Vec<u64>],
    n_rows: usize,
    candidates: &[Vec<usize>],
    min_count: u32,
) -> (Vec<Vec<usize>>, Vec<u32>) {
    let results: Vec<(Vec<usize>, u32)> = candidates
        .par_iter()
        .filter_map(|itemset| {
            let count = count_bitvec_intersection_n(bitvecs, itemset, n_rows);
            if count >= min_count {
                Some((itemset.clone(), count))
            } else {
                None
            }
        })
        .collect();

    let itemsets: Vec<Vec<usize>> = results.iter().map(|(is, _)| is.clone()).collect();
    let counts: Vec<u32> = results.iter().map(|&(_, c)| c).collect();

    (itemsets, counts)
}

/// Count support by intersecting two bitvecs.
#[inline]
fn count_bitvec_intersection_2(bv1: &[u64], bv2: &[u64], _n_rows: usize) -> u32 {
    bv1.iter()
        .zip(bv2.iter())
        .map(|(&a, &b)| (a & b).count_ones())
        .sum()
}

/// Count support by intersecting multiple bitvecs.
#[inline]
fn count_bitvec_intersection_n(bitvecs: &[Vec<u64>], items: &[usize], _n_rows: usize) -> u32 {
    if items.is_empty() {
        return 0;
    }

    let n_words = bitvecs[items[0]].len();
    let mut count = 0u32;

    for (word_idx, &word) in bitvecs[items[0]].iter().enumerate().take(n_words) {
        let mut result = word;
        for &item in &items[1..] {
            result &= bitvecs[item][word_idx];
        }
        count += result.count_ones();
    }

    count
}

/// Build CSR matrix from list of transactions.
///
/// This is a convenience function that creates a CSR matrix directly from
/// transaction lists, avoiding Python-side conversion.
///
/// # Arguments
/// * `transactions` - List of transactions, each transaction is a list of item indices
/// * `n_items` - Total number of unique items
///
/// # Returns
/// Tuple of (indptr, indices) in CSR format
pub fn build_csr_from_transactions(
    transactions: &[Vec<i64>],
    n_items: usize,
) -> (Vec<i64>, Vec<i64>) {
    let n_rows = transactions.len();

    // Calculate total nnz
    let total_nnz: usize = transactions.iter().map(|t| t.len()).sum();

    // Pre-allocate
    let mut indptr = Vec::with_capacity(n_rows + 1);
    let mut indices = Vec::with_capacity(total_nnz);

    indptr.push(0);
    let mut current_ptr = 0i64;

    for transaction in transactions {
        // Filter, sort, and deduplicate items
        let mut items: Vec<i64> = transaction
            .iter()
            .filter(|&&item| item >= 0 && (item as usize) < n_items)
            .copied()
            .collect();
        items.sort_unstable();
        items.dedup();

        indices.extend(&items);
        current_ptr += items.len() as i64;
        indptr.push(current_ptr);
    }

    (indptr, indices)
}

#[cfg(test)]
mod tests {

    /// The min-count rule has three implementations (here,
    /// `core.result._min_count`, `synthetic.SynthSpec.min_count`) and ONE
    /// table. Defect #11 existed because all three carried the same wrong
    /// expression and therefore agreed with each other; a second
    /// hand-maintained table here would reproduce exactly that failure mode,
    /// so this reads the file `tests/test_min_count.py` reads.
    #[test]
    fn exact_min_count_matches_the_shared_fixture() {
        const FIXTURE: &str = include_str!("../../../tests/fixtures/min_count_cases.json");
        let doc: serde_json::Value = serde_json::from_str(FIXTURE).expect("fixture parses");
        let cases = doc["cases"].as_array().expect("cases array");
        assert!(cases.len() >= 15, "fixture shrank: {} cases", cases.len());

        let mut failures = Vec::new();
        for case in cases {
            let s = case["min_support"].as_f64().expect("min_support");
            let n = case["n_rows"].as_u64().expect("n_rows") as usize;
            let want = case["expected"].as_u64().expect("expected");
            let got = exact_min_count(s, n);
            if got != want {
                failures.push(format!(
                    "min_support={} n_rows={} expected={} got={} ({})",
                    s, n, want, got,
                    case["why"].as_str().unwrap_or("")
                ));
            }
        }
        assert!(failures.is_empty(), "min-count disagreements:\n  {}", failures.join("\n  "));
    }

    /// The naive `(s * n as f64).ceil()` must actually be wrong on some fixture
    /// case, or the test above proves nothing about what was fixed.
    #[test]
    fn the_fixture_discriminates_against_the_float_expression() {
        const FIXTURE: &str = include_str!("../../../tests/fixtures/min_count_cases.json");
        let doc: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        let n_diff = doc["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| {
                let s = c["min_support"].as_f64().unwrap();
                let n = c["n_rows"].as_u64().unwrap() as usize;
                let naive = (s * n as f64).ceil() as u64;
                naive != c["expected"].as_u64().unwrap()
            })
            .count();
        assert!(n_diff >= 3, "fixture no longer discriminates: only {} cases differ", n_diff);
    }

    use super::*;

    fn create_test_csr() -> (Vec<i64>, Vec<i64>, usize, usize) {
        // 5 transactions, 5 items:
        // T0: {0, 1, 2}    - items A, B, C
        // T1: {1, 2, 3}    - items B, C, D
        // T2: {0, 2, 4}    - items A, C, E
        // T3: {1, 2}       - items B, C
        // T4: {0, 1, 2, 3} - items A, B, C, D
        let indptr = vec![0i64, 3, 6, 9, 11, 15];
        let indices = vec![0i64, 1, 2, 1, 2, 3, 0, 2, 4, 1, 2, 0, 1, 2, 3];
        (indptr, indices, 5, 5)
    }

    #[test]
    fn test_apriori_basic() {
        let (indptr, indices, n_rows, n_cols) = create_test_csr();

        // min_support = 0.4 means min_count = 2
        let result = apriori_from_csr(&indptr, &indices, n_rows, n_cols, 0.4, 0);

        // All items appear at least twice:
        // 0: 3 times, 1: 4 times, 2: 5 times, 3: 2 times, 4: 1 time
        // So frequent 1-itemsets: {0}, {1}, {2}, {3}
        assert!(!result.itemsets.is_empty());

        let (all_itemsets, _all_counts) = result.flatten();
        assert!(all_itemsets.contains(&vec![0]));
        assert!(all_itemsets.contains(&vec![1]));
        assert!(all_itemsets.contains(&vec![2]));
        assert!(all_itemsets.contains(&vec![3]));
        assert!(!all_itemsets.contains(&vec![4])); // 4 appears only once

        // {1, 2} should be frequent (appears in T0, T1, T3, T4 = 4 times)
        assert!(all_itemsets.contains(&vec![1, 2]));
    }

    #[test]
    fn test_apriori_max_length() {
        let (indptr, indices, n_rows, n_cols) = create_test_csr();

        // Limit to k=2
        let result = apriori_from_csr(&indptr, &indices, n_rows, n_cols, 0.4, 2);

        // Should have at most 2 levels (k=1 and k=2)
        assert!(result.itemsets.len() <= 2);

        // No itemsets should be longer than 2
        let (all_itemsets, _) = result.flatten();
        for itemset in all_itemsets {
            assert!(itemset.len() <= 2);
        }
    }

    #[test]
    fn test_build_csr_from_transactions() {
        let transactions = vec![
            vec![0i64, 1, 2],
            vec![1, 2, 3],
            vec![0, 2],
        ];

        let (indptr, indices) = build_csr_from_transactions(&transactions, 5);

        assert_eq!(indptr, vec![0, 3, 6, 8]);
        assert_eq!(indices, vec![0, 1, 2, 1, 2, 3, 0, 2]);
    }

    #[test]
    fn test_build_csr_from_transactions_with_duplicates() {
        // Duplicates should be removed
        let transactions = vec![
            vec![0i64, 1, 1, 2, 0],  // Should become [0, 1, 2]
        ];

        let (indptr, indices) = build_csr_from_transactions(&transactions, 5);

        assert_eq!(indptr, vec![0, 3]);
        assert_eq!(indices, vec![0, 1, 2]);
    }
}
