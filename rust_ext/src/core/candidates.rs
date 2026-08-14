//! Candidate generation algorithms for Apriori.
//!
//! This module provides efficient candidate generation for itemsets,
//! replacing Python loops with Rust parallelism.

use rayon::prelude::*;
use std::collections::HashSet;

/// Generate k=2 candidate pairs from frequent 1-itemsets.
///
/// Given a sorted list of frequent items, generates all pairs (i, j) where i < j.
/// This is more efficient than Python's itertools.combinations for large item sets.
///
/// # Arguments
/// * `frequent_items` - Sorted list of frequent item indices
///
/// # Returns
/// Vec of (item_i, item_j) pairs where item_i < item_j
///
/// # Example
/// ```
/// let items = vec![0, 2, 5, 7];
/// let pairs = generate_candidates_k2(&items);
/// // Returns: [(0,2), (0,5), (0,7), (2,5), (2,7), (5,7)]
/// ```
pub fn generate_candidates_k2(frequent_items: &[usize]) -> Vec<(usize, usize)> {
    let n = frequent_items.len();
    if n < 2 {
        return Vec::new();
    }

    // Calculate number of pairs: n * (n-1) / 2
    let n_pairs = n * (n - 1) / 2;
    let mut candidates = Vec::with_capacity(n_pairs);

    for (i, &item_i) in frequent_items.iter().enumerate() {
        for &item_j in &frequent_items[i + 1..] {
            candidates.push((item_i, item_j));
        }
    }

    candidates
}

/// Generate k+1 candidates from frequent k-itemsets using the Apriori principle.
///
/// Uses the "F_{k-1} x F_{k-1}" join method:
/// - Two k-itemsets can form a (k+1)-candidate if they share their first k-1 items
/// - The union must have all k-subsets in the frequent set (anti-monotone pruning)
///
/// # Arguments
/// * `frequent_k` - List of frequent k-itemsets, each sorted
///
/// # Returns
/// Vec of (k+1)-itemsets that could potentially be frequent
///
/// # Example
/// ```
/// let frequent_2 = vec![
///     vec![0, 1],
///     vec![0, 2],
///     vec![1, 2],
/// ];
/// let candidates_3 = generate_candidates_kplus1(&frequent_2);
/// // Returns: [[0, 1, 2]] (the only 3-itemset where all 2-subsets are frequent)
/// ```
pub fn generate_candidates_kplus1(frequent_k: &[Vec<usize>]) -> Vec<Vec<usize>> {
    if frequent_k.is_empty() {
        return Vec::new();
    }

    let k = frequent_k[0].len();
    if k == 0 {
        return Vec::new();
    }

    // Build a hash set of frequent k-itemsets for O(1) subset lookup
    let frequent_set: HashSet<Vec<usize>> = frequent_k.iter().cloned().collect();

    // Generate candidates using F_{k-1} x F_{k-1} join
    // Two itemsets can be joined if they share the first k-1 items
    let mut candidates: Vec<Vec<usize>> = Vec::new();

    for (i, itemset_i) in frequent_k.iter().enumerate() {
        for itemset_j in &frequent_k[i + 1..] {
            // Check if first k-1 items match
            if itemset_i[..k - 1] == itemset_j[..k - 1] {
                // Create candidate by merging
                let mut candidate = itemset_i.clone();
                candidate.push(itemset_j[k - 1]);

                // Prune: check if all k-subsets are frequent (anti-monotone property)
                if all_subsets_frequent(&candidate, &frequent_set) {
                    candidates.push(candidate);
                }
            }
        }
    }

    candidates
}

/// Check if all (k-1)-subsets of a k-itemset are in the frequent set.
///
/// This implements the anti-monotone pruning: if any subset is infrequent,
/// the superset cannot be frequent.
fn all_subsets_frequent(itemset: &[usize], frequent_set: &HashSet<Vec<usize>>) -> bool {
    let k = itemset.len();
    if k <= 1 {
        return true;
    }

    // Generate all (k-1)-subsets by removing one element at a time
    for skip_idx in 0..k {
        let subset: Vec<usize> = itemset
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != skip_idx)
            .map(|(_, &v)| v)
            .collect();

        if !frequent_set.contains(&subset) {
            return false;
        }
    }

    true
}

/// Generate candidates for k=2 in parallel (for very large item sets).
///
/// This is useful when you have millions of frequent items and need
/// to generate billions of candidate pairs.
pub fn generate_candidates_k2_parallel(frequent_items: &[usize]) -> Vec<(usize, usize)> {
    let n = frequent_items.len();
    if n < 2 {
        return Vec::new();
    }

    // Parallel generation: each thread handles a range of first items
    (0..n - 1)
        .into_par_iter()
        .flat_map_iter(|i| {
            let item_i = frequent_items[i];
            frequent_items[i + 1..]
                .iter()
                .map(move |&item_j| (item_i, item_j))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_candidates_k2() {
        let items = vec![0, 2, 5, 7];
        let pairs = generate_candidates_k2(&items);

        assert_eq!(pairs.len(), 6); // C(4,2) = 6
        assert!(pairs.contains(&(0, 2)));
        assert!(pairs.contains(&(0, 5)));
        assert!(pairs.contains(&(0, 7)));
        assert!(pairs.contains(&(2, 5)));
        assert!(pairs.contains(&(2, 7)));
        assert!(pairs.contains(&(5, 7)));
    }

    #[test]
    fn test_generate_candidates_k2_empty() {
        assert!(generate_candidates_k2(&[]).is_empty());
        assert!(generate_candidates_k2(&[1]).is_empty());
    }

    #[test]
    fn test_generate_candidates_kplus1() {
        // Frequent 2-itemsets that form a complete triangle
        let frequent_2 = vec![
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
        ];

        let candidates_3 = generate_candidates_kplus1(&frequent_2);

        assert_eq!(candidates_3.len(), 1);
        assert_eq!(candidates_3[0], vec![0, 1, 2]);
    }

    #[test]
    fn test_generate_candidates_kplus1_pruning() {
        // Frequent 2-itemsets that DON'T form a complete triangle
        // Missing (1, 2), so (0, 1, 2) should be pruned
        let frequent_2 = vec![
            vec![0, 1],
            vec![0, 2],
            // vec![1, 2] is missing!
        ];

        let candidates_3 = generate_candidates_kplus1(&frequent_2);

        assert!(candidates_3.is_empty()); // Pruned because {1,2} not frequent
    }

    #[test]
    fn test_generate_candidates_k2_parallel() {
        let items = vec![0, 2, 5, 7];
        let pairs_seq = generate_candidates_k2(&items);
        let pairs_par = generate_candidates_k2_parallel(&items);

        // Should produce same results (though possibly different order)
        assert_eq!(pairs_seq.len(), pairs_par.len());
        for pair in &pairs_seq {
            assert!(pairs_par.contains(pair));
        }
    }
}
