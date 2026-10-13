//! K>=3 prefix group building for GPU candidate counting.
//!
//! Groups frequent (k)-itemsets by their (k-1)-prefix and collects
//! the last item as suffix. Only keeps groups with >= 2 suffixes,
//! since single-suffix groups produce no candidate pairs.
//!
//! This replaces the numpy lexsort + boundary-scan implementation
//! in cuda_kernels.py::build_k3plus_groups_from_flat(), providing:
//! - ~9x less memory (sort u64 indices vs allocating full sorted copies)
//! - Rayon parallel sort (all CPU cores, work-stealing)
//! - GIL-free execution
//!
//! Hot path at K=7: 941M input items, ~54 min total K=7 runtime.

use rayon::prelude::*;

/// Result of K>=3 prefix group building.
pub struct K3PlusGroupsResult {
    /// Flat array of valid group prefix items (i32), length = n_valid * prefix_len
    pub prefix_items: Vec<i32>,
    /// Offsets into prefix_items, length = n_valid + 1. prefix_offsets[i+1] - prefix_offsets[i] = prefix_len
    pub prefix_offsets: Vec<i64>,
    /// Flat array of suffix items (i32) from all valid groups
    pub suffixes: Vec<i32>,
    /// Offsets into suffixes, length = n_valid + 1
    pub suffix_offsets: Vec<i64>,
    /// Cumulative pair count per group, length = n_valid + 1. cumulative_pairs[i] = sum of n_j*(n_j-1)/2 for j < i
    pub cumulative_pairs: Vec<i64>,
    /// Total number of candidate pairs across all groups
    pub total_candidates: i64,
    /// Source row (index into the builder's input flat array) of each suffix
    /// slot; parallels `suffixes`. Empty unless requested (`with_src_rows`):
    /// it costs 8 B per row and only the sparse-CSR GPU path needs it (its
    /// kernels map a candidate's two suffix slots back to prev-level rows).
    pub suffix_src_rows: Vec<i64>,
}

/// Build prefix group arrays from flat (n_freq, k) row-major i32 array.
///
/// # Algorithm
/// 1. Create u64 index array [0..n_freq)
/// 2. Parallel sort indices by lexicographic prefix order (Rayon par_sort_unstable_by)
/// 3. Sequential boundary scan to find groups where prefix changes
/// 4. Filter groups with >= 2 suffixes
/// 5. Build output arrays: prefix_items, prefix_offsets, suffixes, suffix_offsets, cumulative_pairs
///
/// # Arguments
/// * `data` - Row-major flat slice of shape (n_freq, k), i32. NOT modified.
/// * `n_freq` - Number of frequent itemsets (rows)
/// * `k` - Itemset length (columns), must be >= 2
/// * `with_src_rows` - Also record each suffix slot's source row
///   (`suffix_src_rows`); false leaves it empty.
///
/// # Returns
/// `Some(K3PlusGroupsResult)` if valid groups exist, `None` otherwise.
///
/// # Panics
/// - If `data.len() != n_freq * k`
pub fn build_k3plus_groups_from_flat_raw(
    data: &[i32],
    n_freq: usize,
    k: usize,
    with_src_rows: bool,
) -> Option<K3PlusGroupsResult> {
    if n_freq < 2 || k < 2 {
        return None;
    }

    assert_eq!(
        data.len(),
        n_freq * k,
        "data length mismatch: {} != {} * {}",
        data.len(),
        n_freq,
        k
    );
    let prefix_len = k - 1;

    // ── Phase 1: Parallel sort by prefix ──────────────────────────
    // u64 indices: supports >4.3B items (K=8: 12B, K=9: ~39B)
    let mut indices: Vec<u64> = (0..n_freq as u64).collect();

    indices.par_sort_unstable_by(|&a, &b| {
        let off_a = a as usize * k;
        let off_b = b as usize * k;
        // Sort by FULL row (all k columns) not just prefix. Prefix-only sort
        // leaves suffixes in non-deterministic order within same-prefix groups,
        // which downstream consumers (the K>=3 kernels' device subset test,
        // which looks up prefix + (s_i, s_j) with s_i < s_j in the sorted
        // previous level, plus K+1 prefix extraction in apriori.py) require
        // to be ascending.
        for col in 0..k {
            // Safe: off_a + col < n_freq * k = data.len() (indices are in [0, n_freq))
            let va = data[off_a + col];
            let vb = data[off_b + col];
            match va.cmp(&vb) {
                std::cmp::Ordering::Equal => continue,
                ord => return ord,
            }
        }
        std::cmp::Ordering::Equal
    });

    // ── Phase 2: Boundary scan ────────────────────────────────────
    // Sequential O(n) scan — ~0.2s for 941M items (memory-bound)
    // We directly build the valid groups list to avoid double-pass.
    struct GroupInfo {
        start: usize,  // position in sorted indices array
        size: usize,    // number of items in this group
    }

    let mut groups: Vec<GroupInfo> = Vec::with_capacity(n_freq / 4);
    let mut current_start = 0usize;

    for i in 1..n_freq {
        let prev_off = indices[i - 1] as usize * k;
        let curr_off = indices[i] as usize * k;

        let mut is_boundary = false;
        for col in 0..prefix_len {
            if data[prev_off + col] != data[curr_off + col] {
                is_boundary = true;
                break;
            }
        }

        if is_boundary {
            let size = i - current_start;
            if size >= 2 {
                groups.push(GroupInfo {
                    start: current_start,
                    size,
                });
            }
            current_start = i;
        }
    }
    // Last group
    let last_size = n_freq - current_start;
    if last_size >= 2 {
        groups.push(GroupInfo {
            start: current_start,
            size: last_size,
        });
    }

    if groups.is_empty() {
        return None;
    }

    // ── Phase 3: Build output arrays ──────────────────────────────
    let n_valid = groups.len();
    let total_suffixes: usize = groups.iter().map(|g| g.size).sum();

    let mut prefix_items: Vec<i32> = Vec::with_capacity(n_valid * prefix_len);
    let mut prefix_offsets: Vec<i64> = Vec::with_capacity(n_valid + 1);
    let mut suffixes: Vec<i32> = Vec::with_capacity(total_suffixes);
    let mut suffix_offsets: Vec<i64> = Vec::with_capacity(n_valid + 1);
    let mut cumulative_pairs: Vec<i64> = Vec::with_capacity(n_valid + 1);
    let mut suffix_src_rows: Vec<i64> =
        Vec::with_capacity(if with_src_rows { total_suffixes } else { 0 });

    prefix_offsets.push(0);
    suffix_offsets.push(0);
    cumulative_pairs.push(0);

    let mut cum_pairs: i64 = 0;

    for group in &groups {
        // Prefix: first k-1 items from the first row in this group
        let first_row = indices[group.start] as usize;
        let row_off = first_row * k;
        prefix_items.extend_from_slice(&data[row_off..row_off + prefix_len]);
        prefix_offsets.push(prefix_items.len() as i64);

        // Suffixes: last item from each row in this group
        for j in 0..group.size {
            let row_idx = indices[group.start + j] as usize;
            suffixes.push(data[row_idx * k + prefix_len]);
            if with_src_rows {
                suffix_src_rows.push(row_idx as i64);
            }
        }
        suffix_offsets.push(suffixes.len() as i64);

        // Cumulative pairs: n*(n-1)/2
        let s = group.size as i64;
        cum_pairs += s * (s - 1) / 2;
        cumulative_pairs.push(cum_pairs);
    }

    if cum_pairs == 0 {
        return None;
    }

    Some(K3PlusGroupsResult {
        prefix_items,
        prefix_offsets,
        suffixes,
        suffix_offsets,
        cumulative_pairs,
        total_candidates: cum_pairs,
        suffix_src_rows,
    })
}

/// Keep only free-sets (generators): drop itemsets whose count equals a (k-1)-subset's.
///
/// An itemset is a *free-set* (equivalently a generator, or key) when no proper
/// subset has the same support [Bastide et al. 2000]. Dropping any one column
/// and finding the same count means the dropped item is implied by the rest, so
/// the itemset is not free. Free-sets are anti-monotone — every subset of a
/// free-set is free — which is what makes it sound to generate the next level
/// from the survivors alone.
///
/// This is NOT closed-itemset mining: closed asks about equal-support
/// *supersets*, this asks about equal-support *subsets*. The two notions select
/// different itemsets, and the caller must resolve the test against the
/// COMPLETE previous level (not a previously pruned one) for it to be exact.
///
/// # Algorithm
/// 1. Binary-search lookup on prev_flat, which the CALLER must have sorted
///    lexicographically by row. Nothing upstream guarantees that by itself:
///    the group builder sorts its own output, but the K>=3 GPU decode emits
///    each prefix group's pairs in j-major order ((0,1),(0,2),(1,2),(0,3),…),
///    which is not lex order once a group has >= 4 suffixes. The GPU miners
///    therefore lexsort `current_flat` at level end while pruning, and the
///    PyO3 wrappers enforce the invariant with an always-on O(n·k) check
///    (`is_sorted_by_row`) that raises ValueError instead of under-pruning.
/// 2. For each current itemset, try all k drop positions
/// 3. If any (k-1)-subset has equal count in prev → not free (prune)
/// 4. Rayon parallel over current itemsets
///
/// Replaces the earlier HashMap-based lookup. The HashMap had ~50ns/get but
/// required ~5-15s sequential build for 67M+ prev entries. Binary search has
/// ~200ns/get (cache misses on 500 MB prev_flat) but zero build cost. Net:
/// build cost dominates, binary search wins by 3-13s per K-transition.
///
/// # Arguments
/// * `current_flat` - Row-major (n_current, k) i32 array of current-level frequent itemsets
/// * `current_counts` - (n_current,) i64 support counts
/// * `prev_flat` - Row-major (n_prev, k-1) i32 array of prev-level frequent itemsets
/// * `prev_counts` - (n_prev,) i64 support counts
/// * `n_current` - Number of current-level itemsets
/// * `n_prev` - Number of prev-level itemsets
/// * `k` - Current itemset length (columns)
///
/// # Returns
/// Boolean mask of length n_current. `true` = keep (free), `false` = prune (not free).
pub fn prune_non_free_flat_raw(
    current_flat: &[i32],
    current_counts: &[i64],
    prev_flat: &[i32],
    prev_counts: &[i64],
    n_current: usize,
    n_prev: usize,
    k: usize,
) -> Vec<bool> {
    if n_current == 0 || n_prev == 0 || k < 2 {
        return vec![true; n_current];
    }

    let prev_k = k - 1;
    assert_eq!(current_flat.len(), n_current * k);
    assert_eq!(prev_flat.len(), n_prev * prev_k);
    assert_eq!(current_counts.len(), n_current, "current_counts length mismatch");
    assert_eq!(prev_counts.len(), n_prev, "prev_counts length mismatch");

    // prev_flat must be sorted lexicographically by full row: binary search on
    // the sorted slice replaces a HashMap without its 5-15 s sequential build
    // (per lookup ~200 ns vs ~50 ns, but 67M inserts × ~100 ns no longer paid).
    // The sortedness is the caller's job (the GPU miners lexsort at level end
    // while pruning) and the PyO3 wrappers check it on every call; this
    // debug_assert! is a second net for direct Rust callers.
    debug_assert!(
        is_sorted_by_row(prev_flat, n_prev, prev_k),
        "prev_flat must be sorted lexicographically by row for the binary-search lookup"
    );

    // Parallel check — for each current itemset, see if any (k-1)-subset has the
    // same count in prev. If yes, the itemset is not free → prune it.
    (0..n_current)
        .into_par_iter()
        .map(|idx| {
            let row = &current_flat[idx * k..(idx + 1) * k];
            let my_count = current_counts[idx];

            // Try each drop position d: build (k-1)-subset by skipping element d
            let mut subset = Vec::with_capacity(prev_k);
            for d in 0..k {
                subset.clear();
                for (j, &val) in row.iter().enumerate() {
                    if j != d {
                        subset.push(val);
                    }
                }
                if let Some(prev_count) =
                    binary_search_row(prev_flat, prev_counts, n_prev, prev_k, &subset)
                {
                    if prev_count == my_count {
                        return false; // not a free-set → prune
                    }
                }
            }
            true // open → keep
        })
        .collect()
}

/// Verify that flat row-major data is sorted lexicographically by row
/// (non-descending). O(n_rows × row_len). Used by the PyO3 free-set-prune
/// wrappers as an always-on invariant check before the binary-search lookup,
/// and by the debug_assert! in `prune_non_free_flat_raw`.
pub fn is_sorted_by_row(flat: &[i32], n_rows: usize, row_len: usize) -> bool {
    if n_rows < 2 || row_len == 0 {
        return true;
    }
    for i in 1..n_rows {
        let prev = &flat[(i - 1) * row_len..i * row_len];
        let curr = &flat[i * row_len..(i + 1) * row_len];
        if prev.cmp(curr) == std::cmp::Ordering::Greater {
            return false;
        }
    }
    true
}

/// Binary search for `subset` in `prev_flat` (sorted lexicographically by row).
/// Returns `Some(count)` if found, `None` if not present.
///
/// Replaces the HashMap.get() lookup in prune_non_free_flat_raw, eliminating the
/// O(n_prev) sequential HashMap build phase. Caller must ensure prev_flat is
/// sorted-by-row (enforced by the PyO3 wrappers; debug_assert! in
/// prune_non_free_flat_raw).
fn binary_search_row(
    prev_flat: &[i32],
    prev_counts: &[i64],
    n_prev: usize,
    prev_k: usize,
    subset: &[i32],
) -> Option<i64> {
    let mut lo = 0usize;
    let mut hi = n_prev;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let row = &prev_flat[mid * prev_k..(mid + 1) * prev_k];
        match row.cmp(subset) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
            std::cmp::Ordering::Equal => return Some(prev_counts[mid]),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_grouping() {
        // 6 itemsets of length 3 (k=3, prefix_len=2):
        // [1,2,5], [1,2,7], [1,2,9], [1,3,4], [1,3,6], [2,3,8]
        // Prefixes: [1,2], [1,2], [1,2], [1,3], [1,3], [2,3]
        // After sort by prefix: already sorted
        // Groups: [1,2] has 3 suffixes [5,7,9], [1,3] has 2 suffixes [4,6], [2,3] has 1 suffix [8] (filtered)
        let data: Vec<i32> = vec![
            1, 2, 5, // row 0
            1, 2, 7, // row 1
            1, 2, 9, // row 2
            1, 3, 4, // row 3
            1, 3, 6, // row 4
            2, 3, 8, // row 5 — singleton group, should be filtered
        ];

        let result = build_k3plus_groups_from_flat_raw(&data, 6, 3, false).unwrap();

        // 2 valid groups
        assert_eq!(result.prefix_offsets.len(), 3); // n_valid + 1
        assert_eq!(result.suffix_offsets.len(), 3);

        // Group 1: prefix [1,2], 3 suffixes
        assert_eq!(&result.prefix_items[0..2], &[1, 2]);
        assert_eq!(result.suffix_offsets[1] - result.suffix_offsets[0], 3);

        // Group 2: prefix [1,3], 2 suffixes
        assert_eq!(&result.prefix_items[2..4], &[1, 3]);
        assert_eq!(result.suffix_offsets[2] - result.suffix_offsets[1], 2);

        // Total candidates: C(3,2) + C(2,2) = 3 + 1 = 4
        assert_eq!(result.total_candidates, 4);
        assert_eq!(&result.cumulative_pairs, &[0, 3, 4]);
    }

    #[test]
    fn test_unsorted_input() {
        // Same data as above but shuffled — sort should fix it
        let data: Vec<i32> = vec![
            1, 3, 6, // row 0
            2, 3, 8, // row 1
            1, 2, 9, // row 2
            1, 2, 5, // row 3
            1, 3, 4, // row 4
            1, 2, 7, // row 5
        ];

        let result = build_k3plus_groups_from_flat_raw(&data, 6, 3, false).unwrap();
        assert_eq!(result.total_candidates, 4);

        // Suffixes should be sorted within each group (due to stable sort by full prefix)
        // Group [1,2]: suffixes from sorted rows
        let suf_0_start = result.suffix_offsets[0] as usize;
        let suf_0_end = result.suffix_offsets[1] as usize;
        let group1_suffixes = &result.suffixes[suf_0_start..suf_0_end];
        assert_eq!(group1_suffixes.len(), 3);
        // All three suffixes 5, 7, 9 should be present (order depends on sort stability)
        let mut sorted_suf: Vec<i32> = group1_suffixes.to_vec();
        sorted_suf.sort();
        assert_eq!(sorted_suf, vec![5, 7, 9]);
    }

    #[test]
    fn test_all_singletons() {
        // Every prefix is unique → no valid groups
        let data: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert!(build_k3plus_groups_from_flat_raw(&data, 3, 3, false).is_none());
    }

    #[test]
    fn test_too_few_items() {
        let data: Vec<i32> = vec![1, 2, 3];
        assert!(build_k3plus_groups_from_flat_raw(&data, 1, 3, false).is_none());
    }

    #[test]
    fn test_k2_input() {
        // k=2: prefix is 1 column, suffix is 1 column
        let data: Vec<i32> = vec![
            1, 5, // prefix [1], suffix 5
            1, 7, // prefix [1], suffix 7
            2, 3, // prefix [2], suffix 3 — singleton
            3, 1, // prefix [3], suffix 1
            3, 4, // prefix [3], suffix 4
        ];

        let result = build_k3plus_groups_from_flat_raw(&data, 5, 2, false).unwrap();
        assert_eq!(result.total_candidates, 2); // C(2,2) + C(2,2) = 1 + 1 = 2
    }

    // =========================================================================
    // Closed pruning tests
    // =========================================================================

    #[test]
    fn test_prune_non_free_basic() {
        // K=3 current: [1,2,3] count=10, [1,2,4] count=20
        // K=2 prev: [1,2] count=10, [1,3] count=15, [2,3] count=30
        // [1,2,3]: drop col 2 → [1,3] count=15 != 10, drop col 1 → [2,3] count=30 != 10,
        //          drop col 0 → [2,3] count=30 != 10. All differ → KEEP
        // Wait: drop col 2 → [1,2] count=10 == 10 → PRUNE
        let current_flat: Vec<i32> = vec![1, 2, 3, 1, 2, 4];
        let current_counts: Vec<i64> = vec![10, 20];
        let prev_flat: Vec<i32> = vec![1, 2, 1, 3, 2, 3];
        let prev_counts: Vec<i64> = vec![10, 15, 30];

        let mask = prune_non_free_flat_raw(
            &current_flat, &current_counts, &prev_flat, &prev_counts, 2, 3, 3,
        );
        // [1,2,3]: drop 3 → [1,2] has count 10 == 10 → prune
        // [1,2,4]: no (k-1)-subset has count 20 → keep
        assert_eq!(mask, vec![false, true]);
    }

    #[test]
    fn test_prune_non_free_all_open() {
        // All counts differ from any prev subset → keep all
        let current_flat: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
        let current_counts: Vec<i64> = vec![100, 200];
        let prev_flat: Vec<i32> = vec![1, 2, 4, 5];
        let prev_counts: Vec<i64> = vec![50, 60];

        let mask = prune_non_free_flat_raw(
            &current_flat, &current_counts, &prev_flat, &prev_counts, 2, 2, 3,
        );
        assert_eq!(mask, vec![true, true]);
    }

    #[test]
    fn test_prune_non_free_empty_input() {
        let mask = prune_non_free_flat_raw(&[], &[], &[1, 2], &[10], 0, 1, 3);
        assert!(mask.is_empty());
    }

    #[test]
    fn test_prune_non_free_empty_prev() {
        let mask = prune_non_free_flat_raw(&[1, 2, 3], &[10], &[], &[], 1, 0, 3);
        assert_eq!(mask, vec![true]); // nothing to compare against → keep
    }

    // =========================================================================
    // R2: Binary-search lookup tests (replaces HashMap)
    // =========================================================================

    #[test]
    fn test_binary_search_row_found() {
        // Sorted prev_flat: [[1,2], [1,3], [2,3]]
        let prev_flat: Vec<i32> = vec![1, 2, 1, 3, 2, 3];
        let prev_counts: Vec<i64> = vec![10, 15, 30];

        assert_eq!(binary_search_row(&prev_flat, &prev_counts, 3, 2, &[1, 2]), Some(10));
        assert_eq!(binary_search_row(&prev_flat, &prev_counts, 3, 2, &[1, 3]), Some(15));
        assert_eq!(binary_search_row(&prev_flat, &prev_counts, 3, 2, &[2, 3]), Some(30));
    }

    #[test]
    fn test_binary_search_row_not_found() {
        let prev_flat: Vec<i32> = vec![1, 2, 1, 3, 2, 3];
        let prev_counts: Vec<i64> = vec![10, 15, 30];

        // Subset not in prev — return None
        assert_eq!(binary_search_row(&prev_flat, &prev_counts, 3, 2, &[1, 4]), None);
        assert_eq!(binary_search_row(&prev_flat, &prev_counts, 3, 2, &[0, 1]), None);
        assert_eq!(binary_search_row(&prev_flat, &prev_counts, 3, 2, &[3, 4]), None);
    }

    #[test]
    fn test_binary_search_row_empty() {
        // Empty prev → always None regardless of subset
        assert_eq!(binary_search_row(&[], &[], 0, 2, &[1, 2]), None);
    }

    #[test]
    fn test_is_sorted_by_row() {
        // Sorted lex
        assert!(is_sorted_by_row(&[1, 2, 1, 3, 2, 3], 3, 2));
        // Not sorted
        assert!(!is_sorted_by_row(&[2, 3, 1, 2], 2, 2));
        // Single row trivially sorted
        assert!(is_sorted_by_row(&[5, 5, 5], 1, 3));
        // Empty trivially sorted
        assert!(is_sorted_by_row(&[], 0, 2));
    }

    #[test]
    fn test_src_rows_map_back() {
        // Shuffled input: every slot's source row must reproduce prefix + suffix,
        // suffixes stay ascending within a group, and rows are only recorded
        // when requested.
        let data: Vec<i32> = vec![
            1, 3, 6, // row 0
            2, 3, 8, // row 1 — singleton group, dropped
            1, 2, 9, // row 2
            1, 2, 5, // row 3
            1, 3, 4, // row 4
            1, 2, 7, // row 5
        ];
        let (n, k) = (6usize, 3usize);
        let r = build_k3plus_groups_from_flat_raw(&data, n, k, true).unwrap();
        assert_eq!(r.suffix_src_rows.len(), r.suffixes.len());
        let n_groups = r.suffix_offsets.len() - 1;
        for g in 0..n_groups {
            let prefix = &r.prefix_items[r.prefix_offsets[g] as usize..r.prefix_offsets[g + 1] as usize];
            let (s0, s1) = (r.suffix_offsets[g] as usize, r.suffix_offsets[g + 1] as usize);
            for s in s0..s1 {
                let row = r.suffix_src_rows[s] as usize;
                assert_eq!(&data[row * k..row * k + k - 1], prefix);
                assert_eq!(data[row * k + k - 1], r.suffixes[s]);
            }
            assert!(r.suffixes[s0..s1].windows(2).all(|w| w[0] < w[1]));
        }
        let mut rows = r.suffix_src_rows.clone();
        rows.sort();
        assert_eq!(rows, vec![0, 2, 3, 4, 5]); // row 1 is the dropped singleton

        let r2 = build_k3plus_groups_from_flat_raw(&data, n, k, false).unwrap();
        assert!(r2.suffix_src_rows.is_empty());
        assert_eq!(r2.suffixes, r.suffixes);
    }
}
