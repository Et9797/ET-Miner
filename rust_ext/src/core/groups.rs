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
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};

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
        // which downstream consumers (prune_groups_apriori K=3 branch building
        // unsorted [gsuf[i], gsuf[j]] pairs against sorted prev_set keys, plus
        // K+1 prefix extraction in apriori.py) require to be ascending.
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

/// Prune closed itemsets: remove itemsets whose count equals a (k-1)-subset's count.
///
/// An itemset is "non-closed" (prunable) if dropping any one column yields a (k-1)-subset
/// with the same support count — the k-th item adds no discriminative power.
///
/// # Algorithm
/// 1. Binary-search lookup on prev_flat (sorted-by-row since groups.rs:79 fix)
/// 2. For each current itemset, try all k drop positions
/// 3. If any (k-1)-subset has equal count in prev → mark as closed (prune)
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
/// Boolean mask of length n_current. `true` = keep (open), `false` = prune (closed).
pub fn prune_closed_flat_raw(
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

    // prev_flat is sorted-by-full-row
    // sinds groups.rs:79 sort fix (commit 4f0bbab9). Binary search op de sorted
    // slice vervangt de HashMap zonder de 5-15s sequential build cost. Per-lookup
    // is binary search 2-6x slower dan HashMap (~200ns vs ~50ns), maar de gespaarde
    // build dominates: 67M HashMap entries × ~100ns insert = 6.7s wegvallen.
    //
    // Invariant assertion (debug builds only — compiles away in release):
    debug_assert!(
        is_sorted_by_row(prev_flat, n_prev, prev_k),
        "prev_flat must be sorted-by-full-row for binary search lookup. \
         If this fires, did groups.rs:79 sort fix regress?"
    );

    // Parallel check — for each current itemset, see if any (k-1)-subset has the
    // same count in prev. If yes, the itemset is non-closed → prune it.
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
                        return false; // non-closed → prune
                    }
                }
            }
            true // open → keep
        })
        .collect()
}

/// Verify that flat row-major data is sorted lexicographically by row.
/// Used for debug_assert! invariant checking before binary-search lookups.
fn is_sorted_by_row(flat: &[i32], n_rows: usize, row_len: usize) -> bool {
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
/// Replaces the HashMap.get() lookup in prune_closed_flat_raw, eliminating the
/// O(n_prev) sequential HashMap build phase. Caller must ensure prev_flat is
/// sorted-by-row (verified via debug_assert! in prune_closed_flat_raw).
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

/// R3 (2026-05-01): Parallel unique column extraction for `current_live_mgpu`
/// computation in apriori.py. Replaces single-threaded `np.unique(current_flat)`
/// which on (430M, 6) int32 = 2.58B elements took ~30-90s sequential sort+dedup.
///
/// Strategy: atomic bitset sized by max value (vocab size ~149 in V5).
/// Two parallel passes over `flat`:
///   1. Compute max value (reduction)
///   2. Mark `bitset[v] = true` for each v in flat (par_iter atomic OR)
/// Sequential collect of indices where bit is set = O(max_col), trivially small.
///
/// Sub-second on 224 cores for 2.6B elements at ~10 GB/s DDR bandwidth.
pub fn unique_columns_from_flat_raw(flat: &[i32]) -> Vec<i32> {
    if flat.is_empty() {
        return Vec::new();
    }

    // Pass 1: parallel reduce to find max (skip negatives — column indices are non-negative).
    let max_val: i32 = flat.par_iter().copied().filter(|&v| v >= 0).max().unwrap_or(-1);
    if max_val < 0 {
        return Vec::new(); // no valid column indices
    }

    // Pass 2: parallel mark — atomic OR per cell. AtomicBool is ~1 byte typically;
    // for vocab=149 bitset = ~150 bytes, trivially fits L1 cache, no contention.
    let n_buckets = (max_val as usize) + 1;
    let bitset: Vec<AtomicBool> = (0..n_buckets).map(|_| AtomicBool::new(false)).collect();

    flat.par_iter().for_each(|&v| {
        if v >= 0 {
            let idx = v as usize;
            // Defensive: any v > max_val (shouldn't happen since max_val IS max)
            if idx < n_buckets {
                bitset[idx].store(true, Ordering::Relaxed);
            }
        }
    });

    // Sequential collect: O(max_col) where max_col ~149. Negligible.
    bitset
        .iter()
        .enumerate()
        .filter_map(|(i, b)| {
            if b.load(Ordering::Relaxed) {
                Some(i as i32)
            } else {
                None
            }
        })
        .collect()
}

/// Compact variant: prunes non-closed itemsets and returns the surviving
/// `(flat, counts)` arrays directly, eliminating the Python-side numpy fancy
/// index step that bottlenecks at K=6+ on multi-GB arrays.
///
/// Single-threaded Python `current_flat[mask]` on
/// ~430M rows took 30-60s on one core out of 224. Sequential Rust
/// `extend_from_slice` runs at memcpy speed (~7-10 GB/s) and avoids the GIL +
/// numpy dispatch overhead. Speedup ~10-13× verified by Auditor microbenchmark
/// on the materialization step.
///
/// Implementation requirements:
/// 1. Pre-count surviving rows so `Vec::with_capacity` is exact (avoids
///    geometric Vec growth → 2× transient memory waste on 10-160 GB arrays).
/// 2. Sequential `extend_from_slice` preserves `current_flat` row order
///    (parallel reorder would break downstream pattern signature consumers).
/// 3. Empty case returns empty `Vec`s; caller reshapes to `(0, k)` so
///    `prev_frequent_flat.shape[1]` stays well-defined for the next K level.
///
/// # Returns
/// `(out_flat, out_counts)` — flat row-major Vec<i32> of length `n_kept * k`,
/// and Vec<i64> of length `n_kept`. Caller reshapes flat to (n_kept, k).
pub fn prune_closed_flat_compact_raw(
    current_flat: &[i32],
    current_counts: &[i64],
    prev_flat: &[i32],
    prev_counts: &[i64],
    n_current: usize,
    n_prev: usize,
    k: usize,
) -> (Vec<i32>, Vec<i64>) {
    // Re-use mask computation (binary search + Rayon parallel check) from existing path.
    let mask = prune_closed_flat_raw(
        current_flat,
        current_counts,
        prev_flat,
        prev_counts,
        n_current,
        n_prev,
        k,
    );

    // Pre-count for exact capacity. mask.iter() is sequential ~5 GB/s
    // (memory-bound bool scan), negligible vs the compaction cost it saves.
    let n_kept: usize = mask.iter().filter(|&&b| b).count();

    if n_kept == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut out_flat: Vec<i32> = Vec::with_capacity(n_kept * k);
    let mut out_counts: Vec<i64> = Vec::with_capacity(n_kept);

    for (idx, &keep) in mask.iter().enumerate() {
        if keep {
            let row_start = idx * k;
            out_flat.extend_from_slice(&current_flat[row_start..row_start + k]);
            out_counts.push(current_counts[idx]);
        }
    }

    (out_flat, out_counts)
}

/// Apriori-prune prefix groups: remove suffix pairs whose candidates have
/// non-frequent (k-1)-subsets.
///
/// For each group (prefix, suffixes), enumerates all suffix pairs (s_i, s_j).
/// For K=3: checks only (s_i, s_j) membership in prev.
/// For K>=4: also checks k-2 "prefix-drop" subsets.
/// A suffix survives if it participates in at least one valid pair.
/// Groups with < 2 surviving suffixes are dropped.
///
/// # Arguments
/// * `groups` - K3PlusGroupsResult from group building
/// * `prev_flat` - Row-major (n_prev, k-1) i32 array of previous-level frequent itemsets
/// * `n_prev` - Number of prev-level itemsets
/// * `k` - Current candidate length (k = prefix_len + 2 for the pair)
///
/// # Returns
/// `Some(K3PlusGroupsResult)` with pruned groups, or `None` if all candidates pruned.
pub fn prune_groups_apriori_raw(
    groups: &K3PlusGroupsResult,
    prev_flat: &[i32],
    n_prev: usize,
    k: usize,
) -> Option<K3PlusGroupsResult> {
    let prev_k = k - 1;
    assert_eq!(prev_flat.len(), n_prev * prev_k);

    // Build HashSet of prev (k-1)-itemsets for O(1) membership test.
    let mut prev_set: HashSet<Vec<i32>> = HashSet::with_capacity(n_prev);
    for i in 0..n_prev {
        let row = prev_flat[i * prev_k..(i + 1) * prev_k].to_vec();
        prev_set.insert(row);
    }

    let n_groups = groups.suffix_offsets.len() - 1;
    let prefix_len = k - 2; // prefix items per group (candidate = prefix + 2 suffixes)

    // suffix_src_rows is optional but, when present, must parallel suffixes.
    let with_rows = !groups.suffix_src_rows.is_empty();
    if with_rows {
        assert_eq!(
            groups.suffix_src_rows.len(),
            groups.suffixes.len(),
            "suffix_src_rows must parallel suffixes"
        );
    }

    // Process groups in parallel, collect (prefix, valid_suffixes, valid_rows) per group
    let group_results: Vec<(Vec<i32>, Vec<i32>, Vec<i64>)> = (0..n_groups)
        .into_par_iter()
        .map(|g| {
            let pstart = groups.prefix_offsets[g] as usize;
            let pend = groups.prefix_offsets[g + 1] as usize;
            let sstart = groups.suffix_offsets[g] as usize;
            let send = groups.suffix_offsets[g + 1] as usize;

            let prefix = &groups.prefix_items[pstart..pend];
            let gsuf = &groups.suffixes[sstart..send];

            // Track which suffixes participate in at least one valid pair
            let mut valid = vec![false; gsuf.len()];

            for i in 0..gsuf.len() {
                for j in (i + 1)..gsuf.len() {
                    let pair_valid = if k == 3 {
                        // K=3: prefix has 1 element, only check (s_i, s_j)
                        let check = vec![gsuf[i], gsuf[j]];
                        prev_set.contains(&check)
                    } else {
                        // K>=4: build candidate = prefix + (s_i, s_j)
                        // Check k-2 prefix-drop subsets
                        let mut candidate = Vec::with_capacity(k);
                        candidate.extend_from_slice(prefix);
                        candidate.push(gsuf[i]);
                        candidate.push(gsuf[j]);

                        let mut all_freq = true;
                        // For each prefix position d, drop it to form (k-1)-subset
                        for d in 0..prefix_len {
                            let mut subset = Vec::with_capacity(prev_k);
                            for (pos, &val) in candidate.iter().enumerate() {
                                if pos != d {
                                    subset.push(val);
                                }
                            }
                            if !prev_set.contains(&subset) {
                                all_freq = false;
                                break;
                            }
                        }
                        all_freq
                    };

                    if pair_valid {
                        valid[i] = true;
                        valid[j] = true;
                    }
                }
            }

            // Collect valid suffixes (sorted — they're already sorted within the group)
            let valid_sorted: Vec<i32> = gsuf.iter().enumerate()
                .filter(|&(idx, _)| valid[idx])
                .map(|(_, &v)| v)
                .collect();
            // ...and the same slots' source rows, when tracked.
            let valid_rows: Vec<i64> = if with_rows {
                let grows = &groups.suffix_src_rows[sstart..send];
                (0..gsuf.len()).filter(|&idx| valid[idx]).map(|idx| grows[idx]).collect()
            } else {
                Vec::new()
            };

            (prefix.to_vec(), valid_sorted, valid_rows)
        })
        .collect();

    // Sequential assembly of output arrays (fast, just concatenation)
    let mut new_prefix_items: Vec<i32> = Vec::new();
    let mut new_prefix_offsets: Vec<i64> = vec![0];
    let mut new_suffixes: Vec<i32> = Vec::new();
    let mut new_suffix_offsets: Vec<i64> = vec![0];
    let mut new_cumulative_pairs: Vec<i64> = vec![0];
    let mut new_suffix_src_rows: Vec<i64> = Vec::new();
    let mut total_candidates: i64 = 0;

    for (prefix, valid_sorted, valid_rows) in &group_results {
        if valid_sorted.len() >= 2 {
            new_prefix_items.extend_from_slice(prefix);
            new_prefix_offsets.push(new_prefix_items.len() as i64);
            new_suffixes.extend_from_slice(valid_sorted);
            new_suffix_offsets.push(new_suffixes.len() as i64);
            new_suffix_src_rows.extend_from_slice(valid_rows);
            let n = valid_sorted.len() as i64;
            total_candidates += n * (n - 1) / 2;
            new_cumulative_pairs.push(total_candidates);
        }
    }

    if total_candidates == 0 {
        return None;
    }

    Some(K3PlusGroupsResult {
        prefix_items: new_prefix_items,
        prefix_offsets: new_prefix_offsets,
        suffixes: new_suffixes,
        suffix_offsets: new_suffix_offsets,
        cumulative_pairs: new_cumulative_pairs,
        total_candidates,
        suffix_src_rows: new_suffix_src_rows,
    })
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
    fn test_prune_closed_basic() {
        // K=3 current: [1,2,3] count=10, [1,2,4] count=20
        // K=2 prev: [1,2] count=10, [1,3] count=15, [2,3] count=30
        // [1,2,3]: drop col 2 → [1,3] count=15 != 10, drop col 1 → [2,3] count=30 != 10,
        //          drop col 0 → [2,3] count=30 != 10. All differ → KEEP
        // Wait: drop col 2 → [1,2] count=10 == 10 → PRUNE
        let current_flat: Vec<i32> = vec![1, 2, 3, 1, 2, 4];
        let current_counts: Vec<i64> = vec![10, 20];
        let prev_flat: Vec<i32> = vec![1, 2, 1, 3, 2, 3];
        let prev_counts: Vec<i64> = vec![10, 15, 30];

        let mask = prune_closed_flat_raw(
            &current_flat, &current_counts, &prev_flat, &prev_counts, 2, 3, 3,
        );
        // [1,2,3]: drop 3 → [1,2] has count 10 == 10 → prune
        // [1,2,4]: no (k-1)-subset has count 20 → keep
        assert_eq!(mask, vec![false, true]);
    }

    #[test]
    fn test_prune_closed_all_open() {
        // All counts differ from any prev subset → keep all
        let current_flat: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
        let current_counts: Vec<i64> = vec![100, 200];
        let prev_flat: Vec<i32> = vec![1, 2, 4, 5];
        let prev_counts: Vec<i64> = vec![50, 60];

        let mask = prune_closed_flat_raw(
            &current_flat, &current_counts, &prev_flat, &prev_counts, 2, 2, 3,
        );
        assert_eq!(mask, vec![true, true]);
    }

    #[test]
    fn test_prune_closed_empty_input() {
        let mask = prune_closed_flat_raw(&[], &[], &[1, 2], &[10], 0, 1, 3);
        assert!(mask.is_empty());
    }

    #[test]
    fn test_prune_closed_empty_prev() {
        let mask = prune_closed_flat_raw(&[1, 2, 3], &[10], &[], &[], 1, 0, 3);
        assert_eq!(mask, vec![true]); // nothing to compare against → keep
    }

    // =========================================================================
    // Closed pruning compact (eliminates Python fancy-index)
    // =========================================================================

    #[test]
    fn test_prune_closed_compact_basic() {
        // Same fixture as test_prune_closed_basic: [1,2,3] pruned, [1,2,4] kept.
        let current_flat: Vec<i32> = vec![1, 2, 3, 1, 2, 4];
        let current_counts: Vec<i64> = vec![10, 20];
        let prev_flat: Vec<i32> = vec![1, 2, 1, 3, 2, 3];
        let prev_counts: Vec<i64> = vec![10, 15, 30];

        let (out_flat, out_counts) = prune_closed_flat_compact_raw(
            &current_flat, &current_counts, &prev_flat, &prev_counts, 2, 3, 3,
        );
        assert_eq!(out_flat, vec![1, 2, 4]); // only [1,2,4] survives
        assert_eq!(out_counts, vec![20]);
    }

    #[test]
    fn test_prune_closed_compact_preserves_order() {
        // Order-preservation invariant (Schizo-Arch correctness flag): rows must
        // come out in the same order they went in, modulo pruned ones.
        let current_flat: Vec<i32> = vec![5, 1, 9, 3, 2, 7, 8, 4, 6];
        let current_counts: Vec<i64> = vec![100, 200, 300];
        // Empty prev → no pruning, all rows preserved in original order.
        let (out_flat, out_counts) = prune_closed_flat_compact_raw(
            &current_flat, &current_counts, &[], &[], 3, 0, 3,
        );
        assert_eq!(out_flat, current_flat);
        assert_eq!(out_counts, current_counts);
    }

    #[test]
    fn test_prune_closed_compact_partial_prune() {
        // One row pruned, one kept — exercises the count-mismatch branch.
        let current_flat: Vec<i32> = vec![1, 2, 3, 1, 2, 4];
        let current_counts: Vec<i64> = vec![10, 20];
        // [1,2,3]: subset [1,2] count=10 == 10 → pruned
        // [1,2,4]: subset [1,2] count=10 != 20 → kept
        let prev_flat: Vec<i32> = vec![1, 2];
        let prev_counts: Vec<i64> = vec![10];

        let (out_flat, out_counts) = prune_closed_flat_compact_raw(
            &current_flat, &current_counts, &prev_flat, &prev_counts, 2, 1, 3,
        );
        // [1,2,3]: drop col 2 → [1,2] count=10 == 10 → prune
        // [1,2,4]: drop col 2 → [1,2] count=10 != 20 → keep
        assert_eq!(out_flat, vec![1, 2, 4]);
        assert_eq!(out_counts, vec![20]);
    }

    #[test]
    fn test_prune_closed_compact_empty_input() {
        let (out_flat, out_counts) = prune_closed_flat_compact_raw(&[], &[], &[1, 2], &[10], 0, 1, 3);
        assert!(out_flat.is_empty());
        assert!(out_counts.is_empty());
    }

    #[test]
    fn test_prune_closed_compact_matches_mask_path() {
        // Equivalence guarantee: compact must produce same rows/counts as
        // applying the mask path manually. This is the contract that lets us
        // ship without touching the Python rule-gen consumers.
        // R2 update: prev_flat MUST be sorted-by-row for binary search lookup
        // (debug_assert! fires otherwise). Fixture re-sorted from the original.
        let current_flat: Vec<i32> = vec![1, 2, 3, 1, 2, 4, 5, 6, 7, 1, 2, 5];
        let current_counts: Vec<i64> = vec![10, 20, 50, 30];
        // Sorted lex: [1,2], [1,3], [1,5], [2,3], [2,5], [5,6], [5,7], [6,7]
        let prev_flat: Vec<i32> = vec![1, 2, 1, 3, 1, 5, 2, 3, 2, 5, 5, 6, 5, 7, 6, 7];
        let prev_counts: Vec<i64> = vec![10, 15, 8, 30, 12, 50, 25, 35];

        let mask = prune_closed_flat_raw(
            &current_flat, &current_counts, &prev_flat, &prev_counts, 4, 8, 3,
        );
        let (compact_flat, compact_counts) = prune_closed_flat_compact_raw(
            &current_flat, &current_counts, &prev_flat, &prev_counts, 4, 8, 3,
        );

        // Build expected from mask
        let mut expected_flat: Vec<i32> = Vec::new();
        let mut expected_counts: Vec<i64> = Vec::new();
        for (i, &keep) in mask.iter().enumerate() {
            if keep {
                expected_flat.extend_from_slice(&current_flat[i * 3..i * 3 + 3]);
                expected_counts.push(current_counts[i]);
            }
        }
        assert_eq!(compact_flat, expected_flat);
        assert_eq!(compact_counts, expected_counts);
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

    // =========================================================================
    // R3: unique_columns_from_flat_raw tests
    // =========================================================================

    #[test]
    fn test_unique_columns_basic() {
        // Mixed values 1-7 with duplicates
        let flat: Vec<i32> = vec![1, 5, 3, 5, 7, 1, 3, 5, 7];
        let result = unique_columns_from_flat_raw(&flat);
        assert_eq!(result, vec![1, 3, 5, 7]); // sorted ascending (bitset-collected order)
    }

    #[test]
    fn test_unique_columns_empty() {
        assert_eq!(unique_columns_from_flat_raw(&[]), Vec::<i32>::new());
    }

    #[test]
    fn test_unique_columns_single_value() {
        let flat: Vec<i32> = vec![42, 42, 42, 42];
        assert_eq!(unique_columns_from_flat_raw(&flat), vec![42]);
    }

    #[test]
    fn test_unique_columns_v5_vocab_range() {
        // Realistic V5 case: column indices in [0, 149)
        // Build flat with 100k random-ish values in that range, expect 0..149 represented
        let mut flat: Vec<i32> = Vec::with_capacity(100_000);
        for i in 0..100_000 {
            flat.push((i % 149) as i32);
        }
        let result = unique_columns_from_flat_raw(&flat);
        // All values 0..149 should be present
        let expected: Vec<i32> = (0..149).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_unique_columns_negative_values_filtered() {
        // Defensive: negative values shouldn't crash, just be excluded
        let flat: Vec<i32> = vec![-1, 5, -2, 7, 5];
        let result = unique_columns_from_flat_raw(&flat);
        assert_eq!(result, vec![5, 7]);
    }

    #[test]
    fn test_unique_columns_matches_naive() {
        // Sanity vs HashSet-based reference impl on randomized input
        use std::collections::HashSet;
        let flat: Vec<i32> = (0..10_000).map(|i| ((i * 7 + 3) % 100) as i32).collect();

        let result = unique_columns_from_flat_raw(&flat);

        let expected_set: HashSet<i32> = flat.iter().copied().collect();
        let mut expected: Vec<i32> = expected_set.into_iter().collect();
        expected.sort();

        assert_eq!(result, expected);
    }

    // =========================================================================
    // Apriori group pruning tests
    // =========================================================================

    #[test]
    fn test_prune_groups_k3_basic() {
        // K=3: prefix_len=1. Groups have 1-element prefixes, suffix pairs form K=2 candidates.
        // prev_flat (K=2): [1,2], [1,3], [2,3], [1,4]
        // Group: prefix=[1], suffixes=[2,3,4]
        // Pairs: (2,3) → in prev? YES, (2,4) → in prev? NO, (3,4) → in prev? NO
        // Valid suffixes: {2, 3} → 2 survive, 1 pair
        let groups = K3PlusGroupsResult {
            prefix_items: vec![1],
            prefix_offsets: vec![0, 1],
            suffixes: vec![2, 3, 4],
            suffix_offsets: vec![0, 3],
            cumulative_pairs: vec![0, 3],
            total_candidates: 3,
            suffix_src_rows: vec![],
        };

        let prev_flat: Vec<i32> = vec![1, 2, 1, 3, 2, 3, 1, 4];
        let result = prune_groups_apriori_raw(&groups, &prev_flat, 4, 3).unwrap();

        // Only pair (2,3) is valid → suffixes [2, 3], 1 pair
        assert_eq!(result.suffixes.len(), 2);
        assert_eq!(result.total_candidates, 1);
        let mut suf = result.suffixes.clone();
        suf.sort();
        assert_eq!(suf, vec![2, 3]);
    }

    #[test]
    fn test_prune_groups_k3_all_valid() {
        // All pairs exist in prev → nothing pruned
        let groups = K3PlusGroupsResult {
            prefix_items: vec![1],
            prefix_offsets: vec![0, 1],
            suffixes: vec![2, 3],
            suffix_offsets: vec![0, 2],
            cumulative_pairs: vec![0, 1],
            total_candidates: 1,
            suffix_src_rows: vec![],
        };

        let prev_flat: Vec<i32> = vec![2, 3]; // (2,3) exists
        let result = prune_groups_apriori_raw(&groups, &prev_flat, 1, 3).unwrap();
        assert_eq!(result.total_candidates, 1);
    }

    #[test]
    fn test_prune_groups_all_pruned() {
        // No pairs exist in prev → all pruned
        let groups = K3PlusGroupsResult {
            prefix_items: vec![1],
            prefix_offsets: vec![0, 1],
            suffixes: vec![2, 3],
            suffix_offsets: vec![0, 2],
            cumulative_pairs: vec![0, 1],
            total_candidates: 1,
            suffix_src_rows: vec![],
        };

        let prev_flat: Vec<i32> = vec![5, 6]; // (2,3) not in prev
        let result = prune_groups_apriori_raw(&groups, &prev_flat, 1, 3);
        assert!(result.is_none());
    }

    #[test]
    fn test_prune_groups_k4() {
        // K=4: prefix_len=2. prefix=[1,2], suffixes=[3,4]
        // Candidate = [1,2,3,4]
        // Check prefix-drop subsets (d=0 and d=1):
        //   d=0: drop 1 → [2,3,4]
        //   d=1: drop 2 → [1,3,4]
        // prev_flat (K=3): [2,3,4], [1,3,4], [1,2,3], [1,2,4]
        // Both prefix-drop subsets exist → pair is valid
        let groups = K3PlusGroupsResult {
            prefix_items: vec![1, 2],
            prefix_offsets: vec![0, 2],
            suffixes: vec![3, 4],
            suffix_offsets: vec![0, 2],
            cumulative_pairs: vec![0, 1],
            total_candidates: 1,
            suffix_src_rows: vec![],
        };

        let prev_flat: Vec<i32> = vec![2, 3, 4, 1, 3, 4, 1, 2, 3, 1, 2, 4];
        let result = prune_groups_apriori_raw(&groups, &prev_flat, 4, 4).unwrap();
        assert_eq!(result.total_candidates, 1);
    }

    #[test]
    fn test_prune_groups_k4_missing_subset() {
        // K=4: prefix=[1,2], suffixes=[3,4]
        // Candidate [1,2,3,4]
        // prev_flat (K=3): [2,3,4] only — [1,3,4] missing → prune
        let groups = K3PlusGroupsResult {
            prefix_items: vec![1, 2],
            prefix_offsets: vec![0, 2],
            suffixes: vec![3, 4],
            suffix_offsets: vec![0, 2],
            cumulative_pairs: vec![0, 1],
            total_candidates: 1,
            suffix_src_rows: vec![],
        };

        let prev_flat: Vec<i32> = vec![2, 3, 4]; // [1,3,4] missing
        let result = prune_groups_apriori_raw(&groups, &prev_flat, 1, 4);
        assert!(result.is_none());
    }

    #[test]
    fn test_prune_groups_multiple_groups() {
        // Two groups, one survives, one doesn't
        // Group 1: prefix=[1], suffixes=[2,3] → check (2,3) in prev
        // Group 2: prefix=[5], suffixes=[6,7] → check (6,7) in prev
        // prev contains (2,3) but NOT (6,7)
        let groups = K3PlusGroupsResult {
            prefix_items: vec![1, 5],
            prefix_offsets: vec![0, 1, 2],
            suffixes: vec![2, 3, 6, 7],
            suffix_offsets: vec![0, 2, 4],
            cumulative_pairs: vec![0, 1, 2],
            total_candidates: 2,
            suffix_src_rows: vec![],
        };

        let prev_flat: Vec<i32> = vec![2, 3];
        let result = prune_groups_apriori_raw(&groups, &prev_flat, 1, 3).unwrap();
        assert_eq!(result.total_candidates, 1);
        assert_eq!(&result.prefix_items, &[1]);
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

    #[test]
    fn test_prune_carries_src_rows() {
        // Same fixture as test_prune_groups_k3_basic, with rows attached:
        // suffix 4 is pruned, so its row (30) must disappear with it.
        let groups = K3PlusGroupsResult {
            prefix_items: vec![1],
            prefix_offsets: vec![0, 1],
            suffixes: vec![2, 3, 4],
            suffix_offsets: vec![0, 3],
            cumulative_pairs: vec![0, 3],
            total_candidates: 3,
            suffix_src_rows: vec![10, 20, 30],
        };
        let prev_flat: Vec<i32> = vec![1, 2, 1, 3, 2, 3, 1, 4];
        let result = prune_groups_apriori_raw(&groups, &prev_flat, 4, 3).unwrap();
        assert_eq!(result.suffixes, vec![2, 3]);
        assert_eq!(result.suffix_src_rows, vec![10, 20]);

        // Without rows nothing is tracked and nothing is asserted about them.
        let bare = K3PlusGroupsResult { suffix_src_rows: vec![], ..groups };
        let result = prune_groups_apriori_raw(&bare, &prev_flat, 4, 3).unwrap();
        assert!(result.suffix_src_rows.is_empty());
    }
}
