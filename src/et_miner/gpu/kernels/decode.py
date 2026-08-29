"""Decoders: flat GPU result indices back into itemset tuples/arrays."""

from __future__ import annotations

import numpy as np



def decode_k2_pairs_flat(freq_indices, freq_cols):
    """Vectorized decode of pair indices to flat numpy array.

    Returns (n_freq, 2) int32 array instead of Python tuples.
    No Python loops — pure numpy.

    Args:
        freq_indices: numpy array of pair indices that passed threshold.
        freq_cols: sorted list/array of frequent column indices.

    Returns:
        numpy int32 array of shape (n_freq, 2) with column indices.
    """
    freq_cols_arr = np.array(freq_cols, dtype=np.int32)
    fi = freq_indices.astype(np.float64)
    j_vals = np.floor(0.5 + np.sqrt(0.25 + 2.0 * fi)).astype(np.int64)
    i_vals = (freq_indices - j_vals * (j_vals - 1) // 2).astype(np.int64)
    result = np.empty((len(freq_indices), 2), dtype=np.int32)
    result[:, 0] = freq_cols_arr[i_vals]
    result[:, 1] = freq_cols_arr[j_vals]
    return result


def decode_k3plus_candidates(freq_indices, groups_info):
    """Vectorized decode of candidate indices to itemset tuples.

    Uses numpy vectorized binary search + triangular inverse for the
    group lookup, then Python loop for final tuple construction.

    Args:
        freq_indices: numpy array of candidate indices that passed threshold.
        groups_info: K3PlusGroups namedtuple from build_k3plus_groups().

    Returns:
        list of tuples (column index tuples).
    """
    cp_np = groups_info.cumulative_pairs

    # Vectorized binary search: find group for each index
    group_indices = np.searchsorted(cp_np, freq_indices, side="right") - 1
    pair_indices = freq_indices - cp_np[group_indices]

    # Vectorized triangular inverse
    pi_f = pair_indices.astype(np.float64)
    j_vals = np.floor(0.5 + np.sqrt(0.25 + 2.0 * pi_f)).astype(np.int64)
    i_vals = (pair_indices - j_vals * (j_vals - 1) // 2).astype(np.int64)

    # Decode each candidate
    candidates = []
    for idx in range(len(freq_indices)):
        g = int(group_indices[idx])
        prefix = tuple(groups_info.prefix_items[groups_info.prefix_offsets[g] : groups_info.prefix_offsets[g + 1]])
        suf_start = groups_info.suffix_offsets[g]
        suffix_i = int(groups_info.suffixes[suf_start + int(i_vals[idx])])
        suffix_j = int(groups_info.suffixes[suf_start + int(j_vals[idx])])
        candidates.append(prefix + (suffix_i, suffix_j))

    return candidates


def decode_k3plus_flat(freq_indices, groups_info, k):
    """Vectorized decode of candidate indices to flat numpy array.

    Returns (n_freq, k) int32 array instead of Python tuples.
    Minimizes Python loops via vectorized prefix gather.

    Args:
        freq_indices: numpy array of candidate indices that passed threshold.
        groups_info: K3PlusGroups namedtuple from build_k3plus_groups().
        k: current itemset size.

    Returns:
        numpy int32 array of shape (n_freq, k) with column indices.
    """
    n = len(freq_indices)
    cp_np = groups_info.cumulative_pairs

    # Vectorized binary search + triangular inverse
    group_indices = np.searchsorted(cp_np, freq_indices, side="right") - 1
    pair_indices = freq_indices - cp_np[group_indices]
    pi_f = pair_indices.astype(np.float64)
    j_vals = np.floor(0.5 + np.sqrt(0.25 + 2.0 * pi_f)).astype(np.int64)
    i_vals = (pair_indices - j_vals * (j_vals - 1) // 2).astype(np.int64)

    # Vectorized suffix lookup
    suf_starts = groups_info.suffix_offsets[group_indices]
    col_suffix_i = groups_info.suffixes[suf_starts + i_vals]
    col_suffix_j = groups_info.suffixes[suf_starts + j_vals]

    result = np.empty((n, k), dtype=np.int32)

    # Vectorized prefix gather: build prefix columns
    prefix_len = k - 2
    pref_starts = groups_info.prefix_offsets[group_indices]
    for p in range(prefix_len):
        result[:, p] = groups_info.prefix_items[pref_starts + p]

    result[:, -2] = col_suffix_i
    result[:, -1] = col_suffix_j
    return result


