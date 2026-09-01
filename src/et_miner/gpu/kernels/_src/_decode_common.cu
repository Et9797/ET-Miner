// Shared candidate decode for the K>=3 prefix-group kernels.
//
// kernels/loader.py prepends this file (see _KERNEL_PRELUDES) to every .cu
// that lists it, so the arithmetic that must stay identical to the host-side
// decode (kernels/decode.py::decode_k3plus_flat — searchsorted on
// cumulative_pairs, then the triangular inverse) exists exactly once.
//
// Candidate index `cand` -> (group g, suffix slots i < j within the group).
// Returns 0 when `cand` is out of range or the triangular inverse lands
// outside the group; callers then produce no count and write nothing.
// Plain C, sm_60+ only (no templates, no headers — NVRTC provides floor/sqrt).
static __device__ int _decode_candidate(
    const long long* __restrict__ cumulative_pairs,   // (n_groups + 1), [0]-prefixed
    const long long* __restrict__ suffix_offsets,     // (n_groups + 1)
    const long long n_groups,
    const long long cand,
    long long* g_out, long long* i_out, long long* j_out)
{
    if (n_groups <= 0 || cand < 0 || cand >= cumulative_pairs[n_groups]) return 0;

    long long lo = 0, hi = n_groups - 1;
    while (lo < hi) {
        long long mid = (lo + hi + 1) / 2;
        if (cumulative_pairs[mid] <= cand) lo = mid;
        else hi = mid - 1;
    }
    const long long g = lo;
    const long long pair_idx = cand - cumulative_pairs[g];

    const long long suf_start = suffix_offsets[g];
    const long long group_size = suffix_offsets[g + 1] - suf_start;

    const long long j_val = (long long)floor(0.5 + sqrt(0.25 + 2.0 * (double)pair_idx));
    const long long i_val = pair_idx - j_val * (j_val - 1) / 2;
    if (j_val >= group_size || i_val >= j_val || i_val < 0) return 0;

    *g_out = g;
    *i_out = i_val;
    *j_out = j_val;
    return 1;
}
