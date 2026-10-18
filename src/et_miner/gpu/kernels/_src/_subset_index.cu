// Exact (k-1)-subset test of a prefix-group candidate, shared by the K>=3
// counting kernels (kernels/loader.py prepends this file; see
// _KERNEL_PRELUDES).
//
// A candidate is a prefix of `plen` items followed by two suffixes a < b, so
// k = plen + 2. The index is the previous level: row-major int32 rows of
// plen + 1 items in strictly ascending lexicographic order, with one int32
// count and one free flag per row (kernels/subset_index.py builds it). One
// binary search finds any (k-1)-subset.
//
// Mode bits: SUBSET_PRUNE reports CAND_SKIP when a subset that drops a prefix
// item is missing from the index (the two subsets that drop a suffix are the
// candidate's prefix-parents, which are in the index by construction).
// SUBSET_INFER also reports CAND_INFER, with the minimum of the k subset
// counts, when one of the subsets is not free: some y then has
// rows(Y - y) = rows(Y), so rows(X - y) = rows(X) and the minimum is exact.
// SUBSET_BOUNDS (with SUBSET_INFER) settles the remaining candidates of up
// to SUBSET_BOUNDS_MAX_K items by the deduction bounds of depth <= 3
// (Calders & Goethals 2002), from the complete levels k-2 and k-3 with
// counts (deep2_*, deep3_*; the empty set's count deep_n_total at k = 3):
//   u = min( min_a s(X-a), min_{a<b<c} [s(X-a)+s(X-b)+s(X-c)
//                                        - s(X-ab)-s(X-ac)-s(X-bc) + s(X-abc)] )
//   l = max( 0, max_{a<b} [s(X-a) + s(X-b) - s(X-ab)] )
// u < deep_min_count reports CAND_SKIP (the candidate is infrequent); l = u
// reports CAND_INFER with u. Both are inclusion-exclusion inequalities over the
// rows that hold the subset, so they hold exactly for integer counts.
// Plain C, sm_60+.

#define SUBSET_PRUNE 1
#define SUBSET_INFER 2
#define SUBSET_BOUNDS 4

#define CAND_COUNT 0
#define CAND_SKIP 1
#define CAND_INFER 2

#define SUBSET_BOUNDS_MAX_K 24

// The deduction bounds' arguments, after a kernel's write_inferred parameter
// and after _classify_candidate's own arguments (each list starts with a comma).
#define SUBSET_DEEP_PARAMS                                                                 \
    , const int* __restrict__ deep2_rows, const long long deep2_n,                         \
    const int* __restrict__ deep2_counts, const int* __restrict__ deep3_rows,              \
    const long long deep3_n, const int* __restrict__ deep3_counts,                         \
    const long long deep_n_total, const int deep_min_count
#define SUBSET_DEEP_ARGS                                                                   \
    , deep2_rows, deep2_n, deep2_counts, deep3_rows, deep3_n, deep3_counts, deep_n_total, deep_min_count

// Item t of the itemset formed by pref without position `skip` (n_pref items
// kept), then a, then b.
static __device__ __forceinline__ int _subset_item(const int* pref, int n_pref, int skip, int a, int b, int t)
{
    if (t < n_pref) return pref[t < skip ? t : t + 1];
    return t == n_pref ? a : b;
}

// Row of the sorted (n_rows, m) index equal to that itemset, or -1.
static __device__ long long _find_subset(const int* __restrict__ rows, long long n_rows, int m,
                                         const int* pref, int n_pref, int skip, int a, int b)
{
    long long lo = 0, hi = n_rows;
    while (lo < hi) {
        const long long mid = lo + ((hi - lo) >> 1);
        const int* r = rows + mid * (long long)m;
        int cmp = 0;
        for (int t = 0; t < m && cmp == 0; t++) {
            const int v = _subset_item(pref, n_pref, skip, a, b, t);
            cmp = (r[t] < v) ? -1 : ((r[t] > v) ? 1 : 0);
        }
        if (cmp == 0) return mid;
        if (cmp < 0) lo = mid + 1;
        else hi = mid;
    }
    return -1;
}

// Row of the sorted (n_rows, m) index equal to items[0..m), or -1.
static __device__ long long _find_items(const int* __restrict__ rows, long long n_rows, int m, const int* items)
{
    long long lo = 0, hi = n_rows;
    while (lo < hi) {
        const long long mid = lo + ((hi - lo) >> 1);
        const int* r = rows + mid * (long long)m;
        int cmp = 0;
        for (int t = 0; t < m && cmp == 0; t++) cmp = (r[t] < items[t]) ? -1 : ((r[t] > items[t]) ? 1 : 0);
        if (cmp == 0) return mid;
        if (cmp < 0) lo = mid + 1;
        else hi = mid;
    }
    return -1;
}

// The deduction bounds of X = items[0..k) from its (k-1)-subset counts c1
// (c1[t]: X without item t): CAND_SKIP, CAND_INFER (with *inferred) or CAND_COUNT.
static __device__ __noinline__ int _deduce(const int* items, int k, const int* c1, int* inferred SUBSET_DEEP_PARAMS)
{
    int s2[SUBSET_BOUNDS_MAX_K * (SUBSET_BOUNDS_MAX_K - 1) / 2];
    int sub[SUBSET_BOUNDS_MAX_K];
    long long lower = 0, upper = 0x7FFFFFFF;
    for (int t = 0; t < k; t++) upper = c1[t] < upper ? c1[t] : upper;
    for (int j = 1; j < k; j++) {
        for (int i = 0; i < j; i++) {
            int n = 0;
            for (int t = 0; t < k; t++) if (t != i && t != j) sub[n++] = items[t];
            const long long r = _find_items(deep2_rows, deep2_n, k - 2, sub);
            if (r < 0) return CAND_COUNT;
            const int c = deep2_counts[r];
            s2[j * (j - 1) / 2 + i] = c;
            const long long l2 = (long long)c1[i] + c1[j] - c;
            lower = l2 > lower ? l2 : lower;
        }
    }
    for (int l = 2; l < k && lower < upper && upper >= deep_min_count; l++) {
        for (int j = 1; j < l && lower < upper; j++) {
            for (int i = 0; i < j && lower < upper; i++) {
                long long c3 = deep_n_total;
                if (k > 3) {
                    int n = 0;
                    for (int t = 0; t < k; t++) if (t != i && t != j && t != l) sub[n++] = items[t];
                    const long long r = _find_items(deep3_rows, deep3_n, k - 3, sub);
                    if (r < 0) return CAND_COUNT;
                    c3 = deep3_counts[r];
                }
                const long long u3 = (long long)c1[i] + c1[j] + c1[l] - s2[j * (j - 1) / 2 + i]
                                     - s2[l * (l - 1) / 2 + i] - s2[l * (l - 1) / 2 + j] + c3;
                upper = u3 < upper ? u3 : upper;
            }
        }
    }
    if (upper < deep_min_count) return CAND_SKIP;
    if (lower == upper) {
        *inferred = (int)upper;
        return CAND_INFER;
    }
    return CAND_COUNT;
}

// CAND_SKIP, CAND_INFER (with *inferred set) or CAND_COUNT for pref ++ (a, b).
static __device__ int _classify_candidate(const int* __restrict__ rows, long long n_rows,
                                          const int* __restrict__ counts, const unsigned char* __restrict__ free_flags,
                                          int mode, const int* pref, int plen, int a, int b, int* inferred
                                          SUBSET_DEEP_PARAMS)
{
    const int m = plen + 1;
    const int k = plen + 2;
    const int infer = mode & SUBSET_INFER;
    const int bounds = (mode & SUBSET_BOUNDS) && k <= SUBSET_BOUNDS_MAX_K;
    int c1[SUBSET_BOUNDS_MAX_K];
    int lowest = 0x7FFFFFFF;
    int not_free = 0;
    for (int d = 0; d < plen; d++) {
        const long long r = _find_subset(rows, n_rows, m, pref, plen - 1, d, a, b);
        if (r < 0) return CAND_SKIP;
        if (infer) {
            lowest = counts[r] < lowest ? counts[r] : lowest;
            not_free |= free_flags[r] == 0;
            if (bounds) c1[d] = counts[r];
        }
    }
    if (!infer) return CAND_COUNT;
    for (int p = 0; p < 2; p++) {
        const long long r = _find_subset(rows, n_rows, m, pref, plen, plen, p ? b : a, 0);
        if (r < 0) return CAND_COUNT;
        lowest = counts[r] < lowest ? counts[r] : lowest;
        not_free |= free_flags[r] == 0;
        if (bounds) c1[p ? plen : plen + 1] = counts[r];
    }
    if (not_free) {
        *inferred = lowest;
        return CAND_INFER;
    }
    if (!bounds) return CAND_COUNT;
    int items[SUBSET_BOUNDS_MAX_K];
    for (int t = 0; t < plen; t++) items[t] = pref[t];
    items[plen] = a;
    items[plen + 1] = b;
    return _deduce(items, k, c1, inferred SUBSET_DEEP_ARGS);
}
