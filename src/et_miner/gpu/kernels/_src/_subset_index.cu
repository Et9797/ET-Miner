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
// Plain C, sm_60+.

#define SUBSET_PRUNE 1
#define SUBSET_INFER 2

#define CAND_COUNT 0
#define CAND_SKIP 1
#define CAND_INFER 2

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

// CAND_SKIP, CAND_INFER (with *inferred set) or CAND_COUNT for pref ++ (a, b).
static __device__ int _classify_candidate(const int* __restrict__ rows, long long n_rows,
                                          const int* __restrict__ counts, const unsigned char* __restrict__ free_flags,
                                          int mode, const int* pref, int plen, int a, int b, int* inferred)
{
    const int m = plen + 1;
    const int infer = mode & SUBSET_INFER;
    int lowest = 0x7FFFFFFF;
    int not_free = 0;
    for (int d = 0; d < plen; d++) {
        const long long r = _find_subset(rows, n_rows, m, pref, plen - 1, d, a, b);
        if (r < 0) return CAND_SKIP;
        if (infer) {
            lowest = counts[r] < lowest ? counts[r] : lowest;
            not_free |= free_flags[r] == 0;
        }
    }
    if (!infer) return CAND_COUNT;
    for (int p = 0; p < 2; p++) {
        const long long r = _find_subset(rows, n_rows, m, pref, plen, plen, p ? b : a, 0);
        if (r < 0) return CAND_COUNT;
        lowest = counts[r] < lowest ? counts[r] : lowest;
        not_free |= free_flags[r] == 0;
    }
    if (!not_free) return CAND_COUNT;
    *inferred = lowest;
    return CAND_INFER;
}
