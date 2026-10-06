// Row-wise K=2 counting: every row adds 1 to each pair of its frequent-column
// positions. Pair (a < b) of the frequent-column list lives at
// b*(b-1)/2 + a, the layout of count_pairs_k2_dense and decode_k2_pairs_flat,
// so the reduce, the threshold filter and the decode are shared with the
// dense kernels. Only pairs in [pair_lo, pair_lo + n_out) are counted, into
// out[pair - pair_lo]; b_lo / b_hi are the b of the first and the last pair
// of that range (computed exactly on the host), so a row enumerates only the
// pairs whose b falls in the range.
//
// Input per shard: row_ptr (n_rows + 1, int64) and pos (int32), each row's
// frequent-column positions strictly ascending. One warp per row, lanes
// striding over the row's pairs; grid-stride over rows.
//
// count_pairs_k2_rows adds with global atomics. count_pairs_k2_rows_shared
// privatizes the range in shared memory (n_out <= K2_ROWS_SHARED_PAIRS,
// checked on the host) and flushes each block's nonzero counters once.

#define K2_ROWS_SHARED_PAIRS 11264

static __device__ long long _k2_tri(long long t) { return t * (t - 1) / 2; }

// First slot s in [0, m) with p[s] >= v, or m.
static __device__ long long _k2_lower_bound(const int* __restrict__ p, long long m, long long v)
{
    long long lo = 0, hi = m;
    while (lo < hi) {
        long long mid = (lo + hi) / 2;
        if ((long long)p[mid] < v) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// Adds row r's pairs that fall in [pair_lo, pair_lo + n_out) to out.
static __device__ void _k2_count_row(
    const long long* __restrict__ row_ptr, const int* __restrict__ pos, long long r,
    long long pair_lo, long long n_out, long long b_lo, long long b_hi, int lane, int* out)
{
    const long long start = row_ptr[r];
    const long long m = row_ptr[r + 1] - start;
    if (m < 2) return;
    const int* p = pos + start;
    long long t0 = _k2_lower_bound(p, m, b_lo);
    if (t0 < 1) t0 = 1;
    const long long t1 = _k2_lower_bound(p, m, b_hi + 1);
    if (t1 <= t0) return;
    const long long q_end = _k2_tri(t1);
    for (long long q = _k2_tri(t0) + lane; q < q_end; q += 32) {
        // q = t*(t-1)/2 + s with s < t: slot t holds b, slot s holds a.
        long long t = (long long)floor(0.5 + sqrt(0.25 + 2.0 * (double)q));
        while (_k2_tri(t) > q) t--;
        while (_k2_tri(t + 1) <= q) t++;
        const long long s = q - _k2_tri(t);
        const long long idx = _k2_tri((long long)p[t]) + (long long)p[s] - pair_lo;
        if (idx >= 0 && idx < n_out) atomicAdd(out + idx, 1);
    }
}

extern "C" __global__
void count_pairs_k2_rows(
    const long long* __restrict__ row_ptr,
    const int* __restrict__ pos,
    const long long n_rows,
    const long long pair_lo,
    const long long n_out,
    const long long b_lo,
    const long long b_hi,
    int* __restrict__ out  // int32: counts <= n_transactions < 2^31 (guarded host-side)
) {
    const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const long long n_warps = ((long long)gridDim.x * blockDim.x) / 32;
    const int lane = threadIdx.x % 32;
    for (long long r = warp; r < n_rows; r += n_warps)
        _k2_count_row(row_ptr, pos, r, pair_lo, n_out, b_lo, b_hi, lane, out);
}

extern "C" __global__
void count_pairs_k2_rows_shared(
    const long long* __restrict__ row_ptr,
    const int* __restrict__ pos,
    const long long n_rows,
    const long long pair_lo,
    const long long n_out,
    const long long b_lo,
    const long long b_hi,
    int* __restrict__ out
) {
    __shared__ int hist[K2_ROWS_SHARED_PAIRS];
    for (long long x = threadIdx.x; x < n_out; x += blockDim.x) hist[x] = 0;
    __syncthreads();

    const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const long long n_warps = ((long long)gridDim.x * blockDim.x) / 32;
    const int lane = threadIdx.x % 32;
    for (long long r = warp; r < n_rows; r += n_warps)
        _k2_count_row(row_ptr, pos, r, pair_lo, n_out, b_lo, b_hi, lane, hist);
    __syncthreads();

    for (long long x = threadIdx.x; x < n_out; x += blockDim.x) {
        const int c = hist[x];
        if (c) atomicAdd(out + x, c);
    }
}
