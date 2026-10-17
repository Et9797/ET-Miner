// Row-wise K=3 counting over the frequent-pair graph.
//
// The graph's vertices are the active positions 0..m-1 (the frequent columns
// that occur in a frequent pair, ascending). adj_off / adj hold each u's
// frequent partners v > u, ascending; the index of (u, v) in adj is its pair
// index, which is its rank in the lexsorted K=2 level. A counted candidate
// (u, v, w), u < v < w, has all three pairs frequent. Pair p = (u, v) owns the
// slots slot_off[p] .. slot_off[p + 1], one per partner w of v; the candidate
// (u, v, w) counts at slot_off[p] + the rank of w among v's partners. Slots
// whose (u, w) is not a frequent pair are never written and stay 0.
//
// One warp per row (grid-stride over rows), each row's active positions
// strictly ascending and at most K3_ROWS_MAX_LEN of them (checked on the
// host). The warp stages the positions in shared memory and builds each
// position's 64-bit mask of the later positions it forms a frequent pair with.
// Then the row's frequent pairs (i, j) are spread over the lanes; each adds 1
// for every w in nbr[i] & nbr[j]. Pair indices and ranks come from pair_bits
// (m rows of bits_words words, bit v of row u set for a frequent pair) and
// pair_rank (the set bits of row u before each word) when the host built them,
// otherwise from searches of adj. Only slots in [slot_lo, slot_hi) are counted,
// into out[slot - slot_lo]; pair_lo / pair_hi bound the pairs that own them.

#define K3_ROWS_MAX_LEN 64
#define K3_ROWS_WARPS 8

static __device__ long long _k3_lower_bound(const int* __restrict__ a, long long lo, long long hi, int x)
{
    while (lo < hi) {
        long long mid = (lo + hi) / 2;
        if (a[mid] < x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// Rank of v among u's partners (v must be one), from the bitmap or by search.
static __device__ long long _k3_rank(
    const long long* __restrict__ adj_off, const int* __restrict__ adj,
    const unsigned long long* __restrict__ pair_bits, const int* __restrict__ pair_rank,
    long long bits_words, int u, int v)
{
    if (bits_words > 0) {
        const long long word = (long long)u * bits_words + v / 64;
        return pair_rank[word] + __popcll(pair_bits[word] & ((1ULL << (v % 64)) - 1ULL));
    }
    return _k3_lower_bound(adj, adj_off[u], adj_off[u + 1], v) - adj_off[u];
}

extern "C" __global__
void count_k3_rows(
    const long long* __restrict__ row_ptr,
    const int* __restrict__ pos,
    const long long n_rows,
    const long long* __restrict__ adj_off,
    const int* __restrict__ adj,
    const long long* __restrict__ slot_off,
    const unsigned long long* __restrict__ pair_bits,
    const int* __restrict__ pair_rank,
    const long long bits_words,
    const long long pair_lo,
    const long long pair_hi,
    const long long slot_lo,
    const long long slot_hi,
    int* __restrict__ out
) {
    __shared__ int s_pos[K3_ROWS_WARPS][K3_ROWS_MAX_LEN];
    __shared__ unsigned long long s_nbr[K3_ROWS_WARPS][K3_ROWS_MAX_LEN];
    __shared__ int s_first[K3_ROWS_WARPS][K3_ROWS_MAX_LEN + 1];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    int* sp = s_pos[warp];
    unsigned long long* sn = s_nbr[warp];
    int* sf = s_first[warp];

    for (long long r = (long long)blockIdx.x * K3_ROWS_WARPS + warp; r < n_rows;
         r += (long long)gridDim.x * K3_ROWS_WARPS) {
        const long long start = row_ptr[r];
        const int len = (int)(row_ptr[r + 1] - start);
        if (len < 3) continue;  // uniform across the warp
        for (int i = lane; i < len; i += 32) sp[i] = pos[start + i];
        __syncwarp();

        // Lane t takes positions t and 63 - t: their later-position counts sum to 63.
        for (int h = 0; h < 2; h++) {
            const int i = h ? K3_ROWS_MAX_LEN - 1 - lane : lane;
            if (i >= len) continue;
            const int u = sp[i];
            unsigned long long mask = 0;
            if (bits_words > 0) {
                const unsigned long long* row_bits = pair_bits + (long long)u * bits_words;
                for (int j = i + 1; j < len; j++) {
                    const int v = sp[j];
                    if ((row_bits[v / 64] >> (v % 64)) & 1ULL) mask |= 1ULL << j;
                }
            } else {
                long long lo = adj_off[u];
                const long long hi = adj_off[u + 1];
                for (int j = i + 1; j < len && lo < hi; j++) {
                    lo = _k3_lower_bound(adj, lo, hi, sp[j]);
                    if (lo < hi && adj[lo] == sp[j]) mask |= 1ULL << j;
                }
            }
            sn[i] = mask;
        }
        __syncwarp();

        // The row's frequent pairs (i, j), numbered i-major: sf[i] is the first number of i's.
        if (lane == 0) {
            sf[0] = 0;
            for (int i = 0; i < len; i++) sf[i + 1] = sf[i] + __popcll(sn[i]);
        }
        __syncwarp();

        for (int f = lane; f < sf[len]; f += 32) {
            int lo_i = 0, hi_i = len;  // the last i with sf[i] <= f
            while (hi_i - lo_i > 1) {
                const int mid = (lo_i + hi_i) / 2;
                if (sf[mid] <= f) lo_i = mid;
                else hi_i = mid;
            }
            const int i = lo_i;
            unsigned long long bits = sn[i];
            for (int skip = f - sf[i]; skip > 0; skip--) bits &= bits - 1;
            const int j = __ffsll((long long)bits) - 1;
            unsigned long long common = sn[i] & sn[j];
            if (!common) continue;
            const int u = sp[i], v = sp[j];
            const long long p = adj_off[u] + _k3_rank(adj_off, adj, pair_bits, pair_rank, bits_words, u, v);
            if (p < pair_lo || p >= pair_hi) continue;
            const long long s0 = slot_off[p];
            while (common) {
                const int k = __ffsll((long long)common) - 1;
                common &= common - 1;
                const long long slot = s0 + _k3_rank(adj_off, adj, pair_bits, pair_rank, bits_words, v, sp[k]);
                if (slot >= slot_lo && slot < slot_hi) atomicAdd(out + (slot - slot_lo), 1);
            }
        }
        __syncwarp();
    }
}
