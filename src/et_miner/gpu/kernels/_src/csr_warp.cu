// csr_warp.cu — warp-cooperative sorted-set intersection over CSR tidset
// shards, for the sparse-CSR mining path. The loader prepends
// _decode_common.cu (candidate -> group / suffix slots).
//
// Mapping: blockDim.x == 256 = 8 warps; the warp with linear id w handles
// candidate w of the launch (CANDS_PER_BLOCK is coupled to kernels/csr_warp.py).
// Every warp-level loop has a warp-uniform trip count, and the whole warp
// returns together on w >= n before any sync intrinsic runs.
//
// Shard layout: `tids` int32 (rows concatenated, strictly increasing within a
// row), `offsets` int64 (n_rows + 1). A candidate's two parent rows come from
// suffix_src_rows[suffix slot]. Counts are int32 (bounded by n_transactions,
// guarded < 2^31 host-side). Plain C, sm_60+ intrinsics only.

#define CANDS_PER_BLOCK 8

// First position in [lo, hi) with S[pos] >= key, else hi.
static __device__ long long _lower_bound(const int* __restrict__ S, long long lo, long long hi, int key) {
    long long len = hi - lo;
    while (len > 0) {
        const long long half = len >> 1;
        const long long mid = lo + half;
        if (S[mid] < key) {
            lo = mid + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }
    return lo;
}

// |A ∩ B| for strictly increasing A and B. When `out` is non-null the sorted
// intersection is also written to out[0..min(count, cap)). All 32 lanes of the
// warp must call this with identical arguments.
//
// The smaller set is probed 32 elements per chunk (coalesced loads); every
// lane binary-searches its element in the larger set inside a window bounded
// by a cursor (everything below `cur` is smaller than all remaining probe
// keys) and by a warp-uniform search for the chunk maximum, which also yields
// the next cursor. Found lanes are compacted with a ballot, so the written
// order follows the probe order — i.e. ascending — and the count and write
// passes are deterministic and agree exactly.
static __device__ long long _warp_intersect(const int* __restrict__ A, long long nA,
                                            const int* __restrict__ B, long long nB,
                                            int* out, long long cap, int lane) {
    const int* P = A;
    long long nP = nA;
    const int* S = B;
    long long nS = nB;
    if (nB < nA) {
        P = B; nP = nB;
        S = A; nS = nA;
    }
    long long count = 0;
    long long cur = 0;
    for (long long base = 0; base < nP; base += 32) {              // warp-uniform trip count
        const long long idx = base + lane;
        const int valid = idx < nP;
        const long long last = (base + 31 < nP) ? (base + 31) : (nP - 1);   // uniform
        const int key = P[valid ? idx : last];                     // tail lanes mirror the chunk max
        const int kmax = P[last];
        const long long ub = _lower_bound(S, cur, nS, kmax);       // uniform: same loads in all lanes
        const long long hi = (ub < nS) ? (ub + 1) : nS;            // window includes S[ub] (may equal kmax)
        const long long pos = _lower_bound(S, cur, hi, key);       // per-lane search
        const int found = valid && (pos < hi) && (S[pos] == key);
        const unsigned int ballot = __ballot_sync(0xFFFFFFFFu, found);
        if (out != 0 && found) {
            const long long slot = count + __popc(ballot & ((1u << lane) - 1u));
            if (slot < cap) out[slot] = key;
        }
        count += __popc(ballot);
        cur = ub;                                                  // next chunk's keys are > kmax
    }
    return count;
}

// Candidate -> its two parent rows of the shard (via the shared decode).
static __device__ int _decode_rows(const long long* __restrict__ cumulative_pairs,
                                   const long long* __restrict__ suffix_offsets,
                                   const long long* __restrict__ suffix_src_rows,
                                   long long n_groups, long long cand,
                                   long long* row_a, long long* row_b) {
    long long g, i, j;
    if (!_decode_candidate(cumulative_pairs, suffix_offsets, n_groups, cand, &g, &i, &j)) return 0;
    const long long s0 = suffix_offsets[g];
    *row_a = suffix_src_rows[s0 + i];
    *row_b = suffix_src_rows[s0 + j];
    return 1;
}

static __device__ long long _linear_warp(void) {
    return ((long long)blockIdx.y * (long long)gridDim.x + (long long)blockIdx.x) * CANDS_PER_BLOCK
         + (long long)(threadIdx.x >> 5);
}

static __device__ long long _intersect_rows(const int* __restrict__ tids, const long long* __restrict__ offsets,
                                            long long ra, long long rb, int* out, long long cap, int lane) {
    const long long a0 = offsets[ra];
    const long long b0 = offsets[rb];
    return _warp_intersect(tids + a0, offsets[ra + 1] - a0, tids + b0, offsets[rb + 1] - b0, out, cap, lane);
}

#define NO_CAP 0x7FFFFFFFFFFFFFFFLL

// Partial counts for candidates [chunk_start, chunk_start + chunk_size) on this shard.
// With an index of the previous level (index_mode != 0, _subset_index.cu) lane 0
// classifies the candidate first: a skipped one keeps its zero, an inferred one
// gets its count on the device that writes inferred counts and 0 on the others.
extern "C" __global__
void csr_count_range(const int* __restrict__ tids,
                     const long long* __restrict__ offsets,
                     const long long* __restrict__ cumulative_pairs,
                     const long long* __restrict__ suffix_offsets,
                     const long long* __restrict__ suffix_src_rows,
                     const long long n_groups,
                     const long long chunk_start,
                     const long long chunk_size,
                     int* __restrict__ out_counts,
                     const int* __restrict__ group_prefix_items,
                     const long long* __restrict__ group_prefix_offsets,
                     const int* __restrict__ group_suffixes,
                     const int* __restrict__ index_rows,
                     const long long index_n,
                     const int* __restrict__ index_counts,
                     const unsigned char* __restrict__ index_free,
                     const int index_mode,
                     const int write_inferred)
{
    const long long w = _linear_warp();
    if (w >= chunk_size) return;                                   // whole warp leaves together
    const int lane = threadIdx.x & 31;
    long long g, i, j, c = 0;
    if (_decode_candidate(cumulative_pairs, suffix_offsets, n_groups, chunk_start + w, &g, &i, &j)) {
        const long long s0 = suffix_offsets[g];
        if (index_mode != 0) {
            int status = CAND_COUNT, inferred = 0;
            if (lane == 0) {
                const long long p0 = group_prefix_offsets[g];
                status = _classify_candidate(index_rows, index_n, index_counts, index_free, index_mode,
                                             group_prefix_items + p0, (int)(group_prefix_offsets[g + 1] - p0),
                                             group_suffixes[s0 + i], group_suffixes[s0 + j], &inferred);
            }
            status = __shfl_sync(0xFFFFFFFFu, status, 0);         // warp-uniform from here on
            if (status != CAND_COUNT) {
                if (lane == 0 && status == CAND_INFER) out_counts[w] = write_inferred ? inferred : 0;
                return;
            }
        }
        c = _intersect_rows(tids, offsets, suffix_src_rows[s0 + i], suffix_src_rows[s0 + j], (int*)0, 0, lane);
    }
    if (lane == 0) out_counts[w] = (int)c;
}

// Counts for an explicit list of candidate ids (survivors).
extern "C" __global__
void csr_count_gather(const int* __restrict__ tids,
                      const long long* __restrict__ offsets,
                      const long long* __restrict__ cumulative_pairs,
                      const long long* __restrict__ suffix_offsets,
                      const long long* __restrict__ suffix_src_rows,
                      const long long n_groups,
                      const long long* __restrict__ cand_ids,
                      const long long n,
                      int* __restrict__ out_counts)
{
    const long long w = _linear_warp();
    if (w >= n) return;
    const int lane = threadIdx.x & 31;
    long long ra, rb, c = 0;
    if (_decode_rows(cumulative_pairs, suffix_offsets, suffix_src_rows, n_groups, cand_ids[w], &ra, &rb)) {
        c = _intersect_rows(tids, offsets, ra, rb, (int*)0, 0, lane);
    }
    if (lane == 0) out_counts[w] = (int)c;
}

// Materialize the sorted intersections of an explicit list of candidates into a
// new CSR: survivor w's tids go to out_indices[out_offsets[w] .. out_offsets[w+1]).
// out_offsets must be the exclusive scan of csr_count_gather over the same ids.
extern "C" __global__
void csr_write_gather(const int* __restrict__ tids,
                      const long long* __restrict__ offsets,
                      const long long* __restrict__ cumulative_pairs,
                      const long long* __restrict__ suffix_offsets,
                      const long long* __restrict__ suffix_src_rows,
                      const long long n_groups,
                      const long long* __restrict__ cand_ids,
                      const long long n,
                      const long long* __restrict__ out_offsets,
                      int* __restrict__ out_indices)
{
    const long long w = _linear_warp();
    if (w >= n) return;
    const int lane = threadIdx.x & 31;
    long long ra, rb;
    if (_decode_rows(cumulative_pairs, suffix_offsets, suffix_src_rows, n_groups, cand_ids[w], &ra, &rb)) {
        _intersect_rows(tids, offsets, ra, rb, out_indices + out_offsets[w], NO_CAP, lane);
    }
}

// csr_write_gather for slots sized from counts this shard did not produce:
// survivor w writes at most out_offsets[w+1] - out_offsets[w] tids and reports
// its intersection's length in out_lengths[w], so a slot that does not match
// the shard is detected instead of overrun.
extern "C" __global__
void csr_write_gather_checked(const int* __restrict__ tids,
                              const long long* __restrict__ offsets,
                              const long long* __restrict__ cumulative_pairs,
                              const long long* __restrict__ suffix_offsets,
                              const long long* __restrict__ suffix_src_rows,
                              const long long n_groups,
                              const long long* __restrict__ cand_ids,
                              const long long n,
                              const long long* __restrict__ out_offsets,
                              int* __restrict__ out_indices,
                              int* __restrict__ out_lengths)
{
    const long long w = _linear_warp();
    if (w >= n) return;
    const int lane = threadIdx.x & 31;
    long long ra, rb, c = 0;
    if (_decode_rows(cumulative_pairs, suffix_offsets, suffix_src_rows, n_groups, cand_ids[w], &ra, &rb)) {
        c = _intersect_rows(tids, offsets, ra, rb, out_indices + out_offsets[w], out_offsets[w + 1] - out_offsets[w],
                            lane);
    }
    if (lane == 0) out_lengths[w] = (int)c;
}
