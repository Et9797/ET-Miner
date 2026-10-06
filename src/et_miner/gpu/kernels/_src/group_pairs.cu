// Group kernel: one block per prefix group, work proportional to the pairs it counts.
//
// A block takes one prefix group of at most GP_MAX_SUFFIXES suffixes. It first
// classifies the group's pairs with the subset test (_subset_index.cu, prepended
// by the loader): skipped pairs are left unwritten, inferred pairs get the
// inferred count on the device that writes them and 0 on the others, and the
// pairs to count go into a list in shared memory. Then, one tile of GP_TILE_W
// words at a time, it stages the prefix AND and the rows of the suffixes the
// listed pairs use, and counts only the listed pairs. The entries written are
// the per-candidate kernel's, in the same chunk-relative layout: pair (i < j)
// of group g is candidate cumulative_pairs[g] + j*(j-1)/2 + i.
//
// Two counting layouts, chosen per block from the listed pairs n:
//   n <= 8 * GP_LANE_PAIRS: lanes are the words of the tile; warp w owns pairs
//     w, w+8, ... with one register accumulator per pair per lane, reduced
//     across the warp once at the end;
//   larger n: threads are pairs; thread t owns pairs t, t+256, ... and walks
//     the tile's words.
//
// Static shared memory: 65 staged rows x 33 words x 8 B = 17,160 B, two
// 2,016-entry unsigned short pair arrays, small per-suffix arrays: ~26 KB.
// Constraints shared with the rest of _src/: plain C, extern "C", sm_60+
// intrinsics only, blockDim.x == 256.

#define GP_MAX_SUFFIXES 64
#define GP_MAX_PAIRS (GP_MAX_SUFFIXES * (GP_MAX_SUFFIXES - 1) / 2)
#define GP_TILE_W 32
#define GP_LANE_PAIRS 16
#define GP_THREAD_PAIRS ((GP_MAX_PAIRS + 255) / 256)

extern "C" __global__
void count_group_pairs(
    const unsigned long long* __restrict__ bitvecs,
    const int* __restrict__ group_prefix_items,
    const long long* __restrict__ group_prefix_offsets,
    const int* __restrict__ group_suffixes,
    const long long* __restrict__ group_suffix_offsets,
    const long long* __restrict__ cumulative_pairs,
    const long long n_u64s,
    const long long group_start,
    const long long group_end,
    const long long cand_offset,                  // chunk-relative output indexing
    int* __restrict__ result_counts,              // int32, one slot per chunk candidate
    const int* __restrict__ index_rows,           // previous level, sorted (_subset_index.cu)
    const long long index_n,
    const int* __restrict__ index_counts,
    const unsigned char* __restrict__ index_free,
    const int index_mode,                         // 0: count every candidate
    const int write_inferred                      // this device writes inferred counts
) {
    const long long g = group_start + (long long)blockIdx.y * (long long)gridDim.x + (long long)blockIdx.x;
    if (g >= group_end) return;

    const long long suf_start = group_suffix_offsets[g];
    const int n_suf = (int)(group_suffix_offsets[g + 1] - suf_start);
    if (n_suf < 2) return;
    const int n_all = n_suf * (n_suf - 1) / 2;
    const long long pref_start = group_prefix_offsets[g];
    const int prefix_len = (int)(group_prefix_offsets[g + 1] - pref_start);
    const long long cand_base = cumulative_pairs[g] - cand_offset;

    __shared__ int s_pref[62];
    __shared__ int s_used[GP_MAX_SUFFIXES];        // suffix position -> 1 when a listed pair uses it
    __shared__ int s_slot[GP_MAX_SUFFIXES];        // suffix position -> staged row (1 + slot)
    __shared__ int s_used_items[GP_MAX_SUFFIXES];  // staged row - 1 -> item
    __shared__ unsigned short s_pairs[GP_MAX_PAIRS];  // listed pair -> its index p within the group
    __shared__ unsigned short s_rows_of[GP_MAX_PAIRS];  // listed pair -> staged rows, ri | rj << 7
    __shared__ int s_n_pairs;
    __shared__ int s_n_used;
    __shared__ unsigned long long s_rows[GP_MAX_SUFFIXES + 1][GP_TILE_W + 1];  // row 0: prefix AND

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (tid < prefix_len && tid < 62) s_pref[tid] = group_prefix_items[pref_start + tid];
    if (tid < GP_MAX_SUFFIXES) s_used[tid] = 0;
    if (tid == 0) s_n_pairs = 0;
    __syncthreads();

    // Classify: pair p = j*(j-1)/2 + i (i < j), the per-candidate kernel's order.
    for (int p = tid; p < n_all; p += blockDim.x) {
        int j = (int)((1.0f + sqrtf(1.0f + 8.0f * (float)p)) * 0.5f);
        while (j * (j - 1) / 2 > p) j--;
        while ((j + 1) * j / 2 <= p) j++;
        const int i = p - j * (j - 1) / 2;
        int status = CAND_COUNT;
        int inferred = 0;
        if (index_mode != 0) {
            status = _classify_candidate(index_rows, index_n, index_counts, index_free, index_mode, s_pref,
                                         prefix_len, group_suffixes[suf_start + i], group_suffixes[suf_start + j],
                                         &inferred);
        }
        if (status == CAND_INFER) {
            result_counts[cand_base + p] = write_inferred ? inferred : 0;
        } else if (status == CAND_COUNT) {
            const int pos = atomicAdd(&s_n_pairs, 1);
            s_pairs[pos] = (unsigned short)p;
            s_used[i] = 1;
            s_used[j] = 1;
        }
    }
    __syncthreads();
    const int n_pairs = s_n_pairs;
    if (n_pairs == 0) return;

    // Staged rows of the used suffixes, in position order.
    if (warp == 0) {
        const int lo_used = lane < n_suf && s_used[lane];
        const int hi_used = lane + 32 < n_suf && s_used[lane + 32];
        const unsigned int lo = __ballot_sync(0xFFFFFFFFu, lo_used);
        const unsigned int hi = __ballot_sync(0xFFFFFFFFu, hi_used);
        const unsigned int below = (1u << lane) - 1u;
        if (lo_used) {
            const int slot = __popc(lo & below);
            s_slot[lane] = slot;
            s_used_items[slot] = group_suffixes[suf_start + lane];
        }
        if (hi_used) {
            const int slot = __popc(lo) + __popc(hi & below);
            s_slot[lane + 32] = slot;
            s_used_items[slot] = group_suffixes[suf_start + lane + 32];
        }
        if (lane == 0) s_n_used = __popc(lo) + __popc(hi);
    }
    __syncthreads();
    for (int q = tid; q < n_pairs; q += blockDim.x) {
        const int p = s_pairs[q];
        int j = (int)((1.0f + sqrtf(1.0f + 8.0f * (float)p)) * 0.5f);
        while (j * (j - 1) / 2 > p) j--;
        while ((j + 1) * j / 2 <= p) j++;
        const int i = p - j * (j - 1) / 2;
        s_rows_of[q] = (unsigned short)((1 + s_slot[i]) | ((1 + s_slot[j]) << 7));
    }
    const int n_used = s_n_used;
    const int lane_layout = n_pairs <= 8 * GP_LANE_PAIRS;
    unsigned int acc[GP_LANE_PAIRS];
    #pragma unroll
    for (int q = 0; q < GP_LANE_PAIRS; q++) acc[q] = 0;
    __syncthreads();

    for (long long word_base = 0; word_base < n_u64s; word_base += GP_TILE_W) {
        const int total = (1 + n_used) * GP_TILE_W;
        for (int idx = tid; idx < total; idx += blockDim.x) {
            const int row = idx / GP_TILE_W;
            const int wo = idx % GP_TILE_W;
            const long long w = word_base + wo;
            unsigned long long v = 0ULL;
            if (w < n_u64s) {
                if (row == 0) {
                    v = ~0ULL;
                    for (int p = 0; p < prefix_len; p++) v &= bitvecs[(long long)s_pref[p] * n_u64s + w];
                } else {
                    v = bitvecs[(long long)s_used_items[row - 1] * n_u64s + w];
                }
            }
            s_rows[row][wo] = v;
        }
        __syncthreads();
        if (lane_layout) {
            const unsigned long long pa = s_rows[0][lane];
            if (__ballot_sync(0xFFFFFFFFu, pa != 0ULL) != 0u) {
                #pragma unroll
                for (int q = 0; q < GP_LANE_PAIRS; q++) {
                    const int p = warp + 8 * q;
                    if (p < n_pairs) {
                        const int rr = s_rows_of[p];
                        acc[q] += (unsigned int)__popcll(pa & s_rows[rr & 127][lane] & s_rows[rr >> 7][lane]);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int q = 0; q < GP_THREAD_PAIRS; q++) {
                const int p = tid + 256 * q;
                if (p < n_pairs) {
                    const int rr = s_rows_of[p];
                    const int ri = rr & 127;
                    const int rj = rr >> 7;
                    unsigned int c = 0;
                    for (int w = 0; w < GP_TILE_W; w++) {
                        c += (unsigned int)__popcll(s_rows[0][w] & s_rows[ri][w] & s_rows[rj][w]);
                    }
                    acc[q] += c;
                }
            }
        }
        __syncthreads();
    }

    if (lane_layout) {
        #pragma unroll
        for (int q = 0; q < GP_LANE_PAIRS; q++) {
            const int p = warp + 8 * q;
            if (p < n_pairs) {
                unsigned int c = acc[q];
                for (int offset = 16; offset > 0; offset >>= 1) c += __shfl_down_sync(0xFFFFFFFFu, c, offset);
                if (lane == 0) result_counts[cand_base + s_pairs[p]] = (int)c;
            }
        }
    } else {
        #pragma unroll
        for (int q = 0; q < GP_THREAD_PAIRS; q++) {
            const int p = tid + 256 * q;
            if (p < n_pairs) result_counts[cand_base + s_pairs[p]] = (int)acc[q];
        }
    }
}
