
// Shared/tiled K>=2 counting: prefix sharing + suffix-pair tiling.
//
// One block = one (suffix-tile A, suffix-tile B) pair inside one prefix
// group (T=32 suffixes per tile). For each 32-word tile of the bitvectors
// the block cooperatively stages, in shared memory, the prefix AND
// (computed ONCE per block per word instead of once per candidate) and the
// two tiles' suffix rows; every thread then accumulates popcounts for its
// register-blocked pairs entirely out of shared memory. Global traffic per
// word per tile-pair drops from k loads x (pairs served) to
// (k-2) + 2T loads serving up to T*T pairs.
//
// Work mapping (host side, shared_tiled.py):
//   cumulative_tilepairs[g] = cumsum over groups of nt*(nt+1)/2 where
//   nt = ceil(group_size / T); a launch covers the tile-pair range of a
//   group-ALIGNED candidate chunk (dense variant) or everything (fused).
//
// Candidate indexing is bit-identical to the legacy kernels: within a
// group, pair (i < j) of suffix positions maps to cp[g] + j*(j-1)/2 + i.
// K=2 runs as one synthetic group with an empty prefix (prefix AND = ~0).
//
// Layout/occupancy: 256 threads (8 warps); static shared memory
//   s_pref[62]                       248 B   (K <= 62 cap, as legacy)
//   s_pa[TILE_W+1]                   264 B   staged prefix AND per word
//   s_sufA/s_sufB[T][TILE_W+1]     2x8448 B  (+1 pad kills bank conflicts)
// total ~17.5 KB — within the 48 KB static limit, 2 blocks/SM on sm_86.
//
// Constraints shared with the rest of _src/: plain C, extern "C",
// sm_60+ intrinsics only, blockDim.x == 256.

#define TILE_T 32
#define TILE_W 32

// tile-pair enumeration offset: first tile-pair index of row ta
// (ta = 0..nt-1, tb = ta..nt-1, row-major)
#define TP_OFFSET(ta, nt) ((ta) * (nt) - (ta) * ((ta) - 1) / 2)

extern "C" __device__ __forceinline__ void _stage_tiles(
    const unsigned long long* __restrict__ bitvecs,
    const int* __restrict__ group_suffixes,
    const long long suf_start,
    const long long group_size,
    const int* s_pref,
    const int prefix_len,
    const long long ta,
    const long long tb,
    const long long word_base,
    const long long n_u64s,
    unsigned long long* s_pa,
    unsigned long long (*s_sufA)[TILE_W + 1],
    unsigned long long (*s_sufB)[TILE_W + 1]
) {
    // Flat cooperative fill: TILE_W prefix-AND words + 2 * T * TILE_W
    // suffix words. Consecutive threads touch consecutive words of a row —
    // coalesced global loads.
    const int total = TILE_W + 2 * TILE_T * TILE_W;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        if (idx < TILE_W) {
            const long long w = word_base + idx;
            unsigned long long v = 0ULL;
            if (w < n_u64s) {
                v = ~0ULL;  // empty prefix: identity
                for (int p = 0; p < prefix_len; p++) {
                    v &= bitvecs[(long long)s_pref[p] * n_u64s + w];
                }
            }
            s_pa[idx] = v;
        } else {
            const int rem0 = idx - TILE_W;
            const int which = rem0 / (TILE_T * TILE_W);  // 0 = tile A, 1 = tile B
            const int rem = rem0 % (TILE_T * TILE_W);
            const int row = rem / TILE_W;
            const int wo = rem % TILE_W;
            const long long w = word_base + wo;
            const long long pos = (which ? tb : ta) * TILE_T + row;
            unsigned long long v = 0ULL;
            if (pos < group_size && w < n_u64s) {
                const long long col = (long long)group_suffixes[suf_start + pos];
                v = bitvecs[col * n_u64s + w];
            }
            if (which) s_sufB[row][wo] = v;
            else s_sufA[row][wo] = v;
        }
    }
}

// Core body shared by both entries via a macro: decodes the block's
// (group, tile-pair), stages word tiles, accumulates 4 register-blocked
// pairs per thread, then EMIT_PAIR(cand_global, count) per valid pair.
#define SHARED_TILED_BODY(EMIT_PAIR)                                                              \
    const long long tp_idx = (long long)blockIdx.y * (long long)gridDim.x                         \
                           + (long long)blockIdx.x + tilepair_start;                              \
    if (tp_idx >= tilepair_end) return;                                                           \
                                                                                                  \
    /* group = last g with cumulative_tilepairs[g] <= tp_idx */                                   \
    long long lo = 0, hi = n_groups - 1;                                                          \
    while (lo < hi) {                                                                             \
        const long long mid = (lo + hi + 1) / 2;                                                  \
        if (cumulative_tilepairs[mid] <= tp_idx) lo = mid;                                        \
        else hi = mid - 1;                                                                        \
    }                                                                                             \
    const long long g = lo;                                                                       \
    const long long local_tp = tp_idx - cumulative_tilepairs[g];                                  \
                                                                                                  \
    const long long suf_start = group_suffix_offsets[g];                                          \
    const long long group_size = group_suffix_offsets[g + 1] - suf_start;                         \
    const long long nt = (group_size + TILE_T - 1) / TILE_T;                                      \
                                                                                                  \
    /* decode (ta, tb): sqrt seed + integer correction (never trust FP64     */                   \
    /* exactness at the boundary)                                            */                   \
    long long ta = (long long)((2.0 * (double)nt + 1.0                                            \
        - sqrt((2.0 * (double)nt + 1.0) * (2.0 * (double)nt + 1.0)                                \
               - 8.0 * (double)local_tp)) / 2.0);                                                 \
    if (ta < 0) ta = 0;                                                                           \
    if (ta > nt - 1) ta = nt - 1;                                                                 \
    while (ta > 0 && TP_OFFSET(ta, nt) > local_tp) ta--;                                          \
    while (ta < nt - 1 && TP_OFFSET(ta + 1, nt) <= local_tp) ta++;                                \
    const long long tb = ta + (local_tp - TP_OFFSET(ta, nt));                                     \
                                                                                                  \
    const long long pref_start = group_prefix_offsets[g];                                         \
    const int prefix_len = (int)(group_prefix_offsets[g + 1] - pref_start);                       \
                                                                                                  \
    __shared__ int s_pref[62];                                                                    \
    __shared__ unsigned long long s_pa[TILE_W + 1];                                               \
    __shared__ unsigned long long s_sufA[TILE_T][TILE_W + 1];                                     \
    __shared__ unsigned long long s_sufB[TILE_T][TILE_W + 1];                                     \
    __shared__ int s_skip;                                                                        \
                                                                                                  \
    if (threadIdx.x < prefix_len && threadIdx.x < 62) {                                           \
        s_pref[threadIdx.x] = group_prefix_items[pref_start + threadIdx.x];                       \
    }                                                                                             \
    __syncthreads();                                                                              \
                                                                                                  \
    /* register-blocked pair ownership: b = lane, a = warp + 8*q */                               \
    const int lane = (int)(threadIdx.x & 31);                                                     \
    const int warp = (int)(threadIdx.x >> 5);                                                     \
    unsigned int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;                                          \
                                                                                                  \
    for (long long word_base = 0; word_base < n_u64s; word_base += TILE_W) {                      \
        _stage_tiles(bitvecs, group_suffixes, suf_start, group_size, s_pref,                      \
                     prefix_len, ta, tb, word_base, n_u64s, s_pa, s_sufA, s_sufB);                \
        __syncthreads();                                                                          \
        if (threadIdx.x == 0) {                                                                   \
            unsigned long long any = 0ULL;                                                        \
            for (int w = 0; w < TILE_W; w++) any |= s_pa[w];                                      \
            s_skip = (prefix_len > 0 && any == 0ULL) ? 1 : 0;                                     \
        }                                                                                         \
        __syncthreads();                                                                          \
        if (!s_skip) {                                                                            \
            for (int w = 0; w < TILE_W; w++) {                                                    \
                const unsigned long long pb = s_pa[w] & s_sufB[lane][w];                          \
                if (pb == 0ULL) continue;                                                         \
                acc0 += (unsigned int)__popcll(pb & s_sufA[warp][w]);                             \
                acc1 += (unsigned int)__popcll(pb & s_sufA[warp + 8][w]);                         \
                acc2 += (unsigned int)__popcll(pb & s_sufA[warp + 16][w]);                        \
                acc3 += (unsigned int)__popcll(pb & s_sufA[warp + 24][w]);                        \
            }                                                                                     \
        }                                                                                         \
        __syncthreads();                                                                          \
    }                                                                                             \
                                                                                                  \
    /* emit: pair (i < j) -> candidate cp[g] + j*(j-1)/2 + i (legacy order) */                    \
    const long long j_pos = tb * TILE_T + lane;                                                   \
    if (j_pos < group_size) {                                                                     \
        const unsigned int accs[4] = {acc0, acc1, acc2, acc3};                                    \
        for (int q = 0; q < 4; q++) {                                                             \
            const int a = warp + 8 * q;                                                           \
            const long long i_pos = ta * TILE_T + a;                                              \
            if (i_pos >= group_size) continue;                                                    \
            if (ta == tb && a >= lane) continue; /* diagonal tile: need i < j */                  \
            const long long cand = cumulative_pairs[g]                                            \
                + j_pos * (j_pos - 1) / 2 + i_pos;                                                \
            EMIT_PAIR(cand, accs[q]);                                                             \
        }                                                                                         \
    }

extern "C" __global__
void count_shared_tiled_dense(
    const unsigned long long* __restrict__ bitvecs,
    const int* __restrict__ group_prefix_items,
    const long long* __restrict__ group_prefix_offsets,
    const int* __restrict__ group_suffixes,
    const long long* __restrict__ group_suffix_offsets,
    const long long* __restrict__ cumulative_pairs,
    const long long* __restrict__ cumulative_tilepairs,
    const long long n_u64s,
    const long long n_groups,
    const long long tilepair_start,
    const long long tilepair_end,
    const long long cand_offset,          // chunk-relative output indexing
    int* __restrict__ result_counts       // int32, one slot per chunk candidate
) {
    // Each candidate is written by exactly one block (its tile-pair) — no
    // atomics; bit-identical layout to count_k3plus_dense for the chunk.
    #define EMIT_DENSE(cand, count) result_counts[(cand) - cand_offset] = (int)(count);
    SHARED_TILED_BODY(EMIT_DENSE)
    #undef EMIT_DENSE
}

extern "C" __global__
void count_shared_tiled_fused(
    const unsigned long long* __restrict__ bitvecs,
    const int* __restrict__ group_prefix_items,
    const long long* __restrict__ group_prefix_offsets,
    const int* __restrict__ group_suffixes,
    const long long* __restrict__ group_suffix_offsets,
    const long long* __restrict__ cumulative_pairs,
    const long long* __restrict__ cumulative_tilepairs,
    const long long n_u64s,
    const long long n_groups,
    const long long tilepair_start,
    const long long tilepair_end,
    const int min_count,
    long long* __restrict__ result_indices,  // global candidate ids of survivors
    int* __restrict__ result_counts,
    unsigned long long* __restrict__ n_results,
    const long long capacity                 // counting continues past it (writes dropped)
) {
    #define EMIT_FUSED(cand, count)                                                              \
        if ((int)(count) >= min_count) {                                                         \
            const unsigned long long pos = atomicAdd(n_results, 1ULL);                           \
            if (pos < (unsigned long long)capacity) {                                            \
                result_indices[pos] = (cand);                                                    \
                result_counts[pos] = (int)(count);                                               \
            }                                                                                    \
        }
    SHARED_TILED_BODY(EMIT_FUSED)
    #undef EMIT_FUSED
}
