
// Survivor compaction for dense int32 count arrays.
//
// Grid-stride pass over `counts`; every element >= threshold is appended as
// structure-of-arrays (separate int64 index array + int32 count array — an
// interleaved struct would pad to 16 B/survivor instead of 12 B).
//
// Warp-aggregated atomics: one atomicAdd on the global cursor per warp
// (leader lane reserves __popc(ballot) slots), so the all-survivors worst
// case does not serialize on a single counter.
//
// Two-launch protocol (caller-side): capacity == 0 is a count-only pass —
// the atomic cursor still advances so `n_out` reports the exact survivor
// count, but nothing is written. The caller then allocates exactly n_out
// survivors and re-launches with capacity == n_out. The kernel always keeps
// counting past capacity (writes past it are dropped), so overflow is
// detectable and silent truncation is impossible.
//
// Output order is NON-deterministic (atomic append across warps); the host
// wrapper sorts survivors ascending before returning them.
extern "C" __global__
void compact_threshold(
    const int* __restrict__ counts,
    const long long n,
    const int threshold,
    long long* __restrict__ out_indices,  // unused when capacity == 0
    int* __restrict__ out_counts,         // unused when capacity == 0
    unsigned long long* __restrict__ n_out,
    const long long capacity              // 0 => count-only pass
) {
    const long long stride = (long long)gridDim.x * (long long)blockDim.x;
    for (long long i = (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const int c = counts[i];
        const bool pass = (c >= threshold);
        const unsigned int mask = __ballot_sync(__activemask(), pass);
        if (!pass) continue;

        const int lane = (int)(threadIdx.x & 31);
        const int leader = __ffs(mask) - 1;
        unsigned long long base = 0;
        if (lane == leader) {
            base = atomicAdd(n_out, (unsigned long long)__popc(mask));
        }
        // Only pass-lanes reach here, and `mask` is exactly that set — the
        // shuffle's member mask matches its participants.
        base = __shfl_sync(mask, base, leader);

        if (capacity > 0) {
            const unsigned long long pos =
                base + (unsigned long long)__popc(mask & ((1u << lane) - 1u));
            if (pos < (unsigned long long)capacity) {
                out_indices[pos] = i;
                out_counts[pos] = c;
            }
        }
    }
}
