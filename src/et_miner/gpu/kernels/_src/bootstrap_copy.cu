
            extern "C" __global__
            void bootstrap_copy(
                const long long* __restrict__ src_indptr,
                const long long* __restrict__ src_indices,
                const long long* __restrict__ sampled_rows,
                const long long* __restrict__ dst_indptr,
                long long* __restrict__ dst_indices,
                long long n_rows
            ) {
                long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
                if (row >= n_rows) return;

                // Source row to copy from
                long long src_row = sampled_rows[row];
                long long src_start = src_indptr[src_row];
                long long src_end = src_indptr[src_row + 1];

                // Destination in output
                long long dst_start = dst_indptr[row];

                // Copy indices
                for (long long i = 0; i < src_end - src_start; i++) {
                    dst_indices[dst_start + i] = src_indices[src_start + i];
                }
            }
            