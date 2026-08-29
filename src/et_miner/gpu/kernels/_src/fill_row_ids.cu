
            extern "C" __global__
            void fill_row_ids(
                const long long* __restrict__ indptr,
                long long* __restrict__ row_ids,
                long long n_rows
            ) {
                long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
                if (row >= n_rows) return;

                long long start = indptr[row];
                long long end = indptr[row + 1];

                for (long long i = start; i < end; i++) {
                    row_ids[i] = row;
                }
            }
            