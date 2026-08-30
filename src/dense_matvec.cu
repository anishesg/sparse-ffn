#include "dense_matvec.cuh"
#include <cuda_fp16.h>

// Each block handles BLOCK_ROWS output rows.
// Each thread handles one output row.
// Weight rows are loaded in TILE_K-wide chunks through shared memory.
template <int TILE_K, int BLOCK_ROWS>
__global__ void dense_matvec_kernel(
    const __half* __restrict__ W,
    const __half* __restrict__ x,
    __half*       __restrict__ y,
    int rows,
    int cols)
{
    __shared__ __half x_tile[TILE_K];

    const int row = blockIdx.x * BLOCK_ROWS + threadIdx.x;
    float acc = 0.0f;

    for (int tile_start = 0; tile_start < cols; tile_start += TILE_K) {
        // Cooperatively load x tile into shared memory
        if (threadIdx.x < TILE_K) {
            int idx = tile_start + threadIdx.x;
            x_tile[threadIdx.x] = (idx < cols) ? x[idx] : __float2half(0.0f);
        }
        __syncthreads();

        if (row < rows) {
            const int tile_end = min(tile_start + TILE_K, cols);
            const __half* w_row = W + row * cols + tile_start;
            #pragma unroll 8
            for (int k = 0; k < tile_end - tile_start; ++k) {
                acc += __half2float(w_row[k]) * __half2float(x_tile[k]);
            }
        }
        __syncthreads();
    }

    if (row < rows) {
        y[row] = __float2half(acc);
    }
}

void launch_dense_matvec(
    const __half* W,
    const __half* x,
    __half*       y,
    int rows,
    int cols,
    cudaStream_t stream)
{
    constexpr int TILE_K    = 64;
    constexpr int BLOCK_ROWS = 64;
    const int grid = (rows + BLOCK_ROWS - 1) / BLOCK_ROWS;
    dense_matvec_kernel<TILE_K, BLOCK_ROWS><<<grid, BLOCK_ROWS, 0, stream>>>(W, x, y, rows, cols);
}
