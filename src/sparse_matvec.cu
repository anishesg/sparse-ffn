#include "sparse_matvec.cuh"
#include <cuda_fp16.h>

// Column-sparse up_proj kernel.
// One block per output element (active neuron). Each thread block cooperatively
// computes the dot product of x with the column W[:, active_cols[block_idx]].
// Note: up_proj is stored W[intermediate, hidden], so column j is W[j, :].
// Actually W[rows=intermediate, cols=hidden]: row j of W is row j of up_proj.
// We want W[active_col, :] . x, which is just a regular row-dot product.
template <int TILE_K, int BLOCK_SIZE>
__global__ void col_sparse_matvec_kernel(
    const __half* __restrict__ W,
    const __half* __restrict__ x,
    const int*    __restrict__ active_cols,
    __half*       __restrict__ y,
    int rows,
    int hidden_dim,
    int active_count)
{
    // Each block handles one output neuron (one row of up_proj at active index)
    const int out_idx = blockIdx.x;
    if (out_idx >= active_count) return;

    const int col = active_cols[out_idx];
    if (col >= rows) return;

    __shared__ float shmem[BLOCK_SIZE];
    float acc = 0.0f;

    const __half* w_row = W + col * hidden_dim;

    for (int k = threadIdx.x; k < hidden_dim; k += BLOCK_SIZE) {
        acc += __half2float(w_row[k]) * __half2float(x[k]);
    }

    // Block-level reduction
    shmem[threadIdx.x] = acc;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shmem[threadIdx.x] += shmem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        y[out_idx] = __float2half(shmem[0]);
    }
}

void launch_col_sparse_matvec(
    const __half* W,
    const __half* x,
    const int*    active_cols,
    __half*       y,
    int           rows,
    int           hidden_dim,
    int           active_count,
    cudaStream_t  stream)
{
    if (active_count == 0) return;
    constexpr int BLOCK_SIZE = 256;
    constexpr int TILE_K     = 256;
    col_sparse_matvec_kernel<TILE_K, BLOCK_SIZE>
        <<<active_count, BLOCK_SIZE, 0, stream>>>(W, x, active_cols, y, rows, hidden_dim, active_count);
}

// Row-sparse down_proj kernel.
// Each block handles one active neuron (one row of down_proj).
// The block multiplies W[active_rows[block], :] by sparse_x[block]
// and accumulates into y[0..hidden_dim) using tiled non-atomic writes
// followed by an atomic add at the end.
// To avoid per-element atomic, each thread computes its portion and atomically
// adds one float to y, which is competitive at low active_count.
template <int BLOCK_SIZE>
__global__ void row_sparse_matvec_kernel(
    const __half* __restrict__ W,
    const __half* __restrict__ sparse_x,
    const int*    __restrict__ active_rows,
    float*        __restrict__ y_fp32,
    int hidden_dim,
    int intermediate_dim,
    int active_count)
{
    const int neuron = blockIdx.x;
    if (neuron >= active_count) return;

    const int row = active_rows[neuron];
    if (row >= intermediate_dim) return;

    const float scale = __half2float(sparse_x[neuron]);
    const __half* w_row = W + row * hidden_dim;

    for (int k = threadIdx.x; k < hidden_dim; k += BLOCK_SIZE) {
        float contrib = scale * __half2float(w_row[k]);
        atomicAdd(&y_fp32[k], contrib);
    }
}

__global__ void fp32_to_fp16(const float* __restrict__ src, __half* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// We need a persistent fp32 accumulation buffer; caller provides y as fp16.
// We manage a temporary fp32 buffer internally via cudaMallocAsync if available,
// but for simplicity we just accept a pre-zeroed fp32 scratch via launch parameters.
// Expose a simpler interface that takes fp32 scratch.
void launch_row_sparse_matvec(
    const __half* W,
    const __half* sparse_x,
    const int*    active_rows,
    __half*       y,
    int           hidden_dim,
    int           intermediate_dim,
    int           active_count,
    cudaStream_t  stream)
{
    if (active_count == 0) return;

    // Allocate fp32 accumulation buffer
    float* y_fp32 = nullptr;
    cudaMallocAsync(&y_fp32, hidden_dim * sizeof(float), stream);
    cudaMemsetAsync(y_fp32, 0, hidden_dim * sizeof(float), stream);

    constexpr int BLOCK_SIZE = 256;
    row_sparse_matvec_kernel<BLOCK_SIZE>
        <<<active_count, BLOCK_SIZE, 0, stream>>>(
            W, sparse_x, active_rows, y_fp32, hidden_dim, intermediate_dim, active_count);

    const int cvt_grid = (hidden_dim + 255) / 256;
    fp32_to_fp16<<<cvt_grid, 256, 0, stream>>>(y_fp32, y, hidden_dim);

    cudaFreeAsync(y_fp32, stream);
}
