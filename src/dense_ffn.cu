#include "dense_ffn.cuh"
#include "dense_matvec.cuh"
#include <cuda_fp16.h>

__global__ void silu_elementwise_multiply(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half*       __restrict__ out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __half2float(gate[i]);
        float u = __half2float(up[i]);
        float silu_g = g / (1.0f + expf(-g));
        out[i] = __float2half(silu_g * u);
    }
}

void dense_swiglu_ffn(
    const __half* x,
    const __half* gate_proj,
    const __half* up_proj,
    const __half* down_proj,
    __half*       out,
    __half*       workspace,
    int           hidden_dim,
    int           intermediate_dim,
    cudaStream_t  stream)
{
    __half* gate_buf   = workspace;
    __half* up_buf     = workspace + intermediate_dim;
    __half* silu_up    = workspace + 2 * intermediate_dim;

    // gate = gate_proj @ x
    launch_dense_matvec(gate_proj, x, gate_buf, intermediate_dim, hidden_dim, stream);
    // up = up_proj @ x
    launch_dense_matvec(up_proj, x, up_buf, intermediate_dim, hidden_dim, stream);

    // silu_up = SiLU(gate) * up
    const int elwise_grid = (intermediate_dim + 255) / 256;
    silu_elementwise_multiply<<<elwise_grid, 256, 0, stream>>>(gate_buf, up_buf, silu_up, intermediate_dim);

    // out = down_proj @ silu_up
    launch_dense_matvec(down_proj, silu_up, out, hidden_dim, intermediate_dim, stream);
}
