#include "sparse_ffn.cuh"
#include "dense_matvec.cuh"
#include "silu_threshold.cuh"
#include "sparse_matvec.cuh"
#include <cuda_fp16.h>
#include <cstring>

// Workspace layout (all fp16 unless noted, 128-byte aligned offsets):
//   [0]                         gate_buf    [intermediate_dim * sizeof(__half)]
//   [intermediate_dim]          silu_buf    [intermediate_dim * sizeof(__half)]
//   [2*intermediate_dim]        up_sparse   [intermediate_dim * sizeof(__half)]
//   [3*intermediate_dim]        hidden_sp   [intermediate_dim * sizeof(__half)]
//   -- int32 section --
//   [4*intermediate_dim * 2B]   active_idx  [intermediate_dim * sizeof(int)]
//   [4*intermediate_dim * 2B + intermediate_dim * 4B]
//                               active_cnt  [sizeof(int)]
//   -- fp32 scratch --
//   after above                 fp32_scratch [hidden_dim * sizeof(float)]

size_t sparse_ffn_workspace_bytes(int intermediate_dim, int hidden_dim) {
    size_t half_bytes = 4ull * intermediate_dim * sizeof(__half);
    size_t int_bytes  = intermediate_dim * sizeof(int) + sizeof(int);
    size_t fp32_bytes = (size_t)hidden_dim * sizeof(float);
    // Add alignment padding
    return half_bytes + int_bytes + fp32_bytes + 256;
}

// Elementwise multiply of two compact fp16 vectors.
__global__ void elementwise_mul_fp16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half*       __restrict__ c,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = __float2half(__half2float(a[i]) * __half2float(b[i]));
    }
}

void sparse_swiglu_ffn(
    const __half* x,
    const __half* gate_proj,
    const __half* up_proj,
    const __half* down_proj,
    __half*       out,
    void*         workspace,
    int           hidden_dim,
    int           intermediate_dim,
    float         threshold,
    int*          active_count_out,
    cudaStream_t  stream)
{
    char* buf = reinterpret_cast<char*>(workspace);

    // Partition workspace
    __half* gate_buf   = reinterpret_cast<__half*>(buf);
    __half* silu_buf   = gate_buf   + intermediate_dim;
    __half* up_sparse  = silu_buf   + intermediate_dim;
    __half* hidden_sp  = up_sparse  + intermediate_dim;

    // Align int section to 16 bytes
    size_t half_section = 4ull * intermediate_dim * sizeof(__half);
    char* int_start = buf + half_section;
    // Align to 16 bytes
    uintptr_t addr = reinterpret_cast<uintptr_t>(int_start);
    addr = (addr + 15) & ~15ull;
    int* active_idx = reinterpret_cast<int*>(addr);
    int* active_cnt = active_idx + intermediate_dim;

    // fp32 scratch after int section
    char* fp32_start = reinterpret_cast<char*>(active_cnt + 1);
    addr = reinterpret_cast<uintptr_t>(fp32_start);
    addr = (addr + 15) & ~15ull;
    float* fp32_scratch = reinterpret_cast<float*>(addr);

    // Step 1: gate = gate_proj @ x
    launch_dense_matvec(gate_proj, x, gate_buf, intermediate_dim, hidden_dim, stream);

    // Step 2: fused SiLU + compact active neurons
    launch_silu_threshold(gate_buf, silu_buf, active_idx, active_cnt,
                          intermediate_dim, threshold, stream);

    // Retrieve active_count to host so we can conditionally launch step 3-5.
    // Use a pinned-memory copy to avoid synchronizing the entire device.
    int h_active_count = 0;
    cudaMemcpyAsync(&h_active_count, active_cnt, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (active_count_out) *active_count_out = h_active_count;

    if (h_active_count == 0) {
        cudaMemsetAsync(out, 0, hidden_dim * sizeof(__half), stream);
        return;
    }

    // Step 3: column-sparse up_proj
    // up_proj[intermediate_dim x hidden_dim]: row active_idx[i] gives up result for neuron i
    launch_col_sparse_matvec(up_proj, x, active_idx, up_sparse,
                             intermediate_dim, hidden_dim, h_active_count, stream);

    // Step 4: elementwise multiply silu_buf[i] * up_sparse[i]
    const int mul_grid = (h_active_count + 255) / 256;
    elementwise_mul_fp16<<<mul_grid, 256, 0, stream>>>(silu_buf, up_sparse, hidden_sp, h_active_count);

    // Step 5: row-sparse down_proj accumulates into out
    // down_proj[hidden_dim x intermediate_dim]: we need rows at active_idx positions
    launch_row_sparse_matvec_with_scratch(down_proj, hidden_sp, active_idx, out,
                                          fp32_scratch, hidden_dim, intermediate_dim,
                                          h_active_count, stream);
}
