#include "sparse_ffn_batched.cuh"
#include "dense_matvec.cuh"
#include <cuda_fp16.h>
#include <cstring>
#include <vector>

// Workspace layout per token (offsets from token's workspace base):
//   gate_buf    [intermediate_dim * sizeof(__half)]
//   silu_buf    [intermediate_dim * sizeof(__half)]
//   up_sparse   [intermediate_dim * sizeof(__half)]
//   hidden_sp   [intermediate_dim * sizeof(__half)]
//   active_idx  [intermediate_dim * sizeof(int)]    (16-byte aligned after half section)
//   active_cnt  [sizeof(int)]
//   fp32_scratch [hidden_dim * sizeof(float)]       (16-byte aligned)
static const size_t ALIGN = 16;

static size_t align_up(size_t x) { return (x + ALIGN - 1) & ~(ALIGN - 1); }

size_t batched_sparse_ffn_workspace_bytes(int intermediate_dim, int hidden_dim) {
    size_t half_sec = 4ull * intermediate_dim * sizeof(__half);
    size_t int_sec  = align_up(intermediate_dim * sizeof(int) + sizeof(int));
    size_t fp32_sec = align_up((size_t)hidden_dim * sizeof(float));
    return align_up(half_sec) + int_sec + fp32_sec + 64;  // +64 for alignment slack
}

// ---- device helpers ----

__device__ __forceinline__ float silu_b(float x) {
    return x / (1.0f + expf(-x));
}

// Per-token SiLU threshold + compact kernel.
// Grid: (blocks_per_token * batch_size, 1)
// Each batch of blocks_per_token blocks handles one token's intermediate_dim neurons.
__global__ void batched_silu_threshold_kernel(
    const __half* __restrict__ gate_vals,      // [batch_size x intermediate_dim]
    __half*       __restrict__ silu_out,       // [batch_size x intermediate_dim] (compacted)
    int*          __restrict__ active_indices, // [batch_size x intermediate_dim] (compacted)
    int*          __restrict__ active_counts,  // [batch_size]
    int           intermediate_dim,
    int           blocks_per_token,
    float         threshold)
{
    const int token_idx     = blockIdx.x / blocks_per_token;
    const int block_in_tok  = blockIdx.x % blocks_per_token;
    const int local_tid     = block_in_tok * blockDim.x + threadIdx.x;
    const int lane          = threadIdx.x & 31;
    const unsigned mask     = 0xffffffff;

    const __half* g_base  = gate_vals      + (size_t)token_idx * intermediate_dim;
    __half*       s_base  = silu_out       + (size_t)token_idx * intermediate_dim;
    int*          ai_base = active_indices + (size_t)token_idx * intermediate_dim;
    int*          cnt     = active_counts  + token_idx;

    float sv      = 0.0f;
    int is_active = 0;

    if (local_tid < intermediate_dim) {
        sv = silu_b(__half2float(g_base[local_tid]));
        is_active = (fabsf(sv) > threshold) ? 1 : 0;
    }

    unsigned ballot    = __ballot_sync(mask, is_active);
    int warp_count     = __popc(ballot);

    int warp_base = 0;
    if (lane == 0 && warp_count > 0) {
        warp_base = atomicAdd(cnt, warp_count);
    }
    warp_base = __shfl_sync(mask, warp_base, 0);

    if (is_active && local_tid < intermediate_dim) {
        int write_offset = __popc(ballot & ((1u << lane) - 1u));
        int out_idx = warp_base + write_offset;
        s_base[out_idx]  = __float2half(sv);
        ai_base[out_idx] = local_tid;
    }
}

// Per-active-neuron column-sparse up_proj for one token.
// Grid: (active_count, batch_size) -- blockIdx.y = token, blockIdx.x = neuron index
__global__ void batched_col_sparse_kernel(
    const __half* __restrict__ W,              // [intermediate_dim x hidden_dim]
    const __half* __restrict__ x,              // [batch_size x hidden_dim]
    const int*    __restrict__ active_idx_all, // [batch_size x intermediate_dim]
    const int*    __restrict__ active_counts,  // [batch_size]
    __half*       __restrict__ up_sparse,      // [batch_size x intermediate_dim]
    int           intermediate_dim,
    int           hidden_dim,
    int           max_active)
{
    const int token   = blockIdx.y;
    const int n_idx   = blockIdx.x;
    const int n_active = active_counts[token];
    if (n_idx >= n_active) return;

    const int col = active_idx_all[(size_t)token * intermediate_dim + n_idx];

    __shared__ float shmem[256];
    float acc = 0.0f;
    const __half* w_row = W + col * hidden_dim;
    const __half* x_row = x + (size_t)token * hidden_dim;
    for (int k = threadIdx.x; k < hidden_dim; k += 256) {
        acc += __half2float(w_row[k]) * __half2float(x_row[k]);
    }
    shmem[threadIdx.x] = acc;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) shmem[threadIdx.x] += shmem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        up_sparse[(size_t)token * intermediate_dim + n_idx] = __float2half(shmem[0]);
}

// Elementwise multiply silu_buf * up_sparse -> hidden_sp (compacted, per token)
__global__ void batched_elementwise_mul(
    const __half* __restrict__ silu_buf,    // [batch_size x intermediate_dim]
    const __half* __restrict__ up_sparse,   // [batch_size x intermediate_dim]
    __half*       __restrict__ hidden_sp,   // [batch_size x intermediate_dim]
    const int*    __restrict__ active_counts,
    int           intermediate_dim)
{
    const int token = blockIdx.y;
    const int i     = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= active_counts[token]) return;
    size_t base = (size_t)token * intermediate_dim;
    float v = __half2float(silu_buf[base + i]) * __half2float(up_sparse[base + i]);
    hidden_sp[base + i] = __float2half(v);
}

// Row-sparse down_proj accumulate into fp32 scratch, per token.
// Grid: (max_active, batch_size); blockIdx.x = neuron_idx within token, blockIdx.y = token
__global__ void batched_row_sparse_kernel(
    const __half* __restrict__ W,              // [hidden_dim x intermediate_dim]  (down_proj transposed layout)
    const __half* __restrict__ hidden_sp,      // [batch_size x intermediate_dim]
    const int*    __restrict__ active_idx_all, // [batch_size x intermediate_dim]
    const int*    __restrict__ active_counts,  // [batch_size]
    float*        __restrict__ y_fp32,         // [batch_size x hidden_dim]
    int           intermediate_dim,
    int           hidden_dim)
{
    const int token   = blockIdx.y;
    const int n_idx   = blockIdx.x;
    if (n_idx >= active_counts[token]) return;

    const int row = active_idx_all[(size_t)token * intermediate_dim + n_idx];
    const float scale = __half2float(hidden_sp[(size_t)token * intermediate_dim + n_idx]);
    const __half* w_row = W + row * hidden_dim;
    float* y_row = y_fp32 + (size_t)token * hidden_dim;

    for (int k = threadIdx.x; k < hidden_dim; k += 256) {
        atomicAdd(&y_row[k], scale * __half2float(w_row[k]));
    }
}

__global__ void batched_fp32_to_fp16(
    const float* __restrict__ src,  // [batch_size x hidden_dim]
    __half*      __restrict__ dst,  // [batch_size x hidden_dim]
    int batch_size, int hidden_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * hidden_dim;
    if (i < total) dst[i] = __float2half(src[i]);
}

// SiLU elementwise multiply kernel for dense batched
__global__ void batched_silu_elwise(
    const __half* __restrict__ gate,   // [batch_size x inter]
    const __half* __restrict__ up,     // [batch_size x inter]
    __half*       __restrict__ out,    // [batch_size x inter]
    int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    out[i] = __float2half((g / (1.0f + expf(-g))) * u);
}

// ---- host functions ----

void batched_sparse_swiglu(
    const __half* x,
    const __half* gate_proj,
    const __half* up_proj,
    const __half* down_proj,
    __half*       out,
    void*         workspace,
    int           batch_size,
    int           hidden_dim,
    int           intermediate_dim,
    float         threshold,
    cudaStream_t  stream)
{
    // Workspace layout (flat, contiguous):
    //   gate_buf_all   [batch_size x intermediate_dim]  __half
    //   silu_buf_all   [batch_size x intermediate_dim]  __half
    //   up_sparse_all  [batch_size x intermediate_dim]  __half
    //   hidden_sp_all  [batch_size x intermediate_dim]  __half
    //   active_idx_all [batch_size x intermediate_dim]  int32
    //   active_cnt_all [batch_size]                      int32
    //   fp32_scratch   [batch_size x hidden_dim]         float

    char* buf = reinterpret_cast<char*>(workspace);
    size_t inter = (size_t)batch_size * intermediate_dim;

    __half* gate_buf_all  = reinterpret_cast<__half*>(buf);
    __half* silu_buf_all  = gate_buf_all  + inter;
    __half* up_sparse_all = silu_buf_all  + inter;
    __half* hidden_sp_all = up_sparse_all + inter;

    size_t half_end = 4ull * inter * sizeof(__half);
    size_t int_off  = align_up(half_end);
    int* active_idx_all = reinterpret_cast<int*>(buf + int_off);
    int* active_cnt_all = active_idx_all + inter;

    size_t int_end  = int_off + align_up(inter * sizeof(int) + (size_t)batch_size * sizeof(int));
    float* fp32_all = reinterpret_cast<float*>(buf + int_end);

    // Zero active counts and fp32 scratch
    cudaMemsetAsync(active_cnt_all, 0, batch_size * sizeof(int), stream);
    cudaMemsetAsync(fp32_all, 0, (size_t)batch_size * hidden_dim * sizeof(float), stream);

    // Step 1: gate = gate_proj @ x[i] for all tokens (reuse dense matvec per token)
    // Use batched launch: one block-row per output neuron per token.
    // For simplicity call launch_dense_matvec in a loop; later can fuse.
    for (int t = 0; t < batch_size; ++t) {
        launch_dense_matvec(
            gate_proj,
            x + (size_t)t * hidden_dim,
            gate_buf_all + (size_t)t * intermediate_dim,
            intermediate_dim, hidden_dim, stream);
    }

    // Step 2: fused SiLU + compact all tokens in one kernel launch
    constexpr int BLOCK = 256;
    int blocks_per_token = (intermediate_dim + BLOCK - 1) / BLOCK;
    batched_silu_threshold_kernel<<<blocks_per_token * batch_size, BLOCK, 0, stream>>>(
        gate_buf_all, silu_buf_all, active_idx_all, active_cnt_all,
        intermediate_dim, blocks_per_token, threshold);

    // Copy active counts to host to determine grid dims for sparse kernels
    std::vector<int> h_counts(batch_size);
    cudaMemcpyAsync(h_counts.data(), active_cnt_all, batch_size * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int max_active = 0;
    for (int c : h_counts) if (c > max_active) max_active = c;

    if (max_active == 0) {
        cudaMemsetAsync(out, 0, (size_t)batch_size * hidden_dim * sizeof(__half), stream);
        return;
    }

    // Step 3: column-sparse up_proj for all tokens
    dim3 up_grid(max_active, batch_size);
    batched_col_sparse_kernel<<<up_grid, 256, 0, stream>>>(
        up_proj, x, active_idx_all, active_cnt_all,
        up_sparse_all, intermediate_dim, hidden_dim, max_active);

    // Step 4: elementwise multiply silu * up_sparse
    dim3 mul_grid((max_active + 255) / 256, batch_size);
    batched_elementwise_mul<<<mul_grid, 256, 0, stream>>>(
        silu_buf_all, up_sparse_all, hidden_sp_all, active_cnt_all, intermediate_dim);

    // Step 5: row-sparse down_proj
    dim3 down_grid(max_active, batch_size);
    batched_row_sparse_kernel<<<down_grid, 256, 0, stream>>>(
        down_proj, hidden_sp_all, active_idx_all, active_cnt_all,
        fp32_all, intermediate_dim, hidden_dim);

    // Convert fp32 -> fp16
    int total_elems = batch_size * hidden_dim;
    batched_fp32_to_fp16<<<(total_elems + 255) / 256, 256, 0, stream>>>(
        fp32_all, out, batch_size, hidden_dim);
}

void batched_dense_swiglu(
    const __half* x,
    const __half* gate_proj,
    const __half* up_proj,
    const __half* down_proj,
    __half*       out,
    __half*       workspace,
    int           batch_size,
    int           hidden_dim,
    int           intermediate_dim,
    cudaStream_t  stream)
{
    __half* gate_buf = workspace;
    __half* up_buf   = workspace + (size_t)batch_size * intermediate_dim;
    __half* silu_up  = workspace + (size_t)batch_size * intermediate_dim * 2;

    for (int t = 0; t < batch_size; ++t) {
        launch_dense_matvec(gate_proj,
                            x + (size_t)t * hidden_dim,
                            gate_buf + (size_t)t * intermediate_dim,
                            intermediate_dim, hidden_dim, stream);
        launch_dense_matvec(up_proj,
                            x + (size_t)t * hidden_dim,
                            up_buf + (size_t)t * intermediate_dim,
                            intermediate_dim, hidden_dim, stream);
    }

    int total_inter = batch_size * intermediate_dim;
    batched_silu_elwise<<<(total_inter + 255) / 256, 256, 0, stream>>>(
        gate_buf, up_buf, silu_up, total_inter);

    for (int t = 0; t < batch_size; ++t) {
        launch_dense_matvec(down_proj,
                            silu_up + (size_t)t * intermediate_dim,
                            out + (size_t)t * hidden_dim,
                            hidden_dim, intermediate_dim, stream);
    }
}
