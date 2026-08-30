#include "silu_threshold.cuh"
#include <cuda_fp16.h>

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Each warp processes 32 consecutive neurons.
// __ballot_sync produces a 32-bit mask of active lanes.
// __popc prefix sum computes write offsets into the compact output arrays.
// A global atomic on active_count arbitrates the warp's base write offset.
__global__ void silu_threshold_kernel(
    const __half* __restrict__ gate_vals,
    __half*       __restrict__ silu_vals,
    int*          __restrict__ active_indices,
    int*          __restrict__ active_count,
    int           intermediate_dim,
    float         threshold)
{
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const unsigned mask = 0xffffffff;

    float sv = 0.0f;
    int   is_active = 0;

    if (tid < intermediate_dim) {
        sv = silu(__half2float(gate_vals[tid]));
        is_active = (fabsf(sv) > threshold) ? 1 : 0;
    }

    // Warp-level ballot: bitmask of which lanes are active
    unsigned ballot = __ballot_sync(mask, is_active);
    int warp_count  = __popc(ballot);

    // Lane 0 of each warp claims a contiguous block in the output
    int warp_base = 0;
    if (lane == 0 && warp_count > 0) {
        warp_base = atomicAdd(active_count, warp_count);
    }
    warp_base = __shfl_sync(mask, warp_base, 0);

    if (is_active && tid < intermediate_dim) {
        // Compute write offset: count of active lanes with lane index < this lane
        int write_offset = __popc(ballot & ((1u << lane) - 1u));
        int out_idx = warp_base + write_offset;
        silu_vals[out_idx]      = __float2half(sv);
        active_indices[out_idx] = tid;
    }
}

void launch_silu_threshold(
    const __half* gate_vals,
    __half*       silu_vals,
    int*          active_indices,
    int*          active_count,
    int           intermediate_dim,
    float         threshold,
    cudaStream_t  stream)
{
    // Zero the counter before launch
    cudaMemsetAsync(active_count, 0, sizeof(int), stream);

    constexpr int BLOCK = 256;
    const int grid = (intermediate_dim + BLOCK - 1) / BLOCK;
    silu_threshold_kernel<<<grid, BLOCK, 0, stream>>>(
        gate_vals, silu_vals, active_indices, active_count,
        intermediate_dim, threshold);
}
