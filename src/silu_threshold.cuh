#pragma once
#include <cuda_fp16.h>

// Applies SiLU to gate_vals, detects active neurons where |SiLU(x)| > threshold,
// and compacts their indices and values using warp-level ballot/prefix-sum.
//
// Outputs:
//   silu_vals[i]      = SiLU(gate_vals[active_indices[i]])  for i in [0, *active_count)
//   active_indices[i] = original neuron index in [0, intermediate_dim)
//   *active_count     = number of neurons passing threshold
void launch_silu_threshold(
    const __half* gate_vals,      // [intermediate_dim], input gate projection output
    __half*       silu_vals,      // [intermediate_dim], compacted SiLU values at active positions
    int*          active_indices, // [intermediate_dim], compacted active neuron indices
    int*          active_count,   // scalar output: number of active neurons
    int           intermediate_dim,
    float         threshold,
    cudaStream_t  stream = nullptr
);
