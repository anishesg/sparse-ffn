#pragma once
#include <cuda_fp16.h>

// Dense SwiGLU FFN reference.
// Computes: out = down_proj(SiLU(gate_proj(x)) * up_proj(x))
//
// All weight matrices are fp16, row-major.
// gate_proj: [intermediate_dim x hidden_dim]
// up_proj:   [intermediate_dim x hidden_dim]
// down_proj: [hidden_dim x intermediate_dim]
//
// workspace must point to at least 3 * intermediate_dim * sizeof(__half) bytes,
// used for intermediate gate, up, and SiLU*up buffers.
void dense_swiglu_ffn(
    const __half* x,          // [hidden_dim]
    const __half* gate_proj,  // [intermediate_dim x hidden_dim]
    const __half* up_proj,    // [intermediate_dim x hidden_dim]
    const __half* down_proj,  // [hidden_dim x intermediate_dim]
    __half*       out,        // [hidden_dim]
    __half*       workspace,  // [3 * intermediate_dim]
    int           hidden_dim,
    int           intermediate_dim,
    cudaStream_t  stream = nullptr
);
