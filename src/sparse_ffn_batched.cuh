#pragma once
#include <cuda_fp16.h>

// Per-token workspace size for batched sparse SwiGLU.
// Total workspace for batch_size tokens: batch_size * batched_sparse_ffn_workspace_bytes(...)
size_t batched_sparse_ffn_workspace_bytes(int intermediate_dim, int hidden_dim);

// Batched sparse SwiGLU: 2D input [batch_size x hidden_dim].
// Each token gets an independent SiLU threshold scan and sparse matvec.
// workspace must be >= batch_size * batched_sparse_ffn_workspace_bytes() bytes.
void batched_sparse_swiglu(
    const __half* x,           // [batch_size x hidden_dim]
    const __half* gate_proj,   // [intermediate_dim x hidden_dim]
    const __half* up_proj,     // [intermediate_dim x hidden_dim]
    const __half* down_proj,   // [hidden_dim x intermediate_dim]
    __half*       out,         // [batch_size x hidden_dim]
    void*         workspace,
    int           batch_size,
    int           hidden_dim,
    int           intermediate_dim,
    float         threshold,
    cudaStream_t  stream = nullptr
);

// Batched dense SwiGLU: 2D input [batch_size x hidden_dim].
// workspace must be >= batch_size * 3 * intermediate_dim * sizeof(__half) bytes.
void batched_dense_swiglu(
    const __half* x,           // [batch_size x hidden_dim]
    const __half* gate_proj,   // [intermediate_dim x hidden_dim]
    const __half* up_proj,     // [intermediate_dim x hidden_dim]
    const __half* down_proj,   // [hidden_dim x intermediate_dim]
    __half*       out,         // [batch_size x hidden_dim]
    __half*       workspace,   // [batch_size x 3 x intermediate_dim]
    int           batch_size,
    int           hidden_dim,
    int           intermediate_dim,
    cudaStream_t  stream = nullptr
);
