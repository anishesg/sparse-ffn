#pragma once
#include <cuda_fp16.h>

// Sparse SwiGLU FFN with activation-gated neuron pruning.
// Skips up_proj and down_proj computation for neurons where |SiLU(gate)| <= threshold.
//
// Kernel pipeline:
//   1. Dense gate_proj matvec:       gate = gate_proj @ x
//   2. Fused SiLU + compaction:      compact active indices, SiLU values
//   3. Column-sparse up_proj:        up_sparse = up_proj[:, active] @ x
//   4. Elementwise multiply:         hidden_sparse = SiLU(gate)[active] * up_sparse
//   5. Row-sparse down_proj:         out += down_proj[active, :].T @ hidden_sparse
//
// workspace must point to at least workspace_bytes(intermediate_dim, hidden_dim) bytes.
size_t sparse_ffn_workspace_bytes(int intermediate_dim, int hidden_dim);

void sparse_swiglu_ffn(
    const __half* x,          // [hidden_dim]
    const __half* gate_proj,  // [intermediate_dim x hidden_dim]
    const __half* up_proj,    // [intermediate_dim x hidden_dim]
    const __half* down_proj,  // [hidden_dim x intermediate_dim]
    __half*       out,        // [hidden_dim]
    void*         workspace,  // >= sparse_ffn_workspace_bytes bytes
    int           hidden_dim,
    int           intermediate_dim,
    float         threshold,
    int*          active_count_out,  // optional: receives number of active neurons
    cudaStream_t  stream = nullptr
);
