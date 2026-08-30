#pragma once
#include <cuda_fp16.h>

// Column-sparse matvec for up_proj.
// Gathers the active_count columns of W indexed by active_cols,
// and computes y[i] = dot(W[:, active_cols[i]], x) for i in [0, active_count).
//
// W:           [rows x hidden_dim], row-major, fp16  (up_proj weights)
// x:           [hidden_dim], fp16
// active_cols: [active_count], column indices to compute
// y:           [active_count], compacted output (up_proj values at active neurons)
void launch_col_sparse_matvec(
    const __half* W,            // [rows x hidden_dim]
    const __half* x,            // [hidden_dim]
    const int*    active_cols,  // [active_count]
    __half*       y,            // [active_count]
    int           rows,
    int           hidden_dim,
    int           active_count,
    cudaStream_t  stream = nullptr
);

// Row-sparse matvec for down_proj.
// Gathers the active_count rows of W indexed by active_rows,
// multiplies each by the corresponding sparse_x entry,
// and atomically accumulates into y (dense, length hidden_dim).
//
// W:           [hidden_dim x intermediate_dim], row-major, fp16  (down_proj weights)
// sparse_x:    [active_count], values at active neuron positions
// active_rows: [active_count], row indices in [0, intermediate_dim)
// y:           [hidden_dim], output accumulation buffer (must be zeroed before call)
void launch_row_sparse_matvec(
    const __half* W,            // [hidden_dim x intermediate_dim]
    const __half* sparse_x,     // [active_count]
    const int*    active_rows,  // [active_count]
    __half*       y,            // [hidden_dim]
    int           hidden_dim,
    int           intermediate_dim,
    int           active_count,
    cudaStream_t  stream = nullptr
);
