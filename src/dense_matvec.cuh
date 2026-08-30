#pragma once
#include <cuda_fp16.h>

// Dense tiled matvec: y = W * x
// W: [rows x cols] in row-major, fp16
// x: [cols], fp16
// y: [rows], fp16
// Accumulation in fp32, output written as fp16
void launch_dense_matvec(
    const __half* W,  // [rows, cols]
    const __half* x,  // [cols]
    __half*       y,  // [rows]
    int rows,
    int cols,
    cudaStream_t stream = nullptr
);
