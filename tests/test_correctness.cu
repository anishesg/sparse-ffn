#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>
#include "dense_ffn.cuh"
#include "sparse_ffn.cuh"

static void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}
#define CUDA_CHECK(x) cuda_check((x), #x)

// Simple LCG for reproducible random fp16 in [-1, 1]
static float lcg_rand(unsigned& state) {
    state = state * 1664525u + 1013904223u;
    return ((float)(state >> 16) / 32768.0f) - 1.0f;
}

static void fill_random_half(__half* d_buf, int n, unsigned& seed) {
    std::vector<__half> h(n);
    for (int i = 0; i < n; ++i) h[i] = __float2half(lcg_rand(seed) * 0.1f);
    CUDA_CHECK(cudaMemcpy(d_buf, h.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
}

struct TestConfig {
    int   hidden_dim;
    int   intermediate_dim;
    float threshold;
};

int main() {
    TestConfig configs[] = {
        {4096, 11008, 0.001f},
        {4096, 11008, 0.01f},
        {4096, 11008, 0.1f},
        {8192, 28672, 0.001f},
        {8192, 28672, 0.01f},
        {4096, 14336, 0.01f},
    };
    const int num_configs = sizeof(configs) / sizeof(configs[0]);

    int failures = 0;

    for (int ci = 0; ci < num_configs; ++ci) {
        const int H = configs[ci].hidden_dim;
        const int I = configs[ci].intermediate_dim;
        const float thr = configs[ci].threshold;

        unsigned seed = 0x1234ABCD + ci * 31337;

        // Allocate weight matrices
        __half *d_gp, *d_up, *d_dp, *d_x;
        CUDA_CHECK(cudaMalloc(&d_gp, (size_t)I * H * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_up, (size_t)I * H * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_dp, (size_t)H * I * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_x,  H * sizeof(__half)));

        fill_random_half(d_gp, I * H, seed);
        fill_random_half(d_up, I * H, seed);
        fill_random_half(d_dp, H * I, seed);
        fill_random_half(d_x,  H,     seed);

        // Dense reference output
        __half *d_out_dense, *d_ws_dense;
        CUDA_CHECK(cudaMalloc(&d_out_dense, H * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_ws_dense,  3 * I * sizeof(__half)));

        dense_swiglu_ffn(d_x, d_gp, d_up, d_dp, d_out_dense, d_ws_dense, H, I);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Sparse output
        __half* d_out_sparse;
        CUDA_CHECK(cudaMalloc(&d_out_sparse, H * sizeof(__half)));
        size_t ws_bytes = sparse_ffn_workspace_bytes(I, H);
        void* d_ws_sparse;
        CUDA_CHECK(cudaMalloc(&d_ws_sparse, ws_bytes));

        int active_count = 0;
        sparse_swiglu_ffn(d_x, d_gp, d_up, d_dp, d_out_sparse, d_ws_sparse,
                          H, I, thr, &active_count);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy outputs to host
        std::vector<__half> h_dense(H), h_sparse(H);
        CUDA_CHECK(cudaMemcpy(h_dense.data(),  d_out_dense,  H * sizeof(__half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sparse.data(), d_out_sparse, H * sizeof(__half), cudaMemcpyDeviceToHost));

        // Compute metrics
        double dot_ds = 0, norm_d = 0, norm_s = 0;
        double sum_abs_err = 0, sum_abs_ref = 0;
        float  max_abs_err = 0;

        for (int i = 0; i < H; ++i) {
            float d = __half2float(h_dense[i]);
            float s = __half2float(h_sparse[i]);
            dot_ds  += d * s;
            norm_d  += d * d;
            norm_s  += s * s;
            float err = fabsf(d - s);
            if (err > max_abs_err) max_abs_err = err;
            sum_abs_err += err;
            sum_abs_ref += fabsf(d);
        }

        double cos_sim = dot_ds / (sqrt(norm_d) * sqrt(norm_s) + 1e-12);
        double rel_mae = sum_abs_err / (sum_abs_ref + 1e-12);
        float sparsity = 1.0f - (float)active_count / I;

        bool pass = (cos_sim > 0.995);
        printf("[%s] H=%d I=%d thr=%.3f | cos_sim=%.6f max_abs_err=%.5f rel_mae=%.5f sparsity=%.3f active=%d\n",
               pass ? "PASS" : "FAIL",
               H, I, thr,
               cos_sim, max_abs_err, rel_mae, sparsity, active_count);

        if (!pass) ++failures;

        CUDA_CHECK(cudaFree(d_gp));
        CUDA_CHECK(cudaFree(d_up));
        CUDA_CHECK(cudaFree(d_dp));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_out_dense));
        CUDA_CHECK(cudaFree(d_ws_dense));
        CUDA_CHECK(cudaFree(d_out_sparse));
        CUDA_CHECK(cudaFree(d_ws_sparse));
    }

    printf("\n%d/%d configs passed.\n", num_configs - failures, num_configs);
    return failures > 0 ? 1 : 0;
}
