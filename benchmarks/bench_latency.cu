#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_fp16.h>
#include "dense_matvec.cuh"
#include "silu_threshold.cuh"
#include "sparse_matvec.cuh"
#include "dense_ffn.cuh"
#include "sparse_ffn.cuh"

static void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}
#define CUDA_CHECK(x) cuda_check((x), #x)

static float lcg_rand(unsigned& s) {
    s = s * 1664525u + 1013904223u;
    return ((float)(s >> 16) / 32768.0f) - 1.0f;
}

static void fill_random(__half* d, int n, unsigned& seed, float scale = 0.1f) {
    std::vector<__half> h(n);
    for (auto& v : h) v = __float2half(lcg_rand(seed) * scale);
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
}

// Time a lambda over WARMUP+REPS iterations, return median ms.
template <typename F>
static float time_kernel_ms(F fn, int warmup = 5, int reps = 20, cudaStream_t stream = nullptr) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times;
    times.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        fn();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Return median
    std::sort(times.begin(), times.end());
    return times[reps / 2];
}

struct BenchConfig {
    const char* name;
    int hidden_dim;
    int intermediate_dim;
};

int main() {
    BenchConfig configs[] = {
        {"Llama-2 7B",  4096,  11008},
        {"Llama-2 70B", 8192,  28672},
        {"Mixtral 8x7B",4096,  14336},
    };
    float thresholds[] = {0.001f, 0.01f, 0.1f};
    const int num_cfgs  = 3;
    const int num_thrs  = 3;

    for (int ci = 0; ci < num_cfgs; ++ci) {
        const int H = configs[ci].hidden_dim;
        const int I = configs[ci].intermediate_dim;

        printf("\n=== %s (hidden=%d intermediate=%d) ===\n", configs[ci].name, H, I);

        unsigned seed = 0xDEADBEEF + ci * 777;

        __half *d_gp, *d_up, *d_dp, *d_x;
        CUDA_CHECK(cudaMalloc(&d_gp, (size_t)I * H * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_up, (size_t)I * H * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_dp, (size_t)H * I * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_x,  H * sizeof(__half)));
        fill_random(d_gp, I * H, seed);
        fill_random(d_up, I * H, seed);
        fill_random(d_dp, H * I, seed);
        fill_random(d_x,  H,     seed);

        // Dense end-to-end timing
        __half *d_out_dense, *d_ws_dense;
        CUDA_CHECK(cudaMalloc(&d_out_dense, H * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_ws_dense,  3 * I * sizeof(__half)));

        float t_dense = time_kernel_ms([&](){
            dense_swiglu_ffn(d_x, d_gp, d_up, d_dp, d_out_dense, d_ws_dense, H, I);
        });

        // Dense per-kernel timing
        __half *d_gate, *d_up_buf, *d_silu_up, *d_out2;
        CUDA_CHECK(cudaMalloc(&d_gate,    I * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_up_buf,  I * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_silu_up, I * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_out2,    H * sizeof(__half)));

        float t_gate_dense = time_kernel_ms([&](){
            launch_dense_matvec(d_gp, d_x, d_gate, I, H);
        });
        float t_up_dense = time_kernel_ms([&](){
            launch_dense_matvec(d_up, d_x, d_up_buf, I, H);
        });
        float t_down_dense = time_kernel_ms([&](){
            launch_dense_matvec(d_dp, d_silu_up, d_out2, H, I);
        });

        printf("Dense FFN total:     %.4f ms\n", t_dense);
        printf("  gate_proj:         %.4f ms\n", t_gate_dense);
        printf("  up_proj:           %.4f ms\n", t_up_dense);
        printf("  down_proj:         %.4f ms\n", t_down_dense);
        printf("  (SiLU*mul ~0 ms)\n");

        for (int ti = 0; ti < num_thrs; ++ti) {
            const float thr = thresholds[ti];

            size_t ws_bytes = sparse_ffn_workspace_bytes(I, H);
            void* d_ws_sp;
            CUDA_CHECK(cudaMalloc(&d_ws_sp, ws_bytes));
            __half* d_out_sp;
            CUDA_CHECK(cudaMalloc(&d_out_sp, H * sizeof(__half)));

            // Measure active count at this threshold
            int active_count = 0;
            sparse_swiglu_ffn(d_x, d_gp, d_up, d_dp, d_out_sp, d_ws_sp, H, I, thr, &active_count);
            float sparsity = 1.0f - (float)active_count / I;

            float t_sparse = time_kernel_ms([&](){
                sparse_swiglu_ffn(d_x, d_gp, d_up, d_dp, d_out_sp, d_ws_sp, H, I, thr, nullptr);
            });

            // Per-kernel breakdown for sparse path
            // Reuse workspace buffers for individual kernel timing
            char* wsbuf = reinterpret_cast<char*>(d_ws_sp);
            __half* gate_k    = reinterpret_cast<__half*>(wsbuf);
            __half* silu_k    = gate_k + I;
            __half* up_sp_k   = silu_k + I;
            __half* hidden_k  = up_sp_k + I;
            uintptr_t addr    = reinterpret_cast<uintptr_t>(hidden_k + I);
            addr = (addr + 15) & ~15ull;
            int* idx_k  = reinterpret_cast<int*>(addr);
            int* cnt_k  = idx_k + I;
            addr = reinterpret_cast<uintptr_t>(cnt_k + 1);
            addr = (addr + 15) & ~15ull;
            float* fp32_k = reinterpret_cast<float*>(addr);

            // Prime the compaction so we have valid active_count/idx for sub-kernel timing
            launch_silu_threshold(gate_k, silu_k, idx_k, cnt_k, I, thr);
            CUDA_CHECK(cudaDeviceSynchronize());
            int h_ac = 0;
            CUDA_CHECK(cudaMemcpy(&h_ac, cnt_k, sizeof(int), cudaMemcpyDeviceToHost));
            // Seed gate_k with actual gate output for representative timing
            launch_dense_matvec(d_gp, d_x, gate_k, I, H);
            launch_silu_threshold(gate_k, silu_k, idx_k, cnt_k, I, thr);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_ac, cnt_k, sizeof(int), cudaMemcpyDeviceToHost));

            float t_gate_sp = time_kernel_ms([&](){
                launch_dense_matvec(d_gp, d_x, gate_k, I, H);
            });
            float t_thresh = time_kernel_ms([&](){
                launch_silu_threshold(gate_k, silu_k, idx_k, cnt_k, I, thr);
            });
            float t_up_sp = (h_ac > 0) ? time_kernel_ms([&](){
                launch_col_sparse_matvec(d_up, d_x, idx_k, up_sp_k, I, H, h_ac);
            }) : 0.0f;
            float t_down_sp = (h_ac > 0) ? time_kernel_ms([&](){
                launch_row_sparse_matvec_with_scratch(d_dp, hidden_k, idx_k, d_out2,
                                                      fp32_k, H, I, h_ac);
            }) : 0.0f;

            printf("\nSparse thr=%.3f | sparsity=%.3f (%d/%d active) | total=%.4f ms | speedup=%.2fx\n",
                   thr, sparsity, h_ac, I, t_sparse, t_dense / t_sparse);
            printf("  gate_proj (dense):    %.4f ms\n", t_gate_sp);
            printf("  silu+compact:         %.4f ms\n", t_thresh);
            printf("  up_proj (col-sparse): %.4f ms\n", t_up_sp);
            printf("  down_proj (row-sp):   %.4f ms\n", t_down_sp);

            CUDA_CHECK(cudaFree(d_ws_sp));
            CUDA_CHECK(cudaFree(d_out_sp));
        }

        CUDA_CHECK(cudaFree(d_gp));
        CUDA_CHECK(cudaFree(d_up));
        CUDA_CHECK(cudaFree(d_dp));
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_out_dense));
        CUDA_CHECK(cudaFree(d_ws_dense));
        CUDA_CHECK(cudaFree(d_gate));
        CUDA_CHECK(cudaFree(d_up_buf));
        CUDA_CHECK(cudaFree(d_silu_up));
        CUDA_CHECK(cudaFree(d_out2));
    }

    return 0;
}
