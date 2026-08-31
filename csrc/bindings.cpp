#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "dense_ffn.cuh"
#include "sparse_ffn.cuh"
#include "sparse_ffn_batched.cuh"
#include <cuda_fp16.h>
#include <stdexcept>

static void check_fp16_contiguous(const torch::Tensor& t, const char* name) {
    if (t.scalar_type() != torch::kFloat16)
        throw std::invalid_argument(std::string(name) + " must be float16");
    if (!t.is_contiguous())
        throw std::invalid_argument(std::string(name) + " must be contiguous");
}

// single-token dense forward
torch::Tensor dense_swiglu_forward(
    torch::Tensor x,
    torch::Tensor gate_w,
    torch::Tensor up_w,
    torch::Tensor down_w)
{
    check_fp16_contiguous(x, "x");
    check_fp16_contiguous(gate_w, "gate_w");
    check_fp16_contiguous(up_w, "up_w");
    check_fp16_contiguous(down_w, "down_w");

    TORCH_CHECK(x.dim() == 1, "x must be 1D for single-token forward");
    int hidden_dim       = x.size(0);
    int intermediate_dim = gate_w.size(0);
    TORCH_CHECK(gate_w.sizes() == torch::IntArrayRef({intermediate_dim, hidden_dim}),
                "gate_w shape mismatch");
    TORCH_CHECK(up_w.sizes()   == torch::IntArrayRef({intermediate_dim, hidden_dim}),
                "up_w shape mismatch");
    TORCH_CHECK(down_w.sizes() == torch::IntArrayRef({hidden_dim, intermediate_dim}),
                "down_w shape mismatch");

    auto out = torch::empty({hidden_dim}, x.options());
    auto ws  = torch::empty({3 * intermediate_dim}, x.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dense_swiglu_ffn(
        reinterpret_cast<const __half*>(x.data_ptr()),
        reinterpret_cast<const __half*>(gate_w.data_ptr()),
        reinterpret_cast<const __half*>(up_w.data_ptr()),
        reinterpret_cast<const __half*>(down_w.data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()),
        reinterpret_cast<__half*>(ws.data_ptr()),
        hidden_dim, intermediate_dim, stream);
    return out;
}

// single-token sparse forward
std::tuple<torch::Tensor, int> sparse_swiglu_forward(
    torch::Tensor x,
    torch::Tensor gate_w,
    torch::Tensor up_w,
    torch::Tensor down_w,
    float threshold)
{
    check_fp16_contiguous(x, "x");
    check_fp16_contiguous(gate_w, "gate_w");
    check_fp16_contiguous(up_w, "up_w");
    check_fp16_contiguous(down_w, "down_w");

    TORCH_CHECK(x.dim() == 1, "x must be 1D for single-token forward");
    int hidden_dim       = x.size(0);
    int intermediate_dim = gate_w.size(0);
    TORCH_CHECK(gate_w.sizes() == torch::IntArrayRef({intermediate_dim, hidden_dim}),
                "gate_w shape mismatch");
    TORCH_CHECK(up_w.sizes()   == torch::IntArrayRef({intermediate_dim, hidden_dim}),
                "up_w shape mismatch");
    TORCH_CHECK(down_w.sizes() == torch::IntArrayRef({hidden_dim, intermediate_dim}),
                "down_w shape mismatch");

    size_t ws_bytes = sparse_ffn_workspace_bytes(intermediate_dim, hidden_dim);
    auto ws_storage = torch::empty({(int64_t)ws_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(x.device()));
    auto out = torch::empty({hidden_dim}, x.options());

    int active_count = 0;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sparse_swiglu_ffn(
        reinterpret_cast<const __half*>(x.data_ptr()),
        reinterpret_cast<const __half*>(gate_w.data_ptr()),
        reinterpret_cast<const __half*>(up_w.data_ptr()),
        reinterpret_cast<const __half*>(down_w.data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()),
        ws_storage.data_ptr(),
        hidden_dim, intermediate_dim, threshold, &active_count, stream);
    return {out, active_count};
}

// batched dense forward: x is [batch x hidden_dim]
torch::Tensor batched_dense_swiglu_forward(
    torch::Tensor x,
    torch::Tensor gate_w,
    torch::Tensor up_w,
    torch::Tensor down_w)
{
    check_fp16_contiguous(x, "x");
    check_fp16_contiguous(gate_w, "gate_w");
    check_fp16_contiguous(up_w, "up_w");
    check_fp16_contiguous(down_w, "down_w");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [batch x hidden_dim] for batched forward");
    int batch_size       = x.size(0);
    int hidden_dim       = x.size(1);
    int intermediate_dim = gate_w.size(0);
    TORCH_CHECK(gate_w.sizes() == torch::IntArrayRef({intermediate_dim, hidden_dim}),
                "gate_w shape mismatch");
    TORCH_CHECK(up_w.sizes()   == torch::IntArrayRef({intermediate_dim, hidden_dim}),
                "up_w shape mismatch");
    TORCH_CHECK(down_w.sizes() == torch::IntArrayRef({hidden_dim, intermediate_dim}),
                "down_w shape mismatch");

    auto out = torch::empty({batch_size, hidden_dim}, x.options());
    size_t ws_bytes = (size_t)3 * intermediate_dim * sizeof(__half);
    auto ws_storage = torch::empty({batch_size * (int64_t)(3 * intermediate_dim)}, x.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    batched_dense_swiglu(
        reinterpret_cast<const __half*>(x.data_ptr()),
        reinterpret_cast<const __half*>(gate_w.data_ptr()),
        reinterpret_cast<const __half*>(up_w.data_ptr()),
        reinterpret_cast<const __half*>(down_w.data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()),
        reinterpret_cast<__half*>(ws_storage.data_ptr()),
        batch_size, hidden_dim, intermediate_dim, stream);
    return out;
}

// batched sparse forward: x is [batch x hidden_dim]
torch::Tensor batched_sparse_swiglu_forward(
    torch::Tensor x,
    torch::Tensor gate_w,
    torch::Tensor up_w,
    torch::Tensor down_w,
    float threshold)
{
    check_fp16_contiguous(x, "x");
    check_fp16_contiguous(gate_w, "gate_w");
    check_fp16_contiguous(up_w, "up_w");
    check_fp16_contiguous(down_w, "down_w");

    TORCH_CHECK(x.dim() == 2, "x must be 2D [batch x hidden_dim] for batched forward");
    int batch_size       = x.size(0);
    int hidden_dim       = x.size(1);
    int intermediate_dim = gate_w.size(0);
    TORCH_CHECK(gate_w.sizes() == torch::IntArrayRef({intermediate_dim, hidden_dim}),
                "gate_w shape mismatch");
    TORCH_CHECK(up_w.sizes()   == torch::IntArrayRef({intermediate_dim, hidden_dim}),
                "up_w shape mismatch");
    TORCH_CHECK(down_w.sizes() == torch::IntArrayRef({hidden_dim, intermediate_dim}),
                "down_w shape mismatch");

    auto out = torch::empty({batch_size, hidden_dim}, x.options());
    size_t per_token_ws = batched_sparse_ffn_workspace_bytes(intermediate_dim, hidden_dim);
    auto ws_storage = torch::empty({(int64_t)(batch_size * per_token_ws)},
                                   torch::TensorOptions().dtype(torch::kUInt8).device(x.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    batched_sparse_swiglu(
        reinterpret_cast<const __half*>(x.data_ptr()),
        reinterpret_cast<const __half*>(gate_w.data_ptr()),
        reinterpret_cast<const __half*>(up_w.data_ptr()),
        reinterpret_cast<const __half*>(down_w.data_ptr()),
        reinterpret_cast<__half*>(out.data_ptr()),
        ws_storage.data_ptr(),
        batch_size, hidden_dim, intermediate_dim, threshold, stream);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dense_swiglu_forward",          &dense_swiglu_forward,
          "Dense SwiGLU forward (single token)");
    m.def("sparse_swiglu_forward",         &sparse_swiglu_forward,
          "Sparse SwiGLU forward (single token), returns (output, active_count)");
    m.def("batched_dense_swiglu_forward",  &batched_dense_swiglu_forward,
          "Dense SwiGLU forward (batched)");
    m.def("batched_sparse_swiglu_forward", &batched_sparse_swiglu_forward,
          "Sparse SwiGLU forward (batched)");
}
