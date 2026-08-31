"""CUDA-event latency benchmark for sparse vs dense SwiGLU at Llama-2 and Mixtral dims."""

import torch
import sys

WARMUP   = 50
ITERS    = 200
THRESHOLD = 0.01

CONFIGS = [
    ("Llama-2 7B",  4096,  11008),
    ("Llama-2 70B", 8192,  28672),
    ("Mixtral",     4096,  14336),
]


def bench(fn, warmup: int, iters: int) -> float:
    """Returns mean latency in microseconds using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # ms -> us


def run_benchmarks():
    from sparse_ffn.functional import sparse_swiglu, dense_swiglu

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'Config':<16} {'dense_us':>10} {'sparse_us':>10} {'speedup':>8} {'sparsity':>9}")
    print("-" * 58)

    for name, hidden_dim, inter_dim in CONFIGS:
        torch.manual_seed(0)
        x      = torch.randn(hidden_dim, device=device, dtype=torch.float16)
        gate_w = torch.randn(inter_dim, hidden_dim, device=device, dtype=torch.float16) * 0.02
        up_w   = torch.randn(inter_dim, hidden_dim, device=device, dtype=torch.float16) * 0.02
        down_w = torch.randn(hidden_dim, inter_dim, device=device, dtype=torch.float16) * 0.02

        # measure sparsity
        _, active = sparse_swiglu(x, gate_w, up_w, down_w, THRESHOLD)
        torch.cuda.synchronize()
        sparsity = 1.0 - active / inter_dim

        dense_us  = bench(lambda: dense_swiglu(x, gate_w, up_w, down_w),  WARMUP, ITERS)
        sparse_us = bench(lambda: sparse_swiglu(x, gate_w, up_w, down_w, THRESHOLD), WARMUP, ITERS)
        speedup   = dense_us / sparse_us if sparse_us > 0 else float("inf")

        print(f"{name:<16} {dense_us:>10.1f} {sparse_us:>10.1f} {speedup:>8.2f}x {sparsity:>8.1%}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)
    run_benchmarks()
