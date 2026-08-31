"""Throughput benchmark sweeping batch sizes 1-64, reporting tokens/second."""

import torch
import sys

HIDDEN_DIM       = 4096
INTERMEDIATE_DIM = 11008
THRESHOLD        = 0.01
WARMUP           = 30
ITERS            = 100
BATCH_SIZES      = [1, 2, 4, 8, 16, 32, 64]


def bench_us(fn, warmup: int, iters: int) -> float:
    """Returns mean latency in microseconds."""
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
    return start.elapsed_time(end) * 1000.0 / iters


def run():
    from sparse_ffn.functional import batched_sparse_swiglu, batched_dense_swiglu

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: hidden={HIDDEN_DIM}, inter={INTERMEDIATE_DIM}, threshold={THRESHOLD}")
    print()
    print(f"{'batch':>6} {'dense_us':>10} {'sparse_us':>10} {'speedup':>8} "
          f"{'dense_tok/s':>12} {'sparse_tok/s':>12} {'sparsity':>9}")
    print("-" * 75)

    torch.manual_seed(0)
    gate_w = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM, device=device, dtype=torch.float16) * 0.02
    up_w   = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM, device=device, dtype=torch.float16) * 0.02
    down_w = torch.randn(HIDDEN_DIM, INTERMEDIATE_DIM, device=device, dtype=torch.float16) * 0.02

    for B in BATCH_SIZES:
        x = torch.randn(B, HIDDEN_DIM, device=device, dtype=torch.float16)

        dense_us  = bench_us(lambda: batched_dense_swiglu(x, gate_w, up_w, down_w),  WARMUP, ITERS)
        sparse_us = bench_us(lambda: batched_sparse_swiglu(x, gate_w, up_w, down_w, THRESHOLD), WARMUP, ITERS)

        speedup      = dense_us / sparse_us if sparse_us > 0 else float("inf")
        dense_tps    = B / (dense_us * 1e-6)
        sparse_tps   = B / (sparse_us * 1e-6)

        # measure sparsity from first token of this batch
        from sparse_ffn.functional import sparse_swiglu
        _, active = sparse_swiglu(x[0], gate_w, up_w, down_w, THRESHOLD)
        torch.cuda.synchronize()
        sparsity = 1.0 - active / INTERMEDIATE_DIM

        print(f"{B:>6} {dense_us:>10.1f} {sparse_us:>10.1f} {speedup:>8.2f}x "
              f"{dense_tps:>12.0f} {sparse_tps:>12.0f} {sparsity:>8.1%}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)
    run()
