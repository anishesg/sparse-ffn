"""Sparsity-vs-quality sweep: tests threshold values across model dimensions.

Prints a table of (threshold, sparsity fraction, cosine similarity vs dense)
to guide threshold selection for production deployment.
"""

import torch
import torch.nn.functional as F
import sys

THRESHOLDS = [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

CONFIGS = [
    ("Llama-2 7B",  4096,  11008),
    ("Llama-2 70B", 8192,  28672),
    ("Mixtral",     4096,  14336),
]


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return float(F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item())


def run_sweep():
    from sparse_ffn.functional import sparse_swiglu, dense_swiglu

    device = torch.device("cuda")

    for config_name, hidden_dim, inter_dim in CONFIGS:
        print(f"\n=== {config_name} (hidden={hidden_dim}, inter={inter_dim}) ===")
        print(f"{'threshold':>12} {'sparsity':>10} {'cosine_sim':>12} {'active':>10}")
        print("-" * 48)

        torch.manual_seed(42)
        x      = torch.randn(hidden_dim, device=device, dtype=torch.float16)
        gate_w = torch.randn(inter_dim, hidden_dim, device=device, dtype=torch.float16) * 0.02
        up_w   = torch.randn(inter_dim, hidden_dim, device=device, dtype=torch.float16) * 0.02
        down_w = torch.randn(hidden_dim, inter_dim, device=device, dtype=torch.float16) * 0.02

        dense_out = dense_swiglu(x, gate_w, up_w, down_w)
        torch.cuda.synchronize()

        for thresh in THRESHOLDS:
            sparse_out, active = sparse_swiglu(x, gate_w, up_w, down_w, thresh)
            torch.cuda.synchronize()
            sparsity = 1.0 - active / inter_dim
            sim = cosine_sim(sparse_out, dense_out)
            print(f"{thresh:>12g} {sparsity:>9.1%} {sim:>12.6f} {active:>10d}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)
    run_sweep()
