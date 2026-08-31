"""Correctness test comparing sparse_swiglu against dense reference and pure-torch SwiGLU."""

import torch
import torch.nn.functional as F
import math
import sys

HIDDEN_DIM       = 4096
INTERMEDIATE_DIM = 11008
THRESHOLD        = 0.01
COSINE_SIM_MIN   = 0.995


def torch_swiglu(x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch SwiGLU reference: down_proj(SiLU(gate_proj(x)) * up_proj(x))."""
    gate = F.linear(x.float(), gate_w.float())
    up   = F.linear(x.float(), up_w.float())
    silu_gate = gate * torch.sigmoid(gate)
    hidden = silu_gate * up
    out = F.linear(hidden, down_w.float())
    return out.half()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return float(F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item())


def test_sparse_vs_dense_and_torch():
    from sparse_ffn.functional import sparse_swiglu, dense_swiglu

    device = torch.device("cuda")
    torch.manual_seed(42)

    x      = torch.randn(HIDDEN_DIM, device=device, dtype=torch.float16)
    gate_w = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM, device=device, dtype=torch.float16) * 0.02
    up_w   = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM, device=device, dtype=torch.float16) * 0.02
    down_w = torch.randn(HIDDEN_DIM, INTERMEDIATE_DIM, device=device, dtype=torch.float16) * 0.02

    dense_out = dense_swiglu(x, gate_w, up_w, down_w)
    sparse_out, active_count = sparse_swiglu(x, gate_w, up_w, down_w, THRESHOLD)
    torch_out = torch_swiglu(x, gate_w, up_w, down_w)

    torch.cuda.synchronize()

    sparsity = 1.0 - active_count / INTERMEDIATE_DIM
    print(f"active neurons: {active_count}/{INTERMEDIATE_DIM} ({sparsity:.1%} sparse)")

    sim_sparse_dense = cosine_sim(sparse_out, dense_out)
    sim_sparse_torch = cosine_sim(sparse_out, torch_out)
    sim_dense_torch  = cosine_sim(dense_out, torch_out)

    print(f"cosine sim sparse vs dense:  {sim_sparse_dense:.6f}")
    print(f"cosine sim sparse vs torch:  {sim_sparse_torch:.6f}")
    print(f"cosine sim dense  vs torch:  {sim_dense_torch:.6f}")

    assert sim_sparse_dense >= COSINE_SIM_MIN, (
        f"sparse vs dense cosine sim {sim_sparse_dense:.6f} < {COSINE_SIM_MIN}"
    )
    assert sim_sparse_torch >= COSINE_SIM_MIN, (
        f"sparse vs torch cosine sim {sim_sparse_torch:.6f} < {COSINE_SIM_MIN}"
    )
    assert sim_dense_torch >= COSINE_SIM_MIN, (
        f"dense vs torch cosine sim {sim_dense_torch:.6f} < {COSINE_SIM_MIN}"
    )

    print("PASSED: all cosine similarity checks >= 0.995")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)
    test_sparse_vs_dense_and_torch()
