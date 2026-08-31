"""Batched correctness test: batched_sparse_swiglu(B tokens) vs single-token loop."""

import torch
import torch.nn.functional as F
import sys

HIDDEN_DIM       = 4096
INTERMEDIATE_DIM = 11008
THRESHOLD        = 0.01
COSINE_SIM_MIN   = 0.995
BATCH_SIZES      = [1, 4, 16, 64]


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return float(F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item())


def test_batched_vs_single_token_loop():
    from sparse_ffn.functional import sparse_swiglu, batched_sparse_swiglu

    device = torch.device("cuda")
    torch.manual_seed(7)

    gate_w = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM, device=device, dtype=torch.float16) * 0.02
    up_w   = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM, device=device, dtype=torch.float16) * 0.02
    down_w = torch.randn(HIDDEN_DIM, INTERMEDIATE_DIM, device=device, dtype=torch.float16) * 0.02

    print(f"{'batch':>6} {'cosine_sim':>12} {'status':>8}")
    print("-" * 30)

    all_pass = True
    for B in BATCH_SIZES:
        x = torch.randn(B, HIDDEN_DIM, device=device, dtype=torch.float16)

        # Reference: single-token loop
        ref_outs = []
        for i in range(B):
            out_i, _ = sparse_swiglu(x[i], gate_w, up_w, down_w, THRESHOLD)
            ref_outs.append(out_i)
        ref = torch.stack(ref_outs, dim=0)  # [B x hidden_dim]
        torch.cuda.synchronize()

        # Batched kernel
        batched = batched_sparse_swiglu(x, gate_w, up_w, down_w, THRESHOLD)
        torch.cuda.synchronize()

        sim = cosine_sim(batched, ref)
        status = "PASS" if sim >= COSINE_SIM_MIN else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"{B:>6} {sim:>12.6f} {status:>8}")

    if all_pass:
        print("\nPASSED: all batch sizes match single-token loop")
    else:
        raise AssertionError("batched output diverges from single-token loop")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available")
        sys.exit(0)
    test_batched_vs_single_token_loop()
