import torch
import torch.nn as nn
from typing import Optional

from .functional import sparse_swiglu, dense_swiglu, batched_sparse_swiglu, batched_dense_swiglu


class SparseFFNLayer(nn.Module):
    """SwiGLU FFN with dynamic activation-gated sparsity.

    Skips computation for neurons where |SiLU(gate_proj(x))| <= threshold,
    giving speedups proportional to the fraction of dead neurons.

    Args:
        hidden_dim:       input/output dimension
        intermediate_dim: inner projection dimension (e.g. 4 * hidden_dim for Llama)
        threshold:        SiLU magnitude cutoff below which a neuron is skipped
        device:           torch device (must be CUDA)
        dtype:            must be torch.float16
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        threshold: float = 0.01,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        if dtype != torch.float16:
            raise ValueError("SparseFFNLayer only supports float16")
        factory = {"device": device, "dtype": dtype}
        self.gate_proj = nn.Parameter(torch.empty(intermediate_dim, hidden_dim, **factory))
        self.up_proj   = nn.Parameter(torch.empty(intermediate_dim, hidden_dim, **factory))
        self.down_proj = nn.Parameter(torch.empty(hidden_dim, intermediate_dim, **factory))
        self.threshold = threshold
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        nn.init.normal_(self.gate_proj, std=0.02)
        nn.init.normal_(self.up_proj,   std=0.02)
        nn.init.normal_(self.down_proj, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse forward; x may be 1D [hidden_dim] or 2D [batch x hidden_dim]."""
        if x.dim() == 1:
            out, _ = sparse_swiglu(x, self.gate_proj, self.up_proj, self.down_proj, self.threshold)
            return out
        return batched_sparse_swiglu(x, self.gate_proj, self.up_proj, self.down_proj, self.threshold)

    def dense_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dense reference forward path."""
        if x.dim() == 1:
            return dense_swiglu(x, self.gate_proj, self.up_proj, self.down_proj)
        return batched_dense_swiglu(x, self.gate_proj, self.up_proj, self.down_proj)

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, intermediate_dim={self.intermediate_dim}, "
            f"threshold={self.threshold}"
        )
