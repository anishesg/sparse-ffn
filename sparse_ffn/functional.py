"""Stateless functional API for sparse and dense SwiGLU FFN."""

import torch
from typing import Tuple


def _ext():
    """Lazy-load the compiled CUDA extension."""
    try:
        import sparse_ffn._C as _C
        return _C
    except ImportError as e:
        raise ImportError(
            "sparse_ffn._C not found. Run 'pip install -e .' to build the extension."
        ) from e


def sparse_swiglu(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    threshold: float = 0.01,
) -> Tuple[torch.Tensor, int]:
    """Sparse SwiGLU forward for a single token.

    Args:
        x:         [hidden_dim], float16, CUDA
        gate_w:    [intermediate_dim x hidden_dim], float16, CUDA
        up_w:      [intermediate_dim x hidden_dim], float16, CUDA
        down_w:    [hidden_dim x intermediate_dim], float16, CUDA
        threshold: neurons with |SiLU(gate)| <= threshold are skipped

    Returns:
        (output tensor [hidden_dim], number of active neurons used)
    """
    return _ext().sparse_swiglu_forward(x, gate_w, up_w, down_w, threshold)


def dense_swiglu(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> torch.Tensor:
    """Dense SwiGLU forward for a single token.

    Args:
        x:      [hidden_dim], float16, CUDA
        gate_w: [intermediate_dim x hidden_dim], float16, CUDA
        up_w:   [intermediate_dim x hidden_dim], float16, CUDA
        down_w: [hidden_dim x intermediate_dim], float16, CUDA

    Returns:
        output tensor [hidden_dim]
    """
    return _ext().dense_swiglu_forward(x, gate_w, up_w, down_w)


def batched_sparse_swiglu(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    threshold: float = 0.01,
) -> torch.Tensor:
    """Sparse SwiGLU forward for a batch of tokens.

    Args:
        x:         [batch x hidden_dim], float16, CUDA
        gate_w:    [intermediate_dim x hidden_dim], float16, CUDA
        up_w:      [intermediate_dim x hidden_dim], float16, CUDA
        down_w:    [hidden_dim x intermediate_dim], float16, CUDA
        threshold: neurons with |SiLU(gate)| <= threshold are skipped

    Returns:
        output tensor [batch x hidden_dim]
    """
    return _ext().batched_sparse_swiglu_forward(x, gate_w, up_w, down_w, threshold)


def batched_dense_swiglu(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> torch.Tensor:
    """Dense SwiGLU forward for a batch of tokens.

    Args:
        x:      [batch x hidden_dim], float16, CUDA
        gate_w: [intermediate_dim x hidden_dim], float16, CUDA
        up_w:   [intermediate_dim x hidden_dim], float16, CUDA
        down_w: [hidden_dim x intermediate_dim], float16, CUDA

    Returns:
        output tensor [batch x hidden_dim]
    """
    return _ext().batched_dense_swiglu_forward(x, gate_w, up_w, down_w)
