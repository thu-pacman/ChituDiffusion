"""Optional fused QK RMSNorm + RoPE for Flux2 inference.

SPDX-License-Identifier: Apache-2.0

The CUDA algorithm is adapted from SGLang's diffusion QKNorm+RoPE JIT kernel:
https://github.com/sgl-project/sglang/blob/d6ef68881e26/python/sglang/jit_kernel/csrc/diffusion/qknorm_rope.cuh
(reviewed 2026-07-21).  Thanks to the SGLang community for the original
warp-level fusion.  This module deliberately does not import SGLang,
sgl-kernel, vLLM, or TVM FFI.  It uses PyTorch's optional JIT extension loader.
"""

from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_SOURCE = Path(__file__).with_name("csrc") / "qknorm_rope.cu"


def _supported(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    """Check the exact Flux2 eager-inference contract without raising."""
    if not torch.cuda.is_available() or not q.is_cuda:
        return False
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        return False
    if torch.is_grad_enabled() or getattr(torch, "_dynamo", None) and torch._dynamo.is_compiling():
        return False
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype:
        return False
    if q.ndim != 4 or k.ndim != 4 or q.shape != k.shape or q.shape[-1] != 128:
        return False
    if not q.is_contiguous() or not k.is_contiguous():
        return False
    batch, sequence, _, head_dim = q.shape
    if batch < 1 or sequence < 1:
        return False
    if (
        q_weight.shape != (head_dim,)
        or k_weight.shape != (head_dim,)
        or q_weight.dtype != q.dtype
        or k_weight.dtype != q.dtype
        or not q_weight.is_contiguous()
        or not k_weight.is_contiguous()
    ):
        return False
    if (
        cos.dtype != torch.float32
        or sin.dtype != torch.float32
        or cos.shape != (sequence, head_dim)
        or sin.shape != (sequence, head_dim)
        or not cos.is_contiguous()
        or not sin.is_contiguous()
    ):
        return False
    return all(t.device == q.device for t in (k, q_weight, k_weight, cos, sin))


@lru_cache(maxsize=1)
def _load_extension():
    """Build the small CUDA extension only after a supported call."""
    from torch.utils.cpp_extension import load_inline

    source = _SOURCE.read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode()).hexdigest()[:12]
    return load_inline(
        name=f"chitu_qknorm_rope_{digest}",
        cpp_sources="",
        cuda_sources=source,
        functions=None,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=os.getenv("CHITU_QKNORM_ROPE_VERBOSE", "0") == "1",
    )


@lru_cache(maxsize=1)
def _extension_available() -> bool:
    try:
        _load_extension()
        return True
    except Exception as error:  # Build failures must never affect inference.
        logger.warning("Flux2 fused QKNorm+RoPE is unavailable; using PyTorch fallback: %s", error)
        return False


def fused_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> bool:
    """Normalize and rotate contiguous Flux2 BSHD Q/K in place.

    ``cos`` and ``sin`` are the unmodified diffusers multi-axis RoPE outputs
    for one sequence.  The function returns ``False`` for every unsupported
    input or build failure, leaving ``q`` and ``k`` unchanged.
    """
    if not _supported(q, k, q_weight, k_weight, cos, sin) or not _extension_available():
        return False
    try:
        _load_extension().fused_qknorm_rope(q, k, q_weight, k_weight, cos, sin, float(eps))
        return True
    except Exception as error:
        logger.warning("Flux2 fused QKNorm+RoPE failed; using PyTorch fallback: %s", error)
        return False
