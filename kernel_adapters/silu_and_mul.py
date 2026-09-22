"""Optional exact-layout SiLU × mul fusion for dense SwiGLU.

SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the vLLM project

The operator contract and CUDA implementation strategy are adapted from vLLM
commit ``616c9bd0f4efff9761fd762e4247e18b6bd7eb23``:
``vllm/model_executor/layers/activation.py:SiluAndMul`` and
``csrc/libtorch_stable/activation_kernels.cu:silu_and_mul``.

It intentionally does not import vLLM or its ``torch.ops._C`` extension.
Thank you to the vLLM community for publishing and maintaining the original
Apache-2.0 implementation.
"""

from __future__ import annotations

import importlib.util
import logging

import torch

logger = logging.getLogger(__name__)

_TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
if _TRITON_AVAILABLE:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

    @triton.jit
    def _silu_and_mul_kernel(
        output_ptr,
        input_ptr,
        rows,
        width: tl.constexpr,
        block_width: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_width)
        mask = offsets < width
        base = row * (2 * width)
        gate_input = tl.load(input_ptr + base + offsets, mask=mask, other=0.0)
        up_input = tl.load(input_ptr + base + width + offsets, mask=mask, other=0.0)
        gate = gate_input.to(tl.float32)
        # torch.nn.SiLU materializes the activation in the input dtype before
        # Flux2 multiplies it by the second half of the projection.
        activated = (gate * tl.sigmoid(gate)).to(gate_input.dtype)
        output = (activated * up_input).to(gate_input.dtype)
        tl.store(output_ptr + row * width + offsets, output, mask=mask)


def triton_available() -> bool:
    """Whether this process can import the optional Triton dependency."""

    return _TRITON_AVAILABLE


def can_use_silu_and_mul(x: torch.Tensor) -> bool:
    """Return whether ``x`` satisfies the exact Flux2 eager launch contract."""

    return (
        _TRITON_AVAILABLE
        and x.is_cuda
        and x.dtype is torch.bfloat16
        and x.ndim >= 2
        and x.is_contiguous()
        and x.shape[-1] > 0
        and x.shape[-1] % 2 == 0
    )


def silu_and_mul(x: torch.Tensor) -> torch.Tensor | None:
    """Fuse ``silu(x[..., :d]) * x[..., d:]`` for contiguous CUDA BF16.

    The result has ``x.dtype`` and the same leading dimensions as the PyTorch
    expression. ``None`` means the caller must execute its original PyTorch
    expression; this is the CPU, unsupported dtype/layout, or failed-launch
    fallback signal.
    """

    if not can_use_silu_and_mul(x):
        return None

    width = x.shape[-1] // 2
    rows = x.numel() // (2 * width)
    output = torch.empty((*x.shape[:-1], width), dtype=x.dtype, device=x.device)
    if rows == 0:
        return output

    try:
        with torch.get_device_module().device(x.device):
            _silu_and_mul_kernel[(rows,)](
                output,
                x,
                rows,
                width=width,
                block_width=triton.next_power_of_2(width),
                num_warps=8 if width >= 1024 else 4,
            )
    except Exception as error:  # pragma: no cover - requires a broken CUDA launch
        logger.warning("Flux2 SiLU×mul Triton fusion failed; using PyTorch fallback: %s", error)
        return None
    return output
