from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _rms_norm_kernel(
        input_ptr,
        output_ptr,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < hidden_size
        values = tl.load(
            input_ptr + row * hidden_size + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(values * values, axis=0) / hidden_size + eps)
        tl.store(
            output_ptr + row * hidden_size + offsets,
            values * inverse_rms,
            mask=mask,
        )
else:
    _rms_norm_kernel = None


def rms_norm_available() -> bool:
    return triton is not None


def rms_norm(input: torch.Tensor, eps: float) -> torch.Tensor:
    if triton is None or _rms_norm_kernel is None:
        raise RuntimeError("Triton is unavailable")
    if not input.is_cuda:
        raise ValueError("Triton RMSNorm requires a CUDA tensor")
    if not input.is_contiguous():
        input = input.contiguous()
    hidden_size = input.shape[-1]
    block_size = triton.next_power_of_2(hidden_size)
    if block_size > 4096:
        raise ValueError(f"RMSNorm hidden size {hidden_size} is too large")
    output = torch.empty_like(input)
    _rms_norm_kernel[(input.numel() // hidden_size,)](
        input,
        output,
        hidden_size=hidden_size,
        eps=float(eps),
        block_size=block_size,
        num_warps=8,
    )
    return output
