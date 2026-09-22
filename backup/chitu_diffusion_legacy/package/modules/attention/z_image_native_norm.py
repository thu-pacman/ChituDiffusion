"""Opt-in adapter for SGLang-derived Z-Image native norm fusions.

The fused kernels themselves live in :mod:`kernel_adapters.zimage_native_norm`.
This module only replaces the diffusers block ``forward`` method when explicitly
enabled, preserving diffusers as the default model-semantic implementation.
"""

from __future__ import annotations

import types
from collections.abc import Iterable

import torch

from kernel_adapters.zimage_native_norm import (
    ZImageNormMode,
    rmsnorm_scale,
    rmsnorm_tanh_residual,
)


def _rmsnorm_fallback(
    x: torch.Tensor, norm: torch.nn.Module, mode: ZImageNormMode
) -> torch.Tensor:
    if mode == "diffusers_fp32" or x.dtype is not torch.bfloat16:
        return norm(x)
    mean_square = x.square().to(torch.bfloat16).mean(dim=-1, keepdim=True).to(torch.bfloat16)
    rstd = torch.rsqrt((mean_square + norm.eps).to(torch.float32)).to(torch.bfloat16)
    return ((x * rstd).to(torch.bfloat16) * norm.weight).to(torch.bfloat16)


def _rmsnorm_scale_or_native(
    x: torch.Tensor,
    scale: torch.Tensor,
    norm: torch.nn.Module,
    mode: ZImageNormMode = "diffusers_fp32",
) -> torch.Tensor:
    fused = rmsnorm_scale(x, norm.weight, scale, norm.eps, mode=mode)
    return fused if fused is not None else _rmsnorm_fallback(x, norm, mode) * scale


def _rmsnorm_tanh_residual_or_native(
    x: torch.Tensor,
    gate: torch.Tensor,
    residual: torch.Tensor,
    norm: torch.nn.Module,
    mode: ZImageNormMode = "diffusers_fp32",
) -> torch.Tensor:
    fused = rmsnorm_tanh_residual(x, gate, residual, norm.weight, norm.eps, mode=mode)
    return fused if fused is not None else residual + torch.tanh(gate) * _rmsnorm_fallback(x, norm, mode)


def _fused_zimage_block_forward(
    self,
    x: torch.Tensor,
    attn_mask: torch.Tensor,
    freqs_cis: torch.Tensor,
    adaln_input: torch.Tensor | None = None,
    noise_mask: torch.Tensor | None = None,
    adaln_noisy: torch.Tensor | None = None,
    adaln_clean: torch.Tensor | None = None,
) -> torch.Tensor:
    # Mirrors diffusers.ZImageTransformerBlock.forward. Every individual fusion
    # can decline (e.g. CPU, non-BF16, or unsupported layout) and takes the
    # original expression below, so enabling this adapter is safe across devices.
    if self.modulation:
        sequence_length = x.shape[1]
        if noise_mask is not None:
            from diffusers.models.transformers.transformer_z_image import select_per_token

            mod_noisy = self.adaLN_modulation(adaln_noisy)
            mod_clean = self.adaLN_modulation(adaln_clean)
            scale_msa_noisy, gate_msa_noisy, scale_mlp_noisy, gate_mlp_noisy = mod_noisy.chunk(4, dim=1)
            scale_msa_clean, gate_msa_clean, scale_mlp_clean, gate_mlp_clean = mod_clean.chunk(4, dim=1)
            scale_msa_noisy, scale_mlp_noisy = 1.0 + scale_msa_noisy, 1.0 + scale_mlp_noisy
            scale_msa_clean, scale_mlp_clean = 1.0 + scale_msa_clean, 1.0 + scale_mlp_clean
            scale_msa = select_per_token(scale_msa_noisy, scale_msa_clean, noise_mask, sequence_length)
            scale_mlp = select_per_token(scale_mlp_noisy, scale_mlp_clean, noise_mask, sequence_length)
            gate_msa = select_per_token(gate_msa_noisy, gate_msa_clean, noise_mask, sequence_length)
            gate_mlp = select_per_token(gate_mlp_noisy, gate_mlp_clean, noise_mask, sequence_length)
        else:
            mod = self.adaLN_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        mode = self._chitu_zimage_norm_mode
        attn_out = self.attention(
            _rmsnorm_scale_or_native(x, scale_msa, self.attention_norm1, mode),
            attention_mask=attn_mask,
            freqs_cis=freqs_cis,
        )
        x = _rmsnorm_tanh_residual_or_native(attn_out, gate_msa, x, self.attention_norm2, mode)
        ffn_out = self.feed_forward(_rmsnorm_scale_or_native(x, scale_mlp, self.ffn_norm1, mode))
        return _rmsnorm_tanh_residual_or_native(ffn_out, gate_mlp, x, self.ffn_norm2, mode)

    attn_out = self.attention(self.attention_norm1(x), attention_mask=attn_mask, freqs_cis=freqs_cis)
    x = x + self.attention_norm2(attn_out)
    return x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))


def install_zimage_native_norm_fusions(
    blocks: Iterable[torch.nn.Module], mode: ZImageNormMode = "diffusers_fp32"
) -> int:
    """Monkey-patch compatible diffusers blocks and return the patched count."""

    if mode not in {"diffusers_fp32", "native_bf16"}:
        raise ValueError(f"Unsupported Z-Image norm fusion mode: {mode!r}")
    patched = 0
    for block in blocks:
        required = (
            "attention",
            "feed_forward",
            "attention_norm1",
            "attention_norm2",
            "ffn_norm1",
            "ffn_norm2",
            "modulation",
        )
        if getattr(block, "_chitu_zimage_native_norm_patched", False) or not all(
            hasattr(block, attribute) for attribute in required
        ):
            continue
        block.forward = types.MethodType(_fused_zimage_block_forward, block)
        block._chitu_zimage_norm_mode = mode
        block._chitu_zimage_native_norm_patched = True
        patched += 1
    return patched
