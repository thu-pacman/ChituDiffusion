"""Opt-in exact-expression Qwen-Image modulation fusion adapter.

The CUDA kernels are intentionally isolated in :mod:`kernel_adapters`. This
adapter retains the installed diffusers block for every unsupported invocation.
It only restructures Qwen's ``zero_cond_t`` select-0/1 image branch, where the
upstream SGLang fusion has an exact operation-level counterpart.
"""

from __future__ import annotations

import types
from collections.abc import Iterable
from typing import Any

import torch

from kernel_adapters.qwen_image_modulation import (
    layernorm_scale_shift_gate_select01,
    layernorm_scale_shift_gate_select01_reference,
    residual_layernorm_scale_shift_gate_select01,
    residual_layernorm_scale_shift_gate_select01_reference,
)


def _split_select01_modulation(
    mod_params: torch.Tensor, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Split diffusers' doubled zero-condition modulation into select-0/1 rows."""

    shift, scale, gate = mod_params.chunk(3, dim=-1)
    if shift.shape[0] != 2 * batch_size:
        return None
    return (
        scale[:batch_size],
        shift[:batch_size],
        gate[:batch_size],
        scale[batch_size:],
        shift[batch_size:],
        gate[batch_size:],
    )


def _fused_qwen_image_block_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
    modulate_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror diffusers while fusing only its select-0/1 image path."""

    if modulate_index is None:
        return self._chitu_qwen_modulation_original_forward(
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            temb,
            image_rotary_emb,
            joint_attention_kwargs,
            modulate_index,
        )

    batch_size = hidden_states.shape[0]
    img_mod_params = self.img_mod(temb)
    select01_mod1, select01_mod2 = img_mod_params.chunk(2, dim=-1)
    first_params = _split_select01_modulation(select01_mod1, batch_size)
    second_params = _split_select01_modulation(select01_mod2, batch_size)
    if first_params is None or second_params is None:
        return self._chitu_qwen_modulation_original_forward(
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            temb,
            image_rotary_emb,
            joint_attention_kwargs,
            modulate_index,
        )

    txt_temb = temb.chunk(2, dim=0)[0] if self.zero_cond_t else temb
    txt_mod_params = self.txt_mod(txt_temb)
    txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

    img_modulated_and_gate = layernorm_scale_shift_gate_select01(
        hidden_states, *first_params, modulate_index, self.img_norm1.eps
    )
    if img_modulated_and_gate is None:
        img_modulated, img_gate1 = layernorm_scale_shift_gate_select01_reference(
            hidden_states, *first_params, modulate_index, self.img_norm1.eps
        )
    else:
        img_modulated, img_gate1 = img_modulated_and_gate

    txt_normed = self.txt_norm1(encoder_hidden_states)
    txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)
    joint_attention_kwargs = joint_attention_kwargs or {}
    img_attn_output, txt_attn_output = self.attn(
        hidden_states=img_modulated,
        encoder_hidden_states=txt_modulated,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )

    img_modulated_and_residual = residual_layernorm_scale_shift_gate_select01(
        img_attn_output, hidden_states, img_gate1, *second_params, modulate_index, self.img_norm2.eps
    )
    if img_modulated_and_residual is None:
        img_modulated2, hidden_states, img_gate2 = residual_layernorm_scale_shift_gate_select01_reference(
            img_attn_output, hidden_states, img_gate1, *second_params, modulate_index, self.img_norm2.eps
        )
    else:
        img_modulated2, hidden_states, img_gate2 = img_modulated_and_residual
    hidden_states = hidden_states + img_gate2 * self.img_mlp(img_modulated2)

    txt_normed2 = self.txt_norm2(encoder_hidden_states + txt_gate1 * txt_attn_output)
    txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
    encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output + txt_gate2 * self.txt_mlp(txt_modulated2)

    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return encoder_hidden_states, hidden_states


def install_qwen_image_modulation_fusions(blocks: Iterable[torch.nn.Module]) -> int:
    """Patch compatible diffusers Qwen blocks and return their count.

    The patched blocks retain their original ``forward`` for all non-select01
    calls and use the exact PyTorch fallback when Triton is unavailable.
    """

    required = (
        "img_mod",
        "img_norm1",
        "img_norm2",
        "img_mlp",
        "txt_mod",
        "txt_norm1",
        "txt_norm2",
        "txt_mlp",
        "attn",
        "_modulate",
        "zero_cond_t",
    )
    patched = 0
    for block in blocks:
        if getattr(block, "_chitu_qwen_modulation_patched", False) or not all(
            hasattr(block, attribute) for attribute in required
        ):
            continue
        block._chitu_qwen_modulation_original_forward = block.forward
        block.forward = types.MethodType(_fused_qwen_image_block_forward, block)
        block._chitu_qwen_modulation_patched = True
        patched += 1
    return patched
