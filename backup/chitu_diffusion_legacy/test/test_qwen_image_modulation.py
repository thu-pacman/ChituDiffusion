"""Correctness gates for Qwen-Image select-0/1 modulation fusion."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from chitu_diffusion.modules.attention.qwen_image_modulation import install_qwen_image_modulation_fusions
from chitu_diffusion.runtime.adapter.qwen_image import _qwen_image_modulation_fusion_enabled
from kernel_adapters.qwen_image_modulation import (
    layernorm_scale_shift_gate_select01,
    layernorm_scale_shift_gate_select01_reference,
    residual_layernorm_scale_shift_gate_select01,
    residual_layernorm_scale_shift_gate_select01_reference,
    triton_available,
)


def _inputs(device: str = "cpu", dtype: torch.dtype = torch.float32):
    torch.manual_seed(7)
    batch, sequence, dim = 2, 5, 16
    x = torch.randn(batch, sequence, dim, device=device, dtype=dtype)
    residual = torch.randn_like(x)
    residual_gate = torch.randn_like(x)
    modulation = [torch.randn(batch, dim, device=device, dtype=dtype) for _ in range(6)]
    index = torch.randint(0, 2, (batch, sequence), device=device, dtype=torch.int32)
    return x, residual, residual_gate, *modulation, index


def test_qwen_modulation_cpu_uses_exact_reference_fallback():
    x, residual, residual_gate, *modulation, index = _inputs()
    expected, gate = layernorm_scale_shift_gate_select01_reference(x, *modulation, index, 1e-6)
    direct = F.layer_norm(x, (x.shape[-1],), eps=1e-6)
    selected = index.bool().unsqueeze(-1)
    scale = torch.where(selected, modulation[3].unsqueeze(1), modulation[0].unsqueeze(1))
    shift = torch.where(selected, modulation[4].unsqueeze(1), modulation[1].unsqueeze(1))
    expected_gate = torch.where(selected, modulation[5].unsqueeze(1), modulation[2].unsqueeze(1))

    assert layernorm_scale_shift_gate_select01(x, *modulation, index, 1e-6) is None
    torch.testing.assert_close(expected, direct * (1 + scale) + shift, rtol=0, atol=0)
    torch.testing.assert_close(gate, expected_gate, rtol=0, atol=0)

    residual_output, residual_value, residual_gate_value = residual_layernorm_scale_shift_gate_select01_reference(
        x, residual, residual_gate, *modulation, index, 1e-6
    )
    assert residual_layernorm_scale_shift_gate_select01(
        x, residual, residual_gate, *modulation, index, 1e-6
    ) is None
    torch.testing.assert_close(residual_value, residual + residual_gate * x, rtol=0, atol=0)
    torch.testing.assert_close(
        residual_output,
        F.layer_norm(residual_value, (x.shape[-1],), eps=1e-6) * (1 + scale) + shift,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(residual_gate_value, expected_gate, rtol=0, atol=0)


class _Attention(nn.Module):
    def forward(self, *, hidden_states, encoder_hidden_states, **_kwargs):
        return hidden_states + 0.25, encoder_hidden_states - 0.5


class _Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.zero_cond_t = True
        self.img_mod = nn.Sequential(nn.Identity(), nn.Linear(dim, 6 * dim))
        self.txt_mod = nn.Sequential(nn.Identity(), nn.Linear(dim, 6 * dim))
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.img_mlp = nn.Identity()
        self.txt_mlp = nn.Identity()
        self.attn = _Attention()

    @staticmethod
    def _modulate(x, mod_params, index=None):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if index is None:
            return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)
        batch = x.shape[0]
        selected = index.bool().unsqueeze(-1)
        return (
            x * (1 + torch.where(selected, scale[batch:].unsqueeze(1), scale[:batch].unsqueeze(1)))
            + torch.where(selected, shift[batch:].unsqueeze(1), shift[:batch].unsqueeze(1)),
            torch.where(selected, gate[batch:].unsqueeze(1), gate[:batch].unsqueeze(1)),
        )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        temb,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        modulate_index=None,
    ):
        img_mod1, img_mod2 = self.img_mod(temb).chunk(2, dim=-1)
        txt_mod1, txt_mod2 = self.txt_mod(temb.chunk(2, dim=0)[0]).chunk(2, dim=-1)
        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1, modulate_index)
        txt_modulated, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)
        img_attn, txt_attn = self.attn(
            hidden_states=img_modulated, encoder_hidden_states=txt_modulated
        )
        hidden_states = hidden_states + img_gate1 * img_attn
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn
        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2, modulate_index)
        txt_modulated2, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
        return (
            encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated2),
            hidden_states + img_gate2 * self.img_mlp(img_modulated2),
        )


def test_qwen_block_cpu_fallback_preserves_single_block_semantics():
    torch.manual_seed(3)
    block = _Block(8)
    hidden = torch.randn(1, 4, 8)
    text = torch.randn(1, 3, 8)
    temb = torch.randn(2, 8)
    index = torch.tensor([[0, 1, 0, 1]], dtype=torch.int32)
    expected = block(hidden, text, None, temb, modulate_index=index)

    assert install_qwen_image_modulation_fusions([block]) == 1
    assert install_qwen_image_modulation_fusions([block]) == 0
    actual = block(hidden, text, None, temb, modulate_index=index)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_qwen_modulation_environment_is_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("CHITU_QWEN_IMAGE_MODULATION_FUSION", raising=False)
    assert not _qwen_image_modulation_fusion_enabled()
    monkeypatch.setenv("CHITU_QWEN_IMAGE_MODULATION_FUSION", "benchmark_only")
    assert not _qwen_image_modulation_fusion_enabled()
    monkeypatch.setenv("CHITU_QWEN_IMAGE_MODULATION_FUSION", "select01")
    with pytest.raises(ValueError, match="benchmark-only"):
        _qwen_image_modulation_fusion_enabled()
    monkeypatch.setenv("CHITU_QWEN_IMAGE_MODULATION_FUSION", "unexpected")
    with pytest.raises(ValueError, match="off.*benchmark_only"):
        _qwen_image_modulation_fusion_enabled()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_available(),
    reason="requires CUDA with optional Triton",
)
def test_qwen_modulation_triton_matches_strict_bf16_reference():
    x, residual, residual_gate, *modulation, index = _inputs("cuda", torch.bfloat16)
    reference = layernorm_scale_shift_gate_select01_reference(x, *modulation, index, 1e-6)
    candidate = layernorm_scale_shift_gate_select01(x.contiguous(), *modulation, index.contiguous(), 1e-6)
    assert candidate is not None
    torch.testing.assert_close(candidate[0], reference[0], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(candidate[1], reference[1], rtol=0, atol=0)

    reference_residual = residual_layernorm_scale_shift_gate_select01_reference(
        x, residual, residual_gate, *modulation, index, 1e-6
    )
    candidate_residual = residual_layernorm_scale_shift_gate_select01(
        x.contiguous(), residual.contiguous(), residual_gate.contiguous(), *modulation, index.contiguous(), 1e-6
    )
    assert candidate_residual is not None
    for actual, expected in zip(candidate_residual, reference_residual):
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
