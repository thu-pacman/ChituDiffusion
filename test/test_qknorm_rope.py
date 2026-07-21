from __future__ import annotations

import pytest
import torch

from kernel_adapters.qknorm_rope import fused_inplace


def _rope(sequence: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, 128, 2, device=device) / 128))
    angles = torch.arange(sequence, device=device, dtype=torch.float32)[:, None] * inv_freq
    return torch.repeat_interleave(angles.cos(), 2, dim=-1), torch.repeat_interleave(angles.sin(), 2, dim=-1)


def _reference(
    q: torch.Tensor, k: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.nn.functional.rms_norm(q, (128,), q_weight, 1e-6)
    k = torch.nn.functional.rms_norm(k, (128,), k_weight, 1e-6)
    cos, sin = cos[None, :, None], sin[None, :, None]
    q_even, q_odd = q[..., 0::2], q[..., 1::2]
    k_even, k_odd = k[..., 0::2], k[..., 1::2]
    return (
        torch.stack((q_even * cos[..., 0::2] - q_odd * sin[..., 0::2], q_odd * cos[..., 1::2] + q_even * sin[..., 1::2]), -1)
        .flatten(-2)
        .to(q_weight.dtype),
        torch.stack((k_even * cos[..., 0::2] - k_odd * sin[..., 0::2], k_odd * cos[..., 1::2] + k_even * sin[..., 1::2]), -1)
        .flatten(-2)
        .to(k_weight.dtype),
    )


def test_cpu_is_safe_fallback():
    q = torch.randn(1, 4, 2, 128, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    weight = torch.ones(128, dtype=torch.bfloat16)
    cos, sin = _rope(4, "cpu")
    original_q, original_k = q.clone(), k.clone()
    assert not fused_inplace(q, k, weight, weight, cos, sin, 1e-6)
    torch.testing.assert_close(q, original_q)
    torch.testing.assert_close(k, original_k)


def test_flux2_processor_keeps_diffusers_fallback_semantics():
    from diffusers.models.embeddings import apply_rotary_emb

    from chitu_diffusion.models.model_flux2 import _fused_qknorm_rope_or_fallback

    torch.manual_seed(1)
    q, k = (torch.randn(1, 4, 2, 128) for _ in range(2))
    norm_q, norm_k = torch.nn.RMSNorm(128), torch.nn.RMSNorm(128)
    cos, sin = _rope(4, "cpu")

    expected_q = apply_rotary_emb(norm_q(q), (cos, sin), sequence_dim=1)
    expected_k = apply_rotary_emb(norm_k(k), (cos, sin), sequence_dim=1)
    actual_q, actual_k = _fused_qknorm_rope_or_fallback(q, k, norm_q, norm_k, (cos, sin))
    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)


def test_flux2_qknorm_rope_recipe_defaults_to_auto(monkeypatch):
    from chitu_diffusion.models.model_flux2 import _flux2_qknorm_rope_fusion_enabled

    monkeypatch.delenv("CHITU_FLUX2_FUSED_QKNORM_ROPE", raising=False)
    assert _flux2_qknorm_rope_fusion_enabled()
    monkeypatch.setenv("CHITU_FLUX2_FUSED_QKNORM_ROPE", "off")
    assert not _flux2_qknorm_rope_fusion_enabled()
    monkeypatch.setenv("CHITU_FLUX2_FUSED_QKNORM_ROPE", "invalid")
    with pytest.raises(ValueError, match="auto, on, off"):
        _flux2_qknorm_rope_fusion_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_matches_flux2_rope(dtype):
    torch.manual_seed(0)
    q = torch.randn(1, 257, 24, 128, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    q_weight = torch.randn(128, device="cuda", dtype=dtype)
    k_weight = torch.randn(128, device="cuda", dtype=dtype)
    cos, sin = _rope(257, "cuda")
    expected_q, expected_k = _reference(q, k, q_weight, k_weight, cos, sin)

    with torch.inference_mode():
        assert fused_inplace(q, k, q_weight, k_weight, cos, sin, 1e-6)
    torch.testing.assert_close(q, expected_q, atol=8e-2, rtol=1e-2)
    torch.testing.assert_close(k, expected_k, atol=8e-2, rtol=1e-2)
