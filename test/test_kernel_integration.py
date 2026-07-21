import pytest
import torch

from chitu_diffusion.modules.attention.diffusion_attn_backend import DiffusionAttnBackend
from chitu_diffusion.modules.attention.qwen_image_attention import QwenImageChituAttnProcessor
from chitu_diffusion.modules.attention.z_image_native_norm import (
    _rmsnorm_scale_or_native,
    _rmsnorm_tanh_residual_or_native,
    install_zimage_native_norm_fusions,
)
from chitu_diffusion.runtime.adapter.z_image import _resolve_zimage_norm_fusion_mode
from chitu_diffusion.runtime.parallel_utils import update_out_and_lse
from kernel_adapters.attention_state_merge import (
    merge_attention_states_reference,
    merge_mode,
    triton_merge_available,
    try_merge_attention_states,
)
from kernel_adapters.varlen import build_varlen_metadata, pack_qkv, scatter_padded
from kernel_adapters.zimage_native_norm import rmsnorm_scale, rmsnorm_tanh_residual, triton_available


@pytest.mark.parametrize("alias", ["flash3", "flash_v3", "fa3"])
def test_flash_attention3_aliases(alias):
    assert DiffusionAttnBackend._normalize_attn_type(alias) == "flash_attn3"


def test_flash_attention3_is_not_selected_without_supported_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert not DiffusionAttnBackend._is_available("flash_attn3")


def test_auto_prefers_exact_sdpa_baseline():
    assert DiffusionAttnBackend("auto").impl == "torch_sdpa"


def test_cp_attention_state_merge_matches_existing_online_softmax(monkeypatch):
    monkeypatch.setenv("CHITU_CP_ATTN_STATE_MERGE", "off")
    torch.manual_seed(7)
    out = torch.randn(2, 5, 3, 8, dtype=torch.float32)
    lse = torch.randn(2, 5, 3, 1, dtype=torch.float32)
    block_out = torch.randn(2, 5, 3, 8, dtype=torch.bfloat16)
    block_lse = torch.randn(2, 3, 5, dtype=torch.float32)

    expected_out, expected_lse = merge_attention_states_reference(out, lse, block_out, block_lse)
    actual_out, actual_lse = update_out_and_lse(out.clone(), lse.clone(), block_out, block_lse)

    torch.testing.assert_close(actual_out, expected_out, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=2e-6, atol=2e-6)


def test_cp_attention_state_merge_empty_lse_reference_preserves_state():
    out = torch.randn(1, 2, 2, 4, dtype=torch.float32)
    lse = torch.full((1, 2, 2, 1), -torch.inf, dtype=torch.float32)
    block_out = torch.randn(1, 2, 2, 4, dtype=torch.bfloat16)
    block_lse = torch.full((1, 2, 2), -torch.inf, dtype=torch.float32)

    merged_out, merged_lse = merge_attention_states_reference(out, lse, block_out, block_lse)
    torch.testing.assert_close(merged_out, out, rtol=0, atol=0)
    assert torch.isneginf(merged_lse).all()


def test_cp_attention_state_merge_mode_is_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("CHITU_CP_ATTN_STATE_MERGE", raising=False)
    assert merge_mode() == "off"
    monkeypatch.setenv("CHITU_CP_ATTN_STATE_MERGE", "benchmark_only")
    assert merge_mode() == "benchmark_only"
    monkeypatch.setenv("CHITU_CP_ATTN_STATE_MERGE", "triton")
    with pytest.raises(ValueError, match="disabled"):
        merge_mode()
    monkeypatch.setenv("CHITU_CP_ATTN_STATE_MERGE", "auto")
    with pytest.raises(ValueError, match="off, benchmark_only"):
        merge_mode()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_merge_available(),
    reason="requires CUDA and optional Triton",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cp_attention_state_merge_triton_matches_float32_reference(monkeypatch, dtype):
    monkeypatch.setenv("CHITU_CP_ATTN_STATE_MERGE", "benchmark_only")
    torch.manual_seed(11)
    out = torch.randn(2, 257, 8, 128, device="cuda", dtype=torch.float32)
    # Match the real CP state: the first block LSE arrives as [B,H,S] and is
    # retained as a non-contiguous [B,S,H,1] transpose for subsequent merges.
    lse = torch.randn(2, 8, 257, device="cuda", dtype=torch.float32).transpose(-2, -1).unsqueeze(-1)
    assert not lse.is_contiguous()
    block_out = torch.randn(2, 257, 8, 128, device="cuda", dtype=dtype)
    block_lse = torch.randn(2, 8, 257, device="cuda", dtype=torch.float32)
    reference_out, reference_lse = merge_attention_states_reference(out, lse, block_out, block_lse)

    assert try_merge_attention_states(
        out, lse, block_out, block_lse, benchmark=True
    )
    torch.testing.assert_close(out, reference_out, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(lse, reference_lse, rtol=2e-5, atol=2e-5)


def test_varlen_metadata_pack_and_scatter_cpu_fallback():
    mask = torch.tensor([[True, False, True], [False, True, True]])
    metadata = build_varlen_metadata(mask)
    q = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).view(2, 3, 2, 4)
    packed_q, packed_k, packed_v = pack_qkv(q, q + 1, q + 2, metadata)

    assert metadata.cu_seqlens.tolist() == [0, 2, 4]
    assert torch.equal(packed_q, q.reshape(-1, 2, 4)[metadata.indices])
    assert torch.equal(packed_k, packed_q + 1)
    assert torch.equal(packed_v, packed_q + 2)
    scattered = scatter_padded(packed_q, metadata)
    assert torch.equal(scattered[mask], q[mask])
    assert torch.count_nonzero(scattered[~mask]) == 0


def test_torch_sdpa_padded_varlen_masks_keys_and_zeros_invalid_queries():
    torch.manual_seed(0)
    backend = DiffusionAttnBackend("torch_sdpa")
    q = torch.randn(2, 4, 2, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    valid = torch.tensor([[True, False, True, True], [True, True, False, True]])

    output = backend.padded_varlen_attention(q, k, v, valid)[0]
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=valid[:, None, None, :],
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)
    reference = reference.masked_fill(~valid[:, :, None, None], 0)
    torch.testing.assert_close(output, reference)


def test_qwen_joint_mask_requires_bool_padded_layout():
    valid = torch.tensor([[True, False, True, True]])
    QwenImageChituAttnProcessor._validate_joint_valid_mask(
        valid,
        batch_size=1,
        text_sequence_length=2,
        image_sequence_length=2,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="bool valid mask"):
        QwenImageChituAttnProcessor._validate_joint_valid_mask(
            valid.to(torch.int64),
            batch_size=1,
            text_sequence_length=2,
            image_sequence_length=2,
            device=torch.device("cpu"),
        )


class _RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(dim, dtype=torch.bfloat16))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().square().mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).to(self.weight.dtype) * self.weight


def test_zimage_native_norm_cpu_returns_none():
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    weight = torch.ones(16, dtype=torch.bfloat16)
    assert rmsnorm_scale(x, weight, torch.ones(2, 1, 16, dtype=torch.bfloat16), 1e-6) is None
    assert rmsnorm_tanh_residual(x, x[:, :1], x, weight, 1e-6) is None


def test_zimage_norm_fusion_mode_environment(monkeypatch):
    monkeypatch.delenv("CHITU_Z_IMAGE_NORM_FUSION", raising=False)
    monkeypatch.delenv("CHITU_Z_IMAGE_NATIVE_NORM", raising=False)
    assert _resolve_zimage_norm_fusion_mode() == ("diffusers_fp32", False)

    monkeypatch.setenv("CHITU_Z_IMAGE_NORM_FUSION", "auto")
    assert _resolve_zimage_norm_fusion_mode() == ("diffusers_fp32", False)

    monkeypatch.setenv("CHITU_Z_IMAGE_NORM_FUSION", "diffusers_fp32")
    assert _resolve_zimage_norm_fusion_mode() == ("diffusers_fp32", False)

    monkeypatch.delenv("CHITU_Z_IMAGE_NORM_FUSION")
    monkeypatch.setenv("CHITU_Z_IMAGE_NATIVE_NORM", "1")
    assert _resolve_zimage_norm_fusion_mode() == ("native_bf16", True)

    monkeypatch.setenv("CHITU_Z_IMAGE_NORM_FUSION", "invalid")
    with pytest.raises(ValueError, match="auto, off, diffusers_fp32, native_bf16"):
        _resolve_zimage_norm_fusion_mode()


def test_zimage_native_norm_cpu_fallback_matches_diffusers_expression():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    scale = torch.randn(2, 1, 16, dtype=torch.bfloat16)
    gate = torch.randn(2, 1, 16, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    norm = _RMSNorm(16)

    expected_scale = norm(x) * scale
    actual_scale = _rmsnorm_scale_or_native(x, scale, norm)
    expected_residual = residual + torch.tanh(gate) * norm(x)
    actual_residual = _rmsnorm_tanh_residual_or_native(x, gate, residual, norm)

    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)


def test_zimage_native_bf16_cpu_fallback_uses_native_reduction():
    torch.manual_seed(2)
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    scale = torch.randn(2, 1, 16, dtype=torch.bfloat16)
    gate = torch.randn(2, 1, 16, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    norm = _RMSNorm(16)
    mean_square = x.square().to(torch.bfloat16).mean(dim=-1, keepdim=True).to(torch.bfloat16)
    native_norm = (
        (x * torch.rsqrt((mean_square + norm.eps).float()).to(torch.bfloat16)).to(torch.bfloat16)
        * norm.weight
    )

    torch.testing.assert_close(
        _rmsnorm_scale_or_native(x, scale, norm, "native_bf16"),
        native_norm * scale,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        _rmsnorm_tanh_residual_or_native(x, gate, residual, norm, "native_bf16"),
        residual + torch.tanh(gate) * native_norm,
        rtol=0,
        atol=0,
    )


def test_zimage_native_norm_installer_is_idempotent():
    block = torch.nn.Module()
    block.attention = torch.nn.Identity()
    block.feed_forward = torch.nn.Identity()
    block.attention_norm1 = _RMSNorm(16)
    block.attention_norm2 = _RMSNorm(16)
    block.ffn_norm1 = _RMSNorm(16)
    block.ffn_norm2 = _RMSNorm(16)
    block.modulation = False

    assert install_zimage_native_norm_fusions([block]) == 1
    assert install_zimage_native_norm_fusions([block]) == 0


class _Attention(torch.nn.Module):
    def forward(self, x, **_kwargs):
        return x


class _FixedModulation(torch.nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("value", value)

    def forward(self, _x):
        return self.value


def test_zimage_block_applies_gate_tanh_exactly_once():
    torch.manual_seed(1)
    dim = 8
    block = torch.nn.Module()
    block.attention = _Attention()
    block.feed_forward = torch.nn.Identity()
    block.attention_norm1 = _RMSNorm(dim)
    block.attention_norm2 = _RMSNorm(dim)
    block.ffn_norm1 = _RMSNorm(dim)
    block.ffn_norm2 = _RMSNorm(dim)
    block.modulation = True
    modulation = torch.randn(1, 4 * dim, dtype=torch.bfloat16)
    block.adaLN_modulation = _FixedModulation(modulation)
    x = torch.randn(1, 3, dim, dtype=torch.bfloat16)

    scale_msa, gate_msa, scale_mlp, gate_mlp = modulation.unsqueeze(1).chunk(4, dim=2)
    attn_out = block.attention(block.attention_norm1(x) * (1 + scale_msa))
    expected = x + torch.tanh(gate_msa) * block.attention_norm2(attn_out)
    ffn_out = block.feed_forward(block.ffn_norm1(expected) * (1 + scale_mlp))
    expected = expected + torch.tanh(gate_mlp) * block.ffn_norm2(ffn_out)

    assert install_zimage_native_norm_fusions([block], mode="diffusers_fp32") == 1
    actual = block(x, attn_mask=None, freqs_cis=None, adaln_input=torch.zeros(1, dim))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_available(),
    reason="requires a CUDA environment with optional Triton",
)
@pytest.mark.parametrize("mode", ["diffusers_fp32", "native_bf16"])
def test_zimage_native_norm_triton_matches_reference(mode):
    torch.manual_seed(0)
    x = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(2, 1, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(2, 1, 128, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    eps = 1e-6
    if mode == "diffusers_fp32":
        mean_square = x.float().square().mean(dim=-1, keepdim=True)
        norm = (x * torch.rsqrt(mean_square + eps)).to(torch.bfloat16) * weight
    else:
        mean_square = x.square().to(torch.bfloat16).mean(dim=-1, keepdim=True).to(torch.bfloat16)
        norm = (x * torch.rsqrt((mean_square + eps).float()).to(torch.bfloat16)).to(torch.bfloat16) * weight

    scale_output = rmsnorm_scale(x, weight, scale, eps, mode=mode)
    residual_output = rmsnorm_tanh_residual(x, gate, residual, weight, eps, mode=mode)

    assert scale_output is not None
    assert residual_output is not None
    torch.testing.assert_close(scale_output, (norm * scale).to(torch.bfloat16), rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(
        residual_output,
        (residual + torch.tanh(gate).to(torch.bfloat16) * norm).to(torch.bfloat16),
        rtol=2e-2,
        atol=5e-2,
    )
