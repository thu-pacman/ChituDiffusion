import pytest
import torch
import torch.nn.functional as F

from chitu_diffusion.models.model_flux2 import Flux2SwiGLU
from kernel_adapters.silu_and_mul import can_use_silu_and_mul, silu_and_mul, triton_available


def _reference(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x[..., : x.shape[-1] // 2]) * x[..., x.shape[-1] // 2 :]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_flux2_silu_and_mul_cpu_fallback_preserves_pytorch_dtype_and_value(monkeypatch, dtype):
    monkeypatch.setenv("CHITU_FLUX2_SILU_MUL_FUSION", "benchmark_only")
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16, dtype=dtype)

    assert silu_and_mul(x) is None
    actual = Flux2SwiGLU()(x)

    assert actual.dtype is dtype
    torch.testing.assert_close(actual, _reference(x), rtol=0, atol=0)


def test_flux2_silu_and_mul_noncontiguous_falls_back(monkeypatch):
    monkeypatch.setenv("CHITU_FLUX2_SILU_MUL_FUSION", "benchmark_only")
    x = torch.randn(2, 16, 3, dtype=torch.bfloat16).transpose(1, 2)

    assert not can_use_silu_and_mul(x)
    torch.testing.assert_close(Flux2SwiGLU()(x), _reference(x), rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_available(),
    reason="requires a CUDA environment with optional Triton",
)
def test_flux2_silu_and_mul_triton_bitwise_matches_pytorch_bf16():
    torch.manual_seed(7)
    x = torch.randn(2, 257, 1024, device="cuda", dtype=torch.bfloat16)

    actual = silu_and_mul(x)

    assert actual is not None
    assert actual.dtype is x.dtype
    torch.testing.assert_close(actual, _reference(x), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flux2_silu_and_mul_fp16_declines_to_preserve_exactness():
    x = torch.randn(2, 257, 1024, device="cuda", dtype=torch.float16)
    assert silu_and_mul(x) is None
