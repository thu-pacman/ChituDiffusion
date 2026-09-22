import os

import pytest
import torch
import torch.nn.functional as F
from chitu_diffusion.modules.vaes.wan_vae import CausalConv3d

from kernel_adapters.wan_vae_causal_conv3d import (
    cat_pad_stats,
    reset_cat_pad_stats,
    supported,
    triton_available,
    try_causal_conv3d_cat_pad,
)


def _reference(x: torch.Tensor, cache_x: torch.Tensor, padding: tuple[int, ...]) -> torch.Tensor:
    adjusted_padding = list(padding)
    adjusted_padding[4] -= cache_x.shape[2]
    return F.pad(torch.cat([cache_x, x], dim=2), adjusted_padding)


@pytest.mark.parametrize("cache_t", [1, 2])
@pytest.mark.parametrize("spatial_padding", [0, 1])
def test_causal_conv3d_cat_pad_cpu_falls_back_exactly(monkeypatch, cache_t, spatial_padding):
    monkeypatch.setenv("CHITU_WAN_VAE_CAT_PAD", "triton")
    reset_cat_pad_stats()
    x = torch.arange(2 * 3 * 2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 2, 3, 5)
    cache_x = -torch.arange(2 * 3 * cache_t * 3 * 5, dtype=torch.float32).reshape(2, 3, cache_t, 3, 5)
    padding = (spatial_padding, spatial_padding, spatial_padding, spatial_padding, 2, 0)

    assert not supported(x, cache_x, padding)
    assert try_causal_conv3d_cat_pad(x, cache_x, padding) is None
    assert cat_pad_stats().cached_calls == 1
    assert cat_pad_stats().fallback_calls == 1
    assert torch.equal(_reference(x, cache_x, padding), F.pad(torch.cat([cache_x, x], dim=2), [*padding[:4], 2 - cache_t, 0]))


def test_causal_conv3d_no_cache_matches_existing_pad(monkeypatch):
    monkeypatch.setenv("CHITU_WAN_VAE_CAT_PAD", "triton")
    conv = CausalConv3d(2, 3, kernel_size=3, padding=1, bias=False)
    x = torch.randn(1, 2, 2, 4, 5)
    expected = F.conv3d(F.pad(x, conv._padding), conv.weight, None, conv.stride, 0, conv.dilation, conv.groups)

    torch.testing.assert_close(conv(x), expected, rtol=0, atol=0)


def test_causal_conv3d_cached_fallback_preserves_dtype_and_output(monkeypatch):
    monkeypatch.setenv("CHITU_WAN_VAE_CAT_PAD", "off")
    conv = CausalConv3d(2, 3, kernel_size=3, padding=1, bias=False)
    x = torch.randn(1, 2, 1, 4, 5)
    cache_x = torch.randn(1, 2, 2, 4, 5)
    expected = F.conv3d(_reference(x, cache_x, conv._padding), conv.weight, None, conv.stride, 0, conv.dilation, conv.groups)

    actual = conv(x, cache_x)
    assert actual.dtype is x.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not triton_available() or os.getenv("CHITU_RUN_WAN_VAE_CAT_PAD_GPU") != "1",
    reason="requires explicit CHITU_RUN_WAN_VAE_CAT_PAD_GPU=1 on CUDA with Triton",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("cache_t", [1, 2])
@pytest.mark.parametrize("spatial_padding", [0, 1])
def test_causal_conv3d_cat_pad_triton_is_bitwise_exact(monkeypatch, dtype, cache_t, spatial_padding):
    monkeypatch.setenv("CHITU_WAN_VAE_CAT_PAD", "triton")
    torch.manual_seed(17)
    x = torch.randn(1, 8, 2, 7, 9, device="cuda", dtype=dtype)
    cache_x = torch.randn(1, 8, cache_t, 7, 9, device="cuda", dtype=dtype)
    padding = (spatial_padding, spatial_padding, spatial_padding, spatial_padding, 2, 0)

    actual = try_causal_conv3d_cat_pad(x, cache_x, padding)
    assert actual is not None
    assert torch.equal(actual, _reference(x, cache_x, padding))

    conv = CausalConv3d(8, 4, kernel_size=3, padding=1, bias=False).to(device="cuda", dtype=dtype)
    expected_conv = F.conv3d(
        _reference(x, cache_x, conv._padding),
        conv.weight,
        None,
        conv.stride,
        0,
        conv.dilation,
        conv.groups,
    )
    assert torch.equal(conv(x, cache_x), expected_conv)
