# SPDX-License-Identifier: Apache-2.0
"""CUDA correctness gates for Qwen-Image's single-device padded-varlen path."""

import pytest
import torch

from chitu_diffusion.modules.attention.diffusion_attn_backend import DiffusionAttnBackend
from kernel_adapters.varlen import build_varlen_metadata, pack_qkv, scatter_padded


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_pack_scatter_matches_pytorch(dtype):
    torch.manual_seed(0)
    mask = torch.tensor(
        [[True, False, True, True, False], [False, True, True, False, True]],
        device="cuda",
    )
    metadata = build_varlen_metadata(mask)
    q = torch.randn(2, 5, 4, 32, device="cuda", dtype=dtype)
    packed_q, packed_k, packed_v = pack_qkv(q, q + 1, q + 2, metadata)

    expected = q.reshape(-1, 4, 32).index_select(0, metadata.indices)
    assert torch.equal(packed_q, expected)
    assert torch.equal(packed_k, expected + 1)
    assert torch.equal(packed_v, expected + 2)
    scattered = scatter_padded(packed_q, metadata)
    assert torch.equal(scattered[mask], q[mask])
    assert torch.count_nonzero(scattered[~mask]) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_torch_sdpa_masked_fallback_matches_reference(dtype):
    torch.manual_seed(1)
    q = torch.randn(2, 7, 4, 32, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    valid = torch.tensor(
        [[True, False, True, True, False, True, True], [True, True, False, True, True, False, True]],
        device="cuda",
    )
    output = DiffusionAttnBackend("torch_sdpa").padded_varlen_attention(q, k, v, valid)[0]
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=valid[:, None, None, :],
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)
    expected.masked_fill_(~valid[:, :, None, None], 0)
    torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)
