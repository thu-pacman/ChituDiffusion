from __future__ import annotations

import pytest
import torch

from chitu_diffusion.parallel.fast_cp.experimental._ring_merge import (
    fused_merge_flash_partials,
    fused_merge_supported,
)


def _eager_merge(
    running_output: torch.Tensor,
    running_lse: torch.Tensor,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    merged_lse = torch.logaddexp(running_lse, partial_lse)
    running_weight = torch.exp(running_lse - merged_lse).transpose(1, 2).unsqueeze(-1)
    partial_weight = torch.exp(partial_lse - merged_lse).transpose(1, 2).unsqueeze(-1)
    merged = running_output * running_weight + partial_output.float() * partial_weight
    return merged, merged_lse


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("batch", "sequence", "heads", "head_dim"),
    [(1, 512, 40, 128), (2, 37, 5, 80), (1, 9, 1, 256)],
)
@torch.inference_mode()
def test_fused_merge_matches_eager(
    dtype: torch.dtype,
    batch: int,
    sequence: int,
    heads: int,
    head_dim: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("the fused ring merge requires CUDA")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260815)
    running_output = torch.randn(
        batch,
        sequence,
        heads,
        head_dim,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    partial_output = torch.randn(
        batch,
        sequence,
        heads,
        head_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    # Ring LSE values span a wide range across phases; the merge must stay
    # stable when one side dominates.
    running_lse = (
        torch.randn(
            batch,
            heads,
            sequence,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 20.0
    )
    partial_lse = (
        torch.randn(
            batch,
            heads,
            sequence,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 20.0
    )

    expected_output, expected_lse = _eager_merge(
        running_output, running_lse, partial_output, partial_lse
    )
    assert fused_merge_supported(
        running_output, running_lse, partial_output, partial_lse
    )
    assert fused_merge_flash_partials(
        running_output, running_lse, partial_output, partial_lse
    )

    torch.testing.assert_close(running_output, expected_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(running_lse, expected_lse, rtol=1e-6, atol=1e-6)


@pytest.mark.gpu
@torch.inference_mode()
def test_fused_merge_repeats_match_eager_chain() -> None:
    """A ring merges once per phase, so only repeats hit the bound launcher."""

    if not torch.cuda.is_available():
        pytest.skip("the fused ring merge requires CUDA")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260817)
    batch, sequence, heads, head_dim, phases = 2, 129, 6, 128, 8
    running_output = torch.randn(
        batch,
        sequence,
        heads,
        head_dim,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    running_lse = (
        torch.randn(
            batch,
            heads,
            sequence,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 20.0
    )
    partials = [
        (
            torch.randn(
                batch,
                sequence,
                heads,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            ),
            torch.randn(
                batch,
                heads,
                sequence,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 20.0,
        )
        for _ in range(phases)
    ]

    expected_output, expected_lse = running_output.clone(), running_lse.clone()
    for partial_output, partial_lse in partials:
        expected_output, expected_lse = _eager_merge(
            expected_output, expected_lse, partial_output, partial_lse
        )
    for partial_output, partial_lse in partials:
        assert fused_merge_flash_partials(
            running_output, running_lse, partial_output, partial_lse
        )

    torch.testing.assert_close(running_output, expected_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(running_lse, expected_lse, rtol=1e-6, atol=1e-6)


@torch.inference_mode()
def test_fused_merge_declines_unsupported_layouts() -> None:
    running_output = torch.randn(1, 4, 2, 8)
    running_lse = torch.randn(1, 2, 4)
    partial_output = torch.randn(1, 4, 2, 8, dtype=torch.float16)
    partial_lse = torch.randn(1, 2, 4)

    # CPU tensors have no Triton path and must fall through to the eager merge.
    assert not fused_merge_flash_partials(
        running_output, running_lse, partial_output, partial_lse
    )
