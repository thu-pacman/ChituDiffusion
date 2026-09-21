"""Numerics and replay safety for fused Qwen QKV layout conversion."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _reference(tensor, qnorm, knorm, rope, heads):
    query, key, value = tensor.chunk(3, -1)
    outputs = []
    for source, norm in ((query, qnorm), (key, knorm)):
        source = source.unflatten(-1, (heads, -1)).float()
        if norm is not None:
            source = (
                source
                * torch.rsqrt(source.square().mean(-1, keepdim=True) + norm.eps)
                * norm.weight.float()
            )
        if rope is not None:
            pairs = source.reshape(*source.shape[:-1], source.shape[-1] // 2, 2)
            source = torch.view_as_real(
                torch.view_as_complex(pairs) * rope[None, :, None, :]
            ).flatten(-2)
        outputs.append(source.to(tensor.dtype))
    return *outputs, value.unflatten(-1, (heads, -1))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("direct_out", [False, True])
@pytest.mark.parametrize(
    "heads,dim,image_tokens,text_tokens",
    [(3, 64, 19, 7), (24, 128, 512, 64), (5, 96, 21, 0)],
)
@pytest.mark.parametrize(
    "with_norm,with_rope", [(True, True), (False, True), (True, False)]
)
@torch.inference_mode()
def test_joint_qkv_epilogue(
    dtype, direct_out, heads, dim, image_tokens, text_tokens, with_norm, with_rope
):
    from chitu_diffusion.parallel.cp.fast.projection.qk_epilogue import (
        joint_qkv_norm_rope,
    )

    torch.manual_seed(187)

    def norm(eps):
        return (
            SimpleNamespace(
                weight=torch.randn(dim, device="cuda", dtype=dtype), eps=eps
            )
            if with_norm
            else None
        )

    attn = SimpleNamespace(
        heads=heads,
        norm_q=norm(1e-6),
        norm_k=norm(1e-5),
        norm_added_q=norm(1e-6),
        norm_added_k=norm(1e-5),
    )
    image = torch.randn(1, image_tokens, 3 * heads * dim, device="cuda", dtype=dtype)
    text = torch.randn(1, text_tokens, 3 * heads * dim, device="cuda", dtype=dtype)

    def rotary(tokens):
        angles = torch.randn(tokens, dim // 2, device="cuda")
        return torch.polar(torch.ones_like(angles), angles) if with_rope else None

    ropes = rotary(image_tokens), rotary(text_tokens)

    def expected():
        image_ref = _reference(image, attn.norm_q, attn.norm_k, ropes[0], heads)
        text_ref = _reference(
            text, attn.norm_added_q, attn.norm_added_k, ropes[1], heads
        )
        return tuple(torch.cat((t, i), 1) for t, i in zip(text_ref, image_ref))

    before = image.clone(), text.clone()
    # Nonzero offsets and guard regions model a rank-local symmetric slot.
    tokens = image_tokens + text_tokens
    arenas = (
        [
            torch.full((1, tokens + 6, heads, dim), 17, device="cuda", dtype=dtype)
            for _ in range(3)
        ]
        if direct_out
        else []
    )
    out = tuple(arena[:, 3:-3] for arena in arenas) if direct_out else None
    actual = joint_qkv_norm_rope(image, text, attn, ropes, out=out)
    if out is not None:
        assert all(a.data_ptr() == b.data_ptr() for a, b in zip(actual, out))
    for index, (a, b) in enumerate(zip(actual, expected())):
        assert a.is_contiguous()
        torch.testing.assert_close(
            a, b, atol=0.002 if index < 2 else 0, rtol=0.016 if index < 2 else 0
        )
    torch.testing.assert_close(image, before[0], atol=0, rtol=0)
    torch.testing.assert_close(text, before[1], atol=0, rtol=0)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        joint_qkv_norm_rope(image, text, attn, ropes, out=out)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        replay_output = joint_qkv_norm_rope(image, text, attn, ropes, out=out)
    image.normal_()
    text.normal_()
    graph.replay()
    torch.cuda.synchronize()
    for index, (a, b) in enumerate(zip(replay_output, expected())):
        torch.testing.assert_close(
            a, b, atol=0.002 if index < 2 else 0, rtol=0.016 if index < 2 else 0
        )
    for arena in arenas:
        assert torch.all(arena[:, :3] == 17)
        assert torch.all(arena[:, -3:] == 17)
