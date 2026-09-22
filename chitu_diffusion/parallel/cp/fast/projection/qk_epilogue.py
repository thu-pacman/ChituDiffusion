"""Fused per-head RMSNorm and interleaved complex RoPE epilogue."""

import torch
import triton
import triton.language as tl


@triton.jit
def _packed_qkv_epilogue(
    Packed,
    QWeight,
    KWeight,
    Rotary,
    Query,
    Key,
    Value,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    OFFSET: tl.constexpr,
    Q_EPS: tl.constexpr,
    K_EPS: tl.constexpr,
    Q_NORM: tl.constexpr,
    K_NORM: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, BLOCK)
    token, head = rows // HEADS, rows % HEADS
    mask = (token[:, None] < TOKENS) & (columns[None, :] < DIM)
    source = token[:, None] * (3 * HEADS * DIM) + head[:, None] * DIM + columns[None, :]
    q = tl.load(Packed + source, mask, 0).to(tl.float32)
    k = tl.load(Packed + source + HEADS * DIM, mask, 0).to(tl.float32)
    v = tl.load(Packed + source + 2 * HEADS * DIM, mask, 0)
    if Q_NORM:
        q *= tl.rsqrt(tl.sum(q * q, 1) / DIM + Q_EPS)[:, None]
        q *= tl.load(QWeight + columns, columns < DIM, 0)[None, :]
    if K_NORM:
        k *= tl.rsqrt(tl.sum(k * k, 1) / DIM + K_EPS)[:, None]
        k *= tl.load(KWeight + columns, columns < DIM, 0)[None, :]
    if HAS_ROPE:
        pair = tl.broadcast_to((columns ^ 1)[None, :], (ROWS, BLOCK))
        paired_q = tl.gather(q, pair, axis=1)
        paired_k = tl.gather(k, pair, axis=1)
        rope = token[:, None] * DIM + (columns[None, :] // 2) * 2
        cosine = tl.load(Rotary + rope, mask, 0)
        sine = tl.load(Rotary + rope + 1, mask, 0)
        sign = tl.where(columns % 2 == 0, -1.0, 1.0)[None, :]
        q = q * cosine + sign * paired_q * sine
        k = k * cosine + sign * paired_k * sine
    target = (rows[:, None] + OFFSET * HEADS) * DIM + columns[None, :]
    tl.store(Query + target, q, mask)
    tl.store(Key + target, k, mask)
    tl.store(Value + target, v, mask)


def joint_qkv_norm_rope(image_qkv, text_qkv, attn, rotary, *, out=None):
    """Fuse QK epilogues and pack text/image into contiguous joint Q/K/V.

    Inputs are contiguous B1 packed projection outputs. Output order is the local
    text shard followed by the local image shard; no input is modified. Optional
    ``out`` supplies three disjoint contiguous BSHD destinations, including local
    views into symmetric K/V buffers. Their lifetime/synchronization belongs to
    the caller.
    """
    heads = attn.heads
    width = image_qkv.shape[-1] // 3
    dim = width // heads
    for tensor in (image_qkv, text_qkv):
        if (
            tensor.ndim != 3
            or tensor.shape[0] != 1
            or tensor.shape[-1] != 3 * width
            or width % heads
            or dim % 2
            or not tensor.is_cuda
            or not tensor.is_contiguous()
            or tensor.device != image_qkv.device
            or tensor.dtype != image_qkv.dtype
            or tensor.dtype not in (torch.float16, torch.bfloat16)
        ):
            raise ValueError(
                "joint QKV epilogue requires contiguous B1 packed CUDA QKV"
            )
    image_rope, text_rope = rotary or (None, None)
    text_tokens = text_qkv.shape[1]
    shape = (1, text_tokens + image_qkv.shape[1], heads, dim)
    if out is None:
        output = torch.empty(
            (3, *shape), device=image_qkv.device, dtype=image_qkv.dtype
        )
        query, key, value = output.unbind(0)
    else:
        if len(out) != 3:
            raise ValueError("joint QKV output requires three destinations")
        for tensor in out:
            if (
                tensor.shape != shape
                or tensor.dtype != image_qkv.dtype
                or tensor.device != image_qkv.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "joint QKV destinations must be contiguous BSHD with matching shape/device/dtype"
                )
        # Contiguous views may share an allocation, but not overlapping ranges.
        spans = [
            (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
            for t in (*out, image_qkv, text_qkv)
        ]
        for index in range(3):
            start, end = spans[index]
            if any(
                start < other_end and other_start < end
                for other_start, other_end in spans[index + 1 :]
            ):
                raise ValueError(
                    "joint QKV destinations must not overlap each other or inputs"
                )
        query, key, value = out
    for tensor, qnorm, knorm, rope, offset in (
        (text_qkv, attn.norm_added_q, attn.norm_added_k, text_rope, 0),
        (image_qkv, attn.norm_q, attn.norm_k, image_rope, text_tokens),
    ):
        tokens = tensor.shape[1]
        if not tokens:
            continue
        for norm in (qnorm, knorm):
            if norm is not None and norm.weight.shape != (dim,):
                raise ValueError("Q/K RMSNorm must have one weight per head channel")
        if rope is not None:
            if not rope.is_complex() or rope.numel() != tokens * dim // 2:
                raise ValueError("complex RoPE must contain one row per token")
            rope = torch.view_as_real(rope).reshape(tokens, dim).float().contiguous()
        _packed_qkv_epilogue[(triton.cdiv(tokens * heads, 4),)](
            tensor,
            qnorm.weight if qnorm is not None else tensor,
            knorm.weight if knorm is not None else tensor,
            rope if rope is not None else tensor,
            query,
            key,
            value,
            TOKENS=tokens,
            HEADS=heads,
            DIM=dim,
            OFFSET=offset,
            Q_EPS=qnorm.eps if qnorm is not None else 0.0,
            K_EPS=knorm.eps if knorm is not None else 0.0,
            Q_NORM=qnorm is not None,
            K_NORM=knorm is not None,
            HAS_ROPE=rope is not None,
            BLOCK=triton.next_power_of_2(dim),
            ROWS=4,
            num_warps=4,
        )
    return query, key, value


@triton.jit
def _qk_norm_rope_inplace(
    Tensor,
    Weight,
    Rotary,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    HAS_NORM: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HEADS
    head = row % HEADS
    columns = tl.arange(0, BLOCK)
    mask = columns < HEAD_DIM
    offset = token * TOKEN_STRIDE + head * HEAD_DIM + columns
    source = tl.load(Tensor + offset, mask=mask, other=0.0).to(tl.float32)
    if HAS_NORM:
        inverse_rms = tl.rsqrt(tl.sum(source * source, axis=0) / HEAD_DIM + EPS)
        normalized = (
            source * inverse_rms * tl.load(Weight + columns, mask=mask, other=0.0)
        )
    else:
        inverse_rms = 1.0
        normalized = source

    if HAS_ROPE:
        pair_columns = tl.where(columns % 2 == 0, columns + 1, columns - 1)
        pair_source = tl.load(
            Tensor + token * TOKEN_STRIDE + head * HEAD_DIM + pair_columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_NORM:
            pair_normalized = (
                pair_source
                * inverse_rms
                * tl.load(Weight + pair_columns, mask=mask, other=0.0)
            )
        else:
            pair_normalized = pair_source
        rope_offset = token * HEAD_DIM + (columns // 2) * 2
        cosine = tl.load(Rotary + rope_offset)
        sine = tl.load(Rotary + rope_offset + 1)
        normalized = tl.where(
            columns % 2 == 0,
            normalized * cosine - pair_normalized * sine,
            pair_normalized * sine + normalized * cosine,
        )

    tl.store(Tensor + offset, normalized, mask=mask)


def qk_norm_rope_inplace(
    tensor: torch.Tensor,
    norm,
    rotary: torch.Tensor | None,
    *,
    heads: int,
) -> torch.Tensor:
    """Apply a Q/K epilogue in place and return a `[B,S,H,D]` view."""
    if not tensor.is_cuda or tensor.stride(-1) != 1:
        raise ValueError("Q/K epilogue requires a CUDA tensor contiguous in D")
    if tensor.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Q/K epilogue requires FP16 or BF16")
    width = tensor.shape[-1]
    if width % heads:
        raise ValueError("Q/K width must be divisible by the head count")
    head_dim = width // heads
    if tensor.ndim not in (2, 3) or (tensor.ndim == 3 and tensor.shape[0] != 1):
        raise ValueError("Q/K epilogue supports [tokens,width] or B1 tensors")
    tokens = tensor.numel() // width
    if rotary is not None:
        if not rotary.is_complex() or rotary.numel() != tokens * head_dim // 2:
            raise ValueError("complex RoPE must contain one table row per token")
        rotary_real = torch.view_as_real(rotary).reshape(tokens, head_dim)
        if not rotary_real.is_contiguous() or rotary_real.dtype != torch.float32:
            rotary_real = rotary_real.float().contiguous()
    else:
        rotary_real = tensor
    if norm is not None and norm.weight.shape != (head_dim,):
        raise ValueError("Q/K RMSNorm must have one weight per head channel")
    _qk_norm_rope_inplace[(tokens * heads,)](
        tensor,
        norm.weight if norm is not None else tensor,
        rotary_real,
        HEADS=heads,
        HEAD_DIM=head_dim,
        TOKEN_STRIDE=tensor.stride(-2),
        EPS=norm.eps if norm is not None else 0.0,
        HAS_NORM=norm is not None,
        HAS_ROPE=rotary is not None,
        BLOCK=triton.next_power_of_2(head_dim),
        num_warps=1,
    )
    return tensor.unflatten(-1, (heads, head_dim))
