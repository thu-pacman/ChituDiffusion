"""Prepared Wan QKV-to-output block with direct-slot KV projection and CE."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from .....models.wan.attention import _apply_rotary
from ._execution import PreparedAgkv
from .base import (
    ProcessorExecution,
    UnsupportedFastCP,
    validate_common,
    validate_rmsnorm,
)


@triton.jit
def _wan_key_norm_rope_inplace(
    Key,
    Weight,
    Cos,
    Sin,
    WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EPS: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < WIDTH
    source = tl.load(Key + row * WIDTH + columns, mask=mask, other=0.0).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(source * source, axis=0) / WIDTH + EPS)
    normalized = source * inverse_rms * tl.load(Weight + columns, mask=mask, other=0.0)
    if HAS_ROPE:
        head_column = columns % HEAD_DIM
        pair_columns = tl.where(head_column % 2 == 0, columns + 1, columns - 1)
        pair_source = tl.load(
            Key + row * WIDTH + pair_columns, mask=mask, other=0.0
        ).to(tl.float32)
        pair_normalized = (
            pair_source
            * inverse_rms
            * tl.load(Weight + pair_columns, mask=mask, other=0.0)
        )
        rope_offset = row * HEAD_DIM + head_column
        cosine = tl.load(Cos + rope_offset)
        sine = tl.load(Sin + rope_offset)
        normalized = tl.where(
            head_column % 2 == 0,
            normalized * cosine - pair_normalized * sine,
            pair_normalized * sine + normalized * cosine,
        )
    tl.store(Key + row * WIDTH + columns, normalized, mask=mask)


def normalize_rotary(attn, tensor, rotary, *, query):
    norm = attn.norm_q if query else attn.norm_k
    result = norm(tensor).unflatten(-1, (attn.heads, -1))
    return _apply_rotary(result, *rotary) if rotary is not None else result


class PreparedWanAgkv(PreparedAgkv):
    """Opt-in Wan attention processor for fixed buffers and frozen weights.

    Prepare collectively on the runtime lane, under inference_mode(). B1,
    uniform CP2/4/8, SM90 and FP16/BF16 only. RoPE tables and weights are frozen
    for this slot; recreate after changing either. Hidden states may change in
    place. Output is reusable scratch. Close before destroying the runtime.
    """

    _supports_local_key_postprocess = True

    def _gather_rotary(self, rotary_emb, width):
        if rotary_emb is None:
            return None
        pieces = []
        for tensor in rotary_emb:
            if tensor.shape != (
                1,
                self.hidden_states.shape[1],
                1,
                width // self.attn.heads,
            ):
                raise ValueError("Wan RoPE must match the local sequence shard")
            gathered = [torch.empty_like(tensor) for _ in range(self.projection.world)]
            dist.all_gather(gathered, tensor.contiguous(), group=self.projection.pg)
            pieces.append(torch.cat(gathered, dim=1))
        return tuple(pieces)

    def _head(self):
        return (
            normalize_rotary(
                self.attn,
                self.attn.to_q(self.hidden_states),
                self.rotary_emb,
                query=True,
            ),
        )

    def _postprocess_local_key(self, raw_key):
        width = self.attn.to_k.out_features
        head_dim = width // self.attn.heads
        rotary = self.rotary_emb
        _wan_key_norm_rope_inplace[(raw_key.shape[0],)](
            raw_key,
            self.attn.norm_k.weight,
            rotary[0] if rotary is not None else raw_key,
            rotary[1] if rotary is not None else raw_key,
            WIDTH=width,
            HEAD_DIM=head_dim,
            EPS=self.attn.norm_k.eps,
            HAS_ROPE=rotary is not None,
            BLOCK=triton.next_power_of_2(width),
            num_warps=8,
        )

    def _key_value(self, raw_key, value):
        if self.local_key_postprocess:
            return raw_key, value
        return normalize_rotary(
            self.attn, raw_key.flatten(2), self.full_rotary, query=False
        ), value


@dataclass(frozen=True)
class WanAdapter:
    def signature(self):
        return ("wan", "sequence_sharded", "full_width_qk_norm", 1)

    def validate(self, module, inputs, algorithm):
        x = validate_common(
            module,
            inputs,
            {"hidden_states", "rotary_emb", "encoder_hidden_states", "attention_mask"},
        )
        if inputs.get("encoder_hidden_states") is not None:
            raise UnsupportedFastCP("Wan wrapper supports self-attention only")
        for norm in (module.norm_q, module.norm_k):
            validate_rmsnorm(norm, module.heads * 128)
        rope = inputs.get("rotary_emb")
        if rope is not None and (
            len(rope) != 2
            or any(
                tuple(t.shape) != (1, x.shape[1], 1, 128) or not t.is_contiguous()
                for t in rope
            )
        ):
            raise UnsupportedFastCP("Wan RoPE requires local B,S,1,D cos/sin")

    def build(self, runtime, module, inputs, algorithm, schedule, tag):
        if algorithm == "agkv":
            if schedule not in {"auto", "overlap"}:
                raise UnsupportedFastCP("Wan currently provides the overlap template")
            p = PreparedWanAgkv(
                runtime.backend,
                module,
                inputs["hidden_states"],
                inputs.get("rotary_emb"),
                tag=tag,
                remote_streams=runtime.profile.remote_streams,
                peer_order=runtime.profile.peer_order,
            )
            return ProcessorExecution(module, p, dict(p.metadata, schedule="overlap"))
        from .....models.wan.attention import WanCpAttnProcessor

        p = WanCpAttnProcessor(
            runtime.parallel(tag), mode="ulysses", planned_attention=True
        )
        return ProcessorExecution(
            module,
            p,
            {"schedule": "serial", "transport": "nvshmem"},
            planner=p.attention,
        )
