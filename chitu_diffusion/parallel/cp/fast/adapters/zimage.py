"""Prepared Z-Image joint attention with direct-slot image K/V gather."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .....models.zimage.attention import ZImageCpAttnProcessor
from ..projection.qk_epilogue import qk_norm_rope_inplace
from ._execution import PreparedAgkv
from .base import (
    ProcessorExecution,
    UnsupportedFastCP,
    _rope,
    per_head_norms,
    validate_common,
)


def zimage_normalize_rotary(attn, tensor, rotary, *, query):
    result = tensor.unflatten(-1, (attn.heads, -1))
    norm = attn.norm_q if query else attn.norm_k
    if norm is not None:
        result = norm(result)
    if rotary is not None:
        result = ZImageCpAttnProcessor._apply_rotary(result, rotary)
    return result.to(tensor.dtype).contiguous()


class PreparedZImageAgkv(PreparedAgkv):
    """Fixed-storage Z-Image image-sharded, text-replicated joint attention."""

    _supports_local_key_postprocess = True

    def __init__(
        self,
        runtime,
        attn,
        hidden_states,
        image_tokens,
        freqs_cis,
        **kwargs,
    ):
        if not 0 < image_tokens < hidden_states.shape[1]:
            raise ValueError(
                "Z-Image AGKV requires image tokens followed by text tokens"
            )
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError("Z-Image AGKV requires contiguous B1 hidden states")
        self.full_hidden_states = hidden_states
        self.image_tokens = int(image_tokens)
        self.joint_states = hidden_states[:, image_tokens:]
        self.full_freqs_cis = freqs_cis
        image_freqs = None if freqs_cis is None else freqs_cis[:, :image_tokens]
        self.joint_freqs = None if freqs_cis is None else freqs_cis[:, image_tokens:]
        projections = (attn.to_q, attn.to_k, attn.to_v)
        self.joint_weight = torch.cat(
            [projection.weight for projection in projections]
        ).contiguous()
        biases = [projection.bias for projection in projections]
        if any(bias is None for bias in biases) != all(bias is None for bias in biases):
            raise ValueError("Z-Image text Q/K/V must agree on bias presence")
        self.joint_bias = None if biases[0] is None else torch.cat(biases).contiguous()
        super().__init__(
            runtime,
            attn,
            hidden_states[:, :image_tokens],
            image_freqs,
            **kwargs,
        )
        self.metadata["joint_tokens"] = self.joint_states.shape[1]
        self.metadata["normalization"] = (
            "local image K and Q/text per-head RMSNorm+RoPE epilogues"
        )
        self.metadata["text_qkv_gemm_launches"] = 1

    def _gather_rotary(self, rotary_emb, width):
        del width
        if rotary_emb is None:
            return None
        gathered = [torch.empty_like(rotary_emb) for _ in range(self.projection.world)]
        dist.all_gather(gathered, rotary_emb.contiguous(), group=self.projection.pg)
        return torch.cat(gathered, dim=1)

    def _head(self):
        attn = self.attn
        image_query = qk_norm_rope_inplace(
            attn.to_q(self.hidden_states),
            attn.norm_q,
            self.rotary_emb,
            heads=attn.heads,
        )
        joint_qkv = F.linear(self.joint_states, self.joint_weight, self.joint_bias)
        joint_query, joint_key, joint_value = joint_qkv.chunk(3, dim=-1)
        joint_query = qk_norm_rope_inplace(
            joint_query,
            attn.norm_q,
            self.joint_freqs,
            heads=attn.heads,
        )
        joint_key = qk_norm_rope_inplace(
            joint_key,
            attn.norm_k,
            self.joint_freqs,
            heads=attn.heads,
        )
        joint_value = joint_value.unflatten(-1, (attn.heads, -1))
        return torch.cat([image_query, joint_query], dim=1), joint_key, joint_value

    def _postprocess_local_key(self, raw_key):
        qk_norm_rope_inplace(
            raw_key,
            self.attn.norm_k,
            self.rotary_emb,
            heads=self.attn.heads,
        )

    def _key_value(self, raw_key, value):
        image_key = (
            raw_key
            if self.local_key_postprocess
            else zimage_normalize_rotary(
                self.attn,
                raw_key.flatten(2),
                self.full_rotary,
                query=False,
            )
        )
        return torch.cat([image_key, self.head[1]], dim=1), torch.cat(
            [value, self.head[2]], dim=1
        )

    def _finish(self, output):
        output = self.attn.to_out[0](output.flatten(2).contiguous())
        return self.attn.to_out[1](output) if len(self.attn.to_out) > 1 else output

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        freqs_cis=None,
        *,
        overlap=True,
        overlap_q=True,
    ):
        if encoder_hidden_states is not None or attention_mask is not None:
            raise ValueError(
                "prepared Z-Image AGKV supports unmasked self-attention only"
            )
        if (
            hidden_states.shape != self.full_hidden_states.shape
            or hidden_states.stride() != self.full_hidden_states.stride()
            or hidden_states.data_ptr() != self.full_hidden_states.data_ptr()
        ):
            raise ValueError("prepare another Z-Image AGKV slot for different storage")
        if (freqs_cis is None) != (self.full_freqs_cis is None) or (
            freqs_cis is not None
            and (
                freqs_cis.shape != self.full_freqs_cis.shape
                or freqs_cis.data_ptr() != self.full_freqs_cis.data_ptr()
            )
        ):
            raise ValueError("prepare another Z-Image AGKV slot for different RoPE")
        return super().__call__(
            attn,
            hidden_states[:, : self.image_tokens],
            rotary_emb=(
                None if freqs_cis is None else freqs_cis[:, : self.image_tokens]
            ),
            overlap=overlap,
            overlap_q=overlap_q,
        )


class PreparedZImageShardedAgkv(PreparedAgkv):
    """Fixed-storage Z-Image attention with image and text sequence-sharded."""

    _supports_local_key_postprocess = True

    def __init__(self, runtime, attn, hidden_states, freqs_cis, **kwargs):
        self.full_hidden_states = hidden_states
        self.full_freqs_cis = freqs_cis
        super().__init__(runtime, attn, hidden_states, freqs_cis, **kwargs)
        self.metadata["partition"] = "image and text sequence-sharded"
        self.metadata["normalization"] = (
            "local per-head RMSNorm+RoPE before direct-slot CE send"
        )

    def _gather_rotary(self, rotary_emb, width):
        del width
        if rotary_emb is None:
            return None
        gathered = [torch.empty_like(rotary_emb) for _ in range(self.projection.world)]
        dist.all_gather(gathered, rotary_emb.contiguous(), group=self.projection.pg)
        return torch.cat(gathered, dim=1)

    def _head(self):
        return (
            qk_norm_rope_inplace(
                self.attn.to_q(self.hidden_states),
                self.attn.norm_q,
                self.rotary_emb,
                heads=self.attn.heads,
            ),
        )

    def _postprocess_local_key(self, raw_key):
        qk_norm_rope_inplace(
            raw_key,
            self.attn.norm_k,
            self.rotary_emb,
            heads=self.attn.heads,
        )

    def _key_value(self, raw_key, value):
        key = (
            raw_key
            if self.local_key_postprocess
            else zimage_normalize_rotary(
                self.attn,
                raw_key.flatten(2),
                self.full_rotary,
                query=False,
            )
        )
        return key, value

    def _finish(self, output):
        output = self.attn.to_out[0](output.flatten(2).contiguous())
        return self.attn.to_out[1](output) if len(self.attn.to_out) > 1 else output

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        freqs_cis=None,
        *,
        overlap=True,
        overlap_q=True,
    ):
        if encoder_hidden_states is not None or attention_mask is not None:
            raise ValueError(
                "prepared Z-Image AGKV supports unmasked self-attention only"
            )
        if (
            hidden_states.shape != self.full_hidden_states.shape
            or hidden_states.stride() != self.full_hidden_states.stride()
            or hidden_states.data_ptr() != self.full_hidden_states.data_ptr()
        ):
            raise ValueError("prepare another Z-Image AGKV slot for different storage")
        if (freqs_cis is None) != (self.full_freqs_cis is None) or (
            freqs_cis is not None
            and (
                freqs_cis.shape != self.full_freqs_cis.shape
                or freqs_cis.data_ptr() != self.full_freqs_cis.data_ptr()
            )
        ):
            raise ValueError("prepare another Z-Image AGKV slot for different RoPE")
        return super().__call__(
            attn,
            hidden_states,
            rotary_emb=freqs_cis,
            overlap=overlap,
            overlap_q=overlap_q,
        )


@dataclass(frozen=True)
class ZImageAdapter:
    image_tokens: int
    text_distribution: str = "sharded"

    def signature(self):
        return (
            "zimage",
            self.image_tokens,
            self.text_distribution,
            "image_first",
            "per_head_norm",
            1,
        )

    def validate(self, module, inputs, algorithm):
        x = validate_common(
            module,
            inputs,
            {"hidden_states", "freqs_cis", "attention_mask", "encoder_hidden_states"},
        )
        if inputs.get("encoder_hidden_states") is not None:
            raise UnsupportedFastCP(
                "ZImage expects text already appended to hidden_states"
            )
        if (
            self.text_distribution not in {"sharded", "replicated"}
            or not 0 < self.image_tokens < x.shape[1]
        ):
            raise UnsupportedFastCP(
                "ZImage requires image token count and explicit text distribution"
            )
        per_head_norms(module, ("norm_q", "norm_k"))
        _rope(inputs.get("freqs_cis"), x.shape[1])

    def build(self, runtime, module, inputs, algorithm, schedule, tag):
        if algorithm == "agkv":
            if schedule not in {"auto", "overlap"}:
                raise UnsupportedFastCP(
                    "ZImage currently provides the overlap template"
                )
            args = [runtime.backend, module, inputs["hidden_states"]]
            cls = PreparedZImageShardedAgkv
            if self.text_distribution == "replicated":
                cls = PreparedZImageAgkv
                args.append(self.image_tokens)
            args.append(inputs.get("freqs_cis"))
            p = cls(
                *args,
                tag=tag,
                remote_streams=runtime.profile.remote_streams,
                peer_order=runtime.profile.peer_order,
            )
            return ProcessorExecution(module, p, dict(p.metadata, schedule="overlap"))
        from .....models.zimage.attention import ZImageCpAttnProcessor

        p = ZImageCpAttnProcessor(
            runtime.parallel(tag),
            mode="ulysses",
            planned_attention=True,
            sharded_text=self.text_distribution == "sharded",
        )
        p.enable(self.image_tokens)
        return ProcessorExecution(
            module,
            p,
            {"schedule": "serial", "text_distribution": self.text_distribution},
            planner=p.context_parallel,
        )
