"""Adapter contract: model semantics/builders, without runtime ownership."""

from dataclasses import asdict, dataclass
from typing import Protocol

import torch


class UnsupportedFastCP(ValueError):
    """A preparation-time capability miss; never a communication exception."""


class AttentionAdapter(Protocol):
    def validate(self, module, inputs, algorithm): ...
    def signature(self): ...
    def build(self, runtime, module, inputs, algorithm, schedule, tag): ...


def validate_common(module, inputs, allowed):
    if set(inputs) - set(allowed):
        raise UnsupportedFastCP(f"unsupported arguments: {set(inputs) - set(allowed)}")
    if module.training:
        raise UnsupportedFastCP("Fast CP supports eval inference only")
    if getattr(module, "fused_projections", False):
        raise UnsupportedFastCP(
            "fused external projections require a dedicated adapter"
        )
    if any(m._forward_hooks or m._forward_pre_hooks for m in module.modules()):
        raise UnsupportedFastCP(
            "projection/attention hooks require a dedicated adapter"
        )
    for name in ("attention_mask", "encoder_hidden_states_mask"):
        if inputs.get(name) is not None:
            raise UnsupportedFastCP(
                "Fast CP currently requires unmasked dense attention"
            )
    x = inputs.get("hidden_states")
    if (
        not isinstance(x, torch.Tensor)
        or x.ndim != 3
        or x.shape[0] != 1
        or not x.is_contiguous()
        or x.shape[1] <= 0
    ):
        raise UnsupportedFastCP("expected nonempty contiguous B1 hidden states")
    for name in ("to_q", "to_k", "to_v"):
        layer = getattr(module, name, None)
        if type(layer) is not torch.nn.Linear:
            raise UnsupportedFastCP(
                "plain replicated Linear projections required (TP1)"
            )
        if (
            layer.in_features != x.shape[-1]
            or layer.out_features != module.to_q.out_features
        ):
            raise UnsupportedFastCP(
                "equal Q/K/V widths and matching input width required"
            )
    if module.to_q.out_features != module.heads * 128:
        raise UnsupportedFastCP("current profile supports head_dim=128 MHA")
    if (
        getattr(module, "is_causal", False)
        or getattr(module, "scale", 128**-0.5) != 128**-0.5
    ):
        raise UnsupportedFastCP(
            "noncausal attention with the standard head-dimension scale required"
        )
    if type(module.to_out[0]) is not torch.nn.Linear:
        raise UnsupportedFastCP("plain replicated output projection required (TP1)")
    for p in module.parameters():
        if p.device != x.device or p.dtype != x.dtype:
            raise UnsupportedFastCP("parameters must share input device and dtype")
    return x


def per_head_norms(module, names):
    for name in names:
        norm = getattr(module, name, None)
        if norm is not None:
            validate_rmsnorm(norm, 128)


def validate_rmsnorm(norm, width):
    from diffusers.models.normalization import RMSNorm

    if (
        type(norm) not in (torch.nn.RMSNorm, RMSNorm)
        or norm.weight is None
        or tuple(norm.weight.shape) != (width,)
        or norm.eps is None
    ):
        raise UnsupportedFastCP(
            "expected affine RMSNorm with explicit epsilon and matching width"
        )


@dataclass
class ProcessorExecution:
    module: object
    processor: object
    metadata: dict
    planner: object = None

    def __call__(self, inputs):
        return self.processor(self.module, **inputs)

    def describe(self):
        metadata = dict(self.metadata)
        if self.planner is not None:
            plan = self.planner.attention_plan
            metadata["attention_plan"] = None if plan is None else asdict(plan)
        return metadata

    def close(self):
        close = getattr(self.processor, "close", None)
        if close is not None:
            close()


def _rope(tensor, tokens):
    if tensor is not None and (
        not tensor.is_complex()
        or tensor.numel() != tokens * 64
        or not tensor.is_contiguous()
    ):
        raise UnsupportedFastCP(
            "complex RoPE must be contiguous with one row per token"
        )
