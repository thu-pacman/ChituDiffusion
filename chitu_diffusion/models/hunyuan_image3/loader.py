"""Meta construction and shard-local checkpoint loading for Hunyuan Image 3.

The released repository ships the model as Transformers remote code. This module
instantiates that code on the meta device, swaps the decoder stack for the
parallel wrappers, and then streams only this rank's tensor-parallel shards and
expert-parallel slices onto the device. No stage ever holds the full 158 GiB
checkpoint, and ``device_map="auto"`` is never involved.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn

from ...parallel.tp import load_tensor_parallel_checkpoint
from .decoder import ParallelDecoderLayer
from .moe import swiglu_intermediate_size
from .parallel import HunyuanImage3ParallelRuntime
from .sequence import ContextParallelSequenceState

SUPPORTED_BOT_TASKS = ("image",)


@contextmanager
def _default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def checkpoint_expert_index(name: str) -> int | None:
    """Return the routed expert a checkpoint tensor belongs to, if any."""

    parts = name.split(".")
    try:
        position = parts.index("experts")
    except ValueError:
        return None
    return int(parts[position + 1])


@dataclass(frozen=True, slots=True)
class LoadedHunyuanImage3:
    """A rank-local Hunyuan Image 3 model and the released helpers it needs."""

    model: Any
    config: Any
    released: ModuleType
    sequence_state: ContextParallelSequenceState
    device: torch.device
    dtype: torch.dtype

    @property
    def num_hidden_layers(self) -> int:
        return int(self.config.num_hidden_layers)


def _pin_vision_processor_to_tensors(model: nn.Module) -> None:
    """Ask the released SigLIP2 processor for tensors explicitly.

    ``image_processor.preprocess`` squeezes the vision processor's output, which
    only works when the processor returns tensors. Recent Transformers releases
    return lists unless the caller asks, so conditioning images would otherwise
    fail before the parallel decoder ever runs.
    """

    processor = getattr(model, "image_processor", None)
    encoder = getattr(processor, "vision_encoder_processor", None)
    if encoder is None:
        return
    processor.vision_encoder_processor = partial(encoder, return_tensors="pt")


def feed_forward_widths(config: Any) -> tuple[int, ...]:
    """Every SwiGLU half-width the decoder stack will column-shard."""

    widths = {int(config.intermediate_size)}
    for layer_idx in range(int(config.num_hidden_layers)):
        widths.add(swiglu_intermediate_size(config, layer_idx, shared=False))
        if bool(config.use_mixed_mlp_moe):
            widths.add(swiglu_intermediate_size(config, layer_idx, shared=True))
    return tuple(sorted(widths))


def _released_module(model: nn.Module) -> ModuleType:
    module = sys.modules.get(type(model).__module__)
    if module is None:  # pragma: no cover - remote code is always registered
        raise RuntimeError("the released Hunyuan Image 3 module is not importable")
    for symbol in ("apply_rotary_pos_emb", "repeat_kv", "HunyuanRMSNorm"):
        if not hasattr(module, symbol):
            raise RuntimeError(
                f"the released Hunyuan Image 3 module does not expose {symbol}"
            )
    return module


def _rms_norm_factory(released: ModuleType) -> Callable[[int, float], nn.Module]:
    def build(hidden_size: int, eps: float) -> nn.Module:
        return released.HunyuanRMSNorm(hidden_size, eps=eps)

    return build


def _gate_factory(released: ModuleType) -> Callable[[Any, int], nn.Module]:
    def build(config: Any, layer_idx: int) -> nn.Module:
        return released.HunyuanTopKGate(config, layer_idx=layer_idx)

    return build


def build_parallel_decoder_stack(
    model: nn.Module,
    *,
    config: Any,
    released: ModuleType,
    runtime: HunyuanImage3ParallelRuntime,
    dtype: torch.dtype,
) -> ContextParallelSequenceState:
    """Swap every released decoder layer for its parallel counterpart."""

    sequence_state = ContextParallelSequenceState(runtime)
    layer_count = int(config.num_hidden_layers)
    with torch.device("meta"), _default_dtype(dtype):
        layers = nn.ModuleList(
            [
                ParallelDecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                    runtime=runtime,
                    sequence_state=sequence_state,
                    apply_rotary_pos_emb=released.apply_rotary_pos_emb,
                    repeat_kv=released.repeat_kv,
                    rms_norm_factory=_rms_norm_factory(released),
                    gate_factory=_gate_factory(released),
                    dtype=dtype,
                    shards_sequence=layer_idx == 0,
                    gathers_sequence=layer_idx == layer_count - 1,
                )
                for layer_idx in range(layer_count)
            ]
        )
    model.model.layers = layers
    return sequence_state


def load_hunyuan_image3(
    model_path: str | Path,
    *,
    runtime: HunyuanImage3ParallelRuntime,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> LoadedHunyuanImage3:
    """Build and populate this rank's shard of the Hunyuan Image 3 model."""

    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
    )

    root = Path(model_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Hunyuan Image 3 model path not found: {root}")
    config = AutoConfig.from_pretrained(
        root, trust_remote_code=True, local_files_only=True
    )
    config._attn_implementation = "sdpa"
    config.moe_impl = "eager"
    runtime.plan.validate_model_shape(
        num_attention_heads=int(config.num_attention_heads),
        num_key_value_heads=int(config.num_key_value_heads),
        num_experts=int(config.num_experts),
        mlp_widths=feed_forward_widths(config),
    )

    resolved_device = torch.device(
        device
        if device is not None
        else (f"cuda:{runtime.context.local_rank}" if torch.cuda.is_available() else "cpu")
    )
    with _default_dtype(dtype), torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    released = _released_module(model)
    _pin_vision_processor_to_tensors(model)
    sequence_state = build_parallel_decoder_stack(
        model,
        config=config,
        released=released,
        runtime=runtime,
        dtype=dtype,
    )

    def skip(name: str) -> bool:
        expert = checkpoint_expert_index(name)
        return expert is not None and not runtime.owns_checkpoint_expert(expert)

    load_tensor_parallel_checkpoint(
        model,
        root,
        device=resolved_device,
        skip_checkpoint_parameter=skip,
    )
    remaining = [
        name
        for name, value in (*model.named_parameters(), *model.named_buffers())
        if value.is_meta
    ]
    if remaining:
        raise RuntimeError(
            f"{len(remaining)} parameters remain on the meta device, "
            f"starting with {remaining[:5]}"
        )

    model.generation_config = GenerationConfig.from_pretrained(
        root, local_files_only=True
    )
    model.load_tokenizer(
        AutoTokenizer.from_pretrained(
            root, trust_remote_code=True, local_files_only=True
        )
    )
    model.eval()
    model.requires_grad_(False)
    return LoadedHunyuanImage3(
        model=model,
        config=config,
        released=released,
        sequence_state=sequence_state,
        device=resolved_device,
        dtype=dtype,
    )


__all__ = [
    "SUPPORTED_BOT_TASKS",
    "LoadedHunyuanImage3",
    "build_parallel_decoder_stack",
    "checkpoint_expert_index",
    "feed_forward_widths",
    "load_hunyuan_image3",
]
