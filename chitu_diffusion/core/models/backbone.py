from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch


@dataclass
class BackboneBlockInfo:
    index: int
    name: str
    module: torch.nn.Module
    kind: str = "transformer"


class BackboneState(dict):
    """Dictionary state with tensor helpers used by block-level schedulers."""

    def tensor(self) -> torch.Tensor:
        if "x" in self:
            return self["x"]
        if "hidden_states" in self:
            return self["hidden_states"]
        raise KeyError("BackboneState does not expose a cache tensor.")

    def with_tensor(self, tensor: torch.Tensor) -> "BackboneState":
        state = BackboneState(self)
        if "x" in state:
            state["x"] = tensor
        elif "hidden_states" in state:
            state["hidden_states"] = tensor
            text_seq_len = state.get("text_seq_len")
            if text_seq_len is not None and tensor.shape[1] > int(text_seq_len):
                state["single_stream_started"] = True
        else:
            raise KeyError("BackboneState does not expose a writable cache tensor.")
        return state


def detach_backbone_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: detach_backbone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detach_backbone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_backbone_value(item) for item in value)
    return value


def add_backbone_values(left: Any, right: Any) -> Any:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left + right
    if isinstance(left, dict) and isinstance(right, dict):
        return {key: add_backbone_values(left[key], right[key]) for key in left.keys()}
    if isinstance(left, list) and isinstance(right, list):
        return [add_backbone_values(a, b) for a, b in zip(left, right)]
    if isinstance(left, tuple) and isinstance(right, tuple):
        return tuple(add_backbone_values(a, b) for a, b in zip(left, right))
    raise TypeError(f"Cannot add backbone values of type {type(left).__name__} and {type(right).__name__}.")


def sub_backbone_values(left: Any, right: Any) -> Any:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left - right
    if isinstance(left, dict) and isinstance(right, dict):
        return {key: sub_backbone_values(left[key], right[key]) for key in left.keys()}
    if isinstance(left, list) and isinstance(right, list):
        return [sub_backbone_values(a, b) for a, b in zip(left, right)]
    if isinstance(left, tuple) and isinstance(right, tuple):
        return tuple(sub_backbone_values(a, b) for a, b in zip(left, right))
    raise TypeError(f"Cannot subtract backbone values of type {type(left).__name__} and {type(right).__name__}.")


def scale_backbone_value(value: Any, scale: float) -> Any:
    if torch.is_tensor(value):
        return value * scale
    if isinstance(value, dict):
        return {key: scale_backbone_value(item, scale) for key, item in value.items()}
    if isinstance(value, list):
        return [scale_backbone_value(item, scale) for item in value]
    if isinstance(value, tuple):
        return tuple(scale_backbone_value(item, scale) for item in value)
    raise TypeError(f"Cannot scale backbone value of type {type(value).__name__}.")


class BackboneMixin:
    """Default single-stream block backbone contract."""

    def backbone_blocks(self) -> List[BackboneBlockInfo]:
        if not hasattr(self, "blocks"):
            raise ValueError(f"{self.__class__.__name__} does not expose backbone blocks.")
        return [
            BackboneBlockInfo(index=index, name=f"blocks.{index}", module=block)
            for index, block in enumerate(self.blocks)
        ]

    def backbone_attention_modules(self) -> List[Tuple[int, str, torch.nn.Module]]:
        modules: List[Tuple[int, str, torch.nn.Module]] = []
        for block_info in self.backbone_blocks():
            block = block_info.module
            if hasattr(block, "self_attn"):
                modules.append((block_info.index, "self", block.self_attn))
            if hasattr(block, "cross_attn"):
                modules.append((block_info.index, "cross", block.cross_attn))
            if hasattr(block, "attn"):
                modules.append((block_info.index, "self", block.attn))
        return modules

    def backbone_make_state(self, tokens: torch.Tensor, **kwargs) -> BackboneState:
        block_kwargs = dict(kwargs)
        block_kwargs.pop("raw_e", None)
        return BackboneState({"x": tokens, "kwargs": block_kwargs})

    def backbone_prepare_block_state(self, block_info: BackboneBlockInfo, state: BackboneState) -> BackboneState:
        return state

    def backbone_run_block(self, block_info: BackboneBlockInfo, state: BackboneState) -> BackboneState:
        state["x"] = block_info.module(state["x"], **state["kwargs"])
        return state

    def backbone_state_tensor(self, state: Any) -> torch.Tensor:
        if torch.is_tensor(state):
            return state
        return state.tensor() if isinstance(state, BackboneState) else state["x"]

    def backbone_with_state_tensor(self, state: BackboneState, tensor: torch.Tensor) -> BackboneState:
        return state.with_tensor(tensor)

    def backbone_finalize_state(self, state: BackboneState) -> torch.Tensor:
        return state["x"]

    def backbone_cache_state(self, state: BackboneState) -> Any:
        return detach_backbone_value(self.backbone_state_tensor(state))

    def backbone_restore_cached_state(self, state: BackboneState, cached_state: Any) -> BackboneState:
        return self.backbone_with_state_tensor(state, cached_state)

    def backbone_block_delta(self, before: Any, after: Any) -> Any:
        before_tensor = before if torch.is_tensor(before) else self.backbone_state_tensor(before)
        after_tensor = after if torch.is_tensor(after) else self.backbone_state_tensor(after)
        return after_tensor - before_tensor

    def backbone_apply_block_delta(self, state: BackboneState, delta: Any) -> BackboneState:
        return self.backbone_with_state_tensor(state, self.backbone_state_tensor(state) + delta)
