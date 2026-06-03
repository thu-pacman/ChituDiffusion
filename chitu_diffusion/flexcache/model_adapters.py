from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch


def detach_cache_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: detach_cache_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detach_cache_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_cache_value(item) for item in value)
    return value


def add_cache_values(left: Any, right: Any) -> Any:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left + right
    if isinstance(left, dict) and isinstance(right, dict):
        return {key: add_cache_values(left[key], right[key]) for key in left.keys()}
    if isinstance(left, list) and isinstance(right, list):
        return [add_cache_values(a, b) for a, b in zip(left, right)]
    if isinstance(left, tuple) and isinstance(right, tuple):
        return tuple(add_cache_values(a, b) for a, b in zip(left, right))
    raise TypeError(f"Cannot add cache values of type {type(left).__name__} and {type(right).__name__}.")


def sub_cache_values(left: Any, right: Any) -> Any:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return left - right
    if isinstance(left, dict) and isinstance(right, dict):
        return {key: sub_cache_values(left[key], right[key]) for key in left.keys()}
    if isinstance(left, list) and isinstance(right, list):
        return [sub_cache_values(a, b) for a, b in zip(left, right)]
    if isinstance(left, tuple) and isinstance(right, tuple):
        return tuple(sub_cache_values(a, b) for a, b in zip(left, right))
    raise TypeError(f"Cannot subtract cache values of type {type(left).__name__} and {type(right).__name__}.")


def scale_cache_value(value: Any, scale: float) -> Any:
    if torch.is_tensor(value):
        return value * scale
    if isinstance(value, dict):
        return {key: scale_cache_value(item, scale) for key, item in value.items()}
    if isinstance(value, list):
        return [scale_cache_value(item, scale) for item in value]
    if isinstance(value, tuple):
        return tuple(scale_cache_value(item, scale) for item in value)
    raise TypeError(f"Cannot scale cache value of type {type(value).__name__}.")


@dataclass
class FlexCacheBlockInfo:
    index: int
    name: str
    module: torch.nn.Module


class FlexCacheModelAdapter:
    """Thin model-facing API used by FlexCache strategies."""

    def __init__(self, module: torch.nn.Module):
        self.module = module

    @property
    def blocks(self) -> List[FlexCacheBlockInfo]:
        if hasattr(self.module, "flexcache_blocks"):
            return list(self.module.flexcache_blocks())
        if hasattr(self.module, "blocks"):
            return [
                FlexCacheBlockInfo(index=index, name=f"blocks.{index}", module=block)
                for index, block in enumerate(self.module.blocks)
            ]
        raise ValueError(
            f"{self.module.__class__.__name__} does not expose flexcache_blocks() or blocks."
        )

    def make_state(self, tokens: torch.Tensor, kwargs: Dict[str, Any]) -> Any:
        if hasattr(self.module, "flexcache_make_state"):
            return self.module.flexcache_make_state(tokens, **kwargs)
        block_kwargs = dict(kwargs)
        block_kwargs.pop("raw_e", None)
        return {"x": tokens, "kwargs": block_kwargs}

    def run_block(self, block_info: FlexCacheBlockInfo, state: Any) -> Any:
        if hasattr(self.module, "flexcache_run_block"):
            return self.module.flexcache_run_block(block_info.index, state)
        state["x"] = block_info.module(state["x"], **state["kwargs"])
        return state

    def prepare_block_state(self, block_info: FlexCacheBlockInfo, state: Any) -> Any:
        if hasattr(self.module, "flexcache_prepare_block_state"):
            return self.module.flexcache_prepare_block_state(block_info.index, state)
        return state

    def state_tensor(self, state: Any) -> torch.Tensor:
        if hasattr(self.module, "flexcache_state_tensor"):
            return self.module.flexcache_state_tensor(state)
        return state["x"]

    def with_state_tensor(self, state: Any, tensor: torch.Tensor) -> Any:
        if hasattr(self.module, "flexcache_with_state_tensor"):
            return self.module.flexcache_with_state_tensor(state, tensor)
        state = dict(state)
        state["x"] = tensor
        return state

    def finalize_state(self, state: Any) -> torch.Tensor:
        if hasattr(self.module, "flexcache_finalize_state"):
            return self.module.flexcache_finalize_state(state)
        return state["x"]

    def cache_state(self, state: Any) -> Any:
        if hasattr(self.module, "flexcache_cache_state"):
            return self.module.flexcache_cache_state(state)
        return self.state_tensor(state).detach()

    def restore_cached_state(self, state: Any, cached_state: Any) -> Any:
        if hasattr(self.module, "flexcache_restore_cached_state"):
            return self.module.flexcache_restore_cached_state(state, cached_state)
        return self.with_state_tensor(state, cached_state)

    def block_delta(self, before: Any, after: Any) -> Any:
        if hasattr(self.module, "flexcache_block_delta"):
            return self.module.flexcache_block_delta(before, after)
        return self.state_tensor(after) - self.state_tensor(before)

    def apply_block_delta(self, state: Any, delta: Any) -> Any:
        if hasattr(self.module, "flexcache_apply_block_delta"):
            return self.module.flexcache_apply_block_delta(state, delta)
        return self.with_state_tensor(state, self.state_tensor(state) + delta)

    def attention_modules(self) -> List[Tuple[int, str, torch.nn.Module]]:
        if hasattr(self.module, "flexcache_attention_modules"):
            return list(self.module.flexcache_attention_modules())

        modules: List[Tuple[int, str, torch.nn.Module]] = []
        for block_info in self.blocks:
            block = block_info.module
            if hasattr(block, "self_attn"):
                modules.append((block_info.index, "self", block.self_attn))
            if hasattr(block, "cross_attn"):
                modules.append((block_info.index, "cross", block.cross_attn))
            if hasattr(block, "attn"):
                modules.append((block_info.index, "self", block.attn))
        return modules


def get_flexcache_adapter(module: torch.nn.Module) -> FlexCacheModelAdapter:
    return FlexCacheModelAdapter(module)
