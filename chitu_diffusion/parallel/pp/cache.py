from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch

from .config import FppConfig


class RollingKVCache:
    """One request's per-branch, per-layer post-RoPE keys and values.

    Patches update in place: later patches see earlier patches' fresh KV and
    the remaining patches' previous-step KV, matching feat/pp's rolling cache.
    """

    def __init__(self) -> None:
        self.entries: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def update(
        self,
        branch: str,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        bounds: tuple[int, int] | None,
        tokens: int,
        indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError(
                "FPP K/V must have matching [batch, tokens, heads, dim] shapes"
            )
        cache_key = (branch, layer)
        if bounds is None and indices is None:
            if key.shape[1] != tokens:
                raise ValueError("FPP warmup must supply all video tokens")
            self.entries[cache_key] = (key.detach().clone(), value.detach().clone())
            return key, value
        if indices is None:
            start, stop = bounds
            if not 0 <= start < stop <= tokens or key.shape[1] != stop - start:
                raise ValueError("FPP patch bounds do not match its K/V shape")
        elif indices.ndim != 1 or indices.numel() != key.shape[1]:
            raise ValueError("FPP patch indices do not match its K/V shape")
        if cache_key not in self.entries:
            raise RuntimeError(
                "FPP patch attention requires a full warmup for this branch and layer"
            )
        cached_key, cached_value = self.entries[cache_key]
        expected = (key.shape[0], tokens, *key.shape[2:])
        if cached_key.shape != expected or any(
            cached.dtype != current.dtype or cached.device != current.device
            for cached, current in ((cached_key, key), (cached_value, value))
        ):
            raise ValueError(
                "FPP cache shape, dtype, or device changed; refresh the request cache"
            )
        if indices is None:
            cached_key[:, start:stop].copy_(key)
            cached_value[:, start:stop].copy_(value)
        else:
            cached_key.index_copy_(1, indices, key)
            cached_value.index_copy_(1, indices, value)
        return cached_key, cached_value

    def clear(self) -> None:
        self.entries.clear()


@dataclass
class FppState:
    config: FppConfig
    cache: RollingKVCache = field(default_factory=RollingKVCache)
    last_step: int = -1
    full_steps: int = 0
    patch_steps: int = 0

    def needs_full_step(self, step: int, total_steps: int) -> bool:
        return (
            self.config.full_step(step, total_steps)
            or not self.cache.entries
            or self.last_step != step - 1
        )


@dataclass(frozen=True)
class FppAttentionContext:
    cache: RollingKVCache
    branch: str
    bounds: tuple[int, int] | None
    tokens: int
    indices: torch.Tensor | None = None
    cp_group: object | None = None


_ATTENTION_CONTEXT: ContextVar[FppAttentionContext | None] = ContextVar(
    "chitu_fpp_attention", default=None
)


@contextmanager
def fpp_attention_context(context: FppAttentionContext):
    token = _ATTENTION_CONTEXT.set(context)
    try:
        yield
    finally:
        _ATTENTION_CONTEXT.reset(token)


def current_fpp_attention_context() -> FppAttentionContext:
    context = _ATTENTION_CONTEXT.get()
    if context is None:
        raise RuntimeError("FPP attention must run inside a request's pipeline forward")
    return context
