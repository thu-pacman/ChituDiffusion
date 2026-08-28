"""Static key/value cache for the parallel Hunyuan Image 3 decoder.

The released ``HunyuanStaticCache`` subclasses the Transformers ``StaticCache``
and reaches into internals that moved in Transformers 5, so it cannot allocate
its layers under this runtime. Image generation also needs less than the general
cache offers: the sequence never grows, positions arrive as a per-batch index
matrix, and every rank stores only its rank-local attention heads. This cache
covers exactly that, and allocates from the first key states it is handed so the
rank-local head count needs no separate plumbing.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch


class HunyuanImage3Cache:
    """Per-layer static cache holding one denoising request."""

    def __init__(self, *, num_layers: int, max_cache_len: int) -> None:
        if num_layers < 1:
            raise ValueError("a key/value cache needs at least one layer")
        if max_cache_len < 1:
            raise ValueError("max_cache_len must be positive")
        self.max_cache_len = int(max_cache_len)
        self.keys: list[torch.Tensor | None] = [None] * int(num_layers)
        self.values: list[torch.Tensor | None] = [None] * int(num_layers)

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    def byte_size(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*self.keys, *self.values)
            if tensor is not None
        )

    def _allocate(self, layer_idx: int, key_states: torch.Tensor) -> None:
        batch, heads, _, head_dim = key_states.shape
        shape = (batch, heads, self.max_cache_len, head_dim)
        self.keys[layer_idx] = torch.zeros(
            shape, dtype=key_states.dtype, device=key_states.device
        )
        self.values[layer_idx] = torch.zeros(
            shape, dtype=key_states.dtype, device=key_states.device
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write this step's keys and values, then return the whole window."""

        if self.keys[layer_idx] is None:
            self._allocate(layer_idx, key_states)
        keys = self.keys[layer_idx]
        values = self.values[layer_idx]
        assert keys is not None and values is not None
        positions = (cache_kwargs or {}).get("cache_position")
        if positions is None:
            raise ValueError("Hunyuan Image 3 requires explicit cache positions")
        key_states = key_states.to(keys.dtype)
        value_states = value_states.to(values.dtype)
        if positions.dim() == 1:
            keys.index_copy_(2, positions, key_states)
            values.index_copy_(2, positions, value_states)
        elif positions.dim() == 2:
            if positions.shape[0] != keys.shape[0]:
                raise ValueError(
                    f"cache positions cover {positions.shape[0]} sequences but the "
                    f"cache holds {keys.shape[0]}"
                )
            for index in range(positions.shape[0]):
                keys[index].index_copy_(1, positions[index], key_states[index])
                values[index].index_copy_(1, positions[index], value_states[index])
        else:
            raise ValueError("cache positions must be [sequence] or [batch, sequence]")
        return keys, values


__all__ = ["HunyuanImage3Cache"]
