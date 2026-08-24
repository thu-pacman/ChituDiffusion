from __future__ import annotations

import os

import torch
import torch.distributed as dist

from ..agkv_transport import AgkvTransport, AsyncAgkvHandle
from ._runtime import FastUlyssesAllToAll


class FastAgkvTransport:
    name = "fast_agkv"

    def __init__(
        self,
        process_group: object,
        device: torch.device,
        *,
        fallback: AgkvTransport,
    ) -> None:
        self._process_group = process_group
        self._device = device
        self._fallback = fallback
        self._backend: FastUlyssesAllToAll | None = None
        self._closed = False

    @staticmethod
    def _pool_bytes_for(key: torch.Tensor, world_size: int) -> int:
        configured = os.environ.get("CHITU_FAST_AGKV_POOL_BYTES")
        if configured is not None:
            pool_bytes = int(configured)
            if pool_bytes <= 0:
                raise ValueError("CHITU_FAST_AGKV_POOL_BYTES must be positive")
            return pool_bytes
        output_bytes = key.numel() * key.element_size() * world_size
        required = 2 * output_bytes + (16 << 20)
        alignment = 64 << 20
        return ((required + alignment - 1) // alignment) * alignment

    def _ensure_backend(self, key: torch.Tensor) -> FastUlyssesAllToAll:
        if self._backend is None:
            world_size = dist.get_world_size(self._process_group)
            self._backend = FastUlyssesAllToAll(
                self._process_group,
                self._device,
                pool_bytes=self._pool_bytes_for(key, world_size),
            )
        return self._backend

    @property
    def use_ce(self) -> bool:
        value = os.environ.get("CHITU_FAST_AGKV_USE_CE", "0")
        return value.strip().lower() in {"1", "true", "on", "yes"}

    @property
    def async_enabled(self) -> bool:
        value = os.environ.get("CHITU_FAST_AGKV_ASYNC", "auto").strip().lower()
        if value in {"", "auto"}:
            return dist.get_world_size(self._process_group) <= 4
        if value in {"1", "true", "on", "yes"}:
            return True
        if value in {"0", "false", "off", "no"}:
            return False
        raise ValueError(
            f"CHITU_FAST_AGKV_ASYNC must be auto, on, or off; got {value!r}"
        )

    def all_gather_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not FastUlyssesAllToAll.supports_agkv(key, value):
            return self._fallback.all_gather_kv(key, value)
        return self._ensure_backend(key).all_gather_kv(
            key,
            value,
            key_tag="chitu_agkv_key",
            value_tag="chitu_agkv_value",
            use_ce=self.use_ce,
        )

    def begin_all_gather_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> AsyncAgkvHandle:
        if not FastUlyssesAllToAll.supports_agkv(key, value):
            raise ValueError("tensors are unsupported by asynchronous Fast AGKV")
        return self._ensure_backend(key).begin_all_gather_kv(
            key,
            value,
            key_tag="chitu_agkv_key",
            value_tag="chitu_agkv_value",
            use_ce=self.use_ce,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._backend is not None:
            self._backend.destroy()
