from __future__ import annotations

import os

import torch
import torch.distributed as dist

from ...interconnect import local_interconnect
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
        """Whether the all-gather rides the copy engines instead of the SMs.

        The copy engines win when a GPU reaches all of its peers through one
        shared egress port, which is what `auto` resolves. On an 8x RTX PRO 5000
        host the CE all-gather is 1.1-1.2x faster than the SM kernel at CP8 and
        takes no SMs from the attention it overlaps with; on NVLink the SM mesh
        wins instead, because there every peer has its own link to fill.

        Every rank of a Fast CP group runs on one host (enforced in _runtime), so
        this resolves identically across the group -- and it has to: the layered
        copy-engine schedule issues one more barrier per call than the SM path,
        so a split decision would drift the group's barrier epochs apart.
        """
        value = os.environ.get("CHITU_FAST_AGKV_USE_CE", "auto").strip().lower()
        if value in {"", "auto"}:
            return local_interconnect().shared_egress
        if value in {"1", "true", "on", "yes"}:
            return True
        if value in {"0", "false", "off", "no"}:
            return False
        raise ValueError(
            f"CHITU_FAST_AGKV_USE_CE must be auto, on, or off; got {value!r}"
        )

    @property
    def async_enabled(self) -> bool:
        value = os.environ.get("CHITU_FAST_AGKV_ASYNC", "auto").strip().lower()
        if value in {"", "auto"}:
            # The SM kernel takes SMs from the attention it overlaps with, which
            # stopped paying off past four ranks. The copy engines take none, so
            # overlapping keeps paying at any width (CP8, Wan 1.3B shape: 0.789
            # ms synchronous against 0.725 ms overlapped).
            if self.use_ce:
                return True
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
