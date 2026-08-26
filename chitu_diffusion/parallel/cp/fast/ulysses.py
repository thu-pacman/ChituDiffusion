from __future__ import annotations

import os

import torch

from ..ulysses_transport import AsyncUlyssesHandle, UlyssesTransport
from ._runtime import FastUlyssesAllToAll


class FastUlyssesNodeRuntime:
    """One lazily initialized NVSHMEM runtime shared by all local U rows."""

    def __init__(self, process_group: object, device: torch.device) -> None:
        self._process_group = process_group
        self._device = device
        self._backend: FastUlyssesAllToAll | None = None
        self._closed = False

    @property
    def pool_bytes(self) -> int:
        return self._backend.pool_bytes if self._backend is not None else 0

    def backend(self, tensor: torch.Tensor) -> FastUlyssesAllToAll:
        if self._closed:
            raise RuntimeError("Fast Ulysses node runtime is closed")
        if self._backend is None:
            self._backend = FastUlyssesAllToAll(
                self._process_group,
                self._device,
                pool_bytes=FastUlyssesTransport._pool_bytes_for(tensor),
            )
        return self._backend

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._backend is not None:
            self._backend.destroy()


class FastUlyssesSubgroupTransport:
    """Logical U row operating inside one node-local NVSHMEM runtime."""

    name = "fast_ulysses"

    def __init__(
        self,
        runtime: FastUlyssesNodeRuntime,
        *,
        peer_ranks: tuple[int, ...],
        fallback: UlyssesTransport,
    ) -> None:
        if not peer_ranks:
            raise ValueError("Fast Ulysses subgroup cannot be empty")
        self._runtime = runtime
        self._peer_ranks = tuple(int(rank) for rank in peer_ranks)
        self._fallback = fallback
        self._epoch = 0
        self._operation = 0
        subgroup = "_".join(str(rank) for rank in self._peer_ranks)
        self._tag_prefix = f"chitu_ulysses_{subgroup}"

    @property
    def pool_bytes(self) -> int:
        return self._runtime.pool_bytes

    def reset(self) -> None:
        self._epoch += 1
        self._operation = 0

    def all_to_all(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
    ) -> torch.Tensor:
        if not FastUlyssesAllToAll.supports(tensor, scatter_dim, gather_dim):
            return self._fallback.all_to_all(tensor, scatter_dim, gather_dim)
        backend = self._runtime.backend(tensor)
        mode = backend.mode(scatter_dim, gather_dim)
        tag = f"{self._tag_prefix}_{self._operation}"
        self._operation += 1
        output = backend.logical_all_to_all(
            tensor,
            peer_ranks=self._peer_ranks,
            mode=mode,
            tag=tag,
            subgroup_type="ulysses_row",
        )
        # Each rank waits only for its own outgoing CE copies. Publish a
        # subgroup ticket afterwards so consumers cannot read while a peer is
        # still writing its slice.
        backend.logical_barrier(
            peer_ranks=self._peer_ranks,
            tag=f"{self._tag_prefix}_barrier",
            epoch=max(self._epoch, 1) * 16 + self._operation,
        )
        return output

    def all_to_all_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.all_to_all(query, 2, 1),
            self.all_to_all(key, 2, 1),
            self.all_to_all(value, 2, 1),
        )

    def close(self) -> None:
        # The context owns and closes the shared node runtime exactly once.
        return


class FastUlyssesTransport:
    name = "fast_ulysses"

    def __init__(
        self,
        process_group: object,
        device: torch.device,
        *,
        fallback: UlyssesTransport,
    ) -> None:
        self._process_group = process_group
        self._device = device
        self._backend: FastUlyssesAllToAll | None = None
        self._fallback = fallback
        self._closed = False

    @property
    def pool_bytes(self) -> int:
        return self._backend.pool_bytes if self._backend is not None else 0

    @staticmethod
    def _pool_bytes_for(tensor: torch.Tensor) -> int:
        configured = os.environ.get("CHITU_FAST_ULYSSES_POOL_BYTES")
        if configured is not None:
            pool_bytes = int(configured)
            if pool_bytes <= 0:
                raise ValueError("CHITU_FAST_ULYSSES_POOL_BYTES must be positive")
            return pool_bytes
        required = 4 * tensor.numel() * tensor.element_size() + (16 << 20)
        alignment = 64 << 20
        aligned_required = ((required + alignment - 1) // alignment) * alignment
        return max(2 << 30, aligned_required)

    def _ensure_backend(self, tensor: torch.Tensor) -> FastUlyssesAllToAll:
        if self._backend is None:
            self._backend = FastUlyssesAllToAll(
                self._process_group,
                self._device,
                pool_bytes=self._pool_bytes_for(tensor),
            )
        return self._backend

    @property
    def async_ce_enabled(self) -> bool:
        value = os.environ.get("CHITU_FAST_ULYSSES_ASYNC_CE", "1")
        return value.strip().lower() not in {"0", "false", "off", "no"}

    def reset(self) -> None:
        if self._backend is not None:
            self._backend.reset_sequence()

    def all_to_all(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
    ) -> torch.Tensor:
        if FastUlyssesAllToAll.supports(tensor, scatter_dim, gather_dim):
            return self._ensure_backend(tensor).all_to_all(
                tensor, scatter_dim, gather_dim
            )
        return self._fallback.all_to_all(tensor, scatter_dim, gather_dim)

    def all_to_all_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.all_to_all(query, 2, 1),
            self.all_to_all(key, 2, 1),
            self.all_to_all(value, 2, 1),
        )

    def begin_all_to_all(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        *,
        tag: str,
        barrier: bool,
    ) -> AsyncUlyssesHandle:
        if not self.async_ce_enabled:
            raise RuntimeError("Fast Ulysses async CE is disabled")
        if not FastUlyssesAllToAll.supports(tensor, scatter_dim, gather_dim):
            raise ValueError("tensor is unsupported by Fast Ulysses async CE")
        return self._ensure_backend(tensor).begin_all_to_all(
            tensor,
            scatter_dim,
            gather_dim,
            tag=tag,
            barrier=barrier,
            use_ce=True,
        )

    def all_to_all_tagged(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        *,
        tag: str,
    ) -> torch.Tensor:
        if not FastUlyssesAllToAll.supports(tensor, scatter_dim, gather_dim):
            raise ValueError("tensor is unsupported by Fast Ulysses")
        return self._ensure_backend(tensor).all_to_all_tagged(
            tensor,
            scatter_dim,
            gather_dim,
            tag=tag,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._backend is not None:
            self._backend.destroy()
