from __future__ import annotations

import os
from typing import Literal, Protocol, runtime_checkable

import torch
import torch.distributed as dist

UlyssesTransportName = Literal["auto", "torch", "fast_ulysses"]
ULYSSES_TRANSPORTS: tuple[UlyssesTransportName, ...] = (
    "auto",
    "torch",
    "fast_ulysses",
)


class UlyssesTransport(Protocol):
    name: str

    @property
    def pool_bytes(self) -> int: ...

    def reset(self) -> None: ...

    def all_to_all(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
    ) -> torch.Tensor: ...

    def all_to_all_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def close(self) -> None: ...


class AsyncUlyssesHandle(Protocol):
    def wait(self) -> torch.Tensor: ...


@runtime_checkable
class AsyncTaggedUlyssesTransport(Protocol):
    @property
    def async_ce_enabled(self) -> bool: ...

    def reset(self) -> None: ...

    def begin_all_to_all(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        *,
        tag: str,
        barrier: bool,
    ) -> AsyncUlyssesHandle: ...

    def all_to_all_tagged(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        *,
        tag: str,
    ) -> torch.Tensor: ...


def resolve_ulysses_transport(value: str | None = None) -> UlyssesTransportName:
    requested = (
        value
        if value is not None
        else os.environ.get("CHITU_ULYSSES_TRANSPORT", "torch")
    )
    normalized = str(requested).strip().lower().replace("-", "_")
    normalized = {
        "fast": "fast_ulysses",
        "nccl": "torch",
        "native": "torch",
    }.get(normalized, normalized)
    if normalized not in ULYSSES_TRANSPORTS:
        choices = ", ".join(ULYSSES_TRANSPORTS)
        raise ValueError(
            f"Ulysses transport must be one of {choices}, got {requested!r}"
        )
    return normalized  # type: ignore[return-value]


def _collect_fast_capability(
    device: torch.device,
    *,
    require_hopper: bool,
) -> tuple[bool, str]:
    from .fast_cp._runtime import probe_fast_ulysses

    local = probe_fast_ulysses(
        device,
        require_hopper=require_hopper,
        require_subgroups=False,
    )
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return local
    gathered: list[tuple[bool, str] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local, group=dist.group.WORLD)
    failures = [
        f"rank {rank}: {result[1]}"
        for rank, result in enumerate(gathered)
        if result is None or not result[0]
    ]
    if failures:
        return False, "; ".join(failures)
    return True, "; ".join(
        f"rank {rank}: {result[1]}"
        for rank, result in enumerate(gathered)
        if result is not None
    )


def create_ulysses_transport(
    name: str | None,
    *,
    process_group: object | None,
    device: torch.device,
    static_full_world: bool,
) -> UlyssesTransport:
    from .nccl.ulysses import TorchUlyssesTransport

    requested = resolve_ulysses_transport(name)
    fallback = TorchUlyssesTransport(process_group)
    if requested == "torch":
        return fallback
    if not static_full_world:
        if requested == "fast_ulysses":
            raise ValueError(
                "fast_ulysses currently requires static full-world, full-Ulysses CP"
            )
        return fallback
    ready, reason = _collect_fast_capability(device, require_hopper=True)
    if not ready:
        if requested == "fast_ulysses":
            raise RuntimeError(f"fast_ulysses is unavailable: {reason}")
        return fallback
    if process_group is None:
        return fallback

    from .fast_cp.ulysses import FastUlyssesTransport

    return FastUlyssesTransport(
        process_group,
        device,
        fallback=fallback,
    )
