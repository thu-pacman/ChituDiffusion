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
    process_group: object | None,
) -> tuple[bool, str]:
    from .fast._runtime import probe_fast_ulysses

    local = probe_fast_ulysses(
        device,
        require_hopper=require_hopper,
        require_subgroups=False,
    )
    if (
        not dist.is_initialized()
        or process_group is None
        or dist.get_world_size(process_group) == 1
    ):
        return local
    group_ranks = dist.get_process_group_ranks(process_group)
    gathered: list[tuple[bool, str] | None] = [None] * len(group_ranks)
    dist.all_gather_object(gathered, local, group=process_group)
    failures = [
        f"rank {group_ranks[index]}: {result[1]}"
        for index, result in enumerate(gathered)
        if result is None or not result[0]
    ]
    if failures:
        return False, "; ".join(failures)
    return True, "; ".join(
        f"rank {group_ranks[index]}: {result[1]}"
        for index, result in enumerate(gathered)
        if result is not None
    )


def create_ulysses_transport(
    name: str | None,
    *,
    process_group: object | None,
    device: torch.device,
    fast_eligible: bool,
) -> UlyssesTransport:
    from .nccl.ulysses import TorchUlyssesTransport

    requested = resolve_ulysses_transport(name)
    fallback = TorchUlyssesTransport(process_group)
    if requested == "torch":
        return fallback
    if not fast_eligible:
        if requested == "fast_ulysses":
            raise ValueError(
                "fast_ulysses binds one NVSHMEM runtime to a single static lane, "
                "so it is only built for lanes of the configured fast lane width"
            )
        return fallback
    ready, reason = _collect_fast_capability(
        device,
        require_hopper=True,
        process_group=process_group,
    )
    if not ready:
        if requested == "fast_ulysses":
            raise RuntimeError(f"fast_ulysses is unavailable: {reason}")
        return fallback
    if process_group is None:
        return fallback

    from .fast.ulysses import FastUlyssesTransport

    return FastUlyssesTransport(
        process_group,
        device,
        fallback=fallback,
    )
