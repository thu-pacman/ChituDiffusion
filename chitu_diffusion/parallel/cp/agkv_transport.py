from __future__ import annotations

import os
from typing import Literal, Protocol, runtime_checkable

import torch

AgkvTransportName = Literal["auto", "torch", "fast_agkv"]
AGKV_TRANSPORTS: tuple[AgkvTransportName, ...] = (
    "auto",
    "torch",
    "fast_agkv",
)


class AgkvTransport(Protocol):
    name: str

    def all_gather_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def close(self) -> None: ...


class AsyncAgkvHandle(Protocol):
    def wait(self) -> tuple[torch.Tensor, torch.Tensor]: ...


@runtime_checkable
class AsyncAgkvTransport(Protocol):
    @property
    def async_enabled(self) -> bool: ...

    def begin_all_gather_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> AsyncAgkvHandle: ...


def resolve_agkv_transport(value: str | None = None) -> AgkvTransportName:
    requested = (
        value if value is not None else os.environ.get("CHITU_AGKV_TRANSPORT", "torch")
    )
    normalized = str(requested).strip().lower().replace("-", "_")
    normalized = {"fast": "fast_agkv", "nccl": "torch"}.get(normalized, normalized)
    if normalized not in AGKV_TRANSPORTS:
        choices = ", ".join(AGKV_TRANSPORTS)
        raise ValueError(f"AGKV transport must be one of {choices}, got {requested!r}")
    return normalized  # type: ignore[return-value]


def create_agkv_transport(
    value: str | None,
    *,
    process_group: object | None,
    device: torch.device,
    fast_eligible: bool,
) -> AgkvTransport:
    from .fast._runtime import probe_fast_ulysses
    from .nccl.agkv import TorchAgkvTransport

    requested = resolve_agkv_transport(value)
    fallback = TorchAgkvTransport(process_group)
    if requested == "torch":
        return fallback
    available, reason = probe_fast_ulysses(
        device,
        require_hopper=False,
        require_subgroups=False,
        require_fast_agkv=True,
    )
    if available and fast_eligible and process_group is not None:
        from .fast.agkv_transport import FastAgkvTransport

        return FastAgkvTransport(
            process_group,
            device,
            fallback=fallback,
        )
    if requested == "fast_agkv":
        raise RuntimeError(
            "Fast AGKV requires a single-node P2P lane of the configured fast "
            f"lane width: {reason}"
        )
    return fallback
