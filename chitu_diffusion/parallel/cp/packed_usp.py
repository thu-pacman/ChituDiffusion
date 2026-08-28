from __future__ import annotations

import torch
import torch.distributed as dist

from ..tp.topology import get_tp_world_size
from .topology import UspTopology


def validate_tp_ulysses_heads(
    num_attention_heads: int,
    topology: UspTopology,
) -> None:
    divisor = get_tp_world_size() * topology.ulysses_degree
    if num_attention_heads % divisor:
        raise ValueError(
            f"attention heads {num_attention_heads} must be divisible by "
            f"TP x Ulysses degree {divisor}"
        )
    if topology.ring_degree != 1:
        raise ValueError("packed attention currently supports pure Ulysses only")


def all_to_all_packed_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    topology: UspTopology,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transpose local-sequence/local-TP-head QKV to global-sequence/UP-head."""

    if query.shape != key.shape or query.shape != value.shape or query.ndim != 3:
        raise ValueError("Q/K/V must have identical [local_tokens, heads, dim] shapes")
    degree = topology.ulysses_degree
    if degree == 1:
        return query, key, value
    local_tokens, heads, head_dim = query.shape
    if heads % degree:
        raise ValueError(f"local attention heads {heads} are not divisible by {degree}")
    transport = topology.ulysses_transport
    if transport is not None and transport.name == "fast_ulysses":
        transport.reset()
        redistributed = transport.all_to_all_qkv(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
        )
        return tuple(tensor.squeeze(0) for tensor in redistributed)  # type: ignore[return-value]
    local_heads = heads // degree
    packed = torch.stack((query, key, value), dim=-2)
    send = (
        packed.reshape(local_tokens, degree, local_heads, 3, head_dim)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(
        recv,
        send,
        group=topology.ulysses_process_group,
    )
    packed_global = recv.reshape(
        degree * local_tokens, local_heads, 3, head_dim
    )
    return tuple(packed_global.unbind(dim=-2))  # type: ignore[return-value]


def all_to_all_packed_output(
    output: torch.Tensor,
    *,
    topology: UspTopology,
) -> torch.Tensor:
    """Inverse Ulysses transpose back to local-sequence/local-TP-head layout."""

    if output.ndim != 3:
        raise ValueError("attention output must be [global_tokens, local_heads, dim]")
    degree = topology.ulysses_degree
    if degree == 1:
        return output
    global_tokens, local_heads, head_dim = output.shape
    if global_tokens % degree:
        raise ValueError(
            f"global token count {global_tokens} is not divisible by {degree}"
        )
    transport = topology.ulysses_transport
    if transport is not None and transport.name == "fast_ulysses":
        return transport.all_to_all(output.unsqueeze(0), 1, 2).squeeze(0)
    local_tokens = global_tokens // degree
    send = output.reshape(degree, local_tokens, local_heads, head_dim).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(
        recv,
        send,
        group=topology.ulysses_process_group,
    )
    return (
        recv.permute(1, 0, 2, 3)
        .reshape(local_tokens, degree * local_heads, head_dim)
        .contiguous()
    )

