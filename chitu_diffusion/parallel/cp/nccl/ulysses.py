from __future__ import annotations

import torch
import torch.distributed as dist


def torch_all_to_all_4d(
    tensor: torch.Tensor,
    *,
    group: object | None,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Ulysses transport requires a 4D tensor, got {tensor.ndim}D")
    world_size = dist.get_world_size(group) if group is not None else 1
    if world_size == 1:
        return tensor
    batch, sequence, heads, head_dim = tensor.shape
    if scatter_dim == 2 and gather_dim == 1:
        if heads % world_size:
            raise ValueError(
                f"attention heads {heads} must be divisible by Ulysses degree "
                f"{world_size}"
            )
        local_heads = heads // world_size
        send = (
            tensor.reshape(batch, sequence, world_size, local_heads, head_dim)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        return (
            recv.permute(1, 0, 2, 3, 4)
            .reshape(batch, world_size * sequence, local_heads, head_dim)
            .contiguous()
        )
    if scatter_dim == 1 and gather_dim == 2:
        if sequence % world_size:
            raise ValueError(
                f"attention sequence {sequence} must be divisible by Ulysses degree "
                f"{world_size}"
            )
        local_sequence = sequence // world_size
        send = (
            tensor.reshape(batch, world_size, local_sequence, heads, head_dim)
            .permute(1, 0, 2, 3, 4)
            .contiguous()
        )
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        return (
            recv.permute(1, 2, 0, 3, 4)
            .reshape(batch, local_sequence, world_size * heads, head_dim)
            .contiguous()
        )
    raise ValueError(
        "Ulysses supports only heads-to-sequence (2,1) and sequence-to-heads (1,2)"
    )


class TorchUlyssesTransport:
    name = "torch"

    def __init__(self, process_group: object | None) -> None:
        self.process_group = process_group

    @property
    def pool_bytes(self) -> int:
        return 0

    def reset(self) -> None:
        return None

    def all_to_all(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
    ) -> torch.Tensor:
        return torch_all_to_all_4d(
            tensor,
            group=self.process_group,
            scatter_dim=scatter_dim,
            gather_dim=gather_dim,
        )

    def all_to_all_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tensors = (query, key, value)
        if any(tensor.ndim != 4 for tensor in tensors):
            raise ValueError("Ulysses Q/K/V transport requires 4D tensors")
        if key.shape != query.shape or value.shape != query.shape:
            raise ValueError("Ulysses Q/K/V tensors must have identical shapes")
        world_size = (
            dist.get_world_size(self.process_group)
            if self.process_group is not None
            else 1
        )
        if world_size == 1:
            return tensors
        batch, sequence, heads, head_dim = query.shape
        if heads % world_size:
            raise ValueError(
                f"attention heads {heads} must be divisible by Ulysses degree "
                f"{world_size}"
            )
        local_heads = heads // world_size
        sends = tuple(
            tensor.reshape(
                batch,
                sequence,
                world_size,
                local_heads,
                head_dim,
            )
            .permute(2, 0, 1, 3, 4)
            .contiguous()
            for tensor in tensors
        )
        recvs = tuple(torch.empty_like(send) for send in sends)
        coalescing_manager = getattr(dist, "_coalescing_manager", None)
        can_coalesce = (
            coalescing_manager is not None
            and query.is_cuda
            and dist.get_backend(self.process_group) == "nccl"
            and world_size <= 4
        )
        if not can_coalesce:
            for recv, send in zip(recvs, sends, strict=True):
                dist.all_to_all_single(recv, send, group=self.process_group)
        else:
            with coalescing_manager(
                group=self.process_group,
                device=query.device,
            ):
                for recv, send in zip(recvs, sends, strict=True):
                    dist.all_to_all_single(recv, send, group=self.process_group)
        return tuple(
            recv.permute(1, 0, 2, 3, 4)
            .reshape(batch, world_size * sequence, local_heads, head_dim)
            .contiguous()
            for recv in recvs
        )

    def close(self) -> None:
        return None
