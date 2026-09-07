from __future__ import annotations

import torch
import torch.distributed as dist


def all_gather_sequence(
    tensor: torch.Tensor,
    process_group: object | None,
) -> torch.Tensor:
    world_size = dist.get_world_size(process_group) if process_group is not None else 1
    if world_size == 1:
        return tensor
    contiguous = tensor.contiguous()
    gathered = torch.empty(
        world_size * tensor.shape[0],
        *tensor.shape[1:],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(gathered, contiguous, group=process_group)
    rank_major = gathered.unflatten(0, (world_size, tensor.shape[0]))
    return (
        rank_major.permute(1, 0, 2, 3, 4)
        .reshape(
            tensor.shape[0],
            world_size * tensor.shape[1],
            *tensor.shape[2:],
        )
        .contiguous()
    )


def all_gather_sequence_pair(
    key: torch.Tensor,
    value: torch.Tensor,
    process_group: object | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_size = dist.get_world_size(process_group) if process_group is not None else 1
    if world_size == 1:
        return key, value
    if key.shape != value.shape or key.dtype != value.dtype:
        return (
            all_gather_sequence(key, process_group),
            all_gather_sequence(value, process_group),
        )

    inputs = (key.contiguous(), value.contiguous())
    outputs = tuple(
        torch.empty(
            world_size * tensor.shape[0],
            *tensor.shape[1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        for tensor in inputs
    )
    # Keep K/V gathers explicit. PyTorch's private coalescing manager may defer
    # completion to the first consumer on NCCL, causing severe CP8 tail latency.
    for output, tensor in zip(outputs, inputs, strict=True):
        dist.all_gather_into_tensor(output, tensor, group=process_group)

    gathered = []
    for output, tensor in zip(outputs, inputs, strict=True):
        rank_major = output.unflatten(0, (world_size, tensor.shape[0]))
        gathered.append(
            rank_major.permute(1, 0, 2, 3, 4)
            .reshape(
                tensor.shape[0],
                world_size * tensor.shape[1],
                *tensor.shape[2:],
            )
            .contiguous()
        )
    return gathered[0], gathered[1]


class TorchAgkvTransport:
    name = "torch"

    def __init__(self, process_group: object | None) -> None:
        self.process_group = process_group

    def all_gather_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return all_gather_sequence_pair(key, value, self.process_group)

    def close(self) -> None:
        return None
