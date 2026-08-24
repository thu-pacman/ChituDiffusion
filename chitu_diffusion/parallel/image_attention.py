from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .agkv_transport import AgkvTransport
from .nccl.agkv import all_gather_sequence_pair
from .topology import UlyssesTopology

CONTEXT_PARALLEL_MODES = ("agkv", "ulysses")


def resolve_context_parallel_config(
    mode: str = "agkv",
    ulysses_degree: int | None = None,
) -> tuple[str, int | None]:
    """Normalize the model-independent context-parallel backend settings."""
    normalized_mode = str(mode).lower()
    if normalized_mode not in CONTEXT_PARALLEL_MODES:
        choices = ", ".join(CONTEXT_PARALLEL_MODES)
        raise ValueError(f"context-parallel mode must be one of: {choices}")
    degree = int(ulysses_degree) if ulysses_degree is not None else None
    if degree is not None and degree < 1:
        raise ValueError("ulysses_degree must be positive")
    if normalized_mode == "agkv" and degree is not None:
        raise ValueError("ulysses_degree is only valid in Ulysses mode")
    return normalized_mode, degree


def _sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    return (
        F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        )
        .transpose(1, 2)
        .contiguous()
    )


def _pad_heads(tensor: torch.Tensor, degree: int) -> tuple[torch.Tensor, int]:
    original_heads = tensor.shape[2]
    pad_heads = (-original_heads) % degree
    if pad_heads:
        tensor = torch.cat(
            [
                tensor,
                tensor.new_zeros(
                    *tensor.shape[:2],
                    pad_heads,
                    tensor.shape[-1],
                ),
            ],
            dim=2,
        )
    return tensor, original_heads


def _gather_heads(tensor: torch.Tensor, topology: UlyssesTopology) -> torch.Tensor:
    if topology.degree == 1:
        return tensor
    if tensor.shape[1] == 0:
        return tensor.new_empty(
            tensor.shape[0],
            0,
            tensor.shape[2] * topology.degree,
            tensor.shape[3],
        )
    pieces = [torch.empty_like(tensor) for _ in range(topology.degree)]
    dist.all_gather(pieces, tensor.contiguous(), group=topology.process_group)
    return torch.cat(pieces, dim=2).contiguous()


def _ulysses_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topology: UlyssesTopology,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    tensors = tuple(
        _pad_heads(tensor, topology.degree) for tensor in (query, key, value)
    )
    original_heads = tensors[0][1]
    if any(heads != original_heads for _, heads in tensors):
        raise ValueError("Ulysses Q/K/V tensors must have the same head count")
    topology.transport.reset()
    redistributed = topology.transport.all_to_all_qkv(
        tensors[0][0],
        tensors[1][0],
        tensors[2][0],
    )
    return *redistributed, original_heads


class ImageContextParallelAttention:
    """Attention for sequence-sharded image and replicated joint tokens."""

    def __init__(self, mode: str = "agkv") -> None:
        self.mode, _ = resolve_context_parallel_config(mode)

    def __call__(
        self,
        image_query: torch.Tensor,
        image_key: torch.Tensor,
        image_value: torch.Tensor,
        joint_query: torch.Tensor,
        joint_key: torch.Tensor,
        joint_value: torch.Tensor,
        *,
        lane_process_group: object | None,
        ulysses_topology: UlyssesTopology | None = None,
        agkv_transport: AgkvTransport | None = None,
        joint_first: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(joint_output, local_image_output)`` in sequence-major layout."""
        if self.mode == "ulysses":
            if ulysses_topology is None:
                raise ValueError("ulysses_topology is required for Ulysses attention")
            topology = ulysses_topology
            image_query, image_key, image_value, original_heads = _ulysses_qkv(
                image_query, image_key, image_value, topology
            )
            joint_tensors = tuple(
                _pad_heads(tensor, topology.degree)[0]
                for tensor in (joint_query, joint_key, joint_value)
            )
            heads_per_rank = image_query.shape[2]
            head_start = topology.rank * heads_per_rank
            head_stop = head_start + heads_per_rank
            joint_query, joint_key, joint_value = tuple(
                tensor[:, :, head_start:head_stop].contiguous()
                for tensor in joint_tensors
            )
            image_tokens = image_query.shape[1]
            joint_tokens = joint_query.shape[1]
            if joint_first:
                output = _sdpa(
                    torch.cat([joint_query, image_query], dim=1),
                    torch.cat([joint_key, image_key], dim=1),
                    torch.cat([joint_value, image_value], dim=1),
                )
                joint_output = output[:, :joint_tokens]
                image_output = output[:, joint_tokens:]
            else:
                output = _sdpa(
                    torch.cat([image_query, joint_query], dim=1),
                    torch.cat([image_key, joint_key], dim=1),
                    torch.cat([image_value, joint_value], dim=1),
                )
                image_output = output[:, :image_tokens]
                joint_output = output[:, image_tokens:]
            image_output = topology.transport.all_to_all(
                image_output,
                1,
                2,
            )[:, :, :original_heads]
            joint_output = _gather_heads(joint_output, topology)[:, :, :original_heads]
            return joint_output.contiguous(), image_output.contiguous()

        local_image_tokens = image_query.shape[1]
        if agkv_transport is None:
            full_image_key, full_image_value = all_gather_sequence_pair(
                image_key,
                image_value,
                lane_process_group,
            )
        else:
            full_image_key, full_image_value = agkv_transport.all_gather_kv(
                image_key,
                image_value,
            )
        if joint_first:
            joint_tokens = joint_query.shape[1]
            output = _sdpa(
                torch.cat([joint_query, image_query], dim=1),
                torch.cat([joint_key, full_image_key], dim=1),
                torch.cat([joint_value, full_image_value], dim=1),
            )
            return (
                output[:, :joint_tokens].contiguous(),
                output[:, joint_tokens:].contiguous(),
            )
        output = _sdpa(
            torch.cat([image_query, joint_query], dim=1),
            torch.cat([full_image_key, joint_key], dim=1),
            torch.cat([full_image_value, joint_value], dim=1),
        )
        return (
            output[:, local_image_tokens:].contiguous(),
            output[:, :local_image_tokens].contiguous(),
        )


class ImageSelfAttention:
    """Attention for a sequence sharded across one active EPE lane."""

    def __init__(self, mode: str = "agkv") -> None:
        self.mode, _ = resolve_context_parallel_config(mode)

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        lane_process_group: object | None,
        ulysses_topology: UlyssesTopology | None = None,
        agkv_transport: AgkvTransport | None = None,
    ) -> torch.Tensor:
        if self.mode == "ulysses":
            if ulysses_topology is None:
                raise ValueError("ulysses_topology is required for Ulysses attention")
            query, key, value, original_heads = _ulysses_qkv(
                query, key, value, ulysses_topology
            )
            output = _sdpa(query, key, value)
            return ulysses_topology.transport.all_to_all(
                output,
                1,
                2,
            )[:, :, :original_heads].contiguous()
        if agkv_transport is None:
            full_key, full_value = all_gather_sequence_pair(
                key,
                value,
                lane_process_group,
            )
        else:
            full_key, full_value = agkv_transport.all_gather_kv(key, value)
        return _sdpa(query, full_key, full_value)

    def finish_agkv(
        self,
        query: torch.Tensor,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode != "agkv":
            raise RuntimeError("finish_agkv is only valid for AGKV attention")
        return _sdpa(query, full_key, full_value)

    def finish_ulysses(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        topology: UlyssesTopology,
        output_tag: str | None = None,
    ) -> torch.Tensor:
        if self.mode != "ulysses":
            raise RuntimeError("finish_ulysses is only valid for Ulysses attention")
        output = _sdpa(query, key, value)
        transport = topology.transport
        if output_tag is not None:
            from .ulysses_transport import AsyncTaggedUlyssesTransport

            if not isinstance(transport, AsyncTaggedUlyssesTransport):
                raise RuntimeError("tagged output requires a capable Ulysses transport")
            return transport.all_to_all_tagged(
                output,
                1,
                2,
                tag=output_tag,
            ).contiguous()
        return transport.all_to_all(output, 1, 2).contiguous()
