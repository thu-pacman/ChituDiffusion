from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .attention_backend import create_varlen_attention_backend
from .topology import UspTopology

# The Ulysses layout swap and joint-ring schedule follow xDiT/xFuser's
# Apache-2.0 implementation, adapted to take EPAC lane groups per call instead
# of reading yunchang's process-global PROCESS_GROUP.


def _all_to_all_4d(
    tensor: torch.Tensor,
    *,
    group: object | None,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    world_size = dist.get_world_size(group) if group is not None else 1
    if world_size == 1:
        return tensor
    batch, sequence, heads, head_dim = tensor.shape
    if scatter_dim == 2 and gather_dim == 1:
        if heads % world_size:
            raise ValueError(
                f"attention heads {heads} must be divisible by Ulysses degree {world_size}"
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
                f"attention sequence {sequence} must be divisible by Ulysses degree {world_size}"
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
        "USP supports only heads-to-sequence (2,1) and sequence-to-heads (1,2)"
    )


def _gather_replicated_heads(
    tensor: torch.Tensor,
    topology: UspTopology,
) -> torch.Tensor:
    if topology.ulysses_degree == 1:
        return tensor
    if tensor.shape[1] == 0:
        return tensor.new_empty(
            tensor.shape[0],
            0,
            tensor.shape[2] * topology.ulysses_degree,
            tensor.shape[3],
        )
    pieces = [torch.empty_like(tensor) for _ in range(topology.ulysses_degree)]
    dist.all_gather(
        pieces,
        tensor.contiguous(),
        group=topology.ulysses_process_group,
    )
    return torch.cat(pieces, dim=2).contiguous()


def _local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    if query.is_cuda:
        if query.shape[:2] != key.shape[:2] or query.shape[:2] != value.shape[:2]:
            raise ValueError("dense self-attention Q/K/V sequence shapes must match")
        batch, sequence, heads, head_dim = query.shape
        cu_seqlens = torch.arange(
            0,
            (batch + 1) * sequence,
            sequence,
            device=query.device,
            dtype=torch.int32,
        )
        output = create_varlen_attention_backend("auto").forward_varlen(
            query.reshape(batch * sequence, heads, head_dim),
            key.reshape(batch * sequence, heads, head_dim),
            value.reshape(batch * sequence, heads, head_dim),
            cu_seqlens=cu_seqlens,
            max_seqlen=sequence,
        )
        return output.reshape(batch, sequence, heads, head_dim)
    return F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)


def _joint_ring_attention(
    query: torch.Tensor,
    image_key: torch.Tensor,
    image_value: torch.Tensor,
    joint_key: torch.Tensor,
    joint_value: torch.Tensor,
    topology: UspTopology,
) -> torch.Tensor:
    if topology.ring_degree == 1:
        return _local_attention(
            query,
            torch.cat([image_key, joint_key], dim=1),
            torch.cat([image_value, joint_value], dim=1),
        )
    if not query.is_cuda:
        key_pieces = [torch.empty_like(image_key) for _ in range(topology.ring_degree)]
        value_pieces = [
            torch.empty_like(image_value) for _ in range(topology.ring_degree)
        ]
        dist.all_gather(
            key_pieces,
            image_key.contiguous(),
            group=topology.ring_process_group,
        )
        dist.all_gather(
            value_pieces,
            image_value.contiguous(),
            group=topology.ring_process_group,
        )
        return _local_attention(
            query,
            torch.cat([*key_pieces, joint_key], dim=1),
            torch.cat([*value_pieces, joint_value], dim=1),
        )

    raise RuntimeError(
        "CUDA Ring attention is not available: the retired yunchang implementation "
        "must be replaced by native P2P plus LSE merging. Use pure Ulysses "
        "(ring_degree=1) for this model."
    )


class DynamicUspAttention:
    """Ulysses x Ring attention whose process groups are selected per call."""

    @staticmethod
    def _pad_heads(tensor: torch.Tensor, pad_heads: int) -> torch.Tensor:
        if pad_heads <= 0:
            return tensor
        padding = tensor.new_zeros(
            *tensor.shape[:2],
            pad_heads,
            tensor.shape[-1],
        )
        return torch.cat([tensor, padding], dim=2)

    def __call__(
        self,
        image_query: torch.Tensor,
        image_key: torch.Tensor,
        image_value: torch.Tensor,
        joint_query: torch.Tensor,
        joint_key: torch.Tensor,
        joint_value: torch.Tensor,
        *,
        topology: UspTopology,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(joint_output, local_image_output)`` in the input layout."""
        original_heads = image_query.shape[2]
        pad_heads = (
            topology.ulysses_degree - original_heads % topology.ulysses_degree
        ) % topology.ulysses_degree
        tensors = (
            image_query,
            image_key,
            image_value,
            joint_query,
            joint_key,
            joint_value,
        )
        (
            image_query,
            image_key,
            image_value,
            joint_query,
            joint_key,
            joint_value,
        ) = tuple(self._pad_heads(tensor, pad_heads) for tensor in tensors)

        image_query = _all_to_all_4d(
            image_query,
            group=topology.ulysses_process_group,
            scatter_dim=2,
            gather_dim=1,
        )
        image_key = _all_to_all_4d(
            image_key,
            group=topology.ulysses_process_group,
            scatter_dim=2,
            gather_dim=1,
        )
        image_value = _all_to_all_4d(
            image_value,
            group=topology.ulysses_process_group,
            scatter_dim=2,
            gather_dim=1,
        )
        padded_heads = original_heads + pad_heads
        heads_per_rank = padded_heads // topology.ulysses_degree
        head_start = topology.ulysses_rank * heads_per_rank
        head_stop = head_start + heads_per_rank
        joint_query = joint_query[:, :, head_start:head_stop].contiguous()
        joint_key = joint_key[:, :, head_start:head_stop].contiguous()
        joint_value = joint_value[:, :, head_start:head_stop].contiguous()

        full_image_tokens = image_query.shape[1]
        output = _joint_ring_attention(
            torch.cat([image_query, joint_query], dim=1),
            image_key,
            image_value,
            joint_key,
            joint_value,
            topology,
        )
        image_output = _all_to_all_4d(
            output[:, :full_image_tokens],
            group=topology.ulysses_process_group,
            scatter_dim=1,
            gather_dim=2,
        )
        joint_output = _gather_replicated_heads(
            output[:, full_image_tokens:],
            topology,
        )
        if pad_heads:
            image_output = image_output[:, :, :original_heads]
            joint_output = joint_output[:, :, :original_heads]
        return joint_output.contiguous(), image_output.contiguous()

    def self_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        topology: UspTopology,
    ) -> torch.Tensor:
        """Run USP for a sequence-sharded self-attention tensor."""
        empty = query.new_empty(query.shape[0], 0, query.shape[2], query.shape[3])
        _, output = self(
            query,
            key,
            value,
            empty,
            empty,
            empty,
            topology=topology,
        )
        return output
