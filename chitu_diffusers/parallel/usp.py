from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

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
    if tensor.is_cuda:
        from yunchang.comm.all_to_all import SeqAllToAll4D

        return SeqAllToAll4D.apply(group, tensor, scatter_dim, gather_dim)

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
        from flash_attn import flash_attn_func

        return flash_attn_func(
            query,
            key,
            value,
            dropout_p=0.0,
            causal=False,
        )
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

    from yunchang.kernels import AttnType, select_flash_attn_impl
    from yunchang.ring.utils import RingComm, update_out_and_lse

    comm = RingComm(topology.ring_process_group)
    key = image_key.contiguous()
    value = image_value.contiguous()
    output = None
    output_lse = None
    attention = select_flash_attn_impl(AttnType.FA, stage="fwd-only")
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_key = comm.send_recv(key)
            next_value = comm.send_recv(value)
            comm.commit()
        if step + 1 == comm.world_size:
            block_key = torch.cat([key, joint_key], dim=1)
            block_value = torch.cat([value, joint_value], dim=1)
        else:
            block_key = key
            block_value = value
        block_output, block_lse = attention(
            query,
            block_key,
            block_value,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
        output, output_lse = update_out_and_lse(
            output,
            output_lse,
            block_output,
            block_lse,
        )
        if step + 1 != comm.world_size:
            comm.wait()
            key = next_key
            value = next_value
    return output.to(query.dtype)


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
