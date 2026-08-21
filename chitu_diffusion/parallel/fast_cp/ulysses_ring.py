from __future__ import annotations

import torch

from ..nccl.ring import gather_replicated_heads, joint_ring_attention
from ..nccl.ulysses import TorchUlyssesTransport
from ..topology import UspTopology
from ..ulysses_transport import AsyncTaggedUlyssesTransport, UlyssesTransport


def _all_to_all_4d(
    tensor: torch.Tensor,
    *,
    group: object | None,
    scatter_dim: int,
    gather_dim: int,
    transport: UlyssesTransport | None = None,
) -> torch.Tensor:
    selected = transport or TorchUlyssesTransport(group)
    return selected.all_to_all(tensor, scatter_dim, gather_dim)


class FastUlyssesRingAttention:
    """Ulysses redistribution around an NCCL Ring attention phase.

    The Ulysses transport may be the node-local NVSHMEM implementation while
    ``joint_ring_attention`` stays on NCCL and may span nodes. Process groups
    and the selected U×R factorization are supplied by ``UspTopology``.
    """

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
        transport = topology.ulysses_transport or TorchUlyssesTransport(
            topology.ulysses_process_group
        )
        transport.reset()
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

        image_query, image_key, image_value = transport.all_to_all_qkv(
            image_query,
            image_key,
            image_value,
        )
        padded_heads = original_heads + pad_heads
        heads_per_rank = padded_heads // topology.ulysses_degree
        head_start = topology.ulysses_rank * heads_per_rank
        head_stop = head_start + heads_per_rank
        joint_query = joint_query[:, :, head_start:head_stop].contiguous()
        joint_key = joint_key[:, :, head_start:head_stop].contiguous()
        joint_value = joint_value[:, :, head_start:head_stop].contiguous()

        full_image_tokens = image_query.shape[1]
        output = joint_ring_attention(
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
            transport=transport,
        )
        joint_output = gather_replicated_heads(
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
        """Run U×R for a sequence-sharded self-attention tensor."""
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

    def finish_self_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        topology: UspTopology,
        output_tag: str | None = None,
    ) -> torch.Tensor:
        """Finish U×R after Q/K/V have entered the Ulysses layout."""
        transport = topology.ulysses_transport or TorchUlyssesTransport(
            topology.ulysses_process_group
        )
        empty = query.new_empty(query.shape[0], 0, query.shape[2], query.shape[3])
        output = joint_ring_attention(
            query,
            key,
            value,
            empty,
            empty,
            topology,
        )
        if output_tag is not None:
            if not isinstance(transport, AsyncTaggedUlyssesTransport):
                raise RuntimeError("tagged output requires a capable Ulysses transport")
            return transport.all_to_all_tagged(
                output,
                1,
                2,
                tag=output_tag,
            ).contiguous()
        return _all_to_all_4d(
            output,
            group=topology.ulysses_process_group,
            scatter_dim=1,
            gather_dim=2,
            transport=transport,
        ).contiguous()
