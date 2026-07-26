from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .topology import UspTopology
from .usp import DynamicUspAttention


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


def _all_gather_sequence(
    tensor: torch.Tensor,
    process_group: object | None,
) -> torch.Tensor:
    world_size = dist.get_world_size(process_group) if process_group is not None else 1
    if world_size == 1:
        return tensor
    pieces = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(pieces, tensor.contiguous(), group=process_group)
    return torch.cat(pieces, dim=1).contiguous()


class ImageContextParallelAttention:
    """Attention for sequence-sharded image and replicated joint tokens."""

    def __init__(self, mode: str = "agkv") -> None:
        if mode not in {"agkv", "usp"}:
            raise ValueError("context-parallel mode must be one of: agkv, usp")
        self.mode = mode
        self._usp = DynamicUspAttention()

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
        usp_topology: UspTopology | None = None,
        joint_first: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(joint_output, local_image_output)`` in sequence-major layout."""
        if self.mode == "usp":
            if usp_topology is None:
                raise ValueError("usp_topology is required for USP attention")
            return self._usp(
                image_query,
                image_key,
                image_value,
                joint_query,
                joint_key,
                joint_value,
                topology=usp_topology,
            )

        local_image_tokens = image_query.shape[1]
        full_image_key = _all_gather_sequence(image_key, lane_process_group)
        full_image_value = _all_gather_sequence(image_value, lane_process_group)
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
