from __future__ import annotations

import torch
import torch.nn.functional as F

from .agkv_transport import AgkvTransport
from .nccl.agkv import all_gather_sequence_pair
from .topology import UspTopology
from .usp import DynamicUspAttention

CONTEXT_PARALLEL_MODES = ("agkv", "usp")


def resolve_context_parallel_config(
    mode: str = "agkv",
    ulysses_degree: int | None = None,
) -> tuple[str, int | None]:
    """Normalize the model-independent context-parallel backend settings."""
    normalized_mode = str(mode).lower()
    if normalized_mode not in CONTEXT_PARALLEL_MODES:
        choices = ", ".join(CONTEXT_PARALLEL_MODES)
        raise ValueError(f"context-parallel mode must be one of: {choices}")
    degree = (
        int(ulysses_degree)
        if ulysses_degree is not None
        else (None if normalized_mode == "usp" else 1)
    )
    if degree is not None and degree < 1:
        raise ValueError("ulysses_degree must be positive")
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


class ImageContextParallelAttention:
    """Attention for sequence-sharded image and replicated joint tokens."""

    def __init__(self, mode: str = "agkv") -> None:
        self.mode, _ = resolve_context_parallel_config(mode)
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
        agkv_transport: AgkvTransport | None = None,
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
    """Attention for a sequence sharded across one active EPAC lane."""

    def __init__(self, mode: str = "agkv") -> None:
        self.mode, _ = resolve_context_parallel_config(mode)
        self._usp = DynamicUspAttention()

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        lane_process_group: object | None,
        usp_topology: UspTopology | None = None,
        agkv_transport: AgkvTransport | None = None,
    ) -> torch.Tensor:
        if self.mode == "usp":
            if usp_topology is None:
                raise ValueError("usp_topology is required for USP attention")
            return self._usp.self_attention(
                query,
                key,
                value,
                topology=usp_topology,
            )
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

    def finish_usp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        topology: UspTopology,
        output_tag: str | None = None,
    ) -> torch.Tensor:
        if self.mode != "usp":
            raise RuntimeError("finish_usp is only valid for USP attention")
        return self._usp.finish_self_attention(
            query,
            key,
            value,
            topology=topology,
            output_tag=output_tag,
        )
