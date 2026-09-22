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


def uses_fast_cp_transport(parallel: object, mode: str) -> bool:
    """Return whether the active lane uses the Fast CP transport for ``mode``."""
    normalized_mode, _ = resolve_context_parallel_config(mode)
    if normalized_mode == "ulysses":
        transport = parallel.active_ulysses.transport
        return getattr(transport, "name", None) == "fast_ulysses"
    transport = parallel.active_agkv_transport
    return getattr(transport, "name", None) == "fast_agkv"


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


def _group_world_size(process_group: object | None) -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(process_group)


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

    def __init__(self, mode: str = "agkv", *, planned: bool = False) -> None:
        self.mode, _ = resolve_context_parallel_config(mode)
        self.planned = bool(planned)
        self._prepared_attention = None

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cp_degree: int,
    ) -> torch.Tensor:
        if not self.planned:
            return _sdpa(query, key, value)
        if self._prepared_attention is None:
            from .fast.attention import prepare_attention

            self._prepared_attention = prepare_attention(
                query,
                key,
                value,
                cp_degree=cp_degree,
            )
        return self._prepared_attention(query, key, value)

    @property
    def attention_plan(self):
        prepared = self._prepared_attention
        return None if prepared is None else prepared.plan

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
        joint_sharded: bool = False,
        joint_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(joint_output, local_image_output)`` in sequence-major layout.

        A producer may supply preassembled ``joint_qkv`` for sharded text to
        avoid concatenation. Its token order must follow ``joint_first``.
        """
        if joint_qkv is not None and not joint_sharded:
            raise ValueError("preassembled joint QKV requires sharded joint tokens")
        if joint_sharded:
            if joint_qkv is not None:
                query, key, value = joint_qkv
                expected = (
                    image_query.shape[0],
                    image_query.shape[1] + joint_query.shape[1],
                    *image_query.shape[2:],
                )
                if any(t.shape != expected for t in joint_qkv):
                    raise ValueError("preassembled joint QKV must match both sequence shards")
            elif joint_first:
                query = torch.cat([joint_query, image_query], dim=1)
                key = torch.cat([joint_key, image_key], dim=1)
                value = torch.cat([joint_value, image_value], dim=1)
            else:
                query = torch.cat([image_query, joint_query], dim=1)
                key = torch.cat([image_key, joint_key], dim=1)
                value = torch.cat([image_value, joint_value], dim=1)
            joint_tokens = joint_query.shape[1]
            image_tokens = image_query.shape[1]
            if self.mode == "ulysses":
                if ulysses_topology is None:
                    raise ValueError("ulysses_topology is required for Ulysses attention")
                query, key, value, original_heads = _ulysses_qkv(
                    query, key, value, ulysses_topology
                )
                output = self._attend(
                    query,
                    key,
                    value,
                    cp_degree=ulysses_topology.degree,
                )
                output = ulysses_topology.transport.all_to_all(
                    output, 1, 2
                )[:, :, :original_heads]
            else:
                if agkv_transport is None:
                    key, value = all_gather_sequence_pair(
                        key, value, lane_process_group
                    )
                else:
                    key, value = agkv_transport.all_gather_kv(key, value)
                output = self._attend(
                    query,
                    key,
                    value,
                    cp_degree=_group_world_size(lane_process_group),
                )
            if joint_first:
                return (
                    output[:, :joint_tokens].contiguous(),
                    output[:, joint_tokens : joint_tokens + image_tokens].contiguous(),
                )
            return (
                output[:, image_tokens : image_tokens + joint_tokens].contiguous(),
                output[:, :image_tokens].contiguous(),
            )

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
                output = self._attend(
                    torch.cat([joint_query, image_query], dim=1),
                    torch.cat([joint_key, image_key], dim=1),
                    torch.cat([joint_value, image_value], dim=1),
                    cp_degree=topology.degree,
                )
                joint_output = output[:, :joint_tokens]
                image_output = output[:, joint_tokens:]
            else:
                output = self._attend(
                    torch.cat([image_query, joint_query], dim=1),
                    torch.cat([image_key, joint_key], dim=1),
                    torch.cat([image_value, joint_value], dim=1),
                    cp_degree=topology.degree,
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
            output = self._attend(
                torch.cat([joint_query, image_query], dim=1),
                torch.cat([joint_key, full_image_key], dim=1),
                torch.cat([joint_value, full_image_value], dim=1),
                cp_degree=_group_world_size(lane_process_group),
            )
            return (
                output[:, :joint_tokens].contiguous(),
                output[:, joint_tokens:].contiguous(),
            )
        output = self._attend(
            torch.cat([image_query, joint_query], dim=1),
            torch.cat([full_image_key, joint_key], dim=1),
            torch.cat([full_image_value, joint_value], dim=1),
            cp_degree=_group_world_size(lane_process_group),
        )
        return (
            output[:, local_image_tokens:].contiguous(),
            output[:, :local_image_tokens].contiguous(),
        )


class ImageSelfAttention:
    """Attention for a sequence sharded across one active EPE lane."""

    def __init__(self, mode: str = "agkv", *, planned: bool = False) -> None:
        self.mode, _ = resolve_context_parallel_config(mode)
        self.planned = bool(planned)
        self._prepared_attention = None

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cp_degree: int,
    ) -> torch.Tensor:
        if not self.planned:
            return _sdpa(query, key, value)
        if self._prepared_attention is None:
            from .fast.attention import prepare_attention

            self._prepared_attention = prepare_attention(
                query,
                key,
                value,
                cp_degree=cp_degree,
            )
        return self._prepared_attention(query, key, value)

    @property
    def attention_plan(self):
        prepared = self._prepared_attention
        return None if prepared is None else prepared.plan

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
            output = self._attend(
                query,
                key,
                value,
                cp_degree=ulysses_topology.degree,
            )
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
        return self._attend(
            query,
            full_key,
            full_value,
            cp_degree=_group_world_size(lane_process_group),
        )

    def finish_agkv(
        self,
        query: torch.Tensor,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode != "agkv":
            raise RuntimeError("finish_agkv is only valid for AGKV attention")
        local_tokens = max(int(query.shape[1]), 1)
        cp_degree = max(int(full_key.shape[1]) // local_tokens, 1)
        return self._attend(
            query,
            full_key,
            full_value,
            cp_degree=cp_degree,
        )

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
        output = self._attend(
            query,
            key,
            value,
            cp_degree=topology.degree,
        )
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
