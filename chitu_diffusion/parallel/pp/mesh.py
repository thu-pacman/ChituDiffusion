"""Static CFG × PP × CP mesh used only by the FPP request runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import torch.distributed as dist

from .topology import PipelineTopology


@dataclass
class FppMesh:
    pipeline: PipelineTopology
    cp_degree: int = 1
    cp_rank: int = 0
    cfg_degree: int = 1
    cfg_rank: int = 0
    cp_group: object | None = None
    cfg_group: object | None = None
    feedback_group: object | None = None
    owned_groups: list = field(default_factory=list, repr=False)
    boundary_group: object | None = None

    @property
    def output_rank(self) -> int:
        return (self.pipeline.degree - 1) * self.cp_degree

    @classmethod
    def create(cls, pp: int, cp: int = 1, cfg: int = 1) -> FppMesh:
        if any(type(n) is not int or n < 1 for n in (pp, cp, cfg)) or cfg not in (1, 2):
            raise ValueError("FPP needs positive PP/CP degrees and CFG degree 1 or 2")
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        if pp * cp * cfg != world:
            raise ValueError(
                f"FPP world size {world} must equal CFG × PP × CP = {cfg * pp * cp}"
            )
        cfg_rank, offset = divmod(rank, pp * cp)
        _, cp_rank = divmod(offset, cp)
        owned = []

        def group(ranks):
            if len(ranks) == 1:
                return None
            result = dist.new_group(ranks, timeout=timedelta(seconds=180))
            if rank in ranks:
                owned.append(result)
                return result
            return None

        pipeline_group = feedback_group = cp_group = cfg_group = None
        pipeline_ranks = None
        # All ranks create groups in exactly this order. Feedback uses a
        # separate communicator to avoid a two-stage NCCL send/receive cycle.
        for g in range(cfg):
            for c in range(cp):
                ranks = tuple((g * pp + p) * cp + c for p in range(pp))
                forward = group(ranks)
                feedback = group(ranks)
                if rank in ranks:
                    pipeline_ranks = ranks
                    pipeline_group, feedback_group = forward, feedback
        for g in range(cfg):
            for p in range(pp):
                ranks = tuple((g * pp + p) * cp + c for c in range(cp))
                result = group(ranks)
                if rank in ranks:
                    cp_group = result
        for p in range(pp):
            for c in range(cp):
                ranks = tuple((g * pp + p) * cp + c for g in range(cfg))
                result = group(ranks)
                if rank in ranks:
                    cfg_group = result
        boundary_group = None
        if world > 1 and dist.get_backend() == "nccl":
            boundary_group = dist.new_group(
                list(range(world)), backend="gloo", timeout=timedelta(seconds=180)
            )
            owned.append(boundary_group)
        return cls(
            PipelineTopology(pipeline_ranks, rank, pipeline_group),
            cp,
            cp_rank,
            cfg,
            cfg_rank,
            cp_group,
            cfg_group,
            feedback_group,
            owned,
            boundary_group,
        )

    def close(self) -> None:
        for group in reversed(self.owned_groups):
            dist.destroy_process_group(group)
        self.owned_groups.clear()
