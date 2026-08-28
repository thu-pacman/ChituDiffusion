"""Configurable TPxCFGxCPxEP geometry bound to the shared VAEP decode group."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ...parallel.cp import CfgParallelTopology, EpeParallelContext, SequencePartition
from ...parallel.ep import build_expert_parallel_topology, expert_parallel_ranks
from ...parallel.vae import VaeParallelPlacement, create_vae_parallel_placement

ATTENTION_MODES = ("agkv", "ulysses")


@dataclass(frozen=True, slots=True)
class HunyuanImage3ParallelPlan:
    """One launch-time parallel shape for a single-node Hunyuan Image 3 stage.

    The stage world is ``tensor x cfg x context``. Each tensor-parallel plane is
    a contiguous rank range; CFG2 splits that plane into a conditional and an
    unconditional context-parallel lane, while CFG1 runs both conditions on
    every rank of the plane. Expert parallelism is an independent axis, and VAE
    parallelism comes from the stage-wide plan rather than this plan.
    """

    tensor_parallel_degree: int = 2
    cfg_parallel_degree: int = 2
    context_parallel_degree: int = 2
    expert_parallel_degree: int = 4
    attention_mode: str = "agkv"

    def __post_init__(self) -> None:
        for name in (
            "tensor_parallel_degree",
            "cfg_parallel_degree",
            "context_parallel_degree",
            "expert_parallel_degree",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.cfg_parallel_degree > 2:
            raise ValueError(
                "cfg_parallel_degree must be 1 or 2; the released prompt carries "
                "exactly one conditional and one unconditional branch"
            )
        if self.plane_width % self.expert_parallel_degree:
            raise ValueError(
                "expert_parallel_degree must divide cfg_parallel_degree x "
                f"context_parallel_degree ({self.plane_width}), got "
                f"{self.expert_parallel_degree}"
            )
        if self.attention_mode not in ATTENTION_MODES:
            raise ValueError(
                "attention_mode must be one of: " + ", ".join(ATTENTION_MODES)
            )

    @property
    def world_size(self) -> int:
        return (
            self.tensor_parallel_degree
            * self.cfg_parallel_degree
            * self.context_parallel_degree
        )

    @property
    def plane_width(self) -> int:
        """Ranks in one tensor-parallel plane, across every CFG branch."""

        return self.cfg_parallel_degree * self.context_parallel_degree

    @property
    def lane_widths(self) -> tuple[int, ...]:
        """Contiguous plane-local widths that need their own process group."""

        return tuple(
            sorted(
                {
                    1,
                    self.context_parallel_degree,
                    self.expert_parallel_degree,
                    self.plane_width,
                }
            )
        )

    def describe(self) -> str:
        return (
            f"TP{self.tensor_parallel_degree}"
            f"xCFG{self.cfg_parallel_degree}"
            f"xCP{self.context_parallel_degree}"
            f"xEP{self.expert_parallel_degree}"
        )

    def validate_model_shape(
        self,
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        num_experts: int,
        mlp_widths: Sequence[int] = (),
    ) -> None:
        """Reject checkpoints this plan cannot split across the stage."""

        # CFG branches replicate the weights, so heads only split over TP x CP.
        # Ulysses trades sequence for heads; AGKV does not, but keeping one rule
        # lets a stage switch attention mode without reloading a new shape.
        span = self.tensor_parallel_degree * self.context_parallel_degree
        for name, heads in (
            ("num_attention_heads", num_attention_heads),
            ("num_key_value_heads", num_key_value_heads),
        ):
            if heads % span:
                raise ValueError(
                    f"{name}={heads} is not divisible by tensor-parallel x "
                    f"context-parallel degree {span}"
                )
        if num_experts % self.expert_parallel_degree:
            raise ValueError(
                f"num_experts={num_experts} is not divisible by "
                f"expert_parallel_degree {self.expert_parallel_degree}"
            )
        for width in mlp_widths:
            if int(width) % self.tensor_parallel_degree:
                raise ValueError(
                    f"feed-forward width {int(width)} is not divisible by "
                    f"tensor_parallel_degree {self.tensor_parallel_degree}"
                )


def resolve_parallel_plan(
    *,
    world_size: int,
    tensor_parallel_degree: int,
    cfg_parallel_degree: int,
    context_parallel_degree: int | None = None,
    expert_parallel_degree: int | None = None,
    attention_mode: str = "agkv",
) -> HunyuanImage3ParallelPlan:
    """Fill in the degrees the stage world already determines.

    A tensor-parallel plane holds ``world_size / tensor`` ranks and CFG splits
    it, so the context-parallel degree follows unless the caller pins it. Expert
    parallelism defaults to the whole plane, which keeps the checkpoint sharded
    as widely as possible.
    """

    if tensor_parallel_degree < 1 or cfg_parallel_degree < 1:
        raise ValueError("tensor and CFG parallel degrees must be positive")
    branch_ranks = tensor_parallel_degree * cfg_parallel_degree
    if world_size < 1 or world_size % branch_ranks:
        raise ValueError(
            f"tensor-parallel x CFG-parallel degree {branch_ranks} must divide "
            f"the stage world size {world_size}"
        )
    context = context_parallel_degree or world_size // branch_ranks
    if branch_ranks * context != world_size:
        raise ValueError(
            f"TP{tensor_parallel_degree}xCFG{cfg_parallel_degree}xCP{context} "
            f"needs {branch_ranks * context} ranks, got {world_size}"
        )
    plane_width = cfg_parallel_degree * context
    return HunyuanImage3ParallelPlan(
        tensor_parallel_degree=tensor_parallel_degree,
        cfg_parallel_degree=cfg_parallel_degree,
        context_parallel_degree=context,
        expert_parallel_degree=expert_parallel_degree or plane_width,
        attention_mode=attention_mode,
    )


class HunyuanImage3ParallelRuntime:
    """Bind one rank to its tensor, context, expert, and VAE decode groups."""

    def __init__(
        self,
        *,
        plan: HunyuanImage3ParallelPlan,
        context: EpeParallelContext,
        num_experts: int,
        owns_context: bool,
        vae_placement: VaeParallelPlacement | None = None,
    ) -> None:
        if context.world_size != plan.world_size:
            raise ValueError(
                f"{plan.describe()} requires exactly {plan.world_size} ranks, got "
                f"{context.world_size}"
            )
        if context.tensor_parallel.degree != plan.tensor_parallel_degree:
            raise ValueError(
                f"stage tensor-parallel degree {context.tensor_parallel.degree} "
                f"does not match the plan's {plan.tensor_parallel_degree}"
            )
        plane = context.active
        if plane.width != plan.plane_width:
            raise ValueError(
                f"active lane width {plane.width} does not match the configured "
                f"CFG x context plane width {plan.plane_width}"
            )
        self.plan = plan
        self.context = context
        self.plane_ranks = plane.ranks
        self._owns_context = bool(owns_context)
        self.cfg = self._bind_cfg_branch(plane.ranks)
        self.cp = context.lane_topology(self.context_parallel_ranks)
        expert_lane = context.lane_topology(
            expert_parallel_ranks(
                plane.ranks,
                rank=context.rank,
                degree=plan.expert_parallel_degree,
            )
        )
        self.experts = build_expert_parallel_topology(
            num_experts=num_experts,
            ranks=expert_lane.ranks,
            rank=context.rank,
            process_group=expert_lane.process_group,
        )
        self.vae_placement = vae_placement or VaeParallelPlacement()

    def _bind_cfg_branch(
        self, plane_ranks: tuple[int, ...]
    ) -> CfgParallelTopology | None:
        """Return this rank's CFG branch, or None when it runs both."""

        if self.plan.cfg_parallel_degree == 1:
            return None
        cfg = self.context.cfg_parallel_topology(plane_ranks)
        if cfg is None:
            raise ValueError(
                f"CFG-2 topology was not initialized for TP plane {plane_ranks}"
            )
        if cfg.cp_width != self.plan.context_parallel_degree:
            raise ValueError(
                f"CFG branch CP width {cfg.cp_width} does not match "
                f"context_parallel_degree={self.plan.context_parallel_degree}"
            )
        return cfg

    @staticmethod
    def checkpoint_num_experts(model_path: str | Path) -> int:
        """Read the routed expert count from a local checkpoint config."""

        config = json.loads((Path(model_path) / "config.json").read_text())
        experts = config["num_experts"]
        if isinstance(experts, (list, tuple)):
            if len(set(experts)) != 1:
                raise ValueError(
                    "Hunyuan Image 3 expert parallelism requires the same expert "
                    "count in every layer"
                )
            return int(experts[0])
        return int(experts)

    @classmethod
    def from_torchrun(
        cls,
        *,
        plan: HunyuanImage3ParallelPlan,
        num_experts: int,
        vae_parallel_degree: int | None = None,
        vae_parallel_halo: int = 8,
    ) -> "HunyuanImage3ParallelRuntime":
        # Dense AGKV uses the generic transport adapter, but a lane narrower
        # than the plane is NCCL-only. Ulysses remains a correctness fallback
        # and also uses its explicit uneven exchange rather than a packed
        # transport.
        context = EpeParallelContext.from_torchrun(
            allowed_widths=plan.lane_widths,
            tensor_parallel_degree=plan.tensor_parallel_degree,
            ulysses_degree=plan.context_parallel_degree,
            ulysses_transport="torch",
            agkv_transport="torch",
        )
        return cls(
            plan=plan,
            context=context,
            num_experts=num_experts,
            owns_context=True,
            vae_placement=create_vae_parallel_placement(
                plan.world_size if vae_parallel_degree is None else vae_parallel_degree,
                halo=vae_parallel_halo,
            ),
        )

    @property
    def rank(self) -> int:
        return self.context.rank

    @property
    def tensor_parallel_rank(self) -> int:
        return self.context.tensor_parallel.rank_in_group

    @property
    def runs_both_cfg_branches(self) -> bool:
        """True when this rank denoises the conditional and unconditional halves."""

        return self.cfg is None

    @property
    def cfg_branch_index(self) -> int:
        return 0 if self.cfg is None else self.cfg.branch_index

    @property
    def context_parallel_ranks(self) -> tuple[int, ...]:
        return self.plane_ranks if self.cfg is None else self.cfg.cp_ranks

    @property
    def context_parallel_rank(self) -> int:
        if self.cfg is None:
            return self.plane_ranks.index(self.context.rank)
        return self.cfg.cp_rank

    @property
    def context_parallel_degree(self) -> int:
        return self.plan.context_parallel_degree

    @property
    def context_parallel_group(self) -> object | None:
        return self.cp.process_group

    def sequence_partition(self, total_length: int) -> SequencePartition:
        return SequencePartition.balanced(
            total_length,
            degree=self.context_parallel_degree,
            rank=self.context_parallel_rank,
        )

    def owns_checkpoint_expert(self, expert_index: int) -> bool:
        return self.experts.owns(expert_index)

    def close(self) -> None:
        self.vae_placement.close()
        if self._owns_context:
            self.context.close()


__all__ = [
    "ATTENTION_MODES",
    "HunyuanImage3ParallelPlan",
    "HunyuanImage3ParallelRuntime",
    "resolve_parallel_plan",
]
