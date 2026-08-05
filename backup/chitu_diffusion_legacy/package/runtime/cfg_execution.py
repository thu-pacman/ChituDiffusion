from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

import torch
import torch.distributed as dist


BranchState = TypeVar("BranchState")
Prediction = TypeVar("Prediction", bound=torch.Tensor)


@dataclass(frozen=True)
class CfgBranchBatch(Generic[BranchState]):
    cond: BranchState
    uncond: Optional[BranchState] = None


@dataclass(frozen=True)
class CfgBranchPredictions(Generic[Prediction]):
    cond: Prediction
    uncond: Optional[Prediction] = None


@dataclass(frozen=True)
class CfgExecutionSpec:
    guidance_enabled: bool
    cfp_degree: int
    cp_degree: int
    physical_cfg_batch: bool = False

    def __post_init__(self) -> None:
        if self.cfp_degree not in (1, 2):
            raise ValueError(f"cfp_degree must be 1 or 2, got {self.cfp_degree}")
        if self.cp_degree <= 0:
            raise ValueError(f"cp_degree must be positive, got {self.cp_degree}")
        if self.cfp_degree == 2 and not self.guidance_enabled:
            raise ValueError("CFP=2 requires guidance to be enabled")
        if self.cfp_degree == 2 and self.physical_cfg_batch:
            raise ValueError("physical CFG batching is only valid for CFP=1")

    @classmethod
    def from_live_topology(
        cls,
        *,
        guidance_enabled: bool,
        physical_cfg_batch: bool = False,
    ) -> "CfgExecutionSpec":
        from chitu_diffusion.core.distributed.parallel_state import (
            get_cp_group,
            get_live_lane_topology,
        )

        topology = get_live_lane_topology()
        if topology is None or not topology.active:
            cfp_degree = 1
            cp_degree = int(get_cp_group().group_size)
        else:
            cfp_degree = int(topology.cfp_degree)
            cp_degree = int(topology.cp_degree)
        if not guidance_enabled:
            cfp_degree = 1
        return cls(
            guidance_enabled=bool(guidance_enabled),
            cfp_degree=cfp_degree,
            cp_degree=cp_degree,
            physical_cfg_batch=bool(physical_cfg_batch),
        )


class CfgBranchExecutor:
    """Execute logical cond/uncond branches with local or CFP-2 placement."""

    def execute(
        self,
        *,
        spec: CfgExecutionSpec,
        branches: CfgBranchBatch[BranchState],
        forward_branch: Callable[[str, BranchState], Prediction],
        combine: Callable[[Prediction, Prediction], Prediction],
        forward_physical_batch: Optional[
            Callable[[CfgBranchBatch[BranchState]], CfgBranchPredictions[Prediction]]
        ] = None,
    ) -> Prediction:
        if not spec.guidance_enabled:
            return forward_branch("cond", branches.cond)
        if branches.uncond is None:
            raise ValueError("CFG guidance requires an unconditional branch")

        if spec.cfp_degree == 2:
            return self._execute_cfp2(
                branches=branches,
                forward_branch=forward_branch,
                combine=combine,
            )

        if spec.physical_cfg_batch:
            if forward_physical_batch is None:
                raise ValueError(
                    "physical_cfg_batch requires forward_physical_batch callback"
                )
            predictions = forward_physical_batch(branches)
            if predictions.uncond is None:
                raise ValueError("physical CFG batch did not return uncond prediction")
            return combine(predictions.cond, predictions.uncond)

        cond = forward_branch("cond", branches.cond)
        uncond = forward_branch("uncond", branches.uncond)
        return combine(cond, uncond)

    @staticmethod
    def _execute_cfp2(
        *,
        branches: CfgBranchBatch[BranchState],
        forward_branch: Callable[[str, BranchState], Prediction],
        combine: Callable[[Prediction, Prediction], Prediction],
    ) -> Prediction:
        from chitu_diffusion.core.distributed.parallel_state import get_active_cfg_group

        group = get_active_cfg_group()
        if group.group_size != 2:
            raise RuntimeError(
                f"CFP=2 requires a two-rank CFG group, got size={group.group_size}"
            )
        branch_name = "cond" if group.rank_in_group == 0 else "uncond"
        local_branch = branches.cond if branch_name == "cond" else branches.uncond
        assert local_branch is not None
        local_prediction = forward_branch(branch_name, local_branch).contiguous()
        gathered = [torch.empty_like(local_prediction) for _ in range(2)]
        dist.all_gather(gathered, local_prediction, group=group.gpu_group)
        return combine(gathered[0], gathered[1])
