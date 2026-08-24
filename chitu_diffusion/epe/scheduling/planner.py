from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .cost import (
    CalibratedStepCostModel,
    MeasuredStepCostModel,
    MeasuredTransferCostModel,
    RuntimeCostCalibrator,
)
from .policy import EpeSchedulingPolicy
from .types import RequestProfile, SchedulableRequest


@dataclass(frozen=True, slots=True)
class EpeRequest:
    """Model-neutral request view projected into the measured EPE planner."""

    request_id: str
    image_tokens: int
    total_steps: int
    current_step: int
    submitted_at_ms: float
    deadline_at_ms: float | None = None
    priority: int = 0
    batch_size: int = 1
    cfg_conditions: int = 1
    state_bytes: int = 0
    state_tensor_count: int = 1
    current_ranks: tuple[int, ...] | None = None

    @property
    def remaining_steps(self) -> int:
        return max(0, self.total_steps - self.current_step)

    def to_schedulable(self) -> SchedulableRequest:
        return SchedulableRequest(
            request_id=self.request_id,
            submitted_at=self.submitted_at_ms / 1000.0,
            deadline_at=(
                None if self.deadline_at_ms is None else self.deadline_at_ms / 1000.0
            ),
            priority=int(self.priority),
            profile=RequestProfile(
                total_steps=self.total_steps,
                completed_steps=self.current_step,
                image_tokens=self.image_tokens,
                attributes={
                    "batch_size": self.batch_size,
                    "conditions": self.cfg_conditions,
                    "state_bytes": self.state_bytes,
                    "state_tensor_count": self.state_tensor_count,
                },
            ),
            current_ranks=self.current_ranks,
        )


@dataclass(frozen=True, slots=True)
class EpePhaseAssignment:
    request_id: str
    ranks: tuple[int, ...]
    cp_width: int
    start_step: int
    steps: int
    switched: bool
    predicted_step_ms: float
    predicted_phase_ms: float
    image_tokens: int
    batch_size: int
    cfg_conditions: int
    cost_features: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "ranks": list(self.ranks),
            "cp_width": self.cp_width,
            "start_step": self.start_step,
            "steps": self.steps,
            "switched": self.switched,
            "predicted_step_ms": self.predicted_step_ms,
            "predicted_phase_ms": self.predicted_phase_ms,
            "image_tokens": self.image_tokens,
            "batch_size": self.batch_size,
            "cfg_conditions": self.cfg_conditions,
            "cost_features": dict(self.cost_features),
        }


class EpeSchedulingModule(torch.nn.Module):
    """Shared measured-cost scheduling state for image diffusion models."""

    def __init__(
        self,
        parallel: Any,
        *,
        policy: str = "elastic",
        switch_allowed_until_step: int = 0,
        pulse_steps: int = 1,
        balanced_k: bool = True,
        starvation_ms: float = 30_000.0,
        deadline_guard_ms: float = 0.0,
        min_resize_gain_ms: float = 0.0,
        resize_hysteresis_ms: float = 0.0,
        resize_control_cost_ms: float = 0.0,
        max_active_requests: int | None = None,
        online_calibration: bool = True,
        cost_model: MeasuredStepCostModel | None = None,
    ) -> None:
        super().__init__()
        if pulse_steps < 1:
            raise ValueError("pulse_steps must be >= 1")
        if deadline_guard_ms < 0:
            raise ValueError("deadline_guard_ms must be non-negative")
        if min(
            min_resize_gain_ms,
            resize_hysteresis_ms,
            resize_control_cost_ms,
        ) < 0:
            raise ValueError("resize cost and hysteresis controls must be non-negative")
        if max_active_requests is not None and max_active_requests < 1:
            raise ValueError("max_active_requests must be positive")

        self.parallel = parallel
        self.policy = str(policy)
        self.switch_allowed_until_step = int(switch_allowed_until_step)
        self.pulse_steps = int(pulse_steps)
        self.balanced_k = bool(balanced_k)
        self.starvation_ms = float(starvation_ms)
        self.deadline_guard_ms = float(deadline_guard_ms)
        self.min_resize_gain_ms = float(min_resize_gain_ms)
        self.resize_hysteresis_ms = float(resize_hysteresis_ms)
        self.resize_control_cost_ms = float(resize_control_cost_ms)
        self.max_active_requests = (
            None if max_active_requests is None else int(max_active_requests)
        )
        self.cost_model = cost_model or MeasuredStepCostModel()
        self.transfer_cost_model = MeasuredTransferCostModel()
        self.calibrator = RuntimeCostCalibrator(
            self.cost_model,
            enabled=online_calibration,
            warmup_skip=0,
        )
        self._calibrated_cost_model = CalibratedStepCostModel(
            self.cost_model,
            self.calibrator,
        )
        self._planner: EpeSchedulingPolicy | None = None
        self._planner_signature: tuple[object, ...] | None = None

    @property
    def scheduling_widths(self) -> tuple[int, ...]:
        return self.scheduling_policy.constraints.allowed_widths

    @property
    def scheduling_policy(self) -> EpeSchedulingPolicy:
        tp_topology = getattr(self.parallel, "tensor_parallel", None)
        tp_degree = int(getattr(tp_topology, "degree", 1))
        cp_world_size = int(
            getattr(
                self.parallel,
                "cp_world_size",
                self.parallel.world_size // tp_degree,
            )
        )
        signature = (
            self.policy,
            self.switch_allowed_until_step,
            self.balanced_k,
            self.starvation_ms,
            self.deadline_guard_ms,
            self.max_active_requests,
            tuple(self.parallel.allowed_widths),
            cp_world_size,
            tp_degree,
        )
        if self._planner is None or self._planner_signature != signature:
            self._planner = EpeSchedulingPolicy(
                world_size=cp_world_size,
                tp_degree=tp_degree,
                allowed_lane_widths=self.parallel.allowed_widths,
                cost_model=self._calibrated_cost_model,
                transfer_cost_model=self.transfer_cost_model,
                strategy=self.policy,
                switch_allowed_until_step=self.switch_allowed_until_step,
                balanced_k=self.balanced_k,
                starvation_ms=self.starvation_ms,
                deadline_guard_ms=self.deadline_guard_ms,
                min_resize_gain_ms=self.min_resize_gain_ms,
                resize_hysteresis_ms=self.resize_hysteresis_ms,
                resize_control_cost_ms=self.resize_control_cost_ms,
                max_active_requests=self.max_active_requests,
            )
            self._planner_signature = signature
        return self._planner

    def rank_lane(self, width: int) -> tuple[int, ...]:
        tp_topology = getattr(self.parallel, "tensor_parallel", None)
        tp_degree = int(getattr(tp_topology, "degree", 1))
        cp_world_size = int(
            getattr(
                self.parallel,
                "cp_world_size",
                self.parallel.world_size // tp_degree,
            )
        )
        cp_rank = int(
            getattr(self.parallel, "cp_rank", self.parallel.rank % cp_world_size)
        )
        tp_plane = int(
            getattr(self.parallel, "tp_plane", self.parallel.rank // cp_world_size)
        )
        offset = (cp_rank // width) * width
        plane_start = tp_plane * cp_world_size
        return tuple(
            range(plane_start + offset, plane_start + offset + width)
        )

    def initialize_cost_model(self, rows: Iterable[dict[str, Any]]) -> None:
        self.cost_model.initialize(rows)

    def initialize_transfer_cost_model(self, rows: Iterable[dict[str, Any]]) -> None:
        self.transfer_cost_model.initialize(rows)

    def predict_step_ms(self, request: EpeRequest, width: int) -> float:
        return self._calibrated_cost_model.predict_step_ms(
            sequence_length=request.image_tokens,
            width=width,
            batch_size=request.batch_size,
            conditions=request.cfg_conditions,
        )

    def observe_assignment(
        self,
        assignment: dict[str, Any],
        measured_step_ms: float,
    ) -> None:
        self.calibrator.observe(
            sequence_length=int(assignment["image_tokens"]),
            width=int(assignment.get("cp_width", len(assignment["ranks"]))),
            batch_size=int(assignment["batch_size"]),
            conditions=int(assignment["cfg_conditions"]),
            measured_step_ms=float(measured_step_ms),
            cost_features=assignment.get("cost_features"),
        )

    def observe_schedulable(
        self,
        request: SchedulableRequest,
        ranks: tuple[int, ...],
        measured_step_ms: float,
    ) -> None:
        profile = request.profile
        sequence_length = profile.image_tokens
        if sequence_length is None:
            sequence_length = profile.attributes.get("sequence_length")
        if sequence_length is None:
            raise ValueError("online calibration requires a sequence length")
        self.calibrator.observe(
            sequence_length=int(sequence_length),
            width=len(
                {
                    int(rank)
                    % int(
                        getattr(
                            self.parallel,
                            "cp_world_size",
                            self.parallel.world_size,
                        )
                    )
                    for rank in ranks
                }
            ),
            batch_size=int(profile.attributes.get("batch_size", 1)),
            conditions=int(profile.attributes.get("conditions", 1)),
            measured_step_ms=float(measured_step_ms),
            cost_features=profile.attributes.get("cost_features"),
        )

    def plan_phase(
        self,
        requests: Iterable[EpeRequest],
        *,
        now_ms: float | None = None,
    ) -> list[EpePhaseAssignment]:
        active = [request for request in requests if request.remaining_steps > 0]
        if not active:
            return []
        by_id = {request.request_id: request for request in active}
        plans = self.scheduling_policy.plan_at(
            [request.to_schedulable() for request in active],
            pulse_steps=self.pulse_steps,
            now_ms=(time.time() * 1000.0 if now_ms is None else float(now_ms)),
        )
        assignments = []
        for plan in plans:
            request = by_id[plan.request_id]
            assignments.append(
                EpePhaseAssignment(
                    request_id=request.request_id,
                    ranks=tuple(plan.lane_ranks or ()),
                    cp_width=int(
                        plan.metadata.get(
                            "cp_width",
                            len(plan.lane_ranks or ()),
                        )
                    ),
                    start_step=request.current_step,
                    steps=plan.steps,
                    switched=bool(plan.metadata.get("switched", False)),
                    predicted_step_ms=float(plan.metadata["predicted_step_ms"]),
                    predicted_phase_ms=float(plan.metadata["predicted_phase_ms"]),
                    image_tokens=request.image_tokens,
                    batch_size=request.batch_size,
                    cfg_conditions=request.cfg_conditions,
                    cost_features=dict(
                        plan.metadata.get("cost_features", {})
                    ),
                )
            )
        return assignments
