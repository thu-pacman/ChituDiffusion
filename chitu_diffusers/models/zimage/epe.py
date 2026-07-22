from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.distributed as dist

from ...epac.cost import (
    CalibratedStepCostModel,
    MeasuredStepCostModel,
    MeasuredTransferCostModel,
    RuntimeCostCalibrator,
)
from ...epac.epe import EpeSchedulingPolicy
from ...epac.scheduling import RequestProfile, SchedulableRequest
from ...parallel import EpeParallelContext

logger = logging.getLogger(__name__)

EpeCostModel = MeasuredStepCostModel
EpeRuntimeCalibrator = RuntimeCostCalibrator


@dataclass(frozen=True, slots=True)
class EpeRequest:
    """Z-Image request view projected into the model-independent planner."""

    request_id: str
    image_tokens: int
    total_steps: int
    current_step: int
    submitted_at_ms: float
    deadline_at_ms: float | None = None
    batch_size: int = 1
    cfg_conditions: int = 2
    state_bytes: int = 0
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
            priority=0,
            profile=RequestProfile(
                total_steps=self.total_steps,
                completed_steps=self.current_step,
                image_tokens=self.image_tokens,
                attributes={
                    "batch_size": self.batch_size,
                    "conditions": self.cfg_conditions,
                    "state_bytes": self.state_bytes,
                },
            ),
            current_ranks=self.current_ranks,
        )


@dataclass(frozen=True, slots=True)
class EpePhaseAssignment:
    request_id: str
    ranks: tuple[int, ...]
    start_step: int
    steps: int
    switched: bool
    predicted_step_ms: float
    predicted_phase_ms: float
    image_tokens: int
    batch_size: int
    cfg_conditions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "ranks": list(self.ranks),
            "start_step": self.start_step,
            "steps": self.steps,
            "switched": self.switched,
            "predicted_step_ms": self.predicted_step_ms,
            "predicted_phase_ms": self.predicted_phase_ms,
            "image_tokens": self.image_tokens,
            "batch_size": self.batch_size,
            "cfg_conditions": self.cfg_conditions,
        }


class ZImageEpeModule(torch.nn.Module):
    """DiT-owned topology and warmup facade backed by EPAC core policy."""

    def __init__(
        self,
        parallel: EpeParallelContext,
        *,
        policy: str = "elastic",
        switch_allowed_until_step: int = 0,
        pulse_steps: int = 1,
        balanced_k: bool = True,
        starvation_ms: float = 30_000.0,
        deadline_guard_ms: float = 0.0,
        max_active_requests: int | None = None,
        attention_mode: str = "agkv",
        cost_model: object | None = None,
        online_calibration: bool = True,
    ) -> None:
        super().__init__()
        if pulse_steps < 1:
            raise ValueError("pulse_steps must be >= 1")
        if cost_model is not None:
            logger.warning(
                "Ignoring static EPE cost_model=%r; startup warmup initializes "
                "the measured cost model",
                cost_model,
            )
        self.parallel = parallel
        self.policy = str(policy)
        self.switch_allowed_until_step = int(switch_allowed_until_step)
        self.pulse_steps = int(pulse_steps)
        self.balanced_k = bool(balanced_k)
        self.starvation_ms = float(starvation_ms)
        self.deadline_guard_ms = float(deadline_guard_ms)
        if self.deadline_guard_ms < 0:
            raise ValueError("deadline_guard_ms must be non-negative")
        self.max_active_requests = (
            None if max_active_requests is None else int(max_active_requests)
        )
        if self.max_active_requests is not None and self.max_active_requests < 1:
            raise ValueError("max_active_requests must be positive")
        if attention_mode not in {"agkv", "usp"}:
            raise ValueError("attention_mode must be one of: agkv, usp")
        self.attention_mode = attention_mode
        self.cost_model = MeasuredStepCostModel()
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
        return self._get_planner().constraints.allowed_widths

    @property
    def scheduling_policy(self) -> EpeSchedulingPolicy:
        return self._get_planner()

    def _get_planner(self) -> EpeSchedulingPolicy:
        signature = (
            self.policy,
            self.switch_allowed_until_step,
            self.balanced_k,
            self.starvation_ms,
            self.deadline_guard_ms,
            self.max_active_requests,
            tuple(self.parallel.allowed_widths),
        )
        if self._planner is None or self._planner_signature != signature:
            self._planner = EpeSchedulingPolicy(
                world_size=self.parallel.world_size,
                allowed_lane_widths=self.parallel.allowed_widths,
                cost_model=self._calibrated_cost_model,
                transfer_cost_model=self.transfer_cost_model,
                strategy=self.policy,
                switch_allowed_until_step=self.switch_allowed_until_step,
                balanced_k=self.balanced_k,
                starvation_ms=self.starvation_ms,
                deadline_guard_ms=self.deadline_guard_ms,
                max_active_requests=self.max_active_requests,
            )
            self._planner_signature = signature
        return self._planner

    def _rank_lane(self, width: int) -> tuple[int, ...]:
        offset = (self.parallel.rank // width) * width
        return tuple(range(offset, offset + width))

    def initialize_cost_model(self, rows: Iterable[dict[str, Any]]) -> None:
        self.cost_model.initialize(rows)

    def initialize_transfer_cost_model(self, rows: Iterable[dict[str, Any]]) -> None:
        self.transfer_cost_model.initialize(rows)

    @torch.inference_mode()
    def warmup_transformer(
        self,
        transformer: torch.nn.Module,
        *,
        resolutions: Iterable[tuple[int, int]],
        steps: int = 5,
        vae_scale_factor: int = 8,
        text_tokens: int = 32,
        batch_size: int = 1,
        cfg_conditions: int = 2,
    ) -> dict[str, Any]:
        """Measure every configured resolution and canonical lane width."""
        if steps < 1:
            raise ValueError("EPE warmup steps must be >= 1")
        resolutions = tuple((int(height), int(width)) for height, width in resolutions)
        if not resolutions:
            raise ValueError("EPE warmup resolutions must not be empty")

        parameter = next(transformer.parameters())
        device = parameter.device
        dtype = parameter.dtype if parameter.is_floating_point() else torch.float32
        in_channels = int(
            getattr(transformer, "in_channels", transformer.config.in_channels)
        )
        cap_feat_dim = int(transformer.config.cap_feat_dim)
        scale = int(vae_scale_factor) * 2
        if scale <= 0:
            raise ValueError("vae_scale_factor must be positive")

        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        for height, width_pixels in resolutions:
            if height % scale or width_pixels % scale:
                raise ValueError(
                    f"EPE warmup resolution {height}x{width_pixels} must be "
                    f"divisible by {scale}"
                )
            latent_h = 2 * (height // scale)
            latent_w = 2 * (width_pixels // scale)
            image_tokens = (latent_h // 2) * (latent_w // 2)
            invalid = [
                lane_width
                for lane_width in self.parallel.allowed_widths
                if image_tokens % lane_width
            ]
            if invalid:
                raise ValueError(
                    f"EPE warmup image token count {image_tokens} from "
                    f"{height}x{width_pixels} is not divisible by lane widths "
                    f"{invalid}"
                )

            for lane_width in self.parallel.allowed_widths:
                lane = self._rank_lane(lane_width)
                generator = torch.Generator(device=device).manual_seed(
                    1_000_003
                    + height * 97
                    + width_pixels * 193
                    + lane_width * 389
                    + lane[0] * 997
                )
                x = torch.randn(
                    (
                        batch_size * cfg_conditions,
                        in_channels,
                        1,
                        latent_h,
                        latent_w,
                    ),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
                timestep = torch.full(
                    (batch_size * cfg_conditions,),
                    0.5,
                    device=device,
                    dtype=torch.float32,
                )
                cap_feats = [
                    torch.randn(
                        (text_tokens, cap_feat_dim),
                        generator=generator,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in range(batch_size * cfg_conditions)
                ]
                samples_ms: list[float] = []
                with self.parallel.activate(lane):
                    usp_topology = self.parallel.active_usp
                    for _ in range(steps):
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        step_started = time.perf_counter()
                        output = transformer(
                            list(x.unbind(dim=0)),
                            timestep,
                            cap_feats,
                            return_dict=False,
                        )[0]
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        if self.parallel.active.is_leader:
                            samples_ms.append(
                                (time.perf_counter() - step_started) * 1000.0
                            )
                        del output

                local = (
                    {
                        "rank": self.parallel.rank,
                        "lane": list(lane),
                        "ulysses_degree": usp_topology.ulysses_degree,
                        "ring_degree": usp_topology.ring_degree,
                        "samples_ms": samples_ms,
                    }
                    if self.parallel.rank == lane[0]
                    else None
                )
                gathered = [local]
                if self.parallel.world_size > 1:
                    gathered = [None] * self.parallel.world_size
                    dist.all_gather_object(gathered, local)
                if self.parallel.rank == 0:
                    lane_samples = [
                        sample
                        for item in gathered
                        if item is not None
                        for sample in item["samples_ms"]
                    ]
                    rows.append(
                        {
                            "resolution": [height, width_pixels],
                            "image_tokens": image_tokens,
                            "width": lane_width,
                            "batch_size": batch_size,
                            "cfg_conditions": cfg_conditions,
                            "attention_mode": self.attention_mode,
                            "ulysses_degree": next(
                                item["ulysses_degree"]
                                for item in gathered
                                if item is not None
                            ),
                            "ring_degree": next(
                                item["ring_degree"]
                                for item in gathered
                                if item is not None
                            ),
                            "latency_ms": float(statistics.median(lane_samples)),
                            "samples_ms": lane_samples,
                            "lanes": [item for item in gathered if item is not None],
                        }
                    )
                del x, cap_feats, timestep

        report = [
            {
                "source": self.cost_model.source,
                "steps": int(steps),
                "resolutions": [list(item) for item in resolutions],
                "allowed_lane_widths": list(self.parallel.allowed_widths),
                "text_tokens": int(text_tokens),
                "batch_size": int(batch_size),
                "cfg_conditions": int(cfg_conditions),
                "attention_mode": self.attention_mode,
                "total_wall_ms": (time.perf_counter() - started) * 1000.0,
                "rows": rows,
            }
            if self.parallel.rank == 0
            else None
        ]
        if self.parallel.world_size > 1:
            dist.broadcast_object_list(report, src=0)
        assert report[0] is not None
        self.initialize_cost_model(report[0]["rows"])
        logger.info(
            "EPE warmup initialized %d measured cost rows from %d steps",
            len(report[0]["rows"]),
            steps,
        )
        return report[0]

    def predict_step_ms(self, request: EpeRequest, width: int) -> float:
        return self._calibrated_cost_model.predict_step_ms(
            sequence_length=request.image_tokens,
            width=width,
            batch_size=request.batch_size,
            conditions=request.cfg_conditions,
        )

    def observe_assignment(
        self, assignment: dict[str, Any], measured_step_ms: float
    ) -> None:
        self.calibrator.observe(
            sequence_length=int(assignment["image_tokens"]),
            width=len(assignment["ranks"]),
            batch_size=int(assignment["batch_size"]),
            conditions=int(assignment["cfg_conditions"]),
            measured_step_ms=float(measured_step_ms),
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
        planner = self._get_planner()
        plans = planner.plan_at(
            [request.to_schedulable() for request in active],
            pulse_steps=self.pulse_steps,
            now_ms=(time.time() * 1000.0 if now_ms is None else float(now_ms)),
        )
        output = []
        for plan in plans:
            request = by_id[plan.request_id]
            ranks = tuple(plan.lane_ranks or ())
            output.append(
                EpePhaseAssignment(
                    request_id=request.request_id,
                    ranks=ranks,
                    start_step=request.current_step,
                    steps=plan.steps,
                    switched=bool(plan.metadata.get("switched", False)),
                    predicted_step_ms=float(plan.metadata["predicted_step_ms"]),
                    predicted_phase_ms=float(plan.metadata["predicted_phase_ms"]),
                    image_tokens=request.image_tokens,
                    batch_size=request.batch_size,
                    cfg_conditions=request.cfg_conditions,
                )
            )
        return output
