from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.distributed as dist

from ...epe.scheduling.cost import MeasuredStepCostModel, RuntimeCostCalibrator
from ...epe.scheduling.planner import (
    EpePhaseAssignment as EpePhaseAssignment,
)
from ...epe.scheduling.planner import (
    EpeRequest as _EpeRequest,
)
from ...epe.scheduling.planner import (
    EpeSchedulingModule,
)
from ...parallel import EpeParallelContext, resolve_context_parallel_config

logger = logging.getLogger(__name__)

EpeCostModel = MeasuredStepCostModel
EpeRuntimeCalibrator = RuntimeCostCalibrator


@dataclass(frozen=True, slots=True)
class EpeRequest(_EpeRequest):
    """Z-Image planner request with CFG enabled by default."""

    cfg_conditions: int = 2


class ZImageEpeModule(EpeSchedulingModule):
    """Z-Image transformer warmup on top of shared EPE scheduling state."""

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
        cfg_parallel: bool = True,
        cost_model: object | None = None,
        online_calibration: bool = True,
    ) -> None:
        if cost_model is not None:
            logger.warning(
                "Ignoring static EPE cost_model=%r; startup warmup initializes "
                "the measured cost model",
                cost_model,
            )
        attention_mode, _ = resolve_context_parallel_config(attention_mode)
        super().__init__(
            parallel,
            policy=policy,
            switch_allowed_until_step=switch_allowed_until_step,
            pulse_steps=pulse_steps,
            balanced_k=balanced_k,
            starvation_ms=starvation_ms,
            deadline_guard_ms=deadline_guard_ms,
            max_active_requests=max_active_requests,
            online_calibration=online_calibration,
        )
        self.attention_mode = attention_mode
        self.cfg_parallel = bool(cfg_parallel)

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
        if steps < 3:
            raise ValueError("EPE warmup steps must be >= 3")
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
                if image_tokens
                % (
                    lane_width // 2
                    if self.cfg_parallel
                    and cfg_conditions == 2
                    and lane_width >= 2
                    and lane_width % 2 == 0
                    else lane_width
                )
            ]
            if invalid:
                raise ValueError(
                    f"EPE warmup image token count {image_tokens} from "
                    f"{height}x{width_pixels} is not divisible by lane widths "
                    f"{invalid}"
                )

            for lane_width in self.parallel.allowed_widths:
                lane = self.rank_lane(lane_width)
                use_cfp = (
                    self.cfg_parallel
                    and cfg_conditions == 2
                    and lane_width >= 2
                    and lane_width % 2 == 0
                )
                cfg_topology = (
                    self.parallel.cfg_parallel_topology(lane) if use_cfp else None
                )
                if use_cfp and cfg_topology is None:
                    raise RuntimeError(
                        f"CFP topology was not initialized for lane {lane}"
                    )
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
                    if cfg_topology is not None:
                        branch_start = cfg_topology.branch_index * batch_size
                        branch_stop = branch_start + batch_size
                        branch_x = x[branch_start:branch_stop]
                        branch_timestep = timestep[branch_start:branch_stop]
                        branch_cap_feats = cap_feats[branch_start:branch_stop]
                        execution_lane = cfg_topology.cp_ranks
                    else:
                        branch_x = x
                        branch_timestep = timestep
                        branch_cap_feats = cap_feats
                        execution_lane = lane

                    with self.parallel.activate(execution_lane):
                        ulysses_topology = self.parallel.active_ulysses
                        for _ in range(steps):
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            step_started = time.perf_counter()
                            output = transformer(
                                list(branch_x.unbind(dim=0)),
                                branch_timestep,
                                branch_cap_feats,
                                return_dict=False,
                            )[0]
                            if cfg_topology is not None:
                                local_prediction = torch.stack(output)
                                gathered = self.parallel.all_gather_cfg(
                                    local_prediction, cfg_topology
                                )
                                del local_prediction, gathered
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            samples_ms.append(
                                (time.perf_counter() - step_started) * 1000.0
                            )
                            del output

                local = {
                    "rank": self.parallel.rank,
                    "lane": list(lane),
                    "cfp_degree": 2 if cfg_topology is not None else 1,
                    "cp_degree": len(execution_lane),
                    "ulysses_degree": ulysses_topology.degree,
                    "samples_ms": samples_ms,
                }
                gathered = [local]
                if self.parallel.world_size > 1:
                    gathered = [None] * self.parallel.world_size
                    dist.all_gather_object(gathered, local)
                if self.parallel.rank == 0:
                    lane_groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
                    for item in gathered:
                        assert item is not None
                        lane_groups.setdefault(tuple(item["lane"]), []).append(item)
                    lane_samples = [
                        max(member["samples_ms"][sample_index] for member in members)
                        for members in lane_groups.values()
                        for sample_index in range(steps)
                    ]
                    representative = next(item for item in gathered if item is not None)
                    rows.append(
                        {
                            "resolution": [height, width_pixels],
                            "image_tokens": image_tokens,
                            "width": lane_width,
                            "batch_size": batch_size,
                            "cfg_conditions": cfg_conditions,
                            "cfp_degree": representative["cfp_degree"],
                            "cp_degree": representative["cp_degree"],
                            "attention_mode": self.attention_mode,
                            "ulysses_degree": representative["ulysses_degree"],
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
                "cfg_parallel": self.cfg_parallel,
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
