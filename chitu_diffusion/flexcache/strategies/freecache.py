from __future__ import annotations

import math
from types import MethodType
from typing import Any, Callable

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from ..config import FreeCacheConfig, FreeCacheProfile
from ..contracts import CacheStats
from ..presets import preview_profile
from ..spec import FlexCacheModelSpec
from .base import BaseCacheStrategy


class VelocityAnchors:
    """Keep only the two most recent true Fresh velocities, never predictions."""

    def __init__(self) -> None:
        self.sigmas: list[torch.Tensor] = []
        self.velocities: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.velocities)

    def clear(self) -> None:
        self.sigmas.clear()
        self.velocities.clear()

    def record(self, sigma: torch.Tensor, velocity: torch.Tensor) -> None:
        self.sigmas.append(sigma.detach().to(device="cpu", dtype=torch.float32))
        self.velocities.append(velocity.detach())
        del self.sigmas[:-2]
        del self.velocities[:-2]

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size() for t in (*self.sigmas, *self.velocities)
        )

    def extrapolate(
        self,
        *,
        sigma: torch.Tensor,
        coefficient: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.velocities:
            raise RuntimeError("FreeCache reuse requires a Fresh anchor")
        latest = self.velocities[-1].to(device=device, dtype=dtype)
        if coefficient <= 0.0 or len(self.velocities) < 2:
            return latest
        span = float(self.sigmas[-1].item()) - float(self.sigmas[-2].item())
        if not math.isfinite(span) or abs(span) < 1e-12:
            return latest
        previous = self.velocities[-2].to(device=device, dtype=torch.float32)
        current = self.velocities[-1].to(device=device, dtype=torch.float32)
        target = float(sigma.item()) - float(self.sigmas[-1].item())
        prediction = current + coefficient * (current - previous) * (target / span)
        if not torch.isfinite(prediction).all():
            return latest
        return prediction.to(dtype=dtype)


class FreeCacheStrategy(BaseCacheStrategy):
    """Static, single-device velocity reuse at the flow-matching Euler boundary."""

    request_level = True

    def __init__(self, params: FreeCacheConfig) -> None:
        super().__init__()
        self.params = params
        self.anchors = VelocityAnchors()
        self.profile: FreeCacheProfile | None = None
        self.fresh_steps: frozenset[int] = frozenset()
        self._initialized = False
        self._fresh_count = 0

    def begin(self, *, total_steps: int, model_spec: FlexCacheModelSpec) -> None:
        super().begin(total_steps=total_steps, model_spec=model_spec)
        profile = self.params.profile or preview_profile(
            model_spec.family, self.params.fresh_budget or 25
        )
        if (
            profile.model_family != model_spec.family
            or profile.reference_steps != total_steps
        ):
            raise ValueError(
                "FreeCache profile must match the request model and step count"
            )
        self.profile = profile
        self.fresh_steps = frozenset(profile.fresh_steps)
        self.anchors.clear()
        self._initialized = False
        self._fresh_count = 0
        self.stats.strategy = {
            "fresh_budget": len(self.fresh_steps),
            "budget_mode": "profile",
            "variant": model_spec.family,
            "profile_id": profile.profile_id,
            "proposal_coefficient": profile.proposal_coefficient,
        }

    def _initialize(self, state: Any) -> None:
        if self._initialized:
            return
        if len(state.timesteps) != self.total_steps:
            raise ValueError(
                "FreeCache request step count changed after initialization"
            )
        if not isinstance(state.scheduler, FlowMatchEulerDiscreteScheduler):
            raise NotImplementedError(
                "FreeCache preview requires FlowMatchEulerDiscreteScheduler"
            )
        if state.scheduler.config.get("stochastic_sampling", False):
            raise NotImplementedError(
                "FreeCache preview does not support stochastic sampling"
            )
        if len(state.scheduler.sigmas) != self.total_steps + 1:
            raise ValueError("FreeCache requires one sigma per step plus the endpoint")
        self._initialized = True

    @torch.inference_mode()
    def denoise_step(
        self,
        pipeline: Any,
        state: Any,
        *,
        lane_ranks: tuple[int, ...],
        call_original: Callable[[], Any],
    ) -> Any:
        if len(lane_ranks) != 1 or pipeline.parallel_context.world_size != 1:
            raise NotImplementedError(
                "FreeCache preview currently supports single-device generate only"
            )
        if self.profile is None:
            raise RuntimeError("FreeCache must begin before denoising")
        self._initialize(state)
        index = int(state.step_index)
        if index in self.fresh_steps:
            result = self._run_fresh(state, index, call_original)
            self._fresh_count += 1
            self.stats.step_misses += 1
            return result
        velocity = self.anchors.extrapolate(
            sigma=state.scheduler.sigmas[index],
            coefficient=self.profile.proposal_coefficient,
            device=state.latents.device,
            dtype=state.latents.dtype,
        )
        latent_dtype = state.latents.dtype
        if self.profile.model_family == "zimage":
            velocity = velocity.to(torch.float32)
        with pipeline.parallel_context.activate(lane_ranks):
            state.latents = state.scheduler.step(
                velocity,
                state.timesteps[index],
                state.latents,
                return_dict=False,
            )[0]
            if state.latents.dtype != latent_dtype:
                state.latents = state.latents.to(latent_dtype)
            state.step_index += 1
        self.stats.step_hits += 1
        return state

    def _run_fresh(
        self, state: Any, index: int, call_original: Callable[[], Any]
    ) -> Any:
        scheduler = state.scheduler
        original_step = scheduler.step
        captured: dict[str, torch.Tensor] = {}

        def capture_step(_scheduler, model_output, timestep, sample, *args, **kwargs):
            result = original_step(model_output, timestep, sample, *args, **kwargs)
            captured["velocity"] = model_output.detach()
            return result

        scheduler.step = MethodType(capture_step, scheduler)
        try:
            result = call_original()
        finally:
            scheduler.step = original_step
        if "velocity" not in captured:
            raise RuntimeError("FreeCache did not observe the scheduler step")
        self.anchors.record(scheduler.sigmas[index], captured["velocity"])
        self.stats.cache_bytes = self.anchors.nbytes()
        return result

    def end(self) -> CacheStats:
        self.stats.strategy["fresh_steps"] = self._fresh_count
        self.anchors.clear()
        self._initialized = False
        return super().end()
