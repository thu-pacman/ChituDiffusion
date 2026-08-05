from __future__ import annotations

from types import MethodType
from typing import Any, Callable

import torch

from chitu_diffusion.epac.cache import MeanCacheConfig

from ..spec import FlexCacheModelSpec
from ..tree import tree_nbytes
from .base import BaseCacheStrategy

_RULE_ORDER = {
    "chain": 0,
    "v_diff_mean": 1,
    "s2": 2,
    "s3": 3,
    "s4": 4,
    "s5": 5,
    "s6": 6,
}

_SCHEDULES: dict[str, dict[int, tuple[tuple[int, ...], tuple[str, ...]]]] = {
    "qwen_image": {
        25: (
            tuple(range(17)) + (24, 32, 40, 43, 45, 47, 48, 49),
            (
                *(("chain",) * 16),
                "v_diff_mean",
                "v_diff_mean",
                "v_diff_mean",
                "s5",
                "s5",
                "s5",
                "chain",
                "chain",
            ),
        ),
        17: (
            tuple(range(9)) + (10, 17, 24, 31, 38, 45, 48, 49),
            (
                *(("chain",) * 8),
                "s2",
                "v_diff_mean",
                "v_diff_mean",
                "v_diff_mean",
                "v_diff_mean",
                "v_diff_mean",
                "s4",
                "chain",
            ),
        ),
        10: (
            (0, 1, 3, 7, 14, 21, 28, 35, 42, 49),
            (
                "chain",
                "s2",
                "v_diff_mean",
                "v_diff_mean",
                "v_diff_mean",
                "v_diff_mean",
                "v_diff_mean",
                "v_diff_mean",
                "s4",
            ),
        ),
    },
    "zimage": {
        25: (
            (
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                8,
                10,
                11,
                12,
                13,
                15,
                16,
                17,
                18,
                20,
                22,
                29,
                37,
                42,
                45,
                47,
                48,
                49,
            ),
            (
                "chain",
                "chain",
                "chain",
                "chain",
                "chain",
                "chain",
                "s2",
                "s2",
                "chain",
                "chain",
                "chain",
                "s2",
                "chain",
                "chain",
                "chain",
                "s5",
                "s5",
                "v_diff_mean",
                "v_diff_mean",
                "v_diff_mean",
                "s5",
                "s5",
                "chain",
                "chain",
            ),
        ),
        20: (
            (0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 30, 38, 43, 46, 48, 49),
            (
                "chain",
                "chain",
                "chain",
                "chain",
                "s2",
                "s2",
                "s2",
                "s2",
                "s2",
                "s5",
                "s4",
                "s5",
                "s5",
                "v_diff_mean",
                "v_diff_mean",
                "v_diff_mean",
                "s5",
                "s2",
                "chain",
            ),
        ),
        17: (
            (0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 19, 27, 35, 43, 46, 48, 49),
            (
                "chain",
                "chain",
                "chain",
                "chain",
                "s2",
                "s2",
                "s2",
                "s2",
                "s2",
                "v_diff_mean",
                "v_diff_mean",
                "v_diff_mean",
                "v_diff_mean",
                "s5",
                "s2",
                "chain",
            ),
        ),
        15: (
            (0, 1, 2, 3, 5, 7, 10, 12, 17, 25, 33, 41, 45, 48, 49),
            (
                "chain",
                "chain",
                "chain",
                "s2",
                "s2",
                "s2",
                "s2",
                "v_diff_mean",
                "v_diff_mean",
                "v_diff_mean",
                "v_diff_mean",
                "s5",
                "s4",
                "chain",
            ),
        ),
        13: (
            (0, 1, 2, 3, 5, 9, 15, 22, 29, 36, 43, 47, 49),
            (
                "chain",
                "chain",
                "chain",
                "s2",
                "v_diff_mean",
                "v_diff_mean",
                "v_diff_mean",
                "v_diff_mean",
                "v_diff_mean",
                "v_diff_mean",
                "s5",
                "s2",
            ),
        ),
    },
}


class MeanCacheStrategy(BaseCacheStrategy):
    """Official MeanCache sigma-JVP extrapolation at the denoise-step boundary."""

    request_level = True

    def __init__(self, params: MeanCacheConfig) -> None:
        super().__init__()
        self.params = params
        self.fresh_indices: frozenset[int] = frozenset()
        self.edge_orders: tuple[int, ...] = ()
        self.latents_pre: list[torch.Tensor] = []
        self.latents_post: list[torch.Tensor] = []
        self.sigmas_pre: list[torch.Tensor] = []
        self.sigmas_post: list[torch.Tensor] = []
        self.velocities: list[torch.Tensor] = []

    def begin(self, *, total_steps: int, model_spec: FlexCacheModelSpec) -> None:
        super().begin(total_steps=total_steps, model_spec=model_spec)
        if total_steps != 50:
            raise ValueError("official MeanCache schedules require exactly 50 steps")
        if not self.params.use_jvp:
            raise NotImplementedError("official MeanCache requires JVP extrapolation")
        family_schedules = _SCHEDULES.get(model_spec.family)
        if family_schedules is None:
            raise NotImplementedError(
                f"MeanCache is not calibrated for {model_spec.family!r}"
            )
        try:
            fresh, rules = family_schedules[self.params.fresh_steps]
        except KeyError as exc:
            supported = ", ".join(str(value) for value in sorted(family_schedules))
            raise ValueError(
                f"MeanCache fresh_steps={self.params.fresh_steps} is unavailable "
                f"for {model_spec.family}; supported values: {supported}"
            ) from exc
        if len(rules) != len(fresh) - 1:
            raise RuntimeError("invalid MeanCache edge-rule table")
        edge_orders = [0] * total_steps
        for start, end, rule in zip(fresh, fresh[1:], rules):
            edge_orders[start:end] = [_RULE_ORDER[rule]] * (end - start)
        self.fresh_indices = frozenset(fresh)
        self.edge_orders = tuple(edge_orders)
        self.latents_pre.clear()
        self.latents_post.clear()
        self.sigmas_pre.clear()
        self.sigmas_post.clear()
        self.velocities.clear()
        self.stats.strategy = {
            "fresh_steps": self.params.fresh_steps,
            "use_jvp": True,
            "variant": model_spec.family,
            "fresh_indices": fresh,
        }

    @torch.inference_mode()
    def denoise_step(
        self,
        pipeline: Any,
        state: Any,
        *,
        lane_ranks: tuple[int, ...],
        call_original: Callable[[], Any],
    ) -> Any:
        step_index = int(state.step_index)
        if step_index in self.fresh_indices:
            self.stats.step_misses += 1
            return self._run_fresh(state, step_index, call_original)
        self.stats.step_hits += 1
        return self._run_reuse(pipeline, state, step_index, lane_ranks)

    def _run_fresh(
        self,
        state: Any,
        step_index: int,
        call_original: Callable[[], Any],
    ) -> Any:
        scheduler = state.scheduler
        original_step = scheduler.step
        captured: dict[str, Any] = {}

        def capture_step(
            _scheduler: Any,
            model_output: torch.Tensor,
            timestep: torch.Tensor,
            sample: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            result = original_step(model_output, timestep, sample, *args, **kwargs)
            captured.update(
                velocity=model_output.detach(),
                latent_pre=sample.detach(),
                latent_post=result[0].detach(),
            )
            return result

        scheduler.step = MethodType(capture_step, scheduler)
        try:
            result = call_original()
        finally:
            scheduler.step = original_step
        if not captured:
            raise RuntimeError("MeanCache did not observe the scheduler step")
        self._record(
            step_index,
            captured["latent_pre"],
            captured["latent_post"],
            captured["velocity"],
            scheduler,
        )
        return result

    def _run_reuse(
        self,
        pipeline: Any,
        state: Any,
        step_index: int,
        lane_ranks: tuple[int, ...],
    ) -> Any:
        if not self.velocities:
            raise RuntimeError("MeanCache cannot reuse before a fresh denoise step")
        order = self.edge_orders[step_index - 1]
        if order < 1:
            raise RuntimeError(
                f"MeanCache step {step_index} has no valid JVP edge order"
            )
        latent_pre = state.latents.detach()
        velocity = self._predict(
            order=order,
            latent_dtype=state.latents.dtype,
            latent_device=state.latents.device,
            sigma_pre=state.scheduler.sigmas[step_index],
            sigma_post=state.scheduler.sigmas[step_index + 1],
        )
        timestep = state.timesteps[step_index]
        model_output = (
            velocity.to(torch.float32)
            if self.model_spec is not None and self.model_spec.family == "zimage"
            else velocity
        )
        latent_dtype = state.latents.dtype
        with pipeline.parallel_context.activate(lane_ranks):
            state.latents = state.scheduler.step(
                model_output,
                timestep,
                state.latents,
                return_dict=False,
            )[0]
            if state.latents.dtype != latent_dtype:
                state.latents = state.latents.to(latent_dtype)
            state.step_index += 1
        self._record(
            step_index,
            latent_pre,
            state.latents.detach(),
            velocity.detach(),
            state.scheduler,
        )
        return state

    def _predict(
        self,
        *,
        order: int,
        latent_dtype: torch.dtype,
        latent_device: torch.device,
        sigma_pre: torch.Tensor,
        sigma_post: torch.Tensor,
    ) -> torch.Tensor:
        actual_steps = min(order, len(self.velocities))
        start = len(self.velocities) - actual_steps
        x_start = self.latents_pre[start].to(
            device=latent_device, dtype=torch.float32
        )
        x_end = self.latents_post[-1].to(
            device=latent_device, dtype=torch.float32
        )
        s_start = self.sigmas_pre[start].to(
            device=latent_device, dtype=torch.float32
        )
        s_end = self.sigmas_post[-1].to(
            device=latent_device, dtype=torch.float32
        )
        denominator = s_end - s_start
        denominator_b = self._broadcast_sigma(denominator, x_start)
        average_velocity = (x_end - x_start) / denominator_b
        instantaneous = self.velocities[start].to(
            device=latent_device, dtype=torch.float32
        )
        jvp = (instantaneous - average_velocity) / denominator_b
        interval = (sigma_post - sigma_pre).to(
            device=latent_device, dtype=latent_dtype
        )
        return self.velocities[-1].to(
            device=latent_device, dtype=latent_dtype
        ) - jvp.to(latent_dtype) * self._broadcast_sigma(
            interval,
            self.velocities[-1],
        )

    @staticmethod
    def _broadcast_sigma(
        sigma: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        while sigma.ndim < reference.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _record(
        self,
        step_index: int,
        latent_pre: torch.Tensor,
        latent_post: torch.Tensor,
        velocity: torch.Tensor,
        scheduler: Any,
    ) -> None:
        self.latents_pre.append(latent_pre.detach())
        self.latents_post.append(latent_post.to(torch.float32).detach())
        self.sigmas_pre.append(scheduler.sigmas[step_index].detach())
        self.sigmas_post.append(scheduler.sigmas[step_index + 1].detach())
        self.velocities.append(velocity.detach())
        max_history = 6
        for values in (
            self.latents_pre,
            self.latents_post,
            self.sigmas_pre,
            self.sigmas_post,
            self.velocities,
        ):
            if len(values) > max_history:
                del values[:-max_history]
        self.stats.cache_bytes = sum(
            tree_nbytes(value)
            for values in (
                self.latents_pre,
                self.latents_post,
                self.sigmas_pre,
                self.sigmas_post,
                self.velocities,
            )
            for value in values
        )
