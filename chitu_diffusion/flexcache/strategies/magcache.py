from __future__ import annotations

from typing import Any

import torch

from chitu_diffusion.epac.cache import MagCacheConfig

from ..contracts import CacheStepContext
from ..spec import BlockSite, FlexCacheModelSpec
from ..tree import TensorTree
from .base import BaseCacheStrategy
from .magcache_profiles import MagCacheProfile, OFFICIAL_MAGCACHE_PROFILES


def _nearest_resample(values: tuple[float, ...], target_steps: int) -> tuple[float, ...]:
    if target_steps == 1:
        return values[-1:]
    scale = (len(values) - 1) / (target_steps - 1)
    return tuple(values[round(index * scale)] for index in range(target_steps))


def _branch_schedule(
    *,
    ratios: tuple[float, ...],
    threshold: float,
    max_skip_steps: int,
    retention_ratio: float,
    force_fresh: frozenset[int],
    inclusive_threshold: bool,
) -> tuple[bool, ...]:
    retention_steps = int(retention_ratio * len(ratios) + 0.5)
    accumulated_ratio = 1.0
    accumulated_error = 0.0
    accumulated_steps = 0
    schedule: list[bool] = []
    for step_index, ratio in enumerate(ratios):
        reuse = False
        if step_index >= retention_steps:
            accumulated_ratio *= ratio
            accumulated_steps += 1
            accumulated_error += abs(1.0 - accumulated_ratio)
            within_threshold = (
                accumulated_error <= threshold
                if inclusive_threshold
                else accumulated_error < threshold
            )
            reuse = (
                step_index not in force_fresh
                and within_threshold
                and accumulated_steps <= max_skip_steps
            )
            if not reuse:
                accumulated_ratio = 1.0
                accumulated_error = 0.0
                accumulated_steps = 0
        schedule.append(reuse)
    return tuple(schedule)


class MagCacheStrategy(BaseCacheStrategy):
    """Reuse one residual spanning the complete discovered transformer backbone."""

    def __init__(self, params: MagCacheConfig) -> None:
        super().__init__()
        self.params = params
        self.reuse_schedule: tuple[bool, ...] = ()
        self.reuse_current_step = False
        self.backbone_input: torch.Tensor | None = None
        self.previous_residual: torch.Tensor | None = None

    def begin(self, *, total_steps: int, model_spec: FlexCacheModelSpec) -> None:
        super().begin(total_steps=total_steps, model_spec=model_spec)
        profile = self._resolve_profile(model_spec)
        threshold = (
            profile.threshold if self.params.threshold is None else self.params.threshold
        )
        max_skip_steps = (
            profile.max_skip_steps
            if self.params.max_skip_steps is None
            else self.params.max_skip_steps
        )
        retention_ratio = (
            profile.retention_ratio
            if self.params.retention_ratio is None
            else self.params.retention_ratio
        )
        force_fresh = frozenset(
            index
            for index in range(total_steps)
            if round(index * (profile.reference_steps - 1) / max(1, total_steps - 1))
            in profile.force_fresh_reference_indices
        )
        branch_schedules = tuple(
            _branch_schedule(
                ratios=_nearest_resample(ratios, total_steps),
                threshold=threshold,
                max_skip_steps=max_skip_steps,
                retention_ratio=retention_ratio,
                force_fresh=force_fresh,
                inclusive_threshold=profile.inclusive_threshold,
            )
            for ratios in profile.branch_ratios
        )
        self.reuse_schedule = tuple(
            all(schedule[index] for schedule in branch_schedules)
            for index in range(total_steps)
        )
        self.reuse_current_step = False
        self.backbone_input = None
        self.previous_residual = None
        self.stats.strategy = {
            "profile_id": profile.profile_id,
            "reference_steps": profile.reference_steps,
            "threshold": threshold,
            "max_skip_steps": max_skip_steps,
            "retention_ratio": retention_ratio,
            "reuse_steps": tuple(
                index for index, reuse in enumerate(self.reuse_schedule) if reuse
            ),
            "branch_profiles": len(profile.branch_ratios),
            "shared_branch_schedule": True,
        }

    def _resolve_profile(self, model_spec: FlexCacheModelSpec) -> MagCacheProfile:
        if self.params.ratios is not None:
            if model_spec.family not in {"flux1", "qwen_image", "wan"}:
                raise NotImplementedError(
                    "custom MagCache profiles require a single compatible "
                    "Flux.1, Qwen-Image, or Wan transformer backbone"
                )
            assert self.params.threshold is not None
            assert self.params.max_skip_steps is not None
            assert self.params.retention_ratio is not None
            assert self.params.reference_steps is not None
            if len(self.params.ratios) != self.params.reference_steps:
                raise ValueError(
                    "custom MagCache ratios length must equal reference_steps"
                )
            return MagCacheProfile(
                profile_id="custom",
                branch_ratios=(self.params.ratios,),
                threshold=self.params.threshold,
                max_skip_steps=self.params.max_skip_steps,
                retention_ratio=self.params.retention_ratio,
            )
        variant = model_spec.variant()
        profile = OFFICIAL_MAGCACHE_PROFILES.get(variant)
        if profile is None:
            supported = ", ".join(sorted(OFFICIAL_MAGCACHE_PROFILES))
            raise NotImplementedError(
                f"MagCache has no official profile for {variant!r}; "
                f"supported profiles: {supported}"
            )
        return profile

    def model_lookup(
        self,
        context: CacheStepContext,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[bool, TensorTree | None]:
        del args, kwargs
        scheduled = self.reuse_schedule[context.step_index]
        self.reuse_current_step = scheduled and self.previous_residual is not None
        if self.reuse_current_step:
            self.stats.step_hits += 1
        else:
            self.stats.step_misses += 1
        return False, None

    def block_lookup(
        self,
        context: CacheStepContext,
        site: BlockSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[bool, TensorTree | None]:
        del context
        assert self.model_spec is not None
        block_input = self.model_spec.block_input(site, args, kwargs)
        if not self.reuse_current_step or self.previous_residual is None:
            self.stats.block_misses += 1
            return False, None
        self.stats.block_hits += 1
        if site.index == 0:
            if isinstance(block_input, tuple):
                return True, (*block_input[:-1], block_input[-1] + self.previous_residual)
            return True, block_input + self.previous_residual
        return True, block_input

    def block_store(
        self,
        context: CacheStepContext,
        site: BlockSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: TensorTree,
    ) -> None:
        del context
        assert self.model_spec is not None
        if site.index == 0:
            block_input = self.model_spec.block_input(site, args, kwargs)
            hidden_input = block_input[-1] if isinstance(block_input, tuple) else block_input
            if not isinstance(hidden_input, torch.Tensor):
                raise TypeError("MagCache backbone input must contain a tensor")
            self.backbone_input = hidden_input.detach().clone()
        if site.index != len(self.model_spec.blocks) - 1:
            return None
        if self.backbone_input is None:
            raise RuntimeError("MagCache did not capture the backbone input")
        hidden_output = output[-1] if isinstance(output, tuple) else output
        if not isinstance(hidden_output, torch.Tensor):
            raise TypeError("MagCache backbone output must contain a tensor")
        if hidden_output.shape != self.backbone_input.shape:
            if (
                self.model_spec.family == "flux1"
                and hidden_output.ndim >= 2
                and hidden_output.shape[1] > self.backbone_input.shape[1]
                and hidden_output.shape[2:] == self.backbone_input.shape[2:]
            ):
                hidden_output = hidden_output[:, -self.backbone_input.shape[1] :]
            else:
                raise ValueError(
                    "MagCache backbone input/output hidden shapes do not match: "
                    f"{tuple(self.backbone_input.shape)} vs {tuple(hidden_output.shape)}"
                )
        self.previous_residual = (
            hidden_output.detach() - self.backbone_input
        ).detach().clone()
        self.stats.cache_bytes = (
            self.previous_residual.numel() * self.previous_residual.element_size()
        )
        return None
