from __future__ import annotations

from typing import Any

import torch

from chitu_diffusion.epac.cache import TeaCacheConfig

from ..contracts import CacheStepContext
from ..spec import BlockSite, FlexCacheModelSpec
from ..tree import (
    TensorTree,
    first_tensor,
    tree_add,
    tree_clone,
    tree_nbytes,
    tree_sub,
)
from .base import BaseCacheStrategy


class TeaCacheStrategy(BaseCacheStrategy):
    """Reuse backbone group residuals while the modulation signal drifts slowly."""

    _REFERENCE_COEFFICIENTS: dict[str, tuple[float, ...]] = {
        "flux1": (
            4.98651651e2,
            -2.83781631e2,
            5.58554382e1,
            -3.82021401,
            2.64230861e-1,
        ),
        "wan-t2v-1.3b": (
            2.39676752e3,
            -1.31110545e3,
            2.01331979e2,
            -8.29855975,
            1.37887774e-1,
        ),
        "wan-t2v-14b": (
            -5784.54975374,
            5449.50911966,
            -1811.16591783,
            256.27178429,
            -13.02252404,
        ),
    }

    def __init__(
        self,
        params: TeaCacheConfig,
        *,
        warmup_steps: int,
        cooldown_steps: int,
    ) -> None:
        super().__init__()
        self.params = params
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.previous_probe: torch.Tensor | None = None
        self.accumulated = 0.0
        self.reuse_current_step = False
        self.group_inputs: dict[str, TensorTree] = {}
        self.group_residuals: dict[str, TensorTree] = {}
        self.coefficients: tuple[float, ...] = ()

    def begin(self, *, total_steps: int, model_spec: FlexCacheModelSpec) -> None:
        super().begin(total_steps=total_steps, model_spec=model_spec)
        self.previous_probe = None
        self.accumulated = 0.0
        self.reuse_current_step = False
        self.group_inputs.clear()
        self.group_residuals.clear()
        self.coefficients = self._resolve_coefficients(model_spec)
        self.stats.strategy = {
            "threshold": self.params.threshold,
            "variant": model_spec.variant(),
            "coefficients": self.coefficients,
        }

    def _resolve_coefficients(self, model_spec: FlexCacheModelSpec) -> tuple[float, ...]:
        if self.params.coefficients is not None:
            return self.params.coefficients
        variant = model_spec.variant()
        reference = self._REFERENCE_COEFFICIENTS.get(variant)
        if reference is None:
            supported = ", ".join(sorted(self._REFERENCE_COEFFICIENTS))
            raise NotImplementedError(
                f"TeaCache has no calibrated polynomial for {variant!r}; the "
                f"published coefficients cover {supported}. Pass explicit "
                "coefficients to run uncalibrated, e.g. (1.0, 0.0) for an "
                "identity rescale, and re-tune threshold accordingly."
            )
        return reference

    def _polyval(self, value: float) -> float:
        result = 0.0
        for coefficient in self.coefficients:
            result = result * value + coefficient
        return result

    def _forced(self, context: CacheStepContext) -> bool:
        return (
            context.step_index < max(1, self.warmup_steps)
            or context.step_index >= self.total_steps - max(1, self.cooldown_steps)
            or not self.group_residuals
        )

    def model_lookup(
        self,
        context: CacheStepContext,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[bool, TensorTree | None]:
        assert self.model_spec is not None
        probe = first_tensor(self.model_spec.modulation_probe(args, kwargs))
        if self._forced(context) or probe is None or self.previous_probe is None:
            self.reuse_current_step = False
            self.accumulated = 0.0
            if probe is not None:
                self.previous_probe = probe.detach().clone()
            self.stats.step_misses += 1
            return False, None
        with torch.no_grad():
            denominator = self.previous_probe.abs().mean()
            relative = (
                float("inf")
                if denominator == 0
                else (
                    (probe - self.previous_probe).abs().mean() / denominator
                ).item()
            )
        self.previous_probe = probe.detach().clone()
        self.accumulated += self._polyval(relative)
        if self.accumulated < self.params.threshold:
            self.reuse_current_step = True
            self.stats.step_hits += 1
            return False, None
        self.reuse_current_step = False
        self.accumulated = 0.0
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
        residual = self.group_residuals.get(site.group_name)
        if not self.reuse_current_step or residual is None:
            self.stats.block_misses += 1
            return False, None
        self.stats.block_hits += 1
        block_input = self.model_spec.block_input(site, args, kwargs)
        if site.group_index == 0:
            return True, tree_add(block_input, residual)
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
        block_input = self.model_spec.block_input(site, args, kwargs)
        if site.group_index == 0:
            self.group_inputs[site.group_name] = tree_clone(block_input)
        if site.group_index == site.group_size - 1:
            group_input = self.group_inputs.pop(site.group_name, None)
            if group_input is not None:
                self.group_residuals[site.group_name] = tree_clone(
                    tree_sub(output, group_input)
                )
        self.stats.cache_bytes = sum(
            tree_nbytes(value) for value in self.group_residuals.values()
        )
