from __future__ import annotations

import copy
from contextvars import ContextVar, Token
from types import MethodType
from typing import Any

import torch

from chitu_diffusion.flexcache.config import CacheConfig

from .contracts import CacheStats, CacheStepContext, CacheStrategy
from .spec import FlexCacheModelSpec

_ACTIVE_SESSION: ContextVar["CacheSession | None"] = ContextVar(
    "chitu_flexcache_session",
    default=None,
)


def active_cache_session() -> "CacheSession | None":
    return _ACTIVE_SESSION.get()


def _step_signature(value: Any) -> float | None:
    """Reduce a timestep argument to a comparable scalar.

    The signature identifies a denoise step, not a transformer call: pipelines
    that evaluate CFG branches serially invoke the transformer several times
    with the same timestep.
    """

    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().reshape(-1)[0].item())
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and value:
        return _step_signature(value[0])
    return None


class CacheSession:
    """Owns all mutable cache state for exactly one generate request.

    State is partitioned per CFG branch so that a reuse decision taken on the
    conditional pass can never return the unconditional pass' tensors. Branch
    identity is recovered from repeated timesteps rather than from pipeline
    cooperation, which keeps every model pipeline free of cache plumbing.
    """

    def __init__(
        self,
        *,
        config: CacheConfig,
        model_spec: FlexCacheModelSpec,
        strategy: CacheStrategy,
        total_steps: int,
        pipeline: Any | None = None,
    ) -> None:
        config.validate_steps(total_steps)
        self.config = config
        self.model_spec = model_spec
        self.pipeline = pipeline
        self.total_steps = total_steps
        self.step_index = -1
        self.branch_index = 0
        self.branches_seen = 0
        self.stats = CacheStats()
        self._request_strategy = (
            strategy if bool(getattr(strategy, "request_level", False)) else None
        )
        self._template = copy.deepcopy(strategy)
        self._branches: dict[int, CacheStrategy] = {}
        self._active: CacheStrategy | None = None
        self._signature: float | None = None
        self._token: Token[CacheSession | None] | None = None
        self._original_model_forward: Any = None
        self._original_block_forwards: list[tuple[Any, Any]] = []
        self._original_leaf_forwards: list[tuple[Any, Any]] = []
        self._original_denoise_step: Any = None
        self._inside_model_forward = False

    @property
    def steps_seen(self) -> int:
        if self._request_strategy is not None:
            return (
                self._request_strategy.stats.step_hits
                + self._request_strategy.stats.step_misses
            )
        return self.step_index + 1

    def __enter__(self) -> "CacheSession":
        if active_cache_session() is not None:
            raise RuntimeError("nested FlexCache sessions are not supported")
        self._token = _ACTIVE_SESSION.set(self)
        try:
            if self._request_strategy is not None:
                self._request_strategy.begin(
                    total_steps=self.total_steps,
                    model_spec=self.model_spec,
                )
                self._install_denoise_wrapper()
            else:
                self._install_wrappers()
        except BaseException:
            if self._request_strategy is not None:
                self._restore_denoise_wrapper()
            else:
                self._restore_wrappers()
            if self._token is not None:
                _ACTIVE_SESSION.reset(self._token)
                self._token = None
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self._request_strategy is not None:
                self._restore_denoise_wrapper()
                self.stats = self._request_strategy.end()
                self.stats.branches = 1
            else:
                self._restore_wrappers()
                stats = CacheStats()
                for _, strategy in sorted(self._branches.items()):
                    stats.merge(strategy.end())
                stats.branches = self.branches_seen
                self.stats = stats
        finally:
            self._branches.clear()
            self._active = None
            if self._token is not None:
                _ACTIVE_SESSION.reset(self._token)
                self._token = None

    def _install_denoise_wrapper(self) -> None:
        if self.model_spec.family not in {"zimage", "qwen_image", "flux1"}:
            raise NotImplementedError(
                f"request-level cache is unsupported for {self.model_spec.family}"
            )
        if self.pipeline is None:
            raise RuntimeError("request-level cache requires its pipeline instance")
        pipeline = self.pipeline
        original_denoise_step = pipeline.denoise_step
        self._original_denoise_step = original_denoise_step
        session = self

        def cached_denoise_step(
            _pipeline: Any,
            state: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if active_cache_session() is not session:
                return original_denoise_step(state, *args, **kwargs)
            strategy = session._request_strategy
            assert strategy is not None
            lane_ranks = kwargs.get(
                "lane_ranks",
                args[0]
                if args
                else tuple(range(_pipeline.parallel_context.world_size)),
            )
            return strategy.denoise_step(
                _pipeline,
                state,
                lane_ranks=tuple(lane_ranks),
                call_original=lambda: original_denoise_step(
                    state,
                    *args,
                    **kwargs,
                ),
            )

        pipeline.denoise_step = MethodType(cached_denoise_step, pipeline)

    def _restore_denoise_wrapper(self) -> None:
        if self.pipeline is not None and self._original_denoise_step is not None:
            self.pipeline.denoise_step = self._original_denoise_step
        self._original_denoise_step = None

    def _advance(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        signature = _step_signature(self.model_spec.step_probe(args, kwargs))
        if signature is None or self._signature is None:
            self.step_index += 1
            self.branch_index = 0
        elif signature == self._signature:
            self.branch_index += 1
        else:
            self.step_index += 1
            self.branch_index = 0
        self._signature = signature
        self.branches_seen = max(self.branches_seen, self.branch_index + 1)
        strategy = self._branches.get(self.branch_index)
        if strategy is None:
            strategy = copy.deepcopy(self._template)
            strategy.begin(total_steps=self.total_steps, model_spec=self.model_spec)
            self._branches[self.branch_index] = strategy
        self._active = strategy

    def _context(self) -> CacheStepContext:
        return CacheStepContext(
            step_index=self.step_index,
            total_steps=self.total_steps,
            branch_index=self.branch_index,
            model_spec=self.model_spec,
        )

    def _is_recursive_zimage_cfg_batch(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> bool:
        if self.model_spec.family != "zimage":
            return False
        samples = kwargs.get("x", args[0] if args else None)
        if not isinstance(samples, (list, tuple)) or len(samples) <= 1:
            return False
        epe = getattr(self.model_spec.model, "epe", None)
        active = getattr(getattr(epe, "parallel", None), "active", None)
        return int(getattr(active, "width", 1)) > 1

    def _install_wrappers(self) -> None:
        model = self.model_spec.model
        original_model_forward = model.forward
        self._original_model_forward = original_model_forward
        session = self

        def cached_model_forward(_module: Any, *args: Any, **kwargs: Any) -> Any:
            if session._is_recursive_zimage_cfg_batch(args, kwargs):
                return original_model_forward(*args, **kwargs)
            if session._inside_model_forward:
                return original_model_forward(*args, **kwargs)
            session._advance(args, kwargs)
            strategy = session._active
            assert strategy is not None
            context = session._context()
            hit, output = strategy.model_lookup(context, args, kwargs)
            if not hit:
                session._inside_model_forward = True
                try:
                    output = original_model_forward(*args, **kwargs)
                finally:
                    session._inside_model_forward = False
                strategy.model_store(context, args, kwargs, output)
            return output

        model.forward = MethodType(cached_model_forward, model)

        for site in self.model_spec.blocks:
            module = site.module
            original_block_forward = module.forward
            self._original_block_forwards.append((module, original_block_forward))

            def cached_block_forward(
                _module: Any,
                *args: Any,
                __site=site,
                __forward=original_block_forward,
                **kwargs: Any,
            ) -> Any:
                strategy = session._active
                if strategy is None:
                    return __forward(*args, **kwargs)
                context = session._context()
                hit, output = strategy.block_lookup(context, __site, args, kwargs)
                if not hit:
                    output = __forward(*args, **kwargs)
                    strategy.block_store(context, __site, args, kwargs, output)
                return output

            module.forward = MethodType(cached_block_forward, module)

        for site in self.model_spec.leaves:
            module = site.module
            original_leaf_forward = module.forward
            self._original_leaf_forwards.append((module, original_leaf_forward))

            def cached_leaf_forward(
                _module: Any,
                *args: Any,
                __site=site,
                __forward=original_leaf_forward,
                **kwargs: Any,
            ) -> Any:
                strategy = session._active
                if strategy is None:
                    return __forward(*args, **kwargs)
                context = session._context()
                hit, output = strategy.leaf_lookup(context, __site, args, kwargs)
                if not hit:
                    output = __forward(*args, **kwargs)
                    strategy.leaf_store(context, __site, args, kwargs, output)
                return output

            module.forward = MethodType(cached_leaf_forward, module)

    def _restore_wrappers(self) -> None:
        if self._original_model_forward is not None:
            self.model_spec.model.forward = self._original_model_forward
            self._original_model_forward = None
        for module, forward in reversed(self._original_block_forwards):
            module.forward = forward
        self._original_block_forwards.clear()
        for module, forward in reversed(self._original_leaf_forwards):
            module.forward = forward
        self._original_leaf_forwards.clear()
