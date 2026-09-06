from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.distributed as dist

from ..parallel.cp import EpeParallelContext, resolve_context_parallel_config
from ..parallel.vae import VaeParallelPlacement, create_vae_parallel_placement
from .contracts import (
    ExecutorBuildContext,
    TerminalArtifact,
    TransferBundle,
    normalize_terminal_artifact,
)
from .scheduling.types import RequestProfile


def scheduling_options_from_pool(pool: Any) -> dict[str, Any]:
    """Translate embedded pool configuration into shared EPE options."""

    return {
        "policy": pool.policy,
        "switch_allowed_until_step": pool.switch_allowed_until_step,
        "pulse_steps": pool.pulse_steps,
        "balanced_k": pool.balanced_k,
        "starvation_ms": pool.starvation_ms,
        "deadline_guard_ms": pool.deadline_guard_ms,
        "min_resize_gain_ms": pool.min_resize_gain_ms,
        "resize_hysteresis_ms": pool.resize_hysteresis_ms,
        "resize_control_cost_ms": pool.resize_control_cost_ms,
        "max_active_requests": pool.max_inflight_requests,
        "online_calibration": pool.online_calibration,
    }


def build_stage_parallel_context(
    context: ExecutorBuildContext,
    *,
    attention_mode: str | None = None,
    ulysses_degree: int | None = None,
    ulysses_transport: str | None = None,
    agkv_transport: str | None = None,
    fast_lane_width: int | None = None,
) -> tuple[EpeParallelContext, int]:
    """Validate a host stage world and attach EPE to its process group.

    The context-parallel geometry comes from the stage-wide plan. A model whose
    own attention layer decides the exchange shape may override the mode, the
    degree, or a transport here, so it does not inherit the environment default
    and pay for a pool it never uses.

    ``fast_lane_width`` names the one lane width per rank that may own a fast
    transport, and defaults to the full tensor-parallel plane. A model that
    splits that plane further -- CFG parallelism gives each branch half of it --
    passes the width its attention actually exchanges over.
    """

    plan = context.cp
    attention_mode = attention_mode or plan.attention_mode
    ulysses_degree = ulysses_degree or plan.ulysses_degree
    ulysses_transport = ulysses_transport or plan.ulysses_transport
    agkv_transport = agkv_transport or plan.agkv_transport
    world = context.world
    local_suffix = str(world.local_device).rsplit(":", 1)[-1]
    local_rank = int(local_suffix) if local_suffix.isdigit() else 0
    if dist.is_initialized():
        if (
            world.process_group is not None
            and world.process_group is not dist.group.WORLD
        ):
            raise NotImplementedError(
                "the embedded runtime supports an existing WORLD process group, "
                "not an arbitrary subgroup"
            )
        if dist.get_rank() != world.rank or dist.get_world_size() != world.world_size:
            raise ValueError(
                "initialized torch.distributed WORLD mismatches StageWorldSpec"
            )
    else:
        if world.process_group is not None:
            raise ValueError("process_group was supplied before torch.distributed init")
        if not world.owns_process_group:
            raise ValueError(
                "owns_process_group=False requires an initialized process group"
            )
        if world.world_size > 1 and world.master_port == 0:
            raise ValueError("master_port is required for a multi-rank stage world")
        os.environ["MASTER_ADDR"] = world.master_addr
        os.environ["MASTER_PORT"] = str(world.master_port)

    os.environ["RANK"] = str(world.rank)
    os.environ["WORLD_SIZE"] = str(world.world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    _, degree = resolve_context_parallel_config(attention_mode, ulysses_degree)
    parallel = EpeParallelContext.from_torchrun(
        allowed_widths=context.pool.allowed_lane_widths,
        owns_process_group=world.owns_process_group,
        ulysses_degree=degree,
        tensor_parallel_degree=context.tensor_parallel_degree,
        ulysses_transport=ulysses_transport,
        agkv_transport=agkv_transport,
        fast_lane_width=fast_lane_width,
    )
    return parallel, local_rank


def build_stage_vae_placement(
    context: ExecutorBuildContext,
) -> VaeParallelPlacement:
    """Create the stage's single terminal decode placement."""

    return create_vae_parallel_placement(
        context.vae.degree,
        halo=context.vae.halo,
        world_size=context.world.world_size,
        rank=context.world.rank,
    )


class DiffusersBackend(ABC):
    """Reusable Diffusers lifecycle shared by ``generate`` and ``serve``."""

    flexcache_family: str | None = None

    def __init__(
        self,
        pipeline: Any,
        scheduling_module: Any,
        *,
        default_width: int = 1024,
        default_height: int = 1024,
        default_num_steps: int = 50,
        vae_placement: VaeParallelPlacement | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.scheduling_module = scheduling_module
        self.default_width = int(default_width)
        self.default_height = int(default_height)
        self.default_num_steps = int(default_num_steps)
        self.vae_placement = vae_placement or VaeParallelPlacement()
        self.last_cache_stats: dict[str, Any] | None = None

    @property
    def parallel_context(self) -> EpeParallelContext:
        return self.pipeline.parallel_context

    @property
    def parallel_vae(self) -> bool:
        return self.vae_placement.sharded

    @property
    def vae_parallel_halo(self) -> int:
        return self.vae_placement.halo

    @abstractmethod
    def warmup(
        self,
        *,
        resolutions: tuple[tuple[int, int], ...],
        steps: int,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def normalize_request(self, request: Any) -> Any: ...

    @abstractmethod
    def serialize_request(self, request: Any) -> dict[str, Any]: ...

    @abstractmethod
    def prepare_request(self, request: Any) -> Any: ...

    @abstractmethod
    def _request_conditions(self, request: Any) -> int: ...

    @abstractmethod
    def _request_state_bytes(self, request: Any, image_tokens: int) -> int: ...

    @abstractmethod
    def _state_conditions(self, state: Any) -> int: ...

    def request_id(self, request: Any) -> str:
        return str(self.normalize_request(request).request_id)

    def request_deadline_ms(self, request: Any) -> float | None:
        return self.normalize_request(request).deadline_ms

    def validate_request(
        self,
        request: Any,
        *,
        warmup_resolutions: tuple[tuple[int, int], ...],
    ) -> None:
        normalized = self.normalize_request(request)
        cache = getattr(normalized, "cache", None)
        if cache is not None:
            cache.require_serve_available()
        shape = (int(normalized.height), int(normalized.width))
        if shape not in set(warmup_resolutions):
            formatted = ", ".join(
                f"{height}x{width}" for height, width in sorted(warmup_resolutions)
            )
            raise ValueError(
                f"unsupported resolution {shape[0]}x{shape[1]}; "
                f"warmup profile supports: {formatted}"
            )

    def deserialize_request(self, payload: Any) -> Any:
        return self.normalize_request(payload)

    def request_profile(
        self,
        request: Any,
        *,
        completed_steps: int,
    ) -> RequestProfile:
        normalized = self.normalize_request(request)
        height = int(normalized.height)
        width = int(normalized.width)
        image_tokens = (height // 16) * (width // 16)
        return RequestProfile(
            total_steps=int(normalized.num_steps),
            completed_steps=int(completed_steps),
            shape_key=(height, width),
            image_tokens=image_tokens,
            attributes={
                "batch_size": 1,
                "conditions": self._request_conditions(normalized),
                "state_bytes": self._request_state_bytes(
                    normalized,
                    image_tokens,
                ),
            },
        )

    def profile(self, state: Any) -> RequestProfile:
        return RequestProfile(
            total_steps=len(state.timesteps),
            completed_steps=int(state.step_index),
            shape_key=(int(state.height), int(state.width)),
            image_tokens=int(state.image_tokens),
            attributes={
                "batch_size": int(state.actual_batch_size),
                "conditions": self._state_conditions(state),
                "state_bytes": state.latents.numel() * state.latents.element_size(),
            },
        )

    def denoise_step(self, state: Any, *, lane_ranks: tuple[int, ...]) -> None:
        # Lanes are planned across the stage world, but a context-parallel lane
        # lives inside one tensor-parallel plane, so each rank denoises on its
        # own plane's slice. Without tensor parallelism this is the lane itself.
        self.pipeline.denoise_step(
            state, lane_ranks=self.parallel_context.plane_lane(lane_ranks)
        )

    def generate(self, request: Any) -> Any:
        """Run one request synchronously on the full stage world.

        This is the degenerate single-request form of EPE: there is one full-world
        lane, no admission queue, warmup, pulse rendezvous, or state migration.
        Model preparation and finalization are exactly the same operations used by
        the serving backend.
        """

        normalized = self.normalize_request(request)
        state = self.prepare_request(normalized)
        lane_ranks = tuple(range(self.parallel_context.world_size))
        cache = getattr(normalized, "cache", None)
        session = None
        if cache is not None and cache.strategy != "none":
            if self.flexcache_family is None:
                raise NotImplementedError(
                    f"{type(self).__name__} does not define a FlexCache model contract"
                )
            cache.require_generate_available()
            from ..flexcache import CacheSession, FlexCacheModelSpec
            from ..flexcache.strategies import create_cache_strategy

            model_spec = FlexCacheModelSpec.discover(
                self.pipeline.transformer,
                family=self.flexcache_family,
            )
            session = CacheSession(
                config=cache,
                model_spec=model_spec,
                strategy=create_cache_strategy(cache),
                total_steps=int(normalized.num_steps),
                pipeline=self.pipeline,
            )
            session.__enter__()
        try:
            while True:
                before = self.profile(state)
                if before.remaining_steps == 0:
                    break
                self.denoise_step(state, lane_ranks=lane_ranks)
                after = self.profile(state)
                if after.completed_steps <= before.completed_steps:
                    raise RuntimeError("denoise_step did not advance the request state")
        finally:
            if session is not None:
                session.__exit__(None, None, None)
                self.last_cache_stats = session.stats.to_dict()
            else:
                self.last_cache_stats = None

        output_type = str(getattr(normalized, "output_type", "pil"))
        is_leader = self.parallel_context.rank == lane_ranks[0]
        if output_type == "latent":
            output = state.latents if is_leader else None
        else:
            decoded = self.finalize_gpu(
                state,
                lane_ranks=lane_ranks,
                timings={},
            )
            if is_leader:
                if decoded is None:
                    raise RuntimeError("full-world leader did not receive decoded output")
                artifact = normalize_terminal_artifact(decoded)
                assert artifact is not None
                host_artifact = TerminalArtifact(
                    tensors={
                        name: tensor.to(device="cpu")
                        for name, tensor in artifact.tensors.items()
                    },
                    metadata=artifact.metadata,
                )
                output = self.postprocess_artifact(
                    host_artifact,
                    output_type=output_type,
                )
            else:
                output = None
        self.pipeline.maybe_free_model_hooks()
        return self.package_generate_output(output) if is_leader else None

    def synchronize_state(self, state: Any, *, step_index: int) -> None:
        self.pipeline.synchronize_state(state, step_index=step_index)

    def export_state(self, state: Any) -> TransferBundle:
        return TransferBundle(
            tensors={"latents": state.latents},
            step_index=int(state.step_index),
        )

    def _decode_kwargs(self) -> dict[str, Any]:
        return {}

    def finalize_gpu(
        self,
        state: Any,
        *,
        lane_ranks: tuple[int, ...],
        timings: dict[str, object],
    ) -> TerminalArtifact | torch.Tensor | None:
        now = time.time_ns()
        timings.setdefault("vae_start_unix_ns", now)
        timings.setdefault("vae_end_unix_ns", now)
        placement = self.vae_placement
        # Every tensor-parallel plane holds the same activations, so each one
        # decodes its own plane-local lane and only the stage leader keeps the
        # result.
        with self.parallel_context.activate(
            self.parallel_context.plane_lane(lane_ranks)
        ) as lane:
            topology = placement.resolve(lane)
            if topology is None:
                return None
            return self.pipeline.decode_request(
                state,
                topology=topology,
                timings=timings,
                parallel_vae=placement.sharded,
                vae_parallel_halo=placement.halo,
                **self._decode_kwargs(),
            )

    def postprocess(
        self,
        host_output: torch.Tensor,
        *,
        output_type: str = "pil",
    ) -> Any:
        return self.pipeline.image_processor.postprocess(
            host_output,
            output_type=output_type,
        )

    def postprocess_artifact(
        self,
        host_artifact: TerminalArtifact,
        *,
        output_type: str | None = None,
    ) -> Any:
        """Preserve the legacy single-tensor postprocess contract by default."""

        if len(host_artifact.tensors) != 1:
            raise TypeError(
                f"{type(self).__name__} must override postprocess_artifact "
                "to handle multiple terminal tensors"
            )
        tensor = next(iter(host_artifact.tensors.values()))
        if output_type is None:
            return self.postprocess(tensor)
        return self.postprocess(tensor, output_type=output_type)

    @abstractmethod
    def package_generate_output(self, output: Any) -> Any: ...

    def abort_request(self, request_id: str, state: Any | None) -> None:
        del request_id, state

    def close(self) -> None:
        self.vae_placement.close()
        self.pipeline.close()
