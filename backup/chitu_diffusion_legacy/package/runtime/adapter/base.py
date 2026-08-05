from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable, Optional

import torch

from chitu_diffusion.models.parallel import ModelParallelCapabilities
from chitu_diffusion.models.registry import ModelType, get_model_class
from chitu_diffusion.runtime.cfg_execution import (
    CfgBranchBatch,
    CfgBranchExecutor,
    CfgExecutionSpec,
)

logger = getLogger(__name__)


@contextmanager
def device_scope(model: torch.nn.Module, backend):
    """Make `model` available on the current CUDA device for the duration.

    Residency is decided adaptively by `backend.should_keep_resident` (config
    `infer.diffusion.offload_policy`): when VRAM is ample, an already-resident
    model is left in place and we skip the `empty_cache()` that would otherwise
    churn the allocator every stage; under memory pressure (or `always_offload`)
    the model is moved back to its origin device and the cache is emptied, as
    before. This keeps the low-memory path intact while removing per-stage
    offload overhead when it is not needed.
    """
    if model is None or not torch.cuda.is_available():
        yield model
        return

    cur_idx = torch.cuda.current_device()
    try:
        original_device = next(model.parameters()).device
    except StopIteration:
        original_device = None
    already_resident = (
        original_device is not None
        and original_device.type == "cuda"
        and original_device.index == cur_idx
    )
    if not already_resident:
        model.to(cur_idx)

    keep_resident = True
    try:
        keep_resident = backend.should_keep_resident(model, already_resident)
    except Exception:
        # Be conservative on any policy error: fall back to legacy offload.
        keep_resident = False

    try:
        yield model
    finally:
        if keep_resident:
            # Leave the model on GPU and intentionally skip empty_cache() to avoid
            # allocator thrash; the cached blocks are reused by the next stage.
            return
        target_device = original_device if original_device is not None else torch.device("cpu")
        model.to(target_device)
        torch.cuda.empty_cache()
        backend.memory_used(f"Offloaded {target_device}")


@dataclass(frozen=True)
class DiffusionModelSpec:
    keys: tuple[str, ...]
    adapter_cls: type["DiffusionRuntimeAdapter"]

    def create_adapter(self) -> "DiffusionRuntimeAdapter":
        return self.adapter_cls(self)


_MODEL_RUNTIME_REGISTRY: dict[str, DiffusionModelSpec] = {}


def register_model_runtime(*keys: str):
    def decorator(adapter_cls: type["DiffusionRuntimeAdapter"]):
        spec = DiffusionModelSpec(keys=tuple(keys), adapter_cls=adapter_cls)
        for key in keys:
            normalized = str(key).strip()
            if normalized in _MODEL_RUNTIME_REGISTRY:
                logger.warning("Runtime spec '%s' is being re-registered.", normalized)
            _MODEL_RUNTIME_REGISTRY[normalized] = spec
        return adapter_cls

    return decorator


def get_model_runtime_spec(models_config: Any) -> DiffusionModelSpec:
    candidates = [
        str(getattr(models_config, "type", "") or "").strip(),
        str(getattr(models_config, "name", "") or "").strip(),
    ]
    for key in candidates:
        if key in _MODEL_RUNTIME_REGISTRY:
            return _MODEL_RUNTIME_REGISTRY[key]
    raise ValueError(
        f"No diffusion runtime spec registered for model type/name {candidates}. "
        f"Available specs: {sorted(_MODEL_RUNTIME_REGISTRY.keys())}"
    )


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def set_cfg_type(backend, value: str) -> None:
    from chitu_diffusion.runtime.backend import CFGType

    backend.cfg_type = CFGType.POS if value == "pos" else CFGType.NEG


class DiffusionRuntimeAdapter:
    def __init__(self, spec: DiffusionModelSpec):
        self.spec = spec
        self.cfg_executor = CfgBranchExecutor()

    def uses_external_pipeline(self) -> bool:
        return False

    def denoise_completes_in_single_call(self) -> bool:
        return False

    def schedule_each_stage(self) -> bool:
        return False

    def supports_rank0_text_encode(self) -> bool:
        """Whether the generator's ``rank0_gpu`` text-encode optimisation (encode
        once on rank 0, then world-broadcast a single embedding tensor + length
        metadata) is valid for this adapter.

        Only models whose encoded prompt state is fully captured by the single
        tensor returned from ``encode_text`` (plus the length metadata the
        generator broadcasts) may opt in. Models that stash extra per-rank buffer
        state during ``encode_text`` (masks, negative embeddings, ...) MUST leave
        this ``False`` so every rank encodes identically -- otherwise the async
        DP admission path desyncs the collective stream across ranks."""
        return False

    def text_encode_prepares_cfg_state(self) -> bool:
        """Whether one encode call populates both cond and negative task state."""
        return False

    def supports_physical_cfg_batch(
        self, task, generator, backend, spec: CfgExecutionSpec
    ) -> bool:
        """Whether CFP-1 may pack cond+uncond into one physical model batch."""
        return False

    def cfg_branch_batch(self, task) -> CfgBranchBatch:
        """Return adapter branch state without changing the sample batch axis."""
        return CfgBranchBatch(
            cond=task.buffer.text_embeddings,
            uncond=task.buffer.negative_embeddings,
        )

    def cfg_execution_spec(
        self, task, generator, backend, *, guidance_enabled: bool
    ) -> CfgExecutionSpec:
        spec = CfgExecutionSpec.from_live_topology(
            guidance_enabled=guidance_enabled,
            physical_cfg_batch=False,
        )
        return CfgExecutionSpec(
            guidance_enabled=spec.guidance_enabled,
            cfp_degree=spec.cfp_degree,
            cp_degree=spec.cp_degree,
            physical_cfg_batch=self.supports_physical_cfg_batch(
                task, generator, backend, spec
            ),
        )

    def model_shape_spec(self, args: Any) -> Optional["ModelShapeSpec"]:
        """Static architecture facts for the cost model / profiler (optional)."""
        from chitu_diffusion.runtime.cost_model import KNOWN_SPECS

        for key in (
            str(getattr(getattr(args, "models", None), "name", "") or "").strip(),
            str(getattr(getattr(args, "models", None), "type", "") or "").strip(),
        ):
            if key in KNOWN_SPECS:
                return KNOWN_SPECS[key]
        return None

    def latent_token_count(self, width: int, height: int, args: Any) -> int:
        """Image-token count for planner/calibrator keys (model-aware default)."""
        from chitu_diffusion.runtime.cost_model import latent_token_count as _count

        name = str(getattr(getattr(args, "models", None), "name", "") or "Z-Image")
        return int(_count(width, height, model_name=name))

    def configure_external_components(self, backend, attn_backend=None, rope_impl=None) -> None:
        return None

    def configure_after_backend_build(self, backend) -> None:
        return None

    def handles_context_parallel(self, args: Any) -> bool:
        return False

    def parallel_capabilities(self, args: Any) -> ModelParallelCapabilities:
        return ModelParallelCapabilities(
            dit_context_parallel=self.handles_context_parallel(args),
            dit_cfg_parallel=self.supports_cfg(args),
            vae_tile_parallel=True,
            sampler_local_latent=False,
            model_specific_context_parallel=self.handles_context_parallel(args),
        )

    def supports_cfg(self, args: Any) -> bool:
        return all(x > 0 for x in args.models.sampler.guidance_scale)

    def rope_impl(self, args: Any):
        return None

    def loads_transformer_weights(self) -> bool:
        return False

    def build_transformer(self, models_config: Any, attn_backend, rope_impl) -> torch.nn.Module:
        try:
            model_type = ModelType(models_config.type)
        except ValueError as exc:
            raise ValueError(
                f"Model type '{models_config.type}' is not supported. "
                f"Available types: {[t.value for t in ModelType]}"
            ) from exc

        model_cls = get_model_class(model_type)
        return model_cls(
            model_type=models_config.task,
            attn_backend=attn_backend,
            rope_impl=rope_impl,
            **models_config.transformer,
        )

    def load_text_encoder(self, args: Any, init_device: torch.device):
        raise NotImplementedError

    def load_vae(self, args: Any, init_device: torch.device):
        raise NotImplementedError

    def checkpoint_paths(self, args: Any) -> list[str]:
        raise NotImplementedError

    def encode_text(self, task, generator, backend) -> torch.Tensor:
        raise NotImplementedError

    def prepare_denoise(self, task, generator, backend) -> None:
        raise NotImplementedError

    def denoise_step(self, task, generator, backend, run_dit_forward: Callable[..., torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def scheduler_step(
        scheduler,
        noise_pred: torch.Tensor,
        timestep,
        latents: torch.Tensor,
        *,
        step_index: Optional[int] = None,
        **step_kwargs,
    ) -> torch.Tensor:
        """Run one scheduler step, optionally pinned to an explicit ``step_index``.

        FlowMatch-style schedulers advance an internal ``_step_index`` on every
        ``.step`` call. In the pool engine several lanes/tasks can share one
        scheduler instance while sitting at different steps, so a stateful cursor
        is wrong: passing ``step_index`` resets the cursor before stepping so each
        call is deterministic w.r.t. the caller's step, not call order. Adapters
        with a private per-task scheduler can omit ``step_index`` (the natural
        sequential cursor is already correct)."""
        if step_index is not None and hasattr(scheduler, "_step_index"):
            scheduler._step_index = int(step_index)
        return scheduler.step(noise_pred, timestep, latents, return_dict=False, **step_kwargs)[0]

    def denoise_step_group_once(
        self, tasks: list, generator, backend, run_dit_forward: Callable[..., torch.Tensor]
    ) -> None:
        """Advance every task in ``tasks`` by EXACTLY ONE denoise step.

        This is the generic pool-engine contract, expressed on top of the
        per-model one-step ``denoise_step``. Each task carries one
        sample's latent; adapters may override this method to physically batch
        compatible tasks, otherwise members are stepped independently and the
        result is written back in place (the pool engine reads ``task.buffer`` after
        the call rather than a return value). Adapters only need a correct one-step
        ``denoise_step``; they do not need to reimplement this."""
        completes_single = self.denoise_completes_in_single_call()
        for task in tasks:
            assert task.buffer.latents is not None and task.buffer.timesteps is not None
            setattr(task.buffer, "_dit_forward_call_index", 0)
            latents = self.denoise_step(task, generator, backend, run_dit_forward)
            task.buffer.latents = latents
            if completes_single:
                task.buffer.current_step = int(task.req.params.num_inference_steps)
            else:
                task.buffer.current_step = int(task.buffer.current_step) + 1

    def decode_latents(self, task, generator, backend) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def save_output(self, task, output: Optional[torch.Tensor], generator, backend) -> None:
        raise NotImplementedError
