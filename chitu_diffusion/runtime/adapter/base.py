from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable, Optional

import torch

from chitu_diffusion.models.parallel import ModelParallelCapabilities
from chitu_diffusion.models.registry import ModelType, get_model_class

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

    def uses_external_pipeline(self) -> bool:
        return False

    def denoise_completes_in_single_call(self) -> bool:
        return False

    def schedule_each_stage(self) -> bool:
        return False

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

    def decode_latents(self, task, generator, backend) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def save_output(self, task, output: Optional[torch.Tensor], generator, backend) -> None:
        raise NotImplementedError
