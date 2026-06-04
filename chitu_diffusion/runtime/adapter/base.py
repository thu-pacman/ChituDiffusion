from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable, Optional

import torch

from chitu_diffusion.core.models.registry import ModelType, get_model_class

logger = getLogger(__name__)


@contextmanager
def device_scope(model: torch.nn.Module, backend):
    original_device = None
    if model is not None and torch.cuda.is_available():
        try:
            first_param = next(model.parameters())
            original_device = first_param.device
        except StopIteration:
            original_device = None
        model.to(torch.cuda.current_device())

    try:
        backend.memory_used(f"Loaded model to {torch.cuda.current_device()}")
        yield model
    finally:
        if model is not None:
            target_device = original_device if original_device is not None else "cpu"
            model.to(target_device)
            if torch.cuda.is_available():
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

    def supports_cfg(self, args: Any) -> bool:
        return all(x > 0 for x in args.models.sampler.guidance_scale)

    def rope_impl(self, args: Any):
        return None

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
