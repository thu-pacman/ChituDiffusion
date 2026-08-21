from __future__ import annotations

from typing import Any

from diffusers import Flux2KleinPipeline

from ...parallel import EpeParallelContext, resolve_context_parallel_config
from .transformer import Flux2KleinCpTransformer2DModel


class Flux2KleinCpPipeline(Flux2KleinPipeline):
    """Native Diffusers FLUX.2-klein pipeline with fixed full-world CP."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs: Any):
        parallel = kwargs.pop("parallel_context", None)
        ulysses_transport = kwargs.pop("ulysses_transport", None)
        agkv_transport = kwargs.pop("agkv_transport", None)
        attention_mode, ulysses_degree = resolve_context_parallel_config(
            kwargs.pop("attention_mode", "agkv"),
            kwargs.pop("ulysses_degree", None),
        )
        if parallel is None:
            parallel = EpeParallelContext.from_torchrun(
                ulysses_degree=ulysses_degree,
                ulysses_transport=ulysses_transport,
                agkv_transport=agkv_transport,
            )
        transformer = kwargs.pop("transformer", None)
        if transformer is None:
            transformer_kwargs: dict[str, Any] = {
                "subfolder": "transformer",
                "parallel_context": parallel,
                "attention_mode": attention_mode,
            }
            for key in ("torch_dtype", "local_files_only", "variant", "revision"):
                if key in kwargs:
                    transformer_kwargs[key] = kwargs[key]
            transformer = Flux2KleinCpTransformer2DModel.from_pretrained(
                pretrained_model_name_or_path,
                **transformer_kwargs,
            )
        elif not isinstance(transformer, Flux2KleinCpTransformer2DModel):
            raise TypeError("transformer must be a Flux2KleinCpTransformer2DModel")
        else:
            transformer.configure_context_parallel(
                parallel, attention_mode=attention_mode
            )
        pipeline = super().from_pretrained(
            pretrained_model_name_or_path,
            transformer=transformer,
            **kwargs,
        )
        if not pipeline.config.is_distilled:
            raise ValueError(
                "Flux2KleinCpPipeline currently requires a distilled model"
            )
        pipeline._cp_parallel_context = parallel
        return pipeline

    @property
    def parallel_context(self) -> EpeParallelContext:
        parallel = getattr(self, "_cp_parallel_context", None)
        if parallel is None:
            raise RuntimeError("pipeline has no context-parallel context")
        return parallel

    def close(self) -> None:
        self.parallel_context.close()
