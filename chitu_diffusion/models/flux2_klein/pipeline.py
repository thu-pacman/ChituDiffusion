from __future__ import annotations

from functools import wraps
from inspect import signature
from typing import Any

import torch
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput

from ...parallel.cp import EpeParallelContext, resolve_context_parallel_config
from ...parallel.vae import parallel_vae_decode
from .transformer import Flux2KleinCpTransformer2DModel


class Flux2KleinCpPipeline(Flux2KleinPipeline):
    """Native Diffusers FLUX.2-klein pipeline with fixed full-world CP."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs: Any):
        parallel_vae = bool(kwargs.pop("parallel_vae", True))
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
        pipeline.parallel_vae = parallel_vae
        pipeline.last_vae_stats = None
        return pipeline

    @torch.inference_mode()
    @wraps(Flux2KleinPipeline.__call__)
    def __call__(self, *args, **kwargs):
        bound = signature(Flux2KleinPipeline.__call__).bind(self, *args, **kwargs)
        bound.apply_defaults()
        output_type = bound.arguments["output_type"]
        if output_type == "latent":
            self.last_vae_stats = None
            return super().__call__(*args, **kwargs)
        return_dict = bound.arguments["return_dict"]
        bound.arguments.update(output_type="latent", return_dict=False)
        latents = super().__call__(*bound.args[1:], **bound.kwargs)[0]
        decoded = parallel_vae_decode(
            self.vae,
            latents,
            topology=self.parallel_context.active,
            enabled=getattr(self, "parallel_vae", True),
        )
        self.last_vae_stats = self.vae._chitu_vae_decode_stats.copy()
        images = (
            self.image_processor.postprocess(decoded, output_type=output_type)
            if decoded is not None
            else []
        )
        self.maybe_free_model_hooks()
        return Flux2PipelineOutput(images=images) if return_dict else (images,)

    @property
    def parallel_context(self) -> EpeParallelContext:
        parallel = getattr(self, "_cp_parallel_context", None)
        if parallel is None:
            raise RuntimeError("pipeline has no context-parallel context")
        return parallel

    def close(self) -> None:
        self.parallel_context.close()
