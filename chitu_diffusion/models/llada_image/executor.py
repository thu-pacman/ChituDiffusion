from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ...epe.contracts import ExecutorBuildContext
from ...epe.executor import (
    DiffusersBackend,
    build_stage_parallel_context,
    build_stage_vae_placement,
    scheduling_options_from_pool,
)
from ...epe.scheduling.planner import EpeSchedulingModule
from ...epe.scheduling.types import RequestProfile
from ...flexcache.config import CacheConfig
from ...parallel.vae import VaeParallelPlacement
from .api import LLaDAImageRequest
from .epe import LLaDAImageEpeModule
from .pipeline import LLaDAImageDiffusionPipeline, LLaDAImagePipelineOutput


class LLaDAImageDecoderExecutor(DiffusersBackend):
    """LLaDA-Image operations behind the shared Chitu diffusion contract."""

    model_name = "llada-image"

    def __init__(
        self,
        pipeline: LLaDAImageDiffusionPipeline,
        scheduling_module: EpeSchedulingModule,
        *,
        default_width: int = 1024,
        default_height: int = 1024,
        default_num_steps: int = 20,
        vae_placement: VaeParallelPlacement | None = None,
    ) -> None:
        super().__init__(
            pipeline,
            scheduling_module,
            default_width=default_width,
            default_height=default_height,
            default_num_steps=default_num_steps,
            vae_placement=vae_placement,
        )

    def warmup(self, *, resolutions, steps) -> dict[str, Any]:
        report = self.pipeline.warmup_epe(resolutions=resolutions, steps=steps)
        terminal_rows = self.pipeline.warmup_vae(
            resolutions=resolutions,
            steps=steps,
            parallel_vae=self.parallel_vae,
            vae_parallel_halo=self.vae_parallel_halo,
        )
        terminal_by_key = {
            (int(row["image_tokens"]), int(row["width"])): row for row in terminal_rows
        }
        for row in report["rows"]:
            terminal = terminal_by_key[(int(row["image_tokens"]), int(row["width"]))]
            row.update(
                {
                    key: value
                    for key, value in terminal.items()
                    if key
                    in {
                        "state_bytes",
                        "terminal_ms",
                        "vae_latency_ms",
                        "d2h_latency_ms",
                        "terminal_samples_ms",
                    }
                }
            )
        report["parallel_vae"] = self.parallel_vae
        report["vae_parallel_halo"] = self.vae_parallel_halo
        self.scheduling_module.initialize_cost_model(report["rows"])
        return report

    def normalize_request(self, request: Any) -> LLaDAImageRequest:
        if isinstance(request, LLaDAImageRequest):
            return request
        explicit_fields = getattr(request, "model_fields_set", None)
        if hasattr(request, "model_dump"):
            request = request.model_dump()
        if not isinstance(request, Mapping):
            raise TypeError(
                "LLaDA-Image requests must be LLaDAImageRequest or a mapping"
            )
        num_inference_steps = request.get("num_inference_steps")
        if num_inference_steps is None:
            num_inference_steps = request.get("num_steps")
        if num_inference_steps is None:
            num_inference_steps = self.default_num_steps
        guidance_scale = request.get("guidance_scale")
        if guidance_scale is None or (
            explicit_fields is not None and "guidance_scale" not in explicit_fields
        ):
            guidance_scale = 4.5
        return LLaDAImageRequest(
            prompt=request.get("prompt"),
            request_id=str(request.get("request_id") or uuid.uuid4().hex),
            generation_mode=str(request.get("generation_mode", "text")),
            image=request.get("image"),
            negative_prompt=request.get("negative_prompt"),
            width=int(request.get("width") or self.default_width),
            height=int(request.get("height") or self.default_height),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            seed=int(request.get("seed", 0)),
            max_sequence_length=int(request.get("max_sequence_length", 2048)),
            deadline_ms=(
                None
                if request.get("deadline_ms") is None
                else float(request["deadline_ms"])
            ),
            priority=int(request.get("priority", 0)),
            output_type=str(request.get("output_type", "pil")),
            cache=(
                request["cache"]
                if isinstance(request.get("cache"), CacheConfig)
                else CacheConfig.from_mapping(request.get("cache"))
            ),
            extra_inputs=dict(request.get("extra_inputs") or {}),
        )

    def serialize_request(self, request: Any) -> dict[str, Any]:
        normalized = self.normalize_request(request)
        return {
            "request_id": normalized.request_id,
            "prompt": normalized.prompt,
            "generation_mode": normalized.generation_mode,
            "image": normalized.image,
            "negative_prompt": normalized.negative_prompt,
            "width": normalized.width,
            "height": normalized.height,
            "num_inference_steps": normalized.num_inference_steps,
            "guidance_scale": normalized.guidance_scale,
            "seed": normalized.seed,
            "max_sequence_length": normalized.max_sequence_length,
            "deadline_ms": normalized.deadline_ms,
            "priority": normalized.priority,
            "output_type": normalized.output_type,
            "cache": normalized.cache.to_dict(),
            "extra_inputs": dict(normalized.extra_inputs),
        }

    def prepare_request(self, request: Any) -> Any:
        normalized = self.normalize_request(request)
        if normalized.cache.strategy != "none":
            raise NotImplementedError(
                "LLaDA-Image FlexCache requires a validated model contract"
            )
        device = (
            torch.device("cuda", self.parallel_context.local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        kwargs = dict(normalized.extra_inputs)
        reserved = {
            "prompt",
            "image",
            "generation_mode",
            "negative_prompt",
            "width",
            "height",
            "num_inference_steps",
            "guidance_scale",
            "generator",
            "max_sequence_length",
        }
        overlap = reserved.intersection(kwargs)
        if overlap:
            raise ValueError(
                "extra_inputs contains reserved fields: " + ", ".join(sorted(overlap))
            )
        return self.pipeline.prepare_request(
            prompt=normalized.prompt,
            image=normalized.image,
            generation_mode=normalized.generation_mode,
            negative_prompt=normalized.negative_prompt,
            width=normalized.width,
            height=normalized.height,
            num_inference_steps=normalized.num_inference_steps,
            guidance_scale=normalized.guidance_scale,
            generator=torch.Generator(device=device).manual_seed(normalized.seed),
            max_sequence_length=normalized.max_sequence_length,
            **kwargs,
        )

    def request_profile(
        self,
        request: Any,
        *,
        completed_steps: int,
    ) -> RequestProfile:
        normalized = self.normalize_request(request)
        target_tokens = (normalized.height // 16) * (normalized.width // 16)
        image_tokens = target_tokens * (
            2 if normalized.generation_mode == "editing" else 1
        )
        return RequestProfile(
            total_steps=normalized.num_inference_steps,
            completed_steps=int(completed_steps),
            shape_key=(normalized.height, normalized.width),
            image_tokens=image_tokens,
            attributes={
                "batch_size": 1,
                "conditions": self._request_conditions(normalized),
                "state_bytes": self._request_state_bytes(
                    normalized,
                    target_tokens,
                ),
                "generation_mode": normalized.generation_mode,
            },
        )

    def profile(self, state: Any) -> RequestProfile:
        return RequestProfile(
            total_steps=len(state.timesteps),
            completed_steps=int(state.step_index),
            shape_key=(int(state.height), int(state.width)),
            image_tokens=int(state.image_tokens),
            attributes={
                "batch_size": 1,
                "conditions": self._state_conditions(state),
                "state_bytes": state.latents.numel() * state.latents.element_size(),
                "generation_mode": state.generation_mode,
            },
        )

    def _request_conditions(self, request: LLaDAImageRequest) -> int:
        return 2 if request.guidance_scale > 1.0 else 1

    def _request_state_bytes(
        self, request: LLaDAImageRequest, image_tokens: int
    ) -> int:
        del request
        transformer = self.pipeline.transformer
        in_channels = int(transformer.config.in_channels)
        element_size = next(transformer.parameters()).element_size()
        return image_tokens * in_channels * element_size

    def _state_conditions(self, state: Any) -> int:
        return 2 if state.guidance_scale > 1.0 else 1

    def _decode_kwargs(self) -> dict[str, Any]:
        return {
            "parallel_vae": self.parallel_vae,
            "vae_parallel_halo": self.vae_parallel_halo,
        }

    def package_generate_output(self, output: Any) -> LLaDAImagePipelineOutput:
        return LLaDAImagePipelineOutput(images=output)


@dataclass(frozen=True)
class LLaDAImageExecutorFactory:
    model_path: str
    torch_dtype: torch.dtype | None = None
    local_files_only: bool = True
    default_width: int = 1024
    default_height: int = 1024
    default_num_steps: int = 20

    def build(self, context: ExecutorBuildContext) -> LLaDAImageDecoderExecutor:
        parallel, _ = build_stage_parallel_context(context)
        attention_mode = context.cp.attention_mode
        cfg_parallel = context.model.cfg_parallel
        dtype = self.torch_dtype or (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        epe_options = scheduling_options_from_pool(context.pool)
        epe_options["attention_mode"] = attention_mode
        epe_options["cfg_parallel"] = cfg_parallel
        pipeline = LLaDAImageDiffusionPipeline.from_pretrained(
            self.model_path,
            parallel_context=parallel,
            attention_mode=attention_mode,
            ulysses_degree=context.cp.ulysses_degree,
            epe_options=epe_options,
            cfg_parallel=cfg_parallel,
            torch_dtype=dtype,
            local_files_only=self.local_files_only,
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda", pipeline.parallel_context.local_rank))
        pipeline.set_progress_bar_config(disable=True)
        scheduling_module = getattr(pipeline.transformer, "epe", None)
        if scheduling_module is None:
            scheduling_module = LLaDAImageEpeModule(parallel, **epe_options)
        return LLaDAImageDecoderExecutor(
            pipeline,
            scheduling_module,
            default_width=self.default_width,
            default_height=self.default_height,
            default_num_steps=self.default_num_steps,
            vae_placement=build_stage_vae_placement(context),
        )
