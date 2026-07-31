from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from ...epac.image_decoder import ExecutorBuildContext
from ...epac.model_executor import (
    DiffusersImageDecoderExecutor,
    build_stage_parallel_context,
    scheduling_options_from_pool,
)
from .api import EPACRequest
from .pipeline import EpeZImagePipeline, ZImageDenoiseState


class ZImageImageDecoderExecutor(DiffusersImageDecoderExecutor):
    """Z-Image model operations behind the generic decoder contract."""

    def __init__(
        self,
        pipeline: EpeZImagePipeline,
        *,
        default_width: int = 1024,
        default_height: int = 1024,
        default_num_steps: int = 50,
        cfg_parallel: bool = True,
        parallel_vae: bool = True,
        vae_parallel_halo: int = 8,
    ) -> None:
        super().__init__(
            pipeline,
            pipeline.transformer.epe,
            default_width=default_width,
            default_height=default_height,
            default_num_steps=default_num_steps,
        )
        self.cfg_parallel = bool(cfg_parallel)
        self.scheduling_module.cfg_parallel = self.cfg_parallel
        self.parallel_vae = bool(parallel_vae)
        self.vae_parallel_halo = int(vae_parallel_halo)
        if self.vae_parallel_halo < 0:
            raise ValueError("vae_parallel_halo must be non-negative")

    def warmup(self, *, resolutions, steps):
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
        report["cfg_parallel"] = self.cfg_parallel
        report["vae_parallel_halo"] = self.vae_parallel_halo
        self.scheduling_module.initialize_cost_model(report["rows"])
        return report

    def normalize_request(self, request: Any) -> EPACRequest:
        if isinstance(request, EPACRequest):
            return request
        if hasattr(request, "model_dump"):
            request = request.model_dump()
        if not isinstance(request, Mapping):
            raise TypeError("Z-Image requests must be EPACRequest or a mapping")
        return EPACRequest(
            prompt=request.get("prompt"),
            request_id=str(request.get("request_id") or uuid.uuid4().hex),
            negative_prompt=request.get("negative_prompt"),
            width=int(request.get("width") or self.default_width),
            height=int(request.get("height") or self.default_height),
            num_steps=int(request.get("num_steps") or self.default_num_steps),
            guidance_scale=float(request.get("guidance_scale", 5.0)),
            seed=int(request.get("seed", 0)),
            deadline_ms=(
                None
                if request.get("deadline_ms") is None
                else float(request["deadline_ms"])
            ),
            priority=int(request.get("priority", 0)),
        )

    def serialize_request(self, request: Any) -> dict[str, Any]:
        normalized = self.normalize_request(request)
        return {
            "request_id": normalized.request_id,
            "prompt": normalized.prompt,
            "negative_prompt": normalized.negative_prompt,
            "width": normalized.width,
            "height": normalized.height,
            "seed": normalized.seed,
            "num_steps": normalized.num_steps,
            "guidance_scale": normalized.guidance_scale,
            "deadline_ms": normalized.deadline_ms,
            "priority": normalized.priority,
        }

    def prepare_request(self, request: Any) -> ZImageDenoiseState:
        normalized = self.normalize_request(request)
        device = (
            torch.device("cuda", self.parallel_context.local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        generator = torch.Generator(device=device).manual_seed(normalized.seed)
        return self.pipeline.prepare_request(
            prompt=normalized.prompt,
            negative_prompt=normalized.negative_prompt,
            width=normalized.width,
            height=normalized.height,
            num_inference_steps=normalized.num_steps,
            guidance_scale=normalized.guidance_scale,
            generator=generator,
        )

    def _request_conditions(self, request: EPACRequest) -> int:
        return 2 if request.guidance_scale > 0 else 1

    def _request_state_bytes(self, request: EPACRequest, image_tokens: int) -> int:
        del request
        return image_tokens * 64 * 4

    def _state_conditions(self, state: ZImageDenoiseState) -> int:
        return 2 if state.guidance_scale > 0 else 1

    def _decode_kwargs(self) -> dict[str, Any]:
        return {
            "parallel_vae": self.parallel_vae,
            "vae_parallel_halo": self.vae_parallel_halo,
        }


@dataclass(frozen=True)
class ZImageExecutorFactory:
    model_path: str
    torch_dtype: torch.dtype | None = None
    local_files_only: bool = True
    default_width: int = 1024
    default_height: int = 1024
    default_num_steps: int = 50
    attention_mode: str = "agkv"
    ulysses_degree: int | None = None
    cfg_parallel: bool = True
    parallel_vae: bool = True
    vae_parallel_halo: int = 8

    def build(self, context: ExecutorBuildContext) -> ZImageImageDecoderExecutor:
        parallel, _ = build_stage_parallel_context(
            context,
            attention_mode=self.attention_mode,
            ulysses_degree=self.ulysses_degree,
        )
        dtype = self.torch_dtype or (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        epe_options = scheduling_options_from_pool(context.pool)
        epe_options["attention_mode"] = self.attention_mode
        epe_options["cfg_parallel"] = self.cfg_parallel
        pipeline = EpeZImagePipeline.from_pretrained(
            self.model_path,
            parallel_context=parallel,
            attention_mode=self.attention_mode,
            ulysses_degree=self.ulysses_degree,
            epe_options=epe_options,
            torch_dtype=dtype,
            local_files_only=self.local_files_only,
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda", pipeline.parallel_context.local_rank))
        pipeline.set_progress_bar_config(disable=True)
        return ZImageImageDecoderExecutor(
            pipeline,
            default_width=self.default_width,
            default_height=self.default_height,
            default_num_steps=self.default_num_steps,
            cfg_parallel=self.cfg_parallel,
            parallel_vae=self.parallel_vae,
            vae_parallel_halo=self.vae_parallel_halo,
        )
