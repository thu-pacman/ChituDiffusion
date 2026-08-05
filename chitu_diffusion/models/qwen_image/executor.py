from __future__ import annotations

import statistics
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.distributed as dist

from ...epac.cache import CacheConfig
from ...epac.image_decoder import ExecutorBuildContext
from ...epac.model_executor import (
    DiffusersBackend,
    build_stage_parallel_context,
    scheduling_options_from_pool,
)
from ...epac.model_scheduling import EpeSchedulingModule
from ...parallel import parallel_tiled_vae_decode
from .api import QwenImageRequest
from .pipeline import EpeQwenImagePipeline, QwenImageDenoiseState, QwenImagePipelineOutput


class QwenImageDecoderExecutor(DiffusersBackend):
    flexcache_family = "qwen_image"

    model_name = "qwen-image"

    def __init__(
        self,
        pipeline: EpeQwenImagePipeline,
        scheduling_module: EpeSchedulingModule,
        *,
        default_width: int,
        default_height: int,
        default_num_steps: int,
        parallel_vae: bool = True,
        vae_parallel_halo: int = 8,
    ) -> None:
        super().__init__(
            pipeline,
            scheduling_module,
            default_width=default_width,
            default_height=default_height,
            default_num_steps=default_num_steps,
        )
        self.parallel_vae = bool(parallel_vae)
        self.vae_parallel_halo = int(vae_parallel_halo)
        if self.vae_parallel_halo < 0:
            raise ValueError("vae_parallel_halo must be non-negative")

    def warmup(self, *, resolutions, steps) -> dict[str, Any]:
        report = self.pipeline.warmup_epe(resolutions=resolutions, steps=steps)
        terminal_rows = self._warmup_terminal(resolutions=resolutions, steps=steps)
        terminal_by_key = {
            (row["image_tokens"], row["width"]): row for row in terminal_rows
        }
        for row in report["rows"]:
            row.update(terminal_by_key[(row["image_tokens"], row["width"])])
        report["parallel_vae"] = self.parallel_vae
        report["vae_parallel_halo"] = self.vae_parallel_halo
        self.scheduling_module.initialize_cost_model(report["rows"])
        return report

    @torch.inference_mode()
    def _warmup_terminal(self, *, resolutions, steps) -> list[dict[str, Any]]:
        parameter = next(self.pipeline.vae.parameters())
        device, dtype = parameter.device, parameter.dtype
        rows: list[dict[str, Any]] = []
        for height, width in resolutions:
            image_tokens = (height // 16) * (width // 16)
            for lane_width in self.parallel_context.allowed_widths:
                offset = (self.parallel_context.rank // lane_width) * lane_width
                lane = tuple(range(offset, offset + lane_width))
                generator = torch.Generator(device=device).manual_seed(
                    6_000_013 + height * 97 + width * 193 + lane[0] * 997
                )
                latents = torch.randn(
                    (1, 16, 1, height // 8, width // 8),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
                vae_samples_ms: list[float] = []
                d2h_samples_ms: list[float] = []
                with self.parallel_context.activate(lane) as topology:
                    for _ in range(steps):
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        image = parallel_tiled_vae_decode(
                            latents,
                            lambda value: self.pipeline.vae.decode(
                                value, return_dict=False
                            )[0][:, :, 0],
                            topology=topology,
                            latent_split_dim=3,
                            pixel_split_dim=2,
                            scale=int(self.pipeline.vae_scale_factor),
                            halo=self.vae_parallel_halo,
                            enabled=self.parallel_vae,
                        )
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        vae_samples_ms.append((time.perf_counter() - started) * 1000)
                        if topology.is_leader:
                            d2h_started = time.perf_counter()
                            host = image.to("cpu")
                            d2h_samples_ms.append(
                                (time.perf_counter() - d2h_started) * 1000
                            )
                            del host
                        del image
                local = {
                    "rank": self.parallel_context.rank,
                    "lane": list(lane),
                    "vae_samples_ms": vae_samples_ms,
                    "d2h_samples_ms": d2h_samples_ms,
                }
                gathered = [local]
                if self.parallel_context.world_size > 1:
                    gathered = [None] * self.parallel_context.world_size
                    dist.all_gather_object(gathered, local)
                if self.parallel_context.rank == 0:
                    lane_groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
                    for item in gathered:
                        assert item is not None
                        lane_groups.setdefault(tuple(item["lane"]), []).append(item)
                    critical_samples: list[float] = []
                    vae_critical_samples: list[float] = []
                    d2h_samples: list[float] = []
                    for members in lane_groups.values():
                        leader = min(members, key=lambda item: item["rank"])
                        for sample_index in range(steps):
                            vae_ms = max(
                                item["vae_samples_ms"][sample_index] for item in members
                            )
                            d2h_ms = leader["d2h_samples_ms"][sample_index]
                            vae_critical_samples.append(vae_ms)
                            d2h_samples.append(d2h_ms)
                            critical_samples.append(vae_ms + d2h_ms)
                    rows.append(
                        {
                            "resolution": [height, width],
                            "image_tokens": image_tokens,
                            "width": lane_width,
                            "batch_size": 1,
                            "cfg_conditions": 2,
                            "state_bytes": image_tokens * 64 * dtype.itemsize,
                            "terminal_ms": float(statistics.median(critical_samples)),
                            "vae_latency_ms": float(
                                statistics.median(vae_critical_samples)
                            ),
                            "d2h_latency_ms": float(statistics.median(d2h_samples)),
                            "terminal_samples_ms": critical_samples,
                        }
                    )
                del latents
        broadcast = [rows if self.parallel_context.rank == 0 else None]
        if self.parallel_context.world_size > 1:
            dist.broadcast_object_list(broadcast, src=0)
        assert broadcast[0] is not None
        return broadcast[0]

    def normalize_request(self, request: Any) -> QwenImageRequest:
        if isinstance(request, QwenImageRequest):
            return request
        if hasattr(request, "model_dump"):
            request = request.model_dump()
        if not isinstance(request, Mapping):
            raise TypeError("Qwen-Image requests must be QwenImageRequest or a mapping")
        return QwenImageRequest(
            prompt=request.get("prompt"),
            negative_prompt=request.get("negative_prompt", " "),
            request_id=str(request.get("request_id") or uuid.uuid4().hex),
            width=int(request.get("width") or self.default_width),
            height=int(request.get("height") or self.default_height),
            num_steps=int(request.get("num_steps") or self.default_num_steps),
            true_cfg_scale=float(
                request.get("true_cfg_scale", request.get("guidance_scale", 4.0))
            ),
            seed=int(request.get("seed", 0)),
            deadline_ms=request.get("deadline_ms"),
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
            "negative_prompt": normalized.negative_prompt,
            "width": normalized.width,
            "height": normalized.height,
            "num_steps": normalized.num_steps,
            "true_cfg_scale": normalized.true_cfg_scale,
            "seed": normalized.seed,
            "deadline_ms": normalized.deadline_ms,
            "priority": normalized.priority,
            "output_type": normalized.output_type,
            "cache": normalized.cache.to_dict(),
            "extra_inputs": dict(normalized.extra_inputs),
        }

    def _request_conditions(self, request: QwenImageRequest) -> int:
        return (
            2
            if request.true_cfg_scale > 1 and request.negative_prompt is not None
            else 1
        )

    def _request_state_bytes(self, request: QwenImageRequest, image_tokens: int) -> int:
        del request
        element_size = next(self.pipeline.transformer.parameters()).element_size()
        return image_tokens * 64 * element_size

    def prepare_request(self, request: Any) -> QwenImageDenoiseState:
        normalized = self.normalize_request(request)
        device = (
            torch.device("cuda", self.parallel_context.local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        kwargs = dict(normalized.extra_inputs)
        reserved = {
            "prompt",
            "negative_prompt",
            "width",
            "height",
            "num_inference_steps",
            "true_cfg_scale",
            "guidance_scale",
            "generator",
        }
        overlap = reserved.intersection(kwargs)
        if overlap:
            raise ValueError(
                "extra_inputs contains reserved fields: " + ", ".join(sorted(overlap))
            )
        return self.pipeline.prepare_request(
            prompt=normalized.prompt,
            negative_prompt=normalized.negative_prompt,
            width=normalized.width,
            height=normalized.height,
            num_inference_steps=normalized.num_steps,
            true_cfg_scale=normalized.true_cfg_scale,
            guidance_scale=None,
            generator=torch.Generator(device=device).manual_seed(normalized.seed),
            **kwargs,
        )

    def _state_conditions(self, state: QwenImageDenoiseState) -> int:
        return 2 if state.do_true_cfg else 1

    def _decode_kwargs(self) -> dict[str, Any]:
        return {
            "parallel_vae": self.parallel_vae,
            "vae_parallel_halo": self.vae_parallel_halo,
        }

    def package_generate_output(self, output: Any) -> QwenImagePipelineOutput:
        return QwenImagePipelineOutput(images=output)


@dataclass(frozen=True)
class QwenImageExecutorFactory:
    model_path: str
    torch_dtype: torch.dtype | None = None
    local_files_only: bool = True
    default_width: int = 1024
    default_height: int = 1024
    default_num_steps: int = 50
    attention_mode: str = "agkv"
    cfg_parallel: bool = True
    parallel_vae: bool = True
    vae_parallel_halo: int = 8
    ulysses_degree: int | None = None

    def build(self, context: ExecutorBuildContext) -> QwenImageDecoderExecutor:
        parallel, local_rank = build_stage_parallel_context(
            context,
            attention_mode=self.attention_mode,
            ulysses_degree=self.ulysses_degree,
        )
        pipeline = EpeQwenImagePipeline.from_pretrained(
            self.model_path,
            parallel_context=parallel,
            attention_mode=self.attention_mode,
            ulysses_degree=self.ulysses_degree,
            cfg_parallel=self.cfg_parallel,
            torch_dtype=self.torch_dtype
            or (torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            local_files_only=self.local_files_only,
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda", local_rank))
        pipeline.set_progress_bar_config(disable=True)
        scheduling = EpeSchedulingModule(
            parallel,
            **scheduling_options_from_pool(context.pool),
        )
        return QwenImageDecoderExecutor(
            pipeline,
            scheduling,
            default_width=self.default_width,
            default_height=self.default_height,
            default_num_steps=self.default_num_steps,
            parallel_vae=self.parallel_vae,
            vae_parallel_halo=self.vae_parallel_halo,
        )
