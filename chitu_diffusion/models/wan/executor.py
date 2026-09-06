from __future__ import annotations

import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

import torch
import torch.distributed as dist
from diffusers.utils import export_to_video

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
from ...parallel.vae import VaeParallelPlacement, parallel_tiled_vae_decode
from .api import WanRequest
from .pipeline import EpeWanPipeline, WanDenoiseState, WanPipelineOutput


class WanVideoDecoderExecutor(DiffusersBackend):
    output_media_type = "video/mp4"

    flexcache_family = "wan"

    """Independent embedded EPE executor for Wan video requests."""

    model_name = "wan"

    def __init__(
        self,
        pipeline: EpeWanPipeline,
        scheduling_module: EpeSchedulingModule,
        *,
        default_width: int,
        default_height: int,
        default_num_frames: int,
        default_num_steps: int,
        vae_placement: VaeParallelPlacement | None = None,
        warmup_profiles: tuple[tuple[int, int, int], ...] = (),
    ) -> None:
        super().__init__(
            pipeline,
            scheduling_module,
            default_width=default_width,
            default_height=default_height,
            default_num_steps=default_num_steps,
            vae_placement=vae_placement,
        )
        self.default_num_frames = int(default_num_frames)
        self.warmup_profiles = tuple(
            sorted(
                {
                    (int(height), int(width), int(num_frames))
                    for height, width, num_frames in warmup_profiles
                }
            )
        )
        for height, width, num_frames in self.warmup_profiles:
            if min(height, width) < 16 or height % 16 or width % 16:
                raise ValueError(
                    "warmup profile sizes must be positive multiples of 16"
                )
            if num_frames < 1 or (num_frames - 1) % 4:
                raise ValueError("warmup profile frames must equal 4n+1")

    def warmup(self, *, resolutions, steps) -> dict[str, Any]:
        normalized_resolutions = tuple(
            sorted({(int(height), int(width)) for height, width in resolutions})
        )
        profiles = self.warmup_profiles or tuple(
            (height, width, self.default_num_frames)
            for height, width in normalized_resolutions
        )
        unsupported = sorted(
            {
                (height, width)
                for height, width, _ in profiles
                if (height, width) not in normalized_resolutions
            }
        )
        if unsupported:
            raise ValueError(
                f"warmup profiles use unconfigured resolutions: {unsupported}"
            )

        grouped: dict[int, list[tuple[int, int]]] = {}
        for height, width, num_frames in profiles:
            grouped.setdefault(num_frames, []).append((height, width))
        rows: list[dict[str, Any]] = []
        total_wall_ms = 0.0
        for num_frames, frame_resolutions in sorted(grouped.items()):
            profile_resolutions = tuple(sorted(set(frame_resolutions)))
            partial = self.pipeline.warmup_epe(
                resolutions=profile_resolutions,
                steps=steps,
                num_frames=num_frames,
            )
            terminal_rows = self._warmup_terminal(
                resolutions=profile_resolutions,
                steps=steps,
                num_frames=num_frames,
            )
            terminal_by_key = {
                (row["image_tokens"], row["width"]): row for row in terminal_rows
            }
            for row in partial["rows"]:
                row.update(terminal_by_key[(row["image_tokens"], row["width"])])
                rows.append(row)
            total_wall_ms += float(partial["total_wall_ms"])
        frame_counts = sorted(grouped)
        report = {
            "source": "startup_warmup",
            "steps": steps,
            "num_frames": frame_counts[0] if len(frame_counts) == 1 else None,
            "frame_counts": frame_counts,
            "profiles": [list(profile) for profile in profiles],
            "resolutions": [list(value) for value in normalized_resolutions],
            "allowed_lane_widths": list(self.parallel_context.allowed_widths),
            "rows": rows,
            "total_wall_ms": total_wall_ms,
        }
        report["parallel_vae"] = self.parallel_vae
        report["vae_parallel_halo"] = self.vae_parallel_halo
        self.scheduling_module.initialize_cost_model(report["rows"])
        return report

    @torch.inference_mode()
    def _warmup_terminal(
        self, *, resolutions, steps, num_frames: int
    ) -> list[dict[str, Any]]:
        device = next(self.pipeline.vae.parameters()).device
        latent_frames = (num_frames - 1) // self.pipeline.vae_scale_factor_temporal + 1
        rows: list[dict[str, Any]] = []
        for height, width in resolutions:
            image_tokens = latent_frames * (height // 16) * (width // 16)
            for lane_width in self.parallel_context.allowed_widths:
                offset = (self.parallel_context.rank // lane_width) * lane_width
                lane = tuple(range(offset, offset + lane_width))
                generator = torch.Generator(device=device).manual_seed(
                    8_000_023
                    + height * 97
                    + width * 193
                    + num_frames * 389
                    + lane[0] * 997
                )
                latents = torch.randn(
                    (1, 16, latent_frames, height // 8, width // 8),
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
                vae_samples_ms: list[float] = []
                d2h_samples_ms: list[float] = []
                with self.parallel_context.activate(lane) as topology:
                    for _ in range(steps):
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        video = parallel_tiled_vae_decode(
                            self.pipeline._denormalize_latents(latents),
                            lambda value: self.pipeline.vae.decode(
                                value, return_dict=False
                            )[0],
                            topology=topology,
                            latent_split_dim=4,
                            pixel_split_dim=4,
                            scale=int(self.pipeline.vae_scale_factor_spatial),
                            halo=self.vae_parallel_halo,
                            enabled=self.parallel_vae,
                        )
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        vae_samples_ms.append((time.perf_counter() - started) * 1000)
                        if topology.is_leader:
                            d2h_started = time.perf_counter()
                            assert video is not None
                            host = video.to("cpu")
                            d2h_samples_ms.append(
                                (time.perf_counter() - d2h_started) * 1000
                            )
                            del host
                        del video
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
                    terminal_samples: list[float] = []
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
                            terminal_samples.append(vae_ms + d2h_ms)
                    rows.append(
                        {
                            "resolution": [height, width],
                            "num_frames": num_frames,
                            "image_tokens": image_tokens,
                            "width": lane_width,
                            "batch_size": 1,
                            "cfg_conditions": 2,
                            "state_bytes": latents.numel() * latents.element_size(),
                            "terminal_ms": float(statistics.median(terminal_samples)),
                            "vae_latency_ms": float(
                                statistics.median(vae_critical_samples)
                            ),
                            "d2h_latency_ms": float(statistics.median(d2h_samples)),
                            "terminal_samples_ms": terminal_samples,
                        }
                    )
                del latents
        broadcast = [rows if self.parallel_context.rank == 0 else None]
        if self.parallel_context.world_size > 1:
            dist.broadcast_object_list(broadcast, src=0)
        assert broadcast[0] is not None
        return broadcast[0]

    def normalize_request(self, request: Any) -> WanRequest:
        if isinstance(request, WanRequest):
            return request
        if hasattr(request, "model_dump"):
            request = request.model_dump()
        if not isinstance(request, Mapping):
            raise TypeError("Wan requests must be WanRequest or a mapping")
        return WanRequest(
            prompt=request.get("prompt"),
            negative_prompt=request.get("negative_prompt", ""),
            request_id=str(request.get("request_id") or uuid.uuid4().hex),
            width=int(request.get("width") or self.default_width),
            height=int(request.get("height") or self.default_height),
            num_frames=int(request.get("num_frames") or self.default_num_frames),
            num_steps=int(request.get("num_steps") or self.default_num_steps),
            guidance_scale=float(request.get("guidance_scale", 6.0)),
            seed=int(request.get("seed", 0)),
            deadline_ms=request.get("deadline_ms"),
            priority=int(request.get("priority", 0)),
            output_type=str(request.get("output_type", "np")),
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
            "num_frames": normalized.num_frames,
            "num_steps": normalized.num_steps,
            "guidance_scale": normalized.guidance_scale,
            "seed": normalized.seed,
            "deadline_ms": normalized.deadline_ms,
            "priority": normalized.priority,
            "output_type": normalized.output_type,
            "cache": normalized.cache.to_dict(),
            "extra_inputs": dict(normalized.extra_inputs),
        }

    def request_profile(self, request: Any, *, completed_steps: int) -> RequestProfile:
        normalized = self.normalize_request(request)
        latent_frames = (
            normalized.num_frames - 1
        ) // self.pipeline.vae_scale_factor_temporal + 1
        image_tokens = (
            latent_frames * (normalized.height // 16) * (normalized.width // 16)
        )
        return RequestProfile(
            total_steps=normalized.num_steps,
            completed_steps=int(completed_steps),
            shape_key=(normalized.height, normalized.width),
            image_tokens=image_tokens,
            attributes={
                "batch_size": 1,
                "conditions": self._request_conditions(normalized),
                "state_bytes": self._request_state_bytes(normalized, image_tokens),
                "num_frames": normalized.num_frames,
            },
        )

    def _request_conditions(self, request: WanRequest) -> int:
        return 2 if request.guidance_scale > 1 else 1

    def _request_state_bytes(self, request: WanRequest, image_tokens: int) -> int:
        del image_tokens
        latent_frames = (
            request.num_frames - 1
        ) // self.pipeline.vae_scale_factor_temporal + 1
        return (
            16
            * latent_frames
            * (request.height // 8)
            * (request.width // 8)
            * torch.tensor([], dtype=torch.float32).element_size()
        )

    def prepare_request(self, request: Any) -> WanDenoiseState:
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
            "num_frames",
            "num_inference_steps",
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
            num_frames=normalized.num_frames,
            num_inference_steps=normalized.num_steps,
            guidance_scale=normalized.guidance_scale,
            generator=torch.Generator(device=device).manual_seed(normalized.seed),
            **kwargs,
        )

    def _state_conditions(self, state: WanDenoiseState) -> int:
        return 2 if state.do_classifier_free_guidance else 1

    def profile(self, state: WanDenoiseState) -> RequestProfile:
        return RequestProfile(
            total_steps=len(state.timesteps),
            completed_steps=state.step_index,
            shape_key=(state.height, state.width),
            image_tokens=state.image_tokens,
            attributes={
                "batch_size": state.actual_batch_size,
                "conditions": self._state_conditions(state),
                "state_bytes": state.latents.numel() * state.latents.element_size(),
                "num_frames": state.num_frames,
            },
        )

    def postprocess(
        self,
        host_output: torch.Tensor,
        *,
        output_type: str = "np",
    ) -> Any:
        return self.pipeline.video_processor.postprocess_video(
            host_output, output_type=output_type
        )

    def package_generate_output(self, output: Any) -> WanPipelineOutput:
        return WanPipelineOutput(frames=output)

    @staticmethod
    def package_postprocessed_output(videos: Any) -> tuple[bytes, Any]:
        video = videos[0]
        with TemporaryDirectory(prefix="chitu-wan-") as directory:
            output_path = Path(directory) / "output.mp4"
            export_to_video(video, str(output_path), fps=16)
            output_bytes = output_path.read_bytes()
        return output_bytes, video


@dataclass(frozen=True)
class WanExecutorFactory:
    model_path: str
    torch_dtype: torch.dtype | None = None
    local_files_only: bool = True
    default_width: int = 832
    default_height: int = 480
    default_num_frames: int = 17
    default_num_steps: int = 50
    flow_shift: float = 8.0
    warmup_profiles: tuple[tuple[int, int, int], ...] = ()

    def build(self, context: ExecutorBuildContext) -> WanVideoDecoderExecutor:
        parallel, local_rank = build_stage_parallel_context(context)
        placement = build_stage_vae_placement(context)
        pipeline = EpeWanPipeline.from_pretrained(
            self.model_path,
            parallel_context=parallel,
            attention_mode=context.cp.attention_mode,
            ulysses_degree=context.cp.ulysses_degree,
            tensor_parallel_degree=context.tensor_parallel_degree,
            cfg_parallel=context.model.cfg_parallel,
            parallel_vae=placement.sharded,
            vae_parallel_halo=placement.halo,
            flow_shift=self.flow_shift,
            torch_dtype=self.torch_dtype
            or (torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            local_files_only=self.local_files_only,
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda", local_rank))
        pipeline.set_progress_bar_config(disable=True)
        scheduling = EpeSchedulingModule(
            parallel, **scheduling_options_from_pool(context.pool)
        )
        return WanVideoDecoderExecutor(
            pipeline,
            scheduling,
            default_width=self.default_width,
            default_height=self.default_height,
            default_num_frames=self.default_num_frames,
            default_num_steps=self.default_num_steps,
            vae_placement=placement,
            warmup_profiles=self.warmup_profiles,
        )
