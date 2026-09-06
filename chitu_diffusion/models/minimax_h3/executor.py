from __future__ import annotations

import json
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from safetensors.torch import save

from ...epe.contracts import (
    ExecutorBuildContext,
    TerminalArtifact,
    TransferBundle,
)
from ...epe.executor import (
    DiffusersBackend,
    build_stage_parallel_context,
    build_stage_vae_placement,
    scheduling_options_from_pool,
)
from ...epe.scheduling.planner import EpeSchedulingModule
from ...epe.scheduling.types import RequestProfile
from ...parallel.vae import VaeParallelPlacement
from .api import MiniMaxH3Request
from .conditioning import load_conditioning_package, make_synthetic_conditioning
from .cost import H3CostFeatures, MiniMaxH3StepCostModel
from .fl2va_conditioning import (
    FL2VAConditioning,
    FL2VAFrameMetadata,
    patchify_video_latent,
    prepare_fl2va_noise,
)
from .fl2va_geometry import FL2VATimeRequest, resolve_fl2va_plan
from .fl2va_packed_sequence import build_fl2va_packed_sequence
from .loader import load_minimax_h3_transformer
from .packed_sequence import build_packed_sequence
from .pipeline import (
    MiniMaxH3DenoiseState,
    MiniMaxH3LatentPipeline,
    minimax_h3_time_shift_sigmas,
    unpack_latent_tensors,
)


class MiniMaxH3LatentExecutor(DiffusersBackend):
    """EPAC executor whose terminal artifact is a latent safetensors package."""

    model_name = "minimax_h3"
    output_media_type = "application/octet-stream"

    def __init__(
        self,
        pipeline: MiniMaxH3LatentPipeline,
        scheduling_module: EpeSchedulingModule,
        *,
        default_width: int = 8,
        default_height: int = 8,
        default_latent_t: int = 2,
        default_audio_t: int = 3,
        default_audio_channels: int = 2,
        default_text_length: int = 32,
        default_num_steps: int = 50,
        text_processor: Any | None = None,
        text_encoder: torch.nn.Module | None = None,
        video_vae: torch.nn.Module | None = None,
        audio_vae: torch.nn.Module | None = None,
        native_enabled: bool = False,
        default_duration_s: float = 5.0,
        default_fps: int = 24,
        default_output_type: str = "latent",
        ffmpeg_path: str = "ffmpeg",
        vae_placement: VaeParallelPlacement | None = None,
        warmup_media_profiles: tuple[Any, ...] = (),
        warmup_burnin_steps: int = 0,
        cost_profile_path: str | None = None,
    ) -> None:
        super().__init__(
            pipeline,
            scheduling_module,
            default_width=default_width,
            default_height=default_height,
            default_num_steps=default_num_steps,
            vae_placement=vae_placement,
        )
        self.default_latent_t = int(default_latent_t)
        self.default_audio_t = int(default_audio_t)
        self.default_audio_channels = int(default_audio_channels)
        self.default_text_length = int(default_text_length)
        self.text_processor = text_processor
        self.text_encoder = text_encoder
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.native_enabled = bool(native_enabled)
        self.default_duration_s = float(default_duration_s)
        self.default_fps = int(default_fps)
        self.default_output_type = str(default_output_type)
        self.ffmpeg_path = str(ffmpeg_path)
        self.warmup_media_profiles = tuple(warmup_media_profiles)
        self.warmup_burnin_steps = int(warmup_burnin_steps)
        if self.warmup_burnin_steps < 0:
            raise ValueError("warmup_burnin_steps must be non-negative")
        self.cost_profile_path = cost_profile_path
        self._cost_feature_cache: dict[str, H3CostFeatures] = {}

    @staticmethod
    def _is_native(request: MiniMaxH3Request) -> bool:
        return not request.synthetic and request.conditioning_path is None

    def normalize_request(self, request: Any) -> MiniMaxH3Request:
        if isinstance(request, MiniMaxH3Request):
            return request
        if hasattr(request, "model_dump"):
            request = request.model_dump()
        if not isinstance(request, Mapping):
            raise TypeError("MiniMax-H3 requests must be MiniMaxH3Request or a mapping")
        from ...flexcache.config import CacheConfig

        def canonicalize(payload: Mapping[str, Any]) -> dict[str, Any]:
            normalized = dict(payload)
            for alias, canonical in (
                ("condition_path", "conditioning_path"),
                ("latent_h", "height"),
                ("latent_w", "width"),
            ):
                if canonical not in normalized and alias in normalized:
                    normalized[canonical] = normalized[alias]
            return normalized

        outer = canonicalize(request)
        nested = outer.pop("model_inputs", None)
        if nested is not None:
            if not isinstance(nested, Mapping):
                raise TypeError("model_inputs must be a mapping")
            outer.update(canonicalize(nested))
        request = outer
        return MiniMaxH3Request(
            conditioning_path=request.get("conditioning_path"),
            prompt=request.get("prompt"),
            first_frame_path=request.get("first_frame_path"),
            last_frame_path=request.get("last_frame_path"),
            request_id=str(request.get("request_id") or uuid.uuid4().hex),
            width=int(request.get("width") or self.default_width),
            height=int(request.get("height") or self.default_height),
            latent_t=int(request.get("latent_t") or self.default_latent_t),
            audio_t=int(request.get("audio_t") or self.default_audio_t),
            audio_channels=int(
                request.get("audio_channels") or self.default_audio_channels
            ),
            text_length=int(request.get("text_length") or self.default_text_length),
            num_steps=int(request.get("num_steps") or self.default_num_steps),
            flow_shift=(
                None
                if request.get("flow_shift") is None
                else float(request["flow_shift"])
            ),
            audio_flow_shift=(
                None
                if request.get("audio_flow_shift") is None
                else float(request["audio_flow_shift"])
            ),
            synthetic=bool(request.get("synthetic", False)),
            seed=int(request.get("seed", 0)),
            deadline_ms=(
                None
                if request.get("deadline_ms") is None
                else float(request["deadline_ms"])
            ),
            priority=int(request.get("priority", 0)),
            output_type=str(request.get("output_type", self.default_output_type)),
            duration_s=float(
                request.get("duration_s") or self.default_duration_s
            ),
            fps=int(request.get("fps") or self.default_fps),
            cache=(
                request["cache"]
                if isinstance(request.get("cache"), CacheConfig)
                else CacheConfig.from_mapping(request.get("cache"))
            ),
            extra_inputs=dict(request.get("extra_inputs") or {}),
        )

    def serialize_request(self, request: Any) -> dict[str, Any]:
        value = self.normalize_request(request)
        return {
            "request_id": value.request_id,
            "conditioning_path": value.conditioning_path,
            "prompt": value.prompt,
            "first_frame_path": value.first_frame_path,
            "last_frame_path": value.last_frame_path,
            "width": value.width,
            "height": value.height,
            "latent_t": value.latent_t,
            "audio_t": value.audio_t,
            "audio_channels": value.audio_channels,
            "text_length": value.text_length,
            "num_steps": value.num_steps,
            "flow_shift": value.flow_shift,
            "audio_flow_shift": value.audio_flow_shift,
            "synthetic": value.synthetic,
            "seed": value.seed,
            "deadline_ms": value.deadline_ms,
            "priority": value.priority,
            "output_type": value.output_type,
            "duration_s": value.duration_s,
            "fps": value.fps,
            "cache": value.cache.to_dict(),
            "extra_inputs": {},
        }

    def _packed_for_profile(self, request: MiniMaxH3Request):
        if self._is_native(request):
            if self.text_processor is None:
                raise RuntimeError("native MiniMax-H3 processor is not loaded")
            plan = resolve_fl2va_plan(
                FL2VATimeRequest(
                    duration_seconds=request.duration_s,
                    width=request.width,
                    height=request.height,
                    seed=request.seed,
                    fps=request.fps,
                )
            )
            first_image = self._load_image(request.first_frame_path)
            last_image = self._load_image(request.last_frame_path)
            if first_image is not None:
                first_image = first_image.resize((plan.canvas.width, plan.canvas.height))
            if last_image is not None:
                last_image = last_image.resize((plan.canvas.width, plan.canvas.height))
            presentation = self.text_processor(
                request.prompt or "",
                first_image=first_image,
                last_image=last_image,
            )
            conditioning = self._native_conditioning(request)
            return build_fl2va_packed_sequence(
                text_length=int(presentation.input_ids.numel()),
                latent_t=plan.video_latent_t,
                latent_h=plan.latent_height,
                latent_w=plan.latent_width,
                audio_t=plan.audio_latent_t,
                audio_channels=request.audio_channels,
                conditioning=conditioning,
                frame_count=plan.frame_count,
            )
        return build_packed_sequence(
            text_length=request.text_length,
            latent_t=request.latent_t,
            latent_h=request.height,
            latent_w=request.width,
            audio_t=request.audio_t,
            audio_channels=request.audio_channels,
        )

    def _cost_features(self, request: MiniMaxH3Request) -> H3CostFeatures:
        cached = self._cost_feature_cache.get(request.request_id)
        if cached is not None:
            return cached
        features = H3CostFeatures.from_packed(
            self._packed_for_profile(request),
            output_type=request.output_type,
            synthetic=request.synthetic,
        )
        self._cost_feature_cache[request.request_id] = features
        return features

    def _sequence_length(self, request: MiniMaxH3Request) -> int:
        return self._cost_features(request).padded_tokens

    def validate_request(
        self,
        request: Any,
        *,
        warmup_resolutions: tuple[tuple[int, int], ...],
    ) -> None:
        value = self.normalize_request(request)
        if self._is_native(value):
            if not self.native_enabled:
                raise ValueError("native MiniMax-H3 media pipeline is disabled")
            return
        if (value.height, value.width) not in set(warmup_resolutions):
            raise ValueError(
                f"unsupported latent shape {value.height}x{value.width}; "
                "it was not measured during warmup"
            )

    def request_profile(
        self, request: Any, *, completed_steps: int
    ) -> RequestProfile:
        value = self.normalize_request(request)
        features = self._cost_features(value)
        sequence_length = features.padded_tokens
        return RequestProfile(
            total_steps=value.num_steps - 1,
            completed_steps=int(completed_steps),
            shape_key=(
                value.latent_t,
                value.height,
                value.width,
                value.audio_t,
                features.text_tokens,
            ),
            image_tokens=sequence_length,
            attributes={
                "sequence_length": sequence_length,
                "batch_size": 1,
                "conditions": 1,
                "state_bytes": self._request_state_bytes(
                    value,
                    sequence_length,
                    text_tokens=features.text_tokens,
                ),
                "state_tensor_count": 6,
                "cost_features": features.to_dict(),
            },
        )

    def profile(self, state: MiniMaxH3DenoiseState) -> RequestProfile:
        features = H3CostFeatures.from_packed(
            state.packed,
            output_type=state.output_type,
            synthetic=not hasattr(state.packed, "condition_slice"),
        )
        return RequestProfile(
            total_steps=state.video_sigmas.numel() - 1,
            completed_steps=state.step_index,
            shape_key=(state.height, state.width),
            image_tokens=state.packed.sequence_length,
            attributes={
                "sequence_length": state.packed.sequence_length,
                "batch_size": 1,
                "conditions": 1,
                "cost_features": features.to_dict(),
                "state_bytes": (
                    state.video_latents.numel() * state.video_latents.element_size()
                    + state.audio_latents.numel()
                    * state.audio_latents.element_size()
                    + state.refined_prompt_embeds.numel()
                    * state.refined_prompt_embeds.element_size()
                    + state.video_sigmas.numel() * state.video_sigmas.element_size()
                    + state.audio_sigmas.numel() * state.audio_sigmas.element_size()
                    + (
                        0
                        if state.condition_video_rows is None
                        else state.condition_video_rows.numel()
                        * state.condition_video_rows.element_size()
                    )
                ),
                "state_tensor_count": (
                    6 if state.condition_video_rows is not None else 5
                ),
            },
        )

    def _request_conditions(self, request: MiniMaxH3Request) -> int:
        del request
        return 1

    def _request_state_bytes(
        self,
        request: MiniMaxH3Request,
        image_tokens: int,
        *,
        text_tokens: int | None = None,
    ) -> int:
        if self._is_native(request):
            plan = resolve_fl2va_plan(
                FL2VATimeRequest(
                    duration_seconds=request.duration_s,
                    width=request.width,
                    height=request.height,
                    seed=request.seed,
                    fps=request.fps,
                )
            )
            latent_t, height, width, audio_t = (
                plan.video_latent_t,
                plan.latent_height,
                plan.latent_width,
                plan.audio_latent_t,
            )
        else:
            latent_t, height, width, audio_t = (
                request.latent_t,
                request.height,
                request.width,
                request.audio_t,
            )
        config = self.pipeline.transformer.config
        latent_bytes = 4 * (
            config.latents_dim
            * latent_t
            * height
            * width
            + request.audio_channels
            * audio_t
            * config.audio_latents_dim
        )
        static_bytes = (
            2
            * (request.text_length if text_tokens is None else int(text_tokens))
            * config.hidden_size
            + 8 * request.num_steps
        )
        return latent_bytes + static_bytes

    def _state_conditions(self, state: MiniMaxH3DenoiseState) -> int:
        del state
        return 1

    def prepare_request(self, request: Any) -> MiniMaxH3DenoiseState:
        value = self.normalize_request(request)
        if self._is_native(value):
            return self._prepare_native_request(value)
        config = self.pipeline.transformer.config
        conditioning = (
            make_synthetic_conditioning(
                text_length=value.text_length,
                text_dim=config.text_dim,
                latent_t=value.latent_t,
                latent_h=value.height,
                latent_w=value.width,
                latent_channels=config.latents_dim,
                audio_t=value.audio_t,
                audio_channels=value.audio_channels,
                audio_latent_dim=config.audio_latents_dim,
                device=self.pipeline.device,
                seed=value.seed,
            )
            if value.synthetic
            else load_conditioning_package(value.conditioning_path or "")
        )
        state = self.pipeline.prepare_request(
            conditioning, num_inference_steps=value.num_steps
        )
        if value.flow_shift is not None:
            state.video_sigmas = minimax_h3_time_shift_sigmas(
                value.num_steps,
                shift=value.flow_shift,
                device=self.pipeline.device,
            )
        if value.audio_flow_shift is not None:
            state.audio_sigmas = minimax_h3_time_shift_sigmas(
                value.num_steps,
                shift=value.audio_flow_shift,
                device=self.pipeline.device,
            )
        expected = (
            value.latent_t,
            value.height,
            value.width,
            value.audio_t,
            value.audio_channels,
            value.text_length,
        )
        actual = (
            state.video_latents.shape[1],
            state.video_latents.shape[2],
            state.video_latents.shape[3],
            state.audio_latents.shape[1],
            state.audio_latents.shape[0],
            state.refined_prompt_embeds.shape[0],
        )
        if actual != expected:
            raise ValueError(
                f"conditioning package shape {actual} does not match request {expected}"
            )
        return state

    @staticmethod
    def _native_conditioning(request: MiniMaxH3Request) -> FL2VAConditioning:
        frames = []
        if request.first_frame_path:
            frames.append(FL2VAFrameMetadata(0, request.first_frame_path))
        if request.last_frame_path:
            frames.append(FL2VAFrameMetadata(-1, request.last_frame_path))
        return FL2VAConditioning(request.prompt or "", tuple(frames))

    @staticmethod
    def _load_image(path: str | None) -> Any | None:
        if path is None:
            return None
        from PIL import Image

        with Image.open(path) as image:
            return image.convert("RGB").copy()

    @torch.inference_mode()
    def _prepare_native_request(
        self, request: MiniMaxH3Request
    ) -> MiniMaxH3DenoiseState:
        if self.text_processor is None or self.text_encoder is None:
            raise RuntimeError("native MiniMax-H3 text components are not loaded")
        plan = resolve_fl2va_plan(
            FL2VATimeRequest(
                duration_seconds=request.duration_s,
                width=request.width,
                height=request.height,
                seed=request.seed,
                fps=request.fps,
            )
        )
        first_image = self._load_image(request.first_frame_path)
        last_image = self._load_image(request.last_frame_path)
        if first_image is not None:
            first_image = first_image.resize((plan.canvas.width, plan.canvas.height))
        if last_image is not None:
            last_image = last_image.resize((plan.canvas.width, plan.canvas.height))
        encoded_inputs = self.text_processor(
            request.prompt or "",
            first_image=first_image,
            last_image=last_image,
        )
        encoder_kwargs = {
            name: getattr(encoded_inputs, name)
            for name in (
                "pixel_values",
                "image_grid_thw",
                "position_ids",
                "mm_token_type_ids",
            )
            if hasattr(encoded_inputs, name)
            and getattr(encoded_inputs, name) is not None
        }
        prompt_embeds = self.text_encoder(
            encoded_inputs.input_ids.to(self.pipeline.device),
            **{
                name: tensor.to(self.pipeline.device)
                for name, tensor in encoder_kwargs.items()
            },
        )
        noise = prepare_fl2va_noise(
            plan,
            seed=request.seed,
            video_channels=self.pipeline.transformer.config.latents_dim,
            audio_channels=request.audio_channels,
            audio_latent_dim=self.pipeline.transformer.config.audio_latents_dim,
        )
        conditioning = self._native_conditioning(request)
        packed = build_fl2va_packed_sequence(
            text_length=prompt_embeds.shape[0],
            latent_t=plan.video_latent_t,
            latent_h=plan.latent_height,
            latent_w=plan.latent_width,
            audio_t=plan.audio_latent_t,
            audio_channels=request.audio_channels,
            conditioning=conditioning,
            frame_count=plan.frame_count,
        )
        condition_rows = self._encode_condition_frames(
            (first_image, last_image),
            expected_rows=packed.condition_slice.stop - packed.condition_slice.start,
        )
        state = self.pipeline.prepare_native_request(
            prompt_embeds=prompt_embeds,
            video_noise=noise.video,
            audio_rows=noise.audio_rows,
            packed=packed,
            num_inference_steps=request.num_steps,
            condition_video_rows=condition_rows,
            output_type=request.output_type,
            fps=request.fps,
            frame_count=plan.frame_count,
        )
        return state

    def _encode_condition_frames(
        self,
        images: tuple[Any | None, Any | None],
        *,
        expected_rows: int,
    ) -> torch.Tensor | None:
        selected = [image for image in images if image is not None]
        if not selected:
            return None
        # Rank 0 always owns a video VAE: it leads every decode placement.
        source_rank = 0
        if self.parallel_context.rank == source_rank:
            encode = (
                None
                if self.video_vae is None
                else getattr(self.video_vae, "encode", None)
            )
            if not callable(encode):
                raise RuntimeError("native keyframes require a video VAE encode method")
            rows = []
            for image in selected:
                latent = encode(image)
                if latent.ndim == 4:
                    latent = latent.unsqueeze(2)
                if latent.ndim != 5:
                    raise ValueError(
                        "video VAE keyframe encode must return [1,C,T,H,W]"
                    )
                rows.append(patchify_video_latent(latent[:, :, :1]))
            result = torch.cat(rows).to(
                device=self.pipeline.device,
                dtype=torch.float32,
            )
        else:
            result = torch.empty(
                expected_rows,
                self.pipeline.transformer.config.video_patch_dim,
                device=self.pipeline.device,
                dtype=torch.float32,
            )
        if self.parallel_context.world_size > 1:
            dist.broadcast(result, src=source_rank)
        if result.shape[0] != expected_rows:
            raise ValueError("encoded keyframe rows do not match FL2VA packed geometry")
        return result

    def export_state(self, state: MiniMaxH3DenoiseState) -> TransferBundle:
        tensors = {
            "video_latents": state.video_latents,
            "audio_latents": state.audio_latents,
            "refined_prompt_embeds": state.refined_prompt_embeds,
            "video_sigmas": state.video_sigmas,
            "audio_sigmas": state.audio_sigmas,
        }
        if state.condition_video_rows is not None:
            tensors["condition_video_rows"] = state.condition_video_rows
        return TransferBundle(
            tensors=tensors,
            step_index=state.step_index,
            metadata={"sequence_length": state.packed.sequence_length},
        )

    def allocate_request(self, request: Any) -> MiniMaxH3DenoiseState:
        """Allocate migration receive buffers without Qwen or random sampling."""

        value = self.normalize_request(request)
        config = self.pipeline.transformer.config
        if self._is_native(value):
            plan = resolve_fl2va_plan(
                FL2VATimeRequest(
                    duration_seconds=value.duration_s,
                    width=value.width,
                    height=value.height,
                    seed=value.seed,
                    fps=value.fps,
                )
            )
            if self.text_processor is None:
                raise RuntimeError("native MiniMax-H3 processor is not loaded")
            inputs = self.text_processor(
                value.prompt or "",
                first_image=self._load_image(value.first_frame_path),
                last_image=self._load_image(value.last_frame_path),
            )
            text_length = int(inputs.input_ids.numel())
            conditioning = self._native_conditioning(value)
            packed = build_fl2va_packed_sequence(
                text_length=text_length,
                latent_t=plan.video_latent_t,
                latent_h=plan.latent_height,
                latent_w=plan.latent_width,
                audio_t=plan.audio_latent_t,
                audio_channels=value.audio_channels,
                conditioning=conditioning,
                frame_count=plan.frame_count,
            )
            condition_count = packed.condition_slice.stop - packed.condition_slice.start
            return MiniMaxH3DenoiseState(
                video_latents=torch.empty(
                    config.latents_dim,
                    plan.video_latent_t,
                    plan.latent_height,
                    plan.latent_width,
                    dtype=torch.float32,
                    device=self.pipeline.device,
                ),
                audio_latents=torch.empty(
                    value.audio_channels,
                    plan.audio_latent_t,
                    config.audio_latents_dim,
                    dtype=torch.float32,
                    device=self.pipeline.device,
                ),
                refined_prompt_embeds=torch.empty(
                    text_length,
                    config.hidden_size,
                    dtype=torch.bfloat16,
                    device=self.pipeline.device,
                ),
                packed=packed,
                video_sigmas=torch.empty(
                    value.num_steps, dtype=torch.float32, device=self.pipeline.device
                ),
                audio_sigmas=torch.empty(
                    value.num_steps, dtype=torch.float32, device=self.pipeline.device
                ),
                width=plan.latent_width,
                height=plan.latent_height,
                image_tokens=packed.sequence_length,
                condition_video_rows=(
                    torch.empty(
                        condition_count,
                        config.latents_dim
                        * config.patch_size[0]
                        * config.patch_size[1]
                        * config.patch_size[2],
                        dtype=torch.float32,
                        device=self.pipeline.device,
                    )
                    if condition_count
                    else None
                ),
                output_type=value.output_type,
                fps=value.fps,
                frame_count=plan.frame_count,
            )
        packed = build_packed_sequence(
            text_length=value.text_length,
            latent_t=value.latent_t,
            latent_h=value.height,
            latent_w=value.width,
            audio_t=value.audio_t,
            audio_channels=value.audio_channels,
        )
        return MiniMaxH3DenoiseState(
            video_latents=torch.empty(
                config.latents_dim,
                value.latent_t,
                value.height,
                value.width,
                dtype=torch.float32,
                device=self.pipeline.device,
            ),
            audio_latents=torch.empty(
                value.audio_channels,
                value.audio_t,
                config.audio_latents_dim,
                dtype=torch.float32,
                device=self.pipeline.device,
            ),
            refined_prompt_embeds=torch.empty(
                value.text_length,
                config.hidden_size,
                dtype=torch.bfloat16,
                device=self.pipeline.device,
            ),
            packed=packed,
            video_sigmas=torch.empty(
                value.num_steps, dtype=torch.float32, device=self.pipeline.device
            ),
            audio_sigmas=torch.empty(
                value.num_steps, dtype=torch.float32, device=self.pipeline.device
            ),
            width=value.width,
            height=value.height,
            image_tokens=packed.sequence_length,
        )

    def finalize_gpu(
        self,
        state: MiniMaxH3DenoiseState,
        *,
        lane_ranks: tuple[int, ...],
        timings: dict[str, object],
    ) -> TerminalArtifact | torch.Tensor | None:
        now = time.time_ns()
        timings.setdefault("vae_start_unix_ns", now)
        timings.setdefault("vae_end_unix_ns", now)
        timings.setdefault("vae_decode_ms", 0.0)
        if state.output_type == "mp4":
            plane_lane = self.parallel_context.plane_lane(lane_ranks)
            with self.parallel_context.activate(plane_lane) as lane:
                topology = self.vae_placement.resolve(lane)
            if topology is None:
                return None
            if self.video_vae is None:
                raise RuntimeError("VAEP member is missing the video VAE")
            started = time.perf_counter()
            video = self.video_vae.decode_parallel(
                state.video_latents.unsqueeze(0),
                topology=topology,
                enabled=self.parallel_vae,
            )
            if video is None:
                return None
            if self.audio_vae is None:
                raise RuntimeError("VAEP leader is missing the audio VAE")
            audio = self.audio_vae.decode(
                state.audio_latents.transpose(1, 2).contiguous()
            )
            target_samples = round(
                state.frame_count * self.audio_vae.sample_rate / state.fps
            )
            if audio.shape[-1] < target_samples:
                audio = torch.nn.functional.pad(
                    audio, (0, target_samples - audio.shape[-1])
                )
            elif audio.shape[-1] > target_samples:
                audio = audio[..., :target_samples].contiguous()
            timings["vae_decode_ms"] = (time.perf_counter() - started) * 1000.0
            timings["vae_end_unix_ns"] = time.time_ns()
            return TerminalArtifact(
                tensors={"video": video, "audio": audio},
                metadata={
                    "output_type": "mp4",
                    "fps": state.fps,
                    "sample_rate": self.audio_vae.sample_rate,
                    "frame_count": state.frame_count,
                },
            )
        plane_lane = self.parallel_context.plane_lane(lane_ranks)
        with self.parallel_context.activate(plane_lane) as topology:
            packed = self.pipeline.finalize_latent(
                state, topology=topology, timings=timings
            )
        return packed if self.parallel_context.rank == lane_ranks[0] else None

    def close(self) -> None:
        super().close()

    def postprocess(
        self, host_output: torch.Tensor, *, output_type: str = "latent"
    ) -> dict[str, torch.Tensor]:
        del output_type
        video, audio = unpack_latent_tensors(host_output)
        return {"video_latents": video, "audio_latents": audio}

    def postprocess_artifact(
        self,
        host_artifact: TerminalArtifact,
        *,
        output_type: str = "latent",
    ) -> Any:
        selected_type = str(host_artifact.metadata.get("output_type", output_type))
        if selected_type == "mp4":
            from .media_export import export_mp4

            return export_mp4(
                host_artifact.tensors["video"],
                host_artifact.tensors["audio"],
                fps=float(host_artifact.metadata.get("fps", 24)),
                sample_rate=int(host_artifact.metadata.get("sample_rate", 32000)),
                ffmpeg_path=self.ffmpeg_path,
            )
        if len(host_artifact.tensors) != 1:
            raise TypeError("latent terminal artifact must contain one packed tensor")
        return self.postprocess(next(iter(host_artifact.tensors.values())))

    def package_generate_output(self, output: Any) -> Any:
        return output

    @staticmethod
    def media_type_for_output(payload: bytes | None, output: Any) -> str:
        del output
        if payload is not None and (
            payload[4:8] == b"ftyp" or b"ftyp" in payload[:32]
        ):
            return "video/mp4"
        return "application/octet-stream"

    def package_postprocessed_output(
        self,
        latents: Mapping[str, torch.Tensor] | bytes,
    ) -> tuple[bytes, Mapping[str, torch.Tensor] | bytes]:
        if isinstance(latents, bytes):
            return latents, latents
        tensors = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in latents.items()
        }
        metadata = {
            f"{name}_shape": str(list(tensor.shape))
            for name, tensor in tensors.items()
        }
        return save(tensors, metadata=metadata), tensors

    def generate(self, request: Any) -> Mapping[str, torch.Tensor] | None:
        state = self.prepare_request(request)
        ranks = tuple(range(self.parallel_context.world_size))
        while not state.complete:
            self.denoise_step(state, lane_ranks=ranks)
        if self.parallel_context.rank != ranks[0]:
            return None
        return {
            "video_latents": state.video_latents,
            "audio_latents": state.audio_latents,
        }

    @torch.inference_mode()
    def warmup(self, *, resolutions, steps) -> dict[str, Any]:
        if steps <= 0:
            raise ValueError("warmup steps must be positive")
        config = self.pipeline.transformer.config
        rows: list[dict[str, Any]] = []
        started_report = time.perf_counter()
        parallel = self.parallel_context
        warmup_cases: list[
            tuple[MiniMaxH3Request, tuple[int, int], Mapping[str, Any] | None]
        ] = [
            (
                MiniMaxH3Request(
                    synthetic=True,
                    width=int(width),
                    height=int(height),
                    latent_t=self.default_latent_t,
                    audio_t=self.default_audio_t,
                    audio_channels=self.default_audio_channels,
                    text_length=self.default_text_length,
                    num_steps=self.warmup_burnin_steps + steps + 1,
                ),
                (int(height), int(width)),
                None,
            )
            for height, width in resolutions
        ]
        if self.native_enabled:
            for profile in self.warmup_media_profiles:
                media = (
                    profile
                    if isinstance(profile, Mapping)
                    else {
                        "duration_s": profile.duration_s,
                        "fps": profile.fps,
                        "height": profile.height,
                        "width": profile.width,
                        "output_type": profile.output_type,
                        "prompt": profile.prompt,
                        "first_frame_path": profile.first_frame_path,
                        "last_frame_path": profile.last_frame_path,
                        "profile_id": profile.profile_id,
                    }
                )
                warmup_cases.append(
                    (
                        MiniMaxH3Request(
                            prompt=str(media["prompt"]),
                            first_frame_path=media.get("first_frame_path"),
                            last_frame_path=media.get("last_frame_path"),
                            width=int(media["width"]),
                            height=int(media["height"]),
                            duration_s=float(media["duration_s"]),
                            fps=int(media["fps"]),
                            output_type=str(media["output_type"]),
                            num_steps=self.warmup_burnin_steps + steps + 1,
                        ),
                        (int(media["height"]), int(media["width"])),
                        media,
                    )
                )
        for request, resolution, media_profile in warmup_cases:
            height, width = resolution
            cost_features = self._cost_features(request)
            sequence_length = cost_features.padded_tokens
            for cp_width in parallel.allowed_widths:
                cp_offset = (parallel.cp_rank // cp_width) * cp_width
                lane_ranks = tuple(
                    tp * parallel.cp_world_size + cp
                    for tp in range(parallel.tensor_parallel.degree)
                    for cp in range(cp_offset, cp_offset + cp_width)
                )
                state = self.prepare_request(request)
                for _ in range(self.warmup_burnin_steps):
                    self.denoise_step(state, lane_ranks=lane_ranks)
                samples: list[float] = []
                for _ in range(steps):
                    if self.pipeline.device.type == "cuda":
                        torch.cuda.synchronize(self.pipeline.device)
                    started = time.perf_counter()
                    self.denoise_step(state, lane_ranks=lane_ranks)
                    if self.pipeline.device.type == "cuda":
                        torch.cuda.synchronize(self.pipeline.device)
                    if parallel.rank == lane_ranks[0]:
                        samples.append((time.perf_counter() - started) * 1000.0)
                terminal_metrics = None
                if parallel.rank == lane_ranks[0]:
                    terminal_started = time.perf_counter()
                    artifact = self.finalize_gpu(
                        state, lane_ranks=lane_ranks, timings={}
                    )
                    terminal_gpu_ms = (
                        time.perf_counter() - terminal_started
                    ) * 1000.0
                    terminal_d2h_ms = 0.0
                    terminal_cpu_ms = 0.0
                    if isinstance(artifact, TerminalArtifact):
                        d2h_started = time.perf_counter()
                        host_artifact = TerminalArtifact(
                            tensors={
                                name: tensor.to("cpu")
                                for name, tensor in artifact.tensors.items()
                            },
                            metadata=artifact.metadata,
                        )
                        terminal_d2h_ms = (
                            time.perf_counter() - d2h_started
                        ) * 1000.0
                        cpu_started = time.perf_counter()
                        self.postprocess_artifact(host_artifact)
                        terminal_cpu_ms = (
                            time.perf_counter() - cpu_started
                        ) * 1000.0
                    terminal_metrics = (
                        terminal_gpu_ms,
                        terminal_d2h_ms,
                        terminal_cpu_ms,
                    )
                local = (
                    {
                        "lane": lane_ranks,
                        "samples": samples,
                        "terminal": terminal_metrics,
                    }
                    if parallel.rank == lane_ranks[0]
                    else None
                )
                gathered = [local]
                if parallel.world_size > 1:
                    gathered = [None] * parallel.world_size
                    dist.all_gather_object(gathered, local)
                if parallel.rank == 0:
                    measured = [
                        sample
                        for item in gathered
                        if item is not None
                        for sample in item["samples"]
                    ]
                    terminals = [
                        item["terminal"]
                        for item in gathered
                        if item is not None and item["terminal"] is not None
                    ]
                    terminal_gpu_ms = float(
                        statistics.median(value[0] for value in terminals)
                    )
                    terminal_d2h_ms = float(
                        statistics.median(value[1] for value in terminals)
                    )
                    terminal_cpu_ms = float(
                        statistics.median(value[2] for value in terminals)
                    )
                    rows.append(
                        {
                            "resolution": [int(height), int(width)],
                            "sequence_length": sequence_length,
                            "image_tokens": sequence_length,
                            "width": cp_width,
                            "batch_size": 1,
                            "conditions": 1,
                            "cost_features": cost_features.to_dict(),
                            "latency_ms": float(statistics.median(measured)),
                            "samples_ms": measured,
                            "state_bytes": self._request_state_bytes(
                                request,
                                sequence_length,
                                text_tokens=cost_features.text_tokens,
                            ),
                            "state_tensor_count": 6,
                            "terminal_gpu_ms": terminal_gpu_ms,
                            "terminal_d2h_ms": terminal_d2h_ms,
                            "terminal_cpu_ms": terminal_cpu_ms,
                            "terminal_ms": terminal_gpu_ms + terminal_d2h_ms,
                            "completion_tail_ms": (
                                terminal_gpu_ms
                                + terminal_d2h_ms
                                + terminal_cpu_ms
                            ),
                            "media_profile": media_profile,
                            "profile_id": (
                                None
                                if media_profile is None
                                else media_profile.get("profile_id")
                            ),
                        }
                    )
                del state
        external_rows: list[dict[str, Any]] = []
        if self.cost_profile_path:
            profile_path = Path(self.cost_profile_path).expanduser()
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            external_rows = [dict(row) for row in profile.get("rows", ())]
            if not external_rows:
                raise ValueError("H3 cost profile contains no rows")
            rows.extend(external_rows)
        report_box = [
            {
                "source": "startup_warmup",
                "steps": steps,
                "burnin_steps": self.warmup_burnin_steps,
                "resolutions": [list(value) for value in resolutions],
                "allowed_lane_widths": list(parallel.allowed_widths),
                "tensor_parallel_degree": parallel.tensor_parallel.degree,
                "rows": rows,
                "external_cost_profile": self.cost_profile_path,
                "external_cost_rows": len(external_rows),
                "total_wall_ms": (time.perf_counter() - started_report) * 1000.0,
                "latent_channels": config.latents_dim,
            }
            if parallel.rank == 0
            else None
        ]
        if parallel.world_size > 1:
            dist.broadcast_object_list(report_box, src=0)
        report = report_box[0]
        assert report is not None
        self.scheduling_module.initialize_cost_model(report["rows"])
        return report


@dataclass(frozen=True)
class MiniMaxH3ExecutorFactory:
    model_path: str
    attention_backend: str = "auto"
    flow_shift: float = 12.0
    audio_flow_shift: float = 3.0
    default_width: int = 8
    default_height: int = 8
    default_latent_t: int = 2
    default_audio_t: int = 3
    default_audio_channels: int = 2
    default_text_length: int = 32
    default_num_steps: int = 50
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    processor_path: str | None = None
    video_vae_path: str | None = None
    audio_vae_path: str | None = None
    enable_native_media: bool = False
    default_duration_s: float = 5.0
    default_fps: int = 24
    default_output_type: str = "latent"
    ffmpeg_path: str = "ffmpeg"
    warmup_media_profiles: tuple[Any, ...] = ()
    warmup_burnin_steps: int = 0
    cost_profile_path: str | None = None

    def build(self, context: ExecutorBuildContext) -> MiniMaxH3LatentExecutor:
        # The packed model always needs a pure CP topology. The selected mode
        # decides whether attention transposes QKV with Ulysses or gathers K/V;
        # both fast transports are initialized on the same static lane.
        parallel, local_rank = build_stage_parallel_context(
            context,
            attention_mode="ulysses",
            ulysses_degree=context.cp.ulysses_degree or context.cp.world_size,
        )
        device = (
            torch.device("cuda", local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        vae_placement = build_stage_vae_placement(context)
        transformer = load_minimax_h3_transformer(
            self.model_path,
            device=device,
            attention_backend=self.attention_backend,
        )
        text_processor = None
        text_encoder = None
        video_vae = None
        audio_vae = None
        if self.enable_native_media:
            if not all(
                (
                    self.text_encoder_path,
                    self.video_vae_path,
                    self.audio_vae_path,
                )
            ):
                raise ValueError(
                    "native MiniMax-H3 requires text_encoder, video_vae, "
                    "and audio_vae component paths"
                )
            from .qwen3vl_loader import load_minimax_h3_qwen3vl_encoder
            from .qwen3vl_processor import MiniMaxH3Qwen3VLProcessor

            text_processor = MiniMaxH3Qwen3VLProcessor.from_pretrained(
                self.processor_path
                or self.tokenizer_path
                or self.text_encoder_path
                or ""
            )
            text_encoder = load_minimax_h3_qwen3vl_encoder(
                self.text_encoder_path or "", device=device
            )
            # A lane-scoped placement picks its members per request, so every
            # rank has to be able to decode. A fixed group is known up front.
            group = vae_placement.group
            decodes = group.is_member if group is not None else True
            leads = group.is_leader if group is not None else True
            if decodes:
                from .video_vae import load_video_vae

                video_vae = load_video_vae(self.video_vae_path or "", device=device)
            if leads:
                from .audio_vae import load_audio_vae

                audio_vae = load_audio_vae(self.audio_vae_path or "", device=device)
        pipeline = MiniMaxH3LatentPipeline(
            transformer,
            parallel,
            flow_shift=self.flow_shift,
            audio_flow_shift=self.audio_flow_shift,
            attention_mode=context.cp.attention_mode,
        )
        scheduling = EpeSchedulingModule(
            parallel,
            **scheduling_options_from_pool(context.pool),
            cost_model=MiniMaxH3StepCostModel(),
        )
        return MiniMaxH3LatentExecutor(
            pipeline,
            scheduling,
            default_width=self.default_width,
            default_height=self.default_height,
            default_latent_t=self.default_latent_t,
            default_audio_t=self.default_audio_t,
            default_audio_channels=self.default_audio_channels,
            default_text_length=self.default_text_length,
            default_num_steps=self.default_num_steps,
            text_processor=text_processor,
            text_encoder=text_encoder,
            video_vae=video_vae,
            audio_vae=audio_vae,
            native_enabled=self.enable_native_media,
            default_duration_s=self.default_duration_s,
            default_fps=self.default_fps,
            default_output_type=self.default_output_type,
            ffmpeg_path=self.ffmpeg_path,
            vae_placement=vae_placement,
            warmup_media_profiles=self.warmup_media_profiles,
            warmup_burnin_steps=self.warmup_burnin_steps,
            cost_profile_path=self.cost_profile_path,
        )
