"""EPE backend and factory for single-node Hunyuan Image 3 parallel stages."""

from __future__ import annotations

import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist

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
from .api import HunyuanImage3Request
from .loader import load_hunyuan_image3
from .parallel import (
    HunyuanImage3ParallelPlan,
    HunyuanImage3ParallelRuntime,
    resolve_parallel_plan,
)
from .pipeline import (
    CFG_CONDITIONS,
    HunyuanImage3DenoiseState,
    HunyuanImage3Pipeline,
)

WARMUP_PROMPT = "a calibration chart of colored squares on a neutral background"


def denormalize_image(image: torch.Tensor) -> torch.Tensor:
    """Map the released VAE output range onto ``[0, 1]``."""

    return (image / 2 + 0.5).clamp(0, 1)


class HunyuanImage3Executor(DiffusersBackend):
    """TPxCFGxCPxEP backend whose terminal artifact is one image."""

    model_name = "hunyuan-image3"
    output_media_type = "image/png"

    def __init__(
        self,
        pipeline: HunyuanImage3Pipeline,
        scheduling_module: EpeSchedulingModule,
        *,
        default_width: int = 1024,
        default_height: int = 1024,
        default_num_steps: int = 50,
        default_guidance_scale: float = 5.0,
        default_system_prompt: str | None = None,
        default_output_type: str = "pil",
    ) -> None:
        super().__init__(
            pipeline,
            scheduling_module,
            default_width=default_width,
            default_height=default_height,
            default_num_steps=default_num_steps,
            vae_placement=pipeline.vae_placement,
        )
        self.default_guidance_scale = float(default_guidance_scale)
        self.default_system_prompt = default_system_prompt
        self.default_output_type = str(default_output_type)

    @property
    def runtime(self) -> HunyuanImage3ParallelRuntime:
        return self.pipeline.runtime

    def normalize_request(self, request: Any) -> HunyuanImage3Request:
        if isinstance(request, HunyuanImage3Request):
            return request
        if hasattr(request, "model_dump"):
            request = request.model_dump()
        if not isinstance(request, Mapping):
            raise TypeError(
                "Hunyuan Image 3 requests must be HunyuanImage3Request or a mapping"
            )
        return HunyuanImage3Request(
            prompt=request.get("prompt"),
            request_id=str(request.get("request_id") or uuid.uuid4().hex),
            image_paths=tuple(str(item) for item in request.get("image_paths") or ()),
            height=int(request.get("height") or self.default_height),
            width=int(request.get("width") or self.default_width),
            num_steps=int(request.get("num_steps") or self.default_num_steps),
            guidance_scale=float(
                request.get("guidance_scale") or self.default_guidance_scale
            ),
            seed=int(request.get("seed", 0)),
            system_prompt=(
                self.default_system_prompt
                if request.get("system_prompt") is None
                else str(request["system_prompt"])
            ),
            deadline_ms=request.get("deadline_ms"),
            priority=int(request.get("priority", 0)),
            output_type=str(request.get("output_type") or self.default_output_type),
            extra_inputs=dict(request.get("extra_inputs") or {}),
        )

    def serialize_request(self, request: Any) -> dict[str, Any]:
        normalized = self.normalize_request(request)
        return {
            "request_id": normalized.request_id,
            "prompt": normalized.prompt,
            "image_paths": list(normalized.image_paths),
            "height": normalized.height,
            "width": normalized.width,
            "num_steps": normalized.num_steps,
            "guidance_scale": normalized.guidance_scale,
            "seed": normalized.seed,
            "system_prompt": normalized.system_prompt,
            "deadline_ms": normalized.deadline_ms,
            "priority": normalized.priority,
            "output_type": normalized.output_type,
        }

    def validate_request(
        self,
        request: Any,
        *,
        warmup_resolutions: tuple[tuple[int, int], ...],
    ) -> None:
        normalized = self.normalize_request(request)
        resolved = self.pipeline.resolve_image_size(
            (normalized.height, normalized.width)
        )
        supported = {
            self.pipeline.resolve_image_size(shape) for shape in warmup_resolutions
        }
        if resolved not in supported:
            formatted = ", ".join(
                f"{height}x{width}" for height, width in sorted(supported)
            )
            raise ValueError(
                f"unsupported resolution {resolved[0]}x{resolved[1]}; "
                f"warmup profile supports: {formatted}"
            )

    def _request_conditions(self, request: HunyuanImage3Request) -> int:
        del request
        # The released tokenizer always emits a conditional and an
        # unconditional half for image generation.
        return CFG_CONDITIONS

    def _request_state_bytes(
        self,
        request: HunyuanImage3Request,
        image_tokens: int,
    ) -> int:
        del request
        channels = int(self.pipeline.loaded.config.vae["latent_channels"])
        return image_tokens * channels * torch.bfloat16.itemsize

    def _state_conditions(self, state: HunyuanImage3DenoiseState) -> int:
        return state.conditions

    def _load_condition_images(
        self, request: HunyuanImage3Request
    ) -> tuple[Any, ...]:
        if not request.image_paths:
            return ()
        from PIL import Image

        images = []
        for path in request.image_paths:
            resolved = Path(path).expanduser()
            if not resolved.is_file():
                raise FileNotFoundError(f"conditioning image not found: {resolved}")
            with Image.open(resolved) as handle:
                images.append(handle.convert("RGB"))
        return tuple(images)

    def prepare_request(self, request: Any) -> HunyuanImage3DenoiseState:
        normalized = self.normalize_request(request)
        return self.pipeline.prepare_request(
            request_id=normalized.request_id,
            prompt=normalized.prompt or "",
            image_size=(normalized.height, normalized.width),
            num_steps=normalized.num_steps,
            guidance_scale=normalized.guidance_scale,
            seed=normalized.seed,
            condition_images=self._load_condition_images(normalized),
            system_prompt=normalized.system_prompt,
            output_type=normalized.output_type,
        )

    def export_state(self, state: HunyuanImage3DenoiseState) -> TransferBundle:
        raise NotImplementedError(
            "Hunyuan Image 3 keeps a per-rank key/value cache and a fixed "
            "expert-parallel group, so a request cannot migrate between lanes; "
            "run this stage with policy=static_cp"
        )

    def finalize_gpu(
        self,
        state: HunyuanImage3DenoiseState,
        *,
        lane_ranks: tuple[int, ...],
        timings: dict[str, object],
    ) -> TerminalArtifact | torch.Tensor | None:
        now = time.time_ns()
        timings.setdefault("vae_start_unix_ns", now)
        timings.setdefault("vae_end_unix_ns", now)
        timings.setdefault("vae_decode_ms", 0.0)
        with self.parallel_context.activate(lane_ranks):
            if state.output_type == "latent":
                topology = self.pipeline.vae_topology
                if topology is None or not topology.is_leader:
                    return None
                return TerminalArtifact(
                    tensors={"latents": state.latents},
                    metadata={"output_type": "latent"},
                )
            # A stage-scoped VAE group spans ranks the denoising lane may not
            # hold, which is safe because a static-CP stage runs one request at
            # a time. A lane-scoped placement resolves against the activated
            # completing lane instead.
            return self.pipeline.decode_request(state, timings=timings)

    def postprocess(
        self,
        host_output: torch.Tensor,
        *,
        output_type: str = "pil",
    ) -> Any:
        image = denormalize_image(host_output.to(torch.float32))
        if output_type == "pt":
            return image
        array = image.permute(0, 2, 3, 1).numpy()
        if output_type == "np":
            return array
        if output_type != "pil":
            raise ValueError(f"unsupported output_type: {output_type}")
        from PIL import Image

        return [
            Image.fromarray((sample * 255).round().astype("uint8")) for sample in array
        ]

    def postprocess_artifact(
        self,
        host_artifact: TerminalArtifact,
        *,
        output_type: str | None = None,
    ) -> Any:
        selected = str(
            host_artifact.metadata.get("output_type", output_type or "pil")
        )
        if selected == "latent":
            return host_artifact.tensors["latents"]
        return self.postprocess(host_artifact.tensors["image"], output_type=selected)

    def package_generate_output(self, output: Any) -> Any:
        return output

    def close(self) -> None:
        super().close()
        self.runtime.close()

    @torch.inference_mode()
    def warmup(
        self,
        *,
        resolutions: tuple[tuple[int, int], ...],
        steps: int,
    ) -> dict[str, Any]:
        """Measure real denoising and decode latency for every served shape.

        Hunyuan Image 3 has no synthetic transformer entry point: a step needs
        the tokenized prompt and the populated key/value cache. So warmup runs
        genuine requests, which also pays the first-call allocation and
        autotuning costs before the service accepts traffic.
        """

        if steps < 3:
            raise ValueError("EPE warmup steps must be >= 3")
        context = self.pipeline.parallel_context
        lane_width = self.runtime.plan.plane_width
        device = self.pipeline.device
        rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        for height, width in resolutions:
            state = self.pipeline.prepare_request(
                request_id=f"warmup-{height}x{width}",
                prompt=WARMUP_PROMPT,
                image_size=(height, width),
                num_steps=max(steps, 2),
                guidance_scale=self.default_guidance_scale,
                seed=0,
                output_type="pil",
            )
            step_samples_ms: list[float] = []
            for _ in range(steps):
                if state.complete:
                    break
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_started = time.perf_counter()
                self.pipeline.denoise_step(state, lane_ranks=())
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_samples_ms.append((time.perf_counter() - step_started) * 1000)
            vae_samples_ms: list[float] = []
            d2h_samples_ms: list[float] = []
            for _ in range(3):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                decode_started = time.perf_counter()
                artifact = self.pipeline.decode_request(state)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                vae_samples_ms.append((time.perf_counter() - decode_started) * 1000)
                if artifact is not None:
                    host_started = time.perf_counter()
                    host = artifact.tensors["image"].to("cpu")
                    d2h_samples_ms.append((time.perf_counter() - host_started) * 1000)
                    del host
                del artifact
            local = {
                "rank": context.rank,
                "resolution": [state.height, state.width],
                "image_tokens": state.image_tokens,
                "step_samples_ms": step_samples_ms,
                "vae_samples_ms": vae_samples_ms,
                "d2h_samples_ms": d2h_samples_ms,
                "conditions": state.conditions,
            }
            del state
            gathered: list[Any] = [local]
            if context.world_size > 1:
                gathered = [None] * context.world_size
                dist.all_gather_object(gathered, local)
            if context.rank == 0:
                # A step only completes when the slowest rank does, and only the
                # VAE leader pays the device-to-host copy.
                step_critical = [
                    max(item["step_samples_ms"][index] for item in gathered)
                    for index in range(len(local["step_samples_ms"]))
                ]
                vae_critical = [
                    max(item["vae_samples_ms"][index] for item in gathered)
                    for index in range(len(local["vae_samples_ms"]))
                ]
                d2h = [
                    sample
                    for item in gathered
                    if item["d2h_samples_ms"]
                    for sample in item["d2h_samples_ms"]
                ]
                vae_ms = float(statistics.median(vae_critical))
                d2h_ms = float(statistics.median(d2h)) if d2h else 0.0
                rows.append(
                    {
                        "resolution": local["resolution"],
                        "image_tokens": local["image_tokens"],
                        "width": lane_width,
                        "batch_size": 1,
                        "cfg_conditions": local["conditions"],
                        "latency_ms": float(statistics.median(step_critical)),
                        "samples_ms": step_critical,
                        # Zero keeps the runtime from profiling lane-to-lane
                        # state transfers this backend cannot perform.
                        "state_bytes": 0,
                        "vae_latency_ms": vae_ms,
                        "d2h_latency_ms": d2h_ms,
                        "terminal_ms": vae_ms + d2h_ms,
                    }
                )
        report: list[Any] = [
            {
                "source": "startup_warmup",
                "model": self.model_name,
                "steps": steps,
                "resolutions": [list(value) for value in resolutions],
                "topology": self.runtime.plan.describe(),
                "attention_mode": self.runtime.plan.attention_mode,
                "tensor_parallel_degree": self.runtime.plan.tensor_parallel_degree,
                "cfg_parallel_degree": self.runtime.plan.cfg_parallel_degree,
                "context_parallel_degree": (
                    self.runtime.plan.context_parallel_degree
                ),
                "plane_width": lane_width,
                "expert_parallel_degree": self.runtime.plan.expert_parallel_degree,
                "vae_parallel_degree": self.vae_placement.degree,
                "parallel_vae": self.parallel_vae,
                "rows": rows,
                "total_wall_ms": (time.perf_counter() - started) * 1000,
            }
            if context.rank == 0
            else None
        ]
        if context.world_size > 1:
            dist.broadcast_object_list(report, src=0)
        assert report[0] is not None
        self.scheduling_module.initialize_cost_model(report[0]["rows"])
        return report[0]


@dataclass(frozen=True)
class HunyuanImage3ExecutorFactory:
    """Build a Hunyuan Image 3 stage from the configured parallel degrees."""

    model_path: str
    default_width: int = 1024
    default_height: int = 1024
    default_num_steps: int = 50
    default_guidance_scale: float = 5.0
    default_system_prompt: str | None = None
    default_output_type: str = "pil"
    flow_shift: float | None = None

    def resolve_plan(self, context: ExecutorBuildContext) -> HunyuanImage3ParallelPlan:
        """Derive the plan the stage world and configuration imply."""

        model = context.model
        if model.cfg_parallel_degree > CFG_CONDITIONS:
            raise ValueError(
                "Hunyuan Image 3 carries exactly one conditional and one "
                "unconditional branch, so cfg_parallel_degree must be 1 or 2"
            )
        return resolve_parallel_plan(
            world_size=int(context.world.world_size),
            tensor_parallel_degree=int(model.tensor_parallel_degree),
            cfg_parallel_degree=int(model.cfg_parallel_degree),
            expert_parallel_degree=model.expert_parallel_degree,
            attention_mode=context.cp.attention_mode,
        )

    def build(self, context: ExecutorBuildContext) -> HunyuanImage3Executor:
        pool = context.pool
        plan = self.resolve_plan(context)
        if pool.policy != "static_cp":
            raise ValueError(
                "Hunyuan Image 3 requires policy=static_cp; elastic lane resize "
                "would have to move a lane-resident key/value cache"
            )
        if max(pool.allowed_lane_widths) != plan.plane_width:
            raise ValueError(
                "Hunyuan Image 3 requires the widest allowed lane to equal the "
                f"TP-plane width {plan.plane_width}"
            )
        # Attention, expert dispatch, and the plane itself each collect over a
        # different contiguous slice, and the pool decides which slices get a
        # process group.
        missing = sorted(set(plan.lane_widths) - set(pool.allowed_lane_widths))
        if missing:
            raise ValueError(
                f"{plan.describe()} needs allowed_lane_widths to include "
                f"{missing}, got {list(pool.allowed_lane_widths)}"
            )
        # The plan's attention mode picks the DiT exchange; the EPE lane itself
        # is always laid out as a Ulysses group of the context-parallel degree.
        parallel, local_rank = build_stage_parallel_context(
            context,
            attention_mode="ulysses",
            ulysses_degree=plan.context_parallel_degree,
        )
        runtime = HunyuanImage3ParallelRuntime(
            plan=plan,
            context=parallel,
            num_experts=HunyuanImage3ParallelRuntime.checkpoint_num_experts(
                self.model_path
            ),
            owns_context=False,
            vae_placement=build_stage_vae_placement(context),
        )
        loaded = load_hunyuan_image3(
            self.model_path,
            runtime=runtime,
            device=(
                torch.device("cuda", local_rank)
                if torch.cuda.is_available()
                else torch.device("cpu")
            ),
        )
        pipeline = HunyuanImage3Pipeline(
            loaded,
            runtime,
            flow_shift=self.flow_shift,
        )
        scheduling = EpeSchedulingModule(
            parallel,
            **scheduling_options_from_pool(pool),
        )
        return HunyuanImage3Executor(
            pipeline,
            scheduling,
            default_width=self.default_width,
            default_height=self.default_height,
            default_num_steps=self.default_num_steps,
            default_guidance_scale=self.default_guidance_scale,
            default_system_prompt=self.default_system_prompt,
            default_output_type=self.default_output_type,
        )


__all__ = [
    "WARMUP_PROMPT",
    "HunyuanImage3Executor",
    "HunyuanImage3ExecutorFactory",
    "denormalize_image",
]
