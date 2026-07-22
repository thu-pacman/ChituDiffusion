from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from ...epac.adapters import DiffusersModelAdapter
from ...epac.capabilities import PipelineCapabilities
from ...epac.request import DiffusionRequest
from ...epac.scheduling import RequestProfile, StepPlan


@dataclass(slots=True)
class _ZImageStepInputs:
    timestep_value: torch.Tensor
    latent_model_input: torch.Tensor
    timestep_model_input: torch.Tensor
    prompt_model_input: list[torch.Tensor]
    apply_cfg: bool
    current_guidance_scale: float


class ZImageModelAdapter(DiffusersModelAdapter):
    """Step adapter for the current EPE-enabled Diffusers Z-Image prototype."""

    name = "z-image"
    capabilities = PipelineCapabilities(
        context_parallel=True,
        dynamic_topology=True,
        epe=True,
        flexcache=False,
    )

    @classmethod
    def supports(cls, pipeline: Any) -> bool:
        names = {base.__name__ for base in type(pipeline).__mro__}
        return (
            "ZImagePipeline" in names
            and callable(getattr(pipeline, "prepare_request", None))
            and callable(getattr(pipeline, "finalize_request", None))
            and hasattr(pipeline, "transformer")
        )

    def prepare_request(self, pipeline: Any, request: DiffusionRequest) -> Any:
        kwargs = dict(request.inputs)
        supplied_steps = kwargs.pop("num_inference_steps", None)
        if supplied_steps is not None and supplied_steps != request.num_inference_steps:
            raise ValueError(
                "num_inference_steps must be set on DiffusionRequest, not inputs"
            )
        kwargs["num_inference_steps"] = request.num_inference_steps
        return pipeline.prepare_request(**kwargs)

    def prepare_step(self, pipeline: Any, state: Any, plan: StepPlan) -> Any:
        del pipeline, plan
        if state.complete:
            raise RuntimeError("denoise state is already complete")

        timestep_value = state.timesteps[state.step_index]
        timestep = timestep_value.expand(state.latents.shape[0])
        timestep = (1000 - timestep) / 1000
        current_guidance_scale = state.guidance_scale
        if state.guidance_scale > 0 and state.cfg_truncation <= 1:
            if timestep[0].item() > state.cfg_truncation:
                current_guidance_scale = 0.0
        apply_cfg = state.guidance_scale > 0 and current_guidance_scale > 0

        latent_model_input = state.latents
        prompt_model_input = state.prompt_embeds
        timestep_model_input = timestep
        if apply_cfg:
            if state.negative_prompt_embeds is None:
                raise RuntimeError("CFG state has no negative prompt embeddings")
            latent_model_input = latent_model_input.repeat(2, 1, 1, 1)
            prompt_model_input = state.prompt_embeds + state.negative_prompt_embeds
            timestep_model_input = timestep.repeat(2)

        return _ZImageStepInputs(
            timestep_value=timestep_value,
            latent_model_input=latent_model_input,
            timestep_model_input=timestep_model_input,
            prompt_model_input=prompt_model_input,
            apply_cfg=apply_cfg,
            current_guidance_scale=current_guidance_scale,
        )

    def model_forward(
        self,
        pipeline: Any,
        state: Any,
        model_inputs: _ZImageStepInputs,
        plan: StepPlan,
    ) -> Any:
        del state
        lane_context = self._lane_context(pipeline, plan.lane_ranks)
        with lane_context:
            return pipeline.transformer(
                list(
                    model_inputs.latent_model_input.to(pipeline.transformer.dtype)
                    .unsqueeze(2)
                    .unbind(dim=0)
                ),
                model_inputs.timestep_model_input,
                model_inputs.prompt_model_input,
                return_dict=False,
            )[0]

    def process_model_output(
        self,
        pipeline: Any,
        state: Any,
        model_inputs: _ZImageStepInputs,
        model_output: Any,
        plan: StepPlan,
    ) -> torch.Tensor:
        del pipeline, plan
        if model_inputs.apply_cfg:
            positive = model_output[: state.actual_batch_size]
            negative = model_output[state.actual_batch_size :]
            predictions = []
            for positive_value, negative_value in zip(positive, negative):
                positive_value = positive_value.float()
                negative_value = negative_value.float()
                prediction = positive_value + model_inputs.current_guidance_scale * (
                    positive_value - negative_value
                )
                if state.cfg_normalization:
                    original_norm = torch.linalg.vector_norm(positive_value)
                    prediction_norm = torch.linalg.vector_norm(prediction)
                    max_norm = original_norm * float(state.cfg_normalization)
                    if prediction_norm > max_norm:
                        prediction = prediction * (max_norm / prediction_norm)
                predictions.append(prediction)
            noise_pred = torch.stack(predictions)
        else:
            noise_pred = torch.stack([value.float() for value in model_output])

        return -noise_pred.squeeze(2).to(torch.float32)

    def scheduler_step(
        self,
        pipeline: Any,
        state: Any,
        prediction: torch.Tensor,
        plan: StepPlan,
    ) -> None:
        del pipeline, plan
        timestep_value = state.timesteps[state.step_index]
        state.latents = state.scheduler.step(
            prediction,
            timestep_value,
            state.latents,
            return_dict=False,
        )[0]
        state.step_index += 1

    def model_cache_input(
        self, pipeline: Any, state: Any, model_inputs: _ZImageStepInputs
    ) -> list[torch.Tensor]:
        del state
        return list(
            model_inputs.latent_model_input.to(pipeline.transformer.dtype)
            .unsqueeze(2)
            .unbind(dim=0)
        )

    def model_cache_signal(
        self, pipeline: Any, state: Any, model_inputs: _ZImageStepInputs
    ) -> torch.Tensor:
        del state
        return pipeline.transformer.t_embedder(
            model_inputs.timestep_model_input * pipeline.transformer.t_scale
        ).type_as(model_inputs.latent_model_input)

    def is_complete(self, state: Any) -> bool:
        return bool(state.complete)

    def profile(self, state: Any) -> RequestProfile:
        return RequestProfile(
            total_steps=len(state.timesteps),
            completed_steps=state.step_index,
            shape_key=(int(state.height), int(state.width)),
            image_tokens=int(state.image_tokens),
            attributes={
                "batch_size": int(state.actual_batch_size),
                "conditions": 2 if state.guidance_scale > 0 else 1,
            },
        )

    def finalize_request(
        self,
        pipeline: Any,
        state: Any,
        request: DiffusionRequest,
    ) -> Any:
        return pipeline.finalize_request(state, output_type=request.output_type)

    @staticmethod
    def _lane_context(pipeline: Any, lane_ranks: tuple[int, ...] | None):
        parallel = getattr(pipeline, "parallel_context", None)
        if parallel is None:
            if lane_ranks not in (None, (0,)):
                raise ValueError("pipeline has no parallel context for lane_ranks")
            return nullcontext()
        if lane_ranks is None:
            lane_ranks = tuple(range(parallel.world_size))
        return parallel.activate(lane_ranks)
