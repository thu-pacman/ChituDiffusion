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
class _Flux1StepInputs:
    timestep_value: torch.Tensor
    timestep: torch.Tensor


class Flux1ModelAdapter(DiffusersModelAdapter):
    """Resumable Diffusers adapter for FLUX.1-dev."""

    name = "flux1-dev"
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
            "FluxPipeline" in names
            and callable(getattr(pipeline, "prepare_request", None))
            and callable(getattr(pipeline, "finalize_request", None))
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
        timestep = timestep_value.expand(state.latents.shape[0]).to(state.latents.dtype)
        return _Flux1StepInputs(timestep_value=timestep_value, timestep=timestep)

    @torch.inference_mode()
    def model_forward(
        self,
        pipeline: Any,
        state: Any,
        model_inputs: _Flux1StepInputs,
        plan: StepPlan,
    ) -> Any:
        with self._lane_context(pipeline, plan.lane_ranks):
            return pipeline.transformer(
                hidden_states=state.latents,
                timestep=model_inputs.timestep / 1000,
                guidance=state.guidance,
                pooled_projections=state.pooled_prompt_embeds,
                encoder_hidden_states=state.prompt_embeds,
                txt_ids=state.text_ids,
                img_ids=state.latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

    def process_model_output(
        self,
        pipeline: Any,
        state: Any,
        model_inputs: _Flux1StepInputs,
        model_output: Any,
        plan: StepPlan,
    ) -> torch.Tensor:
        del pipeline, state, model_inputs, plan
        return model_output

    @torch.inference_mode()
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
                "conditions": 1,
                "state_bytes": state.latents.numel() * state.latents.element_size(),
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
