from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from ...epac.adapters import DiffusersModelAdapter
from ...epac.capabilities import PipelineCapabilities
from ...epac.request import DiffusionRequest
from ...epac.scheduling import RequestProfile, StepPlan
from ...parallel import CfgParallelTopology
from .pipeline import combine_qwen_cfg_predictions


@dataclass(slots=True)
class _QwenImageStepInputs:
    timestep_value: torch.Tensor
    timestep: torch.Tensor
    cfg_topology: CfgParallelTopology | None


class QwenImageModelAdapter(DiffusersModelAdapter):
    """Resumable Diffusers adapter for Qwen-Image with CFP2."""

    name = "qwen-image"
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
            "QwenImagePipeline" in names
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
        if state.complete:
            raise RuntimeError("denoise state is already complete")
        timestep_value = state.timesteps[state.step_index]
        timestep = timestep_value.expand(state.latents.shape[0]).to(state.latents.dtype)
        parallel = getattr(pipeline, "parallel_context", None)
        lane_ranks = plan.lane_ranks
        if parallel is not None and lane_ranks is None:
            lane_ranks = tuple(range(parallel.world_size))
        cfg_topology = (
            parallel.cfg_parallel_topology(lane_ranks)
            if state.do_true_cfg
            and pipeline.cfg_parallel
            and parallel is not None
            and lane_ranks is not None
            else None
        )
        return _QwenImageStepInputs(
            timestep_value=timestep_value,
            timestep=timestep,
            cfg_topology=cfg_topology,
        )

    @staticmethod
    def _forward_branch(
        pipeline: Any,
        state: Any,
        model_inputs: _QwenImageStepInputs,
        *,
        negative: bool,
    ) -> torch.Tensor:
        embeds = state.negative_prompt_embeds if negative else state.prompt_embeds
        mask = (
            state.negative_prompt_embeds_mask if negative else state.prompt_embeds_mask
        )
        if embeds is None:
            raise RuntimeError("Qwen-Image CFG branch has no prompt embeddings")
        cache_name = "uncond" if negative else "cond"
        with pipeline.transformer.cache_context(cache_name):
            return pipeline.transformer(
                hidden_states=state.latents,
                timestep=model_inputs.timestep / 1000,
                guidance=state.guidance,
                encoder_hidden_states=embeds,
                encoder_hidden_states_mask=mask,
                img_shapes=state.img_shapes,
                attention_kwargs=None,
                return_dict=False,
            )[0]

    @torch.inference_mode()
    def model_forward(
        self,
        pipeline: Any,
        state: Any,
        model_inputs: _QwenImageStepInputs,
        plan: StepPlan,
    ) -> Any:
        parallel = pipeline.parallel_context
        with self._lane_context(pipeline, plan.lane_ranks):
            if model_inputs.cfg_topology is not None:
                topology = model_inputs.cfg_topology
                with parallel.activate(topology.cp_ranks):
                    branch = self._forward_branch(
                        pipeline,
                        state,
                        model_inputs,
                        negative=topology.branch_index == 1,
                    )
                return parallel.all_gather_cfg(branch, topology)

            positive = self._forward_branch(
                pipeline, state, model_inputs, negative=False
            )
            if not state.do_true_cfg:
                return positive
            negative = self._forward_branch(
                pipeline, state, model_inputs, negative=True
            )
            return positive, negative

    def process_model_output(
        self,
        pipeline: Any,
        state: Any,
        model_inputs: _QwenImageStepInputs,
        model_output: Any,
        plan: StepPlan,
    ) -> torch.Tensor:
        del pipeline, model_inputs, plan
        if not state.do_true_cfg:
            return model_output
        positive, negative = model_output
        return combine_qwen_cfg_predictions(
            positive,
            negative,
            true_cfg_scale=state.true_cfg_scale,
        )

    @torch.inference_mode()
    def scheduler_step(
        self,
        pipeline: Any,
        state: Any,
        prediction: torch.Tensor,
        plan: StepPlan,
    ) -> None:
        del pipeline, plan
        latent_dtype = state.latents.dtype
        state.latents = state.scheduler.step(
            prediction,
            state.timesteps[state.step_index],
            state.latents,
            return_dict=False,
        )[0]
        if state.latents.dtype != latent_dtype:
            state.latents = state.latents.to(latent_dtype)
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
                "conditions": 2 if state.do_true_cfg else 1,
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
