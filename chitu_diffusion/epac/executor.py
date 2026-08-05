from __future__ import annotations

from typing import Any

from .adapters import DiffusersModelAdapter
from .optimization import CachedPrediction, ModelStepContext, OptimizationChain
from .request import DiffusionRequest
from .scheduling import StepPlan


class DenoiseStepExecutor:
    """Runs one adapter step and exposes stable cache boundaries."""

    def __init__(
        self,
        pipeline: Any,
        adapter: DiffusersModelAdapter,
        optimizations: OptimizationChain | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.adapter = adapter
        self.optimizations = optimizations or OptimizationChain()

    def execute(
        self,
        request: DiffusionRequest,
        state: Any,
        plan: StepPlan,
    ) -> None:
        model_inputs = self.adapter.prepare_step(self.pipeline, state, plan)
        context = ModelStepContext(
            request=request,
            pipeline=self.pipeline,
            adapter=self.adapter,
            state=state,
            model_inputs=model_inputs,
            plan=plan,
        )
        model_result = self.optimizations.execute_model(
            context,
            lambda: self.adapter.model_forward(
                self.pipeline,
                state,
                model_inputs,
                plan,
            ),
        )
        if isinstance(model_result, CachedPrediction):
            prediction = model_result.value
        else:
            prediction = self.adapter.process_model_output(
                self.pipeline,
                state,
                model_inputs,
                model_result,
                plan,
            )
        prediction = self.optimizations.process_prediction(context, prediction)
        self.adapter.scheduler_step(
            self.pipeline,
            state,
            prediction,
            plan,
        )
        self.optimizations.after_step(context, prediction)

    def on_request_end(self, request_id: str) -> None:
        self.optimizations.on_request_end(request_id)
