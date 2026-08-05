from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence

from .request import DiffusionRequest
from .scheduling import StepPlan

if TYPE_CHECKING:
    from .adapters import DiffusersModelAdapter

ModelCall = Callable[[], Any]


@dataclass(frozen=True, slots=True)
class ModelStepContext:
    """Stable boundary shared by FlexCache-style model optimizations."""

    request: DiffusionRequest
    pipeline: Any
    adapter: DiffusersModelAdapter
    state: Any
    model_inputs: Any
    plan: StepPlan


@dataclass(frozen=True, slots=True)
class CachedPrediction:
    """Signals that a prediction cache skipped the complete DiT forward."""

    value: Any


class DenoiseOptimization(Protocol):
    """Request-local hooks at the DiT and post-CFG prediction boundaries."""

    def execute_model(self, context: ModelStepContext, call_next: ModelCall) -> Any: ...

    def process_prediction(self, context: ModelStepContext, prediction: Any) -> Any: ...

    def after_step(self, context: ModelStepContext, prediction: Any) -> None: ...

    def on_request_end(self, request_id: str) -> None: ...


class OptimizationChain:
    def __init__(self, optimizations: Sequence[DenoiseOptimization] = ()) -> None:
        self._optimizations = tuple(optimizations)

    def execute_model(self, context: ModelStepContext, model_call: ModelCall) -> Any:
        call = model_call
        for optimization in reversed(self._optimizations):
            next_call = call

            def call(
                optimization: DenoiseOptimization = optimization,
                next_call: ModelCall = next_call,
            ) -> Any:
                return optimization.execute_model(context, next_call)

        return call()

    def process_prediction(self, context: ModelStepContext, prediction: Any) -> Any:
        for optimization in self._optimizations:
            prediction = optimization.process_prediction(context, prediction)
        return prediction

    def after_step(self, context: ModelStepContext, prediction: Any) -> None:
        for optimization in self._optimizations:
            optimization.after_step(context, prediction)

    def on_request_end(self, request_id: str) -> None:
        for optimization in reversed(self._optimizations):
            optimization.on_request_end(request_id)
