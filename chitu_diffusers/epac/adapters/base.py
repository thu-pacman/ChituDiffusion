from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..capabilities import PipelineCapabilities
from ..request import DiffusionRequest
from ..scheduling import RequestProfile, StepPlan


class DiffusersModelAdapter(ABC):
    """Extracts a resumable denoise step from one Diffusers model family."""

    name: str
    capabilities: PipelineCapabilities

    @classmethod
    @abstractmethod
    def supports(cls, pipeline: Any) -> bool: ...

    @abstractmethod
    def prepare_request(self, pipeline: Any, request: DiffusionRequest) -> Any: ...

    @abstractmethod
    def prepare_step(self, pipeline: Any, state: Any, plan: StepPlan) -> Any: ...

    @abstractmethod
    def model_forward(
        self,
        pipeline: Any,
        state: Any,
        model_inputs: Any,
        plan: StepPlan,
    ) -> Any: ...

    @abstractmethod
    def process_model_output(
        self,
        pipeline: Any,
        state: Any,
        model_inputs: Any,
        model_output: Any,
        plan: StepPlan,
    ) -> Any: ...

    @abstractmethod
    def scheduler_step(
        self,
        pipeline: Any,
        state: Any,
        prediction: Any,
        plan: StepPlan,
    ) -> None: ...

    @abstractmethod
    def is_complete(self, state: Any) -> bool: ...

    @abstractmethod
    def profile(self, state: Any) -> RequestProfile: ...

    @abstractmethod
    def finalize_request(
        self,
        pipeline: Any,
        state: Any,
        request: DiffusionRequest,
    ) -> Any: ...

    def abort_request(self, pipeline: Any, state: Any) -> None:
        del pipeline, state

    def model_cache_input(
        self, pipeline: Any, state: Any, model_inputs: Any
    ) -> Any | None:
        del pipeline, state, model_inputs
        return None

    def model_cache_signal(
        self, pipeline: Any, state: Any, model_inputs: Any
    ) -> Any | None:
        del pipeline, state, model_inputs
        return None
