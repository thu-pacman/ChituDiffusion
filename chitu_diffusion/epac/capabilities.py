from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PipelineCapabilities:
    """Features an adapter can provide without silent fallback."""

    step_execution: bool = True
    split_model_step: bool = True
    request_batching: bool = False
    context_parallel: bool = False
    dynamic_topology: bool = False
    epe: bool = False
    flexcache: bool = False
    running_abort: bool = True
