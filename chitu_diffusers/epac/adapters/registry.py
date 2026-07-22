from __future__ import annotations

from typing import Any, Iterable

from .base import DiffusersModelAdapter


class AdapterRegistry:
    def __init__(
        self, adapter_types: Iterable[type[DiffusersModelAdapter]] = ()
    ) -> None:
        self._adapter_types = list(adapter_types)

    def register(self, adapter_type: type[DiffusersModelAdapter]) -> None:
        if adapter_type not in self._adapter_types:
            self._adapter_types.append(adapter_type)

    def resolve(self, pipeline: Any) -> DiffusersModelAdapter:
        matches = [
            adapter_type
            for adapter_type in self._adapter_types
            if adapter_type.supports(pipeline)
        ]
        if not matches:
            pipeline_type = type(pipeline)
            raise LookupError(
                "no chitu_diffusers adapter registered for "
                f"{pipeline_type.__module__}.{pipeline_type.__qualname__}"
            )
        if len(matches) > 1:
            names = ", ".join(adapter_type.__name__ for adapter_type in matches)
            raise LookupError(f"multiple adapters match pipeline: {names}")
        return matches[0]()
