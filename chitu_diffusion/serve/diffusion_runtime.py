"""Compatibility alias for the canonical EPE service runtime."""

from .runtime import DiffusionServiceRuntime, _RequestRecord

EpeDiffusionServiceRuntime = DiffusionServiceRuntime

__all__ = [
    "EpeDiffusionServiceRuntime",
    "_RequestRecord",
]
