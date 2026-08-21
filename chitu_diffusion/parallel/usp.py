"""Compatibility facade for Ulysses×Ring attention."""

from .fast_cp.ulysses_ring import FastUlyssesRingAttention

# Public compatibility name used by model-independent attention wrappers.
DynamicUspAttention = FastUlyssesRingAttention

__all__ = ["DynamicUspAttention", "FastUlyssesRingAttention"]
