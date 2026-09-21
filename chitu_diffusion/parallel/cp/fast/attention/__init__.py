"""Shared CP attention preparation, independent of the communication algorithm."""

from .planner import AttentionPlan, prepare_attention

__all__ = ["AttentionPlan", "prepare_attention"]
