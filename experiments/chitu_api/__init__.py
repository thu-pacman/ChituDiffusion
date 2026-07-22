"""Compatibility entry point for the promoted chitu_diffusers service."""

from chitu_diffusers.serve import StageServiceConfig, load_stage_service_config

__all__ = ["StageServiceConfig", "load_stage_service_config"]
