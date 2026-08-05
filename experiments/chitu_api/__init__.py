"""Compatibility entry point for the promoted chitu_diffusion service."""

from chitu_diffusion.serve import StageServiceConfig, load_stage_service_config

__all__ = ["StageServiceConfig", "load_stage_service_config"]
