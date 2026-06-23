from __future__ import annotations

import os
from logging import getLogger
from typing import Any

import torch

from chitu_diffusion.models.parallel import ModelParallelCapabilities
from chitu_diffusion.parallel.state import (
    DitParallelPlan,
    ParallelPlan,
    SamplerParallelPlan,
    VaeParallelPlan,
)

logger = getLogger(__name__)


class ParallelPlanner:
    """Build resolved parallel plans from config, model capabilities, and device state."""

    SUPPORTED_CP_BACKENDS = {"auto", "agcp", "ucp"}

    def __init__(self, args: Any, model_adapter: Any = None):
        self.args = args
        self.model_adapter = model_adapter

    def build(self) -> ParallelPlan:
        diffusion = self.args.infer.diffusion
        capabilities = self._capabilities()
        cp_size = int(getattr(diffusion, "cp_size", 1))
        cfg_size = int(getattr(diffusion, "cfg_size", 1))
        up = int(getattr(diffusion, "up", 1))
        effective_cp_size = cp_size if capabilities.dit_context_parallel else 1
        effective_cfg_size = cfg_size if capabilities.dit_cfg_parallel else 1
        if cp_size > 1 and not capabilities.dit_context_parallel:
            logger.warning(
                "Model adapter %s does not support DiT context parallelism; disabling CP in parallel plan.",
                type(self.model_adapter).__name__ if self.model_adapter is not None else "<none>",
            )
        requested_cp_backend = str(getattr(diffusion, "cp_backend", "auto") or "auto").strip().lower()
        resolved_cp_backend = self.resolve_cp_backend(
            requested_cp_backend,
            cp_size=effective_cp_size,
            up=up,
        )
        return ParallelPlan(
            dit=DitParallelPlan(
                cfg_size=effective_cfg_size,
                cp_size=effective_cp_size,
                up=up,
                cp_backend=resolved_cp_backend if effective_cp_size > 1 else "none",
            ),
            vae=VaeParallelPlan(
                mode="tile",
                enabled=capabilities.vae_tile_parallel and os.getenv("CHITU_VAE_PARALLEL_DECODE", "1") == "1",
            ),
            sampler=SamplerParallelPlan(
                mode="local_latent" if capabilities.sampler_local_latent else "replicated",
                enabled=capabilities.sampler_local_latent and effective_cp_size > 1,
            ),
        )

    def _capabilities(self) -> ModelParallelCapabilities:
        if self.model_adapter is None:
            return ModelParallelCapabilities()
        provider = getattr(self.model_adapter, "parallel_capabilities", None)
        if provider is None:
            return ModelParallelCapabilities()
        return provider(self.args)

    def resolve_cp_backend(self, requested: str, *, cp_size: int, up: int) -> str:
        mode = str(requested or "auto").strip().lower()
        if mode not in self.SUPPORTED_CP_BACKENDS:
            raise ValueError(f"infer.diffusion.cp_backend must be one of auto, agcp, ucp; got {requested!r}.")
        if cp_size <= 1:
            return "none"
        if mode != "auto":
            return mode
        if not torch.cuda.is_available():
            return "ucp"
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
        except Exception:
            return "ucp"
        free_ratio = free_mem / max(total_mem, 1)
        # In high-bandwidth, memory-rich runs AGCP often wins: one collective plus
        # one attention kernel avoids chunk-kernel and LSE-merge overhead.
        return "agcp" if cp_size <= 4 and free_ratio >= 0.20 else "ucp"


def build_parallel_plan(args: Any, model_adapter: Any = None) -> ParallelPlan:
    return ParallelPlanner(args, model_adapter=model_adapter).build()
