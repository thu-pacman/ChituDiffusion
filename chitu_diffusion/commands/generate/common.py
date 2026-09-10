from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch


def generation_timestamp() -> float:
    """Finish outstanding CUDA work before measuring generation boundaries."""
    if torch.cuda.is_available():
        torch.cuda.synchronize(int(os.environ.get("LOCAL_RANK", "0")))
    return time.perf_counter()


def write_generation_metadata(
    output: str | Path,
    *,
    args: argparse.Namespace,
    pipeline: Any,
    elapsed_s: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write generation-only timing and reproducibility fields on the leader."""
    output = Path(output)
    values = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    values.update(
        implementation="epe",
        world_size=pipeline.parallel_context.world_size,
        cache=getattr(pipeline, "last_cache_stats", None),
    )
    values.update(metadata or {})
    vae_stats = getattr(pipeline, "last_vae_stats", None)
    if vae_stats is not None:
        values["requested_parallel_vae"] = values.get("parallel_vae")
        values["requested_vae_parallel_halo"] = values.get("vae_parallel_halo")
        values.update(vae_stats)
    values.update(elapsed_s=elapsed_s, generate_seconds=elapsed_s)
    output.with_suffix(".json").write_text(
        json.dumps(values, indent=2) + "\n", encoding="utf-8"
    )
    print(f"elapsed_s={elapsed_s:.3f}")
    return values
