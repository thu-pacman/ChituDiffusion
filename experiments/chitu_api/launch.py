from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from chitu_diffusion.serve import load_stage_service_config


def build_launch_command(config_path: str | Path) -> tuple[list[str], dict[str, str]]:
    path = Path(config_path).expanduser().resolve()
    config = load_stage_service_config(path)
    checkpoint = Path(config.factory_args.model_path).expanduser()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Z-Image checkpoint directory not found: {checkpoint}")

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={len(config.gpu)}",
        "--module",
        "experiments.chitu_api.serve",
        "--stage-config",
        str(path),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in config.gpu)
    return command, env


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the experimental Chitu API stage")
    parser.add_argument("--stage-config", required=True)
    args = parser.parse_args()
    command, env = build_launch_command(args.stage_config)
    config = load_stage_service_config(args.stage_config)
    print(
        f"Launching {config.name} on physical GPUs {list(config.gpu)}; "
        f"configured endpoint: {config.service.endpoint}",
        flush=True,
    )
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
