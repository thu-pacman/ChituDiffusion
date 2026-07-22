"""Entrypoint for torchrun and the internal ``chitu run`` compatibility path."""

import os
import sys

from chitu_diffusers.serve.torchrun import main, run_config


def _run_chitu_cli() -> None:
    from chitu_diffusion.core.config_loader import load_config

    from chitu_diffusers.serve import StageServiceConfig

    args = load_config(sys.argv[1:])
    config = StageServiceConfig.from_chitu_runtime(
        args,
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )
    run_config(config)


if __name__ == "__main__":
    if "--stage-config" in sys.argv[1:]:
        main()
    else:
        _run_chitu_cli()
