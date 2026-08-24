#!/usr/bin/env python3
"""Launch SGLang diffusion with Chitu's MiniMax-H3 DiT registry adapter."""

from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_lightweight_package(name: str, relative_path: str) -> None:
    """Avoid importing Chitu's unrelated serving stacks in SGLang workers."""
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT / relative_path)]
    module.__package__ = name
    sys.modules[name] = module


# The baseline SGLang environment intentionally contains only dependencies
# needed by H3.  Namespace the relevant Chitu packages so importing this
# adapter does not execute chitu_diffusion/__init__.py or models/__init__.py,
# both of which eagerly import unrelated Diffusers integrations.
_install_lightweight_package("chitu_diffusion", "chitu_diffusion")
_install_lightweight_package("chitu_diffusion.models", "chitu_diffusion/models")
_install_lightweight_package(
    "chitu_diffusion.models.minimax_h3",
    "chitu_diffusion/models/minimax_h3",
)
_install_lightweight_package("chitu_diffusion.parallel", "chitu_diffusion/parallel")

from chitu_diffusion.integrations.sglang.bootstrap import (  # noqa: E402
    install_sglang_h3_adapter,
    validate_launcher_args,
)


# multiprocessing "spawn" imports the main module in each worker.  Installing
# at module import time ensures every worker overrides its process-local
# registry before transformer resolution.
install_sglang_h3_adapter()


def main() -> int:
    args = list(sys.argv[1:])
    if not args or args[0] != "serve":
        args.insert(0, "serve")
    validate_launcher_args(args)
    if "--no-use-fsdp-inference" not in args and "--use-fsdp-inference" not in args:
        args.append("--no-use-fsdp-inference")
    if (
        "--no-enable-breakable-cuda-graph" not in args
        and "--enable-breakable-cuda-graph" not in args
    ):
        args.append("--no-enable-breakable-cuda-graph")
    sys.argv = ["sglang", *args]

    from sglang.cli.main import main as sglang_main

    result = sglang_main()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())

