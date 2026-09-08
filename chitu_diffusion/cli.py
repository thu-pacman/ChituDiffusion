from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

_GENERATE_MODULES = {
    "llada-image": "chitu_diffusion.commands.generate.llada_image",
    "zimage": "chitu_diffusion.commands.generate.zimage",
    "flux1": "chitu_diffusion.commands.generate.flux1",
    "flux2-klein": "chitu_diffusion.commands.generate.flux2_klein",
    "qwen-image": "chitu_diffusion.commands.generate.qwen_image",
    "wan": "chitu_diffusion.commands.generate.wan",
    "hunyuan-image3": "chitu_diffusion.commands.generate.hunyuan_image3",
}


def _run_module(module_name: str, arguments: Sequence[str]) -> None:
    module = importlib.import_module(module_name)
    previous = sys.argv
    try:
        sys.argv = [module_name, *arguments]
        module.main()
    finally:
        sys.argv = previous


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chitu",
        description="Diffusers-native Chitu generation and serving.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="Run one full-world context-parallel request.",
        add_help=False,
    )
    generate.add_argument("--model", choices=tuple(_GENERATE_MODULES), required=True)

    subparsers.add_parser(
        "serve",
        help="Run the persistent EPE service from a native stage config.",
        add_help=False,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    namespace, remaining = parser.parse_known_args(arguments)
    if namespace.command == "generate":
        _run_module(_GENERATE_MODULES[namespace.model], remaining)
    elif namespace.command == "serve":
        _run_module("chitu_diffusion.commands.serve", remaining)
    else:  # pragma: no cover - argparse enforces the command choices.
        parser.error(f"unsupported command: {namespace.command}")


if __name__ == "__main__":
    main()
