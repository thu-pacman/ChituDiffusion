from __future__ import annotations

import argparse
import os

from chitu_diffusion.parallel import AGKV_TRANSPORTS, ULYSSES_TRANSPORTS


def add_parallel_transport_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ulysses-transport",
        choices=ULYSSES_TRANSPORTS,
        default=None,
        help=(
            "USP all-to-all transport. Defaults to CHITU_ULYSSES_TRANSPORT or "
            "torch; fast_ulysses requires full-world, full-Ulysses static CP."
        ),
    )
    parser.add_argument(
        "--agkv-transport",
        choices=AGKV_TRANSPORTS,
        default=None,
        help=(
            "AGKV K/V gather transport. Defaults to CHITU_AGKV_TRANSPORT or "
            "torch; fast_agkv requires static full-world single-node CP."
        ),
    )


def static_parallel_pipeline_kwargs(args: argparse.Namespace) -> dict[str, object]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return {
        "allowed_lane_widths": tuple(sorted({1, world_size})),
        "ulysses_transport": args.ulysses_transport,
        "agkv_transport": args.agkv_transport,
    }
