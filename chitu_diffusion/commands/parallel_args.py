from __future__ import annotations

import argparse
import os

from chitu_diffusion.parallel.cp import AGKV_TRANSPORTS, ULYSSES_TRANSPORTS


def add_parallel_transport_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ulysses-transport",
        choices=ULYSSES_TRANSPORTS,
        default=None,
        help=(
            "Ulysses all-to-all transport. Defaults to CHITU_ULYSSES_TRANSPORT or "
            "torch; fast_ulysses requires a single-node static CP lane."
        ),
    )
    parser.add_argument(
        "--agkv-transport",
        choices=AGKV_TRANSPORTS,
        default=None,
        help=(
            "AGKV K/V gather transport. Defaults to CHITU_AGKV_TRANSPORT or "
            "torch; fast_agkv requires a single-node static CP lane."
        ),
    )


def add_vae_parallel_arguments(
    parser: argparse.ArgumentParser, *, with_degree: bool = False, default: bool = True
) -> None:
    """Add the VAEP flags shared by every standalone generate entry point."""

    parser.add_argument(
        "--parallel-vae",
        action=argparse.BooleanOptionalAction,
        default=default,
        help="Parallelize decoding across the VAE group when the decoder supports it.",
    )
    if with_degree:
        parser.add_argument(
            "--vae-parallel-degree",
            type=int,
            default=None,
            help="VAE group size; defaults to the whole stage.",
        )
    parser.add_argument(
        "--vae-parallel-halo",
        type=int,
        default=8,
        help="Legacy tile overlap; layer-wise decoders infer convolution halos automatically.",
    )


def validate_vae_parallel_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.vae_parallel_halo < 0:
        parser.error("--vae-parallel-halo must be non-negative")
    degree = getattr(args, "vae_parallel_degree", None)
    if degree is not None and degree < 1:
        parser.error("--vae-parallel-degree must be positive")


def vae_parallel_degree(args: argparse.Namespace, world_size: int) -> int:
    """Resolve the VAEP group size a standalone launch should build."""

    if not args.parallel_vae:
        return 1
    return getattr(args, "vae_parallel_degree", None) or world_size


def static_parallel_pipeline_kwargs(args: argparse.Namespace) -> dict[str, object]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # A context-parallel lane lives inside one tensor-parallel plane, so the
    # widest lane is the plane, not the world.
    degree = int(getattr(args, "tensor_parallel_degree", 1) or 1)
    if degree < 1 or world_size % degree:
        raise ValueError(
            f"tensor_parallel_degree {degree} must be a positive divisor of "
            f"world size {world_size}"
        )
    return {
        "allowed_lane_widths": tuple(sorted({1, world_size // degree})),
        "ulysses_transport": args.ulysses_transport,
        "agkv_transport": args.agkv_transport,
    }
