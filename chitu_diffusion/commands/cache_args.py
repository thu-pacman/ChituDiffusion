from __future__ import annotations

import argparse

from ..flexcache.config import (
    CacheCommonConfig,
    CacheConfig,
    MagCacheConfig,
    MeanCacheConfig,
    PABConfig,
    TaylorSeerConfig,
    TeaCacheConfig,
)


def add_cache_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-strategy",
        choices=("none", "magcache", "meancache", "teacache", "taylorseer", "pab"),
        default="none",
    )
    parser.add_argument("--cache-warmup-steps", type=int, default=0)
    parser.add_argument("--cache-cooldown-steps", type=int, default=0)
    parser.add_argument("--magcache-threshold", type=float)
    parser.add_argument("--magcache-max-skip-steps", type=int)
    parser.add_argument("--magcache-retention-ratio", type=float)
    parser.add_argument("--meancache-fresh-steps", type=int, default=25)
    parser.add_argument("--teacache-threshold", type=float, default=0.2)
    parser.add_argument(
        "--teacache-coefficients",
        type=float,
        nargs="+",
        default=None,
        help=(
            "override the rescale polynomial, highest degree first; only the "
            "detected checkpoint's published coefficients are calibrated for "
            "the default threshold"
        ),
    )
    parser.add_argument("--taylorseer-fresh-threshold", type=int, default=3)
    parser.add_argument("--taylorseer-max-order", type=int, default=1)
    parser.add_argument("--taylorseer-first-enhance", type=int, default=3)
    parser.add_argument("--pab-self-interval", type=int, default=2)
    parser.add_argument("--pab-cross-interval", type=int, default=3)
    parser.add_argument("--pab-joint-interval", type=int, default=2)


def cache_config_from_args(args: argparse.Namespace) -> CacheConfig:
    common = CacheCommonConfig(
        warmup_steps=args.cache_warmup_steps,
        cooldown_steps=args.cache_cooldown_steps,
    )
    if args.cache_strategy == "none":
        params = None
    elif args.cache_strategy == "magcache":
        params = MagCacheConfig(
            threshold=args.magcache_threshold,
            max_skip_steps=args.magcache_max_skip_steps,
            retention_ratio=args.magcache_retention_ratio,
        )
    elif args.cache_strategy == "meancache":
        params = MeanCacheConfig(fresh_steps=args.meancache_fresh_steps)
    elif args.cache_strategy == "teacache":
        coefficients = args.teacache_coefficients
        params = TeaCacheConfig(
            threshold=args.teacache_threshold,
            coefficients=tuple(coefficients) if coefficients else None,
        )
    elif args.cache_strategy == "taylorseer":
        params = TaylorSeerConfig(
            fresh_threshold=args.taylorseer_fresh_threshold,
            max_order=args.taylorseer_max_order,
            first_enhance=args.taylorseer_first_enhance,
        )
    elif args.cache_strategy == "pab":
        params = PABConfig(
            self_interval=args.pab_self_interval,
            cross_interval=args.pab_cross_interval,
            joint_interval=args.pab_joint_interval,
        )
    else:  # pragma: no cover - argparse enforces the choices.
        raise ValueError(f"unsupported cache strategy: {args.cache_strategy}")
    return CacheConfig(strategy=args.cache_strategy, common=common, params=params)
