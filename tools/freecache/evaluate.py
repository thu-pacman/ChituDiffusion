"""Paired native-resolution validation of candidate profiles in Slurm debug.

This reports evidence; it never promotes a candidate or rewrites runtime tables.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace

import numpy as np
import torch

from .collect import (
    FAMILIES,
    add_arguments,
    check_gpu_job,
    generate,
    load_model,
    protocol_for,
    read_prompts,
    request_for,
    save,
)
from .preprocess import normalized_setting, sha256


def warmup_periodic(budget, warmup, steps=50):
    if not 2 <= warmup <= budget <= steps:
        raise ValueError("expected 2 <= warmup <= budget <= steps")
    remaining = budget - warmup
    tail = [
        warmup + int(round(i * (steps - warmup) / remaining)) for i in range(remaining)
    ]
    return [*range(warmup), *tail]


def quality(reference, candidate, metric=None):
    if (
        reference.shape != candidate.shape
        or reference.ndim != 4
        or reference.shape[-1] != 3
    ):
        raise ValueError("expected matching native NHWC RGB arrays")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("nonfinite RGB output")
    mse = float(
        np.mean((candidate.astype(np.float64) - reference.astype(np.float64)) ** 2)
    )
    result = {
        "psnr": None if mse == 0 else -10 * math.log10(mse),
        "exact": mse == 0,
        "native_shape": list(reference.shape),
    }
    if metric is not None:

        def tensor(array):
            return torch.from_numpy(array).permute(0, 3, 1, 2).cuda().float() * 2 - 1

        with torch.inference_mode():
            result["one_minus_lpips"] = 1 - float(
                metric(tensor(reference), tensor(candidate)).mean().item()
            )
    return result


def main():
    from pathlib import Path

    from chitu_diffusion import (
        CacheConfig,
        FreeCacheConfig,
        FreeCacheProfile,
        MagCacheConfig,
        MeanCacheConfig,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument(
        "--candidates", type=Path, required=True, help="output of preprocess fit"
    )
    parser.add_argument(
        "--lpips",
        action="store_true",
        help="requires eval extra and cached AlexNet weights",
    )
    parser.add_argument(
        "--magcache",
        action="store_true",
        help="include the model's bundled default where supported",
    )
    parser.add_argument(
        "--meancache",
        action="store_true",
        help="include supported official budgets for Qwen/Z-Image",
    )
    args = parser.parse_args()
    check_gpu_job()
    candidates = json.loads(args.candidates.read_text())
    prompts = read_prompts(args.prompts)
    old = candidates["protocol"]
    if {p["prompt"] for p in prompts} & {p["prompt"] for p in old["prompts"]} or set(
        args.seeds
    ) & set(old["seeds"]):
        parser.error(
            "validation requires separate prompts and seeds from trace collection"
        )
    settings = []
    for record in candidates["profiles"]:
        profile = FreeCacheProfile(**record)
        if profile.model_family != FAMILIES[args.model]:
            parser.error("candidate model family mismatch")
        budget = len(profile.fresh_steps)
        for label, selected in (
            (f"candidate-f{budget}", profile),
            (
                f"warmup-zoh-f{budget}",
                replace(
                    profile,
                    profile_id=f"warmup-zoh-f{budget}",
                    fresh_steps=tuple(warmup_periodic(budget, candidates["warmup"])),
                    proposal_coefficient=0.0,
                ),
            ),
        ):
            settings.append(
                (
                    label,
                    CacheConfig(
                        strategy="freecache", params=FreeCacheConfig(profile=selected)
                    ),
                )
            )
        supported = {"qwen": {10, 17, 25}, "zimage": {13, 15, 17, 20, 25}}
        if args.meancache and budget in supported.get(args.model, set()):
            settings.append(
                (
                    f"meancache-f{budget}",
                    CacheConfig(
                        strategy="meancache", params=MeanCacheConfig(fresh_steps=budget)
                    ),
                )
            )
    if args.meancache and not any(
        label.startswith("meancache") for label, _ in settings
    ):
        parser.error("no requested budget has a supported MeanCache profile")
    if args.magcache:
        if args.model == "zimage":
            parser.error("Z-Image has no bundled MagCache profile")
        settings.append(
            (
                "magcache-default",
                CacheConfig(strategy="magcache", params=MagCacheConfig()),
            )
        )
    args.output.mkdir(parents=True, exist_ok=False)
    pipeline = None
    protocol = {"complete": False}
    rows = []
    started = time.perf_counter()
    try:
        pipeline, request_type = load_model(args)
        protocol = protocol_for(args, pipeline, request_type, prompts)
        if normalized_setting(protocol["setting"]) != normalized_setting(
            old["setting"]
        ):
            raise ValueError("candidate collection and validation settings differ")
        protocol["candidate_sha256"] = sha256(args.candidates)
        protocol["evaluator_sha256"] = sha256(Path(__file__))
        metric = None
        if args.lpips:
            import lpips

            metric = lpips.LPIPS(net="alex").cuda().eval()
        full = FreeCacheProfile(
            "parity-f50", FAMILIES[args.model], 50, tuple(range(50))
        )
        for case, (prompt, seed) in enumerate(
            (p, s) for p in prompts for s in args.seeds
        ):
            request = request_for(
                args, request_type, prompt["prompt"], seed, output_type="np"
            )
            generate(pipeline, request)  # untimed warmup
            origin, origin_seconds = generate(pipeline, request)
            reference = np.asarray(origin.images, dtype=np.float32)
            full_output, _ = generate(
                pipeline,
                replace(
                    request,
                    cache=CacheConfig(
                        strategy="freecache", params=FreeCacheConfig(profile=full)
                    ),
                ),
            )
            if not np.array_equal(reference, np.asarray(full_output.images)):
                raise ValueError("F50 does not exactly match no-cache reference")
            np.save(args.output / f"case-{case}-origin.npy", reference)
            for index in np.random.default_rng(700 + case).permutation(len(settings)):
                label, cache = settings[index]
                output, seconds = generate(pipeline, replace(request, cache=cache))
                candidate = np.asarray(output.images, dtype=np.float32)
                stats = pipeline.last_cache_stats
                if hasattr(stats, "to_dict"):
                    stats = stats.to_dict()
                if cache.strategy == "freecache" and stats["step_misses"] != len(
                    cache.params.profile.fresh_steps
                ):
                    raise ValueError("Fresh count does not match profile")
                np.save(args.output / f"case-{case}-{label}.npy", candidate)
                rows.append(
                    dict(
                        case=case,
                        prompt_id=prompt["id"],
                        seed=seed,
                        label=label,
                        seconds=seconds,
                        origin_seconds=origin_seconds,
                        speedup=origin_seconds / seconds,
                        cache=cache.to_dict(),
                        stats=stats,
                        **quality(reference, candidate, metric),
                    )
                )
                save(args.output / "rows.json", rows)
        protocol["complete"] = True
    except Exception as error:
        protocol["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if pipeline is not None:
            pipeline.close()
        protocol["elapsed_seconds_including_load"] = time.perf_counter() - started
        save(args.output / "protocol.json", protocol)
    print(f"Validated {len(rows)} paired cells; no runtime profiles changed")


if __name__ == "__main__":
    main()
