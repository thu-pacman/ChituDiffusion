"""Collect full-compute velocity Grams and finite-perturbation responses.

Single-process, single-GPU offline tool. GPU execution requires Slurm debug.
Scheduler hooks are temporary and restored even if generation fails.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from .preprocess import sha256

FAMILIES = {"flux": "flux1", "qwen": "qwen_image", "zimage": "zimage"}
INJECTION_STEPS = (4, 6, 8, 10, 13, 16, 20, 24, 28, 32, 36, 40, 43, 45, 47, 48)


def save(path, value):
    """Only called inside a newly created run directory owned by this run."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def check_gpu_job():
    if os.environ.get("SLURM_JOB_PARTITION") != "debug":
        raise RuntimeError("GPU collection/evaluation requires Slurm --partition=debug")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1:
        raise RuntimeError("use one process and exactly one visible GPU")


def add_arguments(parser):
    parser.add_argument("--model", choices=FAMILIES, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--prompts", type=Path, required=True, help="JSON list of {id,prompt}"
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument(
        "--guidance",
        type=float,
        required=True,
        help="Qwen true_cfg_scale; guidance_scale otherwise",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new directory, never overwritten"
    )


def load_model(args):
    from chitu_diffusion import (
        Flux1Pipeline,
        Flux1Request,
        QwenImagePipeline,
        QwenImageRequest,
        ZImagePipeline,
        ZImageRequest,
    )

    pipeline_type, request_type = {
        "flux": (Flux1Pipeline, Flux1Request),
        "qwen": (QwenImagePipeline, QwenImageRequest),
        "zimage": (ZImagePipeline, ZImageRequest),
    }[args.model]
    pipeline = pipeline_type.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        allowed_lane_widths=(1,),
        local_files_only=True,
    )
    return pipeline, request_type


def request_for(args, request_type, prompt, seed, *, output_type="latent", cache=None):
    from chitu_diffusion import CacheConfig

    return request_type(
        prompt=prompt,
        seed=seed,
        width=args.width,
        height=args.height,
        num_steps=50,
        output_type=output_type,
        cache=cache if cache is not None else CacheConfig(),
        **{
            "true_cfg_scale"
            if args.model == "qwen"
            else "guidance_scale": args.guidance
        },
    )


def read_prompts(path):
    prompts = json.loads(path.read_text())
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts must be a nonempty list")
    if any(
        not isinstance(p.get("id"), str)
        or not p["id"]
        or not isinstance(p.get("prompt"), str)
        or not p["prompt"]
        for p in prompts
    ):
        raise ValueError("each prompt requires nonempty string id and prompt")
    if len({p["id"] for p in prompts}) != len(prompts):
        raise ValueError("prompt IDs must be unique")
    return prompts


def protocol_for(args, pipeline, request_type, prompts):
    from dataclasses import asdict

    inner = pipeline.diffusers_pipeline
    config = dict(inner.scheduler.config)
    if "_use_default_values" in config:
        config["_use_default_values"] = sorted(config["_use_default_values"])
    if type(
        inner.scheduler
    ).__name__ != "FlowMatchEulerDiscreteScheduler" or config.get(
        "stochastic_sampling", False
    ):
        raise ValueError(
            "only deterministic FlowMatchEulerDiscreteScheduler is supported"
        )
    request = asdict(
        request_for(args, request_type, prompts[0]["prompt"], args.seeds[0])
    )
    for key in ("request_id", "prompt", "seed", "output_type"):
        request.pop(key, None)
    model_path = Path(args.model_path).resolve()
    model_configs = {
        str(p.relative_to(model_path)): sha256(p)
        for p in sorted(model_path.rglob("*.json"))
        if p.name in {"config.json", "scheduler_config.json", "model_index.json"}
    }
    return {
        "complete": False,
        "setting": {
            "model": args.model,
            "model_family": FAMILIES[args.model],
            "model_path": str(model_path),
            "model_configs_sha256": model_configs,
            "request": request,
            "scheduler_class": type(inner.scheduler).__name__,
            "scheduler_config": config,
            "dtype": "bfloat16",
            "attention_processors": sorted(
                {
                    type(p).__name__
                    for module in inner.transformer.modules()
                    if (p := getattr(module, "processor", None)) is not None
                }
            ),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "diffusers", "transformers", "numpy")
            },
        },
        "prompts": prompts,
        "seeds": args.seeds,
        "prompt_file_sha256": sha256(args.prompts),
        "collector_sha256": sha256(Path(__file__)),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "gpu": torch.cuda.get_device_name(),
        "node": platform.node(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "partition": "debug",
        "weight_hash_scope": "configuration files only; separately retain checkpoint revision/weight hashes",
    }


def sample_from(result):
    return result[0] if isinstance(result, tuple) else result.prev_sample


class Recorder:
    def __init__(self, *, record=True, injection=None):
        self.record = record
        self.injection = injection
        self.velocities, self.state_norms = [], []
        self.sigmas, self.final = None, None
        self.calls = 0

    def step(self, original, scheduler, velocity, *args, **kwargs):
        index = 0 if scheduler.step_index is None else int(scheduler.step_index)
        if index != self.calls:
            raise ValueError(
                "expected one sequential Euler scheduler update per model step"
            )
        self.calls += 1
        sigmas = scheduler.sigmas.detach().cpu().double().tolist()
        if self.sigmas is None:
            self.sigmas = sigmas
        elif self.sigmas != sigmas:
            raise ValueError("sigma grid changed within generation")
        if self.record:
            sample = kwargs["sample"] if "sample" in kwargs else args[1]
            self.velocities.append(velocity.detach().cpu().float().clone())
            self.state_norms.append(
                float(torch.linalg.vector_norm(sample.float()).item())
            )
        result = original(scheduler, velocity, *args, **kwargs)
        sample = sample_from(result)
        if self.injection is not None and index == self.injection[0]:
            delta = self.injection[1].to(device=sample.device, dtype=sample.dtype)
            modified = sample + delta
            self.actual_injection_norm = float(
                torch.linalg.vector_norm((modified.float() - sample.float())).item()
            )
            if isinstance(result, tuple):
                result = (modified, *result[1:])
            else:
                result.prev_sample = modified
            sample = modified
        self.final = sample.detach().cpu().float().clone()
        return result

    def gram_trace(self):
        if self.calls != 50 or len(self.velocities) != 50 or len(self.sigmas) != 51:
            raise ValueError("incomplete 50-step trace")
        matrix = torch.stack([v.flatten() for v in self.velocities]).double()
        return {"sigmas": self.sigmas, "gram": (matrix @ matrix.T).tolist()}


@contextmanager
def observe(pipeline, recorder):
    scheduler_type = type(pipeline.diffusers_pipeline.scheduler)
    original = scheduler_type.step

    def wrapped(scheduler, velocity, *args, **kwargs):
        return recorder.step(original, scheduler, velocity, *args, **kwargs)

    scheduler_type.step = wrapped
    try:
        yield
    finally:
        scheduler_type.step = original


def generate(pipeline, request, recorder=None):
    torch.cuda.synchronize()
    started = time.perf_counter()
    if recorder is None:
        output = pipeline.generate(request)
    else:
        with observe(pipeline, recorder):
            output = pipeline.generate(request)
        if recorder.calls != 50:
            raise ValueError("generation did not execute exactly 50 scheduler steps")
    torch.cuda.synchronize()
    return output, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("trace", "propagation"))
    add_arguments(parser)
    parser.add_argument("--inject-steps", type=int, nargs="+", default=INJECTION_STEPS)
    parser.add_argument(
        "--relative-epsilon",
        type=float,
        help="required for propagation; no cross-model default",
    )
    args = parser.parse_args()
    check_gpu_job()
    if args.mode == "propagation" and (
        args.relative_epsilon is None or not 0 < args.relative_epsilon <= 1
    ):
        parser.error("propagation requires --relative-epsilon in (0,1]")
    if any(i < 2 or i >= 50 for i in args.inject_steps):
        parser.error("injection steps must lie in [2,49]")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must be unique")
    prompts = read_prompts(args.prompts)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    pipeline = None
    protocol = {"complete": False, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    traces, rows = [], []
    try:
        pipeline, request_type = load_model(args)
        protocol = protocol_for(args, pipeline, request_type, prompts)
        protocol.update(
            mode=args.mode,
            gram_accumulation="float64_cpu",
            relative_epsilon=args.relative_epsilon,
            inject_steps=list(args.inject_steps),
            anchor_gap=2,
        )
        save(args.output / "protocol.json", protocol)
        for prompt in prompts:
            for seed in args.seeds:
                request = request_for(args, request_type, prompt["prompt"], seed)
                baseline, _ = generate(pipeline, request)
                recorder = Recorder()
                recorded, _ = generate(pipeline, request, recorder)
                # Detect observer-induced changes without comparing raw latents
                # against unpacked facade output (Flux changes representation).
                if not torch.equal(baseline.images, recorded.images):
                    raise ValueError("observer parity failed")
                if "sigmas" in protocol and protocol["sigmas"] != recorder.sigmas:
                    raise ValueError("sigma grid differs between requests")
                protocol["sigmas"] = recorder.sigmas
                if args.mode == "trace":
                    traces.append(
                        dict(
                            recorder.gram_trace(),
                            prompt_id=prompt["id"],
                            prompt=prompt["prompt"],
                            seed=seed,
                        )
                    )
                    save(
                        args.output / "traces.json",
                        {
                            "complete": False,
                            "steps": 50,
                            "model_family": FAMILIES[args.model],
                            "protocol": protocol,
                            "traces": traces,
                        },
                    )
                else:
                    for step in args.inject_steps:
                        direction = (
                            recorder.velocities[step - 2] - recorder.velocities[step]
                        )
                        norm = torch.linalg.vector_norm(direction)
                        if norm <= 1e-12:
                            raise ValueError(
                                "zero ZOH direction: choose a different probe"
                            )
                        epsilon = args.relative_epsilon * recorder.state_norms[step]
                        probe = Recorder(
                            record=False, injection=(step, direction / norm * epsilon)
                        )
                        generate(pipeline, request, probe)
                        deviation = float(
                            torch.linalg.vector_norm(
                                probe.final - recorder.final
                            ).item()
                        )
                        rows.append(
                            dict(
                                model=args.model,
                                prompt_id=prompt["id"],
                                seed=seed,
                                steps=50,
                                inject_step=step,
                                anchor_gap=2,
                                direction="zoh",
                                relative_epsilon=args.relative_epsilon,
                                epsilon=epsilon,
                                actual_injection_norm=probe.actual_injection_norm,
                                final_deviation=deviation,
                                gain=deviation / max(epsilon, 1e-12),
                            )
                        )
                        with (args.output / "propagation.csv").open("w") as stream:
                            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                            writer.writeheader()
                            writer.writerows(rows)
                save(args.output / "protocol.json", protocol)
        protocol["complete"] = True
    except Exception as error:
        protocol["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if pipeline is not None:
            pipeline.close()
        protocol["elapsed_seconds_including_load"] = time.perf_counter() - started
        save(args.output / "protocol.json", protocol)
        if args.mode == "trace" and traces:
            save(
                args.output / "traces.json",
                {
                    "complete": protocol["complete"],
                    "steps": 50,
                    "model_family": FAMILIES[args.model],
                    "protocol": protocol,
                    "traces": traces,
                },
            )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": True,
                "elapsed_seconds": protocol["elapsed_seconds_including_load"],
            }
        )
    )


if __name__ == "__main__":
    main()
