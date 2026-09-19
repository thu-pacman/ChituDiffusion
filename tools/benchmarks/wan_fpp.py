"""Plan or run the b64ddaa latency matrix and warmup/cooldown quality sweep."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

# name, CFG, PP, CP, patches: experiments/exp02.sh at SmartDiffusion b64ddaa.
MATRIX = (
    ("baseline", 1, 1, 1, 1),
    ("4cp", 1, 1, 4, 1),
    ("4pp", 1, 4, 1, 7),
    ("2cfg2cp", 2, 1, 2, 1),
    ("2cfg2pp", 2, 2, 1, 3),
    ("2cp2pp", 1, 2, 2, 3),
    ("8cp", 1, 1, 8, 1),
    ("8pp", 1, 8, 1, 12),
    ("2cfg4cp", 2, 1, 4, 1),
    ("2cfg4pp", 2, 4, 1, 7),
    ("2cfg2cp2pp", 2, 2, 2, 3),
)
SCHEDULES = tuple((n, 0) for n in range(1, 11)) + (
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1),
    (4, 3),
    (5, 1),
    (5, 3),
    (5, 5),
    (6, 1),
    (7, 1),
)


def build_cases(args):
    selected = [row for row in MATRIX if not args.case or row[0] in args.case]
    if args.case and set(args.case) - {row[0] for row in MATRIX}:
        raise ValueError("unknown matrix case")
    if args.mode == "warmup":
        selected = [
            MATRIX[0],
            (
                "reference",
                1,
                args.reference_stages,
                args.reference_context_degree,
                args.patches,
            ),
        ]
    jobs = []
    for name, cfg, pp, cp, patches in selected:
        schedules = (
            SCHEDULES if name == "reference" else ((args.warmup, args.cooldown),)
        )
        for warmup, cooldown in schedules:
            virtual = name == "reference"
            world = 1 if virtual else cfg * pp * cp
            suffix = f"-w{warmup}-c{cooldown}" if virtual else ""
            for seed in args.seeds:
                output = args.output_dir / f"{name}{suffix}-seed{seed}.mp4"
                command = [
                    "torchrun",
                    "--standalone",
                    f"--nproc_per_node={world}",
                    "-m",
                    "chitu_diffusion.commands.generate.wan",
                    "--model-path",
                    str(args.model_path),
                    "--output",
                    str(output),
                    "--prompt",
                    args.prompt,
                    "--negative-prompt",
                    args.negative_prompt,
                    "--height",
                    str(args.height),
                    "--width",
                    str(args.width),
                    "--frames",
                    str(args.frames),
                    "--steps",
                    str(args.steps),
                    "--sample-solver",
                    "unipc",
                    "--flow-shift",
                    str(args.flow_shift),
                    "--guidance-scale",
                    str(args.guidance_scale),
                    "--seed",
                    str(seed),
                    "--fpp-patches",
                    str(patches),
                    "--fpp-warmup-steps",
                    str(warmup),
                    "--fpp-cooldown-steps",
                    str(cooldown),
                    "--fpp-refresh-interval",
                    "50",
                    "--no-parallel-vae",
                    "--save-frames",
                    "--cfg-parallel" if cfg == 2 else "--no-cfg-parallel",
                ]
                if virtual:
                    command += [
                        "--fpp-reference-stages",
                        str(pp),
                        "--fpp-reference-context-degree",
                        str(cp),
                    ]
                else:
                    command += [
                        "--pipeline-parallel-degree",
                        str(pp),
                        "--context-parallel-degree",
                        str(cp),
                    ]
                jobs.append(
                    dict(
                        case=name,
                        seed=seed,
                        warmup=warmup,
                        cooldown=cooldown,
                        world_size=world,
                        output=str(output),
                        command=command,
                    )
                )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("matrix", "warmup"), default="matrix")
    parser.add_argument("--case", action="append", choices=[row[0] for row in MATRIX])
    parser.add_argument("--prompt", default="A cat walking on grass.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--flow-shift", type=float, default=3)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--cooldown", type=int, default=1)
    parser.add_argument("--reference-stages", type=int, default=4)
    parser.add_argument("--reference-context-degree", type=int, default=1)
    parser.add_argument("--patches", type=int, default=7)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the plan; otherwise only write commands and metadata.",
    )
    args = parser.parse_args()
    if args.mode == "warmup" and args.case:
        parser.error("--case applies only to matrix mode")
    if (
        args.warmup < 1
        or args.cooldown < 0
        or args.reference_stages < 1
        or args.reference_context_degree < 1
    ):
        parser.error("invalid warmup/cooldown or reference degrees")
    if args.mode == "warmup" and args.patches < args.reference_stages:
        parser.error("patches must be >= reference stages")
    jobs = build_cases(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = dict(
        source="SmartDiffusion/exp/fpp@b64ddaa268516901a19372e421af56eaaf6f43c9",
        timing="cold end-to-end generate, excludes loading and output encoding; no claimed speedup",
        jobs=jobs,
    )
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for job in jobs:
        print(shlex.join(job["command"]), flush=True)
        if args.execute:
            subprocess.run(job["command"], check=True)


if __name__ == "__main__":
    main()
