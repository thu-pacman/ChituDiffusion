#!/usr/bin/env python3
"""Build one Fast CP extension for H20 and RTX PRO 5000, without requiring a GPU."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from install_fast_agkv import apply_overlay


def build_environment(
    architectures: list[str], *, cuda_home: Path, nvshmem_home: Path, jobs: int
) -> dict[str, str]:
    if not architectures or any(arch not in {"90", "120"} for arch in architectures):
        raise ValueError(
            "Fast CP build targets must be 90 (H20) and/or 120 (RTX PRO 5000)"
        )
    if jobs < 1:
        raise ValueError("jobs must be positive")
    nvcc = cuda_home / "bin" / "nvcc"
    supported = subprocess.check_output(
        [str(nvcc), "--list-gpu-code"], text=True
    ).split()
    missing = [arch for arch in architectures if f"sm_{arch}" not in supported]
    if missing:
        raise RuntimeError(
            f"{nvcc} cannot compile SM{', SM'.join(missing)}; use CUDA 12.8+ for SM120"
        )
    if not (nvshmem_home / "include" / "nvshmem.h").is_file():
        raise FileNotFoundError(f"NVSHMEM headers are missing in {nvshmem_home}")
    env = os.environ.copy()
    env.update(
        CUDA_HOME=str(cuda_home),
        CUDACXX=str(nvcc),
        CUDAToolkit_ROOT=str(cuda_home),
        NVSHMEM_HOME=str(nvshmem_home),
        FAST_ULYSSES_CUDA_ARCH=";".join(dict.fromkeys(architectures)),
        CMAKE_BUILD_PARALLEL_LEVEL=str(jobs),
    )
    env["PATH"] = os.pathsep.join(
        (str(Path(sys.executable).parent), str(cuda_home / "bin"), env.get("PATH", ""))
    )
    cccl = cuda_home / "include" / "cccl"
    if cccl.is_dir():
        for variable in ("CUDAFLAGS", "CXXFLAGS"):
            env[variable] = (
                env.get(variable, "") + " -I" + shlex.quote(str(cccl))
            ).strip()
    # NVIDIA's Python wheels use lib/; nvcc's default search path is lib64/.
    if (cuda_home / "lib" / "libcudart_static.a").exists():
        env["CUDAFLAGS"] = (
            env.get("CUDAFLAGS", "") + " -L" + shlex.quote(str(cuda_home / "lib"))
        ).strip()
        env["LIBRARY_PATH"] = os.pathsep.join(
            (str(cuda_home / "lib"), env.get("LIBRARY_PATH", ""))
        )
    if not (nvshmem_home / "lib" / "cmake" / "nvshmem").is_dir():
        config = Path(__file__).resolve().parent / "cmake"
        env["FAST_ULYSSES_CMAKE_ARGS"] = (
            env.get("FAST_ULYSSES_CMAKE_ARGS", "")
            + " "
            + shlex.quote(f"-DNVSHMEM_DIR={config}")
        ).strip()
    return env


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    deps_root = Path(
        os.environ.get("FAST_CP_DEPS_ROOT", project_root / ".deps/fast-cp")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkout",
        type=Path,
        nargs="?",
        default=os.environ.get("FAST_CP_BUILD_ROOT", deps_root / "src/fast-ulysses"),
        help="fast-ulysses source checkout (default: .deps/fast-cp/src/fast-ulysses)",
    )
    parser.add_argument(
        "--architectures", nargs="+", choices=("90", "120"), default=["90", "120"]
    )
    parser.add_argument("--cuda-home", type=Path, default=os.environ.get("CUDA_HOME"))
    parser.add_argument(
        "--nvshmem-home",
        type=Path,
        default=os.environ.get("NVSHMEM_HOME", deps_root / "python/nvidia/nvshmem"),
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--wheel-dir",
        type=Path,
        help="also create a distributable wheel in this directory",
    )
    args = parser.parse_args()
    if args.cuda_home is None or args.nvshmem_home is None:
        parser.error("set --cuda-home/CUDA_HOME and --nvshmem-home/NVSHMEM_HOME")
    env = build_environment(
        args.architectures,
        cuda_home=args.cuda_home.resolve(),
        nvshmem_home=args.nvshmem_home.resolve(),
        jobs=args.jobs,
    )
    checkout = args.checkout.resolve()
    apply_overlay(checkout, check_only=False)
    command = [sys.executable, "setup.py", "build_ext", "--inplace"]
    if args.wheel_dir is not None:
        command += ["bdist_wheel", "--dist-dir", str(args.wheel_dir.resolve())]
    subprocess.run(command, cwd=checkout, env=env, check=True)
    cuobjdump = shutil.which("cuobjdump", path=env["PATH"])
    if cuobjdump is None:
        raise RuntimeError("cuobjdump is required to verify the wheel's CUDA images")
    extensions = list((checkout / "fast_ulysses").glob("_C*.so"))
    if len(extensions) != 1:
        raise RuntimeError(f"expected one built extension, found {extensions}")
    images = subprocess.check_output(
        [cuobjdump, "--list-elf", str(extensions[0])], text=True
    )
    for arch in args.architectures:
        if f"sm_{arch}." not in images:
            raise RuntimeError(
                f"built extension does not contain native sm_{arch} images"
            )
    print(
        f"Verified Fast CP native CUDA images: {', '.join('sm_' + a for a in args.architectures)}"
    )
    print(f"Extension: {extensions[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
