from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/install/fast_cp_env.sh"
VARIABLES = (
    "FAST_CP_DEPS_ROOT",
    "FAST_CP_BUILD_ROOT",
    "FLASH_ATTENTION_ROOT",
    "NVSHMEM_HOME",
    "CHITU_EXTRA_PYTHONPATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
)


@pytest.mark.parametrize("custom_root", [False, True])
def test_environment_works_outside_repo_and_survives_slurm_wrapper(
    tmp_path, custom_root
):
    env = os.environ.copy()
    for name in (*VARIABLES, "BASH_ENV", "CHITU_PYTHON_BIN"):
        env.pop(name, None)
    env["PYTHONPATH"] = "/existing/python"
    env["LD_LIBRARY_PATH"] = "/existing/lib"
    prefix = (
        tmp_path / "dependencies with spaces" if custom_root else ROOT / ".deps/fast-cp"
    )
    if custom_root:
        env["FAST_CP_DEPS_ROOT"] = str(prefix)
    env.update(
        CHITU_PROJECT_ROOT=str(ROOT),
        SLURM_PROCID="0",
        SLURM_LOCALID="0",
        SLURM_NTASKS="1",
        SLURM_JOB_ID="0",
    )
    # No scheduler or GPU is involved: emulate only the Slurm environment.
    (tmp_path / "scontrol").write_text("#!/bin/sh\nexit 0\n")
    (tmp_path / "scontrol").chmod(0o755)
    env["PATH"] = str(tmp_path) + os.pathsep + env["PATH"]
    report = (
        "import json, os; print(json.dumps({k: os.environ[k] for k in "
        + repr(VARIABLES)
        + "}))"
    )
    output = subprocess.check_output(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            'set -eu; source "$1"; source "$1"; exec bash "$2" -c "$3"',
            "bash",
            str(SCRIPT),
            str(ROOT / "tools/cluster/srun_wrapper.sh"),
            report,
        ],
        cwd=tmp_path,
        env=env,
        text=True,
    )
    values = json.loads(output.splitlines()[-1])
    extension = str(prefix / "src/fast-ulysses")
    attention = str(prefix / "src/flash-attention")
    assert values["FAST_CP_BUILD_ROOT"] == extension
    assert values["FLASH_ATTENTION_ROOT"] == attention
    assert values["NVSHMEM_HOME"] == str(prefix / "python/nvidia/nvshmem")
    assert values["CHITU_EXTRA_PYTHONPATH"].split(":") == [extension, attention]
    assert values["PYTHONPATH"].split(":") == [str(ROOT), extension, attention]
    assert values["LD_LIBRARY_PATH"].split(":") == [
        str(prefix / "python/nvidia/nvshmem/lib"),
        "/existing/lib",
    ]


def test_environment_preserves_explicit_dependency_locations(tmp_path):
    env = os.environ.copy()
    env.pop("BASH_ENV", None)
    overrides = {
        "FAST_CP_BUILD_ROOT": str(tmp_path / "extension"),
        "FLASH_ATTENTION_ROOT": str(tmp_path / "attention"),
        "NVSHMEM_HOME": str(tmp_path / "nvshmem"),
    }
    env.update(overrides)
    report = (
        "import json, os; print(json.dumps({k: os.environ[k] for k in "
        + repr(tuple(overrides))
        + "}))"
    )
    output = subprocess.check_output(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            'source "$1"; exec "$2" -c "$3"',
            "bash",
            str(SCRIPT),
            sys.executable,
            report,
        ],
        env=env,
        text=True,
    )
    assert json.loads(output) == overrides
