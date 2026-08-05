import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "system_config.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _bool01(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return "1" if str(value).strip().lower() in {"1", "true", "yes", "on"} else "0"


def _bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "true" if str(value).strip().lower() in {"1", "true", "yes", "on"} else "false"


def _normalize_log_ranks(raw: Any) -> tuple[str, str]:
    if isinstance(raw, str):
        text = raw.strip().lower() or "0"
        return text, text
    if isinstance(raw, (list, tuple)):
        items = [str(int(item)) for item in raw]
        return ",".join(items), "[" + ",".join(items) + "]"
    raise ValueError("output.log_ranks must be a list of ranks or 'all'")


def _normalize_eval_type(raw: Any) -> str:
    if raw is None:
        return "[]"
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"", "none", "null"}:
            return "[]"
        if "," in value:
            return "[" + ",".join(item.strip() for item in value.split(",") if item.strip()) + "]"
        return f"[{value}]"
    if isinstance(raw, (list, tuple)):
        items = []
        for item in raw:
            value = str(item).strip().lower()
            if value and value not in {"none", "null"}:
                items.append(value)
        return "[" + ",".join(items) + "]"
    raise ValueError("eval.eval_type must be string/list/null")


def _runtime_python() -> str:
    configured = os.environ.get("CHITU_PYTHON_BIN", "").strip()
    if configured:
        return configured
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.is_file() and os.access(venv_python, os.X_OK):
        return str(venv_python)
    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError("no runtime python found")


def _slugify(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "_", value.lower()).strip("._-")
    return (slug or fallback)[:64]


def _new_task_id() -> str:
    return uuid.uuid4().hex[:8]


def _write_line(line: str, handles: list[Any]) -> None:
    print(line, flush=True)
    for handle in handles:
        handle.write(line + "\n")
        handle.flush()


def _stream_command(cmd: list[str], env: dict[str, str], cwd: Path, handles: list[Any]) -> int:
    if not handles:
        return subprocess.run(cmd, cwd=cwd, env=env).returncode

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        for handle in handles:
            handle.write(line)
            handle.flush()
    return proc.wait()


def run_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config_path = config_path.resolve()

    if not config_path.is_file():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    cfg = _read_yaml(config_path)
    run_tag = str(_get(cfg, "launch.tag", "") or "").strip()
    launch_backend = str(_get(cfg, "launch.backend", "srun")).strip().lower()
    num_nodes = int(args.num_nodes if args.num_nodes is not None else _get(cfg, "launch.num_nodes", 1))
    gpus_per_node = int(
        args.gpus_per_node if args.gpus_per_node is not None else _get(cfg, "launch.gpus_per_node", 1)
    )
    python_script = str(_get(cfg, "launch.python_script", "test/test_generate.py"))
    srun_partition = str(_get(cfg, "launch.srun.partition", "debug"))
    srun_cpus_per_gpu = str(int(_get(cfg, "launch.srun.cpus_per_gpu", 24)))
    srun_mem_per_gpu = str(int(_get(cfg, "launch.srun.mem_per_gpu", 242144)))
    srun_job_name = str(_get(cfg, "launch.srun.job_name", "chitu"))
    chitu_debug = _bool01(_get(cfg, "runtime.chitu_debug", True))
    cuda_launch_blocking = _bool01(_get(cfg, "runtime.cuda_launch_blocking", False))
    model_name = str(_get(cfg, "model.name", "Wan2.1-T2V-1.3B"))
    model_ckpt_dir = str(_get(cfg, "model.ckpt_dir", ""))
    cfp = int(args.cfp if args.cfp is not None else _get(cfg, "parallel.cfp", 1))
    up = int(_get(cfg, "parallel.up", 8))
    attn_type = str(_get(cfg, "infer.attn_type", "torch_sdpa"))
    low_mem_level = int(_get(cfg, "infer.low_mem_level", 0))
    cp_backend = str(_get(cfg, "infer.cp_backend", "auto"))
    eval_reference_path = _get(cfg, "eval.reference_path", None)
    output_root_dir = str(_get(cfg, "output.root_dir", "outputs"))
    output_run_log = bool(_get(cfg, "output.run_log", True))
    output_memory = _bool_text(_get(cfg, "output.memory", True))
    output_timer = _bool_text(_get(cfg, "output.timer", False))
    launch_log_enabled = bool(_get(cfg, "launch.enable_launch_log", False))
    log_rank_text, log_rank_override = _normalize_log_ranks(_get(cfg, "output.log_ranks", [0]))
    eval_type_override = _normalize_eval_type(_get(cfg, "eval.eval_type", []))
    extra_overrides = _get(cfg, "overrides", [])

    if not isinstance(extra_overrides, list):
        raise ValueError("overrides must be a YAML list of dotlist override strings")
    if not model_ckpt_dir:
        print(f"Error: model.ckpt_dir must be configured in {config_path}", file=sys.stderr)
        return 1
    if not Path(model_ckpt_dir).is_dir():
        print(f"Error: model checkpoint directory does not exist: {model_ckpt_dir}", file=sys.stderr)
        return 1
    if cfp not in {1, 2}:
        print(f"Error: parallel.cfp must be 1 or 2, got: {cfp}", file=sys.stderr)
        return 1
    if num_nodes < 1 or gpus_per_node < 1:
        print("Error: launch.num_nodes and launch.gpus_per_node must be >= 1", file=sys.stderr)
        return 1

    total_gpus = num_nodes * gpus_per_node
    if total_gpus % cfp != 0:
        print(f"Error: total_gpus ({total_gpus}) must be divisible by cfp ({cfp})", file=sys.stderr)
        return 1
    cp_size = total_gpus // cfp
    if cp_size < 1:
        print(f"Error: computed cp_size is invalid: {cp_size}", file=sys.stderr)
        return 1

    script_path = PROJECT_ROOT / python_script
    if not script_path.is_file():
        print(f"Error: launch.python_script does not exist: {python_script}", file=sys.stderr)
        return 1

    runtime_python = _runtime_python()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = os.environ.get("CHITU_RUN_TASK_ID", "").strip() or _new_task_id()
    run_dir = Path(output_root_dir) / f"{_slugify(run_tag, 'run')}-{run_timestamp}-{_slugify(task_id, 'task')}"
    command_log = run_dir / "logs" / "command.log"

    env = os.environ.copy()
    env.update(
        {
            "CHITU_DEBUG": chitu_debug,
            "CHITU_RUN_TAG": run_tag,
            "CHITU_RUN_TIMESTAMP": run_timestamp,
            "CHITU_RUN_TASK_ID": task_id,
            "CHITU_RUN_DIR": str(run_dir),
            "CHITU_COMMAND_LOG": str(command_log),
            "CHITU_LOG_RANKS": log_rank_text,
            "CHITU_PYTHON_BIN": runtime_python,
            "CHITU_PROJECT_ROOT": str(PROJECT_ROOT),
            "SRUN_PARTITION": srun_partition,
            "SRUN_CPUS_PER_GPU": srun_cpus_per_gpu,
            "SRUN_MEM_PER_GPU": srun_mem_per_gpu,
            "SRUN_JOB_NAME": srun_job_name,
        }
    )
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if cuda_launch_blocking == "1":
        env["CUDA_LAUNCH_BLOCKING"] = "1"

    base_overrides = [
        f"models={model_name}",
        f"models.ckpt_dir={model_ckpt_dir}",
        f"infer.diffusion.cfg_size={cfp}",
        f"infer.diffusion.cp_size={cp_size}",
        f"infer.diffusion.up={up}",
        f"infer.attn_type={attn_type}",
        f"infer.diffusion.low_mem_level={low_mem_level}",
        f"infer.diffusion.cp_backend={cp_backend}",
        f"eval.eval_type={eval_type_override}",
        f"eval.reference_path={'null' if eval_reference_path is None else eval_reference_path}",
        f"output.root_dir={output_root_dir}",
        f"output.run_log={_bool_text(output_run_log)}",
        f"output.memory={output_memory}",
        f"output.timer={output_timer}",
        f"output.log_ranks={log_rank_override}",
    ]
    if launch_backend in {"srun", "slurm"}:
        launcher_script = PROJECT_ROOT / "script" / "srun_direct.sh"
    elif launch_backend in {"torchrun", "local"}:
        launcher_script = PROJECT_ROOT / "script" / "torchrun_direct.sh"
    else:
        print(f"Error: launch.backend must be one of srun/slurm/torchrun/local, got: {launch_backend}", file=sys.stderr)
        return 1

    cmd = [
        str(launcher_script),
        str(num_nodes),
        str(gpus_per_node),
        python_script,
        *base_overrides,
        *(str(item) for item in extra_overrides),
    ]

    handles: list[Any] = []
    try:
        if output_run_log:
            command_log.parent.mkdir(parents=True, exist_ok=True)
            handles.append(open(command_log, "a", encoding="utf-8", errors="replace"))

        launch_log = None
        if launch_log_enabled:
            output_root = Path(output_root_dir)
            output_root.mkdir(parents=True, exist_ok=True)
            launch_log = output_root / f"launch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            handles.append(open(launch_log, "a", encoding="utf-8", errors="replace"))

        summary = [
            "========== Launch Summary ==========",
            f"config_file: {config_path}",
            f"num_nodes: {num_nodes}",
            f"gpus_per_node: {gpus_per_node}",
            f"total_gpus: {total_gpus}",
            f"launch_backend: {launch_backend}",
            f"cfp(cfg-parallel): {cfp}",
            f"cp(context-parallel): {cp_size}",
            f"model: {model_name}",
            f"ckpt_dir: {model_ckpt_dir}",
            f"python_script: {python_script}",
            f"runtime_python: {runtime_python}",
            f"log_ranks: {log_rank_text}",
        ]
        if run_tag:
            summary.append(f"run_tag: {run_tag}")
        summary.append(f"log_file: {launch_log if launch_log else 'disabled (output.run_log=false)'}")
        summary.extend(["====================================", "Executing: " + " ".join(cmd)])
        for line in summary:
            _write_line(line, handles)

        return _stream_command(cmd, env, PROJECT_ROOT, handles)
    finally:
        for handle in handles:
            handle.close()


def serve_command(args: argparse.Namespace) -> int:
    from service_framework.server import main as service_main

    chitu_bin = shutil.which("chitu")
    cli_cmd = [chitu_bin] if chitu_bin else [sys.executable, "-m", "chitu_diffusion.cli"]
    os.environ.setdefault("CHITU_CLI_CMD", json.dumps(cli_cmd))
    return int(service_main(args) or 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chitu", description="ChituDiffusion command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="launch a configured ChituDiffusion job")
    run_parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG), help="runtime YAML config")
    run_parser.add_argument("--num-nodes", type=int, help="override launch.num_nodes")
    run_parser.add_argument("--gpus-per-node", type=int, help="override launch.gpus_per_node")
    run_parser.add_argument("--cfp", type=int, choices=[1, 2], help="override parallel.cfp")
    run_parser.set_defaults(func=run_command)

    serve_parser = subparsers.add_parser("serve", help="launch the ChituDiffusion HTTP UI")
    serve_parser.add_argument("--config", default="system_config.yaml", help="base system config")
    serve_parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host on the VSCode remote host")
    serve_parser.add_argument("--port", type=int, default=7860, help="HTTP bind port on the VSCode remote host")
    serve_parser.add_argument("--auto-port", action="store_true", help="try later ports if the requested port is busy")
    serve_parser.set_defaults(func=serve_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except BrokenPipeError:
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
