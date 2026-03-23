#!/bin/bash

set -euo pipefail

# Single launch entry for SmartDiffusion.
# Usage:
#   bash run.sh [config_yaml] [--num-nodes N] [--gpus-per-node N] [--cfp 1|2]
# Example:
#   bash run.sh system_config.yaml --gpus-per-node 8 --cfp 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
DEFAULT_CONFIG="$PROJECT_ROOT/system_config.yaml"
CONFIG_FILE="$DEFAULT_CONFIG"
CLI_NUM_NODES=""
CLI_GPUS_PER_NODE=""
CLI_CFP=""

while [ $# -gt 0 ]; do
    case "$1" in
        --num-nodes)
            CLI_NUM_NODES="$2"
            shift 2
            ;;
        --gpus-per-node)
            CLI_GPUS_PER_NODE="$2"
            shift 2
            ;;
        --cfp)
            CLI_CFP="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [config_yaml] [--num-nodes N] [--gpus-per-node N] [--cfp 1|2]"
            exit 0
            ;;
        -* )
            echo "Error: unknown option: $1"
            exit 1
            ;;
        *)
            if [ "$CONFIG_FILE" = "$DEFAULT_CONFIG" ]; then
                CONFIG_FILE="$1"
                shift
            else
                echo "Error: unexpected positional argument: $1"
                exit 1
            fi
            ;;
    esac
done

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE"
    echo "Usage: $0 [config_yaml]"
    exit 1
fi

PYTHON_BIN=""
if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Error: python/python3 is required to parse YAML config"
    exit 1
fi

eval "$("$PYTHON_BIN" - "$CONFIG_FILE" <<'PY'
import shlex
import sys

import yaml

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

def get(path, default=None):
    node = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node

values = {
    "RUN_TAG": str(get("launch.tag", "") or "").strip(),
    "NUM_NODES": int(get("launch.num_nodes", 1)),
    "GPUS_PER_NODE": int(get("launch.gpus_per_node", 1)),
    "PYTHON_SCRIPT": str(get("launch.python_script", "test/test_generate.py")),
    "SRUN_PARTITION": str(get("launch.srun.partition", "debug")),
    "SRUN_CPUS_PER_GPU": int(get("launch.srun.cpus_per_gpu", 24)),
    "SRUN_MEM_PER_GPU": int(get("launch.srun.mem_per_gpu", 242144)),
    "SRUN_JOB_NAME": str(get("launch.srun.job_name", "chitu")),
    "CHITU_DEBUG": bool(get("runtime.chitu_debug", True)),
    "CUDA_LAUNCH_BLOCKING": bool(get("runtime.cuda_launch_blocking", False)),
    "MODEL_NAME": str(get("model.name", "Wan2.1-T2V-1.3B")),
    "MODEL_CKPT_DIR": str(get("model.ckpt_dir", "")),
    "CFP": int(get("parallel.cfp", 1)),
    "UP_LIMIT": int(get("infer.up_limit", 8)),
    "ATTN_TYPE": str(get("infer.attn_type", "flash_attn")),
    "LOW_MEM_LEVEL": int(get("infer.low_mem_level", 0)),
    "ENABLE_FLEXCACHE": bool(get("infer.enable_flexcache", False)),
    "EVAL_REFERENCE_PATH": get("eval.reference_path", None),
    "OUTPUT_ROOT_DIR": str(get("output.root_dir", "outputs")),
    "OUTPUT_ENABLE_RUN_LOG": bool(get("output.enable_run_log", True)),
    "OUTPUT_ENABLE_TIMER_DUMP": bool(get("output.enable_timer_dump", False)),
}

raw_eval_type = get("eval.eval_type", [])
if raw_eval_type is None:
    eval_type_override = "[]"
elif isinstance(raw_eval_type, str):
    v = raw_eval_type.strip().lower()
    if v in {"", "none", "null"}:
        eval_type_override = "[]"
    elif "," in v:
        eval_type_override = "[" + ",".join(
            item.strip() for item in v.split(",") if item.strip()
        ) + "]"
    else:
        eval_type_override = f"[{v}]"
elif isinstance(raw_eval_type, (list, tuple)):
    eval_items = []
    for item in raw_eval_type:
        s = str(item).strip().lower()
        if not s or s in {"none", "null"}:
            continue
        eval_items.append(s)
    eval_type_override = "[" + ",".join(eval_items) + "]"
else:
    raise ValueError("eval.eval_type must be string/list/null")

extra_overrides = get("overrides", [])
if not isinstance(extra_overrides, list):
    raise ValueError("overrides must be a YAML list of Hydra override strings")

for key, value in values.items():
    if isinstance(value, bool):
        out = "true" if value else "false"
    elif value is None:
        out = "null"
    else:
        out = str(value)
    print(f"{key}={shlex.quote(out)}")

print(f"EVAL_TYPE_OVERRIDE={shlex.quote(eval_type_override)}")

print("EXTRA_OVERRIDES=" + shlex.quote(" ".join(str(x) for x in extra_overrides)))
PY
)"

if [ -n "$CLI_NUM_NODES" ]; then
    NUM_NODES="$CLI_NUM_NODES"
fi
if [ -n "$CLI_GPUS_PER_NODE" ]; then
    GPUS_PER_NODE="$CLI_GPUS_PER_NODE"
fi
if [ -n "$CLI_CFP" ]; then
    CFP="$CLI_CFP"
fi

if [ -z "$MODEL_CKPT_DIR" ]; then
    echo "Error: model.ckpt_dir must be configured in $CONFIG_FILE"
    exit 1
fi

if [ ! -d "$MODEL_CKPT_DIR" ]; then
    echo "Error: model checkpoint directory does not exist: $MODEL_CKPT_DIR"
    exit 1
fi

if [ "$CFP" -ne 1 ] && [ "$CFP" -ne 2 ]; then
    echo "Error: parallel.cfp must be 1 or 2, got: $CFP"
    exit 1
fi

if [ "$NUM_NODES" -lt 1 ] || [ "$GPUS_PER_NODE" -lt 1 ]; then
    echo "Error: launch.num_nodes and launch.gpus_per_node must be >= 1"
    exit 1
fi

TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
if [ $((TOTAL_GPUS % CFP)) -ne 0 ]; then
    echo "Error: total_gpus ($TOTAL_GPUS) must be divisible by cfp ($CFP)"
    exit 1
fi
CP_SIZE=$((TOTAL_GPUS / CFP))

if [ "$CP_SIZE" -lt 1 ]; then
    echo "Error: computed cp_size is invalid: $CP_SIZE"
    exit 1
fi

# Normalize YAML booleans to 1/0 for consistent env parsing in Python.
to_env_bool01() {
    case "${1,,}" in
        1|true|yes|on)
            echo 1
            ;;
        *)
            echo 0
            ;;
    esac
}

CHITU_DEBUG="$(to_env_bool01 "$CHITU_DEBUG")"
CUDA_LAUNCH_BLOCKING="$(to_env_bool01 "$CUDA_LAUNCH_BLOCKING")"

export CHITU_DEBUG
export HYDRA_FULL_ERROR=1
export CHITU_RUN_TAG="$RUN_TAG"
if [ "$CUDA_LAUNCH_BLOCKING" = "1" ]; then
    export CUDA_LAUNCH_BLOCKING=1
fi

# Resource settings are centralized here and consumed by script/srun_direct.sh.
export SRUN_PARTITION
export SRUN_CPUS_PER_GPU
export SRUN_MEM_PER_GPU
export SRUN_JOB_NAME

cd "$PROJECT_ROOT"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: launch.python_script does not exist: $PYTHON_SCRIPT"
    exit 1
fi

DATE="$(date +%Y%m%d_%H%M%S)"
LOG_FILE=""
if [ "${OUTPUT_ENABLE_RUN_LOG,,}" = "true" ]; then
    mkdir -p "$OUTPUT_ROOT_DIR"
    LOG_FILE="$OUTPUT_ROOT_DIR/launch_${DATE}.log"
fi

BASE_OVERRIDES=(
    "models=$MODEL_NAME"
    "models.ckpt_dir=$MODEL_CKPT_DIR"
    "infer.diffusion.cfg_size=$CFP"
    "infer.diffusion.cp_size=$CP_SIZE"
    "infer.diffusion.up_limit=$UP_LIMIT"
    "infer.attn_type=$ATTN_TYPE"
    "infer.diffusion.low_mem_level=$LOW_MEM_LEVEL"
    "infer.diffusion.enable_flexcache=$ENABLE_FLEXCACHE"
    "eval.eval_type=$EVAL_TYPE_OVERRIDE"
    "eval.reference_path=$EVAL_REFERENCE_PATH"
    "output.root_dir=$OUTPUT_ROOT_DIR"
    "output.enable_run_log=$OUTPUT_ENABLE_RUN_LOG"
    "output.enable_timer_dump=$OUTPUT_ENABLE_TIMER_DUMP"
)

if [ -n "${EXTRA_OVERRIDES:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARRAY=($EXTRA_OVERRIDES)
else
    EXTRA_ARRAY=()
fi

CMD=(
    "$PROJECT_ROOT/script/srun_direct.sh"
    "$NUM_NODES"
    "$GPUS_PER_NODE"
    "$PYTHON_SCRIPT"
    "${BASE_OVERRIDES[@]}"
    "${EXTRA_ARRAY[@]}"
)

echo "========== Launch Summary =========="
echo "config_file: $CONFIG_FILE"
echo "num_nodes: $NUM_NODES"
echo "gpus_per_node: $GPUS_PER_NODE"
echo "total_gpus: $TOTAL_GPUS"
echo "cfp(cfg-parallel): $CFP"
echo "cp(context-parallel): $CP_SIZE"
echo "model: $MODEL_NAME"
echo "ckpt_dir: $MODEL_CKPT_DIR"
echo "python_script: $PYTHON_SCRIPT"
if [ -n "$RUN_TAG" ]; then
    echo "run_tag: $RUN_TAG"
fi
if [ -n "$LOG_FILE" ]; then
    echo "log_file: $LOG_FILE"
else
    echo "log_file: disabled (output.enable_run_log=false)"
fi
echo "===================================="
echo "Executing: ${CMD[*]}"

if [ -n "$LOG_FILE" ]; then
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
else
    "${CMD[@]}"
fi