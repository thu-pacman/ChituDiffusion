#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CHITU_PROJECT_ROOT=${CHITU_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}

# 设置 PyTorch 分布式所需的环境变量
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(
    scontrol show jobid=$SLURM_JOB_ID 2>/dev/null \
        | tr '=' ' ' \
        | grep BatchHost \
        | awk '{print $2}'
)
if [ -z "$MASTER_ADDR" ]; then
    MASTER_ADDR="${SLURM_LAUNCH_NODE_IPADDR:-127.0.0.1}"
    export MASTER_ADDR
fi

echo "Task $RANK: RANK=$RANK, LOCAL_RANK=$LOCAL_RANK, WORLD_SIZE=$WORLD_SIZE, MASTER_ADDR=$MASTER_ADDR"

cd "$CHITU_PROJECT_ROOT"

# Prefer the repository environment and isolate it from notebook/login Python
# paths. Extra source trees remain available through an explicit opt-in.
if [ -n "${CHITU_PYTHON_BIN:-}" ]; then
    RUNTIME_PYTHON="$CHITU_PYTHON_BIN"
elif [ -x "$CHITU_PROJECT_ROOT/.venv/bin/python" ]; then
    RUNTIME_PYTHON="$CHITU_PROJECT_ROOT/.venv/bin/python"
else
    RUNTIME_PYTHON="python"
fi
if [[ "$RUNTIME_PYTHON" == "$CHITU_PROJECT_ROOT/.venv/bin/python" ]]; then
    export PYTHONPATH="$CHITU_PROJECT_ROOT${CHITU_EXTRA_PYTHONPATH:+:$CHITU_EXTRA_PYTHONPATH}"
else
    case ":${PYTHONPATH:-}:" in
        *":$CHITU_PROJECT_ROOT:"*) ;;
        *) export PYTHONPATH="$CHITU_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
    esac
fi
echo "Task $RANK: using python=$RUNTIME_PYTHON"
exec "$RUNTIME_PYTHON" "$@"
