#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
VENV_PATH=${VENV_PATH:-"$PROJECT_ROOT/.venv"}

if [[ -f "$VENV_PATH/bin/activate" ]]; then
    # Ensure cluster workers use the same project environment.
    source "$VENV_PATH/bin/activate"
fi

# 设置 PyTorch 分布式所需的环境变量
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')

echo "Task $RANK: RANK=$RANK, LOCAL_RANK=$LOCAL_RANK, WORLD_SIZE=$WORLD_SIZE, MASTER_ADDR=$MASTER_ADDR"

# 执行实际的 Python 脚本
RUNTIME_PYTHON="${CHITU_PYTHON_BIN:-python}"
echo "Task $RANK: using python=$RUNTIME_PYTHON"
exec "$RUNTIME_PYTHON" "$@"
