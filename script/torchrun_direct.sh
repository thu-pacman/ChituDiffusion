#!/bin/bash

set -euo pipefail
set -x

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <num_nodes> <num_gpus_per_node> <script> [script_args...]"
    exit 1
fi

NODES=$1
NUM_GPUS=$2
SCRIPT=$3
SCRIPT_ARGS=("${@:4}")

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-$(expr $RANDOM % 10000 + 50000)}
export NODE_RANK=${NODE_RANK:-0}

# Keep NCCL defaults aligned with the Slurm launcher.
export NCCL_GRAPH_MIXING_SUPPORT=${NCCL_GRAPH_MIXING_SUPPORT:-0}
export NCCL_GRAPH_REGISTER=${NCCL_GRAPH_REGISTER:-0}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}
export TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN=${TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN:-0}

if [ -n "${CHITU_PROJECT_ROOT:-}" ]; then
    cd "$CHITU_PROJECT_ROOT"
    case ":${PYTHONPATH:-}:" in
        *":$CHITU_PROJECT_ROOT:"*)
            ;;
        *)
            export PYTHONPATH="$CHITU_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
            ;;
    esac
fi

echo "Running torchrun with $NODES nodes, $NUM_GPUS GPUs per node"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NODE_RANK: $NODE_RANK"

RUNTIME_PYTHON="${CHITU_PYTHON_BIN:-python}"
if [[ "$NODES" -eq 1 ]]; then
    exec "$RUNTIME_PYTHON" -m torch.distributed.run \
        --standalone \
        --nnodes=1 \
        --nproc-per-node="$NUM_GPUS" \
        "$SCRIPT" "${SCRIPT_ARGS[@]}"
fi

exec "$RUNTIME_PYTHON" -m torch.distributed.run \
    --nnodes="$NODES" \
    --nproc-per-node="$NUM_GPUS" \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    "$SCRIPT" "${SCRIPT_ARGS[@]}"
