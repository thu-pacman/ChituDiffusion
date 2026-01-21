#!/bin/bash

set -x 

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <num_nodes> <num_gpus_per_node> <script> [script_args...]"
    exit 1
fi

NODES=$1
NUM_GPUS=$2
SCRIPT=$3
SCRIPT_ARGS=("${@:4}")

JOB_NAME=$USER-chitu
CPUS_PER_GPU=24
MEM_PER_GPU=242144

# 计算资源
NUM_CPUS=$((NUM_GPUS * CPUS_PER_GPU))
NUM_MEMS=$((NUM_GPUS * MEM_PER_GPU))

# 设置 master port
export MASTER_PORT=$(expr $RANDOM % 10000 + 50000)

# 优化 NCCL
export NCCL_GRAPH_MIXING_SUPPORT=0
export NCCL_GRAPH_REGISTER=0

# 获取当前脚本所在目录
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "Running with $NODES nodes, $NUM_GPUS GPUs per node"
echo "MASTER_PORT: $MASTER_PORT"

# 使用 wrapper 脚本
srun --job-name $JOB_NAME \
     --nodes $NODES \
     --ntasks-per-node $NUM_GPUS \
     --cpus-per-task $CPUS_PER_GPU \
     --mem $NUM_MEMS \
     --gres=gpu:$NUM_GPUS \
     --export=ALL \
     bash $SCRIPT_DIR/srun_wrapper.sh $SCRIPT "${SCRIPT_ARGS[@]}"