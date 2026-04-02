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

PARTITION=${SRUN_PARTITION:-debug}
CPUS_PER_GPU=${SRUN_CPUS_PER_GPU:-24}
MEM_PER_GPU=${SRUN_MEM_PER_GPU:-242144}
DEFAULT_JOB_NAME="${USER:-user}-chitu"
JOB_NAME=${SRUN_JOB_NAME:-$DEFAULT_JOB_NAME}

if [ -n "${USER:-}" ] && [[ "$JOB_NAME" != *"$USER"* ]]; then
    JOB_NAME="$USER-$JOB_NAME"
fi

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
echo "Partition: $PARTITION"
echo "Job name: $JOB_NAME"
echo "MASTER_PORT: $MASTER_PORT"

# 若从 login 或过期 salloc 运行，清除旧 SLURM 变量，让 srun 申请新分配，避免 "Invalid job id"
unset SLURM_JOB_ID SLURM_STEP_ID SLURM_NTASKS SLURM_NTASKS_PER_NODE 2>/dev/null || true

# 使用 wrapper 脚本
srun -p "$PARTITION" \
     --job-name $JOB_NAME \
     --nodes $NODES \
     --ntasks-per-node $NUM_GPUS \
     --cpus-per-task $CPUS_PER_GPU \
     --mem $NUM_MEMS \
     --gres=gpu:$NUM_GPUS \
     --export=ALL \
     bash $SCRIPT_DIR/srun_wrapper.sh $SCRIPT "${SCRIPT_ARGS[@]}"