#!/bin/bash

# 设置 PyTorch 分布式所需的环境变量
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(scontrol show jobid=$SLURM_JOB_ID | tr '=' ' ' | grep BatchHost | awk '{print $2}')

echo "Task $RANK: RANK=$RANK, LOCAL_RANK=$LOCAL_RANK, WORLD_SIZE=$WORLD_SIZE, MASTER_ADDR=$MASTER_ADDR"

# 执行实际的 Python 脚本
exec python "$@"