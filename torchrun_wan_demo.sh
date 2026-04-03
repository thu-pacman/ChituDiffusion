#!/bin/bash

# Usage: ./script.sh [num_gpus]
# 用法：./script.sh [GPU数量]
# Default is 2 GPUs if not specified
# 如果未指定，默认使用2个GPU

# Initialize variables
# 初始化变量
num_gpus=${1:-2}
script="./test/test_generate.py"

# Show PYTHONPATH for debugging
# 显示PYTHONPATH用于调试
echo "PYTHONPATH: $PYTHONPATH"

# Set debug environment variables
# 设置调试环境变量
export CHITU_DEBUG=1
# export CUDA_LAUNCH_BLOCKING=1

# Calculate context parallel size (minimum 1)
# 计算序列并行度（最小值为1）
cp_size=$((num_gpus/2))
if [ $cp_size -eq 0 ]; then
    cp_size=1
fi

# Model configurations
# 模型配置
declare -A MODEL_CONFIGS=(
    ["Wan2.1-T2V-1.3B"]="/home/dataset/Wan2.1-T2V-1.3B"
    # Add more models here
    # 在此处添加更多模型
)

# Select model
# 选择模型
select_model() {
    echo "Available models:"
    local i=1
    for model in "${!MODEL_CONFIGS[@]}"; do
        echo "$i) $model"
        models[$i]=$model
        ((i++))
    done

    read -p "Select a model (1-${#MODEL_CONFIGS[@]}): " choice
    if [ -z "${models[$choice]}" ]; then
        echo "Invalid selection. Using default model: Wan2.1-T2V-1.3B"
        model="Wan2.1-T2V-1.3B"
        ckpt_dir=${MODEL_CONFIGS[$model]}
    else
        model=${models[$choice]}
        ckpt_dir=${MODEL_CONFIGS[$model]}
    fi
}

# Call select_model function
# 调用选择模型函数
select_model

# Set distributed training environment variables
# 设置分布式训练环境变量
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"29500"}
export NCCL_GRAPH_MIXING_SUPPORT=0
export NCCL_GRAPH_REGISTER=0

# Get absolute paths
# 获取绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

# Print configuration summary
# 打印配置摘要
echo "=========================================="
echo "Running Chitu Diffusion with torchrun"
echo "Number of GPUs: $num_gpus"
echo "CP Size: $cp_size"
echo "Model: $model"
echo "Checkpoint Dir: $ckpt_dir"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "=========================================="

# Set PYTHONPATH
# 设置PYTHONPATH
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$PROJECT_ROOT"
else
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
fi
echo "PYTHONPATH set to: $PYTHONPATH"

# Change to project root directory
# 切换到项目根目录
cd "$PROJECT_ROOT"

# Check if test file exists
# 检查测试文件是否存在
if [ ! -f "./test/test_generate.py" ]; then
    echo "ERROR: Cannot find test_generate.py at $PROJECT_ROOT/test/test_generate.py"
    exit 1
fi

# Parameter configurations
# 参数配置
basic_params="models=$model models.ckpt_dir=$ckpt_dir"
parallel_params="infer.diffusion.fpp_size=$num_gpus infer.diffusion.cp_size=1 infer.diffusion.up_limit=8"
eval_params="eval.eval_type=vbench"
# 魔法参数！
magic_params="
            infer.attn_type='sparge' \
            infer.diffusion.low_mem_level=1 \
            infer.diffusion.enable_flexcache=true"
# Build and execute command
# 构建并执行命令
command="torchrun \
    --nnodes=1 \
    --nproc-per-node=$num_gpus \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $script \
    $basic_params \
    $parallel_params \
    $magic_params \
    $eval_params"


# Print command for debugging
# 打印命令用于调试
echo "Executing command: $command"

# Execute command
# 执行命令
eval $command