#!/bin/bash

# Usage: ./script.sh <num_gpus>
# 用法：./script.sh <GPU数量>

# Check if number of GPUs is provided
# 检查是否提供了GPU数量参数
if [ -z "$1" ]; then
    echo "Error: Please provide number of GPUs"
    echo "Usage: $0 <num_gpus>"
    exit 1
fi

# Initialize variables
# 初始化变量
num_gpus=$1
script="./test/test_generate.py"

# Show PYTHONPATH for debugging
# 显示PYTHONPATH用于调试
echo "PYTHONPATH: $PYTHONPATH"

# Set debug environment variables
# 设置调试环境变量
export CHITU_DEBUG=1
# export CUDA_LAUNCH_BLOCKING=1

# Calculate context parallel size (minimum 1)

# Model configurations
# 模型配置
declare -A MODEL_CONFIGS=(
    ["Wan2.1-T2V-1.3B"]="/home/zhongrx/cyy/Wan2.1/Wan2.1-T2V-1.3B"
    ["Wan2.1-T2V-14B"]="/home/zhongrx/cyy/Wan2.1/Wan2.1-T2V-14B"
    ["Wan2.2-T2V-A14B"]="/home/zhongrx/cyy/model/Wan22-t2v-a14b"
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

# Parameter configurations
# 参数配置

# 基本参数
basic_params="models=$model models.ckpt_dir=$ckpt_dir"

# 并行参数（根据GPU数自动设置）
enable_cfg=true
if [ "$enable_cfg" = "true" ]; then
    cp_size=$((num_gpus / 2))
else
    cp_size=$num_gpus
fi
parallel_params="infer.diffusion.enable_cfg_parallel=$enable_cfg \
                infer.diffusion.cp_size=$cp_size \
                infer.diffusion.up_limit=1"

# 魔法参数！
magic_params="infer.attn_type='flash_attn' \
            infer.diffusion.low_mem_level=0 \
            infer.diffusion.enable_flexcache=true"

# Build and execute command
command="./script/srun_direct.sh 1 $num_gpus $script \
    $basic_params \
    $parallel_params \
    $magic_params"

# Print command for debugging
echo "Executing command: $command"
echo "执行命令：$command"

# Execute command
eval $command