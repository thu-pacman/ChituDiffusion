#!/bin/bash

# 本地双GPU运行脚本（不使用srun调度器）
# Usage: ./run_local_2gpu.sh [num_gpus]
# 默认使用2张GPU

# 设置错误时退出
set -e

# 初始化变量
num_gpus=${1:-1}
script="./test/test_generate.py"

# 显示PYTHONPATH用于调试
echo "PYTHONPATH: $PYTHONPATH"

# 设置调试环境变量
export CHITU_DEBUG=1
# export CUDA_LAUNCH_BLOCKING=1  # 如需调试CUDA错误，取消注释此行

# 计算context parallel size（序列并行度，最小值为1）
cp_size=$((num_gpus/2))
if [ $cp_size -eq 0 ]; then
    cp_size=1
fi

# 模型配置 - 请根据您的实际情况修改模型路径
declare -A MODEL_CONFIGS=(
    ["Wan2.1-T2V-1.3B"]="/home/zlq/diffusion/Wan2.1-main/Wan2.1-T2V-1.3B"
    # 可以添加更多模型配置
    # ["Wan2.1-T2V-14B"]="/path/to/Wan2.1-T2V-14B"
    # ["Wan2.2-T2V-A14B"]="/path/to/Wan2.2-T2V-A14B"
)

# 选择模型函数
select_model() {
    echo "=========================================="
    echo "可用模型列表："
    local i=1
    for model in "${!MODEL_CONFIGS[@]}"; do
        echo "$i) $model"
        models[$i]=$model
        ((i++))
    done

    read -p "请选择模型 (1-${#MODEL_CONFIGS[@]}): " choice
    if [ -z "${models[$choice]}" ]; then
        echo "无效选择，使用默认模型: Wan2.1-T2V-1.3B"
        model="Wan2.1-T2V-1.3B"
        ckpt_dir=${MODEL_CONFIGS[$model]}
    else
        model=${models[$choice]}
        ckpt_dir=${MODEL_CONFIGS[$model]}
    fi
}

# 调用选择模型函数
select_model

# 检查模型路径是否存在
if [ ! -d "$ckpt_dir" ]; then
    echo "错误: 模型路径不存在: $ckpt_dir"
    echo "请检查 MODEL_CONFIGS 中的路径配置"
    exit 1
fi

# 设置分布式训练环境变量
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"29500"}
export NCCL_GRAPH_MIXING_SUPPORT=0
export NCCL_GRAPH_REGISTER=0

# 获取绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"

# 打印配置摘要
echo "=========================================="
echo "使用 torchrun 运行 Chitu Diffusion"
echo "GPU数量: $num_gpus"
echo "CP Size (序列并行度): $cp_size"
echo "模型: $model"
echo "模型路径: $ckpt_dir"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "项目根目录: $PROJECT_ROOT"
echo "=========================================="

# 设置PYTHONPATH
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$PROJECT_ROOT"
else
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
fi
echo "PYTHONPATH: $PYTHONPATH"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 检查测试文件是否存在
if [ ! -f "$script" ]; then
    echo "错误: 找不到测试文件: $PROJECT_ROOT/$script"
    exit 1
fi

# 检查GPU是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "警告: 未找到 nvidia-smi 命令，无法检查GPU状态"
else
    echo "=========================================="
    echo "GPU 状态:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    echo "=========================================="
fi

# 参数配置
basic_params="models=$model models.ckpt_dir=$ckpt_dir"
parallel_params="infer.diffusion.cp_size=$cp_size infer.diffusion.up_limit=2"

# 可选的低显存模式参数（如果显存不足，可以取消注释）
# magic_params="infer.diffusion.low_mem_level=1"

# FlexCache 参数（启用 TeaCache 加速）
# 注意：teacache_thresh 是在代码中自动设置的，不需要通过配置参数传递
flexcache_params="infer.diffusion.enable_flexcache=true"

# 构建并执行命令
echo "=========================================="
echo "执行命令:"
echo "torchrun --nnodes=1 --nproc-per-node=$num_gpus --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $script $basic_params $parallel_params $flexcache_params"
echo "=========================================="

# 执行命令
torchrun \
    --nnodes=1 \
    --nproc-per-node=$num_gpus \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $script \
    $basic_params \
    $parallel_params \
    $flexcache_params

