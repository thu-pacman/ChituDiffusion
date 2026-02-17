#!/bin/bash

# Unified local runner for SmartDiffusion using torchrun
# Usage: ./run_local.sh [options]
# Options:
#   -g, --gpus <num>        Number of GPUs to use (default: 2)
#   -m, --model <name>      Model name (default: interactive selection)
#   -c, --cp-size <num>     Context parallel size (default: auto-calculated)
#   -s, --script <path>     Python script to run (default: ./test/test_generate.py)
#   --cfg-parallel          Enable CFG parallel mode (sets cfg_size=1)
#   --flexcache             Enable FlexCache acceleration
#   --low-mem               Enable low memory mode
#   -h, --help              Show this help message

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source utility scripts
source "$SCRIPT_DIR/../utils/common.sh"
source "$SCRIPT_DIR/../utils/config.sh"

# Default values
num_gpus=2
script="$DEFAULT_TEST_SCRIPT"
cp_size=""
cfg_parallel=false
enable_flexcache=false
low_mem=false
model_name=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpus)
            num_gpus="$2"
            shift 2
            ;;
        -m|--model)
            model_name="$2"
            shift 2
            ;;
        -c|--cp-size)
            cp_size="$2"
            shift 2
            ;;
        -s|--script)
            script="$2"
            shift 2
            ;;
        --cfg-parallel)
            cfg_parallel=true
            shift
            ;;
        --flexcache)
            enable_flexcache=true
            shift
            ;;
        --low-mem)
            low_mem=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -g, --gpus <num>        Number of GPUs to use (default: 2)"
            echo "  -m, --model <name>      Model name (default: interactive selection)"
            echo "  -c, --cp-size <num>     Context parallel size (default: auto-calculated)"
            echo "  -s, --script <path>     Python script to run (default: ./test/test_generate.py)"
            echo "  --cfg-parallel          Enable CFG parallel mode"
            echo "  --flexcache             Enable FlexCache acceleration"
            echo "  --low-mem               Enable low memory mode"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup environment
print_info "Setting up environment..."
setup_debug_env
setup_distributed_env
setup_pythonpath "$PROJECT_ROOT"

# Change to project root
cd "$PROJECT_ROOT"

# Calculate CP size if not specified
if [ -z "$cp_size" ]; then
    cp_size=$(calculate_cp_size $num_gpus)
fi

# Model selection
if [ -n "$model_name" ]; then
    if ! set_model_from_env "$model_name"; then
        print_error "Failed to set model: $model_name"
        exit 1
    fi
else
    select_model
fi

# Validate model path
if ! validate_model_path "$ckpt_dir"; then
    print_error "Please check MODEL_CONFIGS in scripts/utils/config.sh"
    exit 1
fi

# Check if test script exists
if [ ! -f "$script" ]; then
    print_error "Test script not found: $PROJECT_ROOT/$script"
    exit 1
fi

# Check GPU availability
check_gpu

# Print configuration
print_config_summary $num_gpus $cp_size "$model" "$ckpt_dir"

# Build parameters
basic_params="models=$model models.ckpt_dir=$ckpt_dir"
parallel_params="infer.diffusion.cp_size=$cp_size infer.diffusion.up_limit=2"

# Add CFG parallel parameters if enabled
if [ "$cfg_parallel" = true ]; then
    parallel_params="$parallel_params +infer.diffusion.cfg_size=1"
    print_info "CFG parallel mode enabled"
fi

# Add FlexCache parameters if enabled
flexcache_params=""
if [ "$enable_flexcache" = true ]; then
    flexcache_params="infer.diffusion.enable_flexcache=true"
    print_info "FlexCache acceleration enabled"
fi

# Add low memory mode if enabled
low_mem_params=""
if [ "$low_mem" = true ]; then
    low_mem_params="infer.diffusion.low_mem_level=1"
    print_info "Low memory mode enabled"
fi

# Build command
print_separator
print_info "Running with torchrun..."
echo "Command: torchrun --nnodes=1 --nproc-per-node=$num_gpus --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $script $basic_params $parallel_params $flexcache_params $low_mem_params"
print_separator

# Execute torchrun
torchrun \
    --nnodes=1 \
    --nproc-per-node=$num_gpus \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $script \
    $basic_params \
    $parallel_params \
    $flexcache_params \
    $low_mem_params
