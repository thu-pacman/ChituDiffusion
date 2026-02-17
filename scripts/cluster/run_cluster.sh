#!/bin/bash

# Unified srun launcher for SmartDiffusion cluster execution
# Usage: ./run_cluster.sh [options] [-- script_args...]
# Options:
#   -n, --nodes <num>       Number of nodes (default: 1)
#   -g, --gpus <num>        Number of GPUs per node (default: 2)
#   -p, --partition <name>  SLURM partition name (default: a01)
#   -m, --model <name>      Model name (default: interactive selection)
#   -s, --script <path>     Python script to run (default: ./test/test_generate.py)
#   --multi-node            Use multi-node launcher (srun_multi_node.sh)
#   --flexcache             Enable FlexCache acceleration
#   --low-mem               Enable low memory mode
#   --attn-type <type>      Attention type (flash_attn, sage, sparge, auto)
#   -h, --help              Show this help message

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source utility scripts
source "$SCRIPT_DIR/../utils/common.sh"
source "$SCRIPT_DIR/../utils/config.sh"

# Default values
num_nodes=1
num_gpus=2
partition="a01"
script="$DEFAULT_TEST_SCRIPT"
model_name=""
use_multi_node=false
enable_flexcache=false
low_mem=false
attn_type=""
script_args=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--nodes)
            num_nodes="$2"
            shift 2
            ;;
        -g|--gpus)
            num_gpus="$2"
            shift 2
            ;;
        -p|--partition)
            partition="$2"
            shift 2
            ;;
        -m|--model)
            model_name="$2"
            shift 2
            ;;
        -s|--script)
            script="$2"
            shift 2
            ;;
        --multi-node)
            use_multi_node=true
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
        --attn-type)
            attn_type="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options] [-- script_args...]"
            echo "Options:"
            echo "  -n, --nodes <num>       Number of nodes (default: 1)"
            echo "  -g, --gpus <num>        Number of GPUs per node (default: 2)"
            echo "  -p, --partition <name>  SLURM partition name (default: a01)"
            echo "  -m, --model <name>      Model name (default: interactive selection)"
            echo "  -s, --script <path>     Python script to run (default: ./test/test_generate.py)"
            echo "  --multi-node            Use multi-node launcher"
            echo "  --flexcache             Enable FlexCache acceleration"
            echo "  --low-mem               Enable low memory mode"
            echo "  --attn-type <type>      Attention type (flash_attn, sage, sparge, auto)"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        --)
            shift
            script_args=("$@")
            break
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup environment
print_info "Setting up cluster environment..."
setup_debug_env

# Model selection
if [ -n "$model_name" ]; then
    if ! set_model_from_env "$model_name"; then
        print_error "Failed to set model: $model_name"
        exit 1
    fi
else
    select_model
fi

# Calculate CP size
cp_size=$(calculate_cp_size $num_gpus)

# Build parameters
basic_params="models=$model models.ckpt_dir=$ckpt_dir"
parallel_params="infer.diffusion.cp_size=$cp_size infer.diffusion.up_limit=8"

# Magic parameters
magic_params=""
if [ -n "$attn_type" ]; then
    magic_params="infer.attn_type='$attn_type'"
    print_info "Attention type: $attn_type"
fi

if [ "$low_mem" = true ]; then
    magic_params="$magic_params infer.diffusion.low_mem_level=1"
    print_info "Low memory mode enabled"
fi

if [ "$enable_flexcache" = true ]; then
    magic_params="$magic_params infer.diffusion.enable_flexcache=true"
    print_info "FlexCache acceleration enabled"
fi

# Additional script arguments
extra_args="${script_args[@]}"

# Print configuration
print_separator
print_info "Cluster Configuration"
echo "Nodes: $num_nodes"
echo "GPUs per node: $num_gpus"
echo "Partition: $partition"
echo "Model: $model"
echo "Checkpoint Dir: $ckpt_dir"
echo "Script: $script"
echo "Multi-node mode: $use_multi_node"
print_separator

# Choose the appropriate srun launcher
if [ "$use_multi_node" = true ]; then
    srun_launcher="$SCRIPT_DIR/srun_multi_node.sh"
    print_info "Using multi-node launcher"
else
    srun_launcher="$SCRIPT_DIR/srun_direct.sh"
    print_info "Using direct launcher"
fi

# Export partition for srun scripts to use
export SLURM_PARTITION="$partition"

# Build and execute command
command="$srun_launcher $num_nodes $num_gpus $script \
    $basic_params \
    $parallel_params \
    $magic_params \
    $extra_args"

print_separator
print_info "Executing command:"
echo "$command"
print_separator

# Execute command
eval $command
