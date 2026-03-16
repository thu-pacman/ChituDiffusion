#!/bin/bash

# Single-card Python runner for SmartDiffusion
# Usage: ./run_python.sh [options]
# Options:
#   -m, --model <name>      Model name (default: interactive selection)
#   -s, --script <path>     Python script to run (default: ./test/test_generate.py)
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
script="$DEFAULT_TEST_SCRIPT"
enable_flexcache=false
low_mem=false
model_name=""
script_args=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            model_name="$2"
            shift 2
            ;;
        -s|--script)
            script="$2"
            shift 2
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
            echo "  -m, --model <name>      Model name (default: interactive selection)"
            echo "  -s, --script <path>     Python script to run (default: ./test/test_generate.py)"
            echo "  --flexcache             Enable FlexCache acceleration"
            echo "  --low-mem               Enable low memory mode"
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
print_info "Setting up Python environment for single-card execution..."
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

# Build parameters
basic_params="models=$model models.ckpt_dir=$ckpt_dir"

# Magic parameters
magic_params=""
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
print_info "Single-Card Python Configuration"
echo "Model: $model"
echo "Checkpoint Dir: $ckpt_dir"
echo "Script: $script"
echo "Device: Single GPU (cuda:0)"
print_separator

# Build command
command="python $script \
    $basic_params \
    $magic_params \
    $extra_args"

print_separator
print_info "Executing command:"
echo "$command"
print_separator

# Change to project root and execute
cd "$PROJECT_ROOT"
eval $command
