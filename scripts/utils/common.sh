#!/bin/bash

# Common utility functions for SmartDiffusion launch scripts

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print separator line
print_separator() {
    echo "=========================================="
}

# Get project root directory
get_project_root() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo "$(cd "$script_dir/../.." && pwd)"
}

# Setup Python path
setup_pythonpath() {
    local project_root="$1"
    if [ -z "$PYTHONPATH" ]; then
        export PYTHONPATH="$project_root"
    else
        export PYTHONPATH="$project_root:$PYTHONPATH"
    fi
    print_info "PYTHONPATH: $PYTHONPATH"
}

# Setup distributed training environment
setup_distributed_env() {
    export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
    export MASTER_PORT=${MASTER_PORT:-"29500"}
    export NCCL_GRAPH_MIXING_SUPPORT=0
    export NCCL_GRAPH_REGISTER=0
}

# Check if GPU is available
check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi not found, cannot check GPU status"
        return 1
    fi
    print_separator
    print_info "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    print_separator
    return 0
}

# Calculate context parallel size
calculate_cp_size() {
    local num_gpus=$1
    local cp_size=$((num_gpus/2))
    if [ $cp_size -eq 0 ]; then
        cp_size=1
    fi
    echo $cp_size
}

# Validate model path
validate_model_path() {
    local model_path="$1"
    if [ ! -d "$model_path" ]; then
        print_error "Model path does not exist: $model_path"
        return 1
    fi
    return 0
}

# Print configuration summary
print_config_summary() {
    local num_gpus=$1
    local cp_size=$2
    local model=$3
    local ckpt_dir=$4
    
    print_separator
    print_info "Configuration Summary"
    echo "GPU Count: $num_gpus"
    echo "CP Size (Context Parallel): $cp_size"
    echo "Model: $model"
    echo "Checkpoint Dir: $ckpt_dir"
    echo "MASTER_ADDR: $MASTER_ADDR"
    echo "MASTER_PORT: $MASTER_PORT"
    print_separator
}
