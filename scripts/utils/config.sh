#!/bin/bash

# Model configuration for SmartDiffusion
# Modify these paths according to your environment

# Default model configurations
declare -A DEFAULT_MODEL_CONFIGS=(
    ["Wan2.1-T2V-1.3B"]="/path/to/Wan2.1-T2V-1.3B"
    ["Wan2.1-T2V-14B"]="/path/to/Wan2.1-T2V-14B"
    ["Wan2.2-T2V-A14B"]="/path/to/Wan2.2-T2V-A14B"
)

# Function to select a model interactively
select_model() {
    # Use environment variable MODEL_CONFIGS if set, otherwise use defaults
    if [ ${#MODEL_CONFIGS[@]} -eq 0 ]; then
        for key in "${!DEFAULT_MODEL_CONFIGS[@]}"; do
            MODEL_CONFIGS[$key]="${DEFAULT_MODEL_CONFIGS[$key]}"
        done
    fi
    
    echo "=========================================="
    echo "Available Models:"
    local i=1
    local models=()
    for model_name in "${!MODEL_CONFIGS[@]}"; do
        echo "$i) $model_name"
        models[$i]=$model_name
        ((i++))
    done
    
    read -p "Select a model (1-${#MODEL_CONFIGS[@]}): " choice
    if [ -z "${models[$choice]}" ]; then
        echo "Invalid selection. Using default model: Wan2.1-T2V-1.3B"
        model="Wan2.1-T2V-1.3B"
        ckpt_dir="${MODEL_CONFIGS[$model]}"
    else
        model="${models[$choice]}"
        ckpt_dir="${MODEL_CONFIGS[$model]}"
    fi
    
    echo "Selected model: $model"
    echo "Checkpoint directory: $ckpt_dir"
    echo "=========================================="
}

# Function to set model from environment or arguments
set_model_from_env() {
    local model_name="${1:-$MODEL_NAME}"
    
    if [ -n "$model_name" ]; then
        if [ -n "${MODEL_CONFIGS[$model_name]}" ] || [ -n "${DEFAULT_MODEL_CONFIGS[$model_name]}" ]; then
            model="$model_name"
            ckpt_dir="${MODEL_CONFIGS[$model_name]:-${DEFAULT_MODEL_CONFIGS[$model_name]}}"
            echo "Using model from environment/argument: $model"
            echo "Checkpoint directory: $ckpt_dir"
            return 0
        else
            echo "Warning: Model '$model_name' not found in configurations"
            return 1
        fi
    fi
    return 1
}

# Default test script location
DEFAULT_TEST_SCRIPT="./test/test_generate.py"

# Default debug settings
setup_debug_env() {
    export CHITU_DEBUG=1
    # Uncomment if you need to debug CUDA errors
    # export CUDA_LAUNCH_BLOCKING=1
}
