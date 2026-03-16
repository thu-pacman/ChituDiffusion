#!/bin/bash

# SmartDiffusion Unified Launcher
# This is the main entry point for running SmartDiffusion with different execution backends
#
# Usage: ./launch.sh <mode> [options]
#
# Modes:
#   python    - Single-card execution with plain Python
#   torchrun  - Multi-card execution with torchrun (single node)
#   srun      - Cluster execution with srun (single or multi-node)
#   local     - Alias for torchrun (backward compatibility)
#   cluster   - Alias for srun (backward compatibility)
#   help      - Show detailed help
#
# Examples:
#   # Run on single GPU with Python
#   ./launch.sh python
#
#   # Run with 2 GPUs using torchrun
#   ./launch.sh torchrun -g 2
#
#   # Run on cluster with 8 GPUs
#   ./launch.sh srun -g 8
#
#   # Run with custom script
#   ./launch.sh torchrun -g 2 -s ./test/custom_test.py

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utilities for colored output
source "$SCRIPT_DIR/utils/common.sh"

# Function to show help
show_help() {
    cat << EOF
SmartDiffusion Unified Launcher
================================

Usage: $0 <mode> [options]

MODES:
  python    Single-card execution with plain Python
  torchrun  Multi-card execution with torchrun (single node)
  srun      Cluster execution with srun (single or multi-node)
  local     Alias for torchrun (backward compatibility)
  cluster   Alias for srun (backward compatibility)
  help      Show this help message

PYTHON MODE OPTIONS (Single-Card):
  -m, --model <name>      Model name (default: interactive selection)
  -s, --script <path>     Python script to run (default: ./test/test_generate.py)
  --flexcache             Enable FlexCache acceleration
  --low-mem               Enable low memory mode
  -h, --help              Show detailed help

TORCHRUN MODE OPTIONS (Multi-Card):
  -g, --gpus <num>        Number of GPUs to use (default: 2)
  -m, --model <name>      Model name (default: interactive selection)
  -c, --cp-size <num>     Context parallel size (default: auto-calculated)
  -s, --script <path>     Python script to run (default: ./test/test_generate.py)
  --cfg-parallel          Enable CFG parallel mode
  --flexcache             Enable FlexCache acceleration
  --low-mem               Enable low memory mode
  -h, --help              Show detailed help

SRUN MODE OPTIONS (Cluster):
  -n, --nodes <num>       Number of nodes (default: 1)
  -g, --gpus <num>        Number of GPUs per node (default: 2)
  -p, --partition <name>  SLURM partition name (default: a01)
  -m, --model <name>      Model name (default: interactive selection)
  -s, --script <path>     Python script to run (default: ./test/test_generate.py)
  --multi-node            Use multi-node launcher
  --flexcache             Enable FlexCache acceleration
  --low-mem               Enable low memory mode
  --attn-type <type>      Attention type (flash_attn, sage, sparge, auto)
  -h, --help              Show detailed help

EXAMPLES:
  # Run on single GPU with Python
  $0 python

  # Run on single GPU with specific model
  $0 python -m Wan2.1-T2V-1.3B

  # Run with 2 GPUs using torchrun
  $0 torchrun -g 2

  # Run with 4 GPUs and FlexCache
  $0 torchrun -g 4 --flexcache

  # Run on cluster with single node
  $0 srun -g 8

  # Run on cluster with 2 nodes
  $0 srun -n 2 -g 8 --multi-node

  # Run with custom partition
  $0 srun -g 8 -p gpu_partition

  # Run with custom attention backend
  $0 srun -g 8 --attn-type sparge

CONFIGURATION:
  Model paths and other configurations can be set in:
  - scripts/utils/config.sh
  
  You can also set environment variables:
  - MODEL_NAME: Specify model name
  - MASTER_ADDR: Master node address (default: 127.0.0.1)
  - MASTER_PORT: Master node port (default: 29500)

For more information, see the documentation in docs/ directory.
EOF
}

# Check if mode is provided
if [ $# -eq 0 ]; then
    print_error "No mode specified"
    echo ""
    show_help
    exit 1
fi

# Get mode
mode=$1
shift

# Handle mode
case $mode in
    python)
        print_info "Running in PYTHON mode (single-card)"
        exec "$SCRIPT_DIR/python/run_python.sh" "$@"
        ;;
    torchrun|local)
        if [ "$mode" = "local" ]; then
            print_warning "Note: 'local' mode is deprecated, use 'torchrun' instead"
        fi
        print_info "Running in TORCHRUN mode (multi-card)"
        exec "$SCRIPT_DIR/local/run_local.sh" "$@"
        ;;
    srun|cluster)
        if [ "$mode" = "cluster" ]; then
            print_warning "Note: 'cluster' mode is deprecated, use 'srun' instead"
        fi
        print_info "Running in SRUN mode (cluster)"
        exec "$SCRIPT_DIR/cluster/run_cluster.sh" "$@"
        ;;
    help|-h|--help)
        show_help
        exit 0
        ;;
    *)
        print_error "Unknown mode: $mode"
        echo ""
        show_help
        exit 1
        ;;
esac
