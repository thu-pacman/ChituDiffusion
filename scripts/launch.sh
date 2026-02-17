#!/bin/bash

# SmartDiffusion Unified Launcher
# This is the main entry point for running SmartDiffusion with different execution backends
#
# Usage: ./launch.sh <mode> [options]
#
# Modes:
#   local     - Run locally with torchrun (single or multi-GPU on single node)
#   cluster   - Run on cluster with srun (single or multi-node)
#   help      - Show detailed help
#
# Examples:
#   # Run locally with 2 GPUs
#   ./launch.sh local -g 2
#
#   # Run locally with specific model and FlexCache
#   ./launch.sh local -g 4 -m Wan2.1-T2V-14B --flexcache
#
#   # Run on cluster with 2 nodes, 8 GPUs per node
#   ./launch.sh cluster -n 2 -g 8 --multi-node
#
#   # Run with custom script
#   ./launch.sh local -g 2 -s ./test/custom_test.py

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
  local     Run locally with torchrun (single or multi-GPU on single node)
  cluster   Run on cluster with srun (single or multi-node)
  help      Show this help message

LOCAL MODE OPTIONS:
  -g, --gpus <num>        Number of GPUs to use (default: 2)
  -m, --model <name>      Model name (default: interactive selection)
  -c, --cp-size <num>     Context parallel size (default: auto-calculated)
  -s, --script <path>     Python script to run (default: ./test/test_generate.py)
  --cfg-parallel          Enable CFG parallel mode
  --flexcache             Enable FlexCache acceleration
  --low-mem               Enable low memory mode
  -h, --help              Show detailed help

CLUSTER MODE OPTIONS:
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
  # Run locally with 2 GPUs
  $0 local -g 2

  # Run locally with 4 GPUs and FlexCache
  $0 local -g 4 --flexcache

  # Run locally with specific model
  $0 local -g 2 -m Wan2.1-T2V-14B

  # Run on cluster with single node
  $0 cluster -g 8

  # Run on cluster with 2 nodes
  $0 cluster -n 2 -g 8 --multi-node

  # Run with custom partition
  $0 cluster -g 8 -p gpu_partition

  # Run with custom attention backend
  $0 cluster -g 8 --attn-type sparge

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
    local)
        print_info "Running in LOCAL mode with torchrun"
        exec "$SCRIPT_DIR/local/run_local.sh" "$@"
        ;;
    cluster)
        print_info "Running in CLUSTER mode with srun"
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
