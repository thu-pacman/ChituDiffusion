# SmartDiffusion Launch Scripts

This directory contains all the launch scripts and utilities for running SmartDiffusion with different execution modes.

## Table of Contents

- [Quick Start](#quick-start)
- [Execution Modes](#execution-modes)
- [Directory Structure](#directory-structure)
- [Usage Guide](#usage-guide)
- [System Parameters](#system-parameters)
- [Configuration](#configuration)
- [Migration Guide](#migration-guide)

## Quick Start

SmartDiffusion supports 3 execution modes:

```bash
# Single GPU with Python
./scripts/launch.sh python

# Multi-GPU with torchrun (2-8 GPUs on single node)
./scripts/launch.sh torchrun -g 4

# Cluster with srun (multi-node distributed)
./scripts/launch.sh srun -g 8 -p a01
```

## Execution Modes

### 1. Python Mode (Single-Card)

**When to use:**
- Single GPU execution
- Quick testing and debugging
- Small models that fit on one GPU
- Simplest execution mode

**Command:**
```bash
./scripts/launch.sh python [options]
```

**Options:**
- `-m, --model <name>`: Model name
- `-s, --script <path>`: Python script to run
- `--flexcache`: Enable FlexCache acceleration
- `--low-mem`: Enable low memory mode

**Examples:**
```bash
# Basic usage
./scripts/launch.sh python

# With specific model
./scripts/launch.sh python -m Wan2.1-T2V-1.3B

# With FlexCache
./scripts/launch.sh python --flexcache --low-mem
```

**Technical details:**
- Uses plain Python (no distributed framework)
- Single GPU device (cuda:0)
- No distributed parallelism
- Lowest overhead, fastest startup

---

### 2. Torchrun Mode (Multi-Card)

**When to use:**
- Multi-GPU on single node (2-8 GPUs)
- Distributed training/inference
- Medium to large models
- Need context parallelism or CFG parallelism

**Command:**
```bash
./scripts/launch.sh torchrun [options]
```

**Options:**
- `-g, --gpus <num>`: Number of GPUs (default: 2)
- `-m, --model <name>`: Model name
- `-c, --cp-size <num>`: Context parallel size (auto-calculated by default)
- `-s, --script <path>`: Python script to run
- `--cfg-parallel`: Enable CFG parallel mode
- `--flexcache`: Enable FlexCache acceleration
- `--low-mem`: Enable low memory mode

**Examples:**
```bash
# 2 GPUs with auto CP size
./scripts/launch.sh torchrun -g 2

# 4 GPUs with FlexCache
./scripts/launch.sh torchrun -g 4 --flexcache

# 8 GPUs with CFG parallel
./scripts/launch.sh torchrun -g 8 --cfg-parallel

# Custom CP size
./scripts/launch.sh torchrun -g 4 -c 2
```

**Technical details:**
- Uses PyTorch `torchrun` for distributed execution
- Single node, multiple GPUs
- Supports context parallelism (CP)
- Supports CFG parallelism
- Automatic world size and rank management

---

### 3. Srun Mode (Cluster)

**When to use:**
- SLURM-managed cluster
- Multi-node distributed execution
- Large-scale training/inference
- Need specific SLURM partitions or resources

**Command:**
```bash
./scripts/launch.sh srun [options]
```

**Options:**
- `-n, --nodes <num>`: Number of nodes (default: 1)
- `-g, --gpus <num>`: Number of GPUs per node (default: 2)
- `-p, --partition <name>`: SLURM partition (default: a01)
- `-m, --model <name>`: Model name
- `-s, --script <path>`: Python script to run
- `--multi-node`: Use multi-node launcher (required for >1 node)
- `--flexcache`: Enable FlexCache acceleration
- `--low-mem`: Enable low memory mode
- `--attn-type <type>`: Attention backend (flash_attn, sage, sparge, auto)

**Examples:**
```bash
# Single node, 8 GPUs
./scripts/launch.sh srun -g 8

# Custom partition
./scripts/launch.sh srun -g 8 -p gpu_partition

# Multi-node: 2 nodes × 8 GPUs
./scripts/launch.sh srun -n 2 -g 8 --multi-node

# With sparse attention
./scripts/launch.sh srun -g 8 --attn-type sparge

# All features
./scripts/launch.sh srun -n 2 -g 8 -p custom_partition --multi-node --flexcache --attn-type sage
```

**Technical details:**
- Uses SLURM `srun` for job scheduling
- Supports single-node and multi-node execution
- Configurable SLURM partition
- Advanced SLURM resource management (CPUs, memory, GPUs)
- Automatic distributed environment setup

---

## Directory Structure

```
scripts/
├── launch.sh              # Main unified entry point
├── README.md              # This file
│
├── python/                # Single-card Python execution
│   ├── run_python.sh     # Python launcher
│   └── README.md         # Python mode documentation
│
├── local/                 # Multi-card torchrun execution
│   ├── run_local.sh      # Torchrun launcher
│   └── README.md         # Torchrun mode documentation
│
├── cluster/               # Cluster srun execution
│   ├── run_cluster.sh    # Unified cluster launcher
│   ├── srun_direct.sh    # Direct srun execution
│   ├── srun_wrapper.sh   # Srun environment wrapper
│   ├── srun_multi_node.sh # Multi-node srun launcher
│   └── README.md         # Cluster mode documentation
│
└── utils/                 # Shared utilities
    ├── common.sh         # Common functions (logging, validation)
    └── config.sh         # Configuration (model paths, defaults)
```

## Usage Guide

### Choosing the Right Mode

| Scenario | Mode | Command |
|----------|------|---------|
| Quick test on 1 GPU | `python` | `./scripts/launch.sh python` |
| Development on 2-4 GPUs | `torchrun` | `./scripts/launch.sh torchrun -g 4` |
| Production on single node | `torchrun` | `./scripts/launch.sh torchrun -g 8` |
| Cluster single node | `srun` | `./scripts/launch.sh srun -g 8 -p a01` |
| Cluster multi-node | `srun` | `./scripts/launch.sh srun -n 2 -g 8 --multi-node` |

### Common Options (All Modes)

- `-m, --model <name>`: Specify model name
  - `Wan2.1-T2V-1.3B`
  - `Wan2.1-T2V-14B`
  - `Wan2.2-T2V-A14B`
  - Or use interactive selection (default)

- `-s, --script <path>`: Python script to execute
  - Default: `./test/test_generate.py`
  - Can be any Python script

- `--flexcache`: Enable FlexCache (TeaCache) acceleration
  - Speeds up diffusion inference
  - Recommended for production

- `--low-mem`: Enable low memory mode
  - Reduces memory usage
  - Useful for large models

### Mode-Specific Options

**Torchrun mode only:**
- `-g, --gpus <num>`: Number of GPUs to use
- `-c, --cp-size <num>`: Context parallel size
- `--cfg-parallel`: Enable CFG parallel mode

**Srun mode only:**
- `-n, --nodes <num>`: Number of nodes
- `-g, --gpus <num>`: GPUs per node
- `-p, --partition <name>`: SLURM partition
- `--multi-node`: Multi-node mode flag
- `--attn-type <type>`: Attention backend type

## System Parameters

### Environment Variables

You can set environment variables to override defaults:

```bash
# Model selection
export MODEL_NAME=Wan2.1-T2V-14B

# Distributed training
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# SLURM partition (for srun mode)
export SLURM_PARTITION=gpu_partition

# Then run
./scripts/launch.sh torchrun -g 4
```

### Model Configuration

Edit `scripts/utils/config.sh` to configure model paths:

```bash
declare -A DEFAULT_MODEL_CONFIGS=(
    ["Wan2.1-T2V-1.3B"]="/path/to/Wan2.1-T2V-1.3B"
    ["Wan2.1-T2V-14B"]="/path/to/Wan2.1-T2V-14B"
    ["Wan2.2-T2V-A14B"]="/path/to/Wan2.2-T2V-A14B"
)
```

### SLURM Configuration (Srun Mode)

Default SLURM settings can be found in cluster scripts:

- **Partition**: `a01` (configurable with `-p`)
- **CPUs per GPU**: 24
- **Memory per GPU**: 242144 MB (~236 GB)
- **Job Name**: `$USER-chitu`

To modify defaults, edit:
- `scripts/cluster/srun_direct.sh`
- `scripts/cluster/srun_multi_node.sh`

```bash
# Example: Change CPUs per GPU
CPUS_PER_GPU=32

# Example: Change memory per GPU
MEM_PER_GPU=300000
```

### Distributed Training Parameters

For torchrun and srun modes, the following are automatically configured:

- **World Size**: Total number of processes (= nodes × GPUs)
- **Rank**: Process rank (0 to world_size-1)
- **Local Rank**: GPU index on current node (0 to GPUs-1)
- **Context Parallel Size**: Calculated as `num_gpus / 2` (configurable)

## Configuration

### Quick Configuration Checklist

1. **Set model paths** in `scripts/utils/config.sh`
2. **Choose execution mode**: python, torchrun, or srun
3. **Set partition** (for srun mode): `-p your_partition`
4. **Adjust resources** (for srun mode): Edit CPUS_PER_GPU, MEM_PER_GPU
5. **Enable features**: --flexcache, --low-mem, --attn-type

### Example Configurations

**Small model, quick test:**
```bash
./scripts/launch.sh python -m Wan2.1-T2V-1.3B
```

**Medium model, development:**
```bash
./scripts/launch.sh torchrun -g 4 -m Wan2.1-T2V-14B --flexcache
```

**Large model, production cluster:**
```bash
./scripts/launch.sh srun -n 2 -g 8 -p gpu_partition -m Wan2.2-T2V-A14B \
  --multi-node --flexcache --low-mem --attn-type sage
```

## Migration Guide

### From Old Scripts

Old scripts have been replaced with the unified launcher:

| Old Script | New Command |
|------------|-------------|
| `./run_local_single.sh 1` | `./scripts/launch.sh python` |
| `./run_local_cfg.sh 2` | `./scripts/launch.sh torchrun -g 2` |
| `./run_local_cp.sh 4` | `./scripts/launch.sh torchrun -g 4` |
| `./torchrun_wan_demo.sh 2` | `./scripts/launch.sh torchrun -g 2` |
| `./srun_wan_demo.sh 8` | `./scripts/launch.sh srun -g 8` |

### Backward Compatibility

The old mode names are still supported:

```bash
# Old way (still works, but shows deprecation warning)
./scripts/launch.sh local -g 2
./scripts/launch.sh cluster -g 8

# New way (recommended)
./scripts/launch.sh torchrun -g 2
./scripts/launch.sh srun -g 8
```

## Troubleshooting

### Common Issues

**Issue: "Model not found"**
- Check model paths in `scripts/utils/config.sh`
- Or use `-m <model_name>` to specify model

**Issue: "SLURM partition not found"**
- Check available partitions: `sinfo`
- Specify correct partition: `-p your_partition`

**Issue: "Out of memory"**
- Use `--low-mem` flag
- Reduce number of GPUs
- Use smaller model

**Issue: "Torchrun not found"**
- Make sure PyTorch is installed
- Check your conda/virtual environment

### Getting Help

```bash
# General help
./scripts/launch.sh help

# Mode-specific help
./scripts/python/run_python.sh --help
./scripts/local/run_local.sh --help
./scripts/cluster/run_cluster.sh --help
```

## Advanced Usage

### Custom Scripts

You can run custom Python scripts:

```bash
./scripts/launch.sh torchrun -g 4 -s ./my_custom_script.py
```

### Passing Additional Arguments

Use `--` to pass additional arguments to the Python script:

```bash
./scripts/launch.sh torchrun -g 2 -- extra.param=value another.param=123
```

### Environment Customization

```bash
# Set debug environment
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

# Run with custom environment
./scripts/launch.sh torchrun -g 2
```

## Summary

- **3 execution modes**: `python` (1 GPU), `torchrun` (multi-GPU), `srun` (cluster)
- **Unified entry point**: `scripts/launch.sh <mode> [options]`
- **Organized structure**: Each mode in its own directory with documentation
- **Flexible configuration**: Command-line args, environment variables, config files
- **Backward compatible**: Old mode names still work with deprecation warnings

For more details, see the README in each mode's directory:
- `scripts/python/README.md`
- `scripts/local/README.md`
- `scripts/cluster/README.md`
