# SmartDiffusion Launch Scripts

This directory contains the unified entry point system for launching SmartDiffusion with different execution backends.

## Directory Structure

```
scripts/
├── launch.sh              # Main unified entry point
├── local/                 # Local execution with torchrun
│   ├── run_local.sh      # Local launcher script
│   └── README.md         # Local mode documentation
├── cluster/               # Cluster execution with srun
│   ├── run_cluster.sh    # Cluster launcher script
│   ├── srun_direct.sh    # Direct srun execution
│   ├── srun_wrapper.sh   # Srun wrapper for distributed setup
│   ├── srun_multi_node.sh # Multi-node srun execution
│   └── README.md         # Cluster mode documentation
└── utils/                 # Common utilities
    ├── common.sh         # Common functions (output, validation, etc.)
    └── config.sh         # Configuration (model paths, defaults)
```

## Quick Start

### Local Execution (Single Node)

```bash
# Run with default settings (2 GPUs, interactive model selection)
./scripts/launch.sh local

# Run with 4 GPUs
./scripts/launch.sh local -g 4

# Run with specific model and FlexCache enabled
./scripts/launch.sh local -g 2 -m Wan2.1-T2V-14B --flexcache

# Run with low memory mode
./scripts/launch.sh local -g 2 --low-mem

# Run with CFG parallel mode
./scripts/launch.sh local -g 4 --cfg-parallel
```

### Cluster Execution (SLURM with srun)

```bash
# Run on single node with 8 GPUs
./scripts/launch.sh cluster -g 8

# Run on 2 nodes with 8 GPUs per node (multi-node)
./scripts/launch.sh cluster -n 2 -g 8 --multi-node

# Run with custom partition
./scripts/launch.sh cluster -g 8 -p gpu_partition

# Run with specific attention backend
./scripts/launch.sh cluster -g 8 --attn-type sparge

# Run with FlexCache and low memory mode
./scripts/launch.sh cluster -g 8 --flexcache --low-mem
```

## Configuration

### Model Paths

Edit `scripts/utils/config.sh` to configure your model paths:

```bash
declare -A DEFAULT_MODEL_CONFIGS=(
    ["Wan2.1-T2V-1.3B"]="/path/to/Wan2.1-T2V-1.3B"
    ["Wan2.1-T2V-14B"]="/path/to/Wan2.1-T2V-14B"
    ["Wan2.2-T2V-A14B"]="/path/to/Wan2.2-T2V-A14B"
)
```

### Environment Variables

You can also use environment variables:

```bash
# Set model name
export MODEL_NAME="Wan2.1-T2V-14B"

# Set master address and port for distributed training
export MASTER_ADDR="192.168.1.100"
export MASTER_PORT="29500"

# Then run
./scripts/launch.sh local -g 4
```

## Common Options

### General Options

- `-g, --gpus <num>`: Number of GPUs to use
- `-m, --model <name>`: Model name (skips interactive selection)
- `-s, --script <path>`: Python script to run (default: ./test/test_generate.py)
- `-h, --help`: Show help message

### Performance Options

- `--flexcache`: Enable FlexCache acceleration (TeaCache)
- `--low-mem`: Enable low memory mode (useful for large models)
- `--cfg-parallel`: Enable CFG parallel mode (local only)
- `--attn-type <type>`: Attention backend (flash_attn, sage, sparge, auto) (cluster only)

### Cluster-Specific Options

- `-n, --nodes <num>`: Number of nodes
- `-p, --partition <name>`: SLURM partition name (default: a01)
- `--multi-node`: Use multi-node launcher (required for multi-node execution)

## Migration from Old Scripts

If you were using the old scripts, here's how to migrate:

### Old: `run_local_cfg.sh`
```bash
# Old
./run_local_cfg.sh 4

# New (equivalent with CFG parallel)
./scripts/launch.sh local -g 4 --cfg-parallel
```

### Old: `run_local_single.sh`
```bash
# Old
./run_local_single.sh 1

# New
./scripts/launch.sh local -g 1
```

### Old: `srun_wan_demo.sh`
```bash
# Old
./srun_wan_demo.sh 8

# New
./scripts/launch.sh cluster -g 8
```

### Old: `torchrun_wan_demo.sh`
```bash
# Old
./torchrun_wan_demo.sh 2

# New
./scripts/launch.sh local -g 2
```

## Advanced Usage

### Custom Script

```bash
# Run a custom test script
./scripts/launch.sh local -g 2 -s ./my_custom_script.py
```

### Direct Use of Sub-Scripts

You can also directly use the sub-scripts if needed:

```bash
# Local execution directly
./scripts/local/run_local.sh -g 4 --flexcache

# Cluster execution directly
./scripts/cluster/run_cluster.sh -n 2 -g 8 --multi-node
```

## Troubleshooting

### Model Path Not Found

If you see "Model path does not exist", update the model paths in `scripts/utils/config.sh`.

### GPU Not Available

The scripts will check for GPU availability using `nvidia-smi`. If GPUs are not detected, check your CUDA installation.

### SLURM Job Errors

For SLURM-related errors, check:
1. Partition configuration in `scripts/cluster/srun_direct.sh` (default: `-p a01`)
2. Resource requirements (CPUs, memory per GPU)

## Support

For issues or questions, please check:
- Main README: `README.md`
- Chinese README: `README_zh.md`
- Documentation: `docs/` directory
