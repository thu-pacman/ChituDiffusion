# Local Execution with Torchrun

This directory contains scripts for running SmartDiffusion locally using `torchrun` for single-node multi-GPU execution.

## Main Script

- `run_local.sh`: Unified local launcher that replaces the old `run_local_*.sh` scripts

## Usage

### Basic Usage

```bash
# Run from project root
./scripts/local/run_local.sh -g 2
```

Or use the unified launcher:

```bash
./scripts/launch.sh local -g 2
```

### Options

- `-g, --gpus <num>`: Number of GPUs to use (default: 2)
- `-m, --model <name>`: Model name (default: interactive selection)
- `-c, --cp-size <num>`: Context parallel size (default: auto-calculated as GPUs/2)
- `-s, --script <path>`: Python script to run (default: ./test/test_generate.py)
- `--cfg-parallel`: Enable CFG parallel mode (sets cfg_size=1)
- `--flexcache`: Enable FlexCache acceleration
- `--low-mem`: Enable low memory mode
- `-h, --help`: Show help message

### Examples

```bash
# Single GPU
./scripts/local/run_local.sh -g 1

# 2 GPUs with interactive model selection
./scripts/local/run_local.sh -g 2

# 4 GPUs with specific model
./scripts/local/run_local.sh -g 4 -m Wan2.1-T2V-14B

# With FlexCache acceleration
./scripts/local/run_local.sh -g 2 --flexcache

# With low memory mode
./scripts/local/run_local.sh -g 2 --low-mem

# With CFG parallel
./scripts/local/run_local.sh -g 4 --cfg-parallel

# All features combined
./scripts/local/run_local.sh -g 4 -m Wan2.1-T2V-14B --flexcache --low-mem --cfg-parallel
```

## How It Works

1. **Environment Setup**: Sets up PYTHONPATH, distributed training environment variables
2. **Model Selection**: Interactive or via command-line argument
3. **Parameter Calculation**: Automatically calculates CP size based on GPU count
4. **Validation**: Checks model paths and script existence
5. **Execution**: Runs `torchrun` with appropriate parameters

## Migrating from Old Scripts

The new unified `run_local.sh` script replaces three old scripts:

### `run_local_cfg.sh` → `run_local.sh --cfg-parallel`
```bash
# Old
./run_local_cfg.sh 2

# New
./scripts/local/run_local.sh -g 2 --cfg-parallel
```

### `run_local_cp.sh` → `run_local.sh --cfg-parallel`
```bash
# Old
./run_local_cp.sh 2

# New
./scripts/local/run_local.sh -g 2 --cfg-parallel
```

### `run_local_single.sh` → `run_local.sh`
```bash
# Old
./run_local_single.sh 1

# New
./scripts/local/run_local.sh -g 1
```

## Environment Variables

You can set these environment variables before running:

```bash
# Model configuration
export MODEL_NAME="Wan2.1-T2V-14B"

# Distributed training
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"

# Run script
./scripts/local/run_local.sh -g 4
```

## Troubleshooting

### CUDA Out of Memory

Try enabling low memory mode:
```bash
./scripts/local/run_local.sh -g 2 --low-mem
```

### Model Not Found

1. Check model paths in `scripts/utils/config.sh`
2. Verify the model directory exists
3. Ensure MODEL_CONFIGS is properly set

### Distributed Training Issues

1. Check MASTER_ADDR and MASTER_PORT are not in use
2. Verify NCCL is properly installed
3. Check GPU visibility with `nvidia-smi`

## Advanced Configuration

For advanced configuration, edit:
- Model paths: `scripts/utils/config.sh`
- Common functions: `scripts/utils/common.sh`
- Default values in `run_local.sh` itself
