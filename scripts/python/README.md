# Single-Card Python Execution

This directory contains scripts for running SmartDiffusion on a single GPU using plain Python.

## Scripts

- `run_python.sh`: Single-card Python launcher

## Usage

### Basic Usage

```bash
# Run from project root
./scripts/python/run_python.sh
```

Or use the unified launcher:

```bash
./scripts/launch.sh python
```

### Options

- `-m, --model <name>`: Model name (default: interactive selection)
- `-s, --script <path>`: Python script to run (default: ./test/test_generate.py)
- `--flexcache`: Enable FlexCache acceleration
- `--low-mem`: Enable low memory mode
- `-h, --help`: Show help message

### Examples

```bash
# Run with interactive model selection
./scripts/python/run_python.sh

# Run with specific model
./scripts/python/run_python.sh -m Wan2.1-T2V-1.3B

# Run with FlexCache acceleration
./scripts/python/run_python.sh --flexcache

# Run with low memory mode
./scripts/python/run_python.sh --low-mem

# Run with custom script
./scripts/python/run_python.sh -s ./test/custom_test.py

# Run with all options
./scripts/python/run_python.sh -m Wan2.1-T2V-14B --flexcache --low-mem
```

## When to Use

Use single-card Python execution when:

- You have a small model that fits on a single GPU
- You're doing quick testing or debugging
- You don't need distributed training capabilities
- You want the simplest execution mode

For multi-GPU execution, use the `torchrun` mode instead.
For cluster execution, use the `srun` mode.

## Environment Variables

You can override settings using environment variables:

```bash
export MODEL_NAME=Wan2.1-T2V-1.3B
./scripts/python/run_python.sh
```

## Configuration

Model paths are configured in `scripts/utils/config.sh`.
