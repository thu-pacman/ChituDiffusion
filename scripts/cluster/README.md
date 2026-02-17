# Cluster Execution with SLURM

This directory contains scripts for running SmartDiffusion on SLURM-managed clusters using `srun`.

## Scripts

- `run_cluster.sh`: Unified cluster launcher
- `srun_direct.sh`: Direct srun execution (single or multi-node)
- `srun_wrapper.sh`: Wrapper script for setting up distributed environment variables
- `srun_multi_node.sh`: Multi-node launcher with advanced options

## Usage

### Basic Usage

```bash
# Run from project root
./scripts/cluster/run_cluster.sh -g 8
```

Or use the unified launcher:

```bash
./scripts/launch.sh cluster -g 8
```

### Options

- `-n, --nodes <num>`: Number of nodes (default: 1)
- `-g, --gpus <num>`: Number of GPUs per node (default: 2)
- `-m, --model <name>`: Model name (default: interactive selection)
- `-s, --script <path>`: Python script to run (default: ./test/test_generate.py)
- `--multi-node`: Use multi-node launcher (required for multi-node execution)
- `--flexcache`: Enable FlexCache acceleration
- `--low-mem`: Enable low memory mode
- `--attn-type <type>`: Attention type (flash_attn, sage, sparge, auto)
- `-h, --help`: Show help message

### Examples

```bash
# Single node, 8 GPUs
./scripts/cluster/run_cluster.sh -g 8

# 2 nodes, 8 GPUs per node
./scripts/cluster/run_cluster.sh -n 2 -g 8 --multi-node

# With specific model
./scripts/cluster/run_cluster.sh -g 8 -m Wan2.1-T2V-14B

# With sparge attention backend
./scripts/cluster/run_cluster.sh -g 8 --attn-type sparge

# With all features
./scripts/cluster/run_cluster.sh -n 2 -g 8 --multi-node --flexcache --low-mem --attn-type sparge
```

## SLURM Configuration

The scripts use the following SLURM defaults (can be modified in the scripts):

- **Partition**: `-p a01` (edit in `srun_direct.sh` or `srun_multi_node.sh`)
- **Job Name**: `$USER-chitu`
- **CPUs per GPU**: 24
- **Memory per GPU**: 242144 MB (~236 GB)

### Modifying SLURM Parameters

Edit the scripts to change default SLURM settings:

```bash
# In srun_direct.sh or srun_multi_node.sh
CPUS_PER_GPU=24           # Change CPU allocation
MEM_PER_GPU=242144        # Change memory allocation
PARTITION=a01             # Change partition
```

## How It Works

### Single Node Mode (Default)

Uses `srun_direct.sh`:
1. Calculates resource requirements
2. Sets up master port and NCCL optimization
3. Submits job to SLURM with `srun`
4. Uses `srun_wrapper.sh` to set distributed environment variables
5. Executes Python script on each task

### Multi-Node Mode

Uses `srun_multi_node.sh`:
1. Calculates resource requirements
2. Submits job with `--nodes` and `--ntasks-per-node=1`
3. Uses `torchrun` inside SLURM allocation for better multi-node coordination
4. Sets up rendezvous backend (c10d) for distributed training

## Migrating from Old Scripts

### `srun_wan_demo.sh` → `run_cluster.sh`

```bash
# Old
./srun_wan_demo.sh 8

# New
./scripts/cluster/run_cluster.sh -g 8
```

### Custom srun invocations

The old scripts in the `script/` directory are now organized in `scripts/cluster/`:
- `script/srun_direct.sh` → `scripts/cluster/srun_direct.sh`
- `script/srun_wrapper.sh` → `scripts/cluster/srun_wrapper.sh`
- `script/srun_multi_node.sh` → `scripts/cluster/srun_multi_node.sh`

## Environment Variables

### SLURM Variables (Set by SLURM)

- `SLURM_JOB_ID`: Job ID
- `SLURM_PROCID`: Global task rank
- `SLURM_LOCALID`: Local task rank on node
- `SLURM_NTASKS`: Total number of tasks
- `SLURM_NNODES`: Number of nodes
- `SLURM_GPUS_ON_NODE`: Number of GPUs on current node

### Custom Variables

```bash
# Set before running
export MODEL_NAME="Wan2.1-T2V-14B"

# Run
./scripts/cluster/run_cluster.sh -g 8
```

## Troubleshooting

### SLURM Job Fails to Start

1. Check partition availability: `sinfo -p a01`
2. Verify resource requirements are not exceeding limits
3. Check job queue: `squeue -u $USER`

### "Invalid job id" Error

This usually happens when running from an expired `salloc` session. The scripts automatically clear old SLURM variables to request a fresh allocation.

### Multi-Node Communication Issues

1. Verify nodes can communicate (check network)
2. Check NCCL installation on all nodes
3. Ensure InfiniBand/high-speed network is properly configured
4. Check MASTER_ADDR is correctly set (first node in allocation)

### Memory Issues

Reduce memory per GPU if allocation fails:
```bash
# Edit the script to reduce MEM_PER_GPU
MEM_PER_GPU=120000  # ~117 GB instead of ~236 GB
```

Or enable low memory mode:
```bash
./scripts/cluster/run_cluster.sh -g 8 --low-mem
```

## Advanced Usage

### Custom SLURM Arguments

For advanced SLURM configuration, you can directly use `srun_multi_node.sh` with custom arguments:

```bash
./scripts/cluster/srun_multi_node.sh 2 8 --pty -- ./test/test_generate.py models=Wan2.1-T2V-14B models.ckpt_dir=/path/to/model
```

The `--` separator divides SLURM arguments from torchrun/script arguments.

### Interactive Mode

For interactive debugging on compute nodes:

```bash
./scripts/cluster/srun_multi_node.sh 1 8 --pty -- ./test/test_generate.py models=Wan2.1-T2V-14B
```

## Resource Guidelines

| Model Size | Recommended GPUs | Nodes | Memory |
|------------|------------------|-------|--------|
| 1.3B       | 2-4              | 1     | Normal |
| 14B        | 4-8              | 1-2   | Normal |
| A14B       | 8-16             | 2-4   | High   |

For large models, consider:
- Using `--low-mem` mode
- Increasing CP size (context parallelism)
- Using sparse attention (`--attn-type sparge`)
