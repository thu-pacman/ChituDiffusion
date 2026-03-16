# SmartDiffusion Quick Start Guide

## 3 Ways to Launch SmartDiffusion

SmartDiffusion supports 3 execution modes:

### 1. Python Mode - Single GPU

For quick testing on a single GPU:

```bash
./scripts/launch.sh python
./scripts/launch.sh python -m Wan2.1-T2V-1.3B --flexcache
```

### 2. Torchrun Mode - Multi-GPU (Single Node)

For development and production on 2-8 GPUs:

```bash
./scripts/launch.sh torchrun -g 2
./scripts/launch.sh torchrun -g 4 --flexcache --low-mem
./scripts/launch.sh torchrun -g 8 --cfg-parallel
```

### 3. Srun Mode - Cluster (Multi-Node)

For SLURM cluster execution:

```bash
./scripts/launch.sh srun -g 8 -p a01
./scripts/launch.sh srun -n 2 -g 8 -p gpu_partition --multi-node
./scripts/launch.sh srun -g 8 --attn-type sparge
```

## Quick Reference

### Execution Mode Selection

| Scenario | Mode | Example |
|----------|------|---------|
| Quick test, 1 GPU | `python` | `./scripts/launch.sh python` |
| Dev/prod, 2-8 GPUs | `torchrun` | `./scripts/launch.sh torchrun -g 4` |
| Cluster, multi-node | `srun` | `./scripts/launch.sh srun -n 2 -g 8 --multi-node` |

### Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `-m, --model <name>` | Model name | `-m Wan2.1-T2V-14B` |
| `-s, --script <path>` | Python script to run | `-s ./custom_test.py` |
| `--flexcache` | Enable FlexCache (TeaCache) | `--flexcache` |
| `--low-mem` | Enable low memory mode | `--low-mem` |

### Mode-Specific Options

**Python mode:** (single GPU)
- No additional GPU options needed

**Torchrun mode:** (multi-GPU)
- `-g, --gpus <num>`: Number of GPUs
- `-c, --cp-size <num>`: Context parallel size
- `--cfg-parallel`: Enable CFG parallel

**Srun mode:** (cluster)
- `-n, --nodes <num>`: Number of nodes
- `-g, --gpus <num>`: GPUs per node
- `-p, --partition <name>`: SLURM partition
- `--multi-node`: Multi-node flag
- `--attn-type <type>`: Attention backend

## Detailed Examples

### Python Mode Examples

```bash
# Basic single GPU
./scripts/launch.sh python

# With specific model
./scripts/launch.sh python -m Wan2.1-T2V-1.3B

# With FlexCache and low memory
./scripts/launch.sh python --flexcache --low-mem

# Custom script
./scripts/launch.sh python -s ./test/my_test.py
```

### Torchrun Mode Examples

```bash
# 2 GPUs, default settings
./scripts/launch.sh torchrun -g 2

# 4 GPUs with FlexCache
./scripts/launch.sh torchrun -g 4 --flexcache

# 8 GPUs with CFG parallel
./scripts/launch.sh torchrun -g 8 --cfg-parallel

# With specific model and low memory
./scripts/launch.sh torchrun -g 4 -m Wan2.1-T2V-14B --low-mem

# Custom context parallel size
./scripts/launch.sh torchrun -g 4 -c 2
```

### Srun Mode Examples

```bash
# Single node, 8 GPUs
./scripts/launch.sh srun -g 8 -p a01

# Multi-node: 2 nodes × 8 GPUs
./scripts/launch.sh srun -n 2 -g 8 -p gpu_partition --multi-node

# With sparse attention
./scripts/launch.sh srun -g 8 --attn-type sparge

# With all features
./scripts/launch.sh srun -n 2 -g 8 -p custom_partition \
  --multi-node --flexcache --low-mem --attn-type sage
```

## Attention Types (Srun Mode Only)

| Type | Description |
|------|-------------|
| `flash_attn` | Default FlashAttention (baseline) |
| `sage` | SageAttention (~2x speedup) |
| `sparge` | Sparse + quantized attention |
| `auto` | Automatic selection |

## Configuration

### Set Model Paths

Edit `scripts/utils/config.sh`:

```bash
declare -A DEFAULT_MODEL_CONFIGS=(
    ["Wan2.1-T2V-1.3B"]="/path/to/model"
    ["Wan2.1-T2V-14B"]="/path/to/model"
    ["Wan2.2-T2V-A14B"]="/path/to/model"
)
```

### Environment Variables

```bash
export MODEL_NAME=Wan2.1-T2V-14B
export SLURM_PARTITION=gpu_partition
./scripts/launch.sh torchrun -g 4
```

## Getting Help

```bash
# Main help
./scripts/launch.sh help

# Mode-specific help
./scripts/python/run_python.sh --help
./scripts/local/run_local.sh --help
./scripts/cluster/run_cluster.sh --help
```

## Migration from Old Scripts

| Old Command | New Command |
|-------------|-------------|
| `./run_local_single.sh 1` | `./scripts/launch.sh python` |
| `./run_local_cfg.sh 2` | `./scripts/launch.sh torchrun -g 2` |
| `./run_local_cp.sh 4` | `./scripts/launch.sh torchrun -g 4` |
| `./srun_wan_demo.sh 8` | `./scripts/launch.sh srun -g 8 -p a01` |

Old mode names still work (with deprecation warning):
- `local` → use `torchrun` instead
- `cluster` → use `srun` instead

## Full Documentation

See `scripts/README.md` for complete documentation including:
- Detailed usage guide
- System parameters configuration
- SLURM configuration
- Troubleshooting
- Advanced usage
