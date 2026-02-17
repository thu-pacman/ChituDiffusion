# SmartDiffusion Quick Start Guide

## New Unified Launch System

SmartDiffusion now has a unified entry point for all execution modes!

## Quick Commands

### Local Execution (Single Node with torchrun)

```bash
# Basic usage with 2 GPUs
./scripts/launch.sh local -g 2

# With specific model (no interactive selection)
./scripts/launch.sh local -g 2 -m Wan2.1-T2V-14B

# Enable FlexCache acceleration
./scripts/launch.sh local -g 4 --flexcache

# Enable low memory mode
./scripts/launch.sh local -g 2 --low-mem

# CFG parallel mode
./scripts/launch.sh local -g 4 --cfg-parallel

# All features combined
./scripts/launch.sh local -g 4 -m Wan2.1-T2V-14B --flexcache --low-mem
```

### Cluster Execution (SLURM with srun)

```bash
# Single node with 8 GPUs
./scripts/launch.sh cluster -g 8

# Multi-node: 2 nodes with 8 GPUs each
./scripts/launch.sh cluster -n 2 -g 8 --multi-node

# With sparse quantized attention
./scripts/launch.sh cluster -g 8 --attn-type sparge

# All features combined
./scripts/launch.sh cluster -n 2 -g 8 --multi-node --flexcache --low-mem --attn-type sparge
```

### Get Help

```bash
# Main help
./scripts/launch.sh help

# Local mode help
./scripts/local/run_local.sh --help

# Cluster mode help
./scripts/cluster/run_cluster.sh --help
```

## Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `-g, --gpus <num>` | Number of GPUs | `-g 4` |
| `-m, --model <name>` | Model name | `-m Wan2.1-T2V-14B` |
| `-n, --nodes <num>` | Number of nodes (cluster) | `-n 2` |
| `-s, --script <path>` | Python script to run | `-s ./custom_test.py` |
| `--flexcache` | Enable FlexCache (TeaCache) | `--flexcache` |
| `--low-mem` | Enable low memory mode | `--low-mem` |
| `--cfg-parallel` | Enable CFG parallel (local) | `--cfg-parallel` |
| `--attn-type <type>` | Attention backend (cluster) | `--attn-type sparge` |
| `--multi-node` | Multi-node mode (cluster) | `--multi-node` |

## Attention Types

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
# Set model without interactive selection
export MODEL_NAME="Wan2.1-T2V-14B"

# Distributed training settings
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"
```

## Migration from Old Scripts

| Old Command | New Command |
|-------------|-------------|
| `./run_local_cfg.sh 2` | `./scripts/launch.sh local -g 2 --cfg-parallel` |
| `./run_local_single.sh 1` | `./scripts/launch.sh local -g 1` |
| `./torchrun_wan_demo.sh 2` | `./scripts/launch.sh local -g 2` |
| `./srun_wan_demo.sh 8` | `./scripts/launch.sh cluster -g 8` |

## Troubleshooting

### "Model path does not exist"
→ Update model paths in `scripts/utils/config.sh`

### "Permission denied"
→ Run: `chmod +x scripts/**/*.sh`

### SLURM partition not found
→ Edit partition in `scripts/cluster/srun_*.sh` (default: `a01`)

### Out of memory
→ Use `--low-mem` flag

## Documentation

- Complete guide: `scripts/README.md`
- Local execution: `scripts/local/README.md`
- Cluster execution: `scripts/cluster/README.md`
- Migration guide: `MIGRATION.md`
- Full documentation: `README.md`

## Support

For issues or questions:
1. Check the relevant README in `scripts/`
2. Review the migration guide: `MIGRATION.md`
3. Verify configuration in `scripts/utils/config.sh`
4. Open an issue on GitHub

---

**Pro Tip**: Start simple with `./scripts/launch.sh local -g 2` and add options as needed!
