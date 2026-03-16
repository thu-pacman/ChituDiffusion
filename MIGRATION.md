# Migration Guide: Old Scripts → New Launch System

## Overview

The SmartDiffusion launch system has been reorganized into a unified, mode-based system with 3 execution modes:

1. **python** - Single-card Python execution (1 GPU)
2. **torchrun** - Multi-card torchrun execution (2-8 GPUs, single node)
3. **srun** - Cluster srun execution (single or multi-node)

## Quick Migration Table

| Old Script | New Command | Notes |
|------------|-------------|-------|
| `./run_local_single.sh 1` | `./scripts/launch.sh python` | Now uses Python mode |
| `./run_local_cfg.sh 2` | `./scripts/launch.sh torchrun -g 2` | CFG parallel mode |
| `./run_local_cp.sh 4` | `./scripts/launch.sh torchrun -g 4` | Context parallel mode |
| `./torchrun_wan_demo.sh 2` | `./scripts/launch.sh torchrun -g 2` | Direct replacement |
| `./srun_wan_demo.sh 8` | `./scripts/launch.sh srun -g 8 -p a01` | Add partition flag |

## Detailed Migration

### From run_local_single.sh

**Old:**
```bash
./run_local_single.sh 1
```

**New:**
```bash
./scripts/launch.sh python
# or
./scripts/python/run_python.sh
```

**Changes:**
- Now uses dedicated Python mode (simpler, no torchrun overhead)
- No GPU count needed (always uses 1 GPU)
- Use `-m` to specify model

### From run_local_cfg.sh

**Old:**
```bash
./run_local_cfg.sh 2
./run_local_cfg.sh 4
```

**New:**
```bash
./scripts/launch.sh torchrun -g 2 --cfg-parallel
./scripts/launch.sh torchrun -g 4 --cfg-parallel
```

**Changes:**
- Now requires explicit `--cfg-parallel` flag
- Use `-g` to specify GPU count
- Can combine with other flags: `--flexcache`, `--low-mem`

### From run_local_cp.sh

**Old:**
```bash
./run_local_cp.sh 2
./run_local_cp.sh 4
```

**New:**
```bash
./scripts/launch.sh torchrun -g 2
./scripts/launch.sh torchrun -g 4
```

**Changes:**
- Context parallel (CP) is now the default behavior
- CP size is auto-calculated (num_gpus / 2)
- Use `-c` to manually set CP size: `-c 2`

### From torchrun_wan_demo.sh

**Old:**
```bash
./torchrun_wan_demo.sh 2
./torchrun_wan_demo.sh 4
```

**New:**
```bash
./scripts/launch.sh torchrun -g 2
./scripts/launch.sh torchrun -g 4
```

**Changes:**
- Direct replacement with clearer naming
- Same torchrun backend
- More options available: `--flexcache`, `--low-mem`, `--cfg-parallel`

### From srun_wan_demo.sh

**Old:**
```bash
./srun_wan_demo.sh 8
```

**New:**
```bash
./scripts/launch.sh srun -g 8 -p a01
# or with your partition
./scripts/launch.sh srun -g 8 -p your_partition
```

**Changes:**
- Now requires partition flag `-p` (or uses default `a01`)
- Can specify nodes with `-n`
- Multi-node requires `--multi-node` flag
- More options: `--attn-type`, `--flexcache`, `--low-mem`

### From script/ directory scripts

**Old:**
```bash
./script/srun_direct.sh 1 8 ./test/test_generate.py args...
./script/srun_multi_node.sh 2 8 ./test/test_generate.py args...
```

**New:**
```bash
./scripts/launch.sh srun -g 8 -s ./test/test_generate.py -- args...
./scripts/launch.sh srun -n 2 -g 8 --multi-node -s ./test/test_generate.py -- args...
```

**Changes:**
- Unified interface with clearer options
- Partition is configurable: `-p partition_name`
- Use `--` to separate script arguments

## Mode Naming Changes

### Backward Compatibility

Old mode names still work but show deprecation warnings:

**Old style (deprecated):**
```bash
./scripts/launch.sh local -g 2      # Shows warning
./scripts/launch.sh cluster -g 8    # Shows warning
```

**New style (recommended):**
```bash
./scripts/launch.sh torchrun -g 2   # Clear and explicit
./scripts/launch.sh srun -g 8       # Clear and explicit
```

## New Features Available

### 1. Python Mode (New!)

Single-card execution without torchrun overhead:

```bash
./scripts/launch.sh python
./scripts/launch.sh python -m Wan2.1-T2V-1.3B --flexcache
```

### 2. Configurable Partition

Cluster partition is now configurable:

```bash
./scripts/launch.sh srun -g 8 -p gpu_partition
./scripts/launch.sh srun -g 8 -p a100_partition
```

### 3. Attention Backend Selection

Choose attention implementation (srun mode):

```bash
./scripts/launch.sh srun -g 8 --attn-type sage
./scripts/launch.sh srun -g 8 --attn-type sparge
```

### 4. Command-Line Configuration

All options via command line (no script editing):

```bash
./scripts/launch.sh torchrun -g 4 -m Wan2.1-T2V-14B --flexcache --low-mem
```

## Directory Structure Changes

**Old:**
```
run_local_cfg.sh
run_local_cp.sh
run_local_single.sh
srun_wan_demo.sh
torchrun_wan_demo.sh
script/
  srun_direct.sh
  srun_wrapper.sh
  srun_multi_node.sh
```

**New:**
```
scripts/
  launch.sh              # Main entry point
  python/
    run_python.sh        # Single-card mode
  local/
    run_local.sh         # Multi-card mode
  cluster/
    run_cluster.sh       # Cluster mode
    srun_direct.sh
    srun_wrapper.sh
    srun_multi_node.sh
  utils/
    common.sh
    config.sh
```

## Environment Variables

### Old Usage

Model paths were hardcoded in each script.

### New Usage

Configure once in `scripts/utils/config.sh`:

```bash
declare -A DEFAULT_MODEL_CONFIGS=(
    ["Wan2.1-T2V-1.3B"]="/path/to/model"
    ["Wan2.1-T2V-14B"]="/path/to/model"
    ["Wan2.2-T2V-A14B"]="/path/to/model"
)
```

Or use environment variables:

```bash
export MODEL_NAME=Wan2.1-T2V-14B
export SLURM_PARTITION=gpu_partition
./scripts/launch.sh torchrun -g 4
```

## Common Migration Scenarios

### Scenario 1: Development Workflow

**Old:**
```bash
# Edit run_local_cfg.sh to change GPU count
vim run_local_cfg.sh
./run_local_cfg.sh 2

# Edit again for different model
vim run_local_cfg.sh
./run_local_cfg.sh 2
```

**New:**
```bash
# Everything via command line
./scripts/launch.sh torchrun -g 2 -m Wan2.1-T2V-1.3B
./scripts/launch.sh torchrun -g 4 -m Wan2.1-T2V-14B
./scripts/launch.sh torchrun -g 8 --flexcache
```

### Scenario 2: Cluster Testing

**Old:**
```bash
# Edit srun_wan_demo.sh to change partition
vim srun_wan_demo.sh
./srun_wan_demo.sh 8
```

**New:**
```bash
# Specify partition on command line
./scripts/launch.sh srun -g 8 -p test_partition
./scripts/launch.sh srun -g 8 -p production_partition
```

### Scenario 3: Multi-Node Execution

**Old:**
```bash
./script/srun_multi_node.sh 2 8 ./test/test_generate.py
```

**New:**
```bash
./scripts/launch.sh srun -n 2 -g 8 --multi-node
```

## Troubleshooting Migration Issues

### Issue: "Command not found"

**Problem:** Old scripts don't exist anymore

**Solution:** Use new launch system:
```bash
./scripts/launch.sh <mode> [options]
```

### Issue: "Unknown option"

**Problem:** Old-style positional arguments

**Solution:** Use named options:
```bash
# Old: ./run_local_cfg.sh 2
# New: ./scripts/launch.sh torchrun -g 2 --cfg-parallel
```

### Issue: "Partition not found"

**Problem:** Default partition changed or needs specification

**Solution:** Specify partition explicitly:
```bash
./scripts/launch.sh srun -g 8 -p your_partition
```

### Issue: "Model path not found"

**Problem:** Model paths not configured

**Solution:** Edit `scripts/utils/config.sh` or use `-m` flag:
```bash
./scripts/launch.sh torchrun -g 2 -m Wan2.1-T2V-14B
```

## Getting Help

```bash
# Main help
./scripts/launch.sh help

# Mode-specific help
./scripts/python/run_python.sh --help
./scripts/local/run_local.sh --help
./scripts/cluster/run_cluster.sh --help

# Read documentation
cat scripts/README.md
cat QUICKSTART.md
```

## Summary

| Aspect | Old System | New System |
|--------|-----------|------------|
| Entry points | 5+ scattered scripts | 1 unified launcher |
| Configuration | Edit scripts | Command-line args |
| Modes | Implicit (by script name) | Explicit (`python`, `torchrun`, `srun`) |
| Partition | Hardcoded | Configurable with `-p` |
| Documentation | None | Comprehensive (README, QUICKSTART) |
| Structure | Mixed in root | Organized in `scripts/` |

The new system provides better organization, flexibility, and usability while maintaining full backward compatibility through mode aliases.
