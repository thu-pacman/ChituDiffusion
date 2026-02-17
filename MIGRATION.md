# Migration Guide: Old Scripts to New Unified System

This guide helps you migrate from the old scattered shell scripts to the new unified launch system.

## What Changed?

### Old Structure (Deprecated)
```
SmartDiffusion/
├── run_local_cfg.sh          # Local with CFG parallel
├── run_local_cp.sh           # Local with CP parallel
├── run_local_single.sh       # Single GPU local
├── srun_wan_demo.sh          # SLURM cluster demo
├── torchrun_wan_demo.sh      # Torchrun demo
└── script/                   # Old script directory
    ├── srun_direct.sh
    ├── srun_wrapper.sh
    └── srun_multi_node.sh
```

### New Structure (Current)
```
SmartDiffusion/
├── scripts/                  # New unified scripts directory
│   ├── launch.sh            # Main entry point
│   ├── local/               # Local execution (torchrun)
│   │   └── run_local.sh
│   ├── cluster/             # Cluster execution (srun)
│   │   ├── run_cluster.sh
│   │   ├── srun_direct.sh
│   │   ├── srun_wrapper.sh
│   │   └── srun_multi_node.sh
│   └── utils/               # Common utilities
│       ├── common.sh
│       └── config.sh
└── [old scripts with deprecation notices]
```

## Migration Examples

### 1. Local Execution Scripts

#### `run_local_cfg.sh`
```bash
# Old command
./run_local_cfg.sh 2

# New equivalent (recommended)
./scripts/launch.sh local -g 2 --cfg-parallel

# Alternative (direct)
./scripts/local/run_local.sh -g 2 --cfg-parallel
```

#### `run_local_cp.sh`
```bash
# Old command
./run_local_cp.sh 2

# New equivalent
./scripts/launch.sh local -g 2 --cfg-parallel
```

#### `run_local_single.sh`
```bash
# Old command
./run_local_single.sh 1

# New equivalent
./scripts/launch.sh local -g 1
```

#### `torchrun_wan_demo.sh`
```bash
# Old command
./torchrun_wan_demo.sh 2

# New equivalent
./scripts/launch.sh local -g 2
```

### 2. Cluster Execution Scripts

#### `srun_wan_demo.sh`
```bash
# Old command
./srun_wan_demo.sh 8

# New equivalent
./scripts/launch.sh cluster -g 8

# With additional options
./scripts/launch.sh cluster -g 8 --flexcache --attn-type sparge
```

#### Direct srun scripts
```bash
# Old command
./script/srun_direct.sh 1 8 ./test/test_generate.py models=... 

# New equivalent
./scripts/cluster/run_cluster.sh -g 8 -m <model_name>
# Or use the unified launcher
./scripts/launch.sh cluster -g 8 -m <model_name>
```

## Feature Mapping

### Old Features → New Flags

| Old Script Feature | New Flag/Option |
|-------------------|-----------------|
| CFG parallel in `run_local_cfg.sh` | `--cfg-parallel` |
| FlexCache (hardcoded) | `--flexcache` |
| Low memory mode (commented) | `--low-mem` |
| Attention type (hardcoded) | `--attn-type <type>` |
| Multi-node srun | `--multi-node` |
| Model selection (interactive) | `-m <model_name>` or interactive |
| GPU count | `-g <num>` |
| Custom script | `-s <path>` |

## New Capabilities

The new system adds several improvements:

### 1. Command-line Arguments
No need to edit scripts - configure everything via command-line:
```bash
./scripts/launch.sh local -g 4 -m Wan2.1-T2V-14B --flexcache --low-mem
```

### 2. Unified Entry Point
One command for all execution modes:
```bash
./scripts/launch.sh local ...    # Local execution
./scripts/launch.sh cluster ...  # Cluster execution
./scripts/launch.sh help         # Show help
```

### 3. Better Documentation
- Comprehensive help messages: `--help`
- README files in each directory
- Clear migration path

### 4. Common Utilities
Shared functions for:
- Environment setup
- Model configuration
- Parameter validation
- Colored output

### 5. Flexibility
- Mix and match options
- Environment variable support
- Custom script execution

## Environment Variables

The new system supports environment variables for easier automation:

```bash
# Set model
export MODEL_NAME="Wan2.1-T2V-14B"

# Set distributed training parameters
export MASTER_ADDR="192.168.1.100"
export MASTER_PORT="29500"

# Run without interactive selection
./scripts/launch.sh local -g 4
```

## Configuration

### Model Paths

Edit `scripts/utils/config.sh` to set your model paths:

```bash
declare -A DEFAULT_MODEL_CONFIGS=(
    ["Wan2.1-T2V-1.3B"]="/path/to/Wan2.1-T2V-1.3B"
    ["Wan2.1-T2V-14B"]="/path/to/Wan2.1-T2V-14B"
    ["Wan2.2-T2V-A14B"]="/path/to/Wan2.2-T2V-A14B"
)
```

### SLURM Configuration

Edit `scripts/cluster/srun_direct.sh` or `scripts/cluster/srun_multi_node.sh` to change SLURM defaults:

```bash
CPUS_PER_GPU=24        # CPUs per GPU
MEM_PER_GPU=242144     # Memory per GPU (MB)
# Partition is set in the srun command: -p a01
```

## Backward Compatibility

The old scripts still exist but show deprecation warnings. They will:
1. Display a warning message
2. Show the equivalent new command
3. Ask if you want to run the new command
4. Execute the new command if you agree

This gives you time to update your workflows while still being able to run old scripts.

## Recommended Workflow

1. **Update scripts in your workflows** to use the new system
2. **Update documentation** to reference the new paths
3. **Test the new system** with your specific configurations
4. **Remove references** to old scripts once migration is complete

## Common Issues

### "Model path does not exist"
**Solution**: Update model paths in `scripts/utils/config.sh`

### "Script not found"
**Solution**: Check if you're running from the project root directory

### "Permission denied"
**Solution**: Ensure scripts are executable: `chmod +x scripts/**/*.sh`

### SLURM partition not found
**Solution**: Update partition name in cluster scripts (default is `a01`)

## Getting Help

- Main launcher help: `./scripts/launch.sh help`
- Local mode help: `./scripts/local/run_local.sh --help`
- Cluster mode help: `./scripts/cluster/run_cluster.sh --help`
- Documentation: `scripts/README.md`, `scripts/local/README.md`, `scripts/cluster/README.md`

## Benefits of Migration

1. **Cleaner codebase**: Organized directory structure
2. **Better maintainability**: Shared utilities, less duplication
3. **Easier to use**: Command-line args instead of editing scripts
4. **More flexible**: Mix and match options as needed
5. **Better documentation**: Comprehensive help and examples
6. **Future-proof**: Easy to extend with new features

## Timeline

- **Now**: Old scripts show deprecation warnings but still work
- **Future**: Old scripts may be removed in a later release
- **Recommendation**: Migrate as soon as possible

## Support

If you encounter issues during migration:
1. Check the documentation in `scripts/README.md`
2. Verify your configuration in `scripts/utils/config.sh`
3. Test with the simplest command first: `./scripts/launch.sh local -g 1`
4. Open an issue if problems persist
