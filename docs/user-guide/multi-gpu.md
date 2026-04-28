# Multi-GPU Setup

Configure ChituDiffusion for multi-GPU execution.

## Prerequisites

- Multiple NVIDIA GPUs
- NCCL installed
- GPUs on same node or connected via high-speed interconnect

## Single Node, Multiple GPUs

### Unified Launcher

```bash
bash run.sh system_config.yaml --num-nodes 1 --gpus-per-node 4 --cfp 2
```

`--cfp` (or `parallel.cfp` in config) is CFG parallel factor (`1` or `2`).
Launcher derives `infer.diffusion.cp_size = total_gpus / cfp` automatically.

## Multi-Node Setup

### Using SLURM

```bash
bash run.sh system_config.yaml --num-nodes 2 --gpus-per-node 4 --cfp 2  # 8 GPUs total
```

### Manual Configuration

```bash
# Node 0
export MASTER_ADDR=node0
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=0
export LOCAL_RANK=0

python test_generate.py ...

# Node 1
export RANK=4
export LOCAL_RANK=0

python test_generate.py ...
```

## Parallelism Strategies

### Context Parallelism

Split frames across GPUs:

```bash
infer.diffusion.cp_size=4
```

### CFG Parallelism

Automatic with 2+ GPUs when CFG enabled.

### Combined

```bash
# 4 GPUs: 2 CFG × 2 CP
infer.diffusion.cfg_size=2 \
infer.diffusion.cp_size=2
```

## Troubleshooting

### NCCL Timeout

Increase timeout:

```bash
export NCCL_TIMEOUT=1800
```

### Network Issues

Check connectivity:

```bash
nvidia-smi topo -m
```

## See Also

- [Performance Tuning](performance-tuning.md)
- [Configuration Guide](../getting-started/configuration.md)
