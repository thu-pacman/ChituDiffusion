# Multi-GPU Setup

Configure Smart-Diffusion for multi-GPU execution.

## Prerequisites

- Multiple NVIDIA GPUs
- NCCL installed
- GPUs on same node or connected via high-speed interconnect

## Single Node, Multiple GPUs

### Using torchrun

```bash
torchrun --nproc_per_node=4 test_generate.py \
    models.ckpt_dir=/path/to/checkpoint \
    infer.diffusion.cp_size=2
```

### Using Provided Scripts

```bash
bash run_local_single.sh
```

Edit script to configure GPU count.

## Multi-Node Setup

### Using SLURM

```bash
bash srun_wan_demo.sh 8  # 8 GPUs total
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
