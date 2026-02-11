# Parallelism in Smart-Diffusion

Smart-Diffusion supports multiple parallelism strategies to scale diffusion model inference across multiple GPUs and nodes.

## Overview

Smart-Diffusion implements three main parallelism dimensions:

1. **Context Parallelism (CP)**: Split the sequence (frames) dimension
2. **CFG Parallelism**: Split positive/negative prompts for Classifier-Free Guidance
3. **Data Parallelism** (planned): Process multiple requests in parallel

## Context Parallelism (CP)

Context Parallelism splits the sequence dimension across GPUs, enabling longer video generation with limited per-GPU memory.

### How It Works

```
GPU 0: [frames 0-40]   ←→ All-to-all communication
GPU 1: [frames 41-80]  ←→ for attention
```

Each GPU:
1. Holds a slice of the sequence
2. Computes local attention operations
3. Communicates via all-to-all for global attention
4. Merges results for next layer

### Configuration

```bash
# Use 2 GPUs for context parallelism
torchrun --nproc_per_node=2 test_generate.py \
    infer.diffusion.cp_size=2
```

### Memory Scaling

With CP size = N:
- Memory per GPU: ~1/N of single-GPU memory
- Communication overhead: O(N) all-to-all operations
- Near-linear speedup up to 8 GPUs

### Use Cases

- Generating longer videos (more frames)
- Working with limited VRAM
- Scaling to very high resolutions

## CFG Parallelism

CFG Parallelism splits the positive and negative prompts across two GPUs, effectively doubling CFG computation speed.

### How It Works

```
GPU 0: Positive prompt inference
GPU 1: Negative prompt (unconditioned) inference
```

Results are combined via all-gather:
```python
prediction = uncond + guidance_scale * (cond - uncond)
```

### Configuration

CFG Parallelism is automatically enabled when:
- World size >= 2
- CFG is enabled (`guidance_scale > 1.0`)

To control explicitly:
```python
args.infer.diffusion.cfg_size = 2
```

### Benefits

- 2x speedup for CFG computation
- No additional memory overhead
- Works well with context parallelism

### Limitations

- Only beneficial for 2 GPUs
- Requires CFG to be enabled
- Communication overhead for result merging

## Data Parallelism

*Status: Planned for future release*

Data Parallelism will enable processing multiple user requests in parallel across different GPUs.

### Planned Design

```
GPU 0: Request A
GPU 1: Request B
GPU 2: Request C
GPU 3: Request D
```

### Benefits

- Higher throughput for multi-user scenarios
- Better GPU utilization
- Independent request processing

## Hybrid Parallelism

You can combine different parallelism strategies for optimal performance.

### Example: 4 GPU Setup

```bash
# 2 CFG × 2 CP = 4 GPUs total
torchrun --nproc_per_node=4 test_generate.py \
    infer.diffusion.cfg_size=2 \
    infer.diffusion.cp_size=2
```

This configuration:
- Uses 2 GPUs for CFG (positive/negative)
- Splits each CFG computation across 2 GPUs via CP
- Total: 2 × 2 = 4 GPUs

### Scaling Guidelines

| GPUs | Recommended Strategy |
|------|---------------------|
| 1 | No parallelism |
| 2 | CFG parallelism |
| 4 | 2 CFG × 2 CP |
| 8 | 2 CFG × 4 CP |
| 16+ | 2 CFG × 8+ CP |

## Communication Patterns

### All-to-All (Context Parallelism)

Used for attention computation across sequence chunks:

```python
# Pseudo-code
local_chunk = input[my_rank * chunk_size:(my_rank + 1) * chunk_size]
local_result = local_attention(local_chunk)
global_result = all_to_all(local_result)
```

**Cost**: O(N) where N is CP size

### All-Gather (CFG Parallelism)

Used to combine CFG predictions:

```python
# Pseudo-code
local_pred = model_forward(my_prompt)
[cond_pred, uncond_pred] = all_gather(local_pred)
final = uncond_pred + scale * (cond_pred - uncond_pred)
```

**Cost**: O(1) fixed communication

## Performance Characteristics

### Context Parallelism Scaling

Scaling efficiency benchmarking in progress.

| CP Size | Speedup | Efficiency |
|---------|---------|-----------|
| 1 | 1.0x | 100% |
| 2 | To be tested | To be tested |
| 4 | To be tested | To be tested |
| 8 | To be tested | To be tested |

*Efficiency characteristics will be documented after comprehensive testing*

### CFG Parallelism Speedup

CFG parallelism performance testing in progress.

- 2 GPUs: Performance to be tested
- Communication overhead to be benchmarked
- Scaling characteristics will be documented

## Distributed Setup

### Single Node (Multi-GPU)

```bash
# Automatic detection of all GPUs
torchrun --nproc_per_node=auto test_generate.py
```

### Multi-Node

```bash
# Node 0 (master)
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    test_generate.py

# Node 1-3 (workers)
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=<1,2,3> \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    test_generate.py
```

## Implementation Details

### Distributed Groups

Smart-Diffusion creates separate process groups for different parallelism types:

```python
# CP group: GPUs that share sequence chunks
cp_group = dist.new_group(cp_ranks)

# CFG group: GPUs that handle CFG split
cfg_group = dist.new_group(cfg_ranks)
```

### Sequence Partitioning

```python
def partition_sequence(seq_len, cp_size, rank):
    chunk_size = seq_len // cp_size
    start = rank * chunk_size
    end = start + chunk_size
    return start, end
```

## Troubleshooting

### Communication Hangs

**Symptom**: Training hangs during generation

**Solutions**:
1. Check NCCL version compatibility
2. Verify network connectivity between nodes
3. Enable NCCL debugging: `export NCCL_DEBUG=INFO`
4. Check firewall settings

### Imbalanced Load

**Symptom**: Some GPUs idle while others work

**Solutions**:
1. Ensure sequence length is divisible by CP size
2. Check for uneven task distribution
3. Verify all GPUs have similar performance

### Out of Memory

**Symptom**: OOM errors in distributed setup

**Solutions**:
1. Increase CP size to reduce per-GPU memory
2. Enable low memory mode
3. Reduce batch size or sequence length

## Best Practices

1. **Start Simple**: Test with single GPU before scaling
2. **Profile First**: Identify bottlenecks before adding parallelism
3. **Balance Communication**: More GPUs = more communication overhead
4. **Use CFG Parallelism**: Always enable for 2+ GPU setups with CFG
5. **Monitor Utilization**: Use `nvidia-smi` to check GPU usage

## See Also

- [Architecture Overview](overview.md)
- [Attention Backends](attention-backends.md)
- [Multi-GPU Setup](../user-guide/multi-gpu.md)
