# Performance Tuning

Optimize Smart-Diffusion for maximum performance.

## Quick Wins

### 1. Use SageAttention

```bash
infer.attn_type=sage
```

**Speedup:** ~2x
**Quality loss:** Minimal

### 2. Enable FlexCache

```python
user_params = DiffusionUserParams(
    prompt="...",
    flexcache='teacache'
)
```

**Speedup:** 30-40%
**Quality loss:** Minimal

### 3. Reduce Inference Steps

```python
num_inference_steps=30  # Instead of 50
```

**Speedup:** 40%
**Quality loss:** Slight

## GPU Utilization

### Check Utilization

```bash
nvidia-smi dmon -s u
```

**Target:** >80% GPU utilization

**If low:**
1. Increase batch size (future feature)
2. Use context parallelism
3. Check CPU bottlenecks

## Memory Optimization

### Reduce Memory Usage

**Strategy 1:** Low memory mode
```bash
infer.diffusion.low_mem_level=2
```

**Strategy 2:** Lower resolution
```python
height=480, width=848, num_frames=61
```

**Strategy 3:** SageAttention
```bash
infer.attn_type=sage
```

## Benchmarking

### Measure Performance

```python
import time

start = time.time()
while not DiffusionTaskPool.all_finished():
    chitu_generate()
elapsed = time.time() - start

print(f"Generation took {elapsed:.2f} seconds")
```

### Expected Performance

| Model | Resolution | Frames | Steps | A100 (40GB) | H100 (80GB) |
|-------|------------|--------|-------|-------------|-------------|
| 1.3B | 480x848 | 81 | 50 | ~120s | ~60s |
| 14B | 480x848 | 81 | 50 | ~240s | ~120s |
| 14B | 720x1280 | 121 | 50 | ~600s | ~300s |

*With SageAttention: ~50% faster*

## Multi-GPU Scaling

### Context Parallelism Efficiency

| GPUs | Speedup | Efficiency |
|------|---------|------------|
| 1 | 1.0x | 100% |
| 2 | 1.9x | 95% |
| 4 | 3.6x | 90% |
| 8 | 6.8x | 85% |

### CFG Parallelism

| GPUs | Speedup |
|------|---------|
| 2 | 2.0x |

## Profiling

### Enable Debug Mode

```bash
export CHITU_DEBUG=1
python test_generate.py
```

Shows detailed timing information.

## See Also

- [Multi-GPU Setup](multi-gpu.md)
- [Configuration Guide](../getting-started/configuration.md)
