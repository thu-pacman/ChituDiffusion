# Advanced Features

Explore Smart-Diffusion's advanced capabilities for optimal performance.

## FlexCache

FlexCache enables feature reuse across denoising steps, providing significant speedup with minimal quality loss.

### TeaCache

Temporal adaptive caching strategy from CVPR24.

**Usage:**

```python
user_params = DiffusionUserParams(
    prompt="A cat walking on grass",
    flexcache='teacache'
)
```

**How it works:**
- Monitors feature change rates across denoising steps
- Reuses features when changes are minimal
- Typically provides 30-40% speedup

### Pyramid Attention Broadcast (PAB)

Hierarchical attention broadcasting from ICLR25.

**Usage:**

```python
user_params = DiffusionUserParams(
    prompt="A cat walking on grass",
    flexcache='PAB'
)
```

**How it works:**
- Computes attention at coarse scales
- Broadcasts to finer scales
- Typically provides 40-50% speedup

## Context Parallelism

Split long sequences across multiple GPUs.

**Configuration:**

```bash
python test_generate.py \
    infer.diffusion.cp_size=2 \
    infer.diffusion.up_limit=81
```

**Benefits:**
- Handle longer videos (more frames)
- Linear memory scaling
- Near-linear speedup

## CFG Parallelism

Split positive/negative prompts across 2 GPUs.

**Automatic when:**
- `world_size >= 2`
- `guidance_scale > 0`

**Benefits:**
- 2x speedup for CFG
- No extra memory overhead

## Custom Prompts

### Effective Prompt Engineering

**Good prompts:**
- "A fluffy cat walking slowly through tall green grass on a sunny day"
- "Close-up of ocean waves crashing on a rocky shore at sunset"

**Less effective:**
- "cat grass" (too brief)
- "A cat walking on grass and also playing with a ball while a dog runs" (too complex)

### Negative Prompts (Future)

Support for negative prompts coming soon.

## See Also

- [Performance Tuning](performance-tuning.md)
- [Configuration Guide](../getting-started/configuration.md)
