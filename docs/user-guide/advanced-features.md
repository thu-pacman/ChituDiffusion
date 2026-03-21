# Advanced Features

Explore Smart-Diffusion's advanced capabilities for optimal performance.

## FlexCache

FlexCache enables feature reuse across denoising steps, providing significant speedup with minimal quality loss.

### Unified Parameters

Use the dedicated FlexCache parameter group:

- `strategy`: `teacache`, `pab`, `ditango`
- `cache_ratio`: `0` to `1` quality-efficiency tradeoff (`0` quality-first, `1` speed-first)
- `warmup`: first N steps always full compute
- `cooldown`: last N steps always full compute

Recommended API:

```python
from chitu_diffusion.task import DiffusionUserParams, FlexCacheParams

user_params = DiffusionUserParams(
    prompt="A cat walking on grass",
    num_inference_steps=50,
    flexcache_params=FlexCacheParams(
        strategy="teacache",
        cache_ratio=0.4,
        warmup=5,
        cooldown=5,
    ),
)
```

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

### DiTango

ASE-guided grouped compute/reuse strategy.

**Usage:**

```python
user_params = DiffusionUserParams(
    prompt="A cat walking on grass",
    flexcache='ditango'
)
```

**How it works:**
- Estimates group-level reuse confidence from ASE
- Recomputes groups when confidence is insufficient
- Uses periodic anchor updates to control drift

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
