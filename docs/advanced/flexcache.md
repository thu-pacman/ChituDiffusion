# FlexCache

FlexCache is a feature caching system that accelerates diffusion generation by reusing computations from previous denoising steps.

## Overview

During diffusion denoising, features change slowly between adjacent steps. FlexCache exploits this temporal similarity to skip redundant computations.

**Benefits**:
- Speedup testing in progress
- Minimal quality loss expected
- No additional VRAM required

## Strategies

### TeaCache

**Principle**: Reuse features when they change slowly

**How it Works**:
1. Compute features for step t
2. Check similarity with step t-1
3. If similar, reuse cached features
4. Otherwise, recompute and update cache

**Configuration**:
```python
params = DiffusionUserParams(
    prompt="A cat walking",
    flexcache="teacache"
)
```

**Performance**:
- Speedup: Performance testing in progress
- Quality impact: Minimal expected
- Best for: Standard generation

### PAB (Pyramid Attention Broadcast)

**Principle**: Broadcast high-level features to lower layers

**How it Works**:
1. Compute full attention at pyramid levels
2. Broadcast results to intermediate layers
3. Skip attention computation at intermediate layers

**Configuration**:
```python
params = DiffusionUserParams(
    prompt="A cat walking",
    flexcache="PAB"
)
```

**Performance**:
- Speedup: Performance testing in progress
- Quality impact: Minimal expected
- Best for: High-quality generation

## Usage

### Basic Usage

```python
from chitu_diffusion.task import DiffusionUserParams

# Enable TeaCache
params = DiffusionUserParams(
    prompt="A sunset over mountains",
    flexcache="teacache"
)
```

### Disable Caching

```python
# No caching (default)
params = DiffusionUserParams(
    prompt="A sunset",
    flexcache=None
)
```

### Comparing Strategies

```python
# Test different strategies
strategies = [None, "teacache", "PAB"]

for strategy in strategies:
    params = DiffusionUserParams(
        prompt="A cat",
        seed=42,  # Same seed for fair comparison
        flexcache=strategy
    )
    
    start = time.time()
    # ... generate ...
    elapsed = time.time() - start
    
    print(f"{strategy}: {elapsed:.2f}s")
```

## Performance Comparison

Performance benchmarking in progress for different caching strategies.

| Strategy | Time | Speedup | Quality Score |
|----------|------|---------|---------------|
| None | Baseline | 1.0x | Baseline |
| TeaCache | To be tested | To be tested | To be tested |
| PAB | To be tested | To be tested | To be tested |

## Implementation Details

### Cache Manager

```python
class FlexCacheManager:
    def __init__(self, strategy: str):
        self.strategy = strategy
        self.cache = {}
    
    def should_reuse(self, step_idx: int) -> bool:
        """Decide whether to reuse cache"""
        if self.strategy == "teacache":
            return self._teacache_reuse_logic(step_idx)
        elif self.strategy == "PAB":
            return self._pab_reuse_logic(step_idx)
        return False
```

### Cache Storage

Cache is stored in GPU memory:
```python
cache = {
    "layer_0": tensor([...]),  # Shape: [B, N, D]
    "layer_12": tensor([...]),
    "layer_24": tensor([...]),
}
```

## Advanced Configuration

### Cache Ratio

Control how aggressively to cache:

```python
# More aggressive caching
args.infer.flexcache_ratio = 0.7  # Cache 70% of steps

# Conservative caching
args.infer.flexcache_ratio = 0.3  # Cache 30% of steps
```

### Cache Layers

Select which layers to cache:

```python
# Cache every 6th layer
args.infer.flexcache_layers = [0, 6, 12, 18, 24, 30, 36]
```

## Best Practices

1. **Enable by Default**: FlexCache has minimal downsides
2. **Start with TeaCache**: Simpler and more robust
3. **Use PAB for Quality**: Slightly better quality preservation
4. **Profile Your Workload**: Test on your specific prompts
5. **Combine with Other Optimizations**: Works well with SageAttention

## Troubleshooting

### No Speedup

**Possible Causes**:
- Very dynamic prompts (benefits less from caching)
- Too few inference steps
- Cache overhead dominates

**Solutions**:
1. Increase `num_inference_steps`
2. Try different cache strategy
3. Profile to confirm cache is active

### Quality Degradation

**Symptoms**: Artifacts or reduced detail

**Solutions**:
1. Disable caching for that prompt
2. Reduce `flexcache_ratio`
3. Use PAB instead of TeaCache

## See Also

- [Performance Tuning](../user-guide/performance-tuning.md)
- [Attention Backends](../architecture/attention-backends.md)
- [Low Memory Mode](low-memory.md)
