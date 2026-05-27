# FlexCache

FlexCache is ChituDiffusion's feature-cache acceleration module. It wraps the
DiT forward path, stores selected intermediate tensors, and reuses them on later
denoising steps according to a strategy-specific policy.

## Status

- `teacache`: complete baseline. It stores model-output residuals and reuses
  them when accumulated timestep-embedding change stays under the configured
  threshold.
- `pab`: complete baseline. It stores attention outputs and reuses them at fixed
  self-attention and cross-attention broadcast intervals.
- `model`: initially usable FlexCache strategy. It stores model-output residuals
  and uses residual curvature to schedule the next compute anchor.
- `layer`, `attn`, `seq`: under construction. These entrypoints are kept for
  routing and experiments, but their quality/latency behavior is not finalized.

DiTango is independent and lives outside this module.

## Cache Methods

All strategies use the shared `FlexCacheManager.cache` dictionary. Cache memory
is measured directly from cached tensors by `numel * element_size`, with current
and peak bytes recorded after store events.

- TeaCache and `model`: cache `model_output - input`; reuse returns
  `current_input + cached_residual`.
- PAB: cache attention outputs directly; reuse returns the cached attention
  output.
- `layer` and `attn`: cache block or attention-module outputs directly. These
  granularities are still experimental.

Warmup and cooldown mean full compute is forced for the first and last denoising
steps. Their names are intentionally shared across strategies.

## Configuration

User-facing parameters are normalized into `FlexCacheParams`:

- `strategy`: `teacache`, `pab`, `model`, `layer`, `attn`, or `seq`
- `cache_ratio`: `[0, 1]`, used by FlexCache strategies such as `model`
- `warmup`: first `warmup` denoising steps always compute
- `cooldown`: last `cooldown` denoising steps always compute
- `tau_max`: maximum next-compute interval for curvature policies
- `curvature_interval_power`: curvature-to-interval contrast
- `baseline_params`: explicit TeaCache/PAB options

Baseline strategies read their own controls from `baseline_params`; `cache_ratio`
is not translated into TeaCache or PAB parameters.

## Examples

TeaCache:

```python
FlexCacheParams(
    strategy="teacache",
    warmup=7,
    cooldown=3,
    baseline_params={"teacache_thresh": 0.2},
)
```

PAB:

```python
FlexCacheParams(
    strategy="pab",
    warmup=5,
    cooldown=5,
    baseline_params={"skip_self_range": 2, "skip_cross_range": 3},
)
```

FlexCache model:

```python
FlexCacheParams(
    strategy="model",
    cache_ratio=0.5,
    warmup=7,
    cooldown=3,
    tau_max=8,
)
```

## Files

- `flexcache_manager.py`: shared strategy interface, cache dictionary, and cache
  memory accounting.
- `baseline/teacache.py`: TeaCache baseline.
- `baseline/pab.py`: PAB baseline.
- `strategy/model.py`: model-output residual FlexCache strategy.
- `strategy/layer.py`, `strategy/attn.py`, `strategy/seq.py`: experimental
  granularities.
- `core/anchor_cache.py`: shared anchor/cache-ratio planner and PPM policy
  visualization helpers.

Strategy construction happens in `chitu_diffusion/runtime/generator.py`.
Parameter validation and normalization happen in `chitu_diffusion/runtime/task.py`.
