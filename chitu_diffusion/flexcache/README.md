# FlexCache

FlexCache is ChituDiffusion's feature-cache acceleration module. It wraps the
DiT forward path, stores selected intermediate tensors, and reuses them on later
denoising steps according to a strategy-specific policy.

## Status

- `teacache`: complete strategy. It stores model-output residuals and reuses
  them when accumulated timestep-embedding change stays under the configured
  threshold.
- `pab`: complete strategy. It stores attention outputs and reuses them at fixed
  self-attention and cross-attention broadcast intervals.
- `model`: initially usable FlexCache strategy. It stores model-output residuals
  and uses residual curvature to schedule the next compute anchor.
- `cubic`: Cubic-WAN strategy. It defaults to a Wan1.3 832x480 uniform
  partition: latent-space `64x60` blocks, mapped to token-space `32x30`
  blocks for Wan's `(1, 2, 2)` patch size. It then applies Cubic's
  update-frequency optimizer and selective token forward for Wan T2V.
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
- Cubic: keeps per-branch per-layer self-attention K/V caches plus hidden-state
  residual caches for frozen token reuse.
- `layer` and `attn`: cache block or attention-module outputs directly. These
  granularities are still experimental.

Warmup and cooldown mean full compute is forced for the first and last denoising
steps. Their names are intentionally shared across strategies.

## Runtime Metrics

FlexCache records a unified runtime compute metric and writes the per-task
summary to `metrics/flexcache/rank*.json`. Each strategy reports
`baseline_units` and `actual_units` while it runs, and FlexCache accumulates
them in memory until a `task_summary` event is written at the end of the task.
The saved ratio is `(baseline_units - actual_units) / baseline_units`.

Per-event debug logging can be enabled with
`CHITU_FLEXCACHE_COMPUTE_EVENTS=1`, but it is off by default to avoid high-rate
JSON I/O in layer/attention strategies.

These units are strategy-owned proxy units, not hardware FLOPs: model-level
strategies report model-forward units, attention strategies report attention
module units, and Cubic reports token-forward units. The comparison report reads
these runtime summaries directly instead of estimating savings from
strategy-specific side outputs.

## Configuration

User-facing parameters are normalized into `FlexCacheParams`:

- `strategy`: `teacache`, `pab`, `model`, `layer`, `attn`, `seq`, or `cubic`
- `cache_ratio`: `[0, 1]`, used by FlexCache strategies such as `model`
- `warmup`: first `warmup` denoising steps always compute
- `cooldown`: last `cooldown` denoising steps always compute
- `tau_max`: maximum next-compute interval for curvature policies
- `curvature_interval_power`: curvature-to-interval contrast
- `strategy_params`: strategy-specific options for TeaCache, PAB, and future
  strategies

TeaCache and PAB read their own controls from `strategy_params`; `cache_ratio`
is not translated into TeaCache or PAB parameters.

## Examples

TeaCache:

```python
FlexCacheParams(
    strategy="teacache",
    warmup=7,
    cooldown=3,
    strategy_params={"teacache_thresh": 0.2},
)
```

PAB:

```python
FlexCacheParams(
    strategy="pab",
    warmup=5,
    cooldown=5,
    strategy_params={"skip_self_range": 2, "skip_cross_range": 3},
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

Cubic-WAN:

```python
FlexCacheParams(
    strategy="cubic",
    cache_ratio=0.5,
    warmup=7,
    cooldown=3,
    tau_max=8,
    strategy_params={"target_speedup": 2.0, "partition_mode": "wan13_832x480_uniform"},
)
```

## Files

- `flexcache_manager.py`: shared strategy interface, cache dictionary, and cache
  memory accounting.
- `strategy/teacache.py`: TeaCache strategy.
- `strategy/pab.py`: PAB strategy.
- `strategy/model.py`: model-output residual FlexCache strategy.
- `strategy/cubic.py`: Cubic-WAN strategy wrapper and selective forward adapter.
- `strategy/layer.py`, `strategy/attn.py`, `strategy/seq.py`: experimental
  granularities.
- `core/anchor_cache.py`: shared anchor/cache-ratio planner and PPM policy
  visualization helpers.

Strategy construction happens in `chitu_diffusion/runtime/generator.py`.
Parameter validation and normalization happen in `chitu_diffusion/runtime/task.py`.
