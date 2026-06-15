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
- `cubic`: Cubic-WAN strategy. It defaults to a Wan1.3 832x480 uniform
  partition: latent-space `64x60` blocks, mapped to token-space `32x30`
  blocks for Wan's `(1, 2, 2)` patch size. It then applies Cubic's
  update-frequency optimizer and selective token forward for Wan T2V.
- `taylorseer`: model-specific TaylorSeer strategy for Flux and Wan. It caches
  local module outputs, stores Taylor factors on full refresh steps, and
  predicts skipped steps with the Taylor expansion.
- `blockdance`: BlockDance-style train-free layerwise cache. It keeps early steps
  full, then groups the active window by `group_size`; each group caches the
  output after `boundary_block`, and reuse steps skip shallow/mid blocks while
  recomputing deeper blocks.

DiTango is independent and lives outside this module.

## Cache Methods

All strategies use the shared `FlexCacheManager.cache` dictionary. Cache memory
is measured directly from cached tensors by `numel * element_size`, with current
and peak bytes recorded after store events.

- TeaCache: caches `model_output - input`; reuse returns
  `current_input + cached_residual`.
- PAB: cache attention outputs directly; reuse returns the cached attention
  output.
- Cubic: keeps per-branch per-layer self-attention K/V caches plus hidden-state
  residual caches for frozen token reuse.
- TaylorSeer: caches per-branch per-layer module Taylor factors. Wan caches
  self-attention, cross-attention, and FFN outputs; Flux double blocks cache
  image/text attention and MLP outputs; Flux single blocks cache their fused
  local module output. Distributed ranks cache only their local tensor chunk.
- BlockDance: caches the boundary block hidden state and reuses it as the
  next block input, skipping shallow/mid blocks on reuse steps.

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
JSON I/O in module-level strategies.

These units are strategy-owned proxy units, not hardware FLOPs: TeaCache reports
model-forward units, PAB reports attention-module units, BlockDance reports
transformer-block units, TaylorSeer reports module-output units, and Cubic reports
token-forward units. The comparison report reads these runtime summaries
directly instead of estimating savings from strategy-specific side outputs.

## Configuration

`FlexCacheParams` is the base parameter class. Use a concrete subclass for new
code so each strategy exposes only its own controls:

- `TeaCacheParams`: `warmup`, `cooldown`, `teacache_thresh`, `coefficients`,
  `use_ref_steps`
- `PABParams`: `warmup`, `cooldown`, `skip_self_range`, `skip_cross_range`
- `BlockDanceParams`: `warmup`, `cooldown`, `boundary_block`, `group_size`,
  `start_fraction`, `end_fraction`
- `CubicParams`: `target_speedup`, `warmup`, `cooldown`, `tau_max`,
  optional `block_size`
- `TaylorSeerParams`: `warmup`, `cooldown`, `fresh_threshold`, `max_order`,
  `first_enhance`
- `DiTangoParams`: `cache_ratio`, `warmup`, `cooldown`, `tau_max`,
  `curvature_interval_power`, `intra_group_size_limit`

Dictionary inputs must use the concrete field names of the selected strategy.

## Examples

TeaCache:

```python
TeaCacheParams(
    warmup=1,
    cooldown=1,
    teacache_thresh=0.6,
)
```

PAB:

```python
PABParams(
    warmup=5,
    cooldown=5,
    skip_self_range=2,
    skip_cross_range=3,
)
```

Cubic-WAN:

```python
CubicParams(
    target_speedup=2.0,
    warmup=7,
    cooldown=3,
    tau_max=8,
)
```

BlockDance:

```python
BlockDanceParams(
    boundary_block=20,
    group_size=2,
    start_fraction=0.40,
    end_fraction=0.95,
)
```

TaylorSeer:

```python
TaylorSeerParams(
    warmup=7,
    cooldown=3,
    fresh_threshold=5,
    max_order=1,
    first_enhance=1,
)
```

## Files

- `flexcache_manager.py`: shared strategy interface, cache dictionary, and cache
  memory accounting.
- `strategy/teacache.py`: TeaCache strategy.
- `strategy/pab.py`: PAB strategy.
- `strategy/blockdance.py`: BlockDance-style layerwise cache adapter.
- `strategy/cubic.py`: Cubic-WAN strategy wrapper and selective forward adapter.
- `strategy/taylorseer.py`: TaylorSeer Flux/Wan module-cache adapter.
- `core/anchor_cache.py`: shared anchor/cache-ratio planner and PPM policy
  visualization helpers.

Strategy construction happens in `chitu_diffusion/runtime/generator.py`.
Parameter validation and normalization happen in `chitu_diffusion/runtime/task.py`.
