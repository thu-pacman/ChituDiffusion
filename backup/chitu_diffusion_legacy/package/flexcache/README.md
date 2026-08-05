# FlexCache

FlexCache is ChituDiffusion's feature-cache acceleration module. It wraps the
DiT forward path, stores selected intermediate tensors, and reuses them on later
denoising steps according to a strategy-specific policy.

## Status

- `meancache`: complete step-level strategy for Flux, Qwen-Image, and Wan. It caches
  full noise predictions and can reuse them through the MeanCache JVP /
  average-velocity update outside the model forward.
- `cubic`: Jano-derived region-aware / token-selective strategy. Jano has an
  independent repository at `chen-yy20/Jano`; the ChituDiffusion integration is
  still being updated and tested. The current Wan path defaults to a Wan1.3
  832x480 uniform partition: latent-space `64x60` blocks, mapped to token-space
  `32x30` blocks for Wan's `(1, 2, 2)` patch size.
- `teacache`: complete strategy. It stores model-output residuals and reuses
  them when accumulated timestep-embedding change stays under the configured
  threshold.
- `taylorseer`: model-specific TaylorSeer strategy for Flux and Wan. It caches
  local module outputs, stores Taylor factors on full refresh steps, and
  predicts skipped steps with the Taylor expansion.
- `blockdance`: BlockDance-style train-free layerwise cache. It keeps early steps
  full, then groups the active window by `group_size`; each group caches the
  output after `boundary_block`, and reuse steps skip shallow/mid blocks while
  recomputing deeper blocks.
- `pab`: complete strategy. It stores attention outputs and reuses them at fixed
  self-attention and cross-attention broadcast intervals.

DiTango depends on distributed environments and lives outside this module.

## Technical Sources

FlexCache is an integration layer, not a new paper for each policy. The table
below records the technical source each strategy follows and how it is adapted in
this repository.

| Strategy | Technical source | Venue / status | ChituDiffusion adaptation |
| --- | --- | --- | --- |
| `meancache` | [MeanCache: From Instantaneous to Average Velocity for Accelerating Flow Matching Inference](https://openreview.net/forum?id=GMCyL7Xs9R) | ICLR 2026 | Step-level full-noise cache with optional JVP reuse for Flux, Qwen-Image, and Wan. |
| `teacache` | [Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Timestep_Embedding_Tells_Its_Time_to_Cache_for_Video_Diffusion_CVPR_2025_paper.html) | CVPR 2025 | Model-output residual cache driven by accumulated timestep-embedding change. |
| `pab` | [Real-Time Video Generation with Pyramid Attention Broadcast](https://openreview.net/forum?id=hDBrQ4DApF) | ICLR 2025 | Attention-output broadcast at fixed self/cross-attention intervals. |
| `cubic` | [Jano: Adaptive Diffusion Generation with Early-stage Convergence Awareness](https://arxiv.org/abs/2603.00519) | CVPR Findings 2026 | Region-aware / token-selective forward path exposed as the `cubic` FlexCache strategy. |
| `taylorseer` | [From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_From_Reusing_to_Forecasting_Accelerating_Diffusion_Models_with_TaylorSeers_ICCV_2025_paper.pdf) | ICCV 2025 | Module-output cache plus Taylor expansion forecast for skipped denoising steps. |
| `blockdance` | [BlockDance: Reuse Structurally Similar Spatio-Temporal Features to Accelerate Diffusion Transformers](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_BlockDance_Reuse_Structurally_Similar_Spatio-Temporal_Features_to_Accelerate_Diffusion_Transformers_CVPR_2025_paper.pdf) | CVPR 2025 | Layerwise block reuse around a configurable boundary block and active denoising window. |
| `ditango` | [DiTango: Cost-Effective Parallel Diffusion Generation with Selective Attention State Reuse](https://hpdc.sci.utah.edu/2026/program.html) | HPDC 2026 | Independent planner/runtime path for cache-accelerated parallel attention state reuse. |

## Cache Methods

All strategies use the shared `FlexCacheManager.cache` dictionary. Cache memory
is measured directly from cached tensors by `numel * element_size`, with current
and peak bytes recorded after store events.

- TeaCache: caches `model_output - input`; reuse returns
  `current_input + cached_residual`.
- PAB: cache attention outputs directly; reuse returns the cached attention
  output.
- MeanCache: caches the full noise prediction at fresh steps; reuse returns a
  predicted noise value from the cached state and scheduler sigmas.
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
code so each strategy exposes only its own controls. All strategies share:

- `warmup`: number of initial denoising steps forced to full compute.
- `cooldown`: number of final denoising steps forced to full compute.

Strategy-specific fields:

- `MeanCacheParams`: `fresh_steps`, `use_jvp`
- `CubicParams`: `target_speedup`, `tau_max`, optional `block_size`,
  optional `uniform_square_min_splits`
- `TeaCacheParams`: `teacache_thresh`, `coefficients`, `use_ref_steps`
- `TaylorSeerParams`: `fresh_threshold`, `max_order`, `first_enhance`
- `BlockDanceParams`: `boundary_block`, `group_size`, `start_fraction`,
  `end_fraction`
- `PABParams`: `skip_self_range`, `skip_cross_range`
- `DiTangoParams`: `cache_ratio`, `anchor_interval`, `tau_max`,
  `curvature_interval_power`, `intra_group_size_limit`, plus experimental
  groupwise attention-state reuse controls in `params.py`

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

MeanCache:

```python
MeanCacheParams(
    warmup=0,
    cooldown=0,
    fresh_steps=25,
    use_jvp=True,
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
- `strategy/meancache.py`: MeanCache step-level noise-prediction cache.
- `strategy/blockdance.py`: BlockDance-style layerwise cache adapter.
- `strategy/cubic.py`: Cubic-WAN strategy wrapper and selective forward adapter.
- `strategy/taylorseer.py`: TaylorSeer Flux/Wan module-cache adapter.
- `core/anchor_cache.py`: shared anchor/cache-ratio planner and PPM policy
  visualization helpers.

Strategy construction happens in `chitu_diffusion/runtime/generator.py`.
Parameter validation and normalization happen in `chitu_diffusion/runtime/task.py`.
