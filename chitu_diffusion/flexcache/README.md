# FlexCache Module

FlexCache is ChituDiffusion's feature-cache acceleration module. Its primary
methods are curvature-driven anchor-cache strategies; TeaCache and PAB are kept
as baseline implementations for comparison. DiTango remains independent in
`chitu_diffusion/ditango/`.

## Unified API

User-facing FlexCache parameters are normalized into `FlexCacheParams`:

- `strategy`: main methods `model`, `layer`, `attn`, `seq`; baselines `teacache`, `pab`
- `cache_ratio`: range `[0, 1]`, used by curvature-driven main methods
- `warmup`: first `warmup` denoising steps always run full compute
- `cooldown`: last `cooldown` denoising steps always run full compute
- `tau_max`: maximum curvature-assigned interval for main methods
- `curvature_interval_power`: curvature-to-interval contrast for main methods
- `baseline_params`: explicit TeaCache/PAB parameters

`cache_ratio` is not mapped into baseline parameters. Baselines read their own
controls from `baseline_params`.

## Examples

Curvature-driven model cache:

```python
FlexCacheParams(
    strategy="model",
    cache_ratio=0.45,
    warmup=5,
    cooldown=5,
    tau_max=8,
)
```

TeaCache baseline:

```python
FlexCacheParams(
    strategy="teacache",
    warmup=5,
    cooldown=5,
    baseline_params={"teacache_thresh": 0.2},
)
```

PAB baseline:

```python
FlexCacheParams(
    strategy="pab",
    warmup=5,
    cooldown=5,
    baseline_params={"skip_self_range": 2, "skip_cross_range": 3},
)
```

## Developer Notes

- Parameter normalization happens in `chitu_diffusion/runtime/task.py`.
- Strategy assembly happens in `chitu_diffusion/runtime/generator.py`.
- Main strategies are in `chitu_diffusion/flexcache/strategy/`.
- Baselines are in `chitu_diffusion/flexcache/baseline/`.
- Shared anchor-cache helpers are in `chitu_diffusion/flexcache/core/`.

## Validation Rules

- `cache_ratio` must be in `[0, 1]`
- `warmup >= 0`
- `cooldown >= 0`
- `warmup + cooldown < num_inference_steps`
