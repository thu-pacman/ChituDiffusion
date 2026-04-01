# FlexCache

FlexCache is Smart-Diffusion's unified feature reuse acceleration framework.

## Unified User API

FlexCache is configured with one dedicated parameter group:

- `strategy`: `teacache`, `pab`, or `ditango`
- `cache_ratio`: required, range `[0, 1]`
- `warmup`: required, first `warmup` denoising steps always run full compute
- `cooldown`: required, last `cooldown` denoising steps always run full compute

`cache_ratio` is the only recommended user tuning knob:

- `0.0`: quality-first, conservative cache reuse
- `1.0`: speed-first, aggressive cache reuse

Most users should only tune `cache_ratio`. `warmup` and `cooldown` are advanced controls.

## Strategy Mapping

The same `cache_ratio` scale is mapped to one strategy-specific core parameter:

| Strategy | Internal control | Mapping direction |
|----------|------------------|-------------------|
| TeaCache | `teacache_thresh` | higher ratio -> larger threshold -> more reuse |
| PAB | `skip_self_range` | higher ratio -> larger skip range -> more reuse |
| DiTango | `anchor_rel_err_threshold` + global `ase_threshold` quantile | higher ratio -> stronger reuse preference |

Other strategy internals are fixed by design for API consistency.

DiTango runtime notes:
- `cache_ratio` is used by both anchor gating and global ASE-threshold quantile update.
- Local partition is always recomputed each step for stability.
- The strategy implementation moved to `chitu_diffusion/flex_cache/strategy/ditango/ditango.py`.
- A merged decision map is written to `<output_dir>/ditango_policy_step_layer_group.ppm`.

## Usage

### Recommended style

```python
from chitu_diffusion.task import DiffusionUserParams, FlexCacheParams

params = DiffusionUserParams(
    prompt="A cat walking on grass",
    num_inference_steps=50,
    flexcache_params=FlexCacheParams(
        strategy="ditango",
        cache_ratio=0.45,
        warmup=5,
        cooldown=5,
    ),
)
```

### Legacy compatible style

```python
params = DiffusionUserParams(
    prompt="A cat walking on grass",
    flexcache="teacache",
)
```

### Disable FlexCache

```python
params = DiffusionUserParams(
    prompt="A cat walking on grass",
    flexcache=None,
)
```

## System Switch

Enable FlexCache globally at startup:

```yaml
infer:
  enable_flexcache: true
```

If `enable_flexcache` is false, task-level FlexCache requests are ignored.

## Parameter Validation

FlexCache validates parameters during task preparation:

- `cache_ratio` must be in `[0, 1]`
- `warmup >= 0`
- `cooldown >= 0`
- `warmup + cooldown < num_inference_steps`

## See Also

- [Advanced Features](../user-guide/advanced-features.md)
- [Configuration Guide](../getting-started/configuration.md)
- [Low Memory Mode](low-memory.md)
