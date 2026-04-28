# FlexCache Module

FlexCache is the unified feature reuse acceleration module for ChituDiffusion.

## Unified API

User-facing FlexCache parameters are normalized into one group:

- `strategy`: `teacache`, `pab`, or `ditango`
- `cache_ratio`: required, range `[0, 1]`
- `warmup`: required, first `warmup` denoising steps always run full compute
- `cooldown`: required, last `cooldown` denoising steps always run full compute

`cache_ratio` is the primary quality-efficiency knob:

- `0.0`: quality-first (conservative reuse)
- `1.0`: speed-first (aggressive reuse)

`warmup` and `cooldown` are advanced controls. Most users only need to tune `cache_ratio`.

## Compatibility

Legacy style remains supported:

```python
DiffusionUserParams(
    prompt="A cat walking on grass",
    flexcache="teacache",
)
```

Unified style:

```python
from chitu_diffusion.task import DiffusionUserParams, FlexCacheParams

DiffusionUserParams(
    prompt="A cat walking on grass",
    flexcache_params=FlexCacheParams(
        strategy="ditango",
        cache_ratio=0.45,
        warmup=5,
        cooldown=5,
    ),
)
```

## Internal Mapping

`cache_ratio` is mapped to one strategy-specific control parameter:

- `teacache` -> `teacache_thresh`
- `pab` -> `skip_self_range` (`skip_cross_range` is derived internally)
- `ditango` -> `anchor_rel_err_threshold` + global `ase_threshold` quantile

Other strategy internals are intentionally fixed to keep API compact and predictable.

## Notes for Developers

- Parameter normalization happens in `chitu_diffusion/task.py`.
- Strategy assembly and ratio mapping happen in `chitu_diffusion/generator.py`.
- Strategy implementations are in `chitu_diffusion/flex_cache/strategy/`.
- DiTango implementation is in `chitu_diffusion/flex_cache/strategy/ditango/ditango.py`.
- All strategies now follow the same warmup/cooldown semantics.

## Validation Rules

- `cache_ratio` must be in `[0, 1]`
- `warmup >= 0`
- `cooldown >= 0`
- `warmup + cooldown < num_inference_steps`

Invalid values raise clear parameter errors during task preparation.
