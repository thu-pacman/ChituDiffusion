# FlexCache

[English](README.md) | [简体中文](README.zh-CN.md)

FlexCache is a generate-only, request-local cache layer. It discovers stable
model, block, and leaf sites once, then installs temporary hooks for one
`generate()` request. Model and pipeline source files do not contain
strategy-specific branches.

FlexCache is designed around two non-negotiable properties:

1. A strategy is isolated from the Diffusers model implementation and from
   other requests.
2. Fresh/reuse decisions are orthogonal to parallel layout: CFP and CP ranks
   must take identical control-flow paths.

Persistent EPE serving rejects every non-`none` cache strategy. This boundary
avoids sharing mutable cache state across queued requests or dynamic lanes.

## Supported strategies

| Strategy | Reused value | Calibrated model support | Important constraint |
| --- | --- | --- | --- |
| MagCache | Full image-token backbone residual | FLUX.1, Qwen-Image, Wan 2.1 T2V 1.3B | Uses official 50-step profiles and a branch-shared schedule |
| MeanCache | CFG-combined prediction at the scheduler boundary | Z-Image, Qwen-Image | Official 50-step fresh/JVP tables only |
| TeaCache | Full block-group residual | FLUX.1, Wan T2V 1.3B/14B | Threshold is meaningful only with a matching calibrated polynomial |
| TaylorSeer | Predicted pre-gate block submodule outputs | Z-Image, FLUX.1, Qwen-Image, Wan | Memory grows with blocks, cached sites, and Taylor order |
| PAB | Attention outputs by attention type | Z-Image, FLUX.1, Qwen-Image, Wan | Uses the maintained simplified period rules |

Unsupported model, step-count, profile, or parameter combinations fail early.
In particular, MagCache has no official Z-Image profile and custom ratios are
rejected for its heterogeneous recursive backbone.

## Usage

Use FlexCache through `chitu generate`:

```bash
chitu generate \
  --model flux1 \
  --model-path /path/to/FLUX.1-dev \
  --steps 50 \
  --cache-strategy magcache \
  --output outputs/flux1-magcache.png
```

Strategy-specific arguments use their own prefix:

```bash
# Official MeanCache schedule.
--cache-strategy meancache --meancache-fresh-steps 25

# TeaCache with a model-calibrated threshold.
--cache-strategy teacache --teacache-threshold 0.4

# Maintained TaylorSeer speed/quality default.
--cache-strategy taylorseer --taylorseer-fresh-threshold 3
```

Common warmup/cooldown flags apply only to strategies that use the common
fresh-step envelope. MagCache owns its schedule and rejects those overrides.
Configuration is represented by immutable dataclasses in `epac/cache.py` and
is validated before hooks are installed.

## Layout

```text
flexcache/
├── contracts.py       # CacheStrategy protocol, step context, statistics
├── session.py         # request lifecycle and temporary hook installation
├── spec.py            # model-family discovery and site/probe contracts
├── tree.py            # tensor-tree clone/arithmetic/size helpers
└── strategies/
    ├── base.py        # no-op defaults for every hook
    ├── factory.py     # CacheConfig -> strategy instance
    ├── magcache.py
    ├── magcache_profiles.py
    ├── meancache.py
    ├── pab.py
    ├── taylorseer.py
    └── teacache.py
```

All mutable tensors and counters must live on the strategy instance. A
`CacheSession` creates request-local state and, for serial CFG, independent
strategy instances per branch.

## Request lifecycle

1. `CacheConfig` is validated and `strategies/factory.py` creates a strategy.
2. `FlexCacheModelSpec.discover()` resolves stable model, block, and leaf sites.
3. `CacheSession` installs temporary wrappers on that pipeline/model instance.
4. A timestep signature advances the denoise step and separates serial CFG
   branches.
5. Lookup hooks may bypass work; store hooks update request-local cache state
   and statistics.
6. Session exit restores every wrapper, including when setup or generation
   raises.

Request-level algorithms such as MeanCache patch only the selected pipeline
instance. FlexCache never patches a pipeline class or stores mutable tensors in
global/class state.

## Hook levels

Choose the narrowest hook that represents the value defined by the algorithm:

- `model_lookup/store`: make one decision per transformer invocation. Use this
  for step scheduling and whole-backbone state.
- `block_lookup/store`: reuse or reconstruct a complete transformer block.
- `leaf_lookup/store`: reuse attention, MLP, or projection outputs exposed as
  `LeafSite`s.
- request-level `denoise_step`: only for algorithms such as MeanCache that need
  the CFG-combined prediction and scheduler state. Set `request_level = True`
  and implement `denoise_step(...)`; `CacheSession` installs the temporary
  denoise hook.

The lookup result is `(hit, output)`. Returning `True` skips the wrapped
module. A model-level strategy may return `False` while setting internal state
that causes later block or leaf hooks to hit, as MagCache, TeaCache, and
TaylorSeer do.

## Adding a strategy

1. Add its immutable user parameters as a dataclass in `epac/cache.py`, include
   it in `CacheParams`, `_PARAM_TYPES`, and `CacheStrategyName`, and validate
   unsupported combinations early.
2. Add `strategies/<name>.py`. Subclass `BaseCacheStrategy` and override only
   the hooks the algorithm needs:

   ```python
   from ..contracts import CacheStepContext
   from ..spec import LeafSite
   from ..tree import TensorTree, tree_clone
   from .base import BaseCacheStrategy


   class ExampleStrategy(BaseCacheStrategy):
       def __init__(self, params: ExampleConfig, **common: int) -> None:
           super().__init__()
           self.params = params
           self.cache: dict[str, TensorTree] = {}

       def begin(self, *, total_steps: int, model_spec: FlexCacheModelSpec) -> None:
           super().begin(total_steps=total_steps, model_spec=model_spec)
           self.cache.clear()

       def leaf_lookup(
           self,
           context: CacheStepContext,
           site: LeafSite,
           args: tuple[object, ...],
           kwargs: dict[str, object],
       ) -> tuple[bool, TensorTree | None]:
           value = self.cache.get(site.site_id)
           return (True, tree_clone(value)) if value is not None else (False, None)
   ```

3. Register construction in `strategies/factory.py` and export the class from
   `strategies/__init__.py`.
4. Add CLI arguments in `examples/cache_args.py` only if the strategy is
   intended for example scripts.
5. Add focused tests in `test/test_chitu_diffusion_flexcache.py` for scheduling,
   request isolation, cache statistics, tensor-tree outputs, and unsupported
   configurations. Then run:

   ```bash
   python -m pytest test/test_chitu_diffusion_flexcache.py -q
   python -m ruff check chitu_diffusion/flexcache \
     chitu_diffusion/epac/cache.py test/test_chitu_diffusion_flexcache.py
   ```

## Parallelism requirements

Cache control flow must be orthogonal to CFP and CP:

- Base a fresh/reuse decision on replicated inputs such as step index,
  timestep, or a full-sequence modulation probe.
- Never decide independently from rank-local token shards.
- All ranks in a lane must enter or skip collectives identically.
- Cache post-all-gather, post-CFG values when the algorithm is defined on the
  guided prediction.
- Keep every request's state isolated; do not monkey-patch global model state
  or store tensors on strategy classes.

If an adaptive metric cannot be made rank-identical, synchronize the scalar
decision explicitly or reject that parallel mode. A small divergence from a
reference implementation is preferable to divergent collective control flow.

## Model support

`FlexCacheModelSpec.discover()` in `spec.py` is the only place that should know
model-family module names. When a strategy needs a new reusable site or probe,
extend the model spec first and keep the strategy expressed in terms of
`BlockSite`, `LeafSite`, and `FlexCacheModelSpec` operations. Do not add
strategy-specific conditionals to model forward methods.

## Validation requirements

A strategy is not complete when unit tests pass. Validate it in layers:

1. **Contract tests:** schedule boundaries, hit/miss statistics, tensor-tree
   output, request/CFG isolation, cleanup after exceptions, and rejected
   configurations.
2. **Reference semantics:** on the same Diffusers backend, dtype, scheduler,
   seed, prompt, and dimensions, compare fresh/reuse steps and outputs with the
   reference implementation.
3. **Single-GPU performance:** report end-to-end and, where possible, DiT-only
   latency together with cache memory.
4. **Parallel smoke tests:** run static CP and every supported CFP layout;
   verify that every rank makes the same decision and completes collectives.
5. **Quality sweep:** compare each candidate acceleration point against its own
   uncached backend baseline. Do not compare only across different attention
   backends or dtypes.

Text summaries and reproducible commands belong under
`outputs/flexcache/<run-id>/result.md`. Generated media and raw temporary data
should remain untracked.

Current comparison reports:

- [`MagCache`](../../outputs/flexcache/magcache_compare_20260805/result.md)
- [`MeanCache`](../../outputs/flexcache/meancache_compare_20260804/result.md)

## Known trade-offs

- Caching changes the numerical trajectory. A speedup is not evidence of
  acceptable quality; model/step/profile changes require re-evaluation.
- TaylorSeer may consume gigabytes on video workloads because it stores
  multiple tensors per submodule and derivative order.
- TeaCache's published coefficients are tied to their probe and calibration
  schedule. The reference coefficients are calibrated at 50 steps and can
  degrade at shorter schedules.
- Backend, attention kernel, and dtype differences can dominate pixel-level
  comparisons. Always compare a cache run to the uncached baseline of the same
  backend.
