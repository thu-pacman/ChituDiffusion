# FlexCache 模块

FlexCache 是 Smart-Diffusion 中统一的特征复用加速模块。

## 统一 API

面向用户的 FlexCache 参数统一为一组：

- `strategy`: `teacache`、`pab` 或 `ditango`
- `cache_ratio`: 必填，范围 `[0, 1]`
- `warmup`: 必填，前 `warmup` 步始终完整计算
- `cooldown`: 必填，后 `cooldown` 步始终完整计算

`cache_ratio` 是核心的质量-效率调节参数：

- `0.0`: 质量优先（更保守的复用）
- `1.0`: 速度优先（更激进的复用）

`warmup` 和 `cooldown` 属于高级参数。多数场景只需调 `cache_ratio`。

## 兼容性

旧写法仍然可用：

```python
DiffusionUserParams(
    prompt="A cat walking on grass",
    flexcache="teacache",
)
```

推荐新写法：

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

## 内部映射

`cache_ratio` 会映射到每种策略的一个关键控制参数：

- `teacache` -> `teacache_thresh`
- `pab` -> `skip_self_range`（`skip_cross_range` 在内部固定派生）
- `ditango` -> `ase_threshold`

其他策略细节参数默认写死，以保持接口简洁和行为稳定。

## 开发说明

- 参数归一入口在 `chitu_diffusion/task.py`。
- 策略分发与 ratio 映射在 `chitu_diffusion/generator.py`。
- 策略实现位于 `chitu_diffusion/flex_cache/strategy/`。
- 三种策略统一采用同一 warmup/cooldown 语义。

## 参数校验规则

- `cache_ratio` 必须在 `[0, 1]`
- `warmup >= 0`
- `cooldown >= 0`
- `warmup + cooldown < num_inference_steps`

参数非法会在任务准备阶段给出明确错误。
