# FlexCache 模块

FlexCache 是 ChituDiffusion 的 feature cache 加速模块。主方法统一采用
curvature-driven anchor-cache 范式；TeaCache 和 PAB 作为 baseline 保留用于对比。
DiTango 保持在 `chitu_diffusion/ditango/` 中独立实现。

## 统一 API

面向用户的参数归一到 `FlexCacheParams`：

- `strategy`: 主方法 `model`, `layer`, `attn`, `seq`；baseline `teacache`, `pab`
- `cache_ratio`: 范围 `[0, 1]`，仅用于 curvature 主方法
- `warmup`: 前 `warmup` 步始终完整计算
- `cooldown`: 后 `cooldown` 步始终完整计算
- `tau_max`: 主方法中曲率分配 interval 的上限
- `curvature_interval_power`: 曲率到 interval 映射的对比度
- `baseline_params`: TeaCache/PAB 自己的显式参数

`cache_ratio` 不再映射到 baseline 参数。baseline 只读取 `baseline_params`。

## 示例

curvature-driven model cache:

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

## 开发说明

- 参数归一入口在 `chitu_diffusion/runtime/task.py`。
- 策略分发在 `chitu_diffusion/runtime/generator.py`。
- 主策略位于 `chitu_diffusion/flexcache/strategy/`。
- baseline 位于 `chitu_diffusion/flexcache/baseline/`。
- 共享 anchor-cache 工具位于 `chitu_diffusion/flexcache/core/`。

## 参数校验规则

- `cache_ratio` 必须在 `[0, 1]`
- `warmup >= 0`
- `cooldown >= 0`
- `warmup + cooldown < num_inference_steps`
