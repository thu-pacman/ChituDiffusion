# FlexCache

FlexCache 是 Smart-Diffusion 中统一的特征复用加速框架。

## 统一用户 API

FlexCache 使用一组专门参数：

- `strategy`: `teacache`、`pab` 或 `ditango`
- `cache_ratio`: 必填，范围 `[0, 1]`
- `warmup`: 必填，前 `warmup` 步始终完整计算
- `cooldown`: 必填，后 `cooldown` 步始终完整计算

`cache_ratio` 是推荐的唯一调节旋钮：

- `0.0`: 质量优先，缓存复用更保守
- `1.0`: 速度优先，缓存复用更激进

多数场景只需要调 `cache_ratio`，`warmup` 和 `cooldown` 属于高级参数。

## 策略映射

同一 `cache_ratio` 尺度会映射到各策略的核心控制参数：

| 策略 | 内部主控参数 | 映射方向 |
|------|--------------|----------|
| TeaCache | `teacache_thresh` | ratio 越高，阈值越大，复用越多 |
| PAB | `skip_self_range` | ratio 越高，跳步越大，复用越多 |
| DiTango | `anchor_rel_err_threshold` + 全局 `ase_threshold` 分位 | ratio 越高，越偏向复用 |

其他策略内部参数默认写死，以保证接口一致性。

DiTango 运行时说明：
- `cache_ratio` 同时参与 anchor gate 判定和全局 ASE 阈值分位更新。
- 为保证稳定性，local partition 每步都会重算。
- 策略实现已迁移至 `chitu_diffusion/flex_cache/strategy/ditango/ditango.py`。
- 会输出合并决策可视化：`<output_dir>/ditango_policy_step_layer_group.ppm`。

## 使用方式

### 推荐写法

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

### 兼容旧写法

```python
params = DiffusionUserParams(
    prompt="A cat walking on grass",
    flexcache="teacache",
)
```

### 关闭 FlexCache

```python
params = DiffusionUserParams(
    prompt="A cat walking on grass",
    flexcache=None,
)
```

## 系统开关

在启动配置中全局开启 FlexCache：

```yaml
infer:
  enable_flexcache: true
```

若 `enable_flexcache` 为 false，请求侧 FlexCache 配置会被忽略。

## 参数校验

任务准备阶段会校验：

- `cache_ratio` 必须在 `[0, 1]`
- `warmup >= 0`
- `cooldown >= 0`
- `warmup + cooldown < num_inference_steps`

## 相关文档

- [高级功能](../user-guide/advanced-features.md)
- [配置指南](../getting-started/configuration.md)
- [低内存模式](low-memory.md)
