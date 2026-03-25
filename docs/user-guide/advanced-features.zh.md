# 高级功能

本页介绍 Smart-Diffusion 的高级能力与推荐用法。

## FlexCache

FlexCache 通过跨步特征复用，在保持较好画质的同时提升推理效率。

### 统一参数

FlexCache 统一为以下参数组：

- `strategy`: `teacache`、`pab`、`ditango`
- `cache_ratio`: 0 到 1 的质量-效率权衡参数（推荐唯一调节项）
- `warmup`: 前几步完整计算
- `cooldown`: 后几步完整计算

### 推荐示例

```python
from chitu_diffusion.task import DiffusionUserParams, FlexCacheParams

user_params = DiffusionUserParams(
    prompt="A cat walking on grass",
    num_inference_steps=50,
    flexcache_params=FlexCacheParams(
        strategy="pab",
        cache_ratio=0.35,
        warmup=5,
        cooldown=5,
    ),
)
```

### 三种策略

- TeaCache: 适合通用场景，基于时间步变化自适应复用。
- PAB: 基于注意力广播，常用于质量和速度均衡场景。
- DiTango: 基于 ASE 判据进行分组计算/复用控制。

DiTango 当前实现要点：
- Local partition 每步强制计算，并与 group 状态分开合并。
- Anchor gate 与 group 计算计划会在 CFG 正负分支间同步，避免分支决策不一致。
- `cache_ratio` 同时影响 anchor 触发激进程度和全局 ASE 阈值分位更新。

更多细节见 [FlexCache](../advanced/flexcache.md)。

## Context Parallelism

将长序列切分到多张 GPU 上执行。

```bash
python test_generate.py \
    infer.diffusion.cp_size=2 \
    infer.diffusion.up_limit=81
```

## CFG Parallelism

将正负分支拆分到不同 GPU。

自动触发条件：
- `world_size >= 2`
- `guidance_scale > 0`

## 提示词建议

更有效的提示词通常具备：
- 明确主体和动作
- 明确场景和镜头
- 适当细节，不要过度堆砌

## 相关文档

- [性能调优](performance-tuning.md)
- [配置指南](../getting-started/configuration.md)
