# FlexCache

[English](README.md) | [简体中文](README.zh-CN.md)

FlexCache 是一个 generate-only、request-local 的缓存层。它只发现一次稳定的 model、
block 和 leaf site，然后为单次 `generate()` 请求安装临时 hook。模型和 pipeline 源码中
不包含策略专用分支。

FlexCache 有两个不可放宽的设计约束：

1. 策略必须与 Diffusers 模型实现解耦，并与其他请求隔离。
2. fresh/reuse 判定必须与并行布局正交：CFP 和 CP rank 必须执行相同控制流。

常驻 EPE 服务会拒绝所有非 `none` 缓存策略，避免可变 cache state 跨排队请求或动态 lane
共享。

## 支持的策略

| 策略 | 复用值 | 已标定模型 | 关键约束 |
| --- | --- | --- | --- |
| MagCache | 完整 image-token backbone residual | FLUX.1、Qwen-Image、Wan 2.1 T2V 1.3B | 使用官方 50-step profile 和分支共享日程 |
| MeanCache | scheduler 边界的 CFG 合并 prediction | Z-Image、Qwen-Image | 仅支持官方 50-step fresh/JVP 表 |
| TeaCache | 完整 block-group residual | FLUX.1、Wan T2V 1.3B/14B | threshold 只在匹配的标定多项式下有意义 |
| TaylorSeer | 预测的 gate 前 block 子模块输出 | Z-Image、FLUX.1、Qwen-Image、Wan | 显存随 block、cache site 和 Taylor 阶数增长 |
| PAB | 按 attention 类型缓存的输出 | Z-Image、FLUX.1、Qwen-Image、Wan | 当前使用维护版简化周期规则 |

未支持的模型、步数、profile 或参数组合会提前报错。MagCache 没有官方 Z-Image profile，
其异构递归 backbone 也不接受自定义 ratios。

## 使用方式

通过 `chitu generate` 启用 FlexCache：

```bash
chitu generate \
  --model flux1 \
  --model-path /path/to/FLUX.1-dev \
  --steps 50 \
  --cache-strategy magcache \
  --output outputs/flux1-magcache.png
```

策略私有参数使用独立前缀：

```bash
# 官方 MeanCache 日程。
--cache-strategy meancache --meancache-fresh-steps 25

# 使用模型标定 threshold 的 TeaCache。
--cache-strategy teacache --teacache-threshold 0.4

# 当前维护的 TaylorSeer 速度/质量默认点。
--cache-strategy taylorseer --taylorseer-fresh-threshold 3
```

公共 warmup/cooldown 参数只用于采用公共 fresh-step 外壳的策略。MagCache 自己定义日程，
因此会拒绝这些 override。配置由 `epac/cache.py` 中的 immutable dataclass 表示，并在
安装 hook 前完成校验。

## 目录结构

```text
flexcache/
├── contracts.py       # CacheStrategy 协议、step context、统计
├── session.py         # 请求生命周期和临时 hook 安装
├── spec.py            # 模型家族发现以及 site/probe 协议
├── tree.py            # tensor-tree clone、算术和大小工具
└── strategies/
    ├── base.py        # 所有 hook 的 no-op 默认实现
    ├── factory.py     # CacheConfig -> strategy instance
    ├── magcache.py
    ├── magcache_profiles.py
    ├── meancache.py
    ├── pab.py
    ├── taylorseer.py
    └── teacache.py
```

所有可变 tensor 和 counter 必须保存在 strategy instance 上。`CacheSession` 创建
request-local 状态；串行 CFG 的每个分支使用独立 strategy instance。

## 请求生命周期

1. 校验 `CacheConfig`，由 `strategies/factory.py` 创建 strategy。
2. `FlexCacheModelSpec.discover()` 解析稳定的 model、block 和 leaf site。
3. `CacheSession` 只在当前 pipeline/model instance 上安装临时 wrapper。
4. 使用 timestep signature 推进 denoise step，并区分串行 CFG 分支。
5. lookup hook 可以跳过计算；store hook 更新 request-local cache 和统计。
6. session 退出时恢复所有 wrapper；setup 或 generation 抛错时同样清理。

MeanCache 等 request-level 算法只 patch 当前 pipeline instance。FlexCache 不 patch
pipeline class，也不在全局或 class state 中保存可变 tensor。

## Hook 层级

选择能够表达算法目标值的最窄 hook：

- `model_lookup/store`：每次 transformer 调用做一次判定，用于 step 日程和整 backbone
  状态。
- `block_lookup/store`：复用或重建完整 transformer block。
- `leaf_lookup/store`：复用 `LeafSite` 暴露的 attention、MLP 或 projection 输出。
- request-level `denoise_step`：只用于需要 CFG 合并 prediction 与 scheduler state 的
  MeanCache 类算法。设置 `request_level = True` 并实现 `denoise_step(...)`。

lookup 返回 `(hit, output)`；`True` 表示跳过被 wrapper 包裹的模块。model-level 策略也
可以返回 `False`，同时设置内部状态，让后续 block 或 leaf hook 命中；MagCache、
TeaCache 和 TaylorSeer 都采用这种方式。

## 接入新策略

1. 在 `epac/cache.py` 中增加 immutable 参数 dataclass，并加入 `CacheParams`、
   `_PARAM_TYPES` 和 `CacheStrategyName`。不支持的组合应尽早校验。
2. 新建 `strategies/<name>.py`，继承 `BaseCacheStrategy`，只覆盖算法需要的 hook：

   ```python
   from ..contracts import CacheStepContext
   from ..spec import FlexCacheModelSpec, LeafSite
   from ..tree import TensorTree, tree_clone
   from .base import BaseCacheStrategy


   class ExampleStrategy(BaseCacheStrategy):
       def __init__(self, params: ExampleConfig, **common: int) -> None:
           super().__init__()
           self.params = params
           self.cache: dict[str, TensorTree] = {}

       def begin(
           self, *, total_steps: int, model_spec: FlexCacheModelSpec
       ) -> None:
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
           return (
               (True, tree_clone(value))
               if value is not None
               else (False, None)
           )
   ```

3. 在 `strategies/factory.py` 注册构造逻辑，并从 `strategies/__init__.py` 导出。
4. 只有需要在示例脚本暴露时，才在 `examples/cache_args.py` 增加 CLI 参数。
5. 在 `test/test_chitu_diffusion_flexcache.py` 增加日程、请求隔离、统计、tensor-tree
   输出与不支持配置测试。

```bash
python -m pytest test/test_chitu_diffusion_flexcache.py -q
python -m ruff check chitu_diffusion/flexcache \
  chitu_diffusion/epac/cache.py test/test_chitu_diffusion_flexcache.py
```

## 并行安全要求

- fresh/reuse 判定只能依赖各 rank 复制的输入，例如 step index、timestep 或完整序列
  modulation probe。
- 禁止使用 rank-local token shard 独立决策。
- 同一 lane 的所有 rank 必须一致地进入或跳过 collective。
- 算法定义在 guided prediction 上时，应缓存 all-gather 与 CFG combine 之后的值。
- 每个请求的状态必须隔离；禁止 monkey-patch 全局模型状态或在 strategy class 上保存
  tensor。

若 adaptive metric 无法做到各 rank 一致，应显式同步标量判定，或拒绝该并行模式。相对
reference 的小幅语义差异，也优于 collective 控制流分叉。

## 模型接入边界

`spec.py` 中的 `FlexCacheModelSpec.discover()` 是唯一可以了解模型家族模块命名的位置。
策略需要新的复用 site 或 probe 时，先扩展 model spec，并让策略只使用 `BlockSite`、
`LeafSite` 与 `FlexCacheModelSpec` 操作。禁止在模型 forward 中添加策略条件分支。

## 验收要求

单元测试通过不代表策略接入完成，应分层验证：

1. **协议测试：**日程边界、hit/miss 统计、tensor-tree 输出、request/CFG 隔离、异常后
   清理和不支持配置。
2. **reference 语义：**在相同 Diffusers backend、dtype、scheduler、seed、prompt 和
   尺寸下，对比 fresh/reuse 步骤与输出。
3. **单卡性能：**同时报告端到端、可用时的 DiT-only 耗时，以及 cache 显存。
4. **并行 smoke：**运行 static CP 和每种支持的 CFP 布局，确认各 rank 判定一致且
   collective 完成。
5. **质量 sweep：**每个加速点都与同一 backend 的无缓存 baseline 比较，不能只跨
   attention backend 或 dtype 比图。

文字结果与可复现命令放在 `docs/results/flexcache/<run-id>.md`。生成媒体和临时原始
数据放在已忽略的 `outputs/` 目录。

当前对比记录：

- [`MagCache`](../../docs/results/flexcache/magcache_compare_20260805.md)
- [`MeanCache`](../../docs/results/flexcache/meancache_compare_20260804.md)

## 已知取舍

- cache 会改变数值轨迹；有加速不代表质量可接受。模型、步数或 profile 改变后必须重新
  评估。
- TaylorSeer 为每个子模块和导数阶保存多个 tensor，在视频任务中可能占用数 GB 显存。
- TeaCache 已发布系数与 probe、标定日程绑定。reference 系数按 50 steps 标定，短步数
  质量可能明显退化。
- backend、attention kernel 与 dtype 差异可能主导逐像素对比。cache run 必须与同一
  backend 的无缓存 baseline 比较。
