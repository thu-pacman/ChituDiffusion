# FlexCache

FlexCache 为不同特征复用方法提供统一配置、生命周期和统计接口。它只作用于一次
`generate()`，请求结束或发生异常时会恢复所有临时 hook。

## 基本原理

扩散模型相邻去噪步骤的中间特征通常相近。缓存策略在 fresh step 执行原始计算并保存
结果，在 reuse step 跳过部分计算，用残差、历史输出或预测值代替。

`CacheSession` 在请求开始时定位模型、block 和 attention site，并只包装当前 pipeline
实例。串行 CFG 的 conditional 与 unconditional 分支分别保存状态，因此不会读取对方
的 tensor。

FlexCache 可用于单卡、静态 NCCL CP 和静态 Fast CP。缓存层不创建 process group，
也不选择并行布局。同一 CP lane 的所有 rank 根据 step、timestep 或完整序列 probe
做出相同 fresh/reuse 决定，避免 collective 控制流分叉。

## 优化方案

| 策略 | 复用对象 | 当前支持 |
| --- | --- | --- |
| MagCache | image-token backbone residual | FLUX.1、Qwen-Image、Wan 1.3B 的官方 profile |
| MeanCache | CFG 合并后的 prediction 与 scheduler state | Z-Image、Qwen-Image，固定 50 steps |
| TeaCache | block-group residual | FLUX.1、Wan 1.3B/14B 的默认标定 |
| TaylorSeer | attention、MLP 和 projection 的 Taylor 预测值 | Z-Image、FLUX.1、Qwen-Image、Wan |
| PAB | 按周期复用 attention 输出 | Z-Image、FLUX.1、Qwen-Image、Wan |

### MagCache

MagCache 根据模型预先标定的 magnitude-ratio profile 生成 reuse 日程。fresh step 保存
整个 image-token backbone 的 residual，reuse step 在第一个 block 注入 residual，并
跳过后续 block。

### MeanCache

MeanCache 在 scheduler 边界保存 CFG 合并后的 prediction、latents 和 sigma 历史。
reuse step 使用 sigma-JVP 外推 prediction，再执行正常 scheduler step。它不修改
transformer block。

### TeaCache、TaylorSeer 和 PAB

TeaCache 根据 modulation probe 的累计变化决定是否复用 block-group residual。
TaylorSeer 用有限差分和 Taylor 展开预测 block 内部输出，计算量更低，但缓存 tensor
较多。PAB 按 attention 类型和周期复用输出，MLP 仍正常执行。

每种方法都改变数值轨迹。profile、threshold、fresh step 数或周期改变后，需要重新测量
质量和延迟。

## 优化结果

以下结果均来自单张 NVIDIA H20、BF16，并相对相同 ChituDiffusion backend、prompt 和
seed 的无缓存结果。不同模型和策略的工作负载不同，不能横向比较。

### MagCache

| 模型与配置 | 无缓存 | MagCache | 加速 | PSNR / LPIPS |
| --- | ---: | ---: | ---: | ---: |
| FLUX.1，1024²，28 steps | 24.83s | 8.31s | 2.99x | 17.06 / 0.2963 |
| Qwen-Image，1664×928，50 steps | 100.45s | 73.27s | 1.37x | 35.90 / 0.0172 |
| Wan 1.3B，480×832×81 帧，50 steps | 263.69s | 116.49s | 2.26x | 16.00 / 0.2043 |

### MeanCache

测试使用 512²、50 steps、seed 42。`fresh` 表示 50 个步骤中执行完整模型的步骤数。

| 模型 | fresh | 加速 | PSNR |
| --- | ---: | ---: | ---: |
| Z-Image | 25 / 17 / 13 | 2.26x / 2.83x / 4.23x | 29.25 / 24.90 / 19.76 |
| Qwen-Image | 25 / 17 / 10 | 2.07x / 2.65x / 4.73x | 31.15 / 30.11 / 27.44 |

仓库目前没有条件一致的 TeaCache、TaylorSeer 和 PAB 评测，因此不提供这三种方法的
性能结论。

## 使用示例

CLI 使用 MagCache：

```bash
chitu generate \
  --model wan \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --steps 50 \
  --cache-strategy magcache \
  --output outputs/wan-magcache.mp4
```

Python API 使用 MeanCache：

```python
import torch

from chitu_diffusion import (
    CacheConfig,
    MeanCacheConfig,
    ZImagePipeline,
    ZImageRequest,
)

pipeline = ZImagePipeline.from_pretrained(
    "/path/to/Z-Image",
    torch_dtype=torch.bfloat16,
    cache=CacheConfig(
        strategy="meancache",
        params=MeanCacheConfig(fresh_steps=25),
    ),
)

try:
    result = pipeline.generate(
        ZImageRequest(
            prompt="a red cube on a white table",
            num_steps=50,
            seed=7,
        )
    )
    if pipeline.parallel_context.rank == 0:
        result.images[0].save("zimage-meancache.png")
        print(pipeline.last_cache_stats)
finally:
    pipeline.close()
```

EPE `serve` 不接受 FlexCache 配置。当前缓存状态是 request-local，不能跨排队请求或
动态 lane 共享。
