# Diffusers API

ChituDiffusion 从 Diffusers checkpoint 加载模型，并保留原 pipeline 中与分布式 DiT
无关的组件。用户不需要转换权重格式，也不需要维护另一套 tokenizer、scheduler 或
VAE 实现。

## 基本原理

每个模型适配器包含四层：

1. **Request** 定义 prompt、尺寸、步数、seed 和模型专属参数。
2. **Pipeline** 加载 Diffusers 组件，并提供 `generate()` 和 `serve()` 入口。
3. **Executor** 把 pipeline 生命周期拆成请求准备、单步去噪、decode 和 postprocess。
4. **Attention adapter** 处理模型专属 QKV、RoPE 和 tensor 布局，再调用公共 CP 实现。

静态生成与 EPE 服务共享 executor。两条路径使用相同的 scheduler、latents 和输出处理，
区别只在于 lane 固定还是由 EPE 调度。

## 适配方案

ChituDiffusion 只接管 DiT 去噪部分：

```text
Diffusers checkpoint
├── tokenizer / text encoder   保留
├── scheduler                  保留
├── VAE / image processor      保留
└── transformer / attention    接入 CP、EPE 和 FlexCache
```

模型差异位于 `chitu_diffusion/models/<family>/`。调度、通信和缓存代码不包含模型名称
分支。新增模型时，需要声明 Request、executor contract、attention 布局和支持能力，
无需复制 EPE planner 或 CP transport。

## 当前支持

| 模型 API | 静态生成 | EPE | FlexCache |
| --- | --- | --- | --- |
| `ZImagePipeline` / `ZImageRequest` | 支持 | 支持 | 支持 |
| `Flux1Pipeline` / `Flux1Request` | 支持 | 支持 | 支持 |
| `QwenImagePipeline` / `QwenImageRequest` | 支持 | 支持 | 支持 |
| `WanPipeline` / `WanRequest` | 支持 | 支持 | 支持 |
| `Flux2KleinCpPipeline` | 支持 | 不支持 | 不支持 |

仓库的模型测试检查 typed Request、pipeline contract 和单 rank attention 数值一致性。
GPU 输出质量仍应使用相同 checkpoint、dtype、scheduler、prompt 和 seed 与原生
Diffusers pipeline 对比。

## 使用示例

```python
import torch

from chitu_diffusion import ZImagePipeline, ZImageRequest

pipeline = ZImagePipeline.from_pretrained(
    "/path/to/Z-Image",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

try:
    result = pipeline.generate(
        ZImageRequest(
            prompt="a red cube on a white table",
            width=1024,
            height=1024,
            num_steps=50,
            seed=7,
        )
    )
    if pipeline.parallel_context.rank == 0:
        result.images[0].save("zimage.png")
finally:
    pipeline.close()
```

多卡时每个 rank 执行同一段代码，只有 lane leader 返回最终 Diffusers output。完整示例
位于仓库根目录 `examples/`。
