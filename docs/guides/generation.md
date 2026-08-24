# 生成

`chitu generate` 是正式生成入口，支持 `zimage`、`flux1`、`flux2-klein`、
`qwen-image` 和 `wan`。运行
`chitu generate --model zimage --help` 查看对应模型参数。

```bash
chitu generate \
  --model wan \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --steps 50 \
  --output outputs/wan.mp4
```

单卡和静态 CP 使用同一条命令。多卡时通过 `torchrun` 启动每个 rank：

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate \
  --model qwen-image \
  --model-path /path/to/Qwen-Image \
  --output outputs/qwen.png
```

静态 CP 默认使用 Torch/NCCL transport。Fast CP 需要单机 P2P、Hopper 和单独编译的
扩展，启用方式见[并行指南](../architecture/parallel.md)。

根目录 `examples/` 展示 Pipeline/Request API。命令行实现位于
`chitu_diffusion/commands/`，不属于用户示例。
