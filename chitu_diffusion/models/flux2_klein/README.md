# FLUX.2-klein

该包提供固定 full-world 静态 CP pipeline，不接入 EPE 动态 lane。

VAE 默认使用公共逐层行分片解码，GroupNorm 合并全局统计量，attention 使用
AGKV。`--no-parallel-vae` 使用 leader 整图解码。原 pipeline 的图像条件编码、
latent BatchNorm 与 unpatchify 顺序保留，`output_type="latent"` 跳过解码。
参见 [VAEP 架构](../../../docs/features/vae-parallel.md) 和
[验证范围](../../../docs/validation/vae-parallel.md)。

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate \
  --model flux2-klein \
  --model-path /path/to/FLUX.2-klein \
  --output outputs/flux2-klein.png
```

模型差异位于该目录；process group、NCCL CP 和 Fast CP transport 位于
`chitu_diffusion.parallel`。当前文档不把 FlexCache 或 EPE 服务列为该模型能力。
