# FLUX.2-klein

该包提供固定 full-world 静态 CP pipeline，不接入 EPE 动态 lane。

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate \
  --model flux2-klein \
  --model-path /path/to/FLUX.2-klein \
  --output outputs/flux2-klein.png
```

模型差异位于该目录；process group、NCCL CP 和 Fast CP transport 位于
`chitu_diffusion.parallel`。当前文档不把 FlexCache 或 EPE 服务列为该模型能力。
