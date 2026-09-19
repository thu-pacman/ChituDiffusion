# Wan 2.1 T2V

该包保存 Wan 文本编码、video latent、scheduler、transformer、VAE finalize 和 Request
差异。共享 EPE、CP transport 和 FlexCache 算法位于各自公共包。

```bash
chitu generate \
  --model wan \
  --model-path /path/to/Wan2.1-T2V-1.3B \
  --height 480 \
  --width 832 \
  --frames 17 \
  --output outputs/wan.mp4
```

缓存 profile 与模型和步数绑定。使用前查看 `docs/features/flexcache.md`。

Wan 2.1 T2V 还提供独立的 FPP 细粒度流水线生成预览：
`--pipeline-parallel-degree` 指定 stage 数，`--fpp-patches` 指定 token 分块数。
运行命令、滚动 KV 语义、同步基线与当前限制见
[FPP 指南](../../../docs/features/fpp.md)。
