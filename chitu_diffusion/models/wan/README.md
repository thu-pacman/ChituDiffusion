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

缓存 profile 与模型和步数绑定。使用前查看 `docs/guides/flexcache.md`。
