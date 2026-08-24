# Qwen-Image

该包保存 Qwen-Image 的 Request、Pipeline、executor、transformer tensor 布局和
attention glue。EPE 调度和 CP transport 位于公共运行时。

```bash
chitu generate \
  --model qwen-image \
  --model-path /path/to/Qwen-Image \
  --output outputs/qwen-image.png
```

模型包不创建 process group，也不实现 FlexCache 策略。缓存支持范围见
`docs/guides/flexcache.md`。
