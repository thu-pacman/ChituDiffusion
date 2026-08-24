# Z-Image

该包保存 `ZImagePipeline`、`ZImageRequest`、executor、transformer tensor 布局和
attention glue。EPE 只通过公共 executor contract 使用该包。

```bash
chitu generate \
  --model zimage \
  --model-path /path/to/Z-Image \
  --output outputs/zimage.png
```

服务使用统一入口：

```bash
chitu serve --stage-config examples/stage-zimage.yaml
```

不存在模型专用服务入口。EPE、parallel 和 FlexCache 的边界见 `docs/architecture/`。
