# FLUX.1

该包保存 FLUX.1-dev 的模型差异：

- `api.py`：`Flux1Pipeline` 与 `Flux1Request`。
- `pipeline.py`：Diffusers 分步 denoise 生命周期。
- `transformer.py` 与 `attention.py`：FLUX.1 tensor 布局和 CP glue。
- `executor.py`：EPE 和 embedded runtime 使用的模型 wrapper。

调度、worker、transport 和缓存算法不属于此包。静态 CP 通过
`chitu_diffusion.parallel` 接入，FlexCache 通过 model spec hook 接入。

```bash
chitu generate \
  --model flux1 \
  --model-path /path/to/FLUX.1-dev \
  --output outputs/flux1.png
```

通用模型接入规则见 `docs/guides/model-integration.md`。
