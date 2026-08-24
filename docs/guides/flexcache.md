# FlexCache

FlexCache 是 request-local 的 `generate` 加速层，包含 MagCache、MeanCache、
TeaCache、TaylorSeer 和 PAB。

```bash
chitu generate \
  --model flux1 \
  --model-path /path/to/FLUX.1-dev \
  --steps 50 \
  --cache-strategy magcache \
  --output outputs/flux1-magcache.png
```

边界：

- 可叠加单卡、静态 NCCL CP 或静态 Fast CP 生成。
- 不创建序列并行、process group 或 transport。
- 不接入 EPE `serve`。
- 每个请求和串行 CFG 分支保存独立状态。
- 同一 CP lane 的所有 rank 必须作出相同的 fresh/reuse 决定。

MagCache、MeanCache 和 TeaCache 的 profile 与模型、步数或 probe 绑定。不支持的组合
会在生成前报错。缓存改变数值轨迹，速度提升不能替代质量验收。

实现和策略接入细节见 `chitu_diffusion/flexcache/README.md`。已有结果：

- [MagCache 对比](../results/flexcache/magcache_compare_20260805.md)
- [MeanCache 对比](../results/flexcache/meancache_compare_20260804.md)
