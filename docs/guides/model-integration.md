# 模型接入

`chitu_diffusion/models/<family>/` 只保存模型差异：

- Diffusers pipeline 的分步生命周期；
- checkpoint tensor 和 RoPE/QKV 布局；
- 模型 Request、Pipeline facade 和 executor；
- 模型支持的 CFG、VAE 和 FlexCache site。

模型包不实现 lane 调度、worker、transport 或缓存算法。相应代码分别属于
`epe/`、`parallel/` 和 `flexcache/`。

接入步骤：

1. 建立模型 Request 和 Pipeline facade，保留上游 tokenizer、encoder、scheduler、
   VAE 和 postprocess。
2. 将 DiT 去噪拆成 request prepare、单步 denoise 和 finalize。
3. 通过 `chitu_diffusion.parallel` 的公共接口接入 NCCL CP；需要时再验证 Fast CP。
4. 在 FlexCache model spec 中声明稳定的模型、block 和 leaf site。
5. 增加 CPU contract 测试，并用相同 seed、dtype、scheduler 和尺寸对比原生
   Diffusers GPU baseline。

不得把未运行的模型、并行布局或缓存组合标记为已支持。详细 decoder contract 见
仓库内 `chitu_diffusion/INTEGRATING_IMAGE_DECODER.md`。
