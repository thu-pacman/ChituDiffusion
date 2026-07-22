# Z-Image Integration

This directory owns all Z-Image-specific code:

- the Diffusers-compatible pipeline and request-local denoise state;
- the transformer wrapper and Diffusers attention projection/RoPE glue;
- deterministic startup process-group creation;
- random-tensor warmup for supported image resolutions;
- the `DiffusersModelAdapter` implementation;
- `ZImageImageDecoderExecutor`, which owns request codec, model state, VAE, and
  Diffusers postprocess behind the embedded runtime contract;
- the public `EPACPipeline.from_pretrained()` facade.

It may depend on `chitu_diffusers.epac`, but EPAC core must never import this
package. HTTP schemas and torchrun lifecycle code belong in
`chitu_diffusers.serve`.

AGKV/USP 的通信与 attention 计算位于 `chitu_diffusers.parallel`。Z-Image
processor 只负责把 `[local_image, replicated_text]` Q/K/V 交给公共接口。设置
`attention_mode="usp", ulysses_degree=2` 时，四卡 lane 使用 u2r2；较窄的 elastic lane
自动退化为 u2r1 或 u1r1。

Encoder, scheduler, VAE, and image processor behavior remains inherited from
upstream Diffusers. Only the DiT denoise loop is split into request-local steps.

`ZImageExecutorFactory` is the standalone/embedded construction path. A future
LLaDA2 executor can replace text encoding with VQ/SigVQ preparation while
reusing the same runtime, lane broker, transfer protocol, and SLO planner.
