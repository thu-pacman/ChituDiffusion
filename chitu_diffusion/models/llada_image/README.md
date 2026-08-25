# LLaDA-Image

该包将 LLaDA-Image 的官方 Diffusers pipeline 接入 ChituDiffusion，并保留
text-to-image、MLLM VQ condition 和单图 editing 三种离线生成模式。模型权重必须
是本地 Diffusers checkpoint 目录；仓库不包含权重。
checkpoint 的 text encoder 通过 Transformers `trust_remote_code` 加载其中的
Python 模块，因此模型目录属于可执行制品，只能使用经过校验、只读且可信的本地
checkpoint。多卡调用应使用公开的 `LLaDAImagePipeline` facade 或 CLI；内部
Diffusers pipeline 不承诺分布式 CFG/VQ 语义。

## 本地生成

Text-to-image：

    chitu generate --model llada-image --model-path /path/to/LLaDAImage-diffusers --generation-mode text --prompt "a cinematic photograph of a red fox in the snow" --height 1024 --width 1024 --steps 20 --guidance-scale 4.5 --seed 42 --output outputs/llada-image-text.png

VQ-conditioned 生成会先由 LLaDA text frontend 生成离散图像 token，再交给
SigVQ 和 DiT：

    chitu generate --model llada-image --model-path /path/to/LLaDAImage-diffusers --generation-mode vq --prompt "a red fox in the snow" --output outputs/llada-image-vq.png

Editing 同时使用文本条件、原图的 SigVQ semantic features 和 VAE source
latents：

    chitu generate --model llada-image --model-path /path/to/LLaDAImage-diffusers --generation-mode editing --image /path/to/source.png --prompt "turn it into a watercolor painting" --output outputs/llada-image-edit.png

## 静态并行

AGKV context parallel：

    torchrun --standalone --nproc-per-node=2 -m chitu_diffusion.cli generate --model llada-image --model-path /path/to/LLaDAImage-diffusers --generation-mode text --prompt "a red fox in the snow" --attention-mode agkv --no-cfg-parallel --output outputs/llada-image-agkv.png

Ulysses context parallel：

    torchrun --standalone --nproc-per-node=2 -m chitu_diffusion.cli generate --model llada-image --model-path /path/to/LLaDAImage-diffusers --generation-mode text --prompt "a red fox in the snow" --attention-mode ulysses --ulysses-degree 2 --no-cfg-parallel --output outputs/llada-image-ulysses.png

当 world size 为偶数且 CFG 开启时，--cfg-parallel 会先将 GPU 分成
conditional/unconditional 两个分支，再在每个分支内部应用 CP。VQ 模式只在
world rank 0 生成一次离散 token，并按请求 seed 隔离 RNG，然后广播给所有 rank。

Fast AGKV/Fast Ulysses 扩展可用时沿用通用 transport 选择；扩展不可用时使用
PyTorch/NCCL fallback。LLaDA text frontend 同样优先使用原生 flash_attn 和
veomni，缺少扩展时使用经过测试的 PyTorch fallback。

## Serving

当前 HTTP serving 只接受 text-to-image 请求：

    chitu serve --stage-config /path/to/stage-llada-image.yaml

HTTP schema 不接受本地图片路径、URL、base64 图片、editing 或 vq 模式。
Editing/VQ 仍使用离线 API；在条件状态迁移协议完成前，不宣称支持这两种模式的
elastic serving。

未显式传入 serving 配置时，HTTP 请求的默认去噪步数为 50；离线 CLI/API 的
默认值仍为 20。

当前 LLaDA-Image 的 request-side `state_bytes` 沿用 elastic scheduler 的未知状态
默认值 0，因此迁移成本暂不进入请求规划。worker 准备好 latent 后的精确 FP32
状态大小和 warmup 传输测量仍会保留；待 scheduler 支持接收运行中状态 profile 后，
再启用迁移成本估算。

内置 HTTP 服务没有认证且默认监听 `0.0.0.0`，只能部署在可信内网，或置于带认证
和请求限额的网关后方。
## 边界

- 当前每个请求只支持一张输出图，batch size 必须为 1。
- AGKV、Ulysses 和 CFP 支持 text、VQ 与 editing 离线生成。
- FlexCache 尚未建立 LLaDA-Image 的模型级正确性契约，非 none 策略会提前报错。
- parallel_vae 默认关闭。AutoencoderKLFlux2 包含 GroupNorm 和全局
  mid-block attention，通用 tiled VAE decode 是近似计算；只有显式
  --parallel-vae 才会启用，并在运行时给出警告。
- width=1 使用官方 transformer 路径；Chitu wrapper 只在并行宽度大于 1 时改写
  DiT 的 attention 和 image-token 布局。
