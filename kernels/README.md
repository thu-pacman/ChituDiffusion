# Kernel Integration Workspace

该目录用于审查、适配和验证外部开源社区 kernels，不暗示这些实现由 Chitu 原创：

- `upstreams.toml`：上游仓库、关注路径和主题。
- `upstreams.lock.json`：最近完成人工审查的 commit，不代表依赖锁定。
- `benchmark_attention.py`：使用 diffusion 形状做 correctness/latency gate。
- `worklog.md`：调查、实验、采用和拒绝记录。
- `kernel_adapters/`：运行时兼容层，向 ChituDiffusion 暴露稳定接口。
- `refs/kernels/`：被 git 忽略的浅克隆，仅用于代码审查和移植。

## 快速使用

```bash
# 查看哪些上游在上次审查后发生变化（有变化时退出码为 1）
python script/kernel_watch.py scan

# 拉取关注的上游到 refs/kernels/
python script/kernel_watch.py sync sglang hpc-ops flash-attention

# GPU 上先做小形状 correctness/latency
python kernels/benchmark_attention.py \
  --preset smoke \
  --backends torch_sdpa flash_attn3 flash_attn

# 完成人工审查并更新 worklog 后，记录已审查 commit
python script/kernel_watch.py lock sglang hpc-ops flash-attention
```

FlashAttention 3 可通过官方 `flash_attn_interface`，或可选的 Hugging Face
`kernels` loader 使用；后者用 `uv sync --extra flash3` 安装。运行时选择
`infer.attn_type=flash_attn3`（别名 `fa3`），也可设置
`CHITU_FA3_PROVIDER=official|hf`。FA3 当前只在 Hopper SM90 上可用。

## 按模型 Kernel Recipes

- 按模型的推荐默认值、覆盖项和 H20 实测收益见
  [`recipes.md`](recipes.md)。默认配置采用 `infer.attn_type=auto`，并只自动开启
  已证明有端到端收益的模型专用 fusion。
- Z-Image norm：`CHITU_Z_IMAGE_NORM_FUSION=auto|off|diffusers_fp32|native_bf16`，
  默认 `auto`，等价于 `diffusers_fp32`。后者保持 FP32 reduction policy；
  `native_bf16` 复现
  SGLang native reduction。两者均为 opt-in，均不保证相对 unfused 输出 bit-exact。
- Qwen-Image padded-varlen：配置 `infer.attn_type=flash_attn3` 时，单卡 bool padding
  mask 可走 Triton pack/scatter + FA3 varlen；无 mask、全有效 mask 和 CP 路径不改变。
  H20 microbenchmark 显示 FA3 有收益，FA2 在已测形状反而慢于 masked SDPA。
- Flux2 QK RMSNorm+RoPE：`CHITU_FLUX2_FUSED_QKNORM_ROPE=auto` 为默认；
  仅在 SM90 eager FP16/BF16 且 layout 满足时尝试，其他情况安全回退。H20
  microbenchmark 约 `15x`，端到端约 `1.5%`。
- Flux2 SiLU×mul：`kernel_adapters/silu_and_mul.py` 保留 vLLM
  `616c9bd` 的 Apache-2.0 provenance，并匹配 Flux2 的 contiguous
  `[B,S,2*D] -> [B,S,D]` SwiGLU layout。按 PyTorch 的中间 dtype 舍入修正后，
  H20 BF16 真实 shape bitwise equal 且快 `4.72x`；固定 seed 端到端 PNG 也
  bitwise equal。`CHITU_FLUX2_SILU_MUL_FUSION=auto` 为默认，可设 `off` 禁用。
  FP16 未通过 bitwise gate，会自动回退 PyTorch。
- Qwen-Image select01 modulation/residual/norm：SGLang Triton port 位于
  `kernel_adapters/qwen_image_modulation.py`，并保留与 diffusers expression 相同的
  CPU PyTorch fallback。H20 的真实 `[1,4096,3072]` BF16 shape 未通过
  `rtol=atol=2e-2` 严格门禁（Triton reduction/store rounding），因此当前是
  **benchmark-only**；`CHITU_QWEN_IMAGE_MODULATION_FUSION` 只能为 `off`（默认）
  或 `benchmark_only`。设置旧候选值 `select01` 会明确报错，不会改变默认精度
  policy 或静默启用。
- CP attention-state merge：候选已降级为 benchmark-only；运行时只允许
  `CHITU_CP_ATTN_STATE_MERGE=off`，设置 `triton` 会明确报错。隔离 harness 使用
  `benchmark_only` 测试数学/LSE contract 等价的 float32 running-state merge。
  它适用于 eager Ring/USP 的 `out [B,S,H,D]` FP32、
  strided `lse [B,S,H,1]` FP32、`block_lse [B,H,S]` FP32，以及 BF16/FP16/FP32
  block output。任何其他 layout、dtype、Triton 不可用或 launch 失败均安全回退。
  真实 2×H20 Wan Ring gate 中 300/300 次 merge 确认执行 Triton，但 5-step
  denoise/DiT 均慢 `18.3%`（首步 JIT）；去掉首步后稳态无可测收益。与可复现 off
  输出相比，17 帧视频 mean PSNR `30.035dB`、mean SSIM `0.93211`、mean abs
  `2.962/255`，因此数学等价不能称为扩散轨迹 bit-exact。AGKV 和纯 Ulysses
  没有 per-block online merge，不受影响。
- Wan VAE causal Conv3D cat+pad：`CHITU_WAN_VAE_CAT_PAD=triton` 可为
  `chitu_diffusion/modules/vaes/wan_vae.py` 的 cached `CausalConv3d` 启用精确
  Triton 数据搬运；默认 `off`。它仅接受连续 CUDA `[B,C,T,H,W]`、同 device/dtype
  的 cache，FP16/BF16/FP32、对称 spatial padding、右侧 temporal padding 为 0，
  且 `cache_t <= temporal_left_padding`。不匹配、Triton 不存在或 launch 异常会执行
  原始 `torch.cat` + `F.pad`，不发生任何 dtype/precision 转换。H20 gate 已证明
  输出 bitwise equal。真实 1.3B、256x256、17 帧、1-step E2E 的五个 MP4 完全
  bitwise 相同，但 warm Triton VAE decode 均值 `834.23ms`，明显慢于 off 的
  `427.76ms`；因此默认保持 `off`，`triton` 仅用于实验/诊断，不建议 production
  opt-in。单 shape mover 的 `1.84-2.03x` 不能解释成 VAE/E2E 收益。

详细形状、数值误差、质量与延迟结果见 `kernels/worklog.md`。

### CP merge gate

```bash
# 1 GPU: PyTorch state contract + BF16/FP16 Triton correctness
SRUN_PARTITION=debug SRUN_EXTRA_ARGS=--exclusive \
CHITU_PYTHON_BIN=$PWD/.venv/bin/python \
script/srun_direct.sh 1 1 -m pytest -q test/test_kernel_integration.py

# 2 H20 GPUs: all-rank correctness and local CP-shard merge latency
SRUN_PARTITION=debug SRUN_EXTRA_ARGS=--exclusive \
CHITU_PYTHON_BIN=$PWD/.venv/bin/python \
script/srun_direct.sh 1 2 kernels/benchmark_cp_attention_state_merge.py \
  --require-world-size 2 --seq-len 4096 --heads 24 --head-dim 128 --dtype bf16

# Real Wan eager-Ring A/B: run MODE=off, then MODE=triton.
MODE=off
CHITU_PROJECT_ROOT=$PWD CHITU_PYTHON_BIN=$PWD/.venv/bin/python \
CHITU_RUN_DIR=$PWD/outputs/cp-merge-e2e/wan-ring-$MODE \
CHITU_RUN_TASK_ID=wan-ring-$MODE CHITU_WAN_STEPS=5 CHITU_WAN_FRAMES=17 \
CHITU_CP_ATTN_STATE_MERGE=$MODE CHITU_CP_ATTN_STATE_MERGE_TRACE=1 \
SRUN_PARTITION=debug SRUN_EXTRA_ARGS=--exclusive \
script/srun_direct.sh 1 2 test/test_wan.py \
  models=Wan2.1-T2V-1.3B \
  models.ckpt_dir=/home/chenyy/WORK/models/Wan2.1-T2V-1.3B \
  infer.attn_type=torch_sdpa infer.diffusion.cp_size=2 \
  infer.diffusion.up=1 infer.diffusion.cp_backend=ucp \
  infer.diffusion.ring_cudagraph=false infer.diffusion.compile_mode=off \
  output.timer=true output.memory=false output.log_ranks=all
```

The second command is a merge-only gate and uses the production strided-LSE
layout. For a real Ring run, set `cp_size=2`, `up=1`, `cp_backend=ucp`,
`ring_cudagraph=false`, and optionally
`CHITU_CP_ATTN_STATE_MERGE_TRACE=1`; process-exit counters prove whether Triton
ran or safely fell back. The current Wan E2E result failed both default
performance and trajectory-quality gates, so the feature remains opt-in/off by
default.

### Wan VAE cat+pad gate

```bash
# CPU fallback, cache/no-cache and padding boundaries
.venv/bin/python -m pytest -q test/test_wan_vae_causal_conv3d.py

# Explicit H20 correctness opt-in
SRUN_PARTITION=debug SRUN_EXTRA_ARGS=--exclusive \
CHITU_PYTHON_BIN=$PWD/.venv/bin/python \
CHITU_RUN_WAN_VAE_CAT_PAD_GPU=1 CHITU_WAN_VAE_CAT_PAD=triton \
script/srun_direct.sh 1 1 -m pytest -q test/test_wan_vae_causal_conv3d.py

# Real Wan2.1-1.3B fixed-seed E2E; run once with off and once with triton.
# CHITU_WAN_VAE_CAT_PAD_STATS=1 logs cached/supported/fused/fallback call counts.
CHITU_PROJECT_ROOT=$PWD CHITU_PYTHON_BIN=$PWD/.venv/bin/python \
SRUN_PARTITION=debug SRUN_EXTRA_ARGS=--exclusive \
CHITU_WAN_VAE_CAT_PAD_STATS=1 CHITU_WAN_VAE_CAT_PAD=off \
CHITU_WAN_STEPS=1 CHITU_WAN_FRAMES=17 \
CHITU_WAN_WIDTH=256 CHITU_WAN_HEIGHT=256 \
script/srun_direct.sh 1 1 test/test_wan.py \
  models=Wan2.1-T2V-1.3B \
  models.ckpt_dir=/home/chenyy/WORK/models/Wan2.1-T2V-1.3B \
  infer.diffusion.up=1 infer.diffusion.cp_size=1 infer.diffusion.cfg_size=1 \
  infer.use_cuda_graph=false infer.attn_type=torch_sdpa output.timer=true

# 480p Wan decoder/encoder feature-map shapes
SRUN_PARTITION=debug SRUN_EXTRA_ARGS=--exclusive \
CHITU_PYTHON_BIN=$PWD/.venv/bin/python \
script/srun_direct.sh 1 1 kernels/benchmark_wan_vae_causal_conv3d.py \
  --case wan-decode-480p --dtype bfloat16 --warmup 50 --iters 300
```

## 上游致谢

感谢 [SGLang](https://github.com/sgl-project/sglang) 社区公开 diffusion JIT kernels，
以及其代码中注明来源的 FastVideo 社区；感谢
[vLLM](https://github.com/vllm-project/vllm)、
[vLLM-Omni](https://github.com/vllm-project/vllm-omni)、
[Tencent HPC-Ops](https://github.com/Tencent/HPC-Ops)、
[FlashAttention](https://github.com/Dao-AILab/flash-attention) 和
[Hugging Face kernels](https://github.com/huggingface/kernels) 社区提供可审查的实现、
接口和性能基线。移植文件内保留 SPDX、上游 commit/path provenance 与许可证说明。

## 采用原则

1. 上游 LLM kernel 不等于 diffusion kernel。Paged KV cache、decode attention、
   sampling 和 MoE 默认不适用 DiT。
2. 先证明热点和形状匹配，再复用；先 kernel microbenchmark，再做真实模型端到端 A/B。
3. 量化/稀疏算子除数值误差外必须经过 ChituBench 质量评估。
4. `auto` 只能选择当前设备可运行的实现；显式选择不允许静默 fallback。
5. 独立发布条件成熟时，可直接把 `kernel_adapters/` 和本目录的清单/harness 拆为
   单独仓库；当前放在同一仓库保证接口变更和端到端基准可原子审查。
