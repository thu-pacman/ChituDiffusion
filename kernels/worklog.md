# Kernel Worklog

按时间倒序记录。即使候选被拒绝，也要留下原因，避免重复调查。

## 2026-09-18：Fast CP 原生 baseline 补全

- 完成 H20、BF16、H40/D128、S=4096/8192/16384/32768、CP1/2/4/8 的
  96-cell QKV-to-O sweep。每个序列长度的 CP1/2/4/8 固定在同一节点，全部通过
  dense reference；正式表位于 `outputs/fast-cp/e2e-native-sweep/results.md`。
- CP8 Fast Ulysses 在四个序列上的 scaling 为
  3.75x/5.56x/6.78x/7.51x；SGLang Ulysses 为
  3.54x/5.22x/6.54x/7.36x，xDiT Ulysses 为
  2.82x/4.89x/6.35x/7.22x。各框架均使用自己的 CP1 和原生自动 attention
  backend，跨框架绝对延迟不用于 speedup。
- 完成 Wan 1.3B、Wan 14B、Qwen-Image、Z-Image 的 xDiT 与 SGLang 原生模型
  attention sweep。xDiT 为 15 complete + 1 unsupported，SGLang 为
  25 complete + 3 unsupported；所有 complete cell 均通过 correctness。
  unsupported 来自原生 Ulysses head divisibility 约束。
- 产物：`outputs/fast-cp/model-native-summary.md` 为 96-cell 模型表；
  `outputs/fast-cp/figures/workload-radar/native-latency-workload-radar-all.png`
  为 4 个序列长度加 4 个模型、七种方法的 CP2/4/8 latency 图。半径为同一
  CP/workload 下的最低延迟除以该方法延迟；CSV 同时保存原始毫秒数。原生
  unsupported 配置在图中留空，并在 CSV 中记录状态。

## 2026-09-17：Fast CP benchmark 重置与上游框架基线

- 基线回到 Fast CP 初始实现所在提交 `d4d3a95469017973ca76f242748588adb819ca78`。
  此后未提交的 planner、overlap、breakdown 和 baseline 实验已保存为
  `stash@{0}`（`codex: archive post-fast-cp experiments before baseline rebuild
  2026-09-17`），旧结果移到 `outputs/fast-cp-archive-20260917/`，没有删除。
- 新基准第一阶段统一为：预投影、按 sequence 切分的本地 Q/K/V，经 CP 通信和
  cuDNN dense non-causal BF16 attention，输出本地 O。关闭 CUDA Graph；每个样本
  使用 CUDA event，并取所有 rank 最大值。QKV projection、norm、RoPE、fused
  epilogue 和 graph 留到第二阶段，不能与本表混合。
- 固定 xDiT main `5ede5851a2f950403cdd95b970cd9fc48eb50d87` 与 SGLang main
  `44bd35908230c4f363c83dbe34fe25749d985950`。源码放在
  `refs/fast-cp-frameworks/`，独立环境放在 `.venv-fast-cp-frameworks/`；两者均为
  Python 3.12、Torch 2.13.0+cu130、cuDNN 9.20。SGLang 构建关闭本基准不使用的
  Rust 服务扩展，其 diffusion、JIT、IPC/NCCL 和 attention 依赖保持上游声明。
- 新增统一 JSON schema、固定版本环境脚本、Slurm launcher 和 Markdown 汇总器。
  汇总器拒绝混合 boundary、shape 或 GPU；speedup 只使用相同框架、软件版本、
  序列长度的 CP1。Chitu 当前环境为 Torch 2.9.1+cu130、cuDNN 9.13，跨框架绝对
  延迟必须同时披露该差异；原生实现表保留各 repo 的依赖栈，统一 backend
  仅作为单独的受控对照。
- H20 smoke correctness：B1、S512、H12、D128，CP1/CP2；xDiT Ulysses、SGLang
  Ulysses、SGLang K/V gather、Chitu NCCL AGKV、Fast AGKV 和 Fast Ulysses 全部
  相对全量 K/V cuDNN 参考 `max_abs=0`。实际通信分别记录为 xDiT NCCL
  all-to-all、SGLang CUDA IPC input + NCCL output、SGLang NCCL gather，以及
  Chitu NVSHMEM/NCCL。每项只有 1 warmup、2 samples，仅验证环境、API、路径和
  结果格式，不作为性能数据。
- 决策：正式 H20 H40/D128、S=4096/8192/16384、CP1/2/4/8 测量忠实保留
  三个 repo 的原生通信、attention backend 和软件环境；所有结果记录完整版本，
  不把依赖差异误归因于 CP 算法。

## 2026-07-21：按模型 Kernel Recipe 与默认策略

- 新增 `kernels/recipes.md`，集中记录 Flux1、Flux2、Qwen-Image、Z-Image 和
  Wan 2.1/2.2 的默认 kernel、覆盖项、适用条件与 H20 实测收益。
- 策略从“最终产物必须 bitwise equal 才能默认开启”调整为：模型数学语义等价、
  局部误差符合 dtype/reduction 合约、真实端到端有稳定收益，且没有质量统计退化证据。
  固定 seed 的 PSNR/SSIM 变化只证明扩散轨迹不同，不能单独解释成质量下降。
- 默认 `infer.attn_type=auto`；H20 当前仍选择 dense SDPA。Z-Image 默认启用
  `diffusers_fp32` norm fusion（denoise `+2.67%`），Flux2 默认尝试 SM90
  QK RMSNorm+RoPE（端到端约 `+1.5%`）和 BF16 SiLU×mul（稳态约 `+0.4%`）。
  所有 fusion 在硬件、dtype、layout、编译或 launch 不满足时安全回退。
- Qwen padded-varlen 只在用户选择 FA3 且真实 padding mask 存在时运行；普通单请求
  保持 SDPA。Wan VAE cat+pad 与 CP merge 继续关闭，因为真实路径分别回归和无收益，
  与是否要求 bitwise exact 无关。
- 默认值变更后使用 Slurm `debug --exclusive` 在 H20 运行
  `test/test_qknorm_rope.py test/test_kernel_integration.py`：`26 passed`，覆盖
  Flux2 auto parser、SM90 FP16/BF16 fused correctness、Z-Image auto mode 与 fallback。

## 2026-07-21：vLLM SiLU×mul（Flux2 SwiGLU）精确性门禁

- 上游与许可证：vLLM `616c9bd0f4efff9761fd762e4247e18b6bd7eb23`，Apache-2.0；
  对照 `vllm/model_executor/layers/activation.py:SiluAndMul`、
  `csrc/libtorch_stable/activation_kernels.cu:silu_and_mul` 与
  `chitu_diffusion/models/model_flux2.py:Flux2SwiGLU`。适配实现放在
  `kernel_adapters/silu_and_mul.py`，保留 SPDX、commit/path provenance 和
  vLLM 社区致谢；没有导入 vLLM、`torch.ops._C` 或任何旧版项目专属 kernel 包。
- 合约审查：Flux2 投影输出是连续 `[B,S,2D]`（klein-4B 为
  `D=9216`），前半 gate、后半 up，恰好匹配上游
  `silu(x[..., :D]) * x[..., D:]`。adapter 仅接受 CUDA 连续 BF16；
  CPU、FP16/FP32、奇数末维、非连续 layout 和 Triton/launch 失败均返回 `None`，
  由模型执行原始 PyTorch expression。
- H20 correctness：首次实现把 `gate*sigmoid(gate)*up` 全程保留到 FP32 后一次
  store，因而 BF16/FP16 均未通过 bitwise gate。按真实 PyTorch dtype policy 在
  SiLU 输出后先写回输入 dtype、再执行乘法后重测：BF16 在真实 shape 上
  bitwise equal（max/mean abs `0/0`）；FP16 仍有 max abs `4.8828125e-4`，
  因而 FP16 明确回退，不进入 fusion。
- H20 real Flux2 microbenchmark（`[1,4608,18432]`，20 warmup/100 iters）：
  修正后的 BF16 PyTorch `0.37581ms`、Triton `0.07959ms`（`4.722x`），
  bitwise equal。FP16 kernel 虽约 `4.81x`，但不满足 bitwise gate，运行时拒绝。
  命令：
  `SRUN_PARTITION=debug SRUN_EXTRA_ARGS=--exclusive CHITU_PYTHON_BIN=$PWD/.venv/bin/python script/srun_direct.sh 1 1 kernels/benchmark_flux2_silu_and_mul.py --preset flux2-single-1024 --dtype bf16 --warmup 20 --iters 100`。
- Fixed-seed end-to-end：H20 exclusive、Flux2-klein、1024²、4 steps、seed 42、
  `torch_sdpa_math`。修正后的 BF16 auto 与 off 最终 PNG bitwise equal
  （max/mean abs `0/0`）。首次 Triton compile 污染首 step；排除首 step 后
  off 约 `701.9ms/step`、auto 约 `698.9ms/step`，约 `0.4%` 稳态收益，符合该
  elementwise kernel 在完整 DiT 中占比较小的预期。
- 决策：**采用 BF16 exact auto path。** `CHITU_FLUX2_SILU_MUL_FUSION=auto`
  为默认；`triton` 可显式请求，`off` 禁用，`benchmark_only` 只保留测试语义。
  FP16 和所有不支持条件安全回退原 PyTorch expression。

## 2026-07-21：Wan VAE causal Conv3D cache cat+pad（SGLang 精确搬运）

- 上游与许可证：SGLang `d6ef68881e263812d4901f632786015005c4d050`，Apache-2.0；
  审查 `python/sglang/jit_kernel/diffusion/causal_conv3d_cat_pad.py`、
  `python/sglang/jit_kernel/diffusion/triton/causal_conv3d_pad.py` 和 Wan 使用点
  `python/sglang/multimodal_gen/runtime/models/vaes/wanvae.py`（通过
  `runtime/layers/parallel_conv.py::causal_conv3d_cat_pad`）。适配实现为
  `kernel_adapters/wan_vae_causal_conv3d.py`，保留 Apache SPDX、commit/path 和
  SGLang 社区致谢；不依赖 SGLang、sgl-kernel、TVM FFI 或 CUDA JIT extension。
- 语义逐项核对：Chitu `CausalConv3d.forward` 与 SGLang Wan 均从原 Conv3D padding
  生成 `[w_left,w_right,h_top,h_bottom,2*t_pad,0]`；仅 `cache_x is not None` 且
  temporal-left padding 非零时处理 cache；先将 cache 迁到 x.device，再在 dim=2
  放在 x 前；随后 temporal-left padding 减去 `cache_t`，最后做 `F.pad`。
  因而输出 shape 为 `[B,C,T+pad_t,H+pad_h_top+pad_h_bottom,
  W+pad_w_left+pad_w_right]`，其中每个元素严格为 zero、cache 或 x 的原始 bit。
  Triton 映射等价于 SGLang：`source_t=out_t-(pad_t-cache_t)`，`source_t<cache_t`
  读 cache，其余有效位置读 x；无算术/转换，所以不改变 precision。
- 接入与安全条件：只在显式 `CHITU_WAN_VAE_CAT_PAD=triton` 时尝试。要求连续 CUDA
  5D `[B,C,T,H,W]`、同 device/dtype、FP16/BF16/FP32、相同 B/C/H/W、非空 cache、
  对称 spatial padding、temporal-right padding 为零且 `cache_t<=pad_t`。CPU、
  no-cache、布局/shape/dtype/padding 不符、Triton 缺失或 launch 失败都保留原始
  `torch.cat` + `F.pad`。默认 `off`，不静默转换、不中断现有 precision policy。
- correctness：本地 CPU `test/test_wan_vae_causal_conv3d.py` 为 `6 passed,
  12 skipped`，覆盖 cache/no-cache、cache_t=1/2、spatial padding 0/1 和 fallback。
  H20 debug `--exclusive`、Torch 2.9.1+cu130，带
  `CHITU_RUN_WAN_VAE_CAT_PAD_GPU=1 CHITU_WAN_VAE_CAT_PAD=triton` 为
  `18 passed`：FP16/BF16/FP32、cache_t=1/2、padding 边界均相对
  `F.pad(torch.cat(...))` bitwise equal，max abs error `0.0`。
- H20 microbenchmark（debug `--exclusive`，BF16，精确 mover，非 Conv3D 总耗时）：
  真实 480p Wan decoder feature `[1,384,1,60,104]`、cache_t=2，torch
  `0.06253ms` -> Triton `0.03081ms`（`2.030x`）；真实 480p encoder feature
  `[1,96,4,480,832]`、cache_t=2，`1.94484ms` -> `1.05522ms`（`1.843x`）。
- 真实 checkpoint 更正与调用链：可用 1.3B checkpoint 位于
  `/home/chenyy/WORK/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth`（14B 目录也有同名
  VAE）。使用仓库 `test/test_wan.py`，调用链为
  `generator.vae_decode_step -> WanRuntimeAdapter.decode_latents ->
  WanVAE.decode -> WanVAE_.decode -> Decoder3d/ResidualBlock ->
  CausalConv3d.forward -> try_causal_conv3d_cat_pad`。进程级计数器证明 17 帧 decode
  有 `126` 次 cached 调用；严格 layout gate 支持 `110` 次。`off` 为
  `fused=0/fallback=126`，`triton` 为 `fused=110/fallback=16`，所以不是未触发
  cache 的假基准。
- H20 fixed-seed 短视频 E2E A/B：SGLang debug partition、`--exclusive`、1.3B、
  seed 42、256x256、17 帧、1 step、`torch_sdpa`、CFG/CP/UP 均为 1、
  CUDA graph 关闭。两次 `off` 的 VAE decode 为 `429.54/425.99ms`
  （均值 `427.76ms`），overall `1322.58/1199.80ms`；两次 Triton warm-cache
  decode 为 `849.46/818.99ms`（均值 `834.23ms`，相对 off **0.513x / 慢
  1.95x**），overall `1733.45/1641.21ms`（均值 `1687.33ms`，相对 off
  `0.748x / 慢 1.34x`）。首个 Triton 进程还包含 JIT 冷启动，VAE decode
  `1392.66ms`、overall `2177.09ms`，未混入稳态均值。denoise 不接触该 kernel；
  样本值 `357.73/282.46ms`（off）与 `290.22/390.83/292.53ms`（Triton runs）
  仅反映跨进程波动，不应归因于 mover。
- 输出质量：五个固定输入 MP4（两次 off、cold Triton、两次 warm Triton）SHA256 均为
  `7eabfd21626521f5f5ca501456d3a7fb89668b1823f2620ca66fa2bf31b57023`。
  解码后的 `[17,256,256,3]` uint8 视频逐帧 bitwise equal；每帧 max/mean abs
  均为 `0/0`，全局 max/mean abs `0/0`，PSNR `inf`。这证明精确性，但真实 VAE
  的大量小/中 shape 和 110 次额外 Triton launch 抵消并反转了单 shape mover 收益。
- 决策：**正确但真实 Wan VAE 路径性能回归，默认保持 `off`，不建议 production
  opt-in。** `triton` 只保留为显式实验/诊断策略，不能把前述 `1.84-2.03x`
  mover microbenchmark 报成 VAE 或端到端收益。适用路径为当前 legacy
  `chitu_diffusion/modules/vaes/wan_vae.py::CausalConv3d` 的 cached causal
  Conv3D；`wan_vae_diffusers.py`、no-cache、CPU 和任何非严格 layout 均不变。

## 2026-07-21：Qwen-Image select01 modulation/residual/norm Triton 评估

- 上游：SGLang `429f6b6d157a9d8013903c4e4260b027e7bf5386`，
  `python/sglang/jit_kernel/diffusion/triton/scale_shift.py` 与
  `python/sglang/jit_kernel/diffusion/norm_scale_shift_native.py`，Apache-2.0；
  Qwen-Image runtime 路径 `multimodal_gen/runtime/models/dits/qwen_image.py`
  保留其 FastVideo Apache-2.0 来源。上游 kernel adapter 在
  `kernel_adapters/qwen_image_modulation.py` 保留 SPDX、commit/path 和致谢，不依赖
  SGLang、`sgl-kernel`、TVM FFI 或 CuTeDSL。
- 语义审查：当前 diffusers block 的 `img_norm1 -> select01 scale/shift/gate`，
  以及 `residual + gate1 * attention -> img_norm2 -> select01` 对应上游两个
  kernel。仅在 `zero_cond_t` 生成 `[2B, 3C]` modulation 且提供
  `modulate_index[B,L]` 时具备 select01 语义；普通 Qwen 请求没有该 index，不能把该
  候选宣称为默认路径加速。CPU 和不支持 CUDA 条件严格执行原始 PyTorch
  `LayerNorm`/`where` 表达式。
- H20 correctness：`SRUN_PARTITION=debug CHITU_PROJECT_ROOT=$PWD
  CHITU_PYTHON_BIN=$PWD/.venv/bin/python script/srun_direct.sh 1 1 -m pytest -q
  test/test_qwen_image_modulation.py test/test_kernel_integration.py` 在 H20
  (SM90, Torch `2.9.1+cu130`, CUDA 13.0) 通过 `25 passed`。小 shape CUDA unit
  通过 `rtol=atol=1e-2`；CPU fallback 和单 block 一次性 residual/gate 语义为
  bitwise equal。
- H20 representative microbenchmark（`SRUN_EXTRA_ARGS=--exclusive`，
  `[1,4096,3072]` BF16，100 iters）：首次实现为普通 `7.07x`、residual `5.17x`，
  max abs `0.125/0.25`。随后按真实 diffusers dtype policy 补齐 LayerNorm 输出、
  gate multiply、residual add 和 modulation 的 BF16 中间舍入后重测：普通 reference
  `0.2417ms`、Triton `0.0344ms` (`7.02x`)，max/mean abs
  `0.125/0.000746`；residual reference `0.2895ms`、Triton `0.0484ms`
  (`5.98x`)，max/mean abs `0.1875/0.001077`。residual 误差下降但两者仍未通过
  `rtol=atol=1e-2`。剩余差异主要来自 Triton 与 PyTorch LayerNorm reduction tree；
  因此是数学表达式等价，但不是 bit-exact，不能称为精确输出复现。
- 决策：**benchmark-only / 不接入运行时默认。** 默认 precision policy 未改变；
  `CHITU_QWEN_IMAGE_MODULATION_FUSION=off` 保持默认，`benchmark_only` 仅用于
  文档化；请求 `select01` 显式报错，防止未验证 kernel 改变结果。因 correctness
  gate 未过，未运行端到端 A/B，不能报告端到端性能或质量收益。
- 后续：若需要上线，先实现能匹配 PyTorch LayerNorm reduction 的精确实现，并在真实
  Qwen zero-cond checkpoint 的 block 和端到端固定 seed A/B 同时通过严格门禁，再重新
  考虑 opt-in；不得仅以本轮 kernel latency 解除 gate。

## 2026-07-21：vLLM 精确 CP attention-state merge（真实 Ring 门禁）

- 上游与许可证：vLLM `616c9bd0f4efff9761fd762e4247e18b6bd7eb23`，Apache-2.0；
  审查 `vllm/v1/attention/ops/merge_attn_states.py` 和
  `csrc/libtorch_stable/attention/merge_attn_states.cu`。新
  `kernel_adapters/attention_state_merge.py` 保留 SPDX-equivalent Apache-2.0
  notice、commit/path provenance 和 vLLM 社区致谢。
- 数学与 layout：vLLM 的
  `m=max(Lp,Ls)`、`O=(exp(Lp-m) Op + exp(Ls-m) Os)/(exp(Lp-m)+exp(Ls-m))`、
  `L=log(exp(Lp-m)+exp(Ls-m))+m` 与 Chitu 原有
  `sigmoid/logsigmoid` online softmax 式严格等价。Chitu 的实际状态为
  `out [B,S,H,D]` FP32、`lse [B,S,H,1]` FP32、`block_out [B,S,H,D]` 和
  `block_lse [B,H,S]` FP32；适配层直接保留该 contract，并按 vLLM 的 contract
  将 `±inf` partial LSE 归为 empty state，`(-inf,-inf)` 时保留 prefix output、
  写回 `-inf`，避免 `0/0` NaN。
- 依赖决策：当前 `.venv` 未安装 vLLM；即使已安装，vLLM public op 要求
  token-major `[T,H,D]` 且非 FP8 output 与 input 同 dtype，不能保留 Chitu
  FP32 running output，直接调用会在每轮 merge 降精度。因此实现最小的可选 Triton
  兼容层，而不是新增 vLLM runtime/build dependency。仅 CUDA、`D<=256`、
  FP32 state/LSE、FP32 block LSE、BF16/FP16/FP32 block output 才 launch；
  running LSE 支持真实 CP 从 `[B,H,S]` transpose 得到的 strided `[B,S,H,1]`。
  任何不匹配、Triton 缺失、编译或 launch 异常都回退既有 JIT merge。
- correctness：CPU unit/reference 为 17 passed、4 skipped；exclusive H20
  (`debug`，Torch 2.9.1+cu130) 完整 kernel integration 为 21 passed。Triton
  覆盖 BF16 和 FP16 block state，和 FP32 stable reference 对照，容差
  `rtol=atol=2e-5`。2-GPU H20 `--exclusive`、每 rank
  `[1,4096,24,128]` BF16 shard 的 all-rank max errors：output `9.537e-07`、
  LSE `2.384e-07`。
- production-layout microbenchmark：2-GPU H20 exclusive、每 rank
  `[1,4096,24,128]` BF16、strided LSE，reference `0.3890ms`、Triton
  `0.1315ms`，`2.957x`。这是 local merge gate，不含 Ring P2P、attention
  或 denoise，不能外推为端到端 CP 收益。
- 中性包名迁移后复验：所有 runtime、test、benchmark 与 skill 引用统一改为
  `kernel_adapters`，不保留旧包兼容别名；standalone benchmark 显式加入 repo root，
  避免 editable-install 残留掩盖导入问题。
- 真实调用链与首次阻塞：
  `test/test_wan.py -> generator._run_dit_forward ->
  DiffusionAttention_with_CP._cp_forward -> runtime.parallel_utils.update_out_and_lse ->
  try_merge_attention_states`。配置为
  Wan2.1-T2V-1.3B、2×H20、`cp_size=2/up=1/cp_backend=ucp`、eager
  `ring_cudagraph=false`、`torch_sdpa`。首轮 trace 为每 rank
  `calls=300,triton=0,unsupported=300`，定位到真实 running LSE 非 contiguous；
  kernel 改为显式 stride addressing 后，trace 为每 rank
  `calls=300,triton=300,unsupported=0,errors=0`。
- 真实 E2E A/B：Slurm `debug --exclusive`，seed 42、832x480、17 帧、5 steps、
  同 checkpoint/backend/CFG。稳定复跑 `off` 的 denoise `2540.60ms`、
  `dit_forward total=2506.66ms`；Triton 为 `3006.20ms`、`2965.71ms`，
  分别慢 `18.3%/18.3%`。差异来自首次 Triton JIT：step 0
  `1315.46ms` 对 `856.40ms`；去掉 step 0 后四步均值
  `412.563ms` 对 `412.565ms`，无可测稳态收益（约 `1.00000x`）。
- 输出质量：两次 `off` MP4 逐帧 bitwise equal（17 帧，mean/max abs `0/0`、
  PSNR `inf`、SSIM `1.0`），证明基线可复现。Triton 对稳定 `off` 的逐帧汇总：
  mean abs `2.962/255`、max abs `245/255`、mean/min PSNR
  `30.035/28.591dB`、mean/min SSIM `0.93211/0.92461`。数学/LSE contract
  等价及单次 strict error 不能解释为扩散轨迹 bit-exact。
- 决策：**降级为 benchmark-only，禁止 production opt-in。**
  `CHITU_CP_ATTN_STATE_MERGE=off` 是唯一运行时策略；设置 `triton` 明确报错，
  隔离 harness 才使用 `benchmark_only`。已验证路径为 eager Ring 中调用
  `runtime.parallel_utils.update_out_and_lse` 的连续 state；AGKV/pure Ulysses
  没有 per-block online merge，不受影响。现有结果同时未通过端到端性能和轨迹质量
  默认门禁；未来只有消除 JIT 首步开销、证明稳态净收益并设定可接受的质量阈值后才重审。
- 可复现实验：
  `SRUN_PARTITION=debug SRUN_EXTRA_ARGS=--exclusive CHITU_PYTHON_BIN=$PWD/.venv/bin/python script/srun_direct.sh 1 2 kernels/benchmark_cp_attention_state_merge.py --require-world-size 2 --seq-len 4096 --heads 24 --head-dim 128 --dtype bf16`。

## 2026-07-21：H20 dual-norm 与 Qwen varlen 实测

- 环境：Slurm `debug`、单张 NVIDIA H20（SM90）、Torch `2.9.1+cu130`；端到端对比使用
  `--exclusive`、固定 seed 42、1024² Z-Image、10 steps、CFG=1、`torch_sdpa`。一次未加
  `--exclusive` 的 `native_bf16` 运行中 text encoder、DiT 和 VAE 同时异常变慢，判定为
  节点干扰并废弃，未用于结论。
- Z-Image `[1,4096,3840]` microbenchmark：
  - `diffusers_fp32`：RMSNorm+scale `15.17x`，norm+tanh+residual `11.86x`；分别
    max abs `0.046875/0.0625`、cosine `1.0/0.99999797`。
  - `native_bf16`：对应 SGLang BF16 reference 为 `8.85x/7.62x`；分别
    max abs `0.25/0.125`、cosine `0.99999690/0.99999726`。
  - 两种模式均通过当前 `rtol=2e-2, atol=5e-2` 门禁，但 `allclose` 不能解释为 bit-exact。
- Z-Image 端到端 exclusive denoise：`off=5173.57ms`、`diffusers_fp32=5039.26ms`
  （`1.0267x`）、`native_bf16=5036.62ms`（`1.0272x`）。两种 fusion 在该配置上的性能
  基本相同，选择 BF16 reduction 没有额外可测收益。
- 固定 seed 的最终 PNG 对 `off`：`diffusers_fp32` 为 PSNR `38.86dB`、SSIM `0.98770`、
  mean abs `1.581/255`；`native_bf16` 为 PSNR `36.65dB`、SSIM `0.98240`、
  mean abs `2.100/255`。因此 `diffusers_fp32` 是同一 FP32-reduction **policy**，但由于
  Triton reduction 顺序、融合舍入和 tanh 实现仍不是 bit-exact。按新的 recipe
  门槛，默认 `auto -> diffusers_fp32`：它保持模型 reduction policy 且有 `2.67%`
  denoise 收益；`native_bf16` 仍仅作为实验覆盖项。
- GPU correctness：`test/test_kernel_integration.py test/test_qwen_varlen_attention.py`
  在 H20 上 `20 passed`。
- Qwen masked shape `[1,5120,24,128]`（1024 text slots 中 128 valid，4096 image）：
  - FA2 varlen pipeline `2.902ms`，masked SDPA `2.514ms`，仅 `0.866x`，拒绝作为该形状默认。
  - FA3 varlen pipeline `2.071ms`，masked SDPA `2.513ms`，`1.213x`；max abs
    `9.77e-4`、cosine `0.9999956`，可作为 Hopper masked-batch opt-in。
  - 当前单请求 Qwen 端到端中正/负 prompt 各自为 batch 1 且没有 padding mask，实测日志
    `mask=None`；因此该路径不会触发 varlen，不能把 microbenchmark 收益宣称为当前单请求
    端到端收益。只有未来 padded request batching 或显式 mask 输入才受益。

## 2026-07-21：SGLang fused QK RMSNorm + RoPE（Flux2 第一阶段）

- 上游：SGLang `d6ef68881e26`，`python/sglang/jit_kernel/diffusion/qknorm_rope.py`
  和 `python/sglang/jit_kernel/csrc/diffusion/qknorm_rope.cuh`，Apache-2.0。
- 实现：核心 warp-level Q/K RMSNorm + interleaved RoPE 被独立适配为
  `kernel_adapters/qknorm_rope.py` 与最小 PyTorch CUDA extension。不导入 SGLang、
  sgl-kernel、vLLM 或 TVM FFI；lazy JIT build 仅发生在第一次满足条件的调用。源码保留
  SPDX、commit/path provenance 和对 SGLang 社区的致谢。
- Flux2 语义：已证明可直接使用 diffusers `Flux2PosEmbed` 产生的逐 token、多轴
  `(cos, sin)`（`[S, 128]`），不将其误作一维 position cache。dual processor 按
  text/image 分段，使用各自 `norm_added_*` / `norm_*` 权重后再拼接；single processor
  使用完整序列。因此没有修改 Qwen、Flux1、Z-Image 或 Wan。
- 当前 recipe 默认 `auto` 尝试，也可用 `off` 禁用；其后仍仅限
  eager/inference、CUDA SM90、BF16/FP16、contiguous BSHD、`Dh=128`、
  FP32 contiguous `[S,128]` cos/sin 和同设备输入尝试融合。任何不满足条件、JIT build
  或 launch 失败都走原 diffusers RMSNorm + `apply_rotary_emb`。
- H20 最终验证：SM90 FP16/BF16 correctness 已通过；真实 shape microbenchmark
  约 `15x`，Flux2-klein 端到端约 `1.5%`。默认值更新后再次通过 H20
  `debug --exclusive` 测试，相关 suite 为 `26 passed`。不支持的硬件、layout、
  compile 或 launch 均保持 PyTorch fallback。

## 2026-07-21：Qwen-Image 单卡 masked padded-varlen 第一阶段

- 上游：SGLang `python/sglang/jit_kernel/diffusion/triton/varlen_pack_pad.py`
  与 `multimodal_gen/runtime/layers/attention/layer.py`；Apache-2.0。后者保留的
  FastVideo 来源致谢也已在 Chitu 适配层保留。
- 目标：仅 Qwen-Image 单卡 joint self-attention 的 `[B, text + image]` bool valid
  mask。Flux/Flux2/Z-Image 和所有 CP 路径未接入；masked CP 明确拒绝。
- 实现：`kernel_adapters.varlen` 提供 metadata、pack 和 scatter 稳定 API；Triton 可用时
  使用 fused gather/scatter，否则用精确 PyTorch index 操作。全有效 mask 保持 dense；
  FA2/FA3 可用时走 packed varlen，`torch_sdpa` 使用 bool key mask 并将无效 query
  rows 归零，匹配 pack/scatter 合约。
- correctness：新增 CUDA（无 CUDA 自动 skip）pack/scatter 与 masked SDPA 对照测试；
  另有 CPU fallback 测试。H20 测量尚未在本轮执行。
- microbenchmark：`kernels/benchmark_qwen_varlen_attention.py` 分别输出 metadata、pack、
  attention、scatter、total；只测 FA2/FA3 packed varlen，以免把 SDPA fallback 的不同
  执行图误报为 pack/scatter 性能。
- 决策：适用，等待 H20 correctness 和真实 Qwen-Image 端到端基准；若 pack/scatter
  开销抵消 attention 收益，保持 `torch_sdpa` fallback。

## 2026-07-21：Z-Image 双语义 RMSNorm Triton fusion（第一阶段）

- 上游：SGLang `d6ef68881e263812d4901f632786015005c4d050`，
  `python/sglang/jit_kernel/diffusion/triton/zimage_native_norm.py`，Apache-2.0。
  `kernel_adapters/zimage_native_norm.py` 保留来源、许可证和 SGLang 社区致谢。
- 目标：diffusers `ZImageTransformerBlock` 的 BF16 RMSNorm+scale，以及
  RMSNorm+tanh(gate)+residual；适用条件为 CUDA、Triton、连续 layout、hidden dim
  `<=8192` 和所有输入 BF16。
- 语义差异：diffusers 将输入转为 FP32 后计算 variance/rsqrt，再转回权重 dtype；
  SGLang native kernel 则将平方、均值与 rstd 量化为 BF16。两者并非严格等价，差异来自
  reduction 精度而不是 fusion 本身。
- 当前策略：`CHITU_Z_IMAGE_NORM_FUSION=auto|off|diffusers_fp32|native_bf16`，
  默认 `auto -> diffusers_fp32`；
  `diffusers_fp32` 保持默认模型的 FP32 reduction policy，`native_bf16` 复现 SGLang 的
  实验 reduction policy；二者都不是相对 unfused 路径的 bit-exact 输出。
  旧 `CHITU_Z_IMAGE_NATIVE_NORM=1` 暂映射到 `native_bf16` 并输出弃用日志。
- 实现：block-level monkey patch 将 mode 明确传给 kernel；每次 kernel 不满足条件均
  返回 `None` 并执行相同 mode 的 unfused reduction。gate 始终以原始 logits 传入，
  fused 与 fallback 路径各自恰好执行一次 `tanh`。
- correctness：CPU 覆盖默认/旧环境变量解析、不可用 fallback、adapter 幂等和完整 block
  的单次 tanh 语义；CUDA 测试分别覆盖 FP32 与 BF16 reduction。H20 kernel tolerance、
  microbenchmark 与真实 checkpoint 端到端结果见本文件顶部，均已执行。
- microbenchmark：新增 `kernels/benchmark_zimage_native_norm.py`，真实默认形状为
  `[1,4096,3840]` BF16；分别将 FP32 fusion 对照 diffusers reference、BF16 fusion
  对照 SGLang reference，并报告 max/mean absolute error、cosine、allclose、延迟和 speedup。
- 决策（后续 recipe 更新）：默认 `auto -> diffusers_fp32`。H20 有约 `2.7%`
  denoise 收益，且最终图非 bit-exact 本身不构成质量下降证据。

## 2026-07-21：SGLang / vLLM 同类 diffusion model kernel 审查

### 同类模型覆盖

- SGLang multimodal generation 已有 Flux、Flux 2、Qwen-Image、Z-Image、Wan 等 DiT
  实现，算子形状和 block 语义与 ChituDiffusion 高度重合；它是本轮最有价值的复用源。
- vLLM 主仓仍以自回归、paged KV cache 和 decode 为主，适合提供底层 GEMM/norm/attention
  来源，但大部分 LLM kernel 不能直接用于每个 denoise step 都重算 Q/K/V 的 DiT。
- vLLM-Omni 覆盖 Flux、Flux 2、Qwen-Image、Z-Image、Wan 2.2 等模型，但其 CUDA
  diffusion 路径主要组合 vLLM 通用线性层与现有 attention backend，Hopper 上没有发现
  比 SGLang diffusion JIT 更直接的新模型内 fusion。

### 可复用候选与优先级

1. **P0：varlen FA 的 Triton pack/scatter（精确算子）。** Chitu 已能调用
   FlashAttention varlen API，但 mask 到紧凑 Q/K/V、以及结果 scatter 回 padded layout
   仍缺少 SGLang 的
   `jit_kernel/diffusion/triton/varlen_pack_pad.py` 快路径。它直接覆盖 mask-heavy 的
   Qwen-Image，以及带 padded joint sequence 的 Flux/Flux2/Z-Image；先移植 pack/scatter，
   不移植完整 USP/Ulysses 栈。
2. **P0：Z-Image 原生 norm fusion（精确算子）。** SGLang
   `jit_kernel/diffusion/triton/zimage_native_norm.py` 将
   RMSNorm+scale 以及 RMSNorm+tanh(gate)+residual 分别融合；其 `zimage.py` 在主 block
   的 attention 和 FFN 两侧直接使用。Chitu 当前也使用同一 diffusers Z-Image block
   语义，适合先做独立兼容层与 H20 microbenchmark，不需要改变模型质量。
3. **P0：QK norm + RoPE fusion（精确算子）。** SGLang
   `jit_kernel/diffusion/qknorm_rope.py`/`layers/layernorm.py` 已用于 Z-Image 等模型，
   Flux/Flux2/Qwen-Image/Z-Image 的 head dim 都是 128，满足该 JIT kernel 的
   BF16/FP16、contiguous 4D Q/K 约束；可减少 Q/K normalize、cast、RoPE 的多次
   launch 与中间张量。应提取 SGLang diffusion 版本；vLLM 同名 kernel 要求 packed
   QKV+position_ids，不能直接复用。Wan 的 across-head QK norm 和 3D RoPE 也不能走
   该入口。
4. **P0：modulation/residual/norm fusion（精确算子）。** SGLang 提供
   `norm_scale_shift_native.py`、CuTeDSL
   `scale_residual_norm_scale_shift.py`、`norm_tanh_mul_add_norm_scale.py`，
   并在 Qwen-Image/Wan block 中融合 LayerNorm、scale/shift、gate 和 residual。
   Qwen-Image 的 `triton/scale_shift.py` select01 路径是其中最容易独立提取的实现；
   Wan hidden=5120 满足 CuTe fusion 的维度条件，但构建风险更高。
   该类 elementwise/norm launch 在 attention 已优化后更值得测，但 CuTeDSL 路径需先
   核对 Torch 2.9.1/CUDA 13 兼容性，不能整体依赖要求 Torch 2.11 的 sgl-kernel。
5. **P1：CP attention-state merge（精确算子）。** vLLM
   `v1/attention/ops/merge_attn_states.py` 与
   `csrc/libtorch_stable/attention/merge_attn_states.cu` 实现的 LSE 加权 merge，
   与 Chitu `runtime/parallel_utils.py::_update_out_and_lse` 数学语义相同；可替换 ring CP
   每轮的 JIT-script elementwise 序列。由于只在多卡 CP 生效，应结合现有
   `experiments/fast_cp` profiler 决定是否早于单卡 block fusion。
6. **P1：activation/elementwise 与 VAE fusion（精确算子）。** vLLM
   `silu_and_mul` 精确匹配 Flux2 SwiGLU；SGLang 另有 `residual_gate_add.py`、
   `group_norm_silu.py` 和 `causal_conv3d_{cat_,}pad.py`，分别覆盖 DiT gate/residual、
   VAE GroupNorm+SiLU 和 Wan 类 3D VAE padding。应由 profiler 决定顺序，不能仅凭
   microbenchmark 接入。
7. **P1：FP8 linear/GEMM。** SGLang diffusion 的 FP8 路径支持 BF16/FP16 activation，
   ModelOpt FP8 要求 compute capability >= 8.9，H20（SM90）满足；可针对
   Flux/Qwen/Z-Image/Wan 的 projection/MLP 评估在线或序列化 FP8。它会改变数值，
   必须增加 ChituBench 质量门禁，不能作为默认 exact baseline。
8. **P2：Nunchaku SVDQuant。** SGLang 已在 Flux、Qwen-Image、Z-Image 中组合
   fused QKV、AdaLN 和 NunchakuFeedForward。它依赖专用量化 checkpoint/runtime，
   适合作为独立量化 backend，不应把 Nunchaku 实现复制进 upstream kernel adapters。
9. **benchmark-only：Wan 稀疏 attention。** SGLang 的 Video Sparse Attention
   以 `(4,4,4)` 时空 tile、head dim 64/128 服务 Wan，并另有 sparse-linear、
   Sparse-VideoGen、Sage/SLA 等入口。它们改变 attention 近似且依赖额外包/稀疏策略；
   只有在固定 Wan 配置上同时通过速度、显存和视频质量门禁后才考虑采用。

### 暂不采用

- vLLM-Omni 的 cuDNN attention 自动路由主要针对 Blackwell+cuDNN 9.5；H20 上现有
  实测已表明 PyTorch SDPA 略快于 FA3，因此不应仅复制其 backend 优先级。
- vLLM-Omni Quack FP8 fused-bias GEMM 使用 `tcgen05`，仅 datacenter Blackwell
  SM10.x，不能在 H20 使用；NVFP4/FA4/SageAttention3 也保留到 Blackwell 节点验证。
- paged attention、KV-cache quantization、decode、MoE 和 sampling kernel 与当前
  diffusion 热路径不匹配，继续只作为上游来源观察。

### 下一验证

按“纯 Triton、低耦合”先建立三组独立 harness：varlen pack/scatter、Qwen select01
modulation、Z-Image 两个 native norm fusion；随后评估需要 CUDA JIT 的 QK norm+RoPE。
全部抽取为 `kernel_adapters` 可选实现，使用 `srun_direct.sh` 在 `debug` 分区做
correctness + 真实 shape microbenchmark；只有 microbenchmark 获益后再进入对应模型
端到端和质量对比。CP merge kernel 单独并入 `experiments/fast_cp` 测试，不与单卡结果混合。

## 2026-07-21：建立上游跟进闭环并接入 FlashAttention 3

### 背景

现有统一 attention backend 支持 FlashAttention 2、FlashInfer、SageAttention、
SpargeAttention 和 PyTorch SDPA，但 FlashAttention 3 的导入分支被注释，缺少持续跟踪
上游 commit、diffusion 形状 microbenchmark 和决策记录。

### 上游初筛

- 本轮浅克隆并记录：SGLang `d6ef68881e26`、vLLM `616c9bd0f4ef`、
  HPC-Ops `82d80e8d2ed9`。
- **SGLang / sgl-kernel：优先跟进，但暂不直接依赖。** 已有完整
  `python/sglang/multimodal_gen` diffusion 路径；QK norm + RoPE、MLP fusion
  与 trajectory quality 工具可复用。当前 `sglang-kernel` 明确要求 Torch 2.11，
  而 ChituDiffusion 锁定 Torch 2.9.1，因此应按算子提取 API/实现，不能整体安装。
- **vLLM-Omni：优先跟进。** diffusion attention backend 可作为 cuDNN attention、
  SageAttention 3、FlashInfer 的接口与 fallback 参考。
- **FlashAttention / FlashInfer / SageAttention：优先跟进。** 可直接形成 attention
  baseline，但需要按 GPU 架构、head layout、varlen 和 CP logsumexp 能力分别验证。
- **Tencent HPC-Ops：定向跟进。** 当前 head 要求 SM90、CUDA 12.8+、BF16/FP8。
  `attention_prefill_bf16` 的 NHD 接口与 DiT 接近，可进入 H20 benchmark-only 阶段；
  但它不返回 CP merge 所需 logsumexp。RoPE/QK norm 与 allreduce+residual+RMSNorm
  接口都绑定 paged KV cache 或 LLM RMSNorm 语义，不能直接接入现有 DiT block。
  FP8 block-sparse prefill只有在 ChituDiffusion 能产生合适 block mask、并通过质量门禁后
  才值得接入。
- **vLLM：发现来源而非直接依赖。** 当前树包含 FA4 warmup、HPC attention、
  TurboQuant 等实现，但重点仍是 paged KV-cache/decode。KV-cache 量化、MoE、sampling
  通常不适用于每个 denoise step 重算 Q/K/V 的 DiT；只跟进可复用的 dense attention、
  GEMM、norm/activation fusion 和通信算子。审查期间捕获的 `8688a06..616c9bd`
  增量只修改 translation API sampling 参数，与 kernel 无关。

### 实现

- 新增 `kernels/upstreams.toml` 与 `script/kernel_watch.py`，支持 remote scan、浅克隆到
  `refs/kernels/`，以及记录人工审查 commit。
- 新增项目 skill `.cursor/skills/kernel-upstream-sync/SKILL.md`，固定“发现、筛选、
  correctness、microbenchmark、端到端、worklog、lock”顺序；同一流程也放入仓库
  canonical `.codex/skills/` 并增加 Codex UI metadata。
- 新增 `kernel_adapters` 兼容层，统一官方 `flash_attn_interface` 与 Hugging Face
  `kernels-community/flash-attn3` 的加载和返回值。
- 将 `flash_attn3`/`fa3` 接入统一 attention backend。只在 Hopper SM90 可用；
  显式请求不可用实现时直接报错，不静默 fallback。
- 新增 diffusion attention benchmark，包含 smoke、Z-Image、Qwen-Image 和 Wan
  长序列 preset，输出环境、误差、延迟和相对 PyTorch SDPA speedup 的 JSON。
- 补充 FA3 安装文档、默认配置注释，以及 Flux/Wan 显式 SM90 smoke 配置；FA3 不加入
  跨架构默认矩阵，避免非 Hopper 节点出现预期外失败。
- 发现 Sage/Sparge 存在双 source of truth：`.gitmodules` 使用 `shch-y/*` fork，
  `pyproject.toml` 运行时却安装 `thu-ml/*`。后续升级必须同时核对，不能只更新 submodule。

### 验证状态

- 使用 `script/srun_direct.sh` 在 `debug` 分区申请单卡 NVIDIA H20（SM90），环境为
  Torch 2.9.1+cu130 / CUDA 13.0。HF loader 固定为 `kernels==0.11.7`；更新版本要求
  `huggingface-hub>=1.10`，与当前 Transformers 4.56.2 的 `<1.0` 约束冲突。
- `kernels/benchmark_attention.py`，BF16 correctness 全部通过：
  - 512×8×64：SDPA 0.0221 ms，FA3 0.0906 ms，FA2 0.0390 ms；FA3 不适合短序列。
  - 4096×30×128：SDPA 1.9265 ms，FA3 1.9847 ms，FA2 3.0046 ms；FA3 比 FA2
    快 1.51×，但比 SDPA 慢 3.0%。
  - 4096×24×128：SDPA 1.4871 ms，FA3 1.5303 ms，FA2 2.3203 ms；FA3 比 FA2
    快 1.52×，但比 SDPA 慢 2.9%。
  - 32760×12×128：SDPA 47.2075 ms，FA3 47.4055 ms，FA2 73.5640 ms；FA3 比
    FA2 快 1.55×，与 SDPA 基本持平（慢 0.4%）。
- Flux1-dev 1024²、10 step 端到端：
  - FA2：DiT 770.11 ms/step，denoise 7.728 s，overall 8.819 s。
  - FA3：DiT 762.95 ms/step，denoise 7.655 s，overall 8.968 s。首个 FA3 step
    有一次性加载开销；排除各自最慢 step 后 FA3 为 704.46 ms，FA2 为 764.94 ms，
    稳态快 8.6%。
  - PyTorch SDPA：DiT 711.99 ms/step，denoise 7.144 s，overall 8.244 s；排除最慢
    step 后 699.57 ms，仍比 FA3 快约 0.7%，且无 FA3 首次加载惩罚。
  - FA3 与 FA2 输出图：PSNR 55.45 dB、SSIM 0.99798；FA3 与 SDPA：PSNR
    55.14 dB、SSIM 0.99798。
- **决策：** FA3 作为显式 Hopper backend 保留，可显著替代 FA2；H20 的 exact
  performance baseline 仍为 PyTorch SDPA，`auto` 因此优先 SDPA。尚需验证 CP
  logsumexp 和真实 Wan/Z-Image 端到端。

### 后续候选

1. 从 SGLang diffusion 提取 QK norm + RoPE 和 MLP fusion 的适配面。
2. 在 Blackwell 节点比较 cuDNN attention、SageAttention 3 与 FlashAttention 4。
3. 只在 profiler 证明稀疏 attention 占优且能稳定生成 block mask 后，评估 HPC-Ops
   FP8 block-sparse prefill。

## 2026-07-24：fast-ulysses NVSHMEM all-to-all

- 上游：`triple-mu/fast-ulysses`，commit
  `6e5dcb24dc44e781ac3091d1d9b3f9fef314fb87`，Apache-2.0；检出到
  `refs/fast-ulysses/`，并加入 `kernels/upstreams.toml`。
- 目标：单节点 H20 NVLink 上替换纯 Ulysses CP 的 NCCL
  `all_to_all_single`；保持 AGKV、Ring 和跨节点路径不变，并支持 CFP2
  产生的两个 CP subgroup。
- 环境：8×NVIDIA H20（SM90），Torch 2.9.1+cu130，CUDA 13.0 编译工具，
  NVSHMEM 3.7.2。由于 Torch 2.9.1 元数据固定 NVSHMEM 3.3.24，验证时把
  3.7.2 隔离到 reference checkout，未改变项目锁文件。
- correctness：
  - 上游 `a2a_correctness.py` 在 CP8 对 fp16/bf16、D=64/128/256/512、
    mode0/mode1、auto/TMA/non-TMA/CE、tag alias 和混合 epoch 全部通过。
  - 上游 `a2a_async.py` 的 async、3 in-flight QKV、mode1 roundtrip、
    grouped barrier 和 CE deep-pile 全部通过。
  - `experiments/fast_cp` 的 4096 image token、512 text token、H=24、D=128
    CP8 serial/overlap 均与 NCCL 基线逐元素一致。
  - 2026-07-26 新增 NVSHMEM subgroup 回归：4 GPU 上两个连续 CP team
    使用不同 shape 和不同 collective 次数均逐元素一致且正常 teardown；
    Chitu `cfg_size=2, cp_size=2` 的 `torch_sdpa` attention 与完整序列参考一致。
- microbenchmark：
  - CP8、H=24、D=128、N=4096：mode0 89.2→40.9 μs（2.18×），
    mode1 60.0→40.1 μs（1.50×）。
  - N=32760：mode0 155.5→97.0 μs（1.60×），mode1
    150.5→117.0 μs（1.29×）。
  - Z-Image-like CP core：NCCL serial 1.088 ms，fast-ulysses
    1.014 ms（快 6.8%）；overlap 1.095→1.003 ms（快 8.4%）。
  - CE sync core 为 1.100 ms，没有收益，因此未接入生产路径。
- 端到端：Z-Image 1024²、3 steps、CP2、FlashAttention、同一 H20 节点：
  NCCL denoise 2375.42 ms，fast-ulysses 2130.12 ms，快 10.3%；
  overall 3590.34→3375.08 ms，快 6.0%。两张输出 PNG 逐像素一致
  （max abs=0、SSIM=1.0）。Z-Image 当前 30 heads 不能整除 CP8，CP8 会按既有
  逻辑退回 AGKV，因此该模型的真实 Ulysses A/B 使用 CP2。
- 2026-07-26 Wan subgroup ChituBench：Wan2.1-T2V-1.3B、50 steps、
  81×832×480、FlashAttention、固定 UCP、同一 `bjdb-h20-node-020`，两条 prompt：
  - CFP2+CP2：NCCL/fast DiT-forward 80.521→79.700 s（快 1.0%），
    denoise 80.912→80.091 s。
  - CFP2+CP4：42.779→41.186 s（均值快 3.7%），但首请求快 6.7%、
    第二条稳态请求仅快 0.6%，说明收益包含 NCCL first-collective 冷启动成本。
  - 相同 prompt 的 NCCL/fast、CP2/CP4 输出 MP4 SHA-256 完全一致。
- 2026-07-26 Flux.1 fast-only ChituBench：50 steps、1024²、FlashAttention、
  固定 UCP、同一 `bjdb-h20-node-020`，每个 case 完整 warmup 2 次后测 1 次：
  - CP2/CP4/CP8 DiT-forward 为 20.369/11.230/6.853 s；
    measured end-to-end 为 20.545/11.377/6.980 s，CP4/CP8 相对 CP2
    E2E scaling 为 1.81×/2.94×。
  - CP4/CP8 输出相对 CP2 为 43.47/42.27 dB PSNR、
    0.9957/0.9955 SSIM；差异来自不同 CP reduction ordering。
- 2026-07-26 补充 8-GPU fast-only ChituBench（各完整 warmup 2 次后测 1 次）：
  - Flux2-klein-4B、4 steps、1024²、CP8：DiT-forward 0.232 s，
    denoise 0.236 s，measured end-to-end 0.329 s；相对历史单卡 DiT
    baseline 为 5.86×。
  - Qwen-Image、50 steps、1328²、CFP2+CP4 subgroup：DiT-forward
    23.708 s，denoise 24.111 s，measured end-to-end 24.265 s；
    相对历史单卡 DiT baseline 为 5.86×。Flux2 与 Qwen 分别运行在
    `bjdb-h20-node-020` / `bjdb-h20-node-021`，按本次实验约定不校正节点差异。
- 实现：新增 `kernel_adapters.fast_ulysses` 可选适配器；
  `CHITU_CP_ULYSSES_BACKEND=auto` 默认在全 rank 探测到 Hopper、单节点
  CUDA P2P、可加载扩展及所需 subgroup feature 时启用，否则保持 NCCL；
  `fast_ulysses`/`nccl` 可强制选择。自动/强制 TMA 由
  `CHITU_FAST_ULYSSES_USE_TMA` 控制。固定四个 tag slot 在每次 attention
  前重置，避免 Q/K/V alias 和跨层 symmetric-heap 增长；终止时在所有 rank
  collective destroy。Image CP 和通用 pure-Ulysses 都复用同一个 full-CP
  NVSHMEM group，避免为等价的 full-size UP group 再分配一份 symmetric heap；
  unified Ulysses+Ring 的较小 UP subgroup 保留现有 NCCL 路径。Subgroup 支持通过
  WORLD 一次性 symmetric arena、subgroup-local bump pool 和全 rank
  确定性创建 NVSHMEM team 实现，运行期 allocation/autotune/barrier 不跨 CFP 分支。
- 门禁：只允许单节点、4D Ulysses layout，以及覆盖 WORLD 的连续、不相交
  subgroup partition；强制模式下包、NVSHMEM、拓扑或 subgroup 不满足时显式
  报错，auto 模式则一致回退 NCCL。
  3D varlen、FP8 和 CUDA Graph capture 保留 NCCL 路径（graph replay 不能复用
  host 侧递增的 NVSHMEM barrier epoch）。
- 安装：新增 `uv sync --extra fast-ulysses`，锁定上游 `6e5dcb2`；构建仍要求
  外部 CUDA toolkit 与 NVSHMEM 3.7+。锁定的上游包仅支持 WORLD group，
  CFP2 自动启用还要求带 `SUPPORTS_WORLD_PARTITION_SUBGROUPS` 标记的 Chitu
  subgroup build。
- 决策：**作为 Ulysses 内部 capability-gated auto transport 采用**，但
  NVSHMEM 保持可选依赖。后续评估 async grouped handshake；当前不采用 CE 路径。

### 2026-09-17：Fast CP 原生框架基线与 Wan14B scaling

- 上游：Chitu `d4d3a954`、xDiT `5ede5851`、SGLang Diffusion `44bd3590`。
- 目标：H20，BF16，H40/D128，S=4096/8192/16384；同一节点分别保留三个
  repo 的原生通信、attention backend、布局与同步语义。
- correctness：所有 QKV-to-O 和 Wan attention 端到端 cell 均通过同输入的
  dense reference；Wan 使用全局位置 RoPE，参数和输入为确定性随机张量。
- QKV-to-O：CP8 Fast Ulysses 在 S=4096/8192/16384 为
  0.640/1.751/5.747 ms，相对 Chitu NCCL Ulysses 快 10.4%/13.0%/2.8%；
  Fast AGKV 与 NCCL AGKV 差异不超过 0.4%。S=4096 trace 显示三次输入 TMA、
  attention 和一次输出 TMA 在同一 stream 串行执行，没有计算通信 overlap。
- QKV-to-O CP sweep：H40/S4096 的 CP2/CP4 在 node040，Fast Ulysses 为
  1.547/0.952 ms，相对同模式 NCCL 快 12.8%/9.7%；结合 node021 的 CP8
  0.640 ms，同 CP 内快 9.5%。xDiT Ulysses CP2/4/8 为
  1.995/1.229/0.978 ms，SGLang Ulysses 为 1.836/1.110/0.781 ms；节点不同，
  不用这三列计算精确跨 CP scaling。
- Wan14B：CP1 为 9.623 ms。Fast Ulysses CP2/4/8 为
  5.244/2.787/1.589 ms，即 1.835x/3.453x/6.055x；相对同 CP 的 NCCL
  Ulysses 快 4.9%/5.1%/3.5%。保持 CE kernel、只串行化依赖后的 overlap
  收益为 0.046/0.090/0.028 ms。Fast AGKV CP8 自动关闭 async，1.774 ms，
  比 NCCL AGKV 的 1.741 ms 慢 1.9%。
- scaling gap：Fast Ulysses CP2/4/8 相对理想线性时间仍多
  0.433/0.381/0.386 ms，并行效率为 91.8%/86.3%/75.7%。通信 overlap 为正，
  但不足以填平约 0.4 ms 的固定残差；下一步拆 projection、RoPE/layout、
  attention tail、输入交换和输出交换。
- Wan1.3B：CP1 为 1.578 ms。H12 的短计算路径上，Fast Ulysses CP2/4
  比 NCCL 慢 14.5%/19.9%，CE async schedule 也没有 overlap 收益；CP8
  使用原生 head padding 后与 NCCL Ulysses 基本持平。Fast AGKV CP8 为
  0.621 ms，相比 NCCL AGKV 的 0.950 ms 快 34.6%。
- joint attention：4096 image + 512 replicated text，Qwen-Image CP1 为
  5.815 ms，Fast Ulysses CP2/4/8 为 3.398/2.116/1.541 ms；Z-Image CP1
  为 7.500 ms，Fast Ulysses 为 4.547/3.009/1.957 ms。后者 H30 在 CP4/8
  使用原生 head padding，CP8 仍比 NCCL Ulysses 快 13.5%。两组均包含
  projection、QK norm、RoPE、CP attention 和 output projection，并通过
  dense reference correctness。
- 产物：`tools/benchmarks/fast_cp_frameworks/README.md` 定义公平性协议，
  `RESULTS.md` 保存本轮表格和解释，JSON 与 Nsight trace 位于
  `outputs/fast-cp/`（不纳入版本控制）。
- 待验证：已补 xDiT 原生 Wan attention fixture 和 launcher，静态检查通过；
  debug 分区在完成上述矩阵后最早 21:47 才有资源，尚未发布其 GPU 数字。

## 2026-09-18：Fast CP 原生模型前端与框架基线

- **Wan shape 更正：**此前模型图将通用 CP attention 的 S=4096 误用为 Wan
  模型长度，因此其中 Wan 模型点及相关结论作废。正式负载为 Wan 1.3B
  `B1/S32760/H12/D128`、Wan 14B `B1/S75600/H40/D128`。已用三个仓库的原生
  路径重跑 48 个终态 cell；有效 Wan 1.3B 结果均在 node035，Wan 14B 均在
  node021。每个 cell 使用 3 次 warmup、3 组各 10 次 rank-max CUDA-event 样本，
  所有 complete cell 均通过 dense reference。
- 更正后的 Chitu CP1 为 53.472/951.568 ms。Fast AGKV CP2/4/8：Wan 1.3B
  为 26.444/13.332/7.276 ms（2.02x/4.01x/7.35x），Wan 14B 为
  474.149/237.980/119.371 ms（2.01x/4.00x/7.97x）。CP8 时，Wan 1.3B
  的 SGLang KV gather 为 7.217 ms，比 Fast AGKV 快 0.8%；Wan 14B 的
  Fast AGKV 为六条已实现路径中最快。H12 不能整除 CP8，Fast Ulysses 需要
  pad 到 H16，故 Wan 1.3B CP8 为 9.242 ms。
- 真实长度暴露出 Fast Ulysses symmetric heap 的旧估算只覆盖四份 tensor，
  Wan 14B CP2 会 OOM。串行路径实际同时保留重排后的 Q/K/V 和输出交换缓冲区；
  默认估算已按十份 tensor 加 64 MiB 余量并对齐到 64 MiB，相关大 shape 单测通过。
- 更正后的原始报告位于 `outputs/fast-cp/wan-real-shape-20260918/`，雷达图位于
  `outputs/fast-cp/figures/native-latency-real-shapes-20260918/`。通用 attention
  的 S=4096/8192/16384/32768 矩阵仍有效，但只表示共享 CP attention 算子，
  不再作为 Wan 模型负载。
- 目标：在 H20 上用三个仓库各自的原生实现比较通信、attention 和完整模型
  attention block；负载为 Wan 1.3B S32760、Wan 14B S75600、Qwen-Image 和
  Z-Image，统一使用 BF16、Qwen/Z 的 4096 image + 512 text tokens，以及
  CP2/4/8。
- Chitu 实现：Fast Ulysses 保持串行，按实际 shape 测量 SM/TMA/CE，接近时优先
  不占 SM 的 TMA；Fast AGKV 使用 NVSHMEM direct-slot K/V，并将 CE all-gather 与
  Q 或 text projection 重叠；两者使用通用 KV-split attention planner。Qwen/Z
  前端合并 QKV GEMM，并融合每头 RMSNorm 与 RoPE。
- 公平性修正：Qwen 的 image 和 text sequence 都按 CP 分片，与 SGLang fixture
  一致。Z-Image 保留原生 replicated-text 语义。所有单元从 attention 输入计时到
  output projection，不加载 checkpoint 权重，并逐项通过 dense reference。
- Qwen/Z 的早期 CP8 结果后来都由同模型的正式生产布局重跑覆盖；Wan 的早期
  S4096 数字已按本节开头的 shape 更正作废。下文只保留仍有效的 Qwen/Z
  传输对照和正式结果。
- Qwen CP2 强制路径：TMA 2.774 ms、CE 2.781 ms、SM 2.781 ms。最终 auto 在
  node040 选择两次 TMA，端到端 2.776 ms；输入搬运 TMA/CE/SM 为
  0.061/0.068/0.072 ms，输出为 0.065/0.074/0.069 ms。
- 三框架 CP8 原生结果中，真实 Wan 14B Fast AGKV 为 119.371 ms，SGLang
  Ulysses 为 120.537 ms，xDiT Ulysses 为 123.069 ms，三者都来自 node021。
  Qwen Fast Ulysses 1.149 ms、SGLang KV gather 1.047 ms；Z-Image xDiT
  Ulysses 1.554 ms。Qwen/Z 的 SGLang/xDiT 数字来自其它 H20 节点，只用于实现
  前沿比较，不报告精确跨节点 speedup。
- Z-Image 的上述 xDiT 点并非相同分片语义：xDiT 将 4096 image + 512 text
  统一按 CP 切分，CP8 每卡处理 576 query tokens；Chitu/SGLang 每卡处理 512
  image + 512 replicated text，共 1024 tokens。xDiT 的优势包含减少重复 text
  projection、query attention 和 output projection，不能归因于 NCCL all-to-all。
- Z-Image 后续将 context refiner 输出的 512 caption tokens 一次性按 CP 分片，
  主 transformer 内保持 `[local image, local caption]`，并让 AGKV/Ulysses
  同时覆盖两类 token。node021 同一作业的 CP2/4/8 Fast AGKV 为
  3.378/1.933/1.227 ms，Fast Ulysses 为 3.719/2.199/1.242 ms；CP8 相对各自
  NCCL 快 6.3%/25.7%，相对 CP1 7.016 ms 达到 5.72x/5.65x。13 个 cell 均通过
  dense reference，最大绝对误差为 2.44e-4。该修正后 Fast CP 已快于此前
  xDiT Z-Image CP8 的 1.554 ms；SGLang fixture 仍重复 caption，跨框架图中单独标注。
- 生产路径同步采用上述布局：Qwen-Image 在 `txt_in` 后、Z-Image 在 context
  refiner 后将 text/caption 一次性按 active lane 分片，后续 transformer blocks
  保持本地 image/text shard，不能整除 lane width 时自动回退 replicated text。
  Wan、Qwen-Image、Z-Image 在 active transport 为 `fast_agkv` 或
  `fast_ulysses` 时自动启用 KV-split planner，NCCL 仍使用标准 SDPA。
- 三个模型改为每个 transformer block 持有独立 CP processor，避免多层共用一个
  prepared-attention 槽而使后续层失去 CUDA Graph replay。Qwen/Z 的 packed QKV
  权重也按 attention module 缓存，并在参数 storage、版本、device 或 dtype 改变时
  失效，避免每层、每步重复拼接权重。
- 复现：`tools/benchmarks/fast_cp_frameworks/run_chitu_native_model_sweep.sh` 采集
  Chitu 原生矩阵；`tools/plotting/fast_cp_workload_radar.py` 使用 raw millisecond
  对数半径生成 CP2/4/8 雷达图。完整命令和结果表见同目录 `RESULTS.md`。
- 架构：当前扩展的所有 CUDA image 均包含 `sm_90` 和 `sm_120` cubin；H20
  实机探针通过。PRO 5000 本轮没有可用节点，SM120 只完成构建产物检查，运行时
  correctness 和性能尚未复测。
- 共享 CP attention 正式矩阵：固定 B1/H40/D128/BF16、dense noncausal，覆盖
  S=4096/8192/16384/32768、CP2/4/8，以及 Fast AGKV、Fast Ulysses、xDiT
  Ulysses、xDiT USP、SGLang AGKV 和 SGLang Ulysses，共 84 个 cell。全部在
  node035 采集，采用 5 次 warmup、75 次 rank-max CUDA event，均通过 dense
  reference（最大绝对误差 9.77e-4）。Fast 路径在 12 个 S/CP 组合中均快于外部
  最优路径；优势从 S32768/CP2 的 1.009x 到 S4096/CP8 的 1.454x。xDiT USP
  采用 U=2、R=CP/2；CP2 退化为 Ulysses，CP4/8 的 ring 阶段在本机 dense
  attention 中没有收益。完整表和复现说明见
  `outputs/fast-cp/cp-attention-native-20260918/RESULTS.md`。
- Qwen-Image 生产布局在 node021 重跑：CP1 为 5.308 ms；Fast AGKV
  CP2/4/8 为 2.675/1.544/1.021 ms，相对同作业 NCCL AGKV 快
  10.8%/14.0%/23.4%；Fast Ulysses 为 2.963/1.729/1.149 ms，相对 NCCL
  Ulysses 快 5.6%/3.7%/12.4%。Fast AGKV 使用 CE all-gather 与 Q/text
  projection overlap；Fast Ulysses 自动选择 TMA。全部 13 个 cell 的最大绝对
  误差为 2.44e-4，结果见
  `outputs/fast-cp/optimized-qwen-production-20260918/summary.md`。
- 上述正式重跑发现 packed QKV 缓存在 `torch.inference_mode()` 创建的参数上读取
  `_version` 会抛错。缓存签名现兼容无 version counter 的 inference tensor，同时
  保留普通参数原地更新的失效检测；新增回归测试覆盖 Qwen/Z 两条路径。
- Qwen-Image 跨框架同节点更正：Chitu 与 SGLang 全部在 node020、B1、4096 image
  + 512 text、H24/D128 上重跑。CP2/4/8 Fast AGKV 为
  2.685/1.553/1.021 ms，SGLang KV gather 为 2.625/1.499/1.045 ms；Fast AGKV
  在 CP2/4 慢 2.3%/3.6%，CP8 快 2.3%。Fast Ulysses 为
  2.980/1.735/1.257 ms，SGLang Ulysses 为 2.630/1.495/1.107 ms。此前雷达中
  SGLang Ulysses CP4 并未缺测，其 1.495 ms 与 KV gather 1.499 ms 几乎重合。
- 同形状 QKV-to-O 核心在 node021 单独测量后排名反转：CP2/4/8 Fast AGKV
  1.143/0.698/0.490 ms，SGLang KV gather 1.232/0.751/0.624 ms；Fast Ulysses
  1.186/0.692/0.430 ms，SGLang Ulysses 1.406/0.820/0.694 ms。CP4 attention
  Chitu 为 0.533/0.545 ms，SGLang 为 0.557/0.560 ms；AGKV 通信为
  0.171/0.210 ms。因此模型差距来自 Qwen 前后处理与 layout：SGLang 使用成对
  fused QK RMSNorm+RoPE，并在 Ulysses 输入侧直接把 text/image segments 变换为
  Ulysses layout；Chitu 当前分别执行四个 Q/K epilogue，并在通用 Ulysses 前物化
  三份 joint Q/K/V。下一步应优先复用 fused QK epilogue 和 segmented all-to-all。


### 2026-09-18：Qwen QKV epilogue/layout 融合与 AGKV 调度复测

- 问题：共享 CP attention 的优势被 Qwen 前后处理与固定 overlap 调度抵消。对照冻结
  本轮修改前源码，在同一独占 H20 node020 上进行初测与独立复测（245196/245203）。
  shape 为 B1、4096 image+512 text、H24/D128、BF16；边界是 attention block。
- 实现：每种模态用一个 Triton kernel 同时完成 Q/K RMSNorm、RoPE、V 拷贝和 joint
  Q/K/V 布局写入。通用 Qwen processor 直接传入 joint QKV，跳过重复连续化/拼接；
  prepared packed AGKV 复用该 kernel。AGKV auto 替换静态 CP2/4 overlap、CP8 packed
  规则，在准备阶段实测完整 block 并统一所有 rank 的选择，按 runtime 和配置缓存。
- 独立复测 CP2/4/8：Fast AGKV 从 2.687/1.546/1.024 降至 2.520/1.434/0.963 ms；
  Fast Ulysses 从 2.984/1.736/1.154 降至 2.756/1.612/1.032 ms。SGLang KV gather 为
  2.621/1.501/1.041 ms，Ulysses 为 2.630/1.496/1.110 ms。新 AGKV 比最快 SGLang
  路径低 3.8%/4.2%/7.5%，三种 CP 均选 packed；Fast Ulysses 的输入和输出都选 TMA。
- 机制与限制：packed 将六次 Q/K/V GEMM 合为两次，再进行一次 joint gather；本 shape
  下它比强制 overlap 更快。未把 GEMM、epilogue、同步的贡献分别量化。Ulysses CP2/4
  仍落后，下一候选是 packed/segmented QKV 输入交换，不能称已到优化上限。
- correctness：27 项 GPU 回归、13 项 dense CP 测试通过，另验证24组数值/Graph replay
  与跨 rank 调度缓存命中/丢失恢复。block 最大绝对误差2.44e-4。scratch oracle 的
  空文本 reshape 错误已修复并保留失败日志，未纳入性能结论。
- 决策：保留融合 kernel 和 prepared schedule auto；不保留 CP degree 静态调度规则。
  fused epilogue 已进入通用 processor；prepared AGKV 仍需显式安装，并非完整生成
  pipeline 自动启用。首次候选准备总耗时2.260–3.801 s，额外 symmetric buffer 随
  runtime 释放；这些初始化时间不计入稳态表。
- 证据：`outputs/fast-cp/qwen-epilogue-20260918/EXPERIMENT.md`、`confirm/`、
  `source-digests.json`；当前结果表和雷达图已替换为复测数据，旧数据留在研究记录中。

## 记录模板

### YYYY-MM-DD：候选名称

- 上游：仓库、commit、PR/issue、license
- 目标：模型、层、shape、dtype、GPU、CUDA、PyTorch
- 假设：为什么 LLM/通用 kernel 可能适用于 diffusion
- correctness：参考实现、误差、特殊语义
- microbenchmark：命令、延迟、显存、speedup
- 端到端：ChituBench 命令、延迟/吞吐、质量
- 决策：采用 / benchmark-only / 拒绝
- 后续：负责人或下一验证条件
