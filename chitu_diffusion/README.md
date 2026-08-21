# chitu_diffusion

`chitu_diffusion` 是 ChituDiffusion 的 Diffusers-native 执行后端。本轮维护范围是
高性能 context-parallel EPE DiT，以及共享 backend 上的 `generate`/`serve` 生命周期。
tokenizer、encoder、scheduler、VAE 和 image processor 保持上游 Diffusers 生命周期；
只有 DiT denoise step 进入 EPE 的动态 lane 调度。

`generate()` 是测试和离线使用的单请求入口，同一 backend 会自然退化为 full-world static
CP；`serve()` 是生产入口，在同一 backend 上增加 warmup、队列和 EPE pulse。两条路径共享
request、模型 state、denoise 和 finalize 实现。多卡 `generate()` 要求所有 rank 进入，
仅 full-world leader 返回 Diffusers output，followers 返回 `None`。

## 当前结构

```text
Diffusers pipeline
        |
DiffusersBackend                         model-specific
  prepare_request
  denoise_step
  finalize_gpu / postprocess
        |
EpeDiffusionServiceRuntime               model-independent
  queue / admission / cancel / completion
        |
EpeSchedulingPolicy <--- measured cost   pulse boundary
  request + lane ranks + K steps
```

源码按所有权分为四部分：`epac/` 是模型无关核心，`models/` 只保存 Z-Image、Flux.1 等
模型差异，`parallel/` 是模型无关的并行 group 与通信实现，`serve/` 是 HTTP 与 torchrun 生命周期。
详细协议见 `epac/README.md`。
新增 image decoder 的实现清单和验收条件见
[`INTEGRATING_IMAGE_DECODER.md`](INTEGRATING_IMAGE_DECODER.md)。
Flux.1 的 Diffusers-native 适配范围、代码映射和 bring-up 顺序见
[`models/flux1/README.md`](models/flux1/README.md)。

这两个边界有意分开：

- EPE 只根据 request profile 和 warmup cost table 生成 `StepPlan(request, lane, K)`，
  不接触 tensor、HTTP 或 Omni payload。
- FlexCache 是 `generate()` 的 request-local 优化；`serve()`/EPE 对非 `none`
  策略显式 fail fast。策略决策与并行布局正交：同一 lane 内各 rank 跳步一致。
- model backend 只保存某个 Diffusers 模型家族无法统一的 tensor glue，不负责选择 lane。

## 已实现

| 模块 | 当前能力 |
| --- | --- |
| `epac/model_executor.py` | generate/serve 共享的 request/state、decode/postprocess 生命周期 |
| `epac/pulse.py` | lane lease、deadline、report rendezvous 和 producer-consumer pull 协议 |
| `epac/scheduling.py` | 通用 `StepPlan`、`RequestProfile` 和三策略 lane constraints |
| `epac/epe.py` | measured-cost SLO layout、动态 lane、balanced-K pulse |
| `epac/model_scheduling.py` | 所有模型共享的 cost/calibration/planner facade |
| `epac/api.py` | typed request helper 与同步 full-world Diffusers facade |
| `epac/cost.py` | 启动实测的 DiT step、VAE/D2H terminal 和 state-transfer cost table |
| `flexcache/` | model/block/attention contract、request-local session、MagCache/MeanCache/TeaCache/TaylorSeer/PAB |
| `parallel/` | 动态 lane group、并列的 AGKV/USP attention 后端和 tile-parallel VAE |
| `models/zimage/` | Z-Image pipeline、transformer、backend 和 API 门面 |
| `models/flux1/` | FLUX.1-dev split-step pipeline、动态 AGKV/USP CP、embedded executor 和 API 门面 |
| `models/qwen_image/` | Qwen-Image split-step pipeline、CFP2、动态 AGKV/USP CP 和 API 门面 |
| `models/wan/` | 官方 Diffusers Wan T2V、CFP2、动态 AGKV/USP CP 和并行 VAE |
| `models/flux2_klein/` | 官方 Diffusers FLUX.2-klein 与固定 full-world AGKV/USP CP baseline |
| `serve/` | HTTP connector、原生配置、通用 EPE runtime 和 torchrun 生命周期 |

公共 EPE cost model 兼容 warmup 报告中的 `image_tokens`、`width`、`batch_size`、
`cfg_conditions`、`terminal_ms` 和 `state_bytes`。`terminal_ms` 是当前 lane 的 VAE
critical path 加 leader D2H；`transfer_rows` 则按 state bytes 和 GPU rank pair 实测。

## Pipeline API

安装后可使用统一入口：

```bash
chitu generate --model zimage --model-path /path/to/Z-Image \
  --output outputs/zimage.png
chitu generate --model wan --model-path /path/to/Wan2.1-T2V-1.3B \
  --steps 50 --cache-strategy magcache --output outputs/wan.mp4
chitu serve --stage-config /path/to/stage.yaml
```

新入口不读取旧 `chitu run` 的 OmegaConf 配置。替换前的 runtime、ChituBench、
stage/phased runtime 和 DiTango 仅作为 backup 保留，不再承接新功能。

### FlexCache

FlexCache 是 `generate()` 的 request-local 优化；`serve()`/EPE 对非 `none` 策略
显式 fail fast。公共参数是 `--cache-warmup-steps`、`--cache-cooldown-steps`，
策略参数使用各自前缀。

| 策略 | 复用粒度 | 状态 |
| --- | --- | --- |
| `magcache` | 整个 DiT backbone 的 image-token residual | 官方 Flux.1/Qwen-Image/Wan 1.3B profile；其他模型 fail fast |
| `teacache` | 整个 backbone 的 block-group residual | 仅 Wan T2V 1.3B/14B 与 Flux.1 有标定系数 |
| `taylorseer` | 每个 block 的门控前子模块输出 | 四个模型可用 |
| `pab` | attention 输出，按类型定周期 | 四个模型可用，周期规则为简化版 |
| `meancache` | CFG 后 velocity 与 solver step 前后 latents | 仅 Qwen-Image/Z-Image，固定 50 steps，使用各模型官方 fresh/JVP 表 |

单卡实测（20 steps，H20，默认参数）：

| 配置 | 耗时 | 跳步率 | cache 显存 |
| --- | ---: | ---: | ---: |
| Flux.1 512² baseline | 5.10s | — | — |
| Flux.1 `teacache --teacache-threshold 0.4` | 2.95s | 45% | 18 MB |
| Wan 1.3B 480×832×17 baseline | 17.07s | — | — |
| Wan `teacache --teacache-threshold 0.2` | 11.67s | 45% | 46 MB |
| Wan `taylorseer` | 10.11s | 45% | 8227 MB |

TaylorSeer 的显存代价比 TeaCache 高两个数量级：它为每个 block 的每个子模块各保存
`max_order + 1` 份张量，而 TeaCache 只保存每个 block group 一份 residual。视频模型上
这个差别是 GB 级的，选策略时要一并考虑。

策略状态按 CFG 分支隔离。denoise 步边界由 timestep 签名识别，而不依赖 pipeline
配合，因此 Wan/Qwen 串行执行 cond/uncond 两次 forward 时两条分支各自走完整日程，
不会互相返回对方的缓存。

所有判定信号都取自 transformer 入口的全序列输入，在 CP/CFP 下各 rank 复制，
因此同一个 lane 内所有 rank 的跳步决策一致。代价是探针的 embedding 投影会重算一次。

MagCache 默认直接选取 checkpoint 对应的官方 magnitude-ratio profile 和
threshold/K/retention 参数。Flux.1 为 E024K5R01，Qwen-Image 为 E006K2R02，
Wan 1.3B 为 E012K4R02。Wan 官方 cond/uncond ratio 虽略有差异，但默认参数产生的
50-step reuse schedule 完全相同；FlexCache 显式取所有 branch schedule 的交集，
保证 serial CFG、CFP 和 CP 的控制流一致。命中时只缓存 image-token residual，
并在第一个 block 注入，后续 block 全部在 hook 入口跳过，不修改 model/pipeline 源码。

官方尺寸、H20 单卡对照：Flux.1 1024² 为 24.83s → 8.31s（2.99x，官方报告
约 2.8x）；Qwen-Image 1664×928 为 100.45s → 73.27s（1.37x，官方报告约
1.5x）；Wan 1.3B 480×832×81 为 263.69s → 116.49s（2.26x，官方报告
189s → 68s，即 2.78x）。三者的 reuse schedule 与官方逐 step 一致；绝对耗时差异
来自 GPU、Diffusers/attention backend 和端到端固定开销。按 fresh-step 数计算的
理论上限分别是 3.11x/1.35x/2.38x，实测达到约 96%/101%/95%；官方 Qwen/Wan
README 的近似 speedup 已高于其公开 schedule 自身的理论上限，不能作为同 backend
严格配对计时目标。完整命令、质量指标和并行 smoke 记录见
[`docs/results/flexcache/magcache_compare_20260805.md`](../docs/results/flexcache/magcache_compare_20260805.md)。

TeaCache 的 `threshold` 只有配合官方标定的 rescale 多项式才有意义，而多项式与探针
是一对：Flux.1 测第一个 block 的 `norm1` 调制输出，Wan 测未投影的时间嵌入
（对应 reference 的 no-retention 配置）。检测到的 checkpoint 没有已发布系数时会直接
报错，而不是退化成不做缓存。要在未标定的模型上试跑，显式传
`--teacache-coefficients 1 0` 得到恒等 rescale，并自行重新调 threshold。

TaylorSeer 的 `--taylorseer-fresh-threshold` 默认 3。该配置在 Wan 1.3B、50 steps
实测达到约 1.9–2.0x DiT 加速，且没有 reference 默认 5/6 所产生的严重条纹伪影。
这是速度与质量的折中点，不代表跨模型、跨步数通用；改变模型或步数后仍需重新测图。
Flux.1 会在 block hook 入口直接组合本步 modulation/gate 和 Taylor 预测值，命中时
跳过 attention、MLP 和 projection。1024²/50 steps 实测默认 threshold 3 为 13.71s
（相对 33.21s baseline 为 2.42x）。在相同 vanilla Diffusers FP16 backend 上，
FlexCache f3 为 13.50s，原版 TaylorSeer f3 为 14.03s，生成图片逐像素一致。
因此 block fast path 与 reference 语义一致；EPE/AGKV backend 的质量轨迹仍应相对
它自己的 baseline 独立评估。

MeanCache 通过 request-local denoise/scheduler hook 获取 CFG 后 velocity、真实 sigma
区间以及 solver step 前后的 latents，不修改模型或 pipeline 源码。它只接受官方标定的
50-step 日程；Qwen-Image 支持 fresh budget 25/17/10，Z-Image 支持
25/20/17/15/13，其他步数或 budget 会直接报错。缓存位于 CP all-gather 和 CFG combine
之后，fresh/reuse 决策仅由 step index 决定，因此 CFP/CP 各 rank 的控制流一致。

512²、50 steps、B25、H20 单卡对照：

- Z-Image：原版 13.92s → 6.20s（2.24x），FlexCache 13.97s → 6.17s
  （2.26x）；原版与 FlexCache 的 baseline/B25 图片分别逐像素一致。
- Qwen-Image：原版 19.21s → 8.09s（2.37x），当前 EPAC backend
  16.46s → 7.94s（2.07x）。相对各自 baseline，原版与 FlexCache B25 的
  PSNR/SSIM/LPIPS 分别为 31.15/0.97449/0.03398 和
  31.15/0.97447/0.03400；差异来自 backend 数值轨迹，不是 MeanCache 日程或公式。

两模型均已通过单卡与 static CP2；Qwen-Image 另通过 CFP2。运行参数为
`--cache-strategy meancache --meancache-fresh-steps 25`。B25/B17/B13
（Z-Image）与 B25/B17/B10（Qwen-Image）的完整 speed-quality sweep 见
[`docs/results/flexcache/meancache_compare_20260804.md`](../docs/results/flexcache/meancache_compare_20260804.md)。

```python
import torch

from chitu_diffusion import (
    EPACPipeline,
    EPACRequest,
    EPACServeConfig,
)

pipeline = EPACPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    attention_mode="agkv",
)

# Offline: fixed full-world lane, no warmup or elastic planner.
result = pipeline.generate(
    EPACRequest(
        prompt="a red cube",
        width=1024,
        height=1024,
        num_steps=50,
        seed=7,
    )
)
image = result.images[0]

# Online: startup warmup followed by elastic pulse scheduling.
pipeline.serve(
    EPACServeConfig(
        warmup_resolutions=((512, 512), (1024, 1024), (2048, 2048)),
        warmup_steps=5,
        pulse_steps=5,
        default_deadline_ms=30_000,  # request.deadline_ms can override this SLO
        schedule_strategy="elastic",  # or "static_cp" / "static_dp" baseline
        cfg_parallel=True,
        parallel_vae=True,
        vae_parallel_halo=8,
    )
)
```

`serve()` 是 blocking 调用：所有 torchrun ranks 都进入该方法，rank 0 提供 HTTP，
其他 ranks 运行 follower loop。完整命令见 `chitu_diffusion/examples/README.md`。
所有模型默认使用 `attention_mode="agkv"`。安装 `usp` 可选依赖后，可切换为
`attention_mode="usp", ulysses_degree=2`；cp4/cp2 lane 分别映射为 u2r2/u2r1。

Z-Image 默认优先使用 CFP2：开启 CFG 时，偶数宽度 lane 先拆成 cond/uncond
两条分支，每条分支再使用 `width/2` 路 CP（例如四卡为 CFP2 x CP2）。设置
`cfg_parallel=False` 或 CLI `--no-cfg-parallel` 可回到整 lane 串行执行两条 CFG
分支的纯 CP baseline。warmup 会测量实际采用的 CFP/CP 组合。

Z-Image VAE 包含全局 mid-block attention/normalization，因此空间 tile decode 是质量近似，
不是逐像素等价。4x H20、512、halo=8 的同 seed/request 端到端对照为 34.91 dB
PSNR，最大像素差 10/255，未观察到 tile seam；对要求 bitwise/native decode 的场景可设置
`parallel_vae=False` 或 CLI `--no-parallel-vae`。启动 warmup 会分别测量每个 lane width，
因此 scheduler 使用的是所选模式的真实 terminal cost。

## SLO-aware Elastic Scheduling

`default_deadline_ms` 是从请求到达开始计算的默认端到端 SLO；请求携带的
`deadline_ms` 优先于默认值。两者都不设置时，elastic 退化为 throughput/fairness/
flow-time 调度，而不是伪造 deadline。`deadline_guard_ms`（默认 3000ms）仍用于覆盖
控制面和 CPU 结果发布等未稳定建模的尾部；VAE、D2H 和 state transfer 已进入实测模型。

每个 pulse 都会用启动 warmup 得到的 sequence-length/lane-width cost table 前向预测完整
pending + running 队列。布局评分按 SLO miss 数、最大与总 tardiness、starvation、并发
admission、等价单卡工作吞吐、slowdown 和 flow time 依次排序。等价单卡工作吞吐使用
`Σ(T_cp1 / T_lane)`，避免 raw steps/s 将全部 GPU 分给扩展效率较低的短序列。
`max_inflight_requests` 只限制已创建 request state 的数量；等待请求即使尚未 admission，
也会参与预测，避免队首以外的紧急请求对 planner 不可见。

balanced-K 以完整 phase 而非纯 denoise 时间确定 pulse。若一个 lane 将在 nominal pulse
前完成 denoise，它的预测 VAE/D2H 被视为不可抢占 terminal section；其他独立 lane 会把
这段时间换算成额外的一到数个 denoise step，随后一起进入 pulse。宽 lane 的 VAE 使用
当前动态 group 做空间 tile + halo decode 和 lane-local gather，不创建新通信组，也不做
lane shrink；只有 leader 保留完整 image tensor 并执行 D2H。

## Embedded Runtime API

宿主已经创建 stage processes 时，不需要启动 HTTP，也不需要让 EPAC 重新 spawn worker：

```python
from chitu_diffusion import (
    EmbeddedDiffusionRuntime,
    EmbeddedRuntimeConfig,
    HotSwitchPoolConfig,
    StageWorldSpec,
    ZImageExecutorFactory,
)

world = StageWorldSpec.from_torchrun("image_decode")
pool = HotSwitchPoolConfig(
    allowed_lane_widths=(1, 2, 4),
    default_deadline_ms=30_000,
)
runtime = EmbeddedDiffusionRuntime.create(
    world=world,
    pool=pool,
    executor_factory=ZImageExecutorFactory(model_path=model_path),
    config=EmbeddedRuntimeConfig(output_root="outputs/image_decode"),
)
runtime.start()
```

所有 rank 创建并启动实例；只有 `world.leader_rank` 调用 `submit/cancel/poll`。
`poll()` exactly-once 返回 raw PIL completion，HTTP PNG 或 Omni event 由外层 connector
生成。`stop(graceful=True)` 先停止 ingress、等待 leader queue/inflight drain，再停止
worker 并按 process-group ownership 关闭 executor。

## 本地验证

该目录尚未作为独立 wheel 安装，直接从仓库运行时需要显式保留仓库根目录：

```bash
PYTHONPATH="$PWD" .venv/bin/pytest -q \
  test/test_chitu_diffusion_*.py
```

GPU 端到端验收不能由 CPU 单测替代；本轮替换完成后重新建立 benchmark 和结果基线。

## API 约束

- 对外使用 `EPACRequest`；内部才投影为通用 `DiffusionRequest`。
- 每个 backend 必须创建 request-local scheduler、latent、timesteps 和 step cursor。
- `StepPlan.lane_ranks` 由 EPE 给出；backend 不缓存固定 CP world size。
- `from_pretrained()` 根据 torchrun 环境初始化 world 和 canonical lane groups。
- capabilities 用于显式声明支持范围，未支持能力不得静默退回整段 `pipeline.__call__()`。

## 尚未完成

- 把 warmup tensor 生成与多 rank latency 聚合提升为公共 profiler。
- 扩展四模型 FlexCache 的多 prompt GPU 质量和性能基线；CPU contract、请求隔离和
  MeanCache 单卡/CP/CFP 验证已接入。
- 验收 Flux.1 的 USP、true CFG；当前已验收 AGKV、动态 cp1/cp2/cp4、
  parallel VAE、offline/embedded API 和三策略 benchmark。
- distributed worker 当前只取消 pending request；running request 的 step-boundary cancel
  和任意 rank fatal-state broadcast 尚未接入 control transport。
- `StageWorldSpec` 支持自建或复用 stage WORLD；任意 torch.distributed subgroup 注入尚未
  支持，当前会显式 fail fast。

因此当前交付边界是 **EPAC developer preview + 可运行的 Z-Image 服务适配和 Flux.1
Diffusers-native offline/embedded 适配**。它足以让另一个模型团队按 executor contract
接入并运行 Slurm/torchrun 验证，但尚不应标记为具备跨 rank 容灾保证的 production GA。
LLaDA2 的 VQ/SigVQ、模型 state 和 decode 细节不在本仓库中，本次不做猜测性实现。

现有可运行的 4-GPU 服务、Z-Image 模型实现和压测入口均位于 `chitu_diffusion`。
