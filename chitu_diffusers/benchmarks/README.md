# EPAC Z-Image benchmark

该目录包含从 `experiments/chitu_api` 提升过来的 HTTP trace 回放工具。Poisson mixed
基线为 12 requests；分阶段 DP/CP/SLO trace 为 15 requests。两者均使用
512/1024/2048 和 12 denoise steps。

最终交付数据、表格和精选图统一维护在 `chitu_diffusers/README.md`，本目录只保留
可复现入口和 canonical trace，不再维护独立的过程性 `result.md`。

## Flux.1 Embedded 策略对照

下列命令在四卡上使用相同 closed-batch workload，只切换 lane policy：

```bash
for strategy in static_dp static_cp elastic; do
  CHITU_PYTHON_BIN=/path/to/ChituDiffusion/.venv/bin/python \
  CHITU_PROJECT_ROOT="$PWD" \
    bash script/srun_direct.sh 1 4 \
      chitu_diffusers/benchmarks/flux1_embedded_benchmark.py \
      --model-path /path/to/Flux-1 \
      --strategy "$strategy" \
      --output-dir "outputs/flux1-benchmark/$strategy"
done
```

每个 `metrics.json` 的 request makespan 不包含模型加载和 startup warmup，并记录逐请求
latency、queue delay、图片 pixel hash、实测 cost table 和结束时的 lane snapshot。
`elastic` 即 EPE 对照组。

Flux embedded benchmark 默认启用 active-lane parallel VAE，可用
`--no-parallel-vae` 做 leader-only 对照，或用 `--vae-parallel-halo` 调整 latent halo。

也可以直接回放 Z-Image canonical trace，并用 `--steps` 统一覆盖 trace 内原始步数：

```bash
CHITU_PYTHON_BIN=/path/to/ChituDiffusion/.venv/bin/python \
CHITU_PROJECT_ROOT="$PWD" \
  bash script/srun_direct.sh 1 4 \
    chitu_diffusers/benchmarks/flux1_embedded_benchmark.py \
    --model-path /path/to/Flux-1 \
    --strategy elastic \
    --trace chitu_diffusers/benchmarks/traces/phased_dp_cp_slo_n15_12step.json \
    --steps 50 --pulse-steps 5 --warmup-steps 3 \
    --output-dir outputs/flux1-benchmark/phased50/elastic
```

trace 模式保留原始 `arrival_ms`、尺寸、prompt、seed、priority 和 deadline；后者不会随
`--steps` 自动缩放。因此把 12-step trace 改跑 50 步时，原 deadline 只作为旧负载标记，
不能直接用于跨模型 SLO 结论。

推荐使用 warmup 自适应的统一三 arm 测试，避免跨模型复用固定到达率和 deadline：

```bash
CHITU_PYTHON_BIN=/path/to/ChituDiffusion/.venv/bin/python \
CHITU_PROJECT_ROOT="$PWD" \
  bash script/srun_direct.sh 1 4 \
    -m chitu_diffusers.benchmarks.run_flux1_3arm_adaptive \
    --model-path /path/to/Flux-1 \
    --output-dir outputs/flux1-benchmark/adaptive-3arm \
    --steps 50 --warmup-steps 3 --requests-per-phase 6
```

该入口先独立运行 measured calibration，再从 cp1/cp4 的 denoise 与 terminal warmup
推导 static-DP/static-CP 容量。生成的同一 trace 包含 sparse、capacity crossover 和
overload 三段到达率，然后依次执行 static-DP、串行 static-CP 与 EPE。所有请求使用同一
相对 SLO：`1.2 * max(cp1_step * steps + cp1_terminal)`。结果目录包含
`adaptive_trace.json`、`comparison.{json,md}`、三条 arm 的 metrics/rank timeline，
以及统一的 `strategy_timeline.{png,svg}`。

绘制这种覆盖步数后的 embedded 三策略 timeline 时，同样向绘图器传入覆盖值：

```bash
/path/to/ChituDiffusion/.venv/bin/python -m \
  chitu_diffusers.benchmarks.plot_strategy_timeline \
  --trace chitu_diffusers/benchmarks/traces/mixed_512_1024_2048_r3n12_12step.json \
  --steps 50 \
  --run "Static DP=outputs/flux1-benchmark/mixed50/static_dp" \
  --run "Static CP=outputs/flux1-benchmark/mixed50/static_cp" \
  --run "Elastic=outputs/flux1-benchmark/mixed50/elastic" \
  --output outputs/flux1-benchmark/mixed50/strategy_timeline.png
```

## Wan 自适应 trace

`wan_embedded_benchmark.py` 支持任意单节点 world size、矩形视频分辨率和 trace arrival。
视频完成后只保留数组 hash 与 shape，及时释放完整帧数组，避免长 trace 累积 host 内存。
传入 `--save-videos` 时则用独立的有界 CPU 线程池异步编码 MP4；
`--video-workers` 控制并发编码数。混合时长 trace 会按 `(height, width, frames)` 分别
执行 startup warmup，避免用短视频 cost 预测长视频请求。
推荐先用 2 卡 `static_cp` 单请求得到 CFP2 的实测端到端 latency，再用 8 卡
`--warmup-only` 得到 width 1/2/4/8 cost table：

```bash
bash script/srun_direct.sh 1 2 -m chitu_diffusers.benchmarks.wan_embedded_benchmark \
  --model-path /path/to/Wan2.1-T2V-1.3B --strategy static_cp \
  --output-dir outputs/wan-adaptive/2gpu-cfp \
  --width 1280 --height 720 --frames 81 --steps 50 --warmup-steps 3

bash script/srun_direct.sh 1 8 -m chitu_diffusers.benchmarks.wan_embedded_benchmark \
  --model-path /path/to/Wan2.1-T2V-1.3B --strategy elastic \
  --output-dir outputs/wan-adaptive/calibration --warmup-only \
  --width 1280 --height 720 --frames 81 --steps 50 --warmup-steps 3
```

将 2 卡 `metrics.json` 的单请求 `latency_ms` 原值传给 trace builder。生成器以 8 卡
warmup 的 width=2 成本估计四条 CFP2 lane 的总容量，创建 sparse、balanced 和 overload
三段到达率，并给所有请求写入同一个显式 SLO：

```bash
python -m chitu_diffusers.benchmarks.wan_adaptive_trace \
  --warmup outputs/wan-adaptive/calibration/epe_warmup.json \
  --slo-ms MEASURED_CFP2_LATENCY_MS \
  --output outputs/wan-adaptive/adaptive_trace.json \
  --width 1280 --height 720 --frames 81 --steps 50 --world-size 8

bash script/srun_direct.sh 1 8 -m chitu_diffusers.benchmarks.wan_embedded_benchmark \
  --model-path /path/to/Wan2.1-T2V-1.3B --strategy elastic \
  --trace outputs/wan-adaptive/adaptive_trace.json \
  --output-dir outputs/wan-adaptive/elastic \
  --width 1280 --height 720 --frames 81 --steps 50 --warmup-steps 3

python -m chitu_diffusers.benchmarks.plot_strategy_timeline \
  --trace outputs/wan-adaptive/adaptive_trace.json \
  --run 'Elastic=outputs/wan-adaptive/elastic' --world-size 8 \
  --output outputs/wan-adaptive/timeline.png
```

`warmup-steps` 低于 3 会被拒绝。Wan terminal 默认使用 active-lane parallel VAE 和
leader D2H；`--no-parallel-vae` 可运行 leader-only 对照。cost table、SLO 和 timeline
都包含实测终结成本，不把 denoise-only 外推冒充端到端延迟。

## 策略 / Attention 多到达率对照

下列脚本在同一 Slurm allocation 和节点内，顺序执行 Elastic AGKV、Static DP AGKV、
Static CP AGKV、Elastic USP u2r2 与三个 offered arrival rate。每个点重新启动服务并执行
measured warmup；结束后自动生成完整策略折线图和 0.12 req/s 的 Elastic AGKV timeline。

```bash
mkdir -p outputs/epac-attention
export ZIMAGE_MODEL_PATH=/path/to/Z-Image
export EPAC_ARRIVAL_RATES="0.06 0.12 0.24"
sbatch chitu_diffusers/benchmarks/run_attention_comparison_slurm.sh
```

可通过 `EPAC_VARIANTS` 选择子集，例如只补跑 AGKV baselines：

```bash
EPAC_VARIANTS="static_dp static_cp" \
  sbatch chitu_diffusers/benchmarks/run_attention_comparison_slurm.sh
```

可用 `EPAC_TIMELINE_RATE` 选择典型 timeline，到达率必须包含在
`EPAC_ARRIVAL_RATES` 中。结果位于
`outputs/epac-attention/job-JOBID/{agkv,static_dp,static_cp,usp}/rate-*`；顶层生成
`attention_rate.{png,svg}` 与只展示 Elastic AGKV 的
`attention_timeline.{png,svg}`。折线图仍包含本次运行的全部策略。

单次压测也可设置 `EPAC_ATTENTION_MODE=agkv|usp`、`EPAC_ULYSSES_DEGREE=2` 和
`EPAC_ARRIVAL_RATE` 后运行 `run_benchmark_slurm.sh`。到达率缩放只修改 trace 的
`arrival_ms`，request shape、seed、prompt 和 denoise steps 保持不变。
`EPAC_CFG_PARALLEL=0` 可关闭默认的 CFP2 优先布局，用于纯 CP A/B。

## 只启动服务

```bash
mkdir -p outputs/epac-serve
export ZIMAGE_MODEL_PATH=/path/to/Z-Image
sbatch chitu_diffusers/benchmarks/serve_slurm.sh
```

作业日志中的 `CHITU_API_READY` JSON 包含 compute node endpoint。服务通过 `scancel JOBID`
停止。

## 启动并压测

该脚本在一个 Slurm allocation 内启动 4-rank EPAC 服务，等待 startup warmup 完成，回放
固定 trace，然后停止服务：

```bash
mkdir -p outputs/epac-benchmark
export ZIMAGE_MODEL_PATH=/path/to/Z-Image
sbatch chitu_diffusers/benchmarks/run_benchmark_slurm.sh
```

结果保存到 `outputs/epac-benchmark/job-JOBID/`：

- `service.log`：torchrun、warmup、EPE phase 和 online calibration 日志。
- `epe_warmup.json`：每个 resolution/lane width 的启动实测 cost table。
- `metrics.json`：makespan、throughput、latency、queue delay，以及逐请求图像路径、字节数、
  SHA-256 和实际分辨率。
- `images/reqXXXX.png`：每个完成请求的生成图像。客户端会重新解码 PNG 并核对 trace 中的
  目标分辨率后才写入 metrics。
- `timeline-rankN.jsonl`：rank 独立记录的绝对时间戳；只在服务退出时一次性落盘。

可通过环境变量覆盖 `EPAC_VENV`、`EPAC_PORT`、`EPAC_TRACE`、`EPAC_RUN_DIR`、
`EPAC_GPUS_PER_NODE`、`EPAC_ROOT`、`EPAC_PARALLEL_VAE=0|1`、
`EPAC_VAE_PARALLEL_HALO` 和 `ZIMAGE_MODEL_PATH`。脚本默认以
`SLURM_SUBMIT_DIR` 作为仓库根目录，因此应从仓库根目录执行 `sbatch`。
设置 `EPAC_DEFAULT_DEADLINE_MS` 可为 trace 中未显式填写 `deadline_ms` 的请求配置统一
端到端 SLO；显式 deadline 始终优先。

只验证服务启动、两步 denoise 和 PNG 返回时，可使用最小 smoke trace：

```bash
EPAC_TRACE="$PWD/chitu_diffusers/benchmarks/traces/smoke_512_n1_2step.json" \
  sbatch chitu_diffusers/benchmarks/run_benchmark_slurm.sh
```

服务默认使用当前动态 lane 并行执行 VAE、由 leader 完成 D2H，再用 4 个有界 CPU
worker 异步执行 Diffusers image postprocess 和 PNG encode。直接启动 example 时可通过
`--postprocess-workers N` 调整并发数。请求在异步阶段返回 `postprocessing` 状态，PNG
发布后才变为 `completed`。

`EPAC_SCHEDULE_STRATEGY` 默认为 `elastic`。串行 full-CP baseline 使用：

```bash
EPAC_SCHEDULE_STRATEGY=static_cp \
  sbatch chitu_diffusers/benchmarks/run_benchmark_slurm.sh
```

固定单卡 lane、最多四请求并发的 static-DP baseline 使用：

```bash
EPAC_SCHEDULE_STRATEGY=static_dp \
  sbatch chitu_diffusers/benchmarks/run_benchmark_slurm.sh
```

Static-DP 使用 EPAC singleton-lane producer-consumer transport。每个 rank 独立完成完整
请求，完成后立即向 rank 0 拉取下一个 SLO 排序请求；请求之间没有 world pulse、broadcast
或 all-gather。rank 0 同时承担控制面和 GPU 0 lane，不会浪费一张卡。

## 分阶段 DP / CP / SLO 对照

`phased_dp_cp_slo_n15_12step.json` 根据启动 warmup cost 构造四段到达：开头 8 个
1024 请求形成 dense DP burst；35s、65s 各到达一个稀疏 2048 请求；90s 同时进入一个
22.5s SLO 的 2048 请求和三个普通 1024 请求；130s 追加一个稀疏 512 tail。

下列脚本在同一个 Slurm allocation、同一节点上顺序执行三种策略并自动绘图：

```bash
mkdir -p outputs/epac-phased
export ZIMAGE_MODEL_PATH=/path/to/Z-Image
sbatch chitu_diffusers/benchmarks/run_strategy_comparison_slurm.sh
```

结果位于 `outputs/epac-phased/job-JOBID/{static_dp,static_cp,elastic}/`，共享时间轴为
`outputs/epac-phased/job-JOBID/strategy_timeline.{png,svg}`。timeline 会标出绝对 deadline，
策略标题同时显示 SLO 命中数。VAE、D2H 和 state transfer 已由 startup warmup 实测并
纳入 phase/完成时间预测；服务仍默认用 3000ms `deadline_guard_ms` 为控制面和 CPU
结果发布等未稳定建模的部分预留时间。直接启动时可用
`--deadline-guard-ms` 调整。

多种策略完成后可按请求对齐生成图像总览：

```bash
../ChituDiffusion/.venv/bin/python -m \
  chitu_diffusers.benchmarks.make_image_contact_sheet \
  --run static_dp=outputs/epac-image-validation/job-194935-static_dp/images \
  --run static_cp=outputs/epac-image-validation/job-194935-static_cp/images \
  --run elastic=outputs/epac-image-validation/job-194935-elastic/images \
  --output outputs/epac-image-validation/job-194935-contact_sheet.png
```

## 三策略 timeline

最新的 parallel-VAE / terminal-aware 回归使用 4 x H20 Slurm job `196740`，原始结果在
`outputs/epac-phased/parallel-vae-final/`，顶层 README 保存对应数据表和精选 timeline。

压测脚本默认传入 `--record-timeline`。每个 rank 在内存中记录 prepare、denoise lease、
pulse/control wait、state transfer、VAE、postprocess 和 PNG 的绝对时间戳，服务退出时写入
独立 JSONL。热路径不写文件，也不为 tracing 增加分布式同步。

`plot_strategy_timeline.py` 优先合并这些 rank trace，统一绘制请求到达、static-DP、
static-CP 和 elastic 的真实 4-GPU timeline。缺少 JSONL 的旧压测目录仍回退到 service log
重建。共享 request panel 只标出 Elastic 的首次调度与完成时间；显式 `deadline_ms` 优先，
否则默认 DDL 为多个 run 的 cp1 warmup 单步耗时中位数乘以 denoise steps 和 `1.2`。
可通过 `--default-deadline-factor` 调整该系数。下例使用三次同 trace 压测结果：

```bash
../ChituDiffusion/.venv/bin/python \
  chitu_diffusers/benchmarks/plot_strategy_timeline.py \
  --trace chitu_diffusers/benchmarks/traces/mixed_512_1024_2048_r3n12_12step.json \
  --static-dp-log outputs/epac-async-postprocess/job-194897-static_dp/service.log \
  --static-dp-metrics outputs/epac-async-postprocess/job-194897-static_dp/metrics.json \
  --static-cp-log outputs/epac-async-postprocess/job-194897-static_cp/service.log \
  --static-cp-metrics outputs/epac-async-postprocess/job-194897-static_cp/metrics.json \
  --elastic-log outputs/epac-async-postprocess/job-194897-elastic/service.log \
  --elastic-metrics outputs/epac-async-postprocess/job-194897-elastic/metrics.json \
  --output chitu_diffusers/benchmarks/figures/strategy_timeline.png
```

脚本同时生成 SVG 和摘要 JSON。请求到达和三个 GPU 面板严格使用同一时间尺度，并使用
同一套请求颜色。rank trace 存在时，彩色请求块表示实测 DiT denoise，prepare、VAE、
D2H、控制等待、状态搬运和 idle 分别显示；异步 postprocess/PNG 显示在独立 CPU encode
行，不再占用 GPU lane，也不会把 elastic finalize 隐藏在基于 cost model 的重建时间轴
之外。
