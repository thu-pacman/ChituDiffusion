# Hot Switch 内部使用说明

> 本文面向内部同事使用和继续实验。Hot switch 目前仍是 serving 路径上的内部优化，不是公开宣传特性，也不建议在没有复现和校准的环境里直接当作稳定生产能力使用。

## 一句话概括

当前实现把一组 GPU 看成一个可重新分区的 diffusion serving pool：请求少、GPU 空闲时，可以让单个请求占用更多 GPU 做 SP/CP 以降低单请求延迟；请求突发或排队变深时，在 denoise step 边界把运行中的请求降到更小 width，腾出 GPU 运行更多独立请求，即 request-level DP。调度器的目标顺序是 SLO 优先，然后是公平性、切换次数、尾延迟和吞吐。

```text
idle / low load       : one request may run as SP4 or SP8
burst / deep queue    : multiple requests run as DP lanes, usually width=1
mixed workload        : large requests may use SP, short/low-res requests stay DP
switch point          : denoise step boundary only
```

这里的 `SP` 和 `CP` 在当前代码里基本指同一类 denoise sequence/context parallel execution width；`width=1` 是单 GPU lane，也就是 request-level DP replica；`width>1` 是一个请求跨多个 GPU 的 SP/CP lane。

## 当前进展

### 已经落到 runtime 的部分

- `slo_elastic` pool scheduler 已接入 runtime。rank 0 在每个 phase 边界规划 lane layout，并把计划广播给所有 rank。
- `pure_dp` 和 `pure_sp` 也走同一个 pool engine，用于做 apples-to-apples baseline。
- dynamic SP groups 会在初始化时预热可行的 lane width，避免第一次切换时现场创建 NCCL group。
- pool engine 支持每轮多个 lane 同时执行，每个 lane 可以有不同 width。
- 切换只发生在 denoise step boundary；运行中 lane 不会在一步内部改变 GPU 归属。
- 已有 targeted latent migration / step sync / targeted replication，避免每一轮都无脑把 latent 广播到全 world。
- 已有 multi-step phase 和 balanced-K：一个 phase 可以跑多步，用来摊薄 plan/broadcast/barrier/retire 开销，并让不同 lane 的预计 phase 时间更接近。
- 已接入 runtime online calibration：用实测 lane step latency 修正离线 cost model。
- serving trace replayer 已能按 wall-clock arrival 注入请求，并输出 per-request latency / queue delay / throughput 等指标。

### 仍然要谨慎看待的部分

- runtime 当前仍是同步 phase 模型。phase 边界有 plan、broadcast、barrier、migration、retire 等成本；这和离线 simulator 的理想事件模型不同。
- `slo_elastic` 的策略核心已经比 M7 heuristic 更系统，但真机性能仍依赖 cost model、interconnect、phase 参数和 workload。
- FlexCache emergency action 目前主要是 simulator 里的敏感性研究。runtime 默认关闭，不建议把它当作已经验证的 hot-switch 动作。
- 当前最完整的真机验证路径是 Z-Image。Qwen-Image adapter 具备相关接口和 lane-aware 处理，但还没有整理成和 Z-Image 同级的 canonical pool benchmark。

## 适配模型状态

| 模型 | 当前状态 | 说明 |
| --- | --- | --- |
| `Z-Image` | 主要适配和验证对象 | 已有 `z_image_serve_pool.yaml`、serve trace replayer、dynamic SP lane、rank0 text encode、pool denoise、目标 latent 迁移、baseline 对比和离线 cost model。 |
| `Qwen-Image` | 代码路径已准备，验证较少 | adapter 已处理 dynamic lane topology、pool engine 下的 CP sharding 约束、CFG 能力和基础 smoke/DP 配置；还需要补齐 pool config、cost model 标定和端到端 serving benchmark。 |
| 其他 diffusion adapter | 暂不作为 hot switch 已适配对象 | base adapter 提供 `denoise_step_group_once()` 的通用接口，但实际能否稳定跑 pool engine 取决于模型的 CP/CFG/latent 生命周期是否满足 lane 切换约束。 |

## 关键代码位置

| 路径 | 作用 |
| --- | --- |
| `chitu_diffusion/runtime/scheduler.py` | `slo_elastic` / `pure_dp` / `pure_sp` 的 runtime planning 入口。 |
| `chitu_diffusion/runtime/elastic_policy.py` | SLO-aware layout 枚举、rolling-horizon 评分、barrier-aware forward simulation。 |
| `chitu_diffusion/runtime/generator.py` | pool engine：admission、plan broadcast、lane denoise、migration、retire、timing。 |
| `chitu_diffusion/core/distributed/parallel_state.py` | dynamic SP group 和 active lane topology。 |
| `chitu_diffusion/runtime/cost_model.py` | runtime/offline 共用的 cost model、token 数、online calibration。 |
| `chitu_diffusion/runtime/adapter/base.py` | model adapter 的 pool denoise contract。 |
| `chitu_diffusion/runtime/adapter/z_image.py` | Z-Image 的 CP/dynamic SP/pool 适配主路径。 |
| `chitu_diffusion/runtime/adapter/qwen_image.py` | Qwen-Image 的 lane-aware adapter 路径。 |
| `test/test_z_image_serve.py` | arrival-timed serving trace replayer。 |
| `experiments/cp_dp_hot_switch/` | 离线 simulator、trace generator、cost model、结果记录和分析脚本。 |

## 配置入口

Hot switch 默认关闭。启用 runtime pool engine 的核心配置是：

```yaml
infer:
  diffusion:
    scheduling_policy: slo_elastic   # 或 pure_dp / pure_sp
    dynamic_sp: true
    cost_model: rtx4090              # 或 h20 / 显式 json 路径
    horizon_events: 6
    starvation_ms: 30000.0
    fairness_beta: 3.0
    switch_total_ms: 0.0
    epe:
      cfg_parallel_max: 2
      online_calibration: true
      balanced_k: true
      phase_max_steps: 5
      enable_flexcache: false
```

常用 policy：

| policy | 含义 |
| --- | --- |
| `slo_elastic` | 当前主策略。按 SLO/fairness/latency 等目标动态选 lane layout。 |
| `pure_dp` | 固定 DP baseline。每个 GPU 跑一个 width=1 请求，复用 pool engine。 |
| `pure_sp` | 固定 SP baseline。一次一个请求占满整个 pool，复用 pool engine。 |

常用环境变量：

| 环境变量 | 作用 |
| --- | --- |
| `CHITU_SERVE_TRACE` | 指向 serving trace JSON，`test/test_z_image_serve.py` 必需。 |
| `CHITU_POOL_INGRESS_THREAD=off` | 关闭后台 arrival 注入线程，改用主循环 inline 注入。 |
| `CHITU_POOL_INSTRUMENT=0` | 关闭 pool round/timing 结构化记录，用于更干净的性能 A/B。 |
| `CHITU_POOL_LOCAL_TIMING_DUMP=1` | dump 每 rank lane-local timing，用于分析 barrier / straggler。 |
| `CHITU_POOL_PER_STEP_COST_MS` | 给 planner 加一个每 step lane 残余开销估计。 |
| `CHITU_POOL_BOUNDARY_COST_MS` | 给 planner 加一个 phase boundary 残余开销估计。 |

## 如何跑离线模拟

离线模拟适合先判断一个 workload 是否有 hot switch 空间，速度快、可解释、不会占 GPU。

```bash
cd experiments/cp_dp_hot_switch

# 全策略 x 内置 serve traces，输出指标表和 GPU 时间线 SVG
python3 run_serve_policies.py \
  --cost-model rtx4090 \
  --output-dir out/serve_policies_4090

# H20 cost model
python3 run_serve_policies.py \
  --cost-model h20 \
  --output-dir out/serve_policies_h20

# 只跑某几个 trace / policy
python3 run_serve_policies.py \
  --cost-model rtx4090 \
  --serve-traces traces/serve/bursty_idle_mixed.json traces/serve/poisson_1024_r3n12.json \
  --policies static_dp static_sp4 elastic_hot_switch slo_elastic \
  --decision-log \
  --output-dir out/debug_hot_switch
```

主要输出：

- `serve_policies.md`：策略指标总表。
- `serve_policies.json`：机器可读结果和 decision log。
- `serve_policy_timelines.md` / `serve_policy_timelines/*.svg`：GPU x 时间的执行图。

`out/`、`traces/`、`figures/`、`cost_model.json`、`cost_models/*.json` 都按本地实验数据处理，已被 git ignore。

## 如何生成 trace

推荐先用合成 trace 做明确控制：低负载、突发、混合 shape、饱和队列分别跑一遍。trace JSON 默认留在本地，不提交。

```bash
cd experiments/cp_dp_hot_switch

# Poisson mixed workload
python3 gen_client_trace.py \
  --rate 0.36 \
  --num-requests 24 \
  --sizes 512x512,1024x1024 \
  --size-weights 0.55,0.45 \
  --steps 20 \
  --slo-factor 4 \
  --cost-model rtx4090 \
  --out traces/serve/my_mixed_trace.json

# burst/idle workload：观察 idle 时上 SP、burst 时下 DP
python3 gen_client_trace.py \
  --burst-sizes 1,8,1,6 \
  --intra-burst-ms 70 \
  --lull-ms 22000 \
  --sizes 512x512,1024x1024 \
  --size-weights 0.55,0.45 \
  --steps 20 \
  --slo-factor 4 \
  --cost-model rtx4090 \
  --out traces/serve/my_bursty_trace.json
```

trace 里的 `deadline_ms` 是相对 arrival 的 soft deadline；`slo_elastic` 会读取它，其他 baseline 只用来事后统计。

## 如何跑 Z-Image runtime serving

### 1. 单 GPU baseline

```bash
CHITU_SERVE_TRACE=experiments/cp_dp_hot_switch/traces/serve/bursty_idle_mixed.json \
chitu run test/configs/z_image_serve_baseline.yaml
```

这个配置是单 GPU FIFO baseline，用来确认 harness 和 trace 可以正常跑。

### 2. 4 GPU hot switch pool

```bash
CHITU_SERVE_TRACE=experiments/cp_dp_hot_switch/traces/serve/bursty_idle_mixed.json \
chitu run test/configs/z_image_serve_pool.yaml
```

`z_image_serve_pool.yaml` 默认：

- `models.name=Z-Image`
- `infer.diffusion.scheduling_policy=slo_elastic`
- `infer.diffusion.dynamic_sp=true`
- `infer.diffusion.cost_model=rtx4090`
- `infer.diffusion.epe.enable_flexcache=false`
- `models.sampler.sample_steps=20`

如需跑固定 baseline，用同一个配置模板改 `overrides` 里的 policy：

```yaml
overrides:
  - infer.diffusion.scheduling_policy=pure_dp
  - infer.diffusion.dynamic_sp=true
  - infer.diffusion.cost_model=rtx4090
  - models.sampler.sample_steps=20
```

或：

```yaml
overrides:
  - infer.diffusion.scheduling_policy=pure_sp
  - infer.diffusion.dynamic_sp=true
  - infer.diffusion.cost_model=rtx4090
  - models.sampler.sample_steps=20
```

然后照常运行：

```bash
CHITU_SERVE_TRACE=experiments/cp_dp_hot_switch/traces/serve/bursty_idle_mixed.json \
chitu run <your-z-image-pool-baseline-config.yaml>
```

### 3. 看结果

每次 run 会在 `output.root_dir` 下生成按配置和请求区分的结果目录。重点看：

- `metrics/serve_metrics.json`：请求数、完成数、throughput、latency p50/p95/p99、queue delay。
- `metrics/timing/*.json` 或 timer 输出：pool round、denoise、barrier、migration、retire 等阶段时间。
- run log：每轮 pool layout、admission、completion、异常和 warmup 信息。

对三臂结果做汇总：

```bash
python3 experiments/cp_dp_hot_switch/bench_3way_analyze.py \
  --root outputs/bench_3way_50step \
  --trace experiments/cp_dp_hot_switch/traces/serve/bursty_idle_mixed.json \
  --logs /tmp \
  --out outputs/bench_3way_50step/comparison
```

## 如何尝试 Qwen-Image

Qwen-Image 目前不要直接按“已完整验证”理解。建议步骤是：

1. 先跑现有 smoke / DP 配置，确认模型、checkpoint、attention backend 正常。
2. 基于 `test/configs/z_image_serve_pool.yaml` 新建一个 `qwen_image_serve_pool.yaml`，把 `model.name`、`ckpt_dir`、输出目录和模型采样参数改成 Qwen-Image。
3. 补一个本地 Qwen-Image 对应的 cost model，或者先用显式 JSON 路径做临时实验。
4. 用小 trace 和少步数跑 `pure_dp` / `pure_sp` / `slo_elastic` 三臂，先看 correctness 和完成率，再看性能。
5. 只有当三臂都能稳定完成、输出质量合理、timer 没有明显额外同步后，再扩大 trace 和 step 数。

代码上 Qwen-Image adapter 已经有 dynamic lane topology 处理，但端到端 pool serving 的“推荐配置 + 复现实验”还没像 Z-Image 那样整理完。

## 推荐实验顺序

1. 离线跑 `run_serve_policies.py`，确认目标 trace 上 `slo_elastic` 相比 static DP/SP 有理论收益。
2. 跑单 GPU baseline，确认 trace 和输出目录正常。
3. 跑 4 GPU `pure_dp` 和 `pure_sp`，确认同一 pool engine 的两个固定极端可以稳定完成。
4. 跑 4 GPU `slo_elastic`，和 `pure_dp` / `pure_sp` 比较 throughput、p95、queue delay、SLO miss。
5. 打开 timing dump，定位真实 overhead：plan/broadcast/barrier/migration/retire 哪个最重。
6. 用观测到的 overhead 回填 `CHITU_POOL_PER_STEP_COST_MS` / `CHITU_POOL_BOUNDARY_COST_MS`，再跑一次，看 planner 决策是否更贴近真机。

## 质量和正确性检查

轻量单测：

```bash
python3 -m pytest \
  test/test_slo_scheduler.py \
  test/test_serve_sim.py \
  test/test_cp_dp_simulator.py \
  test/test_cost_model.py \
  test/test_sp_migration.py \
  test/test_work_item.py \
  -q
```

Z-Image GPU smoke / serving：

```bash
chitu run test/configs/z_image_smoke.yaml

CHITU_SERVE_TRACE=experiments/cp_dp_hot_switch/traces/serve/pool_smoke.json \
chitu run test/configs/z_image_serve_pool.yaml
```

做性能结论前，至少确认：

- `num_completed == num_requests`
- baseline 和 hot switch 的 `num_inference_steps` 一致
- `pure_dp` / `pure_sp` / `slo_elastic` 使用同一 trace、同一 checkpoint、同一采样参数
- run log 里没有 fallback、OOM、异步 output 失败或 rank 间不同步
- 关闭或固定影响性能的 debug/timing 开关，或者在三臂里保持一致

## 当前已知风险

- 同步 phase 会引入 barrier idle。混合 lane 很容易出现 fast lane 等 slow lane，这部分必须从 timing 里单独看。
- 低分辨率请求通常 SP-negative，错误上 SP 会拉高 tail latency。
- saturated homogeneous workload 下，最大 SP 并不一定最好；DP concurrency 可能更优。
- cost model 必须按硬件重标定。4090 PCIe 和 H20 NVLink 的通信代价差异很大。
- `cfg_parallel_max=2` 会影响 planner 对 width 的定价；模型若不支持对应 CFG 路径，需要显式降到 1 或做单独验证。
- 目前不是完全异步 serving runtime，不能把 simulator 里的理想时间线当成最终线上开销。

## 相关阅读

- [`README.md`](README.md)：实验目录总览和当前主线。
- [`slo_elastic_scheduler_strategy.md`](slo_elastic_scheduler_strategy.md)：SLO-aware scheduler 设计。
- [`m8_slo_elastic_report.md`](m8_slo_elastic_report.md)：M8 离线结果和时间线。
- [`m6_dynamic_sp_report.md`](m6_dynamic_sp_report.md)：dynamic SP / step-boundary 切换机制。
- [`cost_models/report.md`](cost_models/report.md)：硬件 cost model 和 RTX 4090 / H20 差异。
- [`WORKLOG.md`](WORKLOG.md)：按日期记录的上下文、修复和实验结论。
