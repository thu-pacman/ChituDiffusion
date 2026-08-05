# Hot Switch 实验报告

本文只维护当前实现对应的关键结论。历史里程碑、探索过程、旧图表和旧 trace 不再保留；数值结论若未注明 runtime，均来自 4-GPU 离线 simulator。

## 当前版本

当前主线是 `slo_elastic`：在 denoise step/phase 边界枚举合法 GPU layout，用 rolling-horizon 前向模拟评估请求队列，并按以下优先级选取计划：

1. SLO miss、最大和总 tardiness；
2. 最大 slowdown 和 starvation；
3. 切换次数和切换成本；
4. 总 flow time 和平均 slowdown。

连续指标会量化后再比较，避免为了预测噪声频繁切换。低负载时可将多个 GPU 分给单请求做 SP/CP；队列加深时倾向拆成多个 width=1 lane，以 request-level DP 并发排队请求。runtime 使用同步 multi-step phase、预热通信组、placement affinity、targeted latent migration 和 online cost calibration 执行计划。

FlexCache step reduction 只作为离线敏感性研究。production runtime 默认关闭，不属于当前已验证的 hot-switch 动作。

## 关键离线结果

统一设置：4 GPU、20 denoise steps、`switch_total_ms=85`、`slo_factor=4`。本地 RTX 4090/H20 cost model 和 trace 均为 gitignored 实验输入；因此这些数据用于说明机制和相对趋势，不应直接外推到其他机器或线上流量。

| 场景 | 硬件 | 对比 | p95 | throughput | 其他结论 |
| --- | --- | --- | ---: | ---: | --- |
| 饱和同构 1024 (`r3n12`) | RTX 4090 | `elastic_hot_switch` -> `slo_elastic` | 36.5s -> 29.9s | 0.286 -> 0.359 req/s | max slowdown 3.52 -> 2.75 |
| 饱和同构 1024 (`r3n12`) | H20 | `elastic_hot_switch` -> `slo_elastic` | 39.6s -> 33.1s | 0.266 -> 0.327 req/s | max slowdown 3.48 -> 2.78 |
| heavy mixed | RTX 4090 | `elastic_hot_switch` -> `slo_elastic` | 35.8s -> 36.4s | 0.557 -> 0.537 req/s | SLO miss 17 -> 15；max slowdown 10.74 -> 6.00 |
| realistic SD3 shape mix | RTX 4090 | `elastic_hot_switch` -> `slo_elastic` | 31.0s -> 32.7s | 0.054 -> 0.054 req/s | 0 miss；max slowdown 2.50 -> 1.89 |
| burst/idle mixed | RTX 4090 | `slo_elastic` | 14.7s | - | max slowdown 1.52，优于 M7 的 5.62 和 static SP4 的 9.00 |

结论：

- hot switch 的主要机会来自负载变化，而不是固定高压。空闲阶段用 SP 降低单请求延迟，burst 阶段转 DP 并发，才有切换价值。
- 饱和同构 workload 下，最大 SP 不一定最优。若 SP 加速低于 GPU 数量，DP concurrency 会更快排空队列。
- 混合 shape 下，显式 SLO 和 fairness 能减少大请求对短请求的 head-of-line blocking；代价是 SLO 已满足时，p95 可能略输给针对单一场景调过的 heuristic。
- 512 分辨率在当前 cost model 上通常 SP-negative，正确行为是保持 DP 且不切换。
- GPU busy 不是目标函数。最大化 busy 会偏好宽 SP packing，却可能同时恶化吞吐、排队和 SLO。

## Runtime 现状

Z-Image 是当前完整适配和验证对象。`pure_dp`、`pure_sp` 和 `slo_elastic` 共用同一 pool engine，便于隔离策略收益与 engine overhead。Qwen-Image 已有 lane-aware adapter 路径，但还缺同等级的 canonical 配置、硬件标定和端到端三臂 benchmark。

已有 runtime 数据存在版本不齐的问题：旧同版本三臂结果中，`pure_sp` makespan/p95 为 155.6/115.9s，`pure_dp` 为 144.9/109.5s，旧 `slo_elastic` 为 178.0/141.7s；随后单独更新的 `slo_elastic` 降到 138.3/102.1s、0.065 req/s。后者不能与旧 baseline 严格横比。当前唯一可靠的 runtime 下一步是用同一代码、trace、checkpoint 和采样参数重跑三臂。

runtime 与 simulator 的主要差距是同步 phase 的 plan/broadcast/barrier/migration/retire 成本。结论中必须同时报告 queue delay、p95、SLO miss、throughput 和 barrier idle，不能只看 simulator timeline 或 GPU busy。

## 复现

生成本地 trace：

```bash
cd experiments/cp_dp_hot_switch
python3 gen_client_trace.py \
  --burst-sizes 1,8,1,6 --intra-burst-ms 70 --lull-ms 22000 \
  --sizes 512x512,1024x1024 --size-weights 0.55,0.45 \
  --steps 20 --slo-factor 4 --cost-model rtx4090 \
  --out traces/serve/bursty_idle_mixed.json
```

运行离线策略对比并生成本地指标和 SVG timeline：

```bash
python3 run_serve_policies.py \
  --cost-model rtx4090 \
  --serve-traces traces/serve/bursty_idle_mixed.json \
  --policies static_dp static_sp4 elastic_hot_switch slo_elastic \
  --decision-log --output-dir out/latest
```

运行 Z-Image 同 engine 三臂：

```bash
CHITU_SERVE_TRACE=experiments/cp_dp_hot_switch/traces/serve/bursty_idle_mixed.json \
chitu run test/configs/z_image_serve_pool.yaml
```

分别将 `infer.diffusion.scheduling_policy` 设为 `pure_dp`、`pure_sp`、`slo_elastic`，并保持其余参数完全一致。结果由 `bench_3way_analyze.py` 汇总。

验证：

```bash
python3 -m pytest \
  ../../test/test_slo_scheduler.py ../../test/test_serve_sim.py \
  ../../test/test_cp_dp_simulator.py ../../test/test_cost_model.py \
  ../../test/test_sp_migration.py ../../test/test_work_item.py -q
```

## 结果边界

- 离线 deadline 多为 `slo_factor * fastest solo service time`，不是生产 SLO。
- 离线 cost model 主要覆盖 denoise compute/communication；runtime 固定开销必须真机测量。
- 本地 trace、cost model、JSON、图片和输出目录均不提交。报告只保留足以审计结论的设置、数字和命令。
- 新实验只有在同版本三臂完成后才更新本报告；旧结果直接替换，不新增里程碑报告。
