# CP/DP 热切换实验

在 DiT 推理中做**动态并行热切换**：单请求 / 空闲时用序列并行（SP，`cp2`/`cp4`）压低延迟，多请求 / 突发时切成数据并行（DP，`dp1`）多副本，在 denoise step 边界弹性调整每个请求的 SP degree。本目录记录离线**调度器 + 成本模型**研究，以及当前接入 runtime pool engine 后的真机对比。

```text
空闲：       请求 A 用 GPU0-3 做 SP（cp4），降低单请求延迟
突发到达：   请求 B/C/D 在 A 还有剩余 step 时到达
弹性切换：   在 step 边界降 A 的 width，腾出 GPU 起 B/C/D 的 DP 副本
目标：       SLO 达成优先，其次公平（max_slowdown）、尾延迟、吞吐
```

## 当前主线（canonical）

- **内部使用说明**：[`HOT_SWITCH_README.md`](HOT_SWITCH_README.md) —— 给同事复现实验和试用 runtime hot switch 的 README，包含当前实现状态、模型适配范围、配置项、trace / simulator / runtime serving 的使用方式和已知风险。
- **策略设计**：[`slo_elastic_scheduler_strategy.md`](slo_elastic_scheduler_strategy.md) —— SLO-aware `slo_elastic` 调度器的设计（合法 GPU layout 枚举 → rolling-horizon 前向模拟 → 字典序 SLO-first 目标 + FlexCache 紧急闸门）。
- **最新结果**：[`m8_slo_elastic_report.md`](m8_slo_elastic_report.md) —— `slo_elastic` vs `elastic_hot_switch` / static 的头对头结果、GPU 使用时间线（图见 `figures/`）、FlexCache 敏感性、诚实的代价与局限。
- **工作记录**：[`WORKLOG.md`](WORKLOG.md) —— 按日期的上下文同步、bug 修复、trace 调整、M1~M8 关键结论与本次清理口径（已折叠的旧文档结论都在这里）。

### 代码

| 文件 | 作用 |
| --- | --- |
| `simulate.py` | N-GPU 事件模拟器 `simulate_pool()` + 全部策略（`static_dp/sp2/sp4`、`shape_aware`、`elastic_hot_switch`、`slo_elastic`、`oracle`）+ 成本模型加载 |
| `run_serve_policies.py` | 自包含 runner：cost model × serve trace × 策略跑一遍，输出对比表 + 每 (trace, policy) 的 GPU 使用时间线 SVG |
| `gen_client_trace.py` | 合成 serve trace（Poisson，以及 `--burst-sizes`/`--lull-ms` 的 burst/idle 分相模式） |
| `profile_worker.py` / `profile_stages.py` | 真机标定：DiT 逐 step（DP/SP）与非去噪阶段延迟，产出 `cost_models/*.json` |
| `bench_3way_analyze.py` | 分析 runtime 三臂 serving 结果：`slo_elastic` / `pure_dp` / `pure_sp` 同 pool engine 对比 |

### 数据

- `cost_models/{rtx4090,h20}.json` + [`cost_models/report.md`](cost_models/report.md)：按硬件的逐 step 延迟标定（compute roofline + comm）。`cost_model.json` 是传统默认。
- `traces/serve/*.json`：serving 评估 trace（`realistic_sd3_mixed` 默认；`poisson_*` 负载 sweep；`bursty_idle_mixed` 演示空闲 SP↔突发 DP 的切换；`staggered_mixed` 便于看 lane）。
- `figures/`：m8 报告引用的时间线 PNG（已入库，不依赖 gitignore 的 `out/`）。

### Runtime 接入状态

`slo_elastic` 已接入 serving runtime 的 pool engine：rank 0 每个 phase 规划完整 lane layout，worker rank 在各自 lane subgroup 上执行 bounded denoise phase，随后在 boundary 做迁移/retire。`pure_dp` 和 `pure_sp` 作为固定 layout baseline 复用同一 engine，便于把策略效果和 engine 开销分开看。

当前 runtime 仍是同步 phase 模型，不是完全异步模拟器：phase 边界仍有 plan/broadcast/barrier/retire 成本。实现里已经加入 multi-step phase、placement affinity、targeted latent migration、async output，以及 residual runtime overhead 入口，用于缩小模型预测和真机开销的差距。

### 测试

`test/test_slo_scheduler.py`（`slo_elastic` 策略 §12 九个必测场景）、`test/test_serve_sim.py`（pool 不变量 + elastic-beats-static）、`test/test_cp_dp_simulator.py`、`test/test_cost_model.py`。

## 复现

```bash
cd experiments/cp_dp_hot_switch
# 全策略 × serve trace + 时间线 SVG（RTX 4090 无 NVLink / H20 有 NVLink）
python3 run_serve_policies.py --cost-model rtx4090 --output-dir out/serve_policies_4090
python3 run_serve_policies.py --cost-model h20     --output-dir out/serve_policies_h20
# 测试
python3 -m pytest ../../test/test_slo_scheduler.py ../../test/test_serve_sim.py ../../test/test_cp_dp_simulator.py -q
```

`out/` 是生成物（gitignored），随时可重跑重建。

## 下一步

1. **重跑三臂 runtime benchmark**：用当前代码同时重跑 `slo_elastic` / `pure_dp` / `pure_sp`，更新 `outputs/bench_3way_50step/comparison` 的报告。已有 7/8 `slo_elastic` 结果优于 7/7 pure baselines，但严格结论需要同版本重跑。
2. **标定 residual overhead**：从 pool timers 回填 `CHITU_POOL_PER_STEP_COST_MS` / `CHITU_POOL_BOUNDARY_COST_MS`，让 planner 在真机上感知 phase 边界开销。
3. **FlexCache 谨慎**：目前 production runtime 默认关闭 emergency step-reduction；只有在真机验证“作为调度动作 1+1>2”后才进入线上策略。

runtime 机制层参考：[`m1_cost_model_report.md`](m1_cost_model_report.md)（成本模型）、[`m6_dynamic_sp_report.md`](m6_dynamic_sp_report.md)（step 边界 SP 切换机制）、[`literature_notes.md`](literature_notes.md)（Shift Parallelism / LoongServe / Gyges / Seesaw 等动态并行工作）。
