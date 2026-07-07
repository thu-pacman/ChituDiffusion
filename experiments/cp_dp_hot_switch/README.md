# CP/DP 热切换实验

在 DiT 推理中做**动态并行热切换**：单请求 / 空闲时用序列并行（SP，`cp2`/`cp4`）压低延迟，多请求 / 突发时切成数据并行（DP，`dp1`）多副本，在 denoise step 边界弹性调整每个请求的 SP degree。本目录是离线**调度器 + 成本模型**研究，用来量化收益、定策略，为真正接入 runtime 做准备。

```text
空闲：       请求 A 用 GPU0-3 做 SP（cp4），降低单请求延迟
突发到达：   请求 B/C/D 在 A 还有剩余 step 时到达
弹性切换：   在 step 边界降 A 的 width，腾出 GPU 起 B/C/D 的 DP 副本
目标：       SLO 达成优先，其次公平（max_slowdown）、尾延迟、吞吐
```

## 当前主线（canonical）

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

### 数据

- `cost_models/{rtx4090,h20}.json` + [`cost_models/report.md`](cost_models/report.md)：按硬件的逐 step 延迟标定（compute roofline + comm）。`cost_model.json` 是传统默认。
- `traces/serve/*.json`：serving 评估 trace（`realistic_sd3_mixed` 默认；`poisson_*` 负载 sweep；`bursty_idle_mixed` 演示空闲 SP↔突发 DP 的切换；`staggered_mixed` 便于看 lane）。
- `figures/`：m8 报告引用的时间线 PNG（已入库，不依赖 gitignore 的 `out/`）。

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

## 下一步：接入 runtime

离线已经证明 `slo_elastic` 在 mixed/bursty 上收益真实、公平性显著改善、并修掉了 M7 在饱和同构下的 Pareto loss。接入 runtime 的收敛路径：

1. **统一 action schema**：把 `slo_elastic` 的每次决策映射成 runtime 的 `SchedulingPlan`，显式表达 `{work-item ids, 每请求 target sp_degree/width, GPU 分配, 可选 switch, 可选 FlexCache 动作, predicted cost, deadline slack, 决策理由}`。
2. **generator 只执行 plan**：收口 M4 的 continuous-batch admission 与 M6 的 env-driven SP 切换到 scheduler plan。
3. **request-level DP replica routing**：现有 DP helper 是 `n_sample` seed-slice，需要补“多请求分配到多个 DP replica”的路由，以及 `world_size>1` 下到达间的 idle-sync / rank barrier。
4. **先标定再上线**：用 static baselines（`run_serve_policies` 的 `static_*` policy）校准 cost model，再上线 `slo_elastic`，回填 SLO/queue/mean/p95/busy 指标做在线闭环。
5. **FlexCache 谨慎**：目前是 simulator-only 的紧急闸门；只有在真机验证“作为调度动作 1+1>2”后才进 runtime，否则保持 always-cache 基线。

runtime 机制层参考：[`m1_cost_model_report.md`](m1_cost_model_report.md)（成本模型）、[`m6_dynamic_sp_report.md`](m6_dynamic_sp_report.md)（step 边界 SP 切换机制）、[`literature_notes.md`](literature_notes.md)（Shift Parallelism / LoongServe / Gyges / Seesaw 等动态并行工作）。
