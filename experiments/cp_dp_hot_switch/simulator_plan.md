# 模拟器计划

第一个实现目标应该是离线模拟器，而不是 runtime 热切换原型。模拟器用于判断这个 feature 是否存在有价值的运行区域，避免过早承担 runtime 复杂度。

## 输入

- 逐 step 延迟：
  - `sp_2gpu_step_ms`：一个请求使用两卡 CP/SP 的逐 step 延迟。
  - `dp_1gpu_step_ms`：一个请求在单卡上执行的逐 step 延迟。
  - 可选 `sp_1gpu_step_ms`：如果某个模型的单卡 baseline 与 DP 语义不同，则单独记录。
- 工作负载：
  - 请求到达时间戳；
  - 每个请求的 denoising step 数；
  - 每个请求的分辨率、token/patch 序列长或等价的计算量 proxy；
  - 可选优先级或 deadline。
- 切换模型：
  - `switch_cost_ms`；
  - `switch_allowed_until_step`；
  - `graph_recapture_cost_ms`；
  - `cache_migration_cost_ms`；
  - `communicator_cost_ms`。

## 策略

- `static_dp`：保持两个独立单卡副本。
- `static_sp`：使用两张卡服务一个 CP/SP 请求，其余请求排队。
- `admission_control`：先等待一个短窗口，再决定使用 DP 还是 CP/SP。
- `early_hot_switch`：立刻以 CP/SP 启动；如果另一个请求在早期阈值前到达，则切换到 DP。
- `shape_aware_ratio`：根据不同请求的序列长/分辨率选择 CP/SP degree 和 DP replica 数，例如长序列请求保留更多 CP 资源，短序列请求优先以 DP 形式并发执行。
- `oracle`：使用未来到达信息选择最优动作，作为上界。

## 输出

- 平均延迟；
- P50/P95/P99 延迟；
- 排队延迟；
- GPU busy-time 代理指标；
- 不同序列长请求的 slowdown / fairness；
- 切换次数；
- 浪费或迁移的工作量；
- 用于 debug 的策略 trace。

## 下一个代码产物

新增 `simulate.py`，实现一个确定性的事件模拟器，并配一个小型 JSON 示例 trace。第一次运行可以使用合成的 Poisson 到达过程、从 ChituBench 提取的 step latency，以及几个代表性分辨率/序列长档位。

第一版 trace 至少应覆盖：

- 同分辨率请求：验证基础 CP/SP -> DP hot switch 是否有收益；
- 长请求 + 短请求混合：验证是否应该保留部分 CP/SP 给长请求，同时让短请求以 DP replica 消化；
- 多个短请求 burst：验证 admission control 是否已经足够，hot switch 是否仍有额外收益。
