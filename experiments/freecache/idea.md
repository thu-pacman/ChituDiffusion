# FreeCache Trace Planner 设计草案

## 背景问题

现在的 FreeCache 已经从“在线 tol 决策”逐渐转向“probe 后生成静态查找表”：

- step trace 可以告诉我们哪些时间段速度场平稳，适合 cache。
- 但 full-compute 轨迹上的单步误差不能直接决定 0阶/1阶。
- Qwen-Image 的实验已经说明：离线看 guided velocity 的局部 endpoint error，0阶经常更好；真实闭环生成里，1阶/2阶却明显优于0阶。

所以真正要解决的是：

> 在一次 full-compute probe 过程中，在线模拟许多条 cache 闭环轨迹，在内存中搜索一条最优的 fresh / 0阶 / 1阶 策略路径，最后输出静态查找表。

这里先不考虑 2阶。当前 Qwen / Flux / Wan 的实验证据都不支持 2阶带来稳定收益。

## 目标

给定总步数 `T` 和一个 fresh budget `B`，输出一条策略：

```text
action[t] in {F, R0, R1}

F  = fresh，完整 DiT compute
R0 = 0阶复用，直接使用 last fresh velocity
R1 = 1阶复用，使用 MeanCache/FreeCache 风格 JVP predictor
```

可选目标有两个：

```text
固定 fresh 数量 B：在 B 次 DiT compute 下，使闭环误差最小。
固定误差阈值 eps：在满足误差约束下，使 fresh 数量最少。
```

第一版建议先做固定 `B`，例如 `B=20/25/30`，因为这和速度目标最直接对应。

## 总体流程

```mermaid
flowchart TD
    A[一次 full-compute probe] --> B[每步得到 oracle trace]
    B --> C[x_true[t], v_true[t], x_true[t+1]]
    C --> D[内存中维护候选 cache states]
    D --> E{每个 step 扩展动作}
    E --> F[F: fresh]
    E --> G[R0: reuse 0阶]
    E --> H[R1: reuse 1阶]
    F --> I[推进候选 x_hat]
    G --> I
    H --> I
    I --> J[和 oracle x_true[t+1] 比较误差]
    J --> K[按 fresh_count / error / gap 剪枝]
    K --> D
    K --> L[最终输出查找表]
```

纯文本版本：

```text
full compute:

  t=0        t=1        t=2                 t=T-1
   |          |          |                    |
   v          v          v                    v
  DiT  --->  DiT  --->  DiT  --->  ... --->  DiT
   |          |          |                    |
   +---- oracle trace: x_true[t], v_true[t], x_true[t+1]

in-memory planner:

  state tree / beam:

                state@t
              /    |    \
             F     R0    R1
            /      |      \
       state@t+1 state@t+1 state@t+1

  每一步只保留 Pareto/beam 中的少量候选。
```

## 状态定义

一个候选 state 表示“如果从 step 0 到当前 step 采用某条 cache policy，现在会走到哪里”。

```text
State:
  step: int
  x_hat: Tensor
  fresh_log:
    v: list[Tensor]
    latents_pre: list[Tensor]
    latents: list[Tensor]
    sigmas_pre: list[Tensor]
    sigmas: list[Tensor]
  actions: list[str]       # F / R0 / R1
  fresh_count: int
  last_fresh_step: int
  gap: int                 # 当前距离上一次 fresh 的步数
  err_sum: float
  err_max: float
  err_tail: float
```

其中 `fresh_log` 要尽量复用 runtime FreeCache 的结构，保证 planner 里的 R1 和真实 runtime 的 R1 是同一个预测器。

## 关键转移

每个 step `t`，full-compute probe 会真实得到：

```text
x_true[t]
v_true[t]
x_true[t+1] = scheduler_step(x_true[t], v_true[t])
```

planner 对每个候选 state 扩展三种动作。

### F: fresh

fresh 表示本 step 真实运行 DiT。第一版 probe 里不额外对每条 state 跑 DiT，而是借用 full-compute probe 的 `v_true[t]`：

```text
v_hat = v_true[t]
x_hat_next = scheduler_step(x_hat[t], v_hat)
```

注意：**fresh 不把 `x_hat[t]` reset 到 `x_true[t]`**。

这样才能保留闭环 drift。如果 fresh 直接 reset，就会退化成局部 segment probe，无法解释 Qwen。

然后更新 fresh log：

```text
fresh_log.append(
  latents_pre = x_hat[t],
  latents     = x_hat_next,
  sigma_pre   = sigma[t],
  sigma       = sigma[t+1],
  v           = v_hat,
)
```

### R0: 0阶复用

```text
v_hat = fresh_log.v[-1]
x_hat_next = scheduler_step(x_hat[t], v_hat)
```

### R1: 1阶 JVP 复用

```text
v_hat = jvp_predict_noise_pred(fresh_log, sigmas, step=t, order=1)
x_hat_next = scheduler_step(x_hat[t], v_hat)
```

第一版建议只在 post-CFG guided velocity 上做 planner，因为策略最简洁：

```text
DiT branch compute
  -> CFG all-gather
  -> guidance rescale
  -> guided velocity
  -> planner/runtime cache
  -> scheduler
```

## 误差度量

每次转移后，用 oracle latent 评估候选路径：

```text
err[t+1] = mse(x_hat_next, x_true[t+1]) / mse(x_true[t+1], 0)
```

累计指标：

```text
err_sum += err[t+1]
err_max = max(err_max, err[t+1])
err_tail = err[t+1]
```

第一版排序建议：

```text
score = err_sum + alpha * err_max + beta * err_tail
```

其中 `alpha` 可以让 planner 避免某一步爆炸，`beta` 可以保护尾部质量。

## 剪枝

不做剪枝会有 `3^T` 条路径。T=50 时必须保留小 beam。

建议组合剪枝：

### 1. fresh_count 分桶

固定 budget `B` 时，只保留 `fresh_count <= B` 的 state。

每个 `fresh_count` 桶内保留 top-K：

```text
bucket[fresh_count] = topK(states, key=score)
```

K 可以从 `8/16/32` 开始。

### 2. gap 约束

```text
if gap >= max_gap:
  next action must be F
```

这和现有 FreeCache 的安全边界一致。

### 3. warmup / cooldown 强制 fresh

```text
if t < warmup or t >= T - cooldown:
  next action must be F
```

### 4. domination pruning

如果两个 state 有相同或接近的：

```text
fresh_count
last_fresh_step
gap
```

且一个 state 的：

```text
err_sum <= other.err_sum
err_max <= other.err_max
err_tail <= other.err_tail
```

则删掉被支配的 state。

### 5. hard error guard

```text
if err_tail > hard_tol:
  drop state
```

这会避免明显崩掉的轨迹继续扩展。

## 搜索示意

假设从 step 8 开始允许 cache：

```text
step 8:
  state A: F F F F F F F F

expand:

  A + F  -> A_F
  A + R0 -> A_0
  A + R1 -> A_1

step 9 pruning:

  fresh_count=9: keep A_F
  fresh_count=8: keep best of A_0 / A_1

step 10:

  对保留下来的 state 继续扩展 F/R0/R1
  再按 fresh_count 分桶剪枝
```

最终得到一个策略表：

```text
step:    00 01 02 03 04 05 06 07 08 09 10 11 12 ...
action:   F  F  F  F  F  F  F  F  F R0  F R1  F ...
```

## 和现有实验的关系

这个 planner 解释了为什么之前的几类 probe 会冲突：

### full-compute steptrace

它测的是：

```text
v_true[t] vs v_true[t-1]
v_true[t] vs JVP(full trace)
```

优点：

- 能发现 warmup / cooldown / safe middle region。
- 能发现 phase-sensitive 的危险 step。

缺点：

- 不保留 cache trajectory 的 drift。
- 不能可靠判断 0阶/1阶。

### guided-only segment order probe

它测的是：

```text
从 true anchor 出发，短区间 rollout 到 endpoint。
```

优点：

- 比单步 velocity MSE 更接近 scheduler。

缺点：

- 每个 segment 从 true latent 出发。
- 没有整条闭环轨迹的累计 drift。
- 因此会误判 Qwen：局部 0阶看起来好，但真实生成 1阶/2阶更好。

### Trace Planner

它测的是：

```text
从 step 0 开始维护候选 x_hat，fresh 也不 reset，整条路径闭环推进。
```

优点：

- 直接优化最终 cache policy。
- 能同时决定 fresh 区间和 reuse order。
- 一次 full-compute probe 内完成，不需要每条 policy 单独跑图。

## 近似和风险

第一版 planner 有一个重要近似：

```text
fresh action 使用 v_true(x_true[t])，而不是真正计算 v(x_hat[t])。
```

真实 cache 运行里，fresh step 会在当前 `x_hat[t]` 上跑 DiT。

这个近似成立的前提是：

```text
剪枝让 x_hat[t] 和 x_true[t] 足够接近。
```

因此 planner 输出后，必须再用真实 FreeCache runtime 跑一遍验证。

如果发现 planner 预测和真实质量仍然偏离，可以升级为：

```text
只对 beam 中少数高价值 state，在若干关键 fresh step 上额外做 v(x_hat[t]) 校正。
```

但第一版不建议这么做，因为会破坏“一次 full compute probe 低成本搜索”的简洁性。

## 需要的 runtime 支持

最终 runtime 需要支持一张更细的查找表：

```yaml
flexcache_params:
  strategy: freecache
  policy_table:
    - step: 0
      action: fresh
    - step: 1
      action: fresh
    - step: 9
      action: reuse0
    - step: 11
      action: reuse1
```

或者更紧凑：

```yaml
forced_compute_steps: [0,1,2,3,4,5,6,7,8,10,12,...]
forced_reuse_orders:
  "9": 0
  "11": 1
  "13": 0
```

现有 FreeCache 已有：

```text
forced_compute_steps
global jvp_order
```

下一步只需要增加：

```text
forced_reuse_orders: dict[int, int]
```

这样 planner 输出可以直接进入 runtime 验证。

## 第一版实现计划

1. 增加 `TracePlannerStrategy` 或扩展 `StepTraceStrategy`：
   - 每步 full compute。
   - 保存 oracle `x_true/v_true/sigma` 在内存中。
   - 同步维护 beam states。

2. 先只支持 Qwen post-CFG guided velocity：
   - 和当前 `CHITU_QWEN_STEP_CACHE_AFTER_CFG=1` 对齐。
   - 避免 CFG branch-level 复杂性。

3. 输出：
   - `trace_planner_policy_B20.json`
   - `trace_planner_policy_B25.json`
   - `trace_planner_policy_B30.json`
   - 简单图：step vs action。

4. 增加 FreeCache runtime 支持：
   - `forced_reuse_orders`
   - 每个 reuse step 使用对应 order。

5. 用 planner 输出的 policy 做真实生成验证：
   - Qwen 先测一个 prompt/seed。
   - 对比固定 hq phase jvp1 / post-CFG jvp1。

## 判断成功的标准

第一阶段成功：

```text
planner 输出的 policy 在真实 runtime 中，比手写 interval/phase 不差。
```

更强的成功：

```text
planner 能解释 Qwen:
  为什么同样 phase 下 1阶/2阶优于0阶。

planner 能解释 Wan:
  为什么长 warmup + 中段0阶更好。

planner 能解释 Flux:
  为什么多种策略都差不多。
```

最终目标：

```text
用户只给 total_steps / target_fresh_budget。
FreeCache probe 自动生成:
  warmup
  cooldown
  fresh steps
  per-step reuse order
```

也就是：

```text
FreeCache hyperparameter ~= compute budget
```
