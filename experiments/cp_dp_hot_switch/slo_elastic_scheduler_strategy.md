# SLO-Aware Elastic CP/DP Scheduler Strategy

本文档给实现系统的 agent 使用，目标是在 M7 `elastic_hot_switch` 的基础上，推进一个更优化式、更可解释的 SLO-aware 调度器。

当前 M7 已经证明：在 4-GPU pool 上，shape-aware + burst-triggered CP/DP hot-switch 可以在 mixed / bursty workload 下改善 p95，并且在 homogeneous 或 SP-negative 场景下正确 no-op。但 M7 的策略仍然是启发式的局部经济判断，还没有显式表达 SLO、公平性、同质高负载下的 load-aware downshift，也没有把 FlexCache 作为极端 SLO 保护动作纳入模拟。

## 1. 调度目标

下一版策略建议命名为 `slo_elastic`。

核心目标按优先级排列：

1. 尽量减少 SLO miss 数量。
2. 如果 miss 不可避免，最小化最严重的超时。
3. 降低总超时和尾延迟。
4. 保证公平性，避免短请求被长 SP 请求长期阻塞。
5. 避免不必要的 CP/DP switch。
6. 在上述目标之后，再最大化吞吐和 GPU 利用率。
7. FlexCache 只作为 simulator-only 的极端 SLO rescue action，不作为常规吞吐优化手段。

不要优先实现加权打分函数。为了减少难解释的启发式权重，优先使用字典序目标。

## 2. 决策边界

调度器只在安全点做决策：

- 新请求到达。
- 某个 denoise step 完成。
- 请求完成。
- 触发极端 SLO 压力时，模拟器允许考虑 FlexCache step reduction。

运行中的请求只能在自己的 denoise step boundary 改变 SP degree。

## 3. 状态建模

每个决策点观测：

```text
now_ms
total_gpus
active requests
pending requests
```

每个请求包含：

```text
request_id
arrival_ms
deadline_ms / slo_ms
shape / width / height / token count
remaining_steps
original_steps
current_step
current_degree
current_gpu_allocation
switch_count
flexcache_reduced_steps
```

由 cost model 得到：

```text
step_time(i, k)
remaining_time(i, k) = remaining_steps_i * step_time(i, k)
```

其中 `k=1` 是 DP-style 单卡执行，`k=2/4` 是 SP/CP。

派生指标：

```text
solo_service_time_i = original_steps_i * step_time(i, 1)
predicted_completion_i
tardiness_i = max(0, predicted_completion_i - deadline_i)
slack_i = deadline_i - now_ms - predicted_remaining_time_i
slowdown_i = (predicted_completion_i - arrival_ms_i) / solo_service_time_i
```

`slowdown_i` 是公平性的核心指标。短请求的 `solo_service_time_i` 小，如果被长请求卡住，slowdown 会快速变大，因此它能自然暴露 head-of-line blocking。

## 4. 候选动作枚举

GPU 数较小时，直接枚举合法并行配置是可行的。

以 `G=4` 为例：

```text
[1,1,1,1]  four DP requests
[2,1,1]    one SP2 request plus two DP requests
[2,2]      two SP2 requests
[4]        one SP4 request
```

然后枚举 group 到请求的 assignment：

```text
request -> degree
request -> gpu indices
optional switch event
optional FlexCache step reduction
```

硬约束：

```text
sum(active_widths) <= total_gpus
GPU cannot be double-booked
request cannot execute overlapping steps
degree switch only at request step boundary
degree switch only inside switch window unless explicitly configured otherwise
FlexCache cannot reduce below flexcache_min_steps
```

实现上先支持 `G <= 8` 的枚举即可。请求很多时可以只保留关键候选：

- 已在运行且可 switch 的请求。
- deadline 最近的 pending 请求。
- slack 最小的请求。
- slowdown 最大的请求。
- shortest remaining processing time 的短请求。

## 5. 字典序优化目标

对每个候选 layout 做短视野模拟，得到 objective tuple：

```text
objective =
(
  slo_miss_count,
  max_tardiness_ms,
  total_tardiness_ms,
  max_slowdown,
  starvation_count,
  flexcache_used_count,
  flexcache_reduced_steps_total,
  switch_count,
  total_switch_cost_ms,
  -completed_work_ms,
  -gpu_busy_ratio
)
```

按 tuple 顺序比较。前面的字段优先级高于后面的字段。

含义：

- SLO miss 数量永远优先于吞吐。
- 最严重请求的超时优先于平均表现。
- 公平性优先于减少 switch。
- FlexCache 使用次数和减少 step 数优先于吞吐收益。
- 只有当 SLO、公平性、FlexCache、switch 成本都相同时，才偏向更多 completed work 和更高 GPU busy。

这样可以避免手写大量权重，同时让每次决策可解释。

## 6. Rolling-Horizon 规划

不要只看下一步的即时收益，否则策略容易退化为局部贪心。

每个决策点执行：

```text
1. collect active + pending requests
2. enumerate legal first-step layouts
3. for each first-step layout:
     simulate H events or H milliseconds
     use deterministic step_time from cost model
     use a simple fallback policy for later simulated decisions
     compute objective tuple
4. choose the best first-step layout
5. execute only the first step
6. re-plan at the next decision point
```

推荐初始 horizon：

```text
H_events = 4 to 8 step completions
or
H_ms = 2x to 4x max current step time
```

如果实现复杂度较高，可以先做 one-step enumeration + predicted remaining completion，再升级为 multi-step rolling horizon。

## 7. 公平性

当前 M7 的主要收益来自避免大 SP 请求 head-of-line-block 小请求，但这个行为还没有被显式建模成公平性目标。

下一版应该显式记录：

```text
slowdown_i = predicted_latency_i / solo_service_time_i
wait_ms_i = now_ms - arrival_ms_i if not started
```

建议策略：

```text
include max_slowdown in lexicographic objective
include starvation_count when wait_ms exceeds threshold
optionally enforce slowdown_i <= beta when feasible
```

示例：

```text
beta = 3.0
```

如果没有任何 schedule 能满足 `slowdown_i <= beta`，选择 max slowdown 最小的候选。

不要只用 FIFO。FIFO 会让短请求被长请求卡住。也不要只用 SRPT。SRPT 会让长请求在高负载下饿死。`max_slowdown` 更适合作为混合 workloads 下的公平指标。

## 8. FlexCache 模拟策略

FlexCache cost / quality model 尚未实现。

本阶段只在模拟器内把 FlexCache 建模为 emergency SLO rescue action。不建质量损失函数，也不把它作为常规吞吐优化动作。

### 8.1 抽象模型

FlexCache 表示减少请求剩余 denoise steps：

```text
remaining_steps_i := max(flexcache_min_steps, remaining_steps_i - reduced_steps)
```

节省时间：

```text
time_saved_i = reduced_steps * step_time(i, current_degree)
```

### 8.2 触发条件

调度顺序：

```text
1. 先枚举不使用 FlexCache 的正常 CP/DP layouts。
2. 如果存在正常 layout 可以避免极端 SLO 风险，不使用 FlexCache。
3. 如果所有正常 layouts 都存在严重 SLO 压力，才打开 FlexCache candidates。
4. 选择能改善 SLO miss count 或 worst tardiness 的最小 step reduction。
```

极端 SLO 条件可以先用保守阈值：

```text
tardiness_i > flexcache_emergency_tardiness_ms
```

或：

```text
slack_i(best_possible_degree) < -flexcache_emergency_slack_ms
```

推荐 simulator knobs：

```text
--enable-flexcache-sim
--flexcache-min-steps
--flexcache-max-reduce-steps
--flexcache-emergency-tardiness-ms
--flexcache-emergency-slack-ms
```

### 8.3 目标函数中的位置

由于没有质量模型，不要伪造 quality penalty。

用 hard gate 限制 FlexCache：

```text
FlexCache is forbidden unless the request is in extreme SLO risk.
```

gate 打开之后，objective 中包含：

```text
flexcache_used_count
flexcache_reduced_steps_total
```

这保证：

- FlexCache 只用于改善 SLO。
- 优先减少使用 FlexCache 的请求数。
- 在同等 SLO 改善下，优先减少 step reduction。
- 不让 FlexCache 变成吞吐 hack。

### 8.4 实验口径

报告中必须说明：FlexCache 结果是 upper-bound / sensitivity study，不是 production-ready quality-preserving policy。

它回答的问题是：

```text
如果存在一个 emergency step-reduction mechanism，
调度器在极端 SLO 压力下会多频繁使用它，
以及它理论上能改善多少 tail latency / SLO attainment？
```

## 9. 为什么当前 M7 启发式还不够好

M7 `elastic_hot_switch` 已经是一个强 baseline，但它仍然有几个结构性不足。

### 9.1 它主要是局部 pairwise 判断

M7 的 burst-triggered downshift 会比较：

```text
downshift running request R
vs
让 pending request P 等待自然完成
```

这个 pairwise economics 可以避免无意义切换，但它只看一个 running request 和一个 pending request 的局部收益。它没有完整表达全局目标，例如全队列的 SLO miss 数、最大超时、短请求 slowdown、未来几步的排队演化。

结果是：局部 pairwise 看起来合理的选择，不一定是全局 tail latency 或 SLO 最优。

### 9.2 它缺少显式 SLO 目标

当前 M7 主要优化 p95 / throughput / queue 等统计结果，但策略本身没有把 per-request deadline 写进目标函数。

这会导致两个问题：

- 快要 miss SLO 的请求没有天然优先级。
- 已经有充足 slack 的请求可能仍然占用大 SP group，阻塞更紧急的请求。

SLO-aware 策略应该直接比较 predicted deadline miss，而不是事后只看 p95。

### 9.3 它的 shape-aware degree 是 load-oblivious

M7 会根据单请求 per-step latency 选择 shape-aware initial degree。例如 1024² 可能选择 `k=4`，因为单请求延迟最低。

但在同质高负载下，单请求最优不等于系统最优。

典型问题：

```text
SP4 单请求更快，但 4 路 DP concurrency 可能在饱和队列下有更好吞吐和 p95。
```

M7 报告中 `poisson_1024_r3` 就是这个问题：shape-aware / elastic 倾向 SP4，但 best static 是 DP。原因是 SP4 speedup sub-linear，queue delay 主导 tail latency。

下一版必须让 degree selection 看到 backlog depth 和 deadline pressure。

### 9.4 它没有显式公平性约束

M7 的 mixed workload 收益来自避免大 SP 请求阻塞小 DP-friendly 请求，但这只是策略行为的结果，不是明确建模的约束。

如果 workload 更复杂，短请求仍可能被长请求卡住。仅靠 burst-triggered downshift 不足以保证 fairness。

应该显式引入：

```text
max_slowdown
starvation_count
wait_ms threshold
```

这样短请求长时间排队会直接进入 objective，而不是依赖启发式触发条件。

### 9.5 它没有 rolling-horizon 全局规划

M7 的策略是在线启发式，倾向于 keep-by-default、admit idle、必要时 downshift。

这很好解释，也比较稳，但它没有尝试枚举多个 layout 并模拟未来几个 step 的结果。

由于 DiT step time 可预测，调度器其实可以做更强的短视野优化：

```text
当前 layout 不只影响下一步，
还影响未来几步 request completion、queue release 和 SLO miss。
```

利用 rolling horizon 可以更早发现即将发生的 tail collapse。

### 9.6 它没有 FlexCache emergency action

M7 不考虑“少跑 step 来救 SLO”的选项。

在极端 SLO 压力下，单纯 CP/DP layout 可能已经无解。此时 simulator 应该允许 FlexCache step reduction，观察理论上能救多少 SLO。

注意：这不是质量模型，也不是生产策略，只是为后续 FlexCache 设计提供 sensitivity evidence。

## 10. 需要输出的指标

保留 M7 现有指标，并新增：

```text
slo_miss_count
slo_attainment_by_threshold
max_tardiness_ms
total_tardiness_ms
mean_slowdown
max_slowdown
starvation_count
flexcache_used_count
flexcache_request_fraction
flexcache_reduced_steps_total
flexcache_reduced_steps_mean
slo_saved_by_flexcache_count
```

每个请求输出：

```text
request_id
arrival_ms
deadline_ms
start_ms
finish_ms
latency_ms
tardiness_ms
slowdown
shape
original_steps
executed_steps
flexcache_reduced_steps
switch_count
degree_timeline
```

## 11. 实现建议

建议按以下顺序实现：

1. 在 `simulate_pool()` 旁边增加 `slo_elastic` policy，不替换 M7。
2. 实现合法 GPU partition enumeration。
3. 实现候选 assignment 枚举和基础剪枝。
4. 实现 objective tuple 和 lexicographic comparator。
5. 先做 one-step predicted completion，再扩展 rolling horizon。
6. 加入 slowdown / starvation 指标。
7. 加入 FlexCache simulator-only emergency gate。
8. 扩展 `run_serve_policies.py` 输出新 metrics。
9. 用 `rtx4090` 和 `h20` 都跑一遍，和 M7 `elastic_hot_switch` 对比。

## 12. 必测场景

至少覆盖：

1. 512² on 4090：保持 DP，不应错误使用 SP。
2. mixed burst：应不差于 M7 elastic。
3. realistic_sd3_mixed：应改善或持平 p95 / SLO。
4. homogeneous saturated 1024：应能从 SP4 下调到 DP-like layout，修复 M7 的 Pareto loss。
5. long request + short request：短请求不能被长 SP 请求长期阻塞。
6. impossible SLO：FlexCache 开启后应减少 tardiness 或 SLO miss。
7. non-emergency SLO：FlexCache 不应被使用。
8. switch window：非法 late switch 不应发生。
9. GPU invariant：`sum(widths) <= total_gpus`，无 double-book。

## 13. 决策日志

每次调度决策建议记录：

```text
now_ms
chosen_layout
best_competing_layout
chosen_objective
competing_objective
deciding_objective_field
switches
flexcache_actions
reason
```

实现目标不是只让数字更好，而是让策略可以被解释：

```text
这次为什么从 SP4 降到 DP？
是因为 SLO miss 数减少、max tardiness 降低、short-request slowdown 超阈值，
还是只是因为 GPU busy 更高？
```

只有能回答这个问题，调度策略才足够稳健。
