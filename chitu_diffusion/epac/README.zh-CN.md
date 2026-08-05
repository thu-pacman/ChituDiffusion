# EPAC 核心

[English](README.md) | [简体中文](README.zh-CN.md)

`chitu_diffusion.epac` 只包含模型无关的 Elastic Parallel Caching Engine 代码。该目录
不得导入 Z-Image、FastAPI、torchrun launcher 或旧 ChituDiffusion backend。

EPAC 将固定的分布式 stage 组织成一个或多个上下文并行 lane。它在启动时测量执行代价，
根据吞吐和 SLO 目标安排请求，并可在 pulse 边界调整 lane 宽度。静态生成和常驻服务共享
同一套 executor 与 request state 协议。

## 设计原则

- **模型无关核心：**模型 tensor 约定属于 `chitu_diffusion.models`，不能进入 planner 或
  worker 协议。
- **统一 executor：**static CP、static DP 和 elastic 只在 lane constraint 上不同，不应
  改变模型执行语义。
- **实测驱动：**planner 使用按 workload shape 与 lane width 索引的启动及在线实测代价。
- **collective 安全：**同一 lane 内各 rank 接收相同 lease，并执行一致的分布式控制流。
- **显式 capability：**未支持的调度、迁移或 decode 模式提前报错，不允许隐藏式回退到
  整段 pipeline 调用。

## 执行模型

```text
request queue
    |
request profile + measured cost table
    |
EpeSchedulingPolicy
    |
StepPlan(request, lane_ranks, K)
    |
PulseCoordinator ---- LaneLease ---- lane-local workers
    ^                                      |
    +------------- LaneReport -------------+
```

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `request.py` | 框架无关的 request、status、result 和 metrics |
| `api.py` | typed request helper 与同步 Diffusers facade |
| `image_decoder.py` | stage world、executor、迁移、completion 与 embedded API 类型 |
| `model_executor.py` | 可复用 executor 生命周期和 stage-world 构造 |
| `model_scheduling.py` | 实测代价状态、在线校准与 planner facade |
| `optimization.py` | FlexCache 兼容的 model/prediction hook |
| `cost.py` | 按序列长度和 lane 宽度索引的启动实测代价表 |
| `scheduling.py` | request profile、step plan 与 lane constraint |
| `epe.py` | 代价与 SLO 感知的 layout planner |
| `pulse.py` | epoch、lane lease、deadline、report 与 pull 协议 |
| `worker.py` | lane-local minimum-K、opportunistic step 与 pull loop |
| `worker_pool.py` | 固定 lane pool 与 rank 间点对点通信 |
| `lane_broker.py` | rank-local 进度聚合、pull 与 pulse 重规划 |
| `timeline.py` | rank-local 事件缓冲与关闭时 trace 输出 |

公开接入边界是 `model_executor.py` 与 `image_decoder.py` 中的 executor 生命周期。新增模型
adapter 前请阅读 [`../INTEGRATING_IMAGE_DECODER.md`](../INTEGRATING_IMAGE_DECODER.md)。

## 统一调度策略

三种策略是同一个 planner 和 executor 上的 constraint，不是三套 runtime：

| 策略 | 允许的 lane 宽度 | 运行中 lane 规则 |
| --- | --- | --- |
| `static_dp` | `{1}` | 每个请求保持 singleton lane |
| `elastic` | 配置的因数，如 `{1,2,4}` | 在 pulse 边界调整宽度 |
| `static_cp` | `{world_size}` | 使用一个 full-world lane |

`LaneConstraints.create()` 是唯一将策略名映射为 topology 限制的位置。模型和服务代码不得
另行实现 static-DP/CP planner 分支。

## Pulse 协议

Pulse 是控制面 epoch，不是每固定步数执行一次 `dist.barrier(WORLD)`：

1. `PulseCoordinator.open()` 调用统一调度策略生成初始 request/lane/K layout。
2. 每组互不重叠的 rank 收到具有相同 wall-clock deadline 的 `LaneLease`；空闲 rank 可
   pull 新工作。
3. lane 完成 minimum K 后，只在 `LaneLease.can_start_step()` 预测可在过期前完成时执行
   额外 step。
4. 请求完成后，lane 通过 `choose_pull_request()` 获取全局有序的下一个请求。
5. lease 到期时，每个 lane leader 提交经过 epoch 校验的 `LaneReport`。收齐 report 后，
   下一 pulse 才能回收并重新划分全部 rank。

CP collective 始终在 lane 内同步；只有 lane leader 参与 pulse 控制面。Elastic runtime
使用 `DistributedRankExchange` 与 `PulseLaneBroker` 聚合 lane 级进度，不通过 world
collective 汇报。lane resize 时只将 latent 从旧 owner 传给新加入的 rank。

## 代价与 SLO

`default_deadline_ms` 表示从请求到达到完成的默认端到端 SLO；请求级
`deadline_ms` 优先。`deadline_guard_ms`（默认 3000ms）为未稳定建模的控制面和 CPU
发布预留尾部时间，不修改客户端可见 deadline。

Elastic planner 在每个 pulse 对完整 pending 与 running 队列做前向预测。布局依次比较
SLO miss、tardiness、starvation、并发 admission、cp1 归一化有效吞吐、slowdown 和
flow time。有效吞吐使用 `sum(T_cp1 / T_lane)`，避免短序列 raw steps/s 独占资源。

密集队列采用有界搜索：先插入确定性的 CP 宽度和公平性布局，再探索至多 256 个去重
candidate。启动 profiling 至少测量三步，避免冷启动首步决定中位数。在线校准按 sequence
length、lane width、batch size 和 condition count 精确索引；未见 key 使用启动表插值或
外推，不借用其他分辨率的 correction factor。

## Generate 与 Serve

`generate` 创建一个 full-world static-CP lane，不执行 warmup 或 elastic planning。所有
rank 进入 executor，leader 持有最终 Diffusers result。

`serve` 执行启动 profiling、创建 request-local state，并在 pulse 边界持续规划。stage
leader 负责 ingress 与 completion 发布，所有 rank 参与 lane 执行。

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate --model zimage --model-path /path/to/Z-Image \
  --output outputs/zimage.png

torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  serve --stage-config /path/to/stage.yaml
```

## 扩展 EPAC

接入新模型家族：

1. 在 `chitu_diffusion/models/<family>/` 实现共享 executor operations。
2. 声明 capability 并规范化公开 request，不把模型字段泄露到 EPAC 调度类型。
3. 使用 `StepPlan` 给出的 lane ranks；模型 backend 不得缓存固定 CP world size。
4. scheduler、latent、timesteps 和 step cursor 必须属于单个请求。
5. 增加 CPU contract 测试，再与原生 Diffusers pipeline 做单卡和多 rank 输出验收。

调度改动应扩展 `LaneConstraints` 或统一 policy，不得增加模型专用 planner 分支。

## 验证

在仓库根目录运行：

```bash
python -m pytest -q
python -m ruff check chitu_diffusion/epac chitu_diffusion/models test
```

CPU 测试覆盖 request 状态迁移、lane constraint、planner 排序、lease/report、state
transfer contract 和 executor 生命周期。GPU 验收还必须检查 collective 顺序、确定性的
request ownership、同 seed 质量、lane resize 和 leader-only 结果发布。

## 当前限制

- 尚未实现任意 rank 故障和分布式 fatal-state 恢复。
- running request 的取消尚未在 denoise-step 边界广播。
- 代价校准只覆盖已测 workload key；部署时应 warmup 实际服务的分辨率和 lane 宽度。
- FlexCache 有意不接入常驻 EPE 服务，仅作为静态 `generate` 的请求内优化。

因此 EPAC 当前是开发者预览版运行时，不是 production-GA 容错层。
