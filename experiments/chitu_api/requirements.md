SGLang-Omni Image Decoder 接入 ChituDiffusion Hot Switch 接口需求

1. 文档目标

本文定义 SGLang-Omni 的 LLaDA2-Uni image_decode stage 为接入 ChituDiffusion hot switch 能力，需要 ChituDiffusion 提供的嵌入式运行时接口、模型执行接口、请求生命周期接口和可观测性接口。

目标形态是：

1. SGLang-Omni 通过 StageConfig 为 image_decode 分配一组独占 GPU，并启动一个 stage world。
2. SGLang-Omni 将 LLaDA2-Uni image decoder 的 executor 描述注入 ChituDiffusion。
3. ChituDiffusion 在 stage world 内创建并管理所有 lane 通信组、请求池和 hot-switch scheduler。
4. ChituDiffusion 根据实时请求数量和 SLO，在 DP 与不同宽度的 CP/SP lane 之间动态切换。
5. 只有 stage leader 接收 SGLang-Omni 请求并返回最终图片；follower 只参与 ChituDiffusion 内部计算和通信。
本文是接口需求，不要求在 SGLang-Omni 中重写一套 ChituDiffusion pipeline，也不要求复用 SGLang Diffusion 的静态 SP runtime。

2. 参考代码版本

- SGLang-Omni：kemiao/image-decoder-optimization@37304816e6dea74fb2203f5ce81787010c08ee22（内部版本，近期会向sglang-omni pr）
- ChituDiffusion hot-switch 参考分支：feature/cp-dp-hot-switch-experiment@b988b47a1c25f1ac63a8543bd4470e38ff773d34
当前 ChituDiffusion 分支已经具备以下核心能力：
- DiffusionScheduler.plan_pool_round() 生成每轮 SchedulingPlan。
- LaneTopology 描述 lane 的 offset、width、CP 和 CFP 结构。
- initialize_lane_groups() 预创建可用宽度的通信组。
- activate_lane_topology() 在 denoise phase 边界切换当前 rank 的 active group。
- pool engine 负责 layout broadcast、request admission、latent migration、denoise、状态同步和任务回收。
现阶段的主要缺口不是 hot-switch 算法，而是这些能力仍依赖全局 DiffusionBackend、DiffusionTaskPool、chitu_generate() 和完整 T2I pipeline，缺少可被 SGLang-Omni stage 安全嵌入的实例化 API。

3. 设计边界

3.1 SGLang-Omni 负责

- 解析唯一配置入口 StageConfig。
- 确定 stage 使用的物理 GPU 列表和进程数。
- 为每个 rank 启动独立 OS process，并传入 stage-local rank、world size、device 和 rendezvous 信息。
- 维护 stage leader/follower 角色。
- 将上游 Thinker 生成的 image VQ token、网格尺寸和生成参数提交给 image decoder。
- 将 leader 返回的图片封装成当前 SGLang-Omni event/result 格式。
- 处理 pipeline coordinator、terminal response 和客户端 abort 的对接。
3.2 ChituDiffusion 负责
- 在传入的 stage world 内初始化自己的通信域。
- 预创建允许的 lane group，并保证运行时切换不创建新的 NCCL communicator。
- 维护请求队列、running lanes、step 状态、SLO 和 backpressure。
- 生成并广播每轮 lane layout。
- 负责 hot switch 前后的 latent 所有权、迁移、复制和 step 同步。
- 调用注入的 decoder executor 完成 leader-only prepare、并行 denoise 和 leader-only finalize。
- 只向 leader 暴露完成结果、失败和取消事件。
- 提供健康状态、排队信息、lane layout 和分阶段耗时指标。
3.3 第一阶段不支持
- Chitu image decoder pool 与 Thinker 或其他 stage 共享同一块物理 GPU。
- 在同一 CUDA device 上依赖 SGLang-Omni coordinator 实现跨进程计算互斥。
- 运行中创建或销毁 NCCL process group。
- 在一个 denoise kernel 或单个 attention forward 内中途切换 lane。
- 同时启用 SGLang Diffusion 静态 SP group 和 Chitu dynamic lane group。
- 修改 Thinker TP 链路。
SGLang-Omni 当前允许不同 OS process 使用同一 GPU，但没有计算流隔离、优先级仲裁或互斥调度。因而 hot-switch pool 的 GPU 在第一阶段必须独占；仅配置显存占比不能保证计算不冲突。
4. 总体结构
image_decode 仍然是一个 SGLang-Omni logical stage。ChituDiffusion 的 lanes 是该 stage 内部的动态执行布局，不应该被建模成新的 Omni stages。

5. ChituDiffusion 必须提供的接口
5.1 可实例化的 Embedded Runtime

ChituDiffusion 需要提供非全局单例的 runtime。一个 process 中至少能显式创建、启动和关闭一个实例，并且测试之间可以完整释放状态。

建议接口：

class EmbeddedDiffusionRuntime:
    @classmethod
    def create(
        cls,
        *,
        world: StageWorldSpec,
        pool: HotSwitchPoolConfig,
        executor_factory: ImageDecoderExecutorFactory,
    ) -> "EmbeddedDiffusionRuntime": ...

    def start(self) -> None: ...
    def submit(self, request: ImageDecodeRequest) -> RequestHandle: ...
    def cancel(self, request_id: str, reason: str | None = None) -> bool: ...
    def poll(self, timeout_s: float = 0.0) -> list[ImageDecodeCompletion]: ...
    def health(self) -> RuntimeHealth: ...
    def stop(self, *, graceful: bool = True, timeout_s: float = 30.0) -> None: ...

约束：

- submit() 仅在 stage leader 调用，必须非阻塞。
- 所有 rank 都必须调用 start()，并在后台进入相同的 lockstep control loop。
- follower 不依赖 SGLang-Omni payload fanout 才能推进已有任务；任务和 layout 由 Chitu runtime 内部广播。
- poll() 只在 leader 返回 completion；follower 返回空列表或禁止调用。
- runtime 内部状态不能继续存放在可跨实例污染的 class attributes 中。
- stop() 必须停止 ingress、drain/cancel 请求、同步 ranks，并按所有权销毁 Chitu 创建的 group。
现有 chitu_init()、chitu_generate() 和全局 DiffusionBackend 可以保留为 CLI 兼容层，但应改为调用这个实例化 runtime。

5.2 Stage World 注入

SGLang-Omni 已经为每个 image decoder rank 启动独立 OS process。ChituDiffusion 不应重新 spawn GPU worker，而应接受 stage-local world 描述：

@dataclass(frozen=True)
class StageWorldSpec:
    stage_name: str
    rank: int
    world_size: int
    local_device: str
    physical_device_ids: tuple[int, ...]
    leader_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 0
    process_group: object | None = None
    owns_process_group: bool = True

接口要求：

- 支持通过 master_addr/master_port/rank/world_size 创建 stage-local world。
- 可选支持传入已经初始化的 raw torch.distributed.ProcessGroup；传入时 Chitu 不得销毁外部 group。
- 所有 rank 必须校验 physical_device_ids、world size、local device 和配置完全一致。
- Chitu 内部 rank 一律是 stage-local rank，不得假设它等于整条 Omni pipeline 的 global rank。
- lane 的 GPU 编号使用 stage-local rank；物理 GPU id 只用于日志和 placement 校验。
- 初始化阶段预创建全部允许的 lane group、CFG pair group 和控制面 group。
推荐第一版优先支持 rendezvous 参数，由 Chitu 创建并拥有 stage-local group。不要尝试跨 OS process 传递 Python ProcessGroup 对象。

5.3 Hot-switch Pool 配置

@dataclass(frozen=True)
class HotSwitchPoolConfig:
    policy: str = "slo_elastic"
    allowed_lane_widths: tuple[int, ...] = (1, 2, 4)
    switch_allowed_until_step: int = 8
    phase_max_steps: int = 1
    max_inflight_requests: int = 64
    max_pending_requests: int = 256
    cfg_parallel_max: int = 1
    cost_model: str = "auto"
    online_calibration: bool = True
    default_deadline_ms: float | None = None
    lane_profiles: dict[int, "LaneParallelProfile"] = field(default_factory=dict)

@dataclass(frozen=True)
class LaneParallelProfile:
    width: int
    cp_degree: int
    cfg_degree: int = 1
    attention_mode: str = "auto"  # auto / ulysses / ring / hybrid
    ulysses_degree: int = 1
    ring_degree: int = 1

要求：

- allowed_lane_widths 中每个 width 必须整除 pool world size，并满足 Chitu canonical placement 约束。
- width=1 表示 request-level DP replica，不执行 CP collective。
- 每个 width 的 attention 形式必须显式可配置。当前实验中 width=4 应允许使用 ring_degree=4，不能把 width 固定解释为 Ulysses。
- planner 输出的 lane width、CP/CFG degree 和 attention mode 必须与 executor 实际读取的 active topology 一致。
- 配置必须来自 SGLang-Omni StageConfig；环境变量只允许测试覆盖，并输出 deprecated warning。
5.4 Partial Image Decoder Executor

LLaDA2-Uni 的 image decoder 不是 Chitu 当前完整 T2I pipeline：它不需要 text encoder，输入是 Thinker 生成的 image VQ tokens。实际计算是：

1. SigVQ：VQ token IDs 转为 4096 维 conditioning。
2. ZImage DiT：创建 noise，并执行 Turbo 8-step denoise。
3. VAE：latent 转为最终图片。
ChituDiffusion 需要允许注入 partial pipeline executor，而不是强制创建 TextEncode -> Denoise -> VAEDecode 全链路。

class ImageDecoderExecutorFactory(Protocol):
    def build(self, context: ExecutorBuildContext) -> "ImageDecoderExecutor": ...


class ImageDecoderExecutor(Protocol):
    def prepare_request(
        self,
        request: ImageDecodeRequest,
        context: LeaderContext,
    ) -> DenoiseState: ...

    def denoise_step(
        self,
        state: DenoiseState,
        *,
        step_index: int,
        topology: ActiveLaneTopology,
    ) -> DenoiseState: ...

    def finalize_request(
        self,
        state: DenoiseState,
        context: LeaderContext,
    ) -> ImageDecodeResult: ...

    def abort_request(self, request_id: str) -> None: ...
    def close(self) -> None: ...

执行语义：

- build() 在每个 rank 调用；DiT 权重加载到每个 rank。
- SigVQ 和 VAE 允许只在 leader 加载和执行，follower 对应组件可以为 None。
- prepare_request() 只在 leader 做 VQ embedding、noise/sampler 初始化，结果由 Chitu runtime 分发到首个 lane。
- denoise_step() 在 lane 所有 ranks 调用，并从 ActiveLaneTopology 获取当前 group 和序列切分信息。
- finalize_request() 只在完成请求的 lane leader 或 stage leader 执行；若两者不同，Chitu 先把最终 latent 迁移到 stage leader。
- executor 不负责 lane 选择、任务队列、group 创建和跨 lane latent 迁移。
- Chitu runtime 不解析 SGLang-Omni StagePayload；两侧只通过 typed request/result 交互。
Chitu 现有 DiffusionRuntimeAdapter 已经有 prepare_denoise()、denoise_step() 和 decode_latents() 等相近能力。建议扩展该抽象支持 external conditioning 和 leader-only pre/post，而不是创建一套无继承关系的新模型接口。

5.5 动态 Topology 视图

executor 每次 denoise step 必须读取本轮 topology，不能在模型初始化时缓存固定 sp_size 或固定 group：

@dataclass(frozen=True)
class ActiveLaneTopology:
    lane_id: str
    ranks: tuple[int, ...]
    rank: int
    rank_in_lane: int
    width: int
    cp_degree: int
    cfg_degree: int
    ulysses_degree: int
    ring_degree: int
    cp_group: object | None
    ring_group: object | None
    cfg_group: object | None

接口要求：

- topology 只在 denoise step/phase 边界切换。
- 同一 lane 的所有 rank 在进入 forward 前看到完全相同的 topology version。
- width 从 4 切到 1 时，executor 不执行遗留的 width=4 collective。
- width 从 1 切到 4 前，Chitu 必须保证目标 ranks 都持有一致的 canonical latent、conditioning 和 step index。
- group 必须从预热 registry 中选择；active switch 应是指针/上下文切换，不得触发 new_group()。
这也是现有 SGLang ZImage _run_model_sp() 不能直接无修改用于 hot switch 的原因：它读取的是 SGLang Diffusion 静态 SP group，并假设固定 shard 数。若使用 Chitu 自有 ZImage backend，则 shard/all-gather 必须由 ActiveLaneTopology 驱动。

5.6 请求与结果接口

@dataclass(frozen=True)
class ImageDecodeRequest:
    request_id: str
    vq_token_ids: tuple[int, ...]
    grid_h: int
    grid_w: int
    seed: int
    num_steps: int = 8
    decode_mode: str = "decoder-turbo"
    resolution_multiplier: int = 2
    deadline_ms: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageDecodeCompletion:
    request_id: str
    status: str  # completed / cancelled / failed
    result: ImageDecodeResult | None
    error: StructuredRuntimeError | None
    metrics: RequestRuntimeMetrics

要求：

- request_id 在一个 runtime 实例内唯一。
- submit() 对 queue full、duplicate id、invalid shape 返回结构化错误。
- completion exactly once；取消与正常完成竞争时必须有确定状态。
- seed、sampler、step index 和 conditioning 必须属于 request state，不能使用会被并发请求改写的全局 cursor。
- 结果建议返回 CPU tensor、PIL image 或 raw RGB bytes；PNG/base64 和 Omni event 封装仍由 SGLang-Omni 负责。
- SGLang-Omni 不应直接访问 DiffusionTaskPool.pool 或轮询 Chitu 内部 task 对象。
5.7 Backpressure、取消与错误传播

Chitu runtime 必须提供：

- max_pending_requests 和 max_inflight_requests 限制。
- 非阻塞 admission 结果：accepted、queue_full、invalid、duplicate。
- 根据 request_id 取消 pending 或 running request。
- running request 只在安全 step boundary 停止，并清理 lane-local state。
- 任一 rank 发生不可恢复错误时，将相同 fatal state 广播到 stage world，避免其他 rank 永久阻塞在 collective。
- leader 将 request-scoped error 转为 completion；runtime fatal error 触发 scheduler 崩溃，由 SGLang-Omni StageGroup fail-all。
5.8 可观测性接口

@dataclass(frozen=True)
class RuntimeHealth:
    state: str
    rank: int
    queue_depth: int
    inflight: int
    active_lanes: tuple[LaneSnapshot, ...]
    last_plan_version: int
    fatal_error: str | None

至少暴露以下指标：

- request arrival、admission、first scheduled、finish、cancel 时间。
- queue delay、E2E latency 和 decoder service time。
- 每轮 lane layout、width、request id、planned steps 和 switch 原因。
- plan、broadcast、migration、denoise、barrier、replication、leader finalize 耗时。
- hot-switch 次数、迁移字节、placement reuse/change。
- 每个 width 的实测 step latency，供 online calibration 使用。
- 当前 queue depth、inflight、idle GPU 数和 fatal state。
指标应通过 callback/snapshot API 提供，日志只作为辅助；SGLang-Omni profiler 不应解析 Chitu 文本日志。

6. SGLang-Omni 侧适配要求

6.1 StageConfig 是唯一正式配置入口

建议在 ParallelismConfig 下增加 typed Chitu pool 配置，同时保留 sp 作为 stage world/process 数量：

- name: image_decode
  process: image_decoder
  factory: sglang_omni.models.llada2_uni.stages.create_chitu_image_decode_executor
  gpu: [4, 5, 6, 7]
  parallelism:
    sp: 4
    chitu_pool:
      enabled: true
      policy: slo_elastic
      allowed_lane_widths: [1, 2, 4]
      switch_allowed_until_step: 8
      phase_max_steps: 1
      lane_profiles:
        1: {attention_mode: auto, ulysses_degree: 1, ring_degree: 1}
        2: {attention_mode: auto, ulysses_degree: 2, ring_degree: 1}
        4: {attention_mode: ring, ulysses_degree: 1, ring_degree: 4}
  factory_args:
    backend: chitu
    decode_mode: decoder-turbo
    num_steps: 8
  terminal: true

语义约定：启用 chitu_pool 时，parallelism.sp 表示 stage pool world size 和最大 lane 宽度，不表示运行期间始终以固定 SP=4 执行。

环境变量只用于实验覆盖；正式 serving 不得要求设置 CHITU_* 或 SGLANG_OMNI_* 环境变量才能表达核心并行配置。

6.2 专用 Scheduler 必须继承现有抽象

新增 ChituImageDecoderScheduler(SimpleScheduler)，复用 inbox、outbox、abort 和 Stage contract，但覆盖串行 compute_fn 驱动方式：

- leader loop 将 new_request 转为 runtime.submit()，不等待单个请求完成，因此可以持续接收并发请求。
- leader completion pump 调用 runtime.poll()，并向 outbox 发送对应 result/error。
- follower scheduler 不接收外部 payload，只运行 Chitu runtime worker/control loop。
- stop() 先停止 ingress，再调用 runtime graceful stop。
不能直接把 submit_and_wait() 包成现有 SimpleScheduler(compute_fn)：这会让 scheduler 一次阻塞一个请求，Chitu 永远看不到足够的并发请求，也就无法发挥 DP/CP hot switch。

6.3 Leader/Follower 行为

- stage leader 是唯一拥有 ZMQ control plane、relay reader 和 coordinator output 的 rank。
- follower 不读取上游 relay，不生成 PNG/base64，不向 coordinator 发 completion。
- follower 必须在 stage ready 前完成模型加载和所有 lane group 预热。
- StageGroup 只有在所有 Chitu ranks 都 ready 后才对外服务。
- leader 收到 request abort 后调用 Chitu cancel(request_id)；Chitu 内部广播取消状态。
6.4 Terminal 输出

当前 image_decode 是 terminal stage，因此无需新增下游 relay。Chitu 返回 stage leader 后，现有 LLaDA2-Uni stage adapter 负责：

1. 将图片编码为 PNG。
2. 构造 image_final event。
3. 向 coordinator 返回 terminal result。
若未来 image decoder 变为非 terminal stage，仍只允许 leader 使用 SGLang-Omni relay 向下一 stage 发送结果。

7. 生命周期与时序

sequenceDiagram
    participant O as "Omni StageGroup"
    participant L as "Image decoder leader"
    participant F as "Image decoder followers"
    participant C as "Chitu runtime"
    participant X as "LLaDA executor"

    O->>L: "spawn(rank=0, world=4, gpu=4)"
    O->>F: "spawn(rank=1..3, world=4, gpu=5..7)"
    L->>C: "create(world, pool, executor_factory)"
    F->>C: "create(world, pool, executor_factory)"
    C->>C: "initialize world and prewarm lane groups"
    C->>X: "build executor on every rank"
    C-->>O: "all ranks ready"
    O->>L: "StagePayload with VQ tokens"
    L->>C: "submit(ImageDecodeRequest)"
    C->>C: "plan and broadcast lane layout"
    C->>X: "leader prepare_request"
    loop "At each safe denoise phase"
        C->>C: "activate topology and migrate latent if needed"
        C->>X: "denoise_step(state, active topology)"
    end
    C->>X: "leader finalize_request"
    C-->>L: "ImageDecodeCompletion"
    L-->>O: "terminal image result"

8. 正确性不变量

实现必须始终满足：

1. 同一个 request 的 current_step 在所有持有该 request state 的 rank 上一致。
2. 一个 lane 内所有 rank 在一次 forward 中使用同一个 request、step 和 topology version。
3. hot switch 只发生在 executor 声明的安全边界。
4. 切换后首个 forward 前，目标 lane 已经持有一致的 latent 和 conditioning。
5. 每个 request 最多返回一次 terminal completion。
6. follower 永远不向 Omni coordinator 返回业务结果。
7. runtime stop 或 fatal error 后，不允许 rank 留在未配对 collective 中。
8. 相同 token、seed、steps 和 scheduler 下，width=1/2/4 的生成主体和质量必须通过基准验证。

9.  结论

最合适的集成方式不是让 SGLang-Omni 操作 Chitu 的 lane 或 process group，也不是让 Chitu 接管整条 Omni pipeline。推荐边界是：

- Omni 拥有 logical stage、GPU placement、进程和外部请求路由。
- Chitu 拥有 image decoder stage 内部的 request pool、lane topology、通信组和 hot-switch 调度。
- LLaDA2-Uni 通过 Chitu DiffusionRuntimeAdapter 的扩展接口提供 partial decoder executor。
- Omni 通过继承 SimpleScheduler 的专用 scheduler 完成 inbox/outbox 与 Chitu submit/completion 的桥接。
这个边界可以保留两边现有抽象，同时避免 SGLang static SP group 与 Chitu dynamic group 同时成为并行配置真源。
