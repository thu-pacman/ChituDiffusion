# 文献笔记：LLM Serving 中的动态并行

这份笔记记录与 DiT CP/DP 热切换最接近的 LLM serving 工作。文献中通常不直接使用 "CP/DP hot switch" 这个说法，而更常见的是 dynamic parallelism、elastic parallelism、on-the-fly switching、model re-sharding 或 cross-instance transformation。

## 与 DiT 问题的映射

我们考虑的 DiT feature：

```text
给定 N 张 GPU 和动态到达的请求，在低负载时使用 CP/SP 降低单请求延迟；
当更多请求到达、吞吐和排队延迟更重要时，切换为 DP-style 的独立副本。
```

对应到 LLM 的表述：

```text
TP/CP/SP 可以提升单请求延迟或上下文容量，DP 可以提升多请求吞吐。
静态部署只能选择 tradeoff 上的一个点。动态 serving 试图在安全点改变并行 layout，
同时尽量少迁移请求状态。
```

对 DiT 最可复用的概念：

- state layout invariance；
- safe transition point；
- migration-aware scheduling；
- elastic parallel degree；
- 比较剩余工作量与切换开销的成本模型。

## Shift Parallelism

论文：

- Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads
- arXiv: <https://arxiv.org/abs/2509.16495>
- ACM DL: <https://dl.acm.org/doi/10.1145/3779212.3790219>

核心 motivation：

- TP 可以降低延迟，但通信会伤害整体吞吐。
- DP 可以提升吞吐，但无法降低单请求延迟。
- 论文指出 TP 和 DP 很难直接优雅结合，因为两者的 KV cache layout 不一致。
- 它引入 SP 作为一种 DP-like 的高吞吐模式，并利用 KV cache invariance 在 TP 和 SP 之间动态切换。

核心方法：

- 将 sequence parallelism 适配到 inference。
- 根据 workload pressure 在 TP 和 SP 之间切换。
- 通过选择 KV cache layout 兼容的并行组合来保持请求状态有效。
- 在动态生产 trace 和合成 workload 上评估。

对我们有用的问题定义：

- 可行性的核心不只是“能不能切换执行 kernel”，而是“请求状态能不能跨 layout 保持有效”。
- 对 DiT 而言，类似状态包括 latent tensor、step scheduler state，以及可选的 FlexCache 类加速/cache 状态。

对 DiT 的启发：

- 如果 CP/SP sharding 可以在 step 边界廉价地 materialize 成完整单卡状态，DiT 可能比 LLM 更容易，因为它没有长期存在的 token KV cache。
- 如果 FlexCache state 按 sequence 或 block sharding，DiT 问题就会重新接近 LLM 的 KV layout 兼容性问题。

## FLYING SERVING

论文：

- FLYING SERVING: On-the-Fly Parallelism Switching for Large Language Model Serving
- arXiv: <https://arxiv.org/abs/2602.22593>

核心 motivation：

- 生产 LLM traffic 在非平稳到达下同时需要高吞吐、低延迟和足够的上下文容量。
- 静态 DP/TP 部署无法在 burst、优先级请求或长上下文请求之间自适应，除非付出 disruptive reconfiguration 的代价。

核心方法：

- 在基于 vLLM 的系统中实现 online DP-TP switching，且不重启 engine worker。
- 使用 model weights manager，让 worker 可以按需暴露 TP shard view。
- 使用 KV cache adaptor，让请求 KV state 在 DP/TP layout 间保持有效。
- 预初始化 communicator pool，避免在切换时支付 setup cost。
- 通过 scheduler 协调 transition，避免 worker execution skew 下的 deadlock。

对我们有用的问题定义：

- 热切换是完整系统问题，而不只是 kernel 选择：
  - weight/state virtualization；
  - cache/state adaptation；
  - communicator readiness；
  - scheduler safe point；
  - skewed worker progress 下的正确性。

对 DiT 的启发：

- 第一个 DiT 原型应避免在关键路径动态创建 process group。预创建 group 或固定两卡 topology 更现实。
- DiT step 边界是最可能的安全切换点；layer-level 切换第一阶段大概率不值得。

## LoongServe

论文：

- LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism
- arXiv: <https://arxiv.org/abs/2404.09526>
- SOSP 2024

核心 motivation：

- Long-context LLM serving 在不同请求之间、以及同一请求的不同阶段之间资源需求差异很大。
- 静态并行会浪费资源，因为短请求/早期/晚期阶段可能需要不同的并行 degree。

核心方法：

- 提出 Elastic Sequence Parallelism。
- 实时调整 sequence parallelism 的 degree。
- 降低 KV cache migration overhead。
- 将 partial decoding communication 与 computation overlap。
- 降低实例间 KV cache fragmentation。

对我们有用的问题定义：

- 问题不只是两个命名模式之间的切换，而是为每个请求、每个阶段选择弹性并行 degree。
- scheduler 应该理解 phase-specific resource demand，而不是只维护一个全局 deployment mode。

对 DiT 的启发：

- DiT denoising step 数固定，但不同 step 的 cache 价值和计算特征可能不同。
- 可以定义 early/mid/late denoising phase，只允许在剩余工作足以覆盖迁移成本的阶段热切换。

## Gyges

论文：

- Gyges: Dynamic Cross-Instance Parallelism Transformation for Efficient LLM Inference
- arXiv HTML: <https://arxiv.org/html/2509.19729v1>

核心 motivation：

- 许多请求可以由小实例或副本高效服务，而偶发长上下文请求需要聚合多张 GPU 才能容纳。
- 静态 instance sizing 要么牺牲吞吐，要么牺牲长上下文容量。

核心方法：

- 在多个小 inference instance 和更大的并行 instance 之间 transformation。
- 让 scheduling 感知 transformation，避免 migration 和 reshaping 成本吞掉收益。
- 重点是 dynamic cross-instance parallelism，而不仅是 instance 内部的 kernel 改变。

对我们有用的问题定义：

- 核心对象是 resource layout 之间的 "instance transformation"。
- 评估必须同时包含性能收益和 transformation cost。

对 DiT 的启发：

- CP/SP -> DP 热切换可以被表述为 cross-instance transformation：一个两卡 instance 服务请求 A，变成两个单卡 instance 分别服务 A 和 B。
- 这个 framing 比把切换看作局部 kernel 选择更清晰。

## Seesaw

论文：

- Seesaw: High-throughput LLM Inference via Model Re-sharding
- arXiv: <https://arxiv.org/abs/2503.06433>

核心 motivation：

- LLM prefill 和 decode 具有不同的计算、通信和内存特征。
- 单一静态 sharding 选择无法同时适配两个阶段。

核心方法：

- 在阶段之间重新 sharding model。
- 通过 scheduling 降低 transition overhead，避免过于频繁的 re-shard。

对我们有用的问题定义：

- 在语义明确的 phase boundary 切换，而不是在任意时间点切换。
- transition-minimizing scheduling 是系统的一部分，不应事后补丁式处理。

对 DiT 的启发：

- DiT 没有 prefill/decode boundary，所以需要从 denoising step 或 early/mid/late sampling phase 中构造安全边界。
- 固定 checkpoint step，例如 step 1/2/4/8 或采样前 20%-30%，比“请求一到就切”更容易建模。

## ReMP

论文：

- ReMP: Runtime Model-Parallelism Reconfiguration
- arXiv: <https://arxiv.org/abs/2606.18741>

核心 motivation：

- Model-parallel deployment 往往需要随负载和资源可用性改变 TP/PP layout。
- 重启或重新部署对在线自适应来说太慢。

核心方法：

- 在 runtime 重新配置 model parallelism。
- 将 worker、model state 和 communication topology 解耦到足以改变 runtime layout。

对我们有用的问题定义：

- 将 parallelism layout 视为可变的 runtime object。
- 将 reconfiguration cost 与 steady-state execution cost 分开建模。

对 DiT 的启发：

- 干净的 DiT 实现不应该把 CP/DP 决策埋在 model code 中。
- layout transition 应由 scheduler/runtime 拥有。

## AlpaServe

论文：

- AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving
- OSDI 2023
- Project: <https://alpa.ai/serve>

核心 motivation：

- 多模型 serving 使用固定并行策略时，可能浪费内存和计算。
- 统计复用可以在维持 latency target 的同时提升集群利用率。

核心方法：

- 搜索 model-parallel placement。
- 用 batching 和 placement decision 复用 serving workload。

对我们有用的问题定义：

- 即使没有 per-request hot switching，placement/search 也可以定义强 baseline。
- 静态或 admission-time placement 可能以更低复杂度获得大部分收益。

对 DiT 的启发：

- hot-switch 方案应该与 admission control 对比：短暂等待另一个请求后再启动 DP；或者只在队列为空且仍有 early-switch window 时启动 CP/SP。

## ChituDiffusion 的候选问题定义

```text
我们研究 DiT serving 中的动态 CP/SP-to-DP transformation。
在低负载下，请求可以使用多张 GPU 通过 sequence/context parallelism 降低单请求延迟；
在 bursty load 下，这些 GPU 可能更适合作为独立的数据并行副本，服务不同请求。
目标是在 denoising step 边界决定是否以及何时将一个 in-flight CP/SP 请求
转换为 DP-style 多请求 layout，同时考虑 state migration、communicator readiness、
cache compatibility 和 tail-latency risk。
```

## 成本模型骨架

记：

- `S_remain`：当前 active request 的剩余 denoising step；
- `T_sp2`：一个请求在两卡 CP/SP 上的逐 step 延迟；
- `T_dp1`：一个请求在单卡上的逐 step 延迟；
- `C_switch`：layout transformation 的一次性成本；
- `Q_wait`：如果不切换，新到达请求的预期排队延迟；
- `C_tail`：active request tail latency 增加带来的惩罚。

当下式成立时，切换可能值得：

```text
Q_wait_saved + throughput_gain_over_remaining_steps
    >
C_switch + C_tail + cache_or_graph_penalty
```

第一个模拟器可以 sweep `C_switch`，不必一开始精确测量。

## 首批对比 baseline

- `static_dp`：两个单卡副本，从不使用 CP/SP。
- `static_sp`：每次一个两卡 CP/SP 请求，其余请求排队。
- `admission_control`：如果队列为空，等待一个很小的 batching window，再决定 DP 或 CP/SP。
- `early_hot_switch`：立即启动 CP/SP；如果另一个请求在 early-step threshold 前到达，则切换到 DP-style 副本。
- `oracle`：使用未来到达信息选择最佳 layout，作为上界。
