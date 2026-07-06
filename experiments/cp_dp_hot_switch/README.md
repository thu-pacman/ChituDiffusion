# CP/DP 热切换实验

本目录用于探索 DiT 推理中的动态热切换：在单请求时使用上下文/序列并行降低延迟，在多请求到达时切换为数据并行式的多请求执行。

## 问题

当一个 DiT 请求已经用两卡序列/上下文并行执行时，如果第二个请求在推理中途到达，系统是否可以在 denoising step 边界切换为两个单卡数据并行副本，让每个请求各占一张卡，并提升服务效率？

简化描述：

```text
早期阶段：    请求 A 使用 GPU0+GPU1 做 CP/SP，以降低单请求延迟
新请求到达：  请求 B 在 A 还有剩余 denoising step 时到达
切换：        GPU0 继续/完成 A，GPU1 启动 B，形成 DP 式副本
目标：        降低排队延迟，提高吞吐，同时不明显伤害尾延迟
```

## 工作假设

这个 feature 大概率只在受限策略下有价值：

- 只在 denoising step 边界切换；
- 只在采样早期切换，确保剩余 step 足够多；
- 只有当预期排队收益超过状态迁移、通信器、graph capture、cache layout 等成本时才切换；
- 先从无 FlexCache 或 request-local cache state 的场景开始，再在基础成本模型明确后扩展到 FlexCache-aware 切换。

另一个核心变量是请求异构性：不同请求可能分辨率不同，反映为 DiT token/patch 序列长不同。因此 hot switch 不应只被定义成固定的“两卡 CP -> 两路 DP”，而应把 CP 和 DP 的比例也纳入调度变量。长序列请求更可能需要更多 CP/SP 资源来降低 step latency 或避免显存压力，短序列请求则更适合作为 DP replica 填补空闲 GPU。

第一个实验应该是调度器/成本模型研究，而不是直接做 runtime 实现。

## 初始产物

- [literature_notes.md](literature_notes.md)：记录定义过类似动态并行问题的 LLM serving 论文。
- [simulator_plan.md](simulator_plan.md)：离线模拟器的输入、策略、指标和下一步代码计划。
- [results.md](results.md)：离线模拟、静态 DP/SP baseline、关键结论和局限。
- [WORKLOG.md](WORKLOG.md)：按日期记录上下文同步、bug 修复、trace 调整和 cleanup 口径。

## 当前可复现资产

- `simulate.py` / `run_experiment.py`：抽象 CP/DP hot-switch 事件模拟器。
- `run_static_baselines.py`：固定 DP/SP 部署 baseline，输出 throughput、latency、queue、busy 和固定 SLO attainment。
- `cost_model.json` + `chitu_diffusion/runtime/cost_model.py`：Z-Image worker roofline + communication 成本模型。
- `traces/*.json`：小型抽象场景，用于模拟器策略比较。
- `traces/serve/realistic_sd3_mixed.json`：推荐的默认 serving 策略评估 trace，使用 SD3-like 尺寸组合和非均匀到达。
- `traces/serve/staggered_mixed.json`：小型可视化 trace，方便检查 GPU lane 和 request timeline。
- `traces/serve/poisson_*_r*.json`：低/中/近饱和 Poisson 负载 sweep。

生成输出不提交：`experiments/cp_dp_hot_switch/out/`、raw stage profile JSON、raw SD3 trace 和均匀抽样 SD3 trace
都由 `.gitignore` 覆盖。需要查看图表时重新运行：

```bash
python3 experiments/cp_dp_hot_switch/run_static_baselines.py --output-dir experiments/cp_dp_hot_switch/out
```

需要重新跑抽象策略 sweep：

```bash
python3 experiments/cp_dp_hot_switch/run_experiment.py --output-dir experiments/cp_dp_hot_switch/out
```

## 候选实验计划

1. 基于 ChituBench timing 数据构建离线模拟器：
   - 单卡 DP-style 执行的逐 step 延迟；
   - 两卡 CP/SP 执行的逐 step 延迟；
   - 合成或真实记录的请求到达 trace；
   - 将切换成本作为参数。
2. 评估调度策略：
   - 静态单卡 DP 副本；
   - 静态两卡 CP/SP，一次只服务一个请求；
   - early-window CP/SP -> DP 热切换；
   - 基于剩余 step 和队列长度的阈值策略；
   - 基于请求序列长/分辨率的 CP/DP 比例选择策略。
3. 衡量 serving 指标：
   - 平均延迟；
   - P95/P99 延迟；
   - 排队延迟；
   - GPU 利用率代理指标；
   - 切换次数和浪费/迁移的工作量。
4. 只有在模拟器证明存在有价值区域后，再做 runtime 原型：
   - DiT step 边界作为安全切换点；
   - latent/state 重新分布；
   - distributed group 处理；
   - cache-state 兼容性检查。

## 开放设计点

- 第一个目标模型应该是 Flux、Qwen-Image 还是 Wan？
- 第一个切换目标是 CP/SP -> 独立单卡请求，还是降低 CP/SP degree 后让两个请求仍然部分并行？
- 当请求分辨率不同、序列长不同的时候，如何为每个请求分配 CP/SP degree，并决定集群整体 DP replica 数？
- 在 step 边界必须迁移的最小状态是什么？
- 现有 attention/parallel backend 是否暴露了足够 hook，可以在不重启整个 engine 的情况下切换？
- FlexCache state 会如何改变成本模型？
