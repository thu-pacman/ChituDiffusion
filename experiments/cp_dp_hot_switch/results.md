# CP/DP Hotswitch 离线模拟结果

## M1 解耦成本模型（worker roofline + 通信）

吞吐连续 batching 引擎计划的 M1 已完成，完整报告见
[`m1_cost_model_report.md`](m1_cost_model_report.md)（标定数据 [`cost_model.json`](cost_model.json)）。

要点：H20/bf16/Z-Image 上 **B=1 已 compute-bound**（B2≈1.96×、B4≈3.9× 近线性，
`batch_saturation` 拐点=1），同形状 batching 无免费吞吐；held-out 插值误差 max 4.82% / mean 2.56%
（PASS <10%）。主吞吐杠杆是 **SP degree 右尺寸化**（1024² k=4→188ms，约 3.2×，comm <2%/step）
与 **FlexCache 减步数**，而非同形状 batching。模拟器 `simulate.py` 已消费该模型。


复现：

```bash
python3 experiments/cp_dp_hot_switch/run_experiment.py --output-dir experiments/cp_dp_hot_switch/out
```

原始输出写入 `out/summary.md` 与 `out/summary.json`；`out/` 是可再生成实验输出，已由 `.gitignore`
覆盖，不作为提交资产。

2026-07-06 更新：`static_dp`/DP-like 策略现在使用 per-GPU slot 事件模型；同一 request 的 step
不会重叠执行，`gpu_busy_ratio` 恢复到物理上限 `<=1`。旧版 `mixed_long_short static_dp`
曾出现 `gpu_busy=1.253`，该数值已废弃。

## Trace 与延迟来源

图像生成 serving 没有公开且带分辨率 / DiT 序列长的请求到达 trace（常见的 Azure LLM inference
trace 只有到达时间戳，缺少 DiT shape）。因此这里**到达序列是合成的**，但**每步延迟锚定到本仓库的真实实测**：

- `traces/profiles.json` 的 `dp1 / cp2 / cp4` 逐步延迟来自 `ChituBench/result_parallel_dit.md`
  （flux1-dev、flux2-klein-4B、Qwen-Image）与 `experiments/fast_cp/worklog.md`。
- 例如 Qwen-Image 1328²：单卡 138.819s、纯 CP2 (`cfp1up2`) 80.935s（50 步）→ CP2 单请求 1.71x，
  与实测 1.715x 一致。

四个合成场景（`traces/*.json`）：

| 场景 | 负载 | GPU | 目的 |
| --- | --- | ---: | --- |
| `single_long` | 1 个重请求 (qwen 1328²) | 2 | 低负载下 CP 的单请求延迟收益 |
| `short_burst` | 8 个中请求 (flux1) burst | 2 | 中等并发下 CP2≈2×DP 的临界区 |
| `mixed_long_short` | 1 重 + 4 中，交错到达 | 2 | 异构负载下的排队 / 公平性 |
| `heavy_burst_4gpu` | 10 个中请求持续 burst | 4 | 多卡下过度切分 (cp2) 的利用率损失 |

## 关键结果

| 场景 | 最优吞吐策略 | thrpt 提升 vs static_cp | 备注 |
| --- | --- | ---: | --- |
| single_long | static_cp | — (CP 单请求 1.71x 更快: 80.9s vs 138.8s) | DP 只用半个节点 (gpu_busy 0.5) |
| short_burst | static_dp / early_hot_switch | +8% (0.052 vs 0.048) | 近似平局；CP 均值更优，DP/early 尾延迟更优 |
| mixed_long_short | static_dp / shape_aware_ratio / early_hot_switch 近似同吞吐 | +7% (0.032 vs 0.030) | `shape_aware_ratio` 均值/排队最好；CP 被重请求 head-of-line 阻塞 |
| heavy_burst_4gpu | static_dp / shape_aware_ratio / early_hot_switch 近似同吞吐 | **+81%** (0.087 vs 0.048) | cp2 在 4 卡上浪费一半 (gpu_busy 0.5 vs 0.83) |

## 收益区间结论

1. **存在明确的 CP↔DP 交叉点，这正是自适应调度的价值来源。**
   - 队列深度 ≈ 1（低负载）：CP/SP 胜出，单请求延迟 **1.71x**（Qwen 1328²，实测锚定）。
   - 队列深度 ≥ GPU 数 / cp_size（高负载）：DP 副本胜出，4 卡持续 burst 下吞吐 **~1.8x**、
     均值延迟 1.67x，且把 GPU 利用率从 0.50 拉到 0.83。
   - 因此一个"空队列走 CP、请求堆积走 DP"的负载自适应调度器，理论上可同时拿到低负载 1.7x 延迟
     与高负载 1.8x 吞吐——这是固定部署（只选 CP 或只选 DP）拿不到的。

2. **过度切分的代价被直接量化。** `heavy_burst_4gpu` 里固定 cp2 在 4 卡上只用两张卡
   （gpu_busy 0.50），吞吐仅为 4 路 DP 的 55%。这与 FastCP 的观察一致：序列并行的通信开销 +
   计算打不满，使得高并发下 CP 的效率被 DP 副本超越。

3. **中等并发是临界区，收益对模型很敏感。** flux1 的 CP2 加速比 1.82 ≈ DP 并发度 2，所以
   `short_burst` 近似平局（DP/early 吞吐 +8%、尾延迟更低；CP 均值略优）。CP2 加速比越低（如 Qwen 1.71）、
   序列越长、卡数越多，天平越向 DP 倾斜。

4. **In-flight 热切换的收益主要在排队 / 尾延迟，而非原始吞吐（在"CP 便宜"的成本模型下）。**
   - `mixed_long_short`：`shape_aware_ratio`（重请求留 CP、短请求走 DP 副本）把平均排队延迟从
     88.3s 降到 **15.0s（-83%）**，均值延迟也从 121s 降到 76.8s；吞吐与 static_dp/early
     近似同档（0.032 req/s）。
   - `heavy_burst_4gpu`：`early_hot_switch` 把排队延迟从 static_cp 的 93.3s 降到 **30.2s（-68%）**，
     与 static_dp 的 30.4s 基本持平；switch 次数收敛为 1 次，符合“一次 CP->DP 后按 DP slot 执行”的模型。
   - 因此 `early_hot_switch / oracle` 的原始吞吐仍没有实质超过对应的最优静态 DP-like 策略：在 CP 便宜、
     请求步数不长（几十步）、到达可预测时，**"按负载选对静态 layout"（admission-time placement）已经吃掉大部分收益**。
     step 中途切换更多是缓解 head-of-line 阻塞。切换的边际价值会随 CP 变贵、到达更不可预测、
     单请求步数更多（视频、超高分辨率）而上升。

## 对实现的启示

- 第一优先级是**负载感知的静态 layout 选择 + admission control**：队列空 → CP 起单请求；
  队列深 → 起多路 DP 副本。这条路径低复杂度、收益最确定（低负载 1.7x 延迟 / 高负载 1.8x 吞吐）。
- `shape_aware_ratio` 是异构负载下最稳的单一策略（排队延迟最低、吞吐贴近 DP、单请求贴近 CP），
  适合作为默认在线策略。
- Step 边界热切换值得作为二级优化，重点针对 head-of-line 阻塞与尾延迟；应先把 communicator 预建、
  latent 迁移做廉价（本实验成本模型：switch_total≈45ms，其中 graph recapture / cache 迁移置 0）。

## 局限

- 模拟器的 CP 路径固定为 2 卡一组；4 卡场景下 CP 只用其中 2 卡，尚未建模 cp4 组或多 CP 组。
- 每步延迟用常量均值，未建模 warmup / 分辨率内的 step 方差、VAE / TextEncode 阶段、以及 batching。
- 到达为手工合成的确定序列；未做泊松 / 真实 QPS 分布的敏感性 sweep。以上均为下一步。
