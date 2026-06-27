# FreeCache Planner 方法说明

> 本文是方法定型总结（区别于流水账式 `WORKLOG.md`）。记录截至 2026-06-24 的核心机理、方法论与已定型配方。
> 实验细节、被推翻的单点结论、逐次 run 记录见 `WORKLOG.md`。

## 一、问题与建模

FreeCache/MeanCache 风格的 **step-level 缓存**：扩散去噪 50 步里并非每步都重算 DiT，而是：

- **fresh**：真实 forward；
- **reuse**：用缓存的 velocity + sigma-JVP（任意阶）外推近似。

建模为**离散搜索**：

- 每个 anchor 转移 `t → s` 是一条 **edge**，带 order/span 标签；
- 目标：在预算 `B`（如 25 fresh / 50 step）下选一条 fresh schedule + 每段 reuse 的 order，使最终图像退化最小；
- 关键假设：若 policy 对 prompt/seed 基本无关，就能用**离线 probe** 把搜索变得可解。

## 二、已夯实的机理认知（多模型 / 多手段交叉验证）

1. **velocity 冗余呈 U 型**：早期高、中段低、尾部回升。Wan/Qwen/Flux 三模型 full-compute probe 一致 → 决定「早密晚疏 + 护尾」的 fresh 分布。
2. **cost 必须定义在 velocity 空间，不是 latent 空间**：latent drift 均匀铺开会垫底（19.85），velocity 类全面更高。
3. **模型差异大不能照搬**：Qwen = 好相位 + ≥1 阶 JVP；Wan = 长 warmup + 0 阶 + 护尾；Flux = 宽容低敏感。
4. **Qwen 上 1 阶 JVP 是关键**：0 阶仅 23.9，1 阶 30.3，2 阶 ≈ 1 阶；更高阶（>1）无显著收益。
5. **warmup-heavy + 平滑小 gap > 均匀铺开 / 长 gap**。

## 三、方法论纪律（最值钱）

- **单 seed 不可信**：seed 间 std≈2.4 dB > 策略间差异。评估必须 ≥3 seed 取均值 + 方差。
- **full-compute 单步 latent MSE 不能预测闭环质量**（TracePlanner V1 GPU beam search 失败的根因：预测分高但真实回放 PSNR 22.8，远不如手调）。
- **planner 产出的 policy 必须经真实 FreeCache 回放验证**，不可只信预测分数。

## 四、主线方法：离线 edge-DP planner

放弃在线 GPU beam search（V1 失败），转向 MeanCache 式离线流程：

1. **probe**：跑一次 full-compute，存 step-level velocity 向量（`steptrace` 策略，只存标量/向量，不存大 tensor）。
2. **建图 + edge cost**：每个 `t→s` 转移当作一条边，cost = 该段 reuse 用 sigma-JVP 近似的 velocity 误差。
3. **edge-DP 最短路**：在预算 `B` 下解带约束的最短路 → fresh schedule + 每段 order。
4. **真实回放验证**：用 `forced_compute_steps` / `forced_reuse_orders` 精确注入 schedule，跑真实 FreeCache 多 seed × 多 prompt 回放。

### 用 groundtruth 反选 cost（关键突破）

把已验证的人工 schedule `hqphase` 当**标签**，反过来挑 cost：问题从「planner 产出好不好」变成「哪个 cost 能把 hqphase 排成（近）最优路径」。诊断脚本 `experiments/freecache/tools/edge_dp_vs_hqphase.py`（纯 CPU）。

- **peak-only cost 全错**：hqphase 代价是 DP 最优的 2.7–7 倍 —— peak cost 只压单条最差边，放任长 reuse run（gap7–8 脆弱 schedule），与 hqphase 偏好相反。
- **整合型 sum_* cost 才对**：`sum_dir_err` / `sum_vel_rel_mse` / `sum_chanmax_dir_err`（沿 gap 累加误差）下 hqphase ratio≈1.0–1.2，即 hqphase 就是最优（`sum_chanmax_dir_err` g1 ratio=1.001）。
- **但 cost 单独退化（flat valley）**：等代价路径族里 gap 结构差异巨大（最优 Jaccard 仅 0.43、max_gap 用到 8）→ 这是自动法方差大的根因。
- **缺的一味 = 结构先验 max_gap=4**：加上后 `sum_dir_err` / `sum_vel_rel_mse` 收敛到同一条 schedule（autohq），与 hqphase 几乎同构。

## 五、定型配方

```
edge-DP 整合型 velocity 误差 cost（sum_dir_err / sum_vel_rel_mse, gamma=1）
  + max_gap=4 结构约束
```

- **cost** 负责「该在哪密集计算」（U 型早密晚疏 + 护尾）；
- **max_gap 约束** 负责「别把 reuse run 拉太长」，消除退化、锁定鲁棒成员。

两条 schedule（自动 autohq vs 人工 hqphase）结构几乎一致：

```text
autohq  = [0,1,2,3,4,5,6,7,8,9,11,13,15,17,19,21,24,28,32,36,40,44,47,48,49]
gaps:     1×9 → 2×6 → 3 → 4×5 → 3 → 1,1   (max gap 4)
hqphase = [0,1,2,3,4,5,6,7,8,10,12,14,16,18,20,22,25,28,31,34,38,42,46,48,49]
gaps:     1×8 → 2×7 → 3×4 → 4×3 → 2,1     (max gap 4)
```

## 六、闭环验证结果（autohq 全自动 vs hqphase 人工 groundtruth）

run `qwen_autohq_verify_20260624_1735`，单分配同跑 in-run reference + hqphase_order2 + autohq_order2，landscape+color × 3 seed，1328²/cfp2，jvp_order=2。

| prompt | hqphase（人工） | autohq（自动） | 备注 |
| --- | --- | --- | --- |
| color（难，高频/高饱和） | 29.36 ± 0.88 | **29.62 ± 0.98** | autohq 均值还略高，方差同档 |
| landscape（易，自然场景） | 41.43 ± 1.09 | 40.42 ± 2.30 | 均 40+ dB 近无损 |
| 总体均值(6) | 35.39 | 35.02 | |

- 在最能区分策略的硬 prompt 上**均值/方差打平**（autohq 均值还略高）；
- 之前自动冠军 sum_chanmax 的 std 2.38，加 max_gap=4 后降到 0.98 —— **结构约束（max_gap=4）是把方差从 2.4 压到 0.9 的关键**。

**结论**：从一次 full-compute probe 出发，离线自动产出一条 schedule，真实回放即可追平人工手调 groundtruth（质量 + 方差），全程无需人工调相位。planner 主线目标达成。

## 七、关键工程资产

- FreeCache 策略全模型接线（Qwen/Flux/Wan）+ sigma-JVP 任意阶复用。
- `forced_compute_steps` / `forced_reuse_orders`：精确注入任意 schedule / 逐步 order。
- `steptrace` 策略：full-compute velocity 冗余探针（只存标量）。
- TracePlanner（GPU beam，已弃）+ **Edge DP planner**（`experiments/freecache/tools/traceplanner_edge_dp.py`，多 cost 度量）。
- 诊断脚本 `experiments/freecache/tools/edge_dp_vs_hqphase.py`（CPU，用 groundtruth 反选 cost）。
- single-allocation multi-request sweep（一次资源申请跑多 case，含 in-run reference）。

## 八、候选下一步

1. **多 prompt 平均 probe**（MeanCache 口径，多 prompt×多 seed 求平均 edge cost）再出 schedule，验证是否进一步降方差 / 提泛化。
2. **跨模型迁移**：把「整合型 cost + max_gap 约束」配方套到 Wan / Flux，各自标定 max_gap / warmup 先验。
3. **max_gap 作为可搜索结构超参**：把 max_gap∈{3,4,5} 纳入离线 DP 扫描，用 hq/opt ratio + 回放共同选定。
