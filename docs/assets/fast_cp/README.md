# Fast CP benchmark assets

本目录保存文档直接引用的稳定结果资产。原始逐 case JSON 和运行日志位于
`outputs/fast-cp/`，不纳入版本控制；这里的 CSV 是从对应完整结果机械生成或复制的
可审查快照。

| 文件 | 内容 | 行数（含表头） |
| --- | --- | ---: |
| `fast-cp-h20-cp-attention-unit.csv` | 共享 CP attention 单元测试，包含 84 个框架、方法、序列长度和并行度组合 | 85 |
| `fast-cp-h20-native-latency-radar.csv` | 雷达图使用的原始延迟和缺测状态 | 145 |
| `fast-cp-h20-native-relative-radar.csv` | 每个 workload 相对最快方法归一化后的雷达图数据 | 145 |
| `fast-cp-h20-qwen-direct-slot.csv` | Qwen packed K/V 直写消融，9 组 CP/长度的配对统计 | 10 |
| `fast-cp-h20-qwen-direct-slot-samples.csv` | 上述消融的全部 rank-max CUDA event 样本 | 1621 |
| `fast-cp-h20-wrapper.csv` | 四个 workload、两种算法、CP2/4/8 的绑定入口开销与实际执行计划 | 25 |
| `fast-cp-h20-wrapper-samples.csv` | wrapper 与同一 prepared 实例直接调用的全部 rank-max CUDA event 样本 | 2881 |
| `fast-cp-h20-layout-regression.csv` | 目录重构后完整 24 组 wrapper 回归，单节点同次作业 | 25 |
| `fast-cp-h20-layout-regression-samples.csv` | 目录重构回归的全部 rank-max 样本 | 2881 |

共享 attention 的测量边界是预投影的本地 Q/K/V 到本地 O；雷达图还包含模型
attention block，后者包括 projection、norm、RoPE、通信、attention 和 output
projection。两种边界不能直接相加或互相替代。硬件、软件版本、误差和计时方式见
[`docs/features/fast-cp.md`](../../features/fast-cp.md) 及
[`tools/benchmarks/fast_cp_frameworks/RESULTS.md`](https://github.com/thu-pacman/ChituDiffusion/blob/main/tools/benchmarks/fast_cp_frameworks/RESULTS.md)。

发布新结果时，应先重新生成 `outputs/fast-cp/cp-attention-native-*/results.csv` 和
`outputs/fast-cp/figures/latest/*-values.csv`，再同步替换本目录中的三个 CSV。不要从
Markdown 表格反向恢复数据。

Qwen 直写消融来自独占 H20 node038 的作业 246438：BF16、B1/H24/D128、文本
512 token，计时覆盖 QKV projection 至 output projection。每组新旧路径共享输入与
原始权重，attention plan 一致，六轮交替 AB/BA，每侧每轮15个样本。负的
`latency_reduction_pct` 表示直写更慢；`paired_round_saved_us_min/max` 保留轮间波动。
此消融未重测其他框架，因此不替换原有跨框架雷达图。实验与复现脚本位于
`outputs/fast-cp/qwen-direct-slot-20260921/`，讨论见基准结果文档的直写消融部分。

Wrapper 验收的 Wan 数据来自独占 H20 node039 作业 246599：1.3B 使用
S32760/H12，14B 使用 S75600/H40。Qwen/ZImage 数据来自独占 node020 作业
246516，使用 image4096+text512。`overhead_pct` 为
`(bound_ms/direct_ms - 1) × 100`，负数仅表示测量中绑定调用略快，不能解释为封装
带来加速。四轮交替 AB/BA，每侧每轮 15 个样本，汇总取全部 60 个 rank-max 样本的
中位数。计时不含普通入口的输入 staging 和输出复制。复现与边界说明见
[`fast/test/README.md`](https://github.com/thu-pacman/ChituDiffusion/blob/main/chitu_diffusion/parallel/cp/fast/test/README.md)。

目录重构回归来自独占 H20 node035 作业 246660，使用相同的实际 workload 和四轮
交替 AB/BA 方法。24 组全部通过，绑定入口相对同一 prepared 实例直接调用的延迟
变化为 −0.18%～+0.60%。这两个 CSV 单独记录结构变更后的回归，不改写此前跨框架
benchmark，也不将不同节点的绝对延迟差异归因为重构收益。
本轮使用已有扩展二进制，尚不代表干净源码构建的验收；扩展 checkout 的已知差异
见上述测试 README。
