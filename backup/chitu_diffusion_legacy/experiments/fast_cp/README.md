# Fast CP / DeepCP

Fast CP 是 ChituDiffusion 的 Context Parallel 实验线，目标是形成一套面向 DiT 推理的、
可解释、可测量、可迁移的序列并行调度方法。核心问题不是固定选择某个通信 primitive，
而是根据硬件互联、序列形状、显存预算和计算利用率选择 CP 策略。

## 文档

- [deepcp_for_diffusion.md](deepcp_for_diffusion.md)：完整技术总结、实验规律与后续路线。
- [deepcp_slides.md](deepcp_slides.md)：组会汇报精简版。
- [results_zimage_cp.md](results_zimage_cp.md)：H20 NVLink 上 Z-Image 纯 CP 结论。
- [results_wan_ring_graph.md](results_wan_ring_graph.md)：Wan 1.3B CP8 扩展性与 Ring Graph 结论。
- [harness/README.md](harness/README.md)：可复用 CP core correctness/benchmark harness。

## 已确认结论

1. **CP 是调度问题。** PCIe/no-P2P 主要受通信限制；H20 NVLink 更容易受过度切分和
   小 kernel 利用率限制。
2. **通信模式必须按场景选择。** NVLink 且显存充足时 AGKV/Ulysses 更合适；PCIe
   两卡可偏 Ring，多卡更偏大块 AGKV；跨节点适合分层 Ring+Ulysses。
3. **NVLink 上盲目 overlap 不是主线。** 通信已经很短，stream/event、SM 争用和拆小
   GEMM/SDPA 的成本常抵消收益。只有 comm-bound 场景才有稳定空间。
4. **CUDA Graph 只对 launch-bound 结构有效。** 短序列 Ring 可获益；Wan 1.3B CP8
   的长序列 attention 仍由 GPU 计算和数据搬运主导，attention-only graph 基本无收益。
5. **减少通信有上限。** 当非通信计算地板占主导时，需要贯穿式序列切分、保持大 kernel
   效率，以及更高层的 CP/DP 调度，而不是继续替换 collective API。
6. **CFG 与 CP 正交。** 关闭的是 CFG parallel，不是 CFG 算法；guidance 带来的 forward
   次数同时作用于单卡和 CP，实测不改变 Z-Image 的 CP scaling。

## 当前状态

实验阶段的一次性 probe、trace、microbenchmark demo、过期计划和原始日志已清理。目录保留
最终结论以及最小可复用 CP core harness；生产实现位于正式 runtime/attention 模块。
