# Wan 1.3B CP8 扩展性与 Ring CUDA Graph

## 口径

- 单节点 H20 NVLink，Wan2.1-T2V-1.3B，81 帧、832×480。
- 纯 Ring Context Parallel，重点比较 CP4/CP8 以及 attention-only CUDA Graph。

## 为什么 Wan 在 CP8 仍能扩展

Wan 的视频序列很长：CP8 后每卡仍有约 4095 个 local tokens。H20 上的序列长度扫描显示，
Ring attention 的 launch-bound/compute-bound 分界约在 2K local tokens：

- 低于分界时，CPU launch 和 NCCL rendezvous 暴露，CUDA Graph 能明显减少开销。
- 高于分界时，attention 计算和数据搬运覆盖了 launch 成本，GPU 关键路径仍然充足。

Wan CP8 明显位于第二种 regime。它没有像短序列图像模型那样被切成大量低利用率小 kernel，
因此增加 CP degree 后仍能保持较好的扩展效率。作为对照，Flux CP8 每卡约 576 tokens，
更接近 launch-bound，attention-only graph 收益明显得多。

## Ring Graph 结论

1. **attention-only graph 对 Wan 基本无收益。** CP4/CP8 的 isolated Ring attention
   GPU 时间只改善约 1%，端到端 steady state 持平或略慢。
2. **Graph 确实消除了 CPU enqueue 开销，但该开销不在 Wan 的 GPU 关键路径上。**
   这说明“成功 capture”不等于端到端加速。
3. **Graph 静态输出不能直接复用。** Wan 下游 block 会跨下一次 replay 保留或消费该 tensor；
   关闭 clone 会因静态存储被覆盖而降低输出一致性。安全路径必须保留 clone。
4. **后续若继续 graph，应扩大捕获范围。** 值得测试的是 projection 或完整 transformer
   block，而不是继续优化 attention-only replay。

## 总结

Wan 1.3B 的 CP8 扩展性来自长视频序列带来的充足单卡工作量，而不是 CUDA Graph。它说明
CP 策略必须同时看全局序列长度和切分后的 local workload：相同 Ring 实现，在 Wan 上是
compute/data-movement-bound，在短序列模型上则可能是 launch-bound。
