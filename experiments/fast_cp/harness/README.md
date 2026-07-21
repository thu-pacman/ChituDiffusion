# CP Core Benchmark Harness

该 harness 用合成 Q/K/V 驱动真实 `ImageContextParallelAttention`，用于在不加载模型权重和
去噪 pipeline 的情况下验证 CP correctness、通信/计算 overlap 与 CUDA Graph 调度。

保留它是因为它仍是可复用的 core-level regression/benchmark 入口；一次性 probe、trace 和
microbenchmark demo 已在阶段结束后清理。

## 文件

- `cp_async_harness.py`：correctness、benchmark、component profile 和 trace 入口。
- `overlap_cp.py`：harness-only overlap/graph 调度，被主 harness 直接依赖。
- `run_harness.sh`：单节点 `torchrun` 启动器。

## 使用

```bash
# correctness
bash experiments/fast_cp/harness/run_harness.sh 4 -- --mode agkv --check
bash experiments/fast_cp/harness/run_harness.sh 4 -- --mode ulysses --check

# benchmark
bash experiments/fast_cp/harness/run_harness.sh 4 -- --mode agkv --bench
bash experiments/fast_cp/harness/run_harness.sh 8 -- --mode ulysses --bench
```

默认形状近似 Z-Image 1024×1024。结果只表示单次 CP attention 的机制与理论上限，最终性能
仍须在真实模型的端到端 denoise 墙钟上确认。

## 实验纪律

1. 先用 `--check` 确认 serial 与候选调度数值一致。
2. 再做 component profile，判断通信是否值得 overlap。
3. 只在需要确认调度关系时生成短 trace；trace 写入 `outputs/`，不提交。
4. 最后在同一节点、同一模型配置下做真实模型 A/B。
