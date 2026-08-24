# 实验结果

仓库保留可复现的文字报告和必要图片。原始日志、模型输出和临时 benchmark 文件不属于
文档。

## FlexCache

- [MagCache 对比，2026-08-05](flexcache/magcache_compare_20260805.md)
- [MeanCache 对比，2026-08-04](flexcache/meancache_compare_20260804.md)

报告中的速度和质量只适用于记录的模型、硬件、dtype、scheduler、步数和尺寸。修改这些
条件后需要重新测试。

## Fast CP

Fast CP 的 H20 单机 scaling 和 winner matrix 位于[并行架构](../architecture/parallel.md)。
图表生成入口为：

```bash
python tools/plotting/fast_cp_scaling_matrix.py
```
