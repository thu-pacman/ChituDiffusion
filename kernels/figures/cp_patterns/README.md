# Context-Parallel Pattern Figures

文章配图统一以 `CP=4` 为例。rank 颜色表示 token 的原始 sequence shard，红色箭头
表示跨 GPU 数据移动，深色区域表示 attention compute。

- `cp-patterns-overview.{svg,png}`：Ulysses、Ring、All-Gather KV、Full-Mesh
  Fused Attention 的 2×2 总览。
- `ulysses.{svg,png}`：sequence shard 经 Q/K/V all-to-all 转换为
  full-sequence、`H/CP` head shard，attention 后再做 inverse all-to-all。
- `ring.{svg,png}`：Q 保持本地，K/V block 沿环逐步传递，每步执行 partial
  attention，并通过 online softmax/LSE 合并。
- `agkv.{svg,png}`：Q 保持 sequence-sharded，K/V all-gather 后在每个 rank
  materialize 完整 K/V。
- `full-mesh.{svg,png}`：所有 source shard 直接写入所有 peer 的 symmetric
  K/V buffer，fused attention 按 tile ready 状态消费数据并与通信重叠。

图中内存结论采用本文实验口径：Ulysses 与 Ring 的 attention K/V 状态为
`O(T/CP)`；AGKV 与当前 Full-Mesh prototype 会在每个 rank materialize
`O(T)` 全局 K/V。Full-Mesh 不是 Ring 式低显存方案。

重新生成：

```bash
../ChituDiffusion/.venv/bin/python kernels/plot_cp_patterns.py
```
