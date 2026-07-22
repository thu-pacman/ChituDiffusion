# Parallel

该目录维护 EPAC 中与模型无关的并行拓扑、通信和计算。它不是仅面向 context
parallel 的包：当前实现 CP，后续 CFG parallel 也应在此扩展，并复用同一套动态 lane
和 process-group 生命周期。

- `groups.py`：启动阶段创建候选 lane group，并维护当前 active lane；
- `topology.py`：active lane 对应的 Ulysses/Ring group view；
- `usp.py`：支持调用时注入 group 的 Ulysses x Ring attention；
- `image_attention.py`：`sharded image + replicated joint/text` 布局的 AGKV/USP 接口。

`groups.py` 是通用资源层；后三者是当前 context-parallel 实现。模型目录只能保留
QKV/RoPE/布局等 glue，不应复制通信算子或创建 process group。

USP 的 all-to-all 与 joint-ring schedule 基于 xDiT/xFuser 和 yunchang 的
Apache-2.0 实现。与 xFuser 原始全局 `PROCESS_GROUP` 不同，这里的 group 是每次调用显式
传入的，因此 EPAC 可以在 pulse 之间从 cp4 `u2r2` 切换到 cp2 `u2r1` 或 cp1 `u1r1`。
所有候选 group 必须在服务启动阶段以确定顺序创建；attention forward 不创建或销毁 group。

安装 USP 可选依赖：

```bash
uv sync --extra usp
```

CUDA 路径使用 yunchang `SeqAllToAll4D`、FlashAttention 和 xDiT-style joint ring。
CPU 路径只用于多进程数值回归，不代表服务性能。
