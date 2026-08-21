# Fast 并行后端

本目录集中 Fast Ulysses、Fast AGKV、Fast Ring 及 Ulysses×Ring 编排。
公共协议、能力检测入口和 transport factory 位于 `parallel/` 根目录；模型代码只依赖
这些协议。`parallel/nccl/` 提供基于 `torch.distributed`/NCCL 的通用实现和回退。
Fast Ulysses 与 Fast AGKV transport 是 capability-gated 的显式 opt-in；Fast
AGKV attention 和 Fast Ring 仍是 benchmark-only，不提供稳定
pipeline selector。启用任何 Fast 路径前都必须按目标机器重新构建并验证扩展。
最终性能矩阵与选择策略见
[`kernels/fast_cp_results.md`](../../../kernels/fast_cp_results.md)。

## 状态与约束

- **Fast Ulysses**：已接入 transport factory。每个进程只创建一个 node-local
  NVSHMEM runtime；同节点的不同 U row 通过 logical-subgroup A2A 共享 symmetric
  pool。GPU 间需 CUDA P2P/NVLink 可达，且至少为 Hopper（SM90+）。
  仅支持 eager、连续兼容的 4D `[B,S,H,D]` FP16/BF16 head↔sequence 交换，
  CUDA Graph capture 时回退 NCCL。`auto` capability 失败只把 U row 回退
  Torch/NCCL，不改变外层 Ring topology。
- **Fast AGKV transport**：已接入 transport factory，性能仅在 H20 上验证。要求单节点
  NVLink、static full-world、2–8 ranks；输入为连续 BSHD FP16/BF16，K/V 同形，
  每行 `heads * head_dim * element_size` 必须 16-byte 对齐。动态 lane 或不支持的 shape
  回退 NCCL；异步 lane 默认仅在较小 rank 数启用。
- **Fast AGKV attention**：Fused Full-Mesh 实现，仍是实验性
  research/benchmark 路径，不应视为通用 attention 后端。
  要求精确 SM90、CP2/CP4/CP8、dense non-causal FP16/BF16 attention。
  batch 可为任意正数；head dimension 支持 `[8, 256]` 内 8 的倍数；全局序列
  只需 `S >= CP`，即每个 rank 至少一个 token。它使用定制 FlashAttention
  CuTe overlay 与 full-mesh transport overlay。warm 后通过 direct compiled
  launcher 绕过逐轮 DSL dispatch。
- **Fast Ring**：低显存实验路径，CP2/4/8，双 landing buffer CE ring、CuTe
  partial attention 与 Triton FP32 online-softmax merge。phase zero 先计算，
  forwarding 由 GPU arrival ticket 预提交，ACK 使用独立 stream。它不要求
  `H % CP == 0`，但当前实测未赢过 Fast AGKV/Fast Ulysses。
- Full-Mesh 默认且推荐 `chunk_count=1, scheduler="fixed"`。研究接口支持
  `2/3/4` 个 source×chunk 段；`scheduler="ready_mask"` 会由 producer warp
  leader 选择已到达段，并通过 per-stage block metadata 让 MMA 按同一乱序消费。
  当前只对全局序列和段起点均 256-token 对齐的 shape 启用，其他情况自动回退
  fixed，以避免 TMA tile 跨两个 arrival flag。加入 ready-set cache 和
  intra-WG overlap 后，H20 all-rank critical-path A/B 仍比 C1 慢约 0.4–3.3%；
  因此只保留为研究接口，不会自动启用。

Full-Mesh 的 batch、head 数和序列长度是 runtime shape，不进入编译键。head
dimension 仍按上游 FlashAttention 的 SM90 最优配置专门化并在首次使用时 JIT；
overlay 直接读取该 kernel 的 `tile_n` 生成 slot 阈值，不在 Chitu 侧复制 tile
启发式。因此 `D=64/80/96/128/192/256` 等 shape 会使用各自的上游配置，而
`D=128` 原路径不降级为通用 kernel。

Fast AGKV attention、Fast AGKV transport 和 Fast Ring 依赖单节点 peer access；
Fast Ulysses runtime 只要求其 U row 所在节点满足 peer access，外层 Ring 使用 NCCL
并可跨节点。当前 NVSHMEM 要求 3.7+。`fast-ulysses`
审查基线为 commit `6e5dcb24dc44e781ac3091d1d9b3f9fef314fb87`。

## 包结构

- `ulysses.py`：Fast Ulysses transport、共享 node runtime 和 logical subgroup；
- `agkv.py`：Fused Full-Mesh Fast AGKV attention；
- `agkv_transport.py`：独立 paired-K/V gather transport；
- `ring.py`：单节点 CE Fast Ring；
- `ulysses_ring.py`：U row redistribution、NCCL Ring、inverse Ulysses；
- `_runtime.py`、`_cute.py`、`_ring_merge.py`：非公共 helper。

USP 自动选择 `U = max(d | d divides CP, d divides heads, d <= node width)`，
`R = CP/U`。例如 CP8/H12 为 U4×R2，CP8/H3 为 U1×R8，CP8/H40 为
U8×R1。显式 `ulysses_degree` 是 U 的上限；不整除 heads 时自动向下选择，
head padding 只保留为直接调用低层 API 时的兜底。

## 安装

安装审查过的 Fast Ulysses 依赖：

```bash
uv sync --extra fast-ulysses
```

向仓库管理的 checkout 应用 Fast AGKV overlay：

```bash
python script/install_fast_agkv.py refs/fast-ulysses
```

到达 probe 使用 JIT CUDA observer；CUDA 13 wheel 环境还需要 CCCL headers：

```bash
uv pip install --python .venv/bin/python nvidia-cuda-cccl
```

准备 Full-Mesh 实验依赖：

```bash
python script/kernel_watch.py sync flash-attention
python script/install_full_mesh_cute.py refs/kernels/flash-attention
python script/install_fast_agkv.py refs/fast-ulysses
```

overlay 脚本只修改本地 checkout。之后仍须遵循 upstream 的 CUDA/NVSHMEM 构建
流程，设置与机器匹配的 CUDA toolkit、`NVSHMEM_HOME` 和运行时库路径，并重新编译、
重新安装 `fast-ulysses` 与 FlashAttention 扩展。预编译产物不可假定能跨 CUDA、
NVSHMEM 或 GPU 架构复用。

## 运行时选择

生产 facade 默认使用 Torch/NCCL。只有下面两个 transport 有稳定 selector：

```bash
# CLI
--ulysses-transport fast_ulysses
--agkv-transport fast_agkv

# 等价环境变量
CHITU_ULYSSES_TRANSPORT=fast_ulysses
CHITU_AGKV_TRANSPORT=fast_agkv
```

不满足 node-local peer access、logical subgroup 或 shape gate 时，`auto` 会回退
U row 到 Torch/NCCL；显式请求但本机扩展不可用时会报告原因。Fast AGKV attention
与 Fast Ring 只能由
benchmark/test 从具体模块显式构造，不能通过上述 selector 接入 Diffusers pipeline。

实验调优变量：

| 变量 | 取值/默认值 | 作用 |
| --- | --- | --- |
| `CHITU_FAST_ULYSSES_POOL_BYTES` | 正整数；默认 2 GiB | symmetric pool 大小 |
| `CHITU_FAST_ULYSSES_ASYNC_CE` | `0`/`1`；默认 `1` | Ulysses CE 异步路径 |
| `CHITU_FAST_ULYSSES_USE_TMA` | `auto`/`0`/`1` | TMA transport 选择 |
| `CHITU_FAST_AGKV_POOL_BYTES` | 正整数；实现默认值 | AGKV symmetric pool 大小 |
| `CHITU_FAST_AGKV_USE_CE` | `0`/`1`；默认 `0` | AGKV Copy Engine transport |
| `CHITU_FAST_AGKV_ASYNC` | `auto`/`on`/`off` | AGKV async overlap policy |

_English: Experimental single-node NVSHMEM backends. Rebuild all patched
extensions for the target CUDA, NVSHMEM, and GPU architecture._
