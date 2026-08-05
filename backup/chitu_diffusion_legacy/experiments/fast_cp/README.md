# Hybrid Context Parallel / Fast CP

Fast CP 是 ChituDiffusion 的序列并行优化实验线。这个 feature 的目标不是做某一个固定
attention kernel，而是沉淀一套 **可解释、可测量、可迁移** 的 context parallel runtime：
根据 NVLink、PCIe、跨节点等不同互联，以及模型 token 结构，选择合适的 AGKV、Ring、
Ulysses、Ring+Ulysses 或其他 hybrid 策略。

配套控制面板见 [config.yaml](config.yaml)。其中 `infer.diffusion.*` 是 Chitu 配置字段，
`env.*` 是需要在启动前导出的环境变量。当前这个 YAML 是 feature 模板和开关索引，不会被
runtime 自动读取。

## 设计目标

1. 支持多种 CP 范式：
   - AGKV / all-gather K,V
   - Ring attention
   - Ulysses / all-to-all heads
   - Ring + Ulysses / USP-style hybrid
   - 后续更多 topology-aware hybrid 策略
2. 支持 async QKV projection，用投影、norm、RoPE 或 attention 计算窗口隐藏理论上可隐藏的通信。
3. 将 `txt + img` joint attention 作为独立范式，而不是普通 self-attention 的特殊 case。
4. 支持 `torch.compile` 和 CUDA Graph 开关，并明确它们在哪些 CP 模式上真正有效。
5. 解耦模型适配层，让 Wan、Flux、Z-Image、Qwen-Image 等 DiT 模型尽量复用同一套 CP runtime。

## 核心方法论

序列并行优化的关键不是先决定用 AGKV、Ring 还是 Ulysses，而是先回答这些问题：

1. 当前瓶颈是通信还是计算？
2. 每次切分之后，单卡上的 GEMM / SDPA 是否还能打满？
3. 这次切分节省了多少计算，又新增了多少通信？
4. pack/unpack、stream/event、NCCL、CPU launch 开销是否超过收益？
5. 某个优化的理论上限是多少？
6. 理论上限和实现复杂度相比，性价比是否值得？

因此 fast CP 的实验纪律是：

- 先在小 harness 上 profiling，再迁移到真实模型。
- 先拆分通信和计算，再设计 overlap 或 graph。
- 先估计理论上限，再决定是否工程化。
- 负结果也要记录，因为它能缩小策略空间。

## 开关分级

### 稳定 / 推荐默认

| 开关 | 类型 | 说明 |
| --- | --- | --- |
| `infer.diffusion.cp_size` | config | CP degree。`1` 表示关闭 context parallel。 |
| `infer.diffusion.up` | config | Ulysses 子组大小。`up=cp_size` 是纯 Ulysses，`up=1` 是纯 Ring，`1<up<cp_size` 是 Ring+Ulysses/USP。 |
| `infer.diffusion.cp_backend` | config | `auto` / `ucp` / `agcp`。普通 DiT CP 的策略入口。 |
| `infer.diffusion.compile_mode=off` | config | fast CP 真实性能测试的默认推荐。先关 compile，确认 CP 策略本身。 |
| `CHITU_HARNESS_A2A=single` | env | harness 中 Ulysses all-to-all 的推荐基线。真实 `CommGroup` 已自动优先使用 `all_to_all_single`。 |

### 可用但需要按场景打开

| 开关 | 类型 | 适用场景 | 注意 |
| --- | --- | --- | --- |
| `infer.diffusion.ring_cudagraph=true` | config | Ring / Ring-heavy path | 只适合静态 shape、非 varlen。与 NCCL 混用时通常需要 `NCCL_GRAPH_MIXING_SUPPORT=1`。 |
| `infer.diffusion.compile_mode=default` | config | block 外围融合 | CP comm 路径通常是 eager island。不要假设 compile 能优化 NCCL/stream。 |
| `CHITU_Z_IMAGE_CP_MODE=agkv/ulysses/ring/ring_graph/unified` | env | Z-Image model-specific CP core | 目前是 Z-Image 特定控制面。Qwen/Wan 主要走通用 `infer.diffusion.*`。 |
| `CHITU_Z_IMAGE_CP_PROFILE=1` | env | 分桶分析通信/计算 | 会同步 CUDA event，破坏 overlap，只用于分析。 |
| `CHITU_TORCH_TRACE=/abs/path.json` | env | Chrome trace | 推荐只抓 rank0、短窗口。trace 文件默认放本地，不进仓。 |

### 实验性 feature

这些开关默认应保持关闭。打开前需要先跑 harness，再做真实模型 A/B。

| 开关 | 类型 | 状态 | 当前结论 |
| --- | --- | --- | --- |
| `CHITU_CP_OVERLAP_QKV=1` | env | **实验性** | async QKV projection overlap。正确性可行，但 NVLink 真实模型收益很弱，默认关闭。 |
| `CHITU_CP_FP8_KV=1` | env | **实验性** | 压缩跨卡 K/V 字节。PCIe comm-bound 下有价值，需要质量评估。 |
| `CHITU_CP_KV_CACHE=1` | env | **实验性** | 跨步复用远端 K/V。更快但更近似，需要按模型和步数评估。 |
| `CHITU_CP_RING_NO_OVERLAP=1` | env | **诊断性** | 强制 Ring wait-before-compute，用于测 overlap 天花板，不是优化。 |
| `CHITU_ULYSSES_SPLIT=txt_q/txt_out/txt_both` | env | **harness-only 实验性** | text/image query split。语义正确，但当前 naive split 会变慢。 |
| `CHITU_CP_COMPILE_TAIL=1` | env | **harness-only 实验性** | 编译 harness tail 的小操作。不是主线收益来源。 |

## 推荐策略

### NVLink 单节点

当前建议：

- 先用 Ulysses baseline：`cp_backend=ucp`，`up=cp_size`。
- 保持 `CHITU_CP_OVERLAP_QKV=0`。
- 保持 `infer.diffusion.ring_cudagraph=false`，除非专门测 Ring/RingGraph。
- `torch.compile` 只作为 block 外围优化测试，不要和 CP comm 收益混在一起判断。

原因：

- NVLink 上通信通常短，async overlap 空间小。
- stream/event/record_stream/NCCL launch 额外成本可能抵消收益。
- 真实模型短步测试里 Z/Qwen cp4 基本持平或变慢，Wan cp4 steady 只有约 1.8% 收益。

### PCIe / 无 P2P

当前建议：

- 先打开 profile，确认 comm-bound 程度。
- Z-Image 可横评 `agkv` / `ring` / `unified`。
- 优先考虑少搬字节和减少复制计算，而不是强行 overlap。
- fp8 K/V、KV cache、贯穿式序列切分比单纯 overlap 更值得继续研究。

原因：

- PCIe/no-P2P 下通信通常是主瓶颈。
- 大块连续通信优于碎片化 all-to-all。
- overlap 的理论上限受可隐藏计算窗口限制，通信太大时藏不住。

### Ring Graph

Ring Graph 是目前比较明确值得保留的 CUDA Graph 方向：

- 适合 Ring 或 Ring-heavy hybrid。
- 需要静态 shape。
- 主要收益来自减少 CPU launch / NCCL scheduling 开销。
- 不应直接推广到所有 Ulysses/all-to-all 路径。

## 已实现范围

| 方向 | 当前状态 | 备注 |
| --- | --- | --- |
| 通用 self-attention CP | 已有 | Wan / Flux-like 模型走 `DiffusionAttention_with_CP`。 |
| full text + sharded image CP | 已有 | Qwen-Image-like joint attention 走 `cp_attn_with_full_txt`。 |
| Z-Image replicated text + sharded image core | 已有 | `ImageContextParallelAttention` 支持 `agkv/ulysses/ring/ring_graph/unified`。 |
| `all_to_all_single` Ulysses baseline | 已有 | `CommGroup.all_to_all` 自动使用 packed fast path。 |
| async QKV projection | 原型 | Z/Qwen/Wan 已接入，默认关闭。 |
| compile / graph 一键化 | 局部 | Ring Graph 更可信；compile 对 CP comm 路径有限。 |
| 任意 DiT 模型适配 | 部分 | 仍需要模型暴露 Q/K/V producer、RoPE、mask、txt/img layout。 |

## Harness-first 验证流程

不要直接用 50-step diffusion 跑调度搜索。推荐流程：

1. `experiments/fast_cp/harness` 用合成 Q/K/V 驱动真实 CP core。
2. 先跑 correctness：serial vs overlap / graph 输出一致。
3. 再跑 component profile：通信、projection、SDPA、pack/unpack 各占多少。
4. 再抓 rank0 trace：确认是否真的 overlap，而不是视觉误判。
5. 最后迁移到 Z-Image / Qwen-Image / Wan 做短步和长步复验。

示例：

```bash
export CHITU_PROJECT_ROOT=$PWD
export SRUN_PARTITION=debug
export SRUN_EXTRA_ARGS="--exclusive"

bash experiments/fast_cp/harness/run_harness.sh 4 -- --mode agkv --check
bash experiments/fast_cp/harness/run_harness.sh 4 -- --mode ulysses --bench

CHITU_TORCH_TRACE=$PWD/outputs/fast_cp_trace/agkv_overlap.json \
  bash experiments/fast_cp/harness/run_harness.sh 4 -- --mode agkv --trace --trace-iters 5
```

## 真实模型启动示例

### Wan / Flux-like 通用 CP

```bash
script/srun_direct.sh 1 4 test/test_wan.py \
  models=Wan2.1-T2V-1.3B \
  infer.diffusion.cp_size=4 \
  infer.diffusion.up=4 \
  infer.diffusion.cp_backend=ucp \
  infer.diffusion.ring_cudagraph=false \
  infer.diffusion.compile_mode=off \
  output.timer=True
```

### Z-Image 策略横评

```bash
export CHITU_Z_IMAGE_CP_MODE=agkv       # agkv / ulysses / ring / ring_graph / unified
export CHITU_Z_IMAGE_CP_UP=4            # unified/ulysses subgroup size
export CHITU_CP_OVERLAP_QKV=0           # keep off unless explicitly testing overlap

script/srun_direct.sh 1 4 test/test_z_image.py \
  models=Z-Image \
  infer.diffusion.cp_size=4 \
  infer.diffusion.up=4 \
  infer.diffusion.cp_backend=ucp \
  infer.diffusion.compile_mode=off \
  output.timer=True
```

## 当前阶段结论

短版：

- NVLink 上 async QKV overlap 不是主线收益点。
- Ulysses 当前应以 `all_to_all_single` 作为干净 baseline。
- `torch.compile` 不能直接消除 CP comm 路径，CP attention 多数情况下需要 eager island。
- CUDA Graph 更适合 Ring 这类静态重复结构。
- `txt + img` joint attention 是值得保留的独立范式，但 naive query split 当前不值得迁移。
- 后续更有价值的方向是减少通信字节、减少 pack/unpack、保持大 GEMM/SDPA 效率、建立 topology-aware policy。

详细里程碑见 [milestones/2026-07-01.md](milestones/2026-07-01.md)。
PCIe 系统性工作记录见 [worklog.md](worklog.md)。
