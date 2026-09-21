# Fast CP

Fast CP 将 DiT 的图像或视频 token 分到多张 GPU。每张 GPU 只保存并计算本地 token，
attention 通过通信获得完整上下文。ChituDiffusion 提供 AGKV 和 Ulysses 两种分片方式，
并为二者提供 NCCL 与 Fast transport。

## 基本原理

### AGKV

每个 rank 保留本地 Q，只收集其他 rank 的 K/V。NCCL 路径使用
`all_gather_into_tensor`，Fast AGKV 通过 NVSHMEM 把本地 K/V 直接写入目标 rank 的
最终 buffer。通信结束后，各 rank 使用本地 Q 和全局 K/V 计算 attention。

AGKV 适合 head 数较少或序列较长的模型，因为它不要求按 rank 切分 attention heads。

### Ulysses

Ulysses 在 attention 前把张量从“局部序列、全部 heads”变换为“完整序列、局部
heads”，attention 后执行反向变换。NCCL 路径使用 all-to-all，Fast Ulysses 使用
NVSHMEM transport。

Ulysses 在 head 数不能被并行度整除时会自动 padding；padding 会增加通信和
attention 工作量。序列较短、head 数足够且能够均匀切分时，它通常比 AGKV 更快。

## 优化方案

Fast CP 包含以下优化：

1. **Device-to-device 数据搬运。** Fast AGKV 和 Fast Ulysses 使用 NVSHMEM
   symmetric memory，使发送端直接写入目标 rank 的最终位置，省去 NCCL collective
   前后的中间 buffer、内存重排和额外拷贝。同一节点内的 Ulysses subgroup 复用
   symmetric memory pool，避免每层重新分配通信 buffer。
2. **Attention 计算。** Attention planner 根据本地 shape、dtype、GPU 型号、SM 数
   和 CP degree 动态生成并选择候选，不依赖逐 shape 的静态配置表。候选包括 cuDNN
   attention、不同 tile 的 CuTe FlashAttention，以及 head-tail 和 query-tail 组合。
   planner 让主体任务形成完整 wave，只沿 K/V 维拆分尾部任务以增加可并行 CTA，最后
   使用 FP32 output/LSE 合并 partial attention。
3. **Device 内部数据搬运与重叠。** 运行时分别测量 SM、TMA 和 Copy Engine 路径，
   根据 GPU、tensor shape、搬运方向和互连选择最快实现，并调整多流并发策略。Fast
   AGKV 可使用 CE 传输 K/V，同时执行 Q projection、norm、RoPE 或文本 projection；
   Fast Ulysses 对输入和输出方向独立选择搬运路径。当前 H20 模型测试中多数输入方向
   选择 TMA，个别超长输出 shape 选择 SM；PCIe 设备上的 CE 路径还会根据共享 egress
   带宽限制调整同时在飞的远端目标数。

`torch` transport 使用 NCCL；`auto` 在 Fast transport 不可用时回落到 NCCL；显式
选择 Fast transport 而运行环境不满足要求时直接报错。

`fast/experimental/` 还包含 Fast Ring，用于 benchmark；模型 pipeline
不会自动选择。原独立 fused Fast AGKV 实验已经合并到 Fast AGKV transport，
不再作为第二条路径维护。

Fast transport 当前要求单机 Hopper、CUDA P2P、NVSHMEM，以及针对目标 CUDA 和 GPU
架构编译的扩展。多机静态 CP 使用 NCCL。

一个进程只能持有一个 NVSHMEM runtime，所以每个 rank 只有一条 CP lane 走 Fast，
由 `fast_lane_width` 指定其宽度，默认是整个 TP plane。TP、EP、VAEP 和 CFG 分支合并
都在别的通信域上，与之互不干扰；CFG 把 plane 切成两条半宽 CP lane 时，模型传入实际
宽度即可让两条 lane 各自使用 Fast。scheduler 激活其它宽度的 lane 时自动回落 NCCL。

## 优化结果

测试在单节点 NVIDIA H20 上完成，使用 BF16 dense non-causal attention。共享 CP
attention 单元测试从预投影的本地 Q/K/V 开始，到本地 O 结束，因此同时包含通信和
attention，不包含 projection、norm、RoPE 与 output projection。

| 全局序列长度 | Fast AGKV CP2 | CP4 | CP8 | Fast Ulysses CP2 | CP4 | CP8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1.78x | 2.90x | 4.05x | 1.71x | 2.91x | 4.92x |
| 8192 | 1.87x | 3.37x | 5.55x | 1.83x | 3.46x | 6.33x |
| 16384 | 1.95x | 3.68x | 6.63x | 1.95x | 3.77x | 7.28x |
| 32768 | 1.96x | 3.81x | 7.22x | 1.96x | 3.88x | 7.65x |

加速比相对相同 shape 下的单卡 Chitu attention。短序列的固定通信、同步和 kernel
launch 开销占比更高；随着序列变长，attention 的二次计算量占主导，CP8 更接近线性
扩展。表中 Fast CP 延迟同时包含 NVSHMEM 通信和 attention planner 的收益。

下图把共享序列测试与真实模型 attention block 放在一起。每个顶点独立归一化，得分
为该顶点最低延迟除以当前方法延迟，`100%` 表示最快。Wan 1.3B 和 Wan 14B 使用真实
全局序列长度 32760 和 75600；Qwen-Image 与 Z-Image 使用 4096 image token 和
512 text token。

![H20 原生 CP 延迟归一化雷达图](../assets/fast_cp/fast-cp-h20-native-relative-radar.png)

图表数据以 CSV 形式随文档保存：

- [共享 CP attention 单元测试](../assets/fast_cp/fast-cp-h20-cp-attention-unit.csv)
- [雷达图原始延迟](../assets/fast_cp/fast-cp-h20-native-latency-radar.csv)
- [雷达图归一化数据](../assets/fast_cp/fast-cp-h20-native-relative-radar.csv)

真实模型 attention block 包含 Q/K/V projection、norm、RoPE、通信、attention 和
output projection。Fast AGKV 在 Wan 14B CP8 上达到 `119.371 ms`，在 Qwen-Image
CP8 上达到 `0.963 ms`；Fast Ulysses 在 Z-Image CP8 上达到 `1.242 ms`。这些测试
不加载 checkpoint，也不包含完整 denoising loop、文本编码器和 VAE，因此不代表完整
生成任务的端到端加速。

Qwen-Image 的图文分片 processor 还将配对 QK norm、RoPE 和 joint QKV 布局写入
融合，减少中间张量。prepared AGKV 在准备阶段实测 packed QKV 与 CE overlap 两种
调度；本次 `1×4608×24×128` 测试全部选择 packed，CP2/4/8 分别为
`2.520/1.434/0.963 ms`，比同轮最快 SGLang 路径低 `3.8%/4.2%/7.5%`。Fast Ulysses
为 `2.756/1.612/1.032 ms`，CP2/4 仍有差距。prepared AGKV 由 block benchmark
显式安装，目前不由完整生成 pipeline 自动启用；首次调优时间不计入稳态延迟，
额外候选 buffer 随 runtime 关闭而释放。完整测量口径见[基准结果](../../tools/benchmarks/fast_cp_frameworks/RESULTS.md)。

2026-09-21的独立直写消融进一步去除了prepared Qwen packed AGKV的两次本地K/V
staging copy。H20、image 4096/8192/16384加text 512的测试中，CP2 block延迟降低
约0.35%–0.68%；CP4/CP8收益较小且部分配置退化，尚不能宣称普遍加速。它不改变
attention算法，也不新增计算通信overlap；详细配对结果与波动见基准结果文档，CSV
保存在`docs/assets/fast_cp/fast-cp-h20-qwen-direct-slot.csv`。本轮未重测其他框架，
上面的跨框架结果与雷达图仍对应原测量版本。

## 显式区域 wrapper

`chitu_diffusion.parallel.cp.fast.init_fast_cp` 提供共享 runtime，
`wrap_attention(..., adapter=..., algorithm=...)` 包装 QKV projection 至 output
projection。Wan、Qwen-Image、Z-Image adapter 保留各自的 norm、RoPE 和文本分布
语义，公共层管理准备、通信槽位、生命周期与回退。Qwen 准备阶段继续选择 packed
或 overlap，Ulysses 使用串行调度及 SM/TMA/CE 选择。

普通入口为 `block.prepare(example_inputs=...)` 后 `block(**inputs)`，使用稳定
staging 槽位和独立输出；固定地址入口为 `bound=block.bind(inputs=...)` 后
`bound.run()`，返回借用输出。同形状的层共享通信槽位，避免对称堆随层数线性增长。
当前限制为 H20 NVLink、CP2/4/8、TP1、B1、BF16、D128、等长无 mask 分片。
准备必须集体执行，暂不支持外层 Graph capture、并发 stream 或任意 Python 图自动重写。
这是显式接入接口，尚不自动替换完整生成 pipeline 的 attention 调用。
验收使用 Wan 1.3B 的 `1×32760×12×128`、Wan 14B 的
`1×75600×40×128` 以及 Qwen/ZImage 的实际 attention shape；24 组 CP2/4/8
AGKV/Ulysses 检查全部通过。绑定入口相对同一 prepared 实例直接调用的延迟变化为
−0.13%～+0.94%。
详细 API、资源约束与测试命令见[Fast CP README](../../chitu_diffusion/parallel/cp/fast/README.md)。

代码按职责分为 `runtime/`、`attention/`、`projection/`、`adapters/`、`profiles/`，
公共入口为 `fast/__init__.py` 和 `api.py`。每个模型的契约及执行模板集中在同一个
adapter 文件，共用模型无关的 AGKV 调度。原未发布的平铺模块和 experimental 兼容
转发已删除；`experimental/` 仅保留 Ring，测试集中在 `fast/test/`。

## 当前适用范围

H20/NVLink 上的 transport、attention planner、模型外围融合和显式 wrapper 已形成
可复用的首版接口：

- NVSHMEM transport、symmetric pool、运行时架构检查和 SM/TMA/CE 选择属于公共层，
  可以作为其它模型和单节点互连环境的基础。
- attention planner 的成本参数目前只校准 H20、BF16、head dimension 128；其它 GPU
  会回落到普通 SDPA，不能直接沿用 H20 的候选排序。
- direct-slot KV projection 当前限制 SM90；SM120 虽可构建和运行基础通信扩展，仍需
  单独验证 projection、planner 和模型 block。
- Wan、Qwen-Image 和 Z-Image 的 prepared AGKV 已迁到 `fast/adapters/`，由 wrapper
  管理固定输入槽、初始化和释放；完整生成 pipeline 尚未自动安装这些区域。

下一步迁移到其它 GPU 或模型时，应保持公共 runtime、硬件 profile 与模型 adapter
三层边界，并重新校准 planner 和搬运策略。当前数据只支持 H20/NVLink 上的性能结论，
不支持“相同策略无需校准即可迁移”的结论。

## 使用示例

先按[安装与启用指南](../usage/fast-cp-installation.md)完成 NVSHMEM、扩展编译、
CuTe 依赖和多卡检查。下面的 CLI 示例选择 transport；完整 wrapper 仍需显式
接入模型，不会由 transport 参数自动安装。

运行四卡 Fast AGKV：

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate \
  --model zimage \
  --model-path /path/to/Z-Image \
  --agkv-transport fast_agkv \
  --no-cfg-parallel \
  --output outputs/zimage-fast-agkv.png
```

运行四卡 Fast Ulysses：

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate \
  --model zimage \
  --model-path /path/to/Z-Image \
  --attention-mode ulysses \
  --ulysses-transport fast_ulysses \
  --no-cfg-parallel \
  --output outputs/zimage-fast-ulysses.png
```

不安装 Fast 扩展时，默认 `torch` transport 使用 NCCL：

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate \
  --model zimage \
  --model-path /path/to/Z-Image \
  --no-cfg-parallel \
  --output outputs/zimage-nccl.png
```
