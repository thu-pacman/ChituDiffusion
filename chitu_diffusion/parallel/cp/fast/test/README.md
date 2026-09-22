# Fast CP wrapper 验证

`test_wrapper.py` 检查 Python 接口与资源生命周期；`test_joint_qkv_epilogue.py`
检查融合 epilogue 的数学、非零槽位偏移、边界保护和 Graph 重放。
`distributed_wrapper.py` 在单节点 H20 NVLink 上检查真实通信、模型数学和封装开销。

`test_agkv.py`、`test_ulysses.py` 检查 transport，`test_ring_*.py` 和
`test_distributed_ring.py` 保留实验 Ring 的覆盖。测试集中在本目录，已经加入项目
默认 pytest 搜索路径；需要 CUDA 或多 rank 的测试在环境不足时跳过。

## 测量范围

所有模型使用随机权重、BF16、B1、D128、TP1。Wan 1.3B 使用 H12、全局
S32760；Wan 14B 使用 H40、全局 S75600。Qwen 使用 H24、image 4096 + text 512；
ZImage 使用 H30、image 4096 + text 512。这里不加载 checkpoint，但 projection、
RoPE、通信、attention 与 output projection 的 tensor shape 和执行边界与实际模型
一致。Qwen/ZImage 性能数据采用图文都分片的布局；ZImage 复制文本另做正确性检查。

每个 CP degree 测试四个 workload × AGKV/Ulysses，共 24 组。与 dense reference
比较时使用 `atol=0.01, rtol=0.03`。额外检查不同权重的两层复用同一通信槽位、
输入值更新、不同输入地址、输出所有权、shape A→B→A、显式失效、错误 stream、
借用 runtime 关闭、rank 签名不一致和单 rank 能力不足时的集体 origin 回退。

性能基线是**同一个 prepared 实例的直接调用**，被测对象是 `bound.run()`。
两者使用相同权重、attention plan、通信槽位和 Graph，计时包括 QKV projection
至 output projection。每轮每侧预热 3 次、测量 15 次，共四轮交替 AB/BA。每个样本
先取所有 rank 的最大 CUDA event 时间，汇总取每侧 60 个样本的中位数。这里没有
测量普通 `block(**inputs)` 的 staging 和输出复制开销，也不与旧跨框架数据混合。

## 2026-09-21 验收

Wan 来自作业 246599，独占 `bjdb-h20-node-039`，用时 8 分 20 秒；Qwen/ZImage
来自作业 246516，独占 `bjdb-h20-node-020`。43 项 pytest 通过；CP2/4/8 的 24 组
分布式检查、ZImage 复制文本和集体回退均通过。Qwen 自动选择 packed 直写路径，
Wan/ZImage 使用 CE overlap，Ulysses 保留串行搬运调优和 Attention Planner。

真实 Wan shape 的直接调用与绑定入口中位延迟如下，单位为 ms：

| Workload | CP | AGKV direct → bound | Ulysses direct → bound |
| --- | ---: | ---: | ---: |
| Wan 1.3B，S32760/H12 | 2 | 26.491 → 26.496 | 27.158 → 27.161 |
| Wan 1.3B，S32760/H12 | 4 | 13.317 → 13.325 | 13.721 → 13.723 |
| Wan 1.3B，S32760/H12 | 8 | 7.264 → 7.257 | 9.220 → 9.229 |
| Wan 14B，S75600/H40 | 2 | 474.563 → 474.544 | 478.471 → 478.464 |
| Wan 14B，S75600/H40 | 4 | 237.800 → 237.799 | 240.343 → 240.342 |
| Wan 14B，S75600/H40 | 8 | 119.121 → 119.206 | 120.570 → 120.557 |

绑定入口的延迟变化为 −0.13%～+0.94%，绝对变化为 −18.69～+84.99 μs。
24 组均满足本轮 2% 的开销目标；长 Wan case 的绝对抖动较大，但相对变化不超过
0.10%。负值属于微小测量波动。该结论仅覆盖本次固定
shape 矩阵，不承诺所有负载的开销上界。

汇总和全部 2880 个计时样本位于：

- [wrapper 汇总 CSV](../../../../../docs/assets/fast_cp/fast-cp-h20-wrapper.csv)
- [wrapper 样本 CSV](../../../../../docs/assets/fast_cp/fast-cp-h20-wrapper-samples.csv)

Qwen/ZImage 原始 JSON、Slurm 日志和源码 SHA256 位于
`outputs/fast-cp/wrapper-20260921/`，Wan 位于
`outputs/fast-cp/wrapper-wan-real-20260921/`。
开发过程中曾修复测试 fixture 未显式设置 Wan 原生 processor 的问题；首次封装的
短 Wan Ulysses 调用开销超过 2%，随后减少锁的 Python 包装层和 stream 查询开销。
旧的 Wan H12/S4096 结构测试不再进入汇总；最终 Wan 数据仅采用作业 246599。

## 目录重构回归（2026-09-21）

作业 246660 在独占 H20 node035 完成，用时 8 分 9 秒：91 项单元/GPU 测试通过，
四个 workload × 两种算法 × CP2/4/8 的 24 组分布式检查全部通过，包括真实 Wan
1.3B/14B、跨层槽位复用、ZImage 复制文本与集体回退。另一次 CPU 回归覆盖
`tests/unit/parallel` 和本目录，共 128 项通过、44 项因缺少 GPU/多 rank 环境跳过。
Qwen 模板选择仍为 packed，Wan/ZImage 为 overlap，Ulysses 为 serial。

绑定入口相对直接调用的中位延迟变化为 −0.18%～+0.60%。这验证重构后的封装开销，
不将不同节点的前后绝对延迟差异解释为优化。数据保存在
[回归汇总 CSV](../../../../../docs/assets/fast_cp/fast-cp-h20-layout-regression.csv) 和
[回归样本 CSV](../../../../../docs/assets/fast_cp/fast-cp-h20-layout-regression-samples.csv)。
原始 JSON、日志及重构前备份位于 `outputs/fast-cp/layout-refactor-20260921/`。
首次作业 246659 因 sbatch 临时脚本路径解析错误在启动时退出；路径已修正，最终
数据仅使用作业 246660。此记录与前面的首次 wrapper 验收分别保存。

本轮复用已有扩展二进制，未验证干净环境下的重新编译。安装器 `--check` 返回 1：
本地 `refs/fast-cp-sm90-sm120` 的 `fast_agkv.cu/.cuh` 还包含额外实验实现，与仓库
`csrc/` 不完全一致。重构前备份确认，本次未改动任何 `csrc/` 文件；发布前仍需在
独立 checkout 上应用仓库源码、重新编译并复跑验收。

## 复现

在已编译 Fast CP 扩展的 H20 节点上执行。以下路径对应当前工作区安装位置；换环境
时调整扩展和 NVSHMEM 路径。

一键运行单元测试与完整 24 组矩阵：

```bash
bash chitu_diffusion/parallel/cp/fast/test/run_wrapper.sh outputs/fast-cp/wrapper
```

从仓库根目录提交到 Slurm（可按集群调整 CPU 和内存申请）：

```bash
sbatch --exclusive --nodes=1 --ntasks=1 --gres=gpu:8 \
  --cpus-per-task=192 --mem=1937152 --time=00:15:00 \
  chitu_diffusion/parallel/cp/fast/test/run_wrapper.sh outputs/fast-cp/wrapper
```

可通过 `CHITU_PROJECT_ROOT`、`CHITU_PYTHON_BIN`、`FAST_CP_BUILD_ROOT` 和
`FLASH_ATTENTION_ROOT` 指定安装路径。脚本不会改写 `docs/assets/` 中的正式结果。
手动运行相同检查：

```bash
export PYTHONPATH="$PWD:$PWD/refs/fast-cp-sm90-sm120:$PWD/refs/kernels/flash-attention"
export LD_LIBRARY_PATH="$PWD/refs/fast-cp-sm90-sm120/.deps/nvidia/nvshmem/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 NCCL_GRAPH_MIXING_SUPPORT=0 NCCL_GRAPH_REGISTER=0
export TORCH_NCCL_ENABLE_MONITORING=0
.venv/bin/python -m pytest chitu_diffusion/parallel/cp/fast/test -q
for cp in 2 4 8; do
    .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node "$cp" \
        chitu_diffusion/parallel/cp/fast/test/distributed_wrapper.py \
        --models wan1.3b wan14b qwen zimage \
        --output "outputs/fast-cp/wrapper/cp${cp}.json"
done

.venv/bin/python chitu_diffusion/parallel/cp/fast/test/summarize.py \
    outputs/fast-cp/wrapper/cp{2,4,8}.json \
    --summary docs/assets/fast_cp/fast-cp-h20-wrapper.csv \
    --samples docs/assets/fast_cp/fast-cp-h20-wrapper-samples.csv
```

性能测量应申请独占的 8 GPU 节点，避免其他进程共享 GPU。CPU 环境可以只执行
`test_wrapper.py`；epilogue 测试需要 CUDA，分布式测试需要 H20 NVLink 与扩展。

实验 Ring 的通信协议回归可单独运行：

```bash
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m pytest chitu_diffusion/parallel/cp/fast/test/test_distributed_ring.py -q
```
