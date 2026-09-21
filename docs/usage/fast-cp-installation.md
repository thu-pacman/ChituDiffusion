# 安装与启用 Fast CP

本指南从干净的 ChituDiffusion checkout 开始，安装通信扩展、Attention Planner 的
CuTe 后端，并验证多卡执行。所有命令在仓库根目录运行；不需要模型权重即可完成验证。

## 支持范围

| 入口 | 当前范围 |
| --- | --- |
| Fast AGKV / Ulysses transport | 单节点、GPU P2P 全互通；扩展可构建 SM90 / SM120；连续 FP16/BF16 BSHD，行宽须为 16 字节的整数倍 |
| 完整 attention wrapper | 已验证 H20 NVLink、CP2/4/8、TP1、B1、BF16、D128、等长序列分片、无 mask 的非 causal MHA |
| 模型 adapter | Wan、Qwen-Image、Z-Image；显式接入 QKV projection 至 output projection |

SM120 通信扩展可构建，不代表 PRO 5000 已支持完整 wrapper。仅使用 NCCL 时无需
安装以下扩展。硬件 profile、输入契约和资源管理见
[Fast CP README](https://github.com/thu-pacman/ChituDiffusion/blob/kernels/chitu_diffusion/parallel/cp/fast/README.md)。

## 1. 基础环境

需要 Linux、Python 3.12/3.13、C++17 编译器、CUDA toolkit（含 `nvcc` 和
`cuobjdump`）、兼容的 NVIDIA driver。SM120 编译要求 CUDA 12.8+。下面以 CUDA 13
为例，PyTorch wheel 与 toolkit 应采用匹配的 CUDA 版本。

```bash
uv sync --group dev
source .venv/bin/activate
uv pip install --python .venv/bin/python cmake ninja setuptools wheel

export CUDA_HOME=/usr/local/cuda-13.0  # 改为本机 toolkit 路径
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

仓库目前锁定 Torch 2.10.0，默认使用 cu130 index；此前 H20 回归使用
Torch 2.9.1+cu130。二者不是同一环境，升级后需要重新编译扩展和验收，不能直接复用
旧 wheel 或假定得到相同延迟。基础环境选择见[安装](installation.md)。

## 2. NVSHMEM 与通信扩展

准备包含 `include/nvshmem.h` 和 `lib/libnvshmem_host.so*` 的 NVSHMEM 安装。
已有系统安装可直接设置 `NVSHMEM_HOME`；以下安装方式对应此前构建使用的
NVSHMEM 3.7.2 host library，单独放置以免与 Torch 环境中的 NVSHMEM 包混淆：

```bash
uv pip install --python .venv/bin/python --target .deps/fast-cp \
  'nvidia-nvshmem-cu12==3.7.2'
export NVSHMEM_HOME="$PWD/.deps/fast-cp/nvidia/nvshmem"
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

在独立 checkout 中构建，避免使用带有历史实验改动的源码目录：

```bash
git clone https://github.com/triple-mu/fast-ulysses.git refs/fast-cp-build
git -C refs/fast-cp-build checkout 6e5dcb24dc44e781ac3091d1d9b3f9fef314fb87
python tools/install/build_fast_cp.py refs/fast-cp-build \
  --architectures 90 --cuda-home "$CUDA_HOME" \
  --nvshmem-home "$NVSHMEM_HOME" --wheel-dir dist/fast-cp --jobs 4
python tools/install/install_fast_agkv.py refs/fast-cp-build --check
uv pip install --python .venv/bin/python --no-deps --force-reinstall \
  dist/fast-cp/fast_ulysses-*.whl
```

只部署 H20 使用 `90`；包含 SM120 的扩展使用 `--architectures 90 120`。构建脚本
应用仓库 `csrc/` 源码、编译并检查目标架构的 native CUDA image。`--check` 返回 0
表示 overlay 源码一致，**不检查二进制是否重新编译**。安装时 wheel 目录应只包含
本环境需要的一个版本。

`uv sync --extra fast-ulysses` 只安装上游依赖，不能代替上述 Chitu 扩展构建。
`install_fast_agkv.py` 单独运行也只修改源码，不编译、不安装 wheel。

## 3. Attention Planner 的 CuTe 后端

cuDNN 和 Triton 随 PyTorch 环境提供。完整候选集还需要 CuTe FlashAttention。
以下固定源码版本与此前回归的依赖版本；通过源码路径加载，不构建无关的 FA2 扩展：

```bash
git clone https://github.com/Dao-AILab/flash-attention.git refs/kernels/flash-attention
git -C refs/kernels/flash-attention checkout d7e4dba3e568106b0f1b6323b07c1272f53679b3
uv pip install --python .venv/bin/python \
  'nvidia-cutlass-dsl==4.6.1' 'quack-kernels==0.6.2' \
  'apache-tvm-ffi==0.1.13.post2' 'torch-c-dlpack-ext==0.1.5' \
  'cuda-python==13.3.1'
export FLASH_ATTENTION_ROOT="$PWD/refs/kernels/flash-attention"
export PYTHONPATH="$PWD:$FLASH_ATTENTION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

Planner 使用该版本的内部 forward 接口，因此不应任意替换为另一个 `flash-attn`
release。CuTe 无法导入时，相关计划会退回普通 SDPA；运行成功不一定表示优化已启用。
不要把旧实验目录中的 streaming/segmented-KV 修改复制过来，它们不属于此安装路径。

将 `CUDA_HOME`、`NVSHMEM_HOME`、`LD_LIBRARY_PATH` 和 `PYTHONPATH` 设置保存在
部署环境或作业脚本中。额外依赖通过 `uv pip` 安装；再次执行 `uv sync` 可能移除它们
或替换扩展，需重新执行相应安装步骤。运行命令直接使用 `.venv/bin/python`，或使用
`uv run --no-sync`。

## 4. 验证安装

先检查实际加载位置与必需算子：

```bash
python - <<'PY'
import torch
import fast_ulysses
from chitu_diffusion.parallel.cp.fast.attention.cute import load_cute_flash_attn_fwd

print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("Fast CP:", fast_ulysses.__file__)
for name in ("fast_cp_device_info", "all_gather_kv_4d",
             "kv_projection_prepare", "kv_projection_send_ce_group",
             "projection_sync_prepare", "projection_sync"):
    assert hasattr(torch.ops.fast_ulysses, name), f"missing operator: {name}"
print("CuTe:", load_cute_flash_attn_fwd().__module__)
PY
```

在有两张 H20 的节点上运行小规模正确性与性能检查：

```bash
export OMP_NUM_THREADS=1 NCCL_GRAPH_MIXING_SUPPORT=0 NCCL_GRAPH_REGISTER=0
export TORCH_NCCL_ENABLE_MONITORING=0
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m tools.benchmarks.fast_cp_frameworks.chitu \
  --method fast_agkv --sequence 4096 --heads 40 --head-dim 128 \
  --batch 1 --component e2e --warmup 3 --iterations 10 --repeats 3 \
  --output outputs/fast-cp/install-smoke.json
```

Slurm 集群可在最后一条命令前加
`srun -p debug --exclusive --nodes=1 --ntasks=1 --gres=gpu:2 --cpus-per-task=16`。
JSON 应有 `status: complete`、`accuracy.checked: true`；查看
`implementation.attention_plan` 确认实际计划。这里测试 Q/K/V 到 O，不包含 projection。
全局序列长必须能被 CP 整除；修改参数即可测试其他形状。

完整 wrapper 验收需要独占 8 张 H20，使用真实 Wan 1.3B/14B shape 和 Qwen/ZImage
shape，无需 checkpoint：

```bash
export FAST_CP_BUILD_ROOT="$PWD/refs/fast-cp-build"
bash chitu_diffusion/parallel/cp/fast/test/run_wrapper.sh outputs/fast-cp/install-wrapper
```

`FAST_CP_BUILD_ROOT` 覆盖测试脚本原有的本地 checkout 默认值。若 checkout 下没有
`.deps/`，脚本会保留上面设置的 `LD_LIBRARY_PATH`。Slurm 提交和验收边界见
[测试说明](https://github.com/thu-pacman/ChituDiffusion/blob/kernels/chitu_diffusion/parallel/cp/fast/test/README.md)。

## 5. 启用方式

**已有 pipeline：选择通信 transport。** 例如四卡 AGKV：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m chitu_diffusion.cli generate --model zimage --model-path /path/to/Z-Image \
  --attention-mode agkv --agkv-transport fast_agkv --no-cfg-parallel \
  --output outputs/zimage-fast-agkv.png
```

Ulysses 使用 `--attention-mode ulysses --ulysses-transport fast_ulysses`。
`--no-cfg-parallel` 避免四个 rank 被分给 CFG，使该示例使用 CP4。
默认 transport 是 Torch/NCCL；也可通过 `CHITU_AGKV_TRANSPORT` 或
`CHITU_ULYSSES_TRANSPORT` 设置。Ulysses 默认 `CHITU_FAST_ULYSSES_TRANSFER=auto`，
按实际 shape 比较 SM/TMA/CE。

**完整优化区域：显式接入 wrapper。** 先由应用完成 `dist.init_process_group` 和
CP 分组，再调用 `init_fast_cp(cp_group, device=...)`，用模型 adapter 包装 attention，
集体 `prepare()` 后执行。可用 adapter 为 `WanAdapter`、`QwenImageAdapter` 和
`ZImageAdapter`；完整示例与输入分片契约见
[wrapper 用法](https://github.com/thu-pacman/ChituDiffusion/blob/kernels/chitu_diffusion/parallel/cp/fast/README.md#使用-wrapper)。
它覆盖 Planner、NVSHMEM 和模型级 packed/CE overlap 调度。CLI transport 开关
不会自动安装这个 wrapper，也不能单独复现它的全部性能收益。

## 常见问题与验收状态

| 现象 | 检查方式 |
| --- | --- |
| 缺少 `fast_cp_device_info` / `kv_projection_*` | 加载了上游或旧扩展；检查 `fast_ulysses.__file__`，应用 overlay 后重编译 |
| `libnvshmem_host.so` 无法加载 | 检查 `NVSHMEM_HOME` 和动态库搜索路径，构建与运行使用同一安装 |
| `undefined symbol` / `no kernel image` | 核对 Python/Torch/CUDA ABI 与目标 SM，重新构建 |
| CuTe 导入错误或计划退回 SDPA | 检查源码版本、CUTLASS DSL 依赖和 `PYTHONPATH` |
| P2P 不支持、wrapper 选择 origin | 核对可见 GPU 拓扑、硬件 profile 和输入契约；普通 NCCL 可作为兼容路径 |
| 对称堆不足 | 增加 `init_fast_cp(pool_bytes=...)` 或 transport 的 pool 配置；无需按每层单独建堆 |

当前已有 H20 多卡运行回归；**尚未完成本指南在全新环境中的完整构建与 wrapper
验收**。此前 GPU 测试复用了带实验改动的扩展和 CuTe checkout，且 Torch 版本与当前
项目锁定版本不同。因此该指南给出可检查的安装流程，不将旧测试结果视为干净安装的
证明；新部署应至少完成本页的 smoke test，发布前完成完整矩阵。

本次文档核对已验证：固定上游版本的干净源码可应用 overlay，安装器 3 项测试通过；
干净 CuTe 源码可导入，具有 Planner 所需的 forward/merge 接口。这些检查不涉及
新环境的 CUDA 编译或 kernel 执行。
