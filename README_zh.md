# ChituDiffusion

[英文版](./README.md)

本仓库是Chitu-Diffusion的纯享版。

[Chitu](https://github.com/thu-pacman/chitu )是由清华大学PACMAN团队与清程极智联合开发的高性能LLM推理框架。我们致力于为迅速发展的Diffusion生态系统提供支持。因此，我们在Chitu的API和调度理念下重构了DiT模型，在保持调度灵活性的同时提供极致性能，旨在提供一个真正简单易用的AIGC加速框架。

Chitu-Diffusion目前处于测试和开发阶段。我们正在努力使其变得更好！我们欢迎任何感兴趣的人加入我们的团队，使用、测试并参与开发。

我们目前支持Wan-T2V系列，并持续增加对新模型、算子和算法优化的支持。

# 安装

## 环境

推荐的软件环境：Python3.12，cuda 12.4

根据`chitu/diffusion/requirements.txt`安装并运行：
```
pip install -e .
```

> 建议通过wheel安装Flash Attention：https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.1.post2  

### 使用uv设置环境
推荐使用`uv`来设置环境，以获得更流畅的体验。
#### 克隆仓库并更新sparge/sage attn子模块
```bash
git clone git@github.com:chen-yy20/SmartDiffusion.git
git submodule update --init --recursive
```

#### 安装`uv` 
```bash
curl -LsSf https://astral.sh/uv/install.sh  | sh
```
文档：https://docs.astral.sh/uv/getting-started/installation/ 

#### 指定构建配方
修改`pyproject.toml`中的`[tool.uv.extra-build-variables]`项：
- 指定`TORCH_CUDA_ARCH_LIST`，只为所需的计算架构编译内核。
- flash_attn默认从GitHub源仓库拉取二进制包，如果遇到网络/符号链接问题，可以取消以下注释以从源代码编译（需要32核、256GB内存，大约10分钟）。

```toml
[tool.uv.extra-build-variables]
# flash_attn = { FLASH_ATTN_CUDA_ARCHS = "80",FLASH_ATTENTION_FORCE_BUILD = "TRUE" }
## 根据你的GPU架构设置"TORCH_CUDA_ARCH_LIST"（例如Ampere：8.0 / Hopper：9.0）
sageattention = { EXT_PARALLEL= "4", NVCC_APPEND_FLAGS="--threads 8", MAX_JOBS="32", "TORCH_CUDA_ARCH_LIST" = "9.0"}
spas_sage_attn = { EXT_PARALLEL= "4", NVCC_APPEND_FLAGS="--threads 8", MAX_JOBS="32", "TORCH_CUDA_ARCH_LIST" = "9.0"}
```

#### 一键安装依赖

```bash
# 1. 必装（[project.dependencies] 基础依赖）
uv sync -v 2>&1 | tee uv_sync.log

# 2. 按需安装（[project.optional-dependencies] 可选依赖）
# 安装 SageAttention
uv sync -v --extra sage 2>&1 | tee build_sage.log

# 安装 SpargeAttention
uv sync -v --extra sparge 2>&1 | tee build_sparge.log

# 安装 VBench 评测工具
uv sync -v --extra vbench 2>&1 | tee build_vbench.log

# 安装评测指标依赖（FID/FVD/PSNR/SSIM/LPIPS）
uv sync -v --extra eval 2>&1 | tee build_eval.log

# 扩展一键安装（sage + sparge + vbench + eval）
uv sync -v --all-extras 2>&1 | tee build_full.log
```

## 模型检查点
> 支持的模型-id：
> * Wan-AI/Wan2.1-T2V-1.3B
> * Wan-AI/Wan2.1-T2V-14B
> * Wan-AI/Wan2.2-T2V-A14B

# 运行演示

**模型架构参数**（层数、注意力头数等）是静态的，设置在`chitu/config/models/<diffusion-model>.yaml`中。

**用户参数**（生成步骤、形状等）是动态的。`Chitu`提供了`DiffusionUserParams`，可以在每次请求时设置它们。

**系统参数**（并行性、算子、加速算法等）在`Chitu`的启动参数中设置。

测试脚本：`test/test_generate.py`
统一启动方式（仅支持 srun）：

```bash
bash run.sh system_config.yaml
```

可选覆盖参数（单卡/单机多卡/多机多卡）：

```bash
bash run.sh system_config.yaml --num-nodes 2 --gpus-per-node 8 --cfp 2
```

运行补充说明：
- `parallel.cfp`（或 `--cfp`）当前仅支持 `1` 或 `2`，并映射到 `infer.diffusion.cfg_size`。
- `infer.diffusion.cp_size` 会由 `(num_nodes * gpus_per_node) / cfp` 自动推导。
- `launch.tag` 会导出为 `CHITU_RUN_TAG`，并用于给输出目录名前缀打标。
- `launch.enable_launch_log=true` 时，启动日志会写入 `output.root_dir/launch_<timestamp>.log`。
- 可通过 `CHITU_PYTHON_BIN` 指定运行时 Python；默认优先级为 `.venv/bin/python` -> `python` -> `python3`。

推荐同步配置 `system_config.yaml` 的输出段：

```yaml
output:
    root_dir: outputs
    enable_run_log: true
    enable_timer_dump: true
    hydra_dump_mode: off   # default/video_dir/off
```

其中 `hydra_dump_mode=video_dir` 会将 Hydra 的 `.hydra` 元数据移动到视频输出目录。
当 `enable_timer_dump=true` 时，每次运行会在输出目录写入 `time_stats.csv`。

---

# 魔法参数 !!!

## `infer.attn_type`

**选择你的Diffusion后端**：此参数控制你的注意力后端。Diffusion注意力通常是具有长上下文的3D全注意力，由于其$O(n^2)$复杂度，计算开销很高。我们提供了更智能的注意力实现，以减少注意力开销。

### 注意力类型描述

| attn-type | 描述 | 性能 |
|-------|-------------|-------|
| **flash_attn** | 默认注意力实现。高性能全注意力内核，无精度损失。 | 待测试。 |
| [**sage**](https://github.com/thu-ml/SageAttention ) |  (NIPS25 spotlight)无需训练的量化注意力实现。 | 待测试。 |
| [**sparge**](https://github.com/thu-ml/SpargeAttn ) |  (ICML25)基于sage-attention的无需训练的稀疏量化注意力。  | 待测试。 |
| **auto** | 自动选择最佳注意力后端。 | - |

---

## `infer.diffusion.low_mem_level`

**低内存级别**：此参数控制模型使用的GPU内存比例，有效防止内存溢出（OOM）问题，并允许模型在有限的GPU内存下运行。

### 参数级别描述

| 级别 | 描述 |
|-------|-------------|
| **0** | 所有模型直接加载到GPU中。 |
| **1** | VAE启用分块。 |
| **2** | T5模型卸载到CPU。 |
| **>3** | DIT模型卸载到CPU。 |

---

通过合理设置`infer.diffusion.low_mem_level`，你可以根据可用的GPU内存灵活调整模型加载策略，确保模型在有限资源下高效运行。

## `infer.diffusion.enable_flexcache`

**启用FlexCache**：Diffusion后端初始化FlexCache管理器，统一支持基于特征复用的有损加速算法。

### 参数描述
在启动脚本中设置：`infer.diffusion.enable_flexcache=true`并设置相应的用户参数。

目前支持：

| 方法 | cache_type | 性能 |
|-------|-------------| ---------|
| `teacache` |[Teacache](https://github.com/ali-vilab/TeaCache)(CVPR24-spotlight ) | 待测试。 |
| `pab` | [Pyramid Attention Broadcast](https://oahzxl.github.io/PAB/)(ICLR25 ) | 待测试。 |
| `ditango` | DiTango (ASE + Anchor 门控分组复用) | 待测试。 |

DiTango 当前实现说明：
- Local partition 每步都强制计算，并与 group 状态分开合并，保证稳定性。
- Anchor 判定是 step 级决策，并在 CFG 正负分支间保持一致。
- `cache_ratio` 同时影响 anchor 触发激进程度与全局 ASE 阈值分位更新。
- 策略实现位于 `chitu_diffusion/flex_cache/strategy/ditango/ditango.py`。
- 会在输出目录生成合并决策可视化：`<output_dir>/ditango_policy_step_layer_group.ppm`。
---

推荐使用统一参数接口：

```python
from chitu_diffusion.task import DiffusionUserParams, FlexCacheParams

DiffusionUserParams(
    role="Alex",
    prompt="A cat walking on grass",
    flexcache_params=FlexCacheParams(
        strategy="teacache",  # teacache / pab / ditango
        cache_ratio=0.4,       # 0质量优先, 1速度优先
        warmup=5,
        cooldown=5,
    ),
)
```

兼容旧写法：

```
DiffusionUserParams(
    role="Alex",
    prompt="A cat walking on grass",
    ...
    flexcache='<cache_type>',
)
```

## `eval`

**启用自动评测**：支持多种评测策略组合执行。

在 `system_config.yaml` 或 Hydra 覆盖中配置：

```yaml
eval:
    eval_type: [vbench, fid, psnr]
    reference_path: /path/to/reference_videos
```

可选项：
- `vbench`
- `fid`（需要 `reference_path`）
- `fvd`（需要 `reference_path`）
- `psnr`（需要 `reference_path`）
- `ssim`（需要 `reference_path`）
- `lpips`（需要 `reference_path`）

说明：
- `eval_type: []` 或 `null` 表示关闭评测。
- 当 `reference_path` 缺失/无效时，依赖参考视频的指标会被跳过并打印 warning，不会阻断其他策略。
- 结果输出：`vbench` 写入 `./vbench_out/`；其余指标写入 `./eval_out/`。