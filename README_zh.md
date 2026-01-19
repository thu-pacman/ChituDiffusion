# Smart-Diffusion

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
# 1. 只安装flash_attn
uv sync -v 2>&1 | tee uv_sync.log

# 2. 安装sparge attn和sage attn以加速量化注意力。
# 编译时间约为10分钟（32核，256GB内存）
uv sync -v --all-extras 2>&1 | tee build.log 
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

测试脚本：`chitu/diffusion/test_generate.py`
单卡/分布式启动：`bash srun_wan_demo.sh <num_gpus>`

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
| `PAB` | [Pyramid Attention Broadcast](https://oahzxl.github.io/PAB/)(ICLR25 ) | 待测试。 |
---

你可以在`DiffusionUserParams`中设置`flexcache`，如下所示：
```
DiffusionUserParams(
    role="Alex",
    prompt="A cat walking on grass",
    ...
    flexcache='<cache_type>',
)
```