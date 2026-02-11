# 安装指南

本指南将帮助您在系统上安装 Smart-Diffusion。

## 前置要求

在安装 Smart-Diffusion 之前，请确保您具备：

- **Python**：3.12 或更高版本
- **CUDA**：12.4 或更高版本（推荐 12.8）
- **GPU**：具有以下计算能力的 NVIDIA GPU：
    - 8.0+（Ampere：A100、A10 等）
    - 9.0+（Hopper：H100、H20 等）
    - 9.0+（Blackwell：B100、B200、5090 等）

## 安装方法

### 方法 1：使用 uv（推荐）

`uv` 是一个快速的 Python 包管理器，可简化安装过程。

#### 步骤 1：克隆仓库

```bash
git clone git@github.com:chen-yy20/SmartDiffusion.git
cd SmartDiffusion
git submodule update --init --recursive
```

#### 步骤 2：安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

更多安装选项，请参阅 [uv 文档](https://docs.astral.sh/uv/getting-started/installation/)。

#### 步骤 3：配置 CUDA 版本

检查您的 CUDA 版本：

```bash
nvcc --version
```

编辑 `pyproject.toml` 以匹配您的 CUDA 版本。例如，对于 CUDA 12.8：

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
```

可用的 CUDA 版本：`cu124`、`cu126`、`cu128`、`cu130`

#### 步骤 4：配置 GPU 架构

根据您的 GPU 在 `pyproject.toml` 中设置 `TORCH_CUDA_ARCH_LIST`：

```toml
[tool.uv.extra-build-variables]
sageattention = { 
    EXT_PARALLEL= "4", 
    NVCC_APPEND_FLAGS="--threads 8", 
    MAX_JOBS="32", 
    "TORCH_CUDA_ARCH_LIST" = "8.0;9.0"  # 根据您的 GPU 调整
}
spas_sage_attn = { 
    EXT_PARALLEL= "4", 
    NVCC_APPEND_FLAGS="--threads 8", 
    MAX_JOBS="32", 
    "TORCH_CUDA_ARCH_LIST" = "8.0;9.0"  # 根据您的 GPU 调整
}
```

**GPU 架构对应表**：
- Ampere (A100, A10): `8.0`
- Hopper (H100): `9.0`
- Blackwell (B100, 5090): `9.0`

#### 步骤 5：安装依赖

```bash
uv sync
```

这将安装所有必需的依赖项，包括 PyTorch、diffusers 和其他库。

#### 步骤 6：下载模型

Smart-Diffusion 需要预训练模型。您可以：

**选项 A：自动下载**（需要 Hugging Face 访问权限）

```bash
# 设置 Hugging Face token
export HF_TOKEN=your_token_here

# 模型将在首次运行时自动下载
```

**选项 B：手动下载**

```bash
# 从 ModelScope 下载（中国用户推荐）
git clone https://www.modelscope.cn/Wan-AI/Wan2.1-T2V-1.3B.git models/Wan2.1-T2V-1.3B
git clone https://www.modelscope.cn/Wan-AI/Wan2.1-T2V-14B.git models/Wan2.1-T2V-14B
```

### 方法 2：使用 pip

如果您更喜欢使用传统的 pip：

```bash
# 克隆仓库
git clone git@github.com:chen-yy20/SmartDiffusion.git
cd SmartDiffusion
git submodule update --init --recursive

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装 PyTorch（根据您的 CUDA 版本调整）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 安装其他依赖
pip install -r requirements.txt

# 安装 Smart-Diffusion
pip install -e .
```

## 验证安装

验证安装是否成功：

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

预期输出：
```
PyTorch: 2.6.0+cu128
CUDA: True
```

## 故障排除

### CUDA 不可用

**症状**：`torch.cuda.is_available()` 返回 `False`

**解决方案**：
1. 验证 NVIDIA 驱动：`nvidia-smi`
2. 检查 CUDA 版本匹配
3. 重新安装正确的 PyTorch 版本

### 编译错误

**症状**：安装 SageAttention 时出现编译错误

**解决方案**：
1. 确保已安装 CUDA toolkit：`nvcc --version`
2. 检查 GPU 架构设置
3. 增加编译资源（如果内存不足）

### 内存不足

**症状**：安装过程中系统挂起

**解决方案**：
1. 减少 `MAX_JOBS`
2. 一次安装一个包
3. 增加系统交换空间

## 下一步

安装完成后：

1. [运行快速入门示例](quick-start.zh.md)
2. [配置您的设置](configuration.md)
3. [探索用户指南](../user-guide/basic-usage.md)

## 卸载

要卸载 Smart-Diffusion：

```bash
# 如果使用 uv
uv cache clean

# 如果使用 pip
pip uninstall smart-diffusion
```

## 更新

要更新到最新版本：

```bash
git pull origin main
git submodule update --init --recursive
uv sync  # 或 pip install -e . --upgrade
```

## 获取帮助

如果您遇到问题：

1. 检查 [常见问题](../faq.zh.md)
2. 在 [GitHub Issues](https://github.com/chen-yy20/SmartDiffusion/issues) 上搜索
3. 加入 [讨论](https://github.com/chen-yy20/SmartDiffusion/discussions)
