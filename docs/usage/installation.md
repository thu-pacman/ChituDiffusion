# 安装

ChituDiffusion 支持 Linux、Python 3.12 或 3.13，并要求 NVIDIA GPU 和可用的 CUDA
环境。项目使用 `pyproject.toml` 管理依赖和构建。

## 基础环境

克隆仓库后，先检查 `pyproject.toml` 中的 PyTorch source：

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu130" }
torchvision = { index = "pytorch-cu130" }
```

仓库已定义 `pytorch-cu124`、`pytorch-cu126`、`pytorch-cu128` 和
`pytorch-cu130`。把 `torch` 与 `torchvision` 改为机器对应的 index，然后安装：

```bash
uv sync --group dev
source .venv/bin/activate
chitu --help
```

安装结果包含 ChituDiffusion、Torch、Diffusers、测试、构建和文档工具。

## 可选依赖

Fast CP 需要 NVSHMEM、带 Chitu 源码 overlay 的 Fast Ulysses 扩展，以及供
Attention Planner 使用的 CuTe 后端。按[Fast CP 安装与启用](fast-cp-installation.md)
完成构建、安装和多卡验证。单独安装 `--extra fast-ulysses` 不包含完整 Chitu 扩展；
只使用单卡或 NCCL CP 时不需要它。

其他 attention backend 按需安装：

```bash
uv sync --group dev --extra flash
```

## 检查安装

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
chitu --help
python -m pytest -q -m "not gpu and not distributed and not benchmark"
```
