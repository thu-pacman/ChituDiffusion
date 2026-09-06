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

Fast CP 需要 Fast Ulysses 与 NVSHMEM：

```bash
uv sync --group dev --extra fast-ulysses
python tools/install/install_fast_agkv.py refs/fast-ulysses
```

第二步把 `chitu_diffusion/parallel/cp/fast/csrc/` 下的扩展源码装入 checkout。
Fast transport 必须针对目标 CUDA、NVSHMEM 和 GPU 架构编译。只使用单卡或 NCCL CP
时不需要这些扩展。

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
