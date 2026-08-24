# 快速开始

## 安装

要求 Linux、Python 3.12、NVIDIA GPU 和可用的 CUDA 环境。先按机器 CUDA 版本选择
`pyproject.toml` 中的 PyTorch index，再安装：

```bash
uv sync --group dev
source .venv/bin/activate
chitu --help
```

`pyproject.toml` 是唯一构建配置。项目不提供 `setup.py`。

## 单卡生成

```bash
chitu generate \
  --model zimage \
  --model-path /path/to/Z-Image \
  --prompt "a red cube on a white table" \
  --output outputs/zimage.png
```

## 静态多卡生成

所有 rank 必须执行同一个命令，只有 leader 写出结果：

```bash
torchrun --standalone --nproc-per-node=4 -m chitu_diffusion.cli \
  generate \
  --model flux1 \
  --model-path /path/to/FLUX.1-dev \
  --prompt "a lighthouse above a stormy sea" \
  --output outputs/flux1.png
```

继续阅读[生成指南](guides/generation.md)、[服务指南](guides/serving.md)和
[并行架构](architecture/parallel.md)。
