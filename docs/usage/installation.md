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

## 受限网络与文件系统

`uv sync` 解析项目时也会解析未启用的 extras，因此基础安装仍可能访问
`fast-ulysses` 的 Git 源。若输出长时间停在创建 `.venv`，先检查 GitHub
连接；仅省略 `--extra fast-ulysses` 不会跳过这个解析步骤。

GitHub HTTPS 不通、但已配置 GitHub SSH key 的环境，可以将
`[tool.uv.sources]` 中的对应条目改为下面的显式 SSH URL，保留固定 revision：

```toml
fast-ulysses = { git = "ssh://git@github.com/triple-mu/fast-ulysses.git", rev = "6e5dcb24dc44e781ac3091d1d9b3f9fef314fb87" }
```

若当前条目已经是 SSH URL，无需再改。不要只依赖 Git CLI 的 `insteadOf`
配置：本次受限环境中它没有改写 uv 实际访问的 HTTPS 地址。

完全无法访问 GitHub、且只需要单卡或 Torch/NCCL CP 时，在本地
`pyproject.toml` 中同时移除以下三项，再执行 `uv sync --group dev`：

- `[project.optional-dependencies]` 中的 `fast-ulysses = ["fast-ulysses"]`。
- `[tool.uv.sources]` 中的 `fast-ulysses` Git 源。
- `[tool.uv.extra-build-dependencies]` 中的 `fast-ulysses` 构建依赖。

只删除 Git 源仍会解析同名 PyPI 依赖，不能保证基础安装独立于这个可选扩展。
之后要启用 Fast CP 时，先恢复这三项配置。其余依赖的包索引仍需可访问。

部分 ZFS 环境会在安装 wheel 时出现 `Failed to clone`、
`Resource temporarily unavailable (os error 11)`。改用文件复制：

```bash
UV_LINK_MODE=copy uv sync --group dev
```

使用 `hf-mirror.com` 下载权重时，若 Xet CAS 返回 401，可关闭 Xet：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --local-dir /path/to/Wan2.1-T2V-1.3B-Diffusers
```

下载完成后通过 `--model-path` 指向本地目录。

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

### Transformers 导入提示

当前固定的 `transformers==5.12.1` 在导入 Qwen2.5-VL 模块时，可能直接打印
`Qwen2_5_VLCausalLMOutputWithPast.__init__` 的 `loss`、`logits` 未写入 docstring
的 `[ERROR]` 提示。这两条来自依赖的文档生成检查，不表示模型加载或推理失败。
Chitu 的公共入口按需导入模型，`chitu --help` 和不需要该模块的模型导入不再
触发这些提示；实际使用该依赖路径时仍可能看到它们。其他错误需单独排查。
