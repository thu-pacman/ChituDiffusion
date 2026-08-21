# Omni Baseline Environments

vLLM-Omni 与 SGLang Diffusion 使用独立虚拟环境，避免其固定的
PyTorch、Transformers、Diffusers 和 CUDA 扩展覆盖 ChituDiffusion 主环境。
两个环境均位于仓库根目录并被 `.gitignore` 排除。

| Baseline | 环境路径 | 框架版本 | PyTorch / CUDA |
| --- | --- | --- | --- |
| vLLM-Omni | `.venv-vllm-omni` | `vllm-omni==0.26.0`, `vllm==0.26.0` | `2.11.0+cu130` / `13.0` |
| SGLang Diffusion | `.venv-sglang-diffusion` | `sglang[diffusion]==0.5.16` | `2.11.0+cu130` / `13.0` |

## 安装

从仓库根目录执行：

```bash
uv venv --python 3.12 --seed .venv-vllm-omni
uv pip install \
  --python .venv-vllm-omni/bin/python \
  "vllm-omni==0.26.0"

# 登录节点不可见 GPU 时，--torch-backend=auto 会错误选择 CPU wheel。
# 因此显式安装 CUDA 13.0 PyTorch 和 vLLM 官方 CUDA release wheel。
uv pip install \
  --python .venv-vllm-omni/bin/python \
  --torch-backend=cu130 \
  --reinstall-package torch \
  --reinstall-package torchvision \
  --reinstall-package torchaudio \
  --reinstall-package vllm \
  "https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0-cp38-abi3-manylinux_2_28_x86_64.whl" \
  "torch==2.11.0" torchvision torchaudio

uv venv --python 3.12 --seed .venv-sglang-diffusion
uv pip install \
  --python .venv-sglang-diffusion/bin/python \
  --prerelease=allow \
  "sglang[diffusion]==0.5.16"
```

`--prerelease=allow` 是 SGLang Diffusion 的 `flash-attn-4` 等预发布依赖所需。
不要在 `../ChituDiffusion/.venv` 中执行这些安装命令。

## 验证

基础依赖一致性：

```bash
.venv-vllm-omni/bin/python -m pip check
.venv-sglang-diffusion/bin/python -m pip check
```

H20 节点验证同时覆盖 CUDA 可见性、原生扩展和 CP attention 模块：

```bash
srun -p debug --exclusive --nodes 1 --ntasks 1 --gres gpu:1 \
  .venv-vllm-omni/bin/python -c \
  'import torch, vllm._C_stable_libtorch, vllm_omni.diffusion.attention.parallel.ulysses; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'

srun -p debug --exclusive --nodes 1 --ntasks 1 --gres gpu:1 \
  .venv-sglang-diffusion/bin/python -c \
  'import torch, sgl_kernel, sglang.multimodal_gen.runtime.layers.usp; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
```

2026-08-09 的实际验证结果：两套环境均通过 `pip check`，在 NVIDIA H20
上识别为 compute capability `9.0`，CUDA 可用；vLLM stable-ABI 扩展、
vLLM-Omni Ulysses、SGLang kernel 和 diffusion USP 模块均成功导入。

## 隔离约束

- 基准脚本必须显式选择对应环境的 `bin/python`，不要依赖已激活 shell。
- 两套环境独立升级；升级任一框架后需重新记录完整版本并重跑 H20 import gate。
- 主 ChituDiffusion 环境继续用于 Fast CP；Omni 环境只用于各自上游 baseline，
  避免用不同 Torch/CUDA 栈的数据冒充同环境对照。
