# ChituDiffusion 离线安装（Python 3.12 / CUDA 13.0）

本文冻结 ChituDiffusion、原生 Ulysses、MiniMax-H3 DiT 所需环境。目标机器为
Linux x86_64、Python 3.12、CUDA 13.0、H20（SM90）或 Pro 5000（SM120）。

## 目标版本

| 包 | 版本 |
| --- | --- |
| torch / torchvision / torchaudio | 2.10.0+cu130 / 0.25.0+cu130 / 2.10.0+cu130 |
| triton | 3.6.0 |
| diffusers / accelerate | 0.38.0 / 1.13.0 |
| transformers | 5.12.1 |
| numpy / scipy | 2.2.6 / 1.17.1 |
| safetensors | 0.8.0 |
| einops | 0.8.2 |
| flash-attn-4 | 4.0.0b25 |
| nvidia-cutlass-dsl[cu13] | 4.6.0.dev0 |
| quack-kernels | 0.5.3 |

环境固定在 torch 2.10.0+cu130。现有 vLLM、FlashInfer、DeepGEMM 和 hpc-ops
包含 torch 2.10 ABI 产物，不能直接升级到 torch 2.11。

不安装 `flash_attn`（FlashAttention 2）或 `yunchang`。Ulysses collective
由 ChituDiffusion 基于 `torch.distributed.all_to_all_single` 原生实现。
Ring 策略仍保留，但其 CUDA 实现需要后续用原生 P2P 和 LSE merge 重写。

## 已生成的离线包

```text
/cfs_cloud_code/royroychen/wheelhouse/
├── wheelhouse-chitudiffusion-py312-cu130-sm120.tar.zst
├── wheelhouse-chitudiffusion-py312-cu130-sm120.tar.zst.sha256
└── wheelhouse-chitudiffusion-py312-cu130-sm120/
    ├── wheels/
    ├── install.sh
    ├── verify.sh
    ├── requirements-offline.txt
    ├── requirements-core.txt
    ├── requirements-lock.txt
    ├── WHEELS.txt
    ├── SHA256SUMS
    └── BUILD-INFO.txt
```

包只面向 **Linux x86_64 + Python 3.12 + CUDA 13 兼容驱动**。它不包含
ChituDiffusion 源码，安装时使用 CFS 上当前工作目录，因此不会覆盖未提交代码。

在目标机器执行：

```bash
cd /cfs_cloud_code/royroychen/wheelhouse
sha256sum -c wheelhouse-chitudiffusion-py312-cu130-sm120.tar.zst.sha256
mkdir -p /dockerdata
tar -I zstd -xf \
  wheelhouse-chitudiffusion-py312-cu130-sm120.tar.zst \
  -C /dockerdata

/dockerdata/wheelhouse-chitudiffusion-py312-cu130-sm120/install.sh \
  /dockerdata/chitudiffusion-venv-sm120 \
  /cfs_cloud_code/royroychen/Chitu/ChituDiffusion

/dockerdata/wheelhouse-chitudiffusion-py312-cu130-sm120/verify.sh \
  /dockerdata/chitudiffusion-venv-sm120
```

安装脚本会校验全部 wheel，以 `--no-index` 安装，并以 editable 模式注册当前
ChituDiffusion 源码。若 Python 3.12 不在 PATH，可设置
`PYTHON_BIN=/path/to/python3.12`。

安装器先解析核心依赖，再安装 Cutlass DSL，最后以 `--no-deps` 固定
FA4 b25 和 Quack 0.5.3，防止解析器替换已冻结的 Torch/CUDA 组件。
`verify.sh` 会执行完整 `pip check`。

## 在联网机器重新生成 wheelhouse

推荐直接运行仓库中的构建脚本：

```bash
cd /cfs_cloud_code/royroychen/Chitu/ChituDiffusion
BUNDLE_NAME=wheelhouse-chitudiffusion-py312-cu130-sm120 \
BUNDLE_PARENT=/cfs_cloud_code/royroychen/wheelhouse \
  bash offline/build-wheelhouse.sh
```

脚本只下载 Linux x86_64 / Python 3.12 二进制 wheel，并在无网络容器中完成
安装和静态验证。不要把三个 FA4 兼容包放进同一次普通 pip 解析；构建脚本会
先解析 `requirements-core.txt`，固定 `cuda-python==13.0.3` 以兼容
Torch 的 `cuda-bindings==13.0.3`，再分别下载 Cutlass DSL 和 FA4/Quack。

然后把本仓库和 wheelhouse 一起打包：

```bash
cd wheelhouse-chitudiffusion-py312-cu130-sm120
sha256sum wheels/* > SHA256SUMS
cd ..
tar --zstd -cf wheelhouse-chitudiffusion-py312-cu130-sm120.tar.zst \
  wheelhouse-chitudiffusion-py312-cu130-sm120
sha256sum wheelhouse-chitudiffusion-py312-cu130-sm120.tar.zst \
  > wheelhouse-chitudiffusion-py312-cu130-sm120.tar.zst.sha256
```

`flash-attn-4` 是 CuTeDSL 实现，不含绑定特定 torch ABI 的扩展；但其运行依赖
`nvidia-cutlass-dsl==4.6.0.dev0` 和 `quack-kernels==0.5.3`，必须一起下载。
不要使用解析器可能选择的 `cutlass-dsl 4.7.0 + quack-kernels 0.5.0`：
该组合在导入时会因 `cute.core.ThrMma` API 不兼容而失败。

## 目标机器离线安装

```bash
sha256sum -c wheelhouse-chitudiffusion-py312-cu130-sm120.tar.zst.sha256
tar --zstd -xf wheelhouse-chitudiffusion-py312-cu130-sm120.tar.zst

wheelhouse-chitudiffusion-py312-cu130-sm120/install.sh \
  /dockerdata/chitudiffusion-venv-sm120 \
  /cfs_cloud_code/royroychen/Chitu/ChituDiffusion

wheelhouse-chitudiffusion-py312-cu130-sm120/verify.sh \
  /dockerdata/chitudiffusion-venv-sm120
```

## 验证

优先运行离线包内的 `verify.sh`。它会检查固定版本、CUDA 13、SM90/SM120，并在
GPU 可用时执行真实 FA4 varlen smoke test；无 GPU 构建机只做静态检查。

```bash
/dockerdata/chitudiffusion-venv-sm120/bin/python - <<'PY'
import torch
from flash_attn.cute.interface import flash_attn_varlen_func

assert torch.__version__.startswith("2.10.0"), torch.__version__
assert torch.version.cuda == "13.0", torch.version.cuda
assert torch.cuda.is_available()
capability = torch.cuda.get_device_capability()
assert capability in {(9, 0), (12, 0)}, capability

q = torch.randn(64, 8, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
cu = torch.tensor([0, 48, 64], device="cuda", dtype=torch.int32)
result = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu,
    cu_seqlens_k=cu,
    max_seqlen_q=48,
    max_seqlen_k=48,
    causal=False,
)
out = result[0] if isinstance(result, tuple) else result
assert out.shape == q.shape
assert torch.isfinite(out).all()
print("FA4 varlen smoke test passed:", out.shape)
PY

/dockerdata/chitudiffusion-venv-sm120/bin/python -m pytest -q
```

