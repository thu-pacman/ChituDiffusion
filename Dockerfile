FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/chitu-venv \
    UV_LINK_MODE=copy \
    UV_INDEX_URL=https://pypi.org/simple \
    TORCH_CUDA_ARCH_LIST=9.0 \
    FLASH_ATTN_CUDA_ARCHS=90 \
    MAX_JOBS=16

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libglib2.0-0 \
    libgl1 \
    ninja-build \
    pkg-config \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m pip install --break-system-packages uv

WORKDIR /opt/ChituDiffusion

COPY . .

RUN python3.12 -c "from pathlib import Path; p = Path('pyproject.toml'); text = p.read_text().replace('https://pypi.tuna.tsinghua.edu.cn/simple', 'https://pypi.org/simple'); skip = ('vbench', 'sageattention', 'spas_sage_attn', 'flash_attn', 'flashinfer'); text = '\n'.join(line for line in text.splitlines() if not any(item in line for item in skip)) + '\n'; p.write_text(text)"

RUN uv sync --no-dev

ENV PATH="/opt/chitu-venv/bin:${PATH}"

CMD ["bash"]
