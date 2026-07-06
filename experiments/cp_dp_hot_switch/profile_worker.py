#!/usr/bin/env python3
"""Profile the Z-Image worker roofline (compute) and NVLink all-to-all (comm).

Two independent modes feed the decoupled cost model in
``chitu_diffusion/runtime/cost_model.py``:

* ``--mode compute`` (single GPU): instantiate the real Z-Image DiT transformer,
  sweep a ``batch x seq_len`` grid of synthetic denoise-step forwards, and record
  the median per-step latency. This is the worker roofline: it shows where the
  worker is memory-bound (small B*L, batching is ~free) vs compute-bound (large
  B*L, only sequence sharding helps).

* ``--mode comm`` (>=2 GPUs, torchrun/srun): measure effective NVLink all-to-all
  bandwidth for the Ulysses layout, to calibrate ``CommModel.nvlink_gbps``.

Both modes merge their results into a single ``cost_model.json`` (via
``CostModel.save``), which the simulator and scheduler then consume.

Examples::

    # compute grid on one debug GPU
    srun -p debug --gres=gpu:1 python experiments/cp_dp_hot_switch/profile_worker.py \
        --mode compute --ckpt /home/chenyy/WORK/models/Z-Image \
        --out experiments/cp_dp_hot_switch/cost_model.json

    # comm calibration on 4 GPUs
    torchrun --nproc_per_node=4 experiments/cp_dp_hot_switch/profile_worker.py \
        --mode comm --out experiments/cp_dp_hot_switch/cost_model.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from chitu_diffusion.runtime.cost_model import (  # noqa: E402
    CommModel,
    CostModel,
    RooflineComputeModel,
    Z_IMAGE_SPEC,
    image_token_count,
)


DEFAULT_RESOLUTIONS = [(512, 512), (768, 768), (1024, 1024), (1280, 1280), (1536, 1536)]
DEFAULT_BATCHES = [1, 2, 4, 8]


def _extract_ckpt_from_argv(argv: list[str]) -> str | None:
    for token in argv:
        if token.startswith("models.ckpt_dir="):
            return token.split("=", 1)[1]
    return None


def _load_existing(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


# --------------------------------------------------------------------------- #
# Compute grid (single GPU)
# --------------------------------------------------------------------------- #
def profile_compute(args) -> None:
    import torch

    ckpt = args.ckpt or _extract_ckpt_from_argv(sys.argv) or os.environ.get("CHITU_Z_IMAGE_CKPT")
    if not ckpt:
        raise SystemExit("compute mode needs --ckpt <Z-Image dir> (or models.ckpt_dir=... / CHITU_Z_IMAGE_CKPT)")

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

    print(f"[compute] loading Z-Image transformer from {ckpt}", flush=True)
    transformer = (
        ZImageTransformer2DModel.from_pretrained(
            ckpt, subfolder="transformer", torch_dtype=dtype, local_files_only=True
        )
        .to(device=device, dtype=dtype)
        .eval()
        .requires_grad_(False)
    )
    # Match the serving attention backend (flash_attn) instead of the diffusers
    # native SDPA path, so the roofline reflects production kernels. Falls back
    # to native attention if the Chitu processor cannot be installed standalone.
    attn_impl = args.attn
    if attn_impl and attn_impl != "native":
        try:
            from chitu_diffusion.modules.attention.diffusion_attn_backend import DiffusionAttnBackend
            from chitu_diffusion.modules.attention.z_image_attention import ZImageChituAttnProcessor

            processor = ZImageChituAttnProcessor(DiffusionAttnBackend(attn_impl))
            for block in list(transformer.noise_refiner) + list(transformer.context_refiner) + list(transformer.layers):
                block.attention.set_processor(processor)
            print(f"[compute] installed Chitu attention processor: {attn_impl}", flush=True)
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[compute] WARNING: could not install {attn_impl} processor ({exc!r}); using native attention", flush=True)
    in_channels = int(transformer.config.in_channels)
    cap_dim = int(transformer.config.cap_feat_dim)
    text_tokens = int(args.text_tokens)

    def one_forward(batch: int, width: int, height: int) -> None:
        lh, lw = height // 8, width // 8
        latents = torch.randn(batch, in_channels, lh, lw, device=device, dtype=dtype)
        latent_list = list(latents.unsqueeze(2).unbind(dim=0))
        tvec = torch.full((batch,), 0.5, device=device, dtype=dtype)
        embeds = [torch.randn(text_tokens, cap_dim, device=device, dtype=dtype) for _ in range(batch)]
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            transformer(latent_list, tvec, embeds, return_dict=False)

    grid: dict[tuple[int, int], float] = {}
    resolutions = args.resolutions or DEFAULT_RESOLUTIONS
    batches = args.batches or DEFAULT_BATCHES
    for (width, height) in resolutions:
        img_tokens = image_token_count(width, height)
        seq_len = text_tokens + img_tokens
        for batch in batches:
            try:
                for _ in range(args.warmup):
                    one_forward(batch, width, height)
                torch.cuda.synchronize()
                samples = []
                for _ in range(args.iters):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    one_forward(batch, width, height)
                    end.record()
                    torch.cuda.synchronize()
                    samples.append(start.elapsed_time(end))
                ms = statistics.median(samples)
                grid[(batch, seq_len)] = ms
                print(
                    f"[compute] B={batch:2d} {width}x{height} img_tokens={img_tokens} "
                    f"seq_len={seq_len} -> {ms:.2f} ms",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[compute] B={batch} {width}x{height} OOM, skipping", flush=True)

    out = Path(args.out)
    data = _load_existing(out)
    model = CostModel.from_dict(data) if data else CostModel(
        compute=RooflineComputeModel(spec=Z_IMAGE_SPEC), comm=CommModel(spec=Z_IMAGE_SPEC)
    )
    model.compute.grid.update(grid)
    model.save(out)
    print(f"[compute] wrote {len(grid)} grid points to {out}", flush=True)


# --------------------------------------------------------------------------- #
# Comm calibration (multi GPU)
# --------------------------------------------------------------------------- #
def profile_comm(args) -> None:
    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    spec = Z_IMAGE_SPEC

    measured_gbps: list[float] = []
    for (width, height) in (args.resolutions or DEFAULT_RESOLUTIONS):
        img_tokens = image_token_count(width, height)
        if img_tokens % world != 0:
            continue
        seq_local = img_tokens // world
        buf = torch.randn(1, seq_local, spec.dim, device=device, dtype=dtype)

        def a2a() -> None:
            out = torch.empty_like(buf)
            dist.all_to_all_single(out, buf.contiguous())

        for _ in range(args.warmup):
            a2a()
        torch.cuda.synchronize()
        dist.barrier()
        samples = []
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            a2a()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        ms = statistics.median(samples)
        moved_bytes = buf.numel() * buf.element_size() * (world - 1) / world
        gbps = moved_bytes / (ms / 1e3) / 1e9
        measured_gbps.append(gbps)
        if rank == 0:
            print(f"[comm] world={world} {width}x{height} seq_local={seq_local} -> {ms:.3f} ms  {gbps:.1f} GB/s", flush=True)

    if rank == 0 and measured_gbps:
        eff = statistics.median(measured_gbps)
        out = Path(args.out)
        data = _load_existing(out)
        model = CostModel.from_dict(data) if data else CostModel(
            compute=RooflineComputeModel(spec=spec), comm=CommModel(spec=spec)
        )
        model.comm.nvlink_gbps = eff
        model.save(out)
        print(f"[comm] calibrated nvlink_gbps={eff:.1f} (world={world}) -> {out}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def _parse_pairs(text: str | None):
    if not text:
        return None
    pairs = []
    for tok in text.split(","):
        w, h = tok.lower().split("x")
        pairs.append((int(w), int(h)))
    return pairs


def _parse_ints(text: str | None):
    if not text:
        return None
    return [int(t) for t in text.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["compute", "comm"], default="compute")
    ap.add_argument("--ckpt", default=None, help="Z-Image checkpoint dir (compute mode)")
    ap.add_argument("--out", default=str(Path(_REPO_ROOT) / "experiments/cp_dp_hot_switch/cost_model.json"))
    ap.add_argument("--resolutions", type=_parse_pairs, default=None, help="e.g. 512x512,1024x1024")
    ap.add_argument("--batches", type=_parse_ints, default=None, help="e.g. 1,2,4,8")
    ap.add_argument("--text-tokens", type=int, default=Z_IMAGE_SPEC.text_tokens, dest="text_tokens")
    ap.add_argument("--attn", default="flash_attn", help="attention backend for compute grid: flash_attn|sageattn|torch_sdpa|native")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    args, _unknown = ap.parse_known_args()

    if args.mode == "compute":
        profile_compute(args)
    else:
        profile_comm(args)


if __name__ == "__main__":
    main()
