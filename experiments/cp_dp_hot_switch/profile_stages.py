#!/usr/bin/env python3
"""Profile the NON-denoise Z-Image pipeline stages (text-encode, VAE-decode, image save).

This extends the M1 worker-roofline work (``profile_worker.py`` /
``cost_model.py`` / ``EXPERIMENT_REPORT.md``) from the denoise DiT step to the
other pipeline stages that the runtime drives per request:

  * ``encode_text``  (z_image.py:435)  -> Qwen3 text encoder forward, [B, 512]
  * ``decode_latents`` (z_image.py:1265) -> AutoencoderKL VAE decode, [B,16,H/8,W/8]
  * ``save_output``  (z_image.py:1315) -> PIL postprocess + PNG encode + disk write

Hypothesis under test: the non-denoise stages are **memory-bound / launch-bound**,
so accumulating several requests and batching these stages (before dispatching to
denoise) would amortize weight reads + kernel launches and pay off. We test that by
sweeping batch size B and looking at the *per-item* latency (latency/B): a sharp
drop with B is the memory-bound signature; a flat per-item latency is compute-bound
(like denoise at 1024², see M1).

The profiler builds the *real* model components once (mirroring
``ZImageRuntimeAdapter._ensure_pipeline`` / ``_encode_prompt`` / ``decode_latents``
exactly) and times each stage directly for controlled batch sizes, WITHOUT going
through the scheduler/generator. It is standalone (no chitu_init / torch.distributed)
so it runs on a single debug GPU, exactly like the M1 ``profile_worker.py --mode
compute`` precedent.

Example (single debug GPU)::

    srun -p debug --gres=gpu:1 \
      /home/chenyy/WORK/cyy/ChituDiffusion/.venv/bin/python \
      experiments/cp_dp_hot_switch/profile_stages.py \
      --ckpt /home/chenyy/WORK/models/Z-Image \
      --out experiments/cp_dp_hot_switch/stage_profile.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Representative prompt (the same one used by test/test_z_image.py). Note the
# Z-Image pipeline pads every prompt to max_sequence_length=512 tokens
# (padding="max_length"), so the text-encode forward shape is [B, 512]
# irrespective of the actual prompt length -- text-encode is a fixed-shape,
# perfectly batchable op.
REPRESENTATIVE_PROMPT = (
    'A compact workstation desk with a small sign reading "Z-Image x ChituDiffusion", '
    "soft morning light, crisp product photography, detailed cables and notebooks."
)

DEFAULT_BATCHES = [1, 2, 4, 8]
DEFAULT_RESOLUTIONS = [(1024, 1024), (512, 512)]
MAX_SEQUENCE_LENGTH = 512


def _median_cuda_ms(fn, warmup: int, iters: int):
    """Return the median GPU latency (ms) of `fn` via CUDA events."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _param_bytes(module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def _nonembedding_param_bytes(model) -> int:
    """Weight bytes that a forward reads *in full* every call (transformer layers).

    Excludes the input-embedding matrix: a forward only gathers B*L rows of it, so
    it is not part of the weight-load traffic that batching amortizes."""
    total = 0
    for name, p in model.named_parameters():
        low = name.lower()
        if "embed" in low or low.endswith("lm_head.weight"):
            continue
        total += p.numel() * p.element_size()
    return total


def build_components(ckpt: str, device, dtype):
    import torch
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from diffusers.image_processor import VaeImageProcessor
    from transformers import Qwen2Tokenizer, Qwen3Model

    print(f"[load] tokenizer  <- {ckpt}", flush=True)
    tokenizer = Qwen2Tokenizer.from_pretrained(ckpt, subfolder="tokenizer", local_files_only=True)

    print(f"[load] text_encoder (Qwen3Model) <- {ckpt}", flush=True)
    text_encoder = (
        Qwen3Model.from_pretrained(ckpt, subfolder="text_encoder", dtype=dtype, local_files_only=True)
        .to(device=device, dtype=dtype)
        .eval()
        .requires_grad_(False)
    )

    print(f"[load] vae (AutoencoderKL) <- {ckpt}", flush=True)
    vae = (
        AutoencoderKL.from_pretrained(ckpt, subfolder="vae", torch_dtype=dtype, local_files_only=True)
        .to(device=device, dtype=dtype)
        .eval()
        .requires_grad_(False)
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)
    return tokenizer, text_encoder, vae, vae_scale_factor, image_processor


def encode_text_forward(tokenizer, text_encoder, prompts, device):
    """Faithful copy of ZImagePipeline._encode_prompt (the batchable op)."""
    import torch

    templated = []
    for prompt_item in prompts:
        messages = [{"role": "user", "content": prompt_item}]
        templated.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        )
    text_inputs = tokenizer(
        templated,
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    ids = text_inputs.input_ids.to(device)
    masks = text_inputs.attention_mask.to(device).bool()
    embeds = text_encoder(input_ids=ids, attention_mask=masks, output_hidden_states=True).hidden_states[-2]
    return [embeds[i][masks[i]] for i in range(len(embeds))]


def profile_text_encode(tokenizer, text_encoder, device, batches, warmup, iters):
    results = []
    # Report the padded sequence length actually fed to the encoder.
    for b in batches:
        prompts = [REPRESENTATIVE_PROMPT] * b
        ms = _median_cuda_ms(lambda: encode_text_forward(tokenizer, text_encoder, prompts, device), warmup, iters)
        results.append({"batch": b, "seq_len": MAX_SEQUENCE_LENGTH, "latency_ms": ms, "per_item_ms": ms / b})
        print(f"[text] B={b:2d} seq={MAX_SEQUENCE_LENGTH} -> {ms:8.2f} ms  ({ms / b:7.2f} ms/item)", flush=True)
    return results


def profile_vae_decode(vae, vae_scale_factor, device, dtype, batches, resolutions, warmup, iters, tiling):
    import torch

    if tiling:
        vae.enable_tiling()
    else:
        if hasattr(vae, "disable_tiling"):
            vae.disable_tiling()
        if hasattr(vae, "disable_slicing"):
            vae.disable_slicing()
    tiling_active = bool(getattr(vae, "use_tiling", False))
    latent_channels = int(vae.config.latent_channels)
    scaling = float(vae.config.scaling_factor)
    shift = float(vae.config.shift_factor)

    results = []
    for (w, h) in resolutions:
        lh, lw = h // vae_scale_factor, w // vae_scale_factor
        for b in batches:
            # Build a latent as decode_latents does: divide by scaling, add shift.
            raw = torch.randn(b, latent_channels, lh, lw, device=device, dtype=dtype)
            z = (raw / scaling) + shift

            def _decode():
                return vae.decode(z, return_dict=False)[0]

            try:
                ms = _median_cuda_ms(_decode, warmup, iters)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[vae ] B={b} {w}x{h} OOM, skipping", flush=True)
                continue
            results.append(
                {
                    "batch": b,
                    "width": w,
                    "height": h,
                    "latent_hw": [lh, lw],
                    "latency_ms": ms,
                    "per_item_ms": ms / b,
                    "tiling": tiling_active,
                }
            )
            print(
                f"[vae ] B={b:2d} {w}x{h} latent={lh}x{lw} tiling={tiling_active} "
                f"-> {ms:8.2f} ms  ({ms / b:7.2f} ms/item)",
                flush=True,
            )
    return results


def profile_image_save(vae, vae_scale_factor, image_processor, device, dtype, warmup, iters, resolution):
    """Wall time per image: PIL postprocess + PNG encode + disk write (CPU/disk)."""
    import torch

    w, h = resolution
    lh, lw = h // vae_scale_factor, w // vae_scale_factor
    scaling = float(vae.config.scaling_factor)
    shift = float(vae.config.shift_factor)
    raw = torch.randn(1, int(vae.config.latent_channels), lh, lw, device=device, dtype=dtype)
    z = (raw / scaling) + shift
    with torch.no_grad():
        decoded = vae.decode(z, return_dict=False)[0]
    images = image_processor.postprocess(decoded, output_type="pil")
    img = images[0]

    tmpdir = tempfile.mkdtemp(prefix="zimg_save_")
    # postprocess-only time (tensor -> PIL), measured separately (CPU).
    post_samples = []
    for i in range(warmup + iters):
        t0 = time.perf_counter()
        _ = image_processor.postprocess(decoded, output_type="pil")
        t1 = time.perf_counter()
        if i >= warmup:
            post_samples.append((t1 - t0) * 1e3)

    # save (PNG encode + disk write) time, matching save_output kwargs.
    save_samples = []
    for i in range(warmup + iters):
        path = os.path.join(tmpdir, f"img_{i}.png")
        t0 = time.perf_counter()
        img.save(path, quality=95, subsampling=0)
        t1 = time.perf_counter()
        if i >= warmup:
            save_samples.append((t1 - t0) * 1e3)

    post_ms = statistics.median(post_samples)
    save_ms = statistics.median(save_samples)
    print(
        f"[save] {w}x{h}  postprocess={post_ms:7.2f} ms  png_encode+write={save_ms:7.2f} ms "
        f"total={post_ms + save_ms:7.2f} ms/image",
        flush=True,
    )
    return {
        "width": w,
        "height": h,
        "postprocess_ms": post_ms,
        "save_ms": save_ms,
        "total_ms": post_ms + save_ms,
    }


def _parse_pairs(text):
    if not text:
        return None
    out = []
    for tok in text.split(","):
        a, b = tok.lower().split("x")
        out.append((int(a), int(b)))
    return out


def _parse_ints(text):
    if not text:
        return None
    return [int(t) for t in text.split(",")]


def main():
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=os.environ.get("CHITU_Z_IMAGE_CKPT", "/home/chenyy/WORK/models/Z-Image"))
    ap.add_argument("--out", default=str(Path(_REPO_ROOT) / "experiments/cp_dp_hot_switch/stage_profile.json"))
    ap.add_argument("--batches", type=_parse_ints, default=None, help="e.g. 1,2,4,8")
    ap.add_argument("--resolutions", type=_parse_pairs, default=None, help="e.g. 1024x1024,512x512")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--tiling", action="store_true", help="enable VAE tiling for the decode sweep")
    ap.add_argument("--also-tiling", action="store_true", help="additionally run a tiled VAE sweep for contrast")
    args, _unknown = ap.parse_known_args()

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    batches = args.batches or DEFAULT_BATCHES
    resolutions = args.resolutions or DEFAULT_RESOLUTIONS

    gpu_name = torch.cuda.get_device_name(device)
    print(f"[env] device={device} gpu={gpu_name} dtype={dtype}", flush=True)

    tokenizer, text_encoder, vae, vae_scale_factor, image_processor = build_components(args.ckpt, device, dtype)

    te_bytes_full = _param_bytes(text_encoder)
    te_bytes_weights = _nonembedding_param_bytes(text_encoder)
    vae_bytes = _param_bytes(vae)
    vae_dec_bytes = _param_bytes(vae.decoder)
    print(
        f"[info] text_encoder params: full={te_bytes_full/1e9:.2f} GB "
        f"(transformer-weights only={te_bytes_weights/1e9:.2f} GB); "
        f"vae full={vae_bytes/1e9:.2f} GB decoder={vae_dec_bytes/1e9:.2f} GB",
        flush=True,
    )

    with torch.no_grad():
        text_results = profile_text_encode(tokenizer, text_encoder, device, batches, args.warmup, args.iters)
        vae_results = profile_vae_decode(
            vae, vae_scale_factor, device, dtype, batches, resolutions, args.warmup, args.iters, tiling=args.tiling
        )
        vae_results_tiled = []
        if args.also_tiling:
            vae_results_tiled = profile_vae_decode(
                vae, vae_scale_factor, device, dtype, batches, resolutions, args.warmup, args.iters, tiling=True
            )
            # restore
            if hasattr(vae, "disable_tiling"):
                vae.disable_tiling()
        save_result = profile_image_save(
            vae, vae_scale_factor, image_processor, device, dtype, args.warmup, args.iters, resolutions[0]
        )

    payload = {
        "meta": {
            "gpu": gpu_name,
            "dtype": str(dtype),
            "ckpt": args.ckpt,
            "warmup": args.warmup,
            "iters": args.iters,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "vae_scale_factor": vae_scale_factor,
            "text_encoder_bytes_full": te_bytes_full,
            "text_encoder_bytes_weights": te_bytes_weights,
            "vae_bytes_full": vae_bytes,
            "vae_decoder_bytes": vae_dec_bytes,
        },
        "text_encode": text_results,
        "vae_decode": vae_results,
        "vae_decode_tiled": vae_results_tiled,
        "image_save": save_result,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[done] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
