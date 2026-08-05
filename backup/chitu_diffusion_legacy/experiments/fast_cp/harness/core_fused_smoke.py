"""Smoke-test the migrated core CP fused producer path.

This intentionally avoids model weights. It drives the real
``ImageContextParallelAttention`` with synthetic packed Q/K/V projection
closures and compares:

* old eager ``attention()`` path
* new ``attention_fused_packed()`` path

Launch, for example:

    CHITU_PYTHON_BIN=$PWD/.venv/bin/python \
    script/srun_direct.sh 1 2 experiments/fast_cp/harness/core_fused_smoke.py \
      --mode ulysses
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from chitu_diffusion.core.distributed.comm_group import CommGroup  # noqa: E402
from chitu_diffusion.core.distributed.parallel_state import initialize_diffusion_parallel_groups  # noqa: E402
from chitu_diffusion.modules.attention.diffusion_attn_backend import DiffusionAttention_with_CP  # noqa: E402
from chitu_diffusion.modules.attention.image_cp_attention import ImageContextParallelAttention  # noqa: E402


def sdpa_backend(q, k, v, causal: bool = False):
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=causal,
    )
    return (out.transpose(1, 2).contiguous(),)


class SdpaBackend:
    impl = "torch_sdpa"

    def __call__(self, q, k, v, *_, **kwargs):
        return sdpa_backend(q, k, v, causal=bool(kwargs.get("causal", False))) + (None, None)


class SyntheticProjections:
    def __init__(self, *, batch, img_local, txt_tokens, heads, head_dim, hidden, dtype, device, seed):
        gen = torch.Generator(device=device).manual_seed(seed)
        self.img_len = img_local
        self.heads = heads
        self.head_dim = head_dim
        self.hidden = torch.randn(batch, img_local + txt_tokens, hidden, dtype=dtype, device=device, generator=gen)
        out_dim = heads * head_dim
        self.weights = {
            name: torch.randn(hidden, out_dim, dtype=dtype, device=device, generator=gen) / hidden ** 0.5
            for name in ("q", "k", "v")
        }

    def produce(self, name: str) -> torch.Tensor:
        x = self.hidden @ self.weights[name]
        return x.unflatten(-1, (self.heads, self.head_dim)).contiguous()

    def packed(self):
        return self.produce("q"), self.produce("k"), self.produce("v")


def run_serial(cp, proj):
    q, k, v = proj.packed()
    il = proj.img_len
    return cp.attention(
        q[:, il:], k[:, il:], v[:, il:],
        q[:, :il], k[:, :il], v[:, :il],
    )


def run_fused(cp, proj):
    return cp.attention_fused_packed(
        img_len=proj.img_len,
        produce_q=lambda: proj.produce("q"),
        produce_k=lambda: proj.produce("k"),
        produce_v=lambda: proj.produce("v"),
        num_heads=proj.heads,
    )


def make_tensor(shape, *, dtype, device, seed):
    return torch.randn(*shape, dtype=dtype, device=device, generator=torch.Generator(device=device).manual_seed(seed))


def run_full_txt_smoke(args, *, rank, world, local_rank, device, dtype):
    initialize_diffusion_parallel_groups(cfg_size=1, cp_size=world, up=world)
    cp = DiffusionAttention_with_CP(SdpaBackend(), ulysses_limit=world, ring_cudagraph=False, cp_backend="ucp")
    img_local = args.img_tokens // world
    txt_q = make_tensor((args.batch, args.txt_tokens, args.heads, args.head_dim), dtype=dtype, device=device, seed=args.seed)
    txt_k = make_tensor((args.batch, args.txt_tokens, args.heads, args.head_dim), dtype=dtype, device=device, seed=args.seed + 1)
    txt_v = make_tensor((args.batch, args.txt_tokens, args.heads, args.head_dim), dtype=dtype, device=device, seed=args.seed + 2)
    img_q = make_tensor((args.batch, img_local, args.heads, args.head_dim), dtype=dtype, device=device, seed=args.seed + 100 + rank)
    img_k = make_tensor((args.batch, img_local, args.heads, args.head_dim), dtype=dtype, device=device, seed=args.seed + 200 + rank)
    img_v = make_tensor((args.batch, img_local, args.heads, args.head_dim), dtype=dtype, device=device, seed=args.seed + 300 + rank)

    with torch.no_grad():
        txt_s, img_s = cp.cp_attn_with_full_txt(
            txt_q, txt_k, txt_v, img_q, img_k, img_v, image_seq_len=args.img_tokens, causal=False
        )
        txt_f, img_f = cp.cp_attn_with_full_txt_fused(
            produce_txt_q=lambda: txt_q,
            produce_txt_k=lambda: txt_k,
            produce_txt_v=lambda: txt_v,
            produce_img_q=lambda: img_q,
            produce_img_k=lambda: img_k,
            produce_img_v=lambda: img_v,
            image_seq_len=args.img_tokens,
            causal=False,
            num_heads=args.heads,
        )
    return txt_s, img_s, txt_f, img_f


def run_packed_smoke(args, *, rank, world, local_rank, device, dtype):
    group = CommGroup([list(range(world))], global_rank=rank, local_rank=local_rank)
    cp = ImageContextParallelAttention(sdpa_backend, mode=args.mode, profile=False)
    cp.enable(group)
    proj = SyntheticProjections(
        batch=args.batch,
        img_local=args.img_tokens // world,
        txt_tokens=args.txt_tokens,
        heads=args.heads,
        head_dim=args.head_dim,
        hidden=args.hidden,
        dtype=dtype,
        device=device,
        seed=args.seed + rank,
    )

    with torch.no_grad():
        txt_s, img_s = run_serial(cp, proj)
        txt_f, img_f = run_fused(cp, proj)
    return txt_s, img_s, txt_f, img_f


def max_diff(a, b):
    return (a.float() - b.float()).abs().max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", choices=["packed", "full_txt"], default="packed")
    ap.add_argument("--mode", choices=["agkv", "ulysses"], default="agkv")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--img-tokens", type=int, default=512)
    ap.add_argument("--txt-tokens", type=int, default=64)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tol", type=float, default=2e-2)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world = dist.get_world_size()
    if args.img_tokens % world != 0:
        raise ValueError(f"--img-tokens must be divisible by world size, got {args.img_tokens} and {world}")
    if args.mode == "ulysses" and args.heads % world != 0:
        raise ValueError(f"--heads must be divisible by world size for Ulysses, got {args.heads} and {world}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.api == "full_txt":
        txt_s, img_s, txt_f, img_f = run_full_txt_smoke(
            args, rank=rank, world=world, local_rank=local_rank, device=device, dtype=dtype
        )
    else:
        txt_s, img_s, txt_f, img_f = run_packed_smoke(
            args, rank=rank, world=world, local_rank=local_rank, device=device, dtype=dtype
        )
    torch.cuda.synchronize()

    diff = torch.stack([max_diff(txt_s, txt_f), max_diff(img_s, img_f)])
    dist.all_reduce(diff, op=dist.ReduceOp.MAX)
    passed = bool(diff.max().item() <= args.tol)
    flag = torch.tensor([1 if passed else 0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)

    if rank == 0:
        print(
            f"[core-smoke] api={args.api} mode={args.mode} world={world} "
            f"txt_max={diff[0].item():.3e} img_max={diff[1].item():.3e} "
            f"tol={args.tol:.1e} -> {'PASS' if flag.item() == 1 else 'FAIL'}",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()
    if flag.item() != 1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
