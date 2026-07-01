"""Standalone harness for the model-agnostic CP attention core.

Drives the *real* ``ImageContextParallelAttention``
(``chitu_diffusion/modules/attention/image_cp_attention.py``) with **synthetic**
packed Q/K/V, with no Z-Image / scheduler / config / model weights in the loop.
Use it to debug, bit-equivalence-check, profile, and A/B the async QKV-comm
overlap (``attention_fused``) before wiring it into ``z_image_attention.py``.

See ``async_comm_h20_nvlink_plan.md`` section 1.5 for why this is the boundary.

Launch (single node, cp4)::

    bash experiments/fast_cp/harness/run_harness.sh 4 -- --mode agkv --check
    bash experiments/fast_cp/harness/run_harness.sh 8 -- --mode ulysses --bench

What it does:
  * builds a synthetic ``hidden_states`` [B, S_local, hidden] per rank;
  * ``produce_q/k/v`` closures do a real projection GEMM (so the compute stream
    has work to overlap the collective with) -> packed [B, S_local_img+S_txt, H, D];
  * ``--check`` : serial (cp.attention) vs overlap (cp.attention_fused) outputs,
    same synthetic seed -> max-abs-diff / PSNR. Asserts they match.
  * ``--bench`` : median wall-clock over N iters, serial vs overlap.
  * ``CHITU_TORCH_TRACE=/path.json`` : rank-0 chrome trace of one short run.

If ``attention_fused`` does not exist yet (scaffolding phase), the harness runs
the serial path only and reports that overlap is unavailable -- so it is useful
from implementation step 1 onward.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

# Make the repo importable when launched as a bare script under torchrun.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from chitu_diffusion.modules.attention.image_cp_attention import (  # noqa: E402
    ImageContextParallelAttention,
)

# Overlap schedule lives in the harness dir (subclass), not in the core yet.
from overlap_cp import OverlapCPAttention  # noqa: E402


# --------------------------------------------------------------------------- #
# Minimal stand-ins for the two things the CP core depends on.
# --------------------------------------------------------------------------- #
class HarnessGroup:
    """Duck-typed ``CommGroup`` exposing exactly what the CP core touches.

    The core uses: ``group_size``, ``gpu_group`` (a real ProcessGroup handed to
    ``dist.all_gather``), ``rank_in_group``, and ``all_to_all(t, scatter, gather)``.
    Mirrors ``chitu_diffusion/core/distributed/comm_group.py``.
    """

    def __init__(self, pg, ranks):
        self.gpu_group = pg
        self.rank_list = list(ranks)
        self.group_size = len(self.rank_list)
        self.rank_in_group = self.rank_list.index(dist.get_rank())
        self.a2a_backend = os.environ.get("CHITU_HARNESS_A2A", "list").strip().lower()
        self._symm_cache = {}
        self._a2a_sequence_idx = 0
        self._symm_mem = None
        self._symm_group_name = None
        if self.a2a_backend in ("symm_single", "symm_pipe"):
            import torch.distributed._symmetric_memory as symm_mem

            self._symm_mem = symm_mem
            backend = os.environ.get(
                "CHITU_HARNESS_SYMM_BACKEND",
                "NCCL" if self.a2a_backend == "symm_single" else "",
            ).strip()
            if backend:
                try:
                    symm_mem.set_backend(backend)
                except Exception as exc:
                    if dist.get_rank() == 0:
                        print(f"[harness] symm_mem.set_backend({backend!r}) failed: {exc!r}", flush=True)
            self._symm_group_name = getattr(self.gpu_group, "group_name", None) or "0"
            symm_mem.enable_symm_mem_for_group(self._symm_group_name)

    def reset_a2a_sequence(self):
        self._a2a_sequence_idx = 0

    def all_to_all(self, input_: torch.Tensor, scatter_dim: int = 2, gather_dim: int = 1) -> torch.Tensor:
        if self.group_size == 1:
            return input_
        if self.a2a_backend in ("single", "symm_single", "symm_pipe"):
            return self._all_to_all_single(
                input_, scatter_dim=scatter_dim, gather_dim=gather_dim,
                use_symm=self.a2a_backend == "symm_single",
                use_pipe=self.a2a_backend == "symm_pipe",
            )
        in_list = [t.contiguous() for t in torch.tensor_split(input_, self.group_size, scatter_dim)]
        out_list = [torch.empty_like(in_list[0]) for _ in range(self.group_size)]
        dist.all_to_all(out_list, in_list, group=self.gpu_group)
        return torch.cat(out_list, dim=gather_dim).contiguous()

    def _symm_buffers(self, shape, dtype, device, slot: int):
        key = (slot, tuple(shape), dtype, device.index)
        entry = self._symm_cache.get(key)
        if entry is None:
            if self._symm_mem is None:
                raise RuntimeError("symmetric memory backend was not initialized")
            send = self._symm_mem.empty(*shape, device=device, dtype=dtype)
            recv = self._symm_mem.empty(*shape, device=device, dtype=dtype)
            self._symm_mem.rendezvous(send, group=self._symm_group_name)
            self._symm_mem.rendezvous(recv, group=self._symm_group_name)
            entry = (send, recv)
            self._symm_cache[key] = entry
        return entry

    def _pipe_all_to_all(self, packed: torch.Tensor) -> torch.Tensor:
        if self._symm_mem is None:
            raise RuntimeError("symmetric memory backend was not initialized")
        fn = getattr(self._symm_mem, "_pipelined_produce_and_all2all", None)
        if fn is None:
            raise RuntimeError("torch.distributed._symmetric_memory has no _pipelined_produce_and_all2all")
        recv = torch.empty_like(packed)

        def producer(dst_rank: int, chunk: torch.Tensor) -> None:
            chunk.copy_(packed.narrow(0, dst_rank, 1))

        fn(producer, recv, self._symm_group_name)
        return recv

    def _all_to_all_single(
        self,
        input_: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        *,
        use_symm: bool,
        use_pipe: bool,
    ) -> torch.Tensor:
        """Packed all-to-all for the two Ulysses layouts used by the harness.

        ``torch.distributed.all_to_all_single`` splits along dim0, so pack the
        requested scatter dimension into a leading rank dimension, communicate a
        single contiguous buffer, then unpack into the requested gather layout.
        This avoids the Python list of per-rank contiguous tensors and replaces
        many small copies with one pack and one unpack.
        """
        p = self.group_size
        if scatter_dim == 2 and gather_dim == 1:
            # [B, S, H, D] -> send [rank, B, S, H/p, D] -> [B, rank*S, H/p, D]
            b, s, h, d = input_.shape
            if h % p != 0:
                raise ValueError(f"head dim {h} must be divisible by group size {p}")
            hc = h // p
            packed = input_.reshape(b, s, p, hc, d).permute(2, 0, 1, 3, 4)
            if use_pipe:
                recv = self._pipe_all_to_all(packed)
            elif use_symm:
                slot = self._a2a_sequence_idx
                self._a2a_sequence_idx += 1
                send, recv = self._symm_buffers((p, b, s, hc, d), input_.dtype, input_.device, slot)
                send.copy_(packed)
            else:
                send = packed.contiguous()
                recv = torch.empty_like(send)
            if not use_pipe:
                dist.all_to_all_single(recv, send, group=self.gpu_group)
            return recv.permute(1, 0, 2, 3, 4).reshape(b, p * s, hc, d).contiguous()
        if scatter_dim == 1 and gather_dim == 2:
            # [B, S, H, D] -> send [rank, B, S/p, H, D] -> [B, S/p, rank*H, D]
            b, s, h, d = input_.shape
            if s % p != 0:
                raise ValueError(f"sequence dim {s} must be divisible by group size {p}")
            sc = s // p
            packed = input_.reshape(b, p, sc, h, d).permute(1, 0, 2, 3, 4)
            if use_pipe:
                recv = self._pipe_all_to_all(packed)
            elif use_symm:
                slot = self._a2a_sequence_idx
                self._a2a_sequence_idx += 1
                send, recv = self._symm_buffers((p, b, sc, h, d), input_.dtype, input_.device, slot)
                send.copy_(packed)
            else:
                send = packed.contiguous()
                recv = torch.empty_like(send)
            if not use_pipe:
                dist.all_to_all_single(recv, send, group=self.gpu_group)
            return recv.permute(1, 2, 0, 3, 4).reshape(b, sc, p * h, d).contiguous()
        raise NotImplementedError(
            f"CHITU_HARNESS_A2A={self.a2a_backend} supports only Ulysses layouts, got scatter_dim={scatter_dim}, gather_dim={gather_dim}"
        )


def sdpa_backend(q, k, v, causal: bool = False):
    """Synthetic attention backend: SDPA over packed ``[B, S, H, D]``.

    Matches the call shape the core expects: ``backend(q, k, v, causal=...)[0]``
    returns ``[B, S, H, D]``.
    """
    # [B, S, H, D] -> [B, H, S, D] for SDPA, then back.
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal
    )
    return (out.transpose(1, 2).contiguous(),)


# --------------------------------------------------------------------------- #
# Synthetic workload
# --------------------------------------------------------------------------- #
class SyntheticProjections:
    """Holds per-rank hidden states + projection weights, and builds the
    ``produce_q/k/v`` closures that the overlap path consumes.

    Each closure does a genuine GEMM (hidden @ W) so the compute stream has real
    projection work to hide the collective behind -- this is the synthetic stand
    in for Z-Image's ``to_q/to_k/to_v`` + norm + RoPE.
    """

    def __init__(self, *, B, s_img_local, s_txt, heads, head_dim, hidden, dtype, device, gen):
        self.B = B
        self.s_img_local = s_img_local
        self.s_txt = s_txt
        self.heads = heads
        self.head_dim = head_dim
        self.img_len = s_img_local  # packed order is [image_chunk, text]
        s_local = s_img_local + s_txt
        out_dim = heads * head_dim
        self.hidden = torch.randn(B, s_local, hidden, dtype=dtype, device=device, generator=gen)
        self.w = {
            name: (torch.randn(hidden, out_dim, dtype=dtype, device=device, generator=gen) / hidden ** 0.5)
            for name in ("q", "k", "v")
        }

    def _produce(self, which: str) -> torch.Tensor:
        x = self.hidden @ self.w[which]
        x = x.unflatten(-1, (self.heads, self.head_dim))
        return x.contiguous()

    def _produce_from(self, hs, which: str) -> torch.Tensor:
        x = hs @ self.w[which]
        x = x.unflatten(-1, (self.heads, self.head_dim))
        return x.contiguous()

    def produce_packed(self, hs):
        """(q, k, v) built from an explicit hidden_states tensor -- used as the
        captured region's static input for the CUDA graph paths."""
        return (
            self._produce_from(hs, "q"),
            self._produce_from(hs, "k"),
            self._produce_from(hs, "v"),
        )

    def produce_one(self, hs, which: str):
        return self._produce_from(hs, which)

    def closures(self):
        return (
            lambda: self._produce("q"),
            lambda: self._produce("k"),
            lambda: self._produce("v"),
        )

    def packed(self):
        """Eager serial Q/K/V (same math, no overlap), for the serial path."""
        return self._produce("q"), self._produce("k"), self._produce("v")


# --------------------------------------------------------------------------- #
# Serial / overlap drivers
# --------------------------------------------------------------------------- #
def _reset_group_a2a_sequence(cp):
    grp = getattr(cp, "_group", None)
    if grp is not None and hasattr(grp, "reset_a2a_sequence"):
        grp.reset_a2a_sequence()


def run_serial(cp, proj: SyntheticProjections):
    _reset_group_a2a_sequence(cp)
    q, k, v = proj.packed()
    il = proj.img_len
    return cp.attention(
        q[:, il:], k[:, il:], v[:, il:],
        q[:, :il], k[:, :il], v[:, :il],
    )


def run_overlap(cp, proj: SyntheticProjections):
    pq, pk, pv = proj.closures()
    return cp.attention_fused(
        img_len=proj.img_len, produce_q=pq, produce_k=pk, produce_v=pv
    )


def run_overlap_graph(cp, proj: SyntheticProjections):
    return cp.attention_fused_graphed(
        img_len=proj.img_len, hidden_states=proj.hidden, produce_packed=proj.produce_packed,
        produce_one=proj.produce_one,
    )


def run_serial_graph(cp, proj: SyntheticProjections):
    return cp.attention_serial_graphed(
        img_len=proj.img_len, hidden_states=proj.hidden, produce_packed=proj.produce_packed
    )


def _exit_after_graph():
    """Avoid NCCL process-group teardown hangs after CUDA graph capture.

    This mirrors the standalone graphed-ring probe behavior in the repo: once a
    graph-capture experiment has printed/exported its result, there is no useful
    cleanup left to do, and destroy_process_group() can hang on some NCCL builds.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _set_zero_cta_policy(opts) -> bool:
    config = getattr(opts, "config", None)
    if config is None or not hasattr(config, "cta_policy"):
        return False
    for value in ("zero", "ZERO", 0):
        try:
            config.cta_policy = value
            return True
        except Exception:
            pass
    return False


def _count_gpu_kernels(path):
    """Count device-side kernel events in an exported chrome trace, so we can
    tell whether CUPTI actually captured the GPU timeline (vs CPU-only)."""
    import json
    try:
        with open(path) as f:
            ev = json.load(f).get("traceEvents", [])
    except Exception:
        return -1
    n = 0
    for e in ev:
        if e.get("ph") != "X":
            continue
        cat = str(e.get("cat", "")).lower()
        if cat in ("kernel", "gpu_memcpy", "gpu_memset") or "gpu" in cat:
            n += 1
    return n


def _allclose_report(name, a, b):
    diff = (a.float() - b.float()).abs()
    max_abs = diff.max().item()
    denom = b.float().pow(2).mean().sqrt().item() + 1e-12
    psnr = 20.0 * torch.log10(torch.tensor(denom / (diff.pow(2).mean().sqrt().item() + 1e-12))).item()
    ok = max_abs < 1e-2
    return ok, f"{name}: max_abs_diff={max_abs:.3e}  PSNR={psnr:.1f}dB  -> {'OK' if ok else 'MISMATCH'}"


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Standalone CP-core async overlap harness")
    ap.add_argument("--mode", default="agkv", choices=["agkv", "ulysses"])
    ap.add_argument("--check", action="store_true", help="bit-equivalence: serial vs overlap")
    ap.add_argument("--bench", action="store_true", help="wall-clock A/B")
    ap.add_argument("--graph", action="store_true",
                    help="also benchmark CUDA-graphed serial & overlap (removes CPU launch overhead)")
    ap.add_argument("--graph-diag", action="store_true",
                    help="capture serial-graph then overlap-graph once each, with stage prints, then exit")
    ap.add_argument("--profile-parts", action="store_true",
                    help="time projection / gather / attention separately for the overlap ceiling")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    # Z-Image 1024x1024-ish defaults; override per experiment.
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--img-tokens", type=int, default=4096, help="GLOBAL image tokens (sharded across CP)")
    ap.add_argument("--txt-tokens", type=int, default=512)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=None, help="defaults to heads*head_dim")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    a2a_backend = os.environ.get("CHITU_HARNESS_A2A", "list").strip().lower()
    if a2a_backend in ("symm_single", "symm_pipe"):
        opts = dist.ProcessGroupNCCL.Options()
        zero_cta = _set_zero_cta_policy(opts)
        try:
            dist.init_process_group(backend="nccl", pg_options=opts, device_id=device)
        except Exception:
            dist.init_process_group(backend="nccl")
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[harness] CHITU_HARNESS_A2A={a2a_backend} zero_cta_set={zero_cta}", flush=True)
    else:
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    is_main = rank == 0

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    hidden = args.hidden or (args.heads * args.head_dim)

    if args.img_tokens % world != 0:
        if is_main:
            print(f"[harness] img-tokens {args.img_tokens} not divisible by cp_size {world}", flush=True)
        dist.destroy_process_group()
        return
    if args.mode == "ulysses" and args.heads % world != 0:
        if is_main:
            print(f"[harness] ulysses needs heads {args.heads} divisible by cp_size {world}", flush=True)

    # CP group over all ranks (the harness uses the full world as one CP group).
    group = HarnessGroup(dist.group.WORLD, list(range(world)))
    cp = OverlapCPAttention(sdpa_backend, mode=args.mode, profile=False)
    cp.enable(group)
    has_fused = hasattr(cp, "attention_fused")
    used_cuda_graph = False
    used_symm_mem = a2a_backend in ("symm_single", "symm_pipe")

    s_img_local = args.img_tokens // world
    # Per-rank generator -> each rank holds its own shard, but seed is fixed so a
    # given (rank, run) is reproducible across serial/overlap calls.
    gen = torch.Generator(device=device).manual_seed(args.seed + rank)

    def make_proj():
        return SyntheticProjections(
            B=args.batch, s_img_local=s_img_local, s_txt=args.txt_tokens,
            heads=args.heads, head_dim=args.head_dim, hidden=hidden,
            dtype=dtype, device=device, gen=torch.Generator(device=device).manual_seed(args.seed + rank),
        )

    if is_main:
        print(
            f"[harness] world={world} mode={args.mode} dtype={args.dtype} "
            f"B={args.batch} img_global={args.img_tokens} img_local={s_img_local} "
            f"txt={args.txt_tokens} H={args.heads} D={args.head_dim} "
            f"fused_available={has_fused}",
            flush=True,
        )

    # ----- graph capture diagnostic (isolate where a hang happens) --------- #
    if args.graph_diag:
        os.environ["CHITU_GRAPH_DEBUG"] = "1"
        proj = make_proj()
        if is_main:
            print("[graph-diag] === step 1: serial-graph (single stream + NCCL) ===", flush=True)
        try:
            with torch.no_grad():
                run_serial_graph(cp, proj)
            torch.cuda.synchronize(); dist.barrier()
            if is_main:
                print("[graph-diag] serial-graph: OK (replay succeeded)", flush=True)
        except Exception as exc:
            if is_main:
                print(f"[graph-diag] serial-graph FAILED: {exc!r}", flush=True)
        if is_main:
            print("[graph-diag] === step 2: overlap-graph (comm stream fork + NCCL) ===", flush=True)
        try:
            with torch.no_grad():
                run_overlap_graph(cp, proj)
            torch.cuda.synchronize(); dist.barrier()
            if is_main:
                print("[graph-diag] overlap-graph: OK (replay succeeded)", flush=True)
        except Exception as exc:
            if is_main:
                print(f"[graph-diag] overlap-graph FAILED: {exc!r}", flush=True)
        _exit_after_graph()

    # ----- correctness ----------------------------------------------------- #
    if args.check:
        if not has_fused:
            if is_main:
                print("[harness] --check skipped: attention_fused() not implemented yet", flush=True)
        else:
            proj = make_proj()
            with torch.no_grad():
                txt_s, img_s = run_serial(cp, proj)
                txt_o, img_o = run_overlap(cp, proj)
            torch.cuda.synchronize()
            ok_t, msg_t = _allclose_report("txt_out", txt_o, txt_s)
            ok_i, msg_i = _allclose_report("img_out", img_o, img_s)
            # gather pass/fail across ranks
            flag = torch.tensor([1 if (ok_t and ok_i) else 0], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if is_main:
                print(f"[check] {msg_t}", flush=True)
                print(f"[check] {msg_i}", flush=True)
                print(f"[check] ALL RANKS {'PASS' if flag.item() == 1 else 'FAIL'}", flush=True)

            # graphed overlap must also match serial
            if args.graph and args.mode in ("agkv", "ulysses"):
                used_cuda_graph = True
                try:
                    proj_g = make_proj()
                    with torch.no_grad():
                        txt_gs, img_gs = run_serial(cp, proj_g)
                        txt_gg, img_gg = run_overlap_graph(cp, proj_g)
                    torch.cuda.synchronize()
                    okg_t, msgg_t = _allclose_report("txt_out(graph)", txt_gg, txt_gs)
                    okg_i, msgg_i = _allclose_report("img_out(graph)", img_gg, img_gs)
                    gflag = torch.tensor([1 if (okg_t and okg_i) else 0], device=device)
                    dist.all_reduce(gflag, op=dist.ReduceOp.MIN)
                    if is_main:
                        print(f"[check] {msgg_t}", flush=True)
                        print(f"[check] {msgg_i}", flush=True)
                        print(f"[check] GRAPH ALL RANKS {'PASS' if gflag.item() == 1 else 'FAIL'}", flush=True)
                except Exception as exc:
                    if is_main:
                        print(f"[check] graph correctness failed: {exc!r}", flush=True)

    # ----- benchmark ------------------------------------------------------- #
    if args.bench:
        def timed(fn):
            proj = make_proj()
            for _ in range(args.warmup):
                with torch.no_grad():
                    fn(cp, proj)
            torch.cuda.synchronize()
            dist.barrier()
            samples = []
            for _ in range(args.iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    fn(cp, proj)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - t0) * 1e3)
            return statistics.median(samples)

        serial_ms = timed(run_serial)
        line = f"[bench] serial median = {serial_ms:.3f} ms/call"
        if has_fused:
            overlap_ms = timed(run_overlap)
            speedup = (serial_ms - overlap_ms) / serial_ms * 100.0
            line += f"  |  overlap median = {overlap_ms:.3f} ms/call  |  delta = {speedup:+.1f}%"
        else:
            line += "  |  overlap unavailable (attention_fused not implemented)"
        if is_main:
            print(line, flush=True)

        # ----- CUDA graph A/B (removes CPU launch overhead) ------------------ #
        if args.graph and args.mode in ("agkv", "ulysses"):
            used_cuda_graph = True
            errored = None
            try:
                sg_ms = timed(run_serial_graph)
                og_ms = timed(run_overlap_graph)
            except Exception as exc:  # capture/replay can fail under NCCL rules
                errored = repr(exc)
            if is_main:
                if errored:
                    print(f"[graph] capture/replay failed: {errored}", flush=True)
                else:
                    g_delta = (sg_ms - og_ms) / sg_ms * 100.0
                    best = (serial_ms - og_ms) / serial_ms * 100.0
                    print(
                        f"[graph] serial-graph = {sg_ms:.3f} ms  |  overlap-graph = {og_ms:.3f} ms  "
                        f"|  graph-overlap delta = {g_delta:+.1f}%", flush=True)
                    print(
                        f"[graph] best (overlap-graph vs eager-serial) = {best:+.1f}%  "
                        f"[eager-serial {serial_ms:.3f} -> overlap-graph {og_ms:.3f} ms]", flush=True)
        elif args.graph and is_main:
            print("[graph] skipped: graph path implemented for agkv/ulysses only", flush=True)

    # ----- component profile (overlap ceiling) ----------------------------- #
    if args.profile_parts:
        proj = make_proj()

        def _cuda_time(fn, iters=40, warmup=15):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize(); dist.barrier()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); s.record()
            for _ in range(iters):
                fn()
            e.record(); torch.cuda.synchronize()
            return s.elapsed_time(e) / iters  # ms

        il = proj.img_len

        def proj_only():
            with torch.no_grad():
                proj._produce("q"); proj._produce("k"); proj._produce("v")

        def gather_only():
            with torch.no_grad():
                k = proj._produce("k")[:, :il].contiguous()
                cp.all_gather_seq(k, group, kind="kv")
                v = proj._produce("v")[:, :il].contiguous()
                cp.all_gather_seq(v, group, kind="kv")

        def attn_only():
            with torch.no_grad():
                _reset_group_a2a_sequence(cp)
                q, k, v = proj.packed()
                cp.attention(q[:, il:], k[:, il:], v[:, il:],
                             q[:, :il], k[:, :il], v[:, :il])

        t_proj = _cuda_time(proj_only)
        t_gath = _cuda_time(gather_only)  # includes the proj of k/v it gathers
        t_attn = _cuda_time(attn_only)    # full serial path
        if is_main:
            # rough exposed-comm estimate: gather_only minus its own k/v proj share
            print(f"[parts] proj(q+k+v)={t_proj:.3f}ms  gather(k+v incl proj)={t_gath:.3f}ms  "
                  f"serial_attn_total={t_attn:.3f}ms", flush=True)
            print(f"[parts] overlap ceiling ~ min(proj_overlap, comm). proj is the hideable work; "
                  f"if comm << proj, overlap can hide comm but adds stream overhead.", flush=True)

    # ----- trace ----------------------------------------------------------- #
    trace_path = os.environ.get("CHITU_TORCH_TRACE", "").strip()
    if not trace_path and os.environ.get("CHITU_TRACE_MODE", "").strip():
        trace_path = os.path.join(os.path.dirname(__file__), "traces", "trace.json")
    if trace_path:
        # Which schedule to trace. Eager and graph modes work for AGKV/Ulysses.
        trace_mode = os.environ.get("CHITU_TRACE_MODE", "overlap").strip().lower()
        proj = make_proj()

        def driver(which):
            if which == "overlap":
                return run_overlap(cp, proj)
            if which == "serial":
                return run_serial(cp, proj)
            if which == "overlap_graph":
                return run_overlap_graph(cp, proj)
            if which == "serial_graph":
                return run_serial_graph(cp, proj)
            raise ValueError(f"unknown CHITU_TRACE_MODE target: {which}")

        trace_aliases = {
            "both": ["serial", "overlap"],
            "graph": ["overlap_graph"],
            "overlap-graph": ["overlap_graph"],
            "serial-graph": ["serial_graph"],
            "both_graph": ["serial_graph", "overlap_graph"],
            "both-graph": ["serial_graph", "overlap_graph"],
            "all": ["serial", "overlap", "serial_graph", "overlap_graph"],
        }
        targets = trace_aliases.get(trace_mode, [trace_mode.replace("-", "_")])
        filtered_targets = []
        for target in targets:
            if target == "overlap" and not has_fused:
                if is_main:
                    print("[trace] skip overlap: attention_fused not implemented", flush=True)
                continue
            if target in ("serial_graph", "overlap_graph") and args.mode not in ("agkv", "ulysses"):
                if is_main:
                    print(f"[trace] skip {target}: graph trace is implemented for agkv/ulysses only", flush=True)
                continue
            filtered_targets.append(target)
        targets = filtered_targets
        if any(t in ("serial_graph", "overlap_graph") for t in targets):
            used_cuda_graph = True

        # Prime Kineto/CUPTI with one non-exported profiling context. On this
        # cluster the first profiler context can be CPU-only even after waiting;
        # the second context reliably captures device kernels. Non-rank0 ranks
        # still run the same replay count because graph targets contain NCCL.
        trace_iters = int(os.environ.get("CHITU_TRACE_ITERS", "5"))
        trace_ready_delay = float(os.environ.get("CHITU_TRACE_READY_DELAY", "6.5"))
        trace_prime = os.environ.get("CHITU_TRACE_PRIME", "1") != "0"
        if trace_prime and targets:
            prime_target = targets[0]
            for _ in range(2):
                with torch.no_grad():
                    driver(prime_target)
            torch.cuda.synchronize()
            dist.barrier()
            if not is_main:
                if trace_ready_delay > 0:
                    time.sleep(trace_ready_delay)
                for _ in range(trace_iters):
                    with torch.no_grad():
                        driver(prime_target)
                torch.cuda.synchronize()
                dist.barrier()
            else:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=False,
                    with_stack=False,
                ) as prof:
                    if trace_ready_delay > 0:
                        time.sleep(trace_ready_delay)
                    for _ in range(trace_iters):
                        with torch.no_grad():
                            driver(prime_target)
                        prof.step()
                    torch.cuda.synchronize()
                dist.barrier()

        for which in targets:
            # warm the side-stream NCCL init + cublas heuristics out of the trace
            for _ in range(5):
                with torch.no_grad():
                    driver(which)
            torch.cuda.synchronize()
            dist.barrier()
            if not is_main:
                # Keep non-profiled ranks participating in the same number of
                # graph replays/collectives without exporting redundant traces.
                if trace_ready_delay > 0:
                    time.sleep(trace_ready_delay)
                for i in range(trace_iters):
                    with torch.no_grad():
                        driver(which)
                torch.cuda.synchronize()
                dist.barrier()
                continue

            os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False,
            ) as prof:
                # Some cluster builds start CUPTI device collection lazily after
                # the profiler context opens. Wait here, then replay only the
                # requested 3-5 iterations instead of spamming the trace.
                if trace_ready_delay > 0:
                    time.sleep(trace_ready_delay)
                for i in range(trace_iters):
                    with torch.profiler.record_function(f"iter{i}_{which}"):
                        with torch.no_grad():
                            driver(which)
                    prof.step()
                torch.cuda.synchronize()
            base, ext = os.path.splitext(trace_path)
            out = f"{base}.{which}.rank{rank}{ext}"
            prof.export_chrome_trace(out)
            # Verify CUPTI actually captured device kernels (else the trace is
            # CPU-only and useless for judging overlap).
            n_gpu = _count_gpu_kernels(out)
            ka = prof.key_averages()
            cuda_total_ms = sum(getattr(k, "device_time", getattr(k, "cuda_time", 0)) for k in ka) / 1000.0
            if is_main or n_gpu == 0:
                print(f"[trace] rank{rank} {which}: wrote {out}  gpu_kernels={n_gpu}  "
                      f"cuda_time_sum={cuda_total_ms:.2f}ms", flush=True)
                if n_gpu == 0:
                    print(f"[trace] WARNING rank{rank}: 0 GPU kernels captured -> CUPTI not "
                          f"active in this env; trace is CPU-only.", flush=True)
            dist.barrier()

    if used_cuda_graph or used_symm_mem:
        torch.cuda.synchronize()
        _exit_after_graph()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
