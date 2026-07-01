"""Probe PyTorch Symmetric Memory + all_to_all_single availability.

Run under srun_direct, e.g.:

  export CHITU_PROJECT_ROOT=$PWD CHITU_PYTHON_BIN=$PWD/.venv/bin/python
  bash script/srun_direct.sh 1 4 experiments/fast_cp/harness/probe_symm_mem_a2a.py

This is intentionally standalone: it only answers "can this environment allocate
symmetric memory, rendezvous it across ranks, and use all_to_all_single on it?".
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist


def _hard_exit() -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def log(msg: str) -> None:
    rank = int(os.environ.get("RANK", "-1"))
    if rank == 0:
        print(msg, flush=True)


def _set_zero_cta_policy(opts) -> bool:
    """Best-effort setup for NCCL zero-CTA policy.

    PyTorch has changed the enum surface a few times, so try the common string
    and integer forms. Failure here should not stop the probe because plain
    symmetric allocation may still be available.
    """
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


def _init_process_group(timeout_s: int):
    opts = dist.ProcessGroupNCCL.Options()
    zero_cta = _set_zero_cta_policy(opts)
    try:
        dist.init_process_group(
            backend="nccl",
            pg_options=opts,
            device_id=torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))),
        )
        return zero_cta, None
    except Exception as exc:
        # Fall back to normal NCCL init so the probe can still test whether the
        # Python APIs exist and fail later with a more specific message.
        dist.init_process_group(backend="nccl")
        return zero_cta, repr(exc)


def _time_cuda(fn, iters: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _all_to_all_single_roundtrip(send: torch.Tensor, recv: torch.Tensor, group) -> None:
    dist.all_to_all_single(recv, send, group=group)


def _check_a2a_result(recv: torch.Tensor, rank: int, world: int) -> float:
    # send is filled with rank*1000 + destination_rank. After all-to-all, recv[i]
    # should contain source_rank i's payload for this rank.
    expected = torch.empty_like(recv)
    chunk = recv.numel() // world
    flat = expected.view(world, chunk)
    for src in range(world):
        flat[src].fill_(src * 1000 + rank)
    return (recv.float() - expected.float()).abs().max().item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--chunk-elems", type=int, default=1024 * 1024)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--backend", default="NCCL", choices=["NCCL", "CUDA", "NVSHMEM"])
    ap.add_argument("--trace", default="", help="optional rank0 chrome trace path")
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    zero_cta, init_fallback = _init_process_group(timeout_s=60)
    rank = dist.get_rank()
    world = dist.get_world_size()
    group = dist.group.WORLD
    device = torch.device("cuda", local_rank)

    if args.chunks != world:
        log(f"[probe] overriding --chunks {args.chunks} -> world size {world}")
        args.chunks = world
    numel = args.chunks * args.chunk_elems

    if rank == 0:
        try:
            nccl_version = torch.cuda.nccl.version()
        except Exception as exc:
            nccl_version = f"unavailable: {exc!r}"
        print(
            f"[probe] torch={torch.__version__} cuda={torch.version.cuda} nccl={nccl_version} "
            f"world={world} local_rank={local_rank} zero_cta_set={zero_cta} init_fallback={init_fallback}",
            flush=True,
        )

    # Baseline: normal all_to_all_single.
    send = torch.empty(numel, device=device, dtype=torch.float32)
    recv = torch.empty_like(send)
    send.view(world, -1).copy_(
        torch.stack(
            [torch.full((args.chunk_elems,), rank * 1000 + dst, device=device) for dst in range(world)]
        )
    )
    _all_to_all_single_roundtrip(send, recv, group)
    torch.cuda.synchronize()
    err = _check_a2a_result(recv, rank, world)
    flag = torch.tensor([0 if err == 0 else 1], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    log(f"[probe] baseline all_to_all_single max_err={err} all_ranks_ok={flag.item() == 0}")
    base_ms = _time_cuda(lambda: _all_to_all_single_roundtrip(send, recv, group), args.iters)
    log(f"[probe] baseline all_to_all_single {base_ms:.3f} ms/call")

    try:
        import torch.distributed._symmetric_memory as symm_mem
    except Exception as exc:
        log(f"[probe] import torch.distributed._symmetric_memory failed: {exc!r}")
        _hard_exit()

    log(
        "[probe] symm_mem api: "
        f"backend_before={symm_mem.get_backend(device)} "
        f"nvshmem_available={getattr(symm_mem, 'is_nvshmem_available', lambda: 'n/a')()}"
    )
    try:
        symm_mem.set_backend(args.backend)
        log(f"[probe] symm_mem.set_backend({args.backend!r}) OK")
    except Exception as exc:
        log(f"[probe] symm_mem.set_backend({args.backend!r}) failed: {exc!r}")

    group_name = getattr(group, "group_name", None)
    if not group_name:
        group_name = "default"
    log(f"[probe] group_name={group_name!r}")

    try:
        symm_mem.enable_symm_mem_for_group(group_name)
        log("[probe] enable_symm_mem_for_group OK")
    except Exception as exc:
        log(f"[probe] enable_symm_mem_for_group failed: {exc!r}")

    try:
        ssend = symm_mem.empty(numel, device=device, dtype=torch.float32)
        srecv = symm_mem.empty(numel, device=device, dtype=torch.float32)
        ssend.view(world, -1).copy_(
            torch.stack(
                [torch.full((args.chunk_elems,), rank * 1000 + dst, device=device) for dst in range(world)]
            )
        )
        sm_send = symm_mem.rendezvous(ssend, group=group_name)
        sm_recv = symm_mem.rendezvous(srecv, group=group_name)
        log(f"[probe] rendezvous OK send={type(sm_send).__name__} recv={type(sm_recv).__name__}")
    except Exception as exc:
        log(f"[probe] symmetric allocation/rendezvous failed: {exc!r}")
        _hard_exit()

    try:
        _all_to_all_single_roundtrip(ssend, srecv, group)
        torch.cuda.synchronize()
        serr = _check_a2a_result(srecv, rank, world)
        sflag = torch.tensor([0 if serr == 0 else 1], device=device)
        dist.all_reduce(sflag, op=dist.ReduceOp.MAX)
        log(f"[probe] symm tensor all_to_all_single max_err={serr} all_ranks_ok={sflag.item() == 0}")
        symm_ms = _time_cuda(lambda: _all_to_all_single_roundtrip(ssend, srecv, group), args.iters)
        log(f"[probe] symm tensor all_to_all_single {symm_ms:.3f} ms/call")
    except Exception as exc:
        log(f"[probe] symm tensor all_to_all_single failed: {exc!r}")

    # Internal experimental hook: useful to know if this build exposes it, but
    # optional because the function is private and may change across PyTorch.
    fn = getattr(symm_mem, "_pipelined_produce_and_all2all", None)
    log(f"[probe] has _pipelined_produce_and_all2all={fn is not None}")
    pipe_recv = None
    if fn is not None:
        try:
            pipe_recv = torch.empty_like(send.view(world, -1))

            def pipe_roundtrip() -> None:
                def producer(dst_rank: int, chunk: torch.Tensor) -> None:
                    chunk.copy_(send.view(world, -1).narrow(0, dst_rank, 1))

                fn(producer, pipe_recv, group_name)

            pipe_roundtrip()
            torch.cuda.synchronize()
            perr = _check_a2a_result(pipe_recv.reshape(-1), rank, world)
            pflag = torch.tensor([0 if perr == 0 else 1], device=device)
            dist.all_reduce(pflag, op=dist.ReduceOp.MAX)
            log(f"[probe] pipelined_produce_and_all2all max_err={perr} all_ranks_ok={pflag.item() == 0}")
            pipe_ms = _time_cuda(pipe_roundtrip, args.iters)
            log(f"[probe] pipelined_produce_and_all2all {pipe_ms:.3f} ms/call")
        except Exception as exc:
            log(f"[probe] pipelined_produce_and_all2all failed: {exc!r}")

    if args.trace:
        trace_iters = min(args.iters, 5)
        for _ in range(3):
            _all_to_all_single_roundtrip(send, recv, group)
            _all_to_all_single_roundtrip(ssend, srecv, group)
            if fn is not None and pipe_recv is not None:
                def producer(dst_rank: int, chunk: torch.Tensor) -> None:
                    chunk.copy_(send.view(world, -1).narrow(0, dst_rank, 1))

                fn(producer, pipe_recv, group_name)
        torch.cuda.synchronize()
        dist.barrier()
        if rank != 0:
            for _ in range(trace_iters):
                _all_to_all_single_roundtrip(send, recv, group)
                _all_to_all_single_roundtrip(ssend, srecv, group)
                if fn is not None and pipe_recv is not None:
                    def producer(dst_rank: int, chunk: torch.Tensor) -> None:
                        chunk.copy_(send.view(world, -1).narrow(0, dst_rank, 1))

                    fn(producer, pipe_recv, group_name)
            torch.cuda.synchronize()
            dist.barrier()
        else:
            os.makedirs(os.path.dirname(args.trace) or ".", exist_ok=True)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False,
            ) as prof:
                # Match the harness profiler quirk: let CUPTI device collection
                # become active, but only replay a handful of calls.
                time.sleep(float(os.environ.get("CHITU_TRACE_READY_DELAY", "6.5")))
                for i in range(trace_iters):
                    with torch.profiler.record_function(f"baseline_a2a_{i}"):
                        _all_to_all_single_roundtrip(send, recv, group)
                    with torch.profiler.record_function(f"symm_a2a_{i}"):
                        _all_to_all_single_roundtrip(ssend, srecv, group)
                    if fn is not None and pipe_recv is not None:
                        with torch.profiler.record_function(f"pipe_a2a_{i}"):
                            def producer(dst_rank: int, chunk: torch.Tensor) -> None:
                                chunk.copy_(send.view(world, -1).narrow(0, dst_rank, 1))

                            fn(producer, pipe_recv, group_name)
                    prof.step()
                torch.cuda.synchronize()
            prof.export_chrome_trace(args.trace)
            log(f"[probe] wrote trace {args.trace}")
            dist.barrier()

    _hard_exit()


if __name__ == "__main__":
    raise SystemExit(main())
