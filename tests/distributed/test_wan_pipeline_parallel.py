"""Actual P2P, shard loading and Wan FPP numerical checks with tiny weights.

CPU/Gloo by default. Set CHITU_FPP_TEST_CUDA=1 under CUDA_VISIBLE_DEVICES to
exercise the same production path with NCCL (three visible GPUs are needed).
The single-stage FPP reference distinguishes intended stale-KV approximation
from distributed implementation errors; patches=1 additionally checks dense
Diffusers parity. No model downloads or text/VAE weights are needed.
"""

from __future__ import annotations

import os
import random
import socket
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler

from chitu_diffusion.models.wan.pipeline import EpeWanPipeline, WanDenoiseState
from chitu_diffusion.models.wan.transformer import EpeWanTransformer3DModel
from chitu_diffusion.parallel.cp import EpeParallelContext
from chitu_diffusion.parallel.pp import (
    FppConfig,
    FppState,
    PipelineTopology,
    partition_bounds,
)

TINY = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=4,
    attention_head_dim=12,
    in_channels=4,
    out_channels=4,
    text_dim=16,
    freq_dim=32,
    ffn_dim=64,
    num_layers=5,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    rope_max_seq_len=64,
)


def _port():
    # Avoid the ephemeral client-port range: spawned workers can otherwise
    # consume the chosen port before rank zero starts its TCPStore.
    for port in random.SystemRandom().sample(range(15000, 25000), 100):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no available test rendezvous port")


def _pipeline(model, parallel, config):
    pipeline = EpeWanPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=None,
        transformer=model,
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )
    pipeline._epe_parallel_context = parallel
    pipeline._epe_cfg_parallel = False
    pipeline._epe_fpp_config = config
    pipeline.last_fpp_stats = None
    return pipeline


def _state(seed, *, device, steps, guidance, scheduler_kind, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, 4, 2, 6, 10, generator=generator).to(device)
    # Match prepare_request: scheduler latents stay FP32, embeddings use the
    # transformer compute dtype (including native FP32 conditioning islands).
    prompt = torch.randn(1, 7, 16, generator=generator).to(device=device, dtype=dtype)
    negative = torch.randn(1, 7, 16, generator=generator).to(device=device, dtype=dtype)
    scheduler = (
        FlowMatchEulerDiscreteScheduler(shift=3)
        if scheduler_kind == "euler"
        else UniPCMultistepScheduler()
    )
    scheduler.set_timesteps(steps, device=device)
    return WanDenoiseState(
        latents,
        prompt,
        negative if guidance > 1 else None,
        scheduler.timesteps,
        scheduler,
        guidance,
        1,
        80,
        48,
        5,
        30,
    )


def _worker(rank, world, port, root, patches, scheduler_kind, dtype_name, results):
    torch.set_num_threads(1)
    cuda = os.environ.get("CHITU_FPP_TEST_CUDA") == "1"
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world),
        LOCAL_RANK=str(rank),
    )
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo", timeout=timedelta(seconds=90))
    parallel = EpeParallelContext.from_torchrun(
        allowed_widths=(1, world), owns_process_group=True
    )
    topology = PipelineTopology(tuple(range(world)), rank, dist.group.WORLD)
    model = EpeWanTransformer3DModel.from_pretrained(
        root,
        subfolder=None,
        pipeline_topology=topology,
        torch_dtype=getattr(torch, dtype_name),
    ).to(device)
    begin, end = partition_bounds(TINY["num_layers"], world)[rank]
    block_indices = {
        int(name.split(".")[1])
        for name, _ in model.named_parameters()
        if name.startswith("blocks.")
    }
    assert block_indices == set(range(begin, end))
    assert not any(parameter.is_meta for parameter in model.parameters())
    config = FppConfig(schedule="step", patches=patches)
    pipeline = _pipeline(model, parallel, config)
    # Two live requests are interleaved, with different seeds, CFG and lengths.
    states = [
        _state(
            seed,
            device=device,
            steps=steps,
            guidance=guidance,
            scheduler_kind=scheduler_kind,
            dtype=model.dtype,
        )
        for seed, steps, guidance in ((11, 4, 6.0), (29, 3, 1.0))
    ]
    histories = [[], []]
    for step in range(4):
        for index, state in enumerate(states):
            if state.complete:
                continue
            pipeline.denoise_step(state, lane_ranks=topology.ranks)
            assert state.step_index == step + 1
            assert state.scheduler.step_index == step + 1
            histories[index].append(state.latents.detach().cpu())
    for state in states:
        assert state.complete and not state.fpp_state.cache.entries
    results[rank] = histories
    results[(rank, "stats")] = [
        (state.fpp_state.full_steps, state.fpp_state.patch_steps) for state in states
    ]
    parallel.close()


def _reference(root, patches, scheduler_kind, *, dense=False, dtype_name="float32"):
    # Use the same compute backend as the distributed run. In particular,
    # CPU vs CUDA convolution/attention rounding is not a PP correctness test.
    device = "cuda:0" if os.environ.get("CHITU_FPP_TEST_CUDA") == "1" else "cpu"
    model = EpeWanTransformer3DModel.from_pretrained(
        root, torch_dtype=getattr(torch, dtype_name)
    ).to(device)
    if not dense:
        model.configure_pipeline_parallel(PipelineTopology((0,), 0))
    parallel = SimpleNamespace(
        activate=lambda ranks: nullcontext(), cfg_parallel_topology=lambda ranks: None
    )
    pipeline = _pipeline(model, parallel, FppConfig(schedule="step", patches=patches))
    histories = []
    for seed, steps, guidance in ((11, 4, 6.0), (29, 3, 1.0)):
        state = _state(
            seed,
            device=device,
            steps=steps,
            guidance=guidance,
            scheduler_kind=scheduler_kind,
            dtype=model.dtype,
        )
        history = []
        for _ in range(steps):
            pipeline.denoise_step(state, lane_ranks=(0,))
            history.append(state.latents.detach().cpu().clone())
        histories.append(history)
    return histories


@pytest.mark.parametrize(
    ("world", "patches", "scheduler_kind", "dtype_name"),
    [
        (2, 1, "euler", "float32"),
        (2, 3, "euler", "float32"),
        (3, 21, "euler", "float32"),
        (2, 3, "unipc", "float32"),
        (2, 1, "euler", "bfloat16"),
        (2, 3, "euler", "bfloat16"),
    ],
)
def test_pipeline_matches_single_stage_and_preserves_request_history(
    tmp_path, world, patches, scheduler_kind, dtype_name
):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = EpeWanTransformer3DModel(**TINY).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.05)
    model.save_pretrained(tmp_path)
    expected = _reference(
        tmp_path, patches, scheduler_kind, dense=patches == 1, dtype_name=dtype_name
    )
    with mp.get_context("spawn").Manager() as manager:
        results = manager.dict()
        mp.spawn(
            _worker,
            args=(
                world,
                _port(),
                str(tmp_path),
                patches,
                scheduler_kind,
                dtype_name,
                results,
            ),
            nprocs=world,
            join=True,
        )
        for rank in range(world):
            for history, reference in zip(results[rank], expected, strict=True):
                for actual, wanted in zip(history, reference, strict=True):
                    tolerance = (
                        dict(rtol=0.015, atol=0.015)
                        if dtype_name == "bfloat16"
                        else dict(rtol=1e-5, atol=2e-6)
                    )
                    torch.testing.assert_close(actual, wanted, **tolerance)
            assert results[(rank, "stats")] == (
                [(4, 0), (3, 0)] if patches == 1 else [(2, 2), (2, 1)]
            )


def _subgroup_worker(rank, port, root, results):
    from chitu_diffusion.models.wan.pipeline_parallel import wan_pipeline_forward

    torch.set_num_threads(1)
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE="3"
    )
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    group = dist.new_group([1, 2])
    if rank in (1, 2):
        topology = PipelineTopology((1, 2), rank, group)
        model = EpeWanTransformer3DModel.from_pretrained(
            root, subfolder=None, pipeline_topology=topology, torch_dtype=torch.float32
        )
        state = _state(11, device="cpu", steps=4, guidance=1, scheduler_kind="euler")
        output = wan_pipeline_forward(
            model,
            state.latents,
            state.timesteps[0].expand(1),
            state.prompt_embeds,
            state=FppState(FppConfig(schedule="step", patches=3)),
            branch="cond",
            step_index=0,
            full_step=True,
        )
        results[rank] = output
    dist.barrier()
    dist.destroy_process_group()


def test_pipeline_loader_preserves_native_mixed_precision(tmp_path):
    model = EpeWanTransformer3DModel(**TINY).eval()
    model.save_pretrained(tmp_path)
    native = EpeWanTransformer3DModel.from_pretrained(
        tmp_path, torch_dtype=torch.bfloat16
    )
    native_tensors = dict(native.named_parameters()) | dict(native.named_buffers())
    for rank in range(2):
        shard = EpeWanTransformer3DModel.from_pretrained(
            tmp_path,
            subfolder=None,
            pipeline_topology=PipelineTopology((0, 1), rank),
            torch_dtype=torch.bfloat16,
        )
        for name, value in (
            dict(shard.named_parameters()) | dict(shard.named_buffers())
        ).items():
            torch.testing.assert_close(value, native_tensors[name], rtol=0, atol=0)


def test_pipeline_p2p_uses_global_peers_in_a_nonzero_subgroup(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = EpeWanTransformer3DModel(**TINY).eval()
    model.save_pretrained(tmp_path)
    state = _state(11, device="cpu", steps=4, guidance=1, scheduler_kind="euler")
    with torch.no_grad():
        expected = model(
            state.latents,
            state.timesteps[0].expand(1),
            state.prompt_embeds,
            return_dict=False,
        )[0]
    with mp.get_context("spawn").Manager() as manager:
        results = manager.dict()
        mp.spawn(
            _subgroup_worker,
            args=(_port(), str(tmp_path), results),
            nprocs=3,
            join=True,
        )
        for rank in (1, 2):
            torch.testing.assert_close(results[rank], expected, rtol=1e-5, atol=2e-6)


def _public_worker(rank, port, root, results):
    from chitu_diffusion import WanPipeline, WanRequest

    torch.set_num_threads(1)
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_RANK=str(rank),
    )
    pipeline = WanPipeline.from_pretrained(
        root,
        pipeline_parallel_degree=2,
        fpp_config=FppConfig(schedule="step", patches=3),
        torch_dtype=torch.float32,
        device="cpu",
        flow_shift=3,
        tokenizer=None,
        text_encoder=None,
        vae=None,
    )
    state = _state(11, device="cpu", steps=4, guidance=6, scheduler_kind="euler")
    request = WanRequest(
        prompt=None,
        width=80,
        height=48,
        num_frames=5,
        num_steps=4,
        output_type="latent",
        guidance_scale=6,
        extra_inputs={
            "prompt_embeds": state.prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
            "latents": state.latents,
        },
    )
    try:
        output = pipeline.generate(request)
        if rank == 0:
            results["latents"] = output.frames
            results["stats"] = pipeline.last_fpp_stats
    finally:
        pipeline.close()


@pytest.mark.skipif(
    os.environ.get("CHITU_FPP_TEST_CUDA") == "1",
    reason="public loader smoke is CPU-only; NCCL is covered by the model tests",
)
def test_public_pipeline_load_and_generate_use_fpp(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = EpeWanTransformer3DModel(**TINY).eval()
    pipeline = _pipeline(model, None, FppConfig(schedule="step", patches=3))
    pipeline.save_pretrained(tmp_path)
    expected = _reference(tmp_path / "transformer", 3, "euler")[0][-1]
    with mp.get_context("spawn").Manager() as manager:
        results = manager.dict()
        mp.spawn(
            _public_worker, args=(_port(), str(tmp_path), results), nprocs=2, join=True
        )
        torch.testing.assert_close(results["latents"], expected, rtol=1e-5, atol=2e-6)
        assert results["stats"] == {
            "pipeline_degree": 2,
            "patches": 3,
            "full_steps": 2,
            "patch_steps": 2,
        }
