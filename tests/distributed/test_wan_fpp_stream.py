"""Cross-step execution versus a serial reference with the same feedback delay.

No downloads: tiny random Wan weights, actual Gloo/NCCL P2P and collectives.
CHITU_FPP_TEST_CUDA=1 selects NCCL; the mixed mesh needs eight visible GPUs.
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_wan_pipeline_parallel import TINY, _pipeline, _port, _state

from chitu_diffusion.models.wan.fpp_scheduler import make_wan_scheduler
from chitu_diffusion.models.wan.fpp_stream import run_wan_fpp
from chitu_diffusion.models.wan.transformer import EpeWanTransformer3DModel
from chitu_diffusion.parallel.pp import FppConfig, PipelineTopology
from chitu_diffusion.parallel.pp.mesh import FppMesh


def _run(
    root,
    mesh,
    stages,
    context,
    solver,
    dtype,
    *,
    reference=False,
    patches=7,
    steps=8,
    refresh=3,
):
    device = (
        torch.device("cuda", 0 if reference else dist.get_rank())
        if os.environ.get("CHITU_FPP_TEST_CUDA") == "1"
        else torch.device("cpu")
    )
    model = EpeWanTransformer3DModel.from_pretrained(
        root,
        subfolder=None,
        pipeline_topology=mesh.pipeline,
        torch_dtype=getattr(torch, dtype),
    ).to(device)
    config = FppConfig(
        patches=patches,
        refresh_interval=refresh,
        reference_stages=stages if reference else None,
        reference_context_degree=context if reference else 1,
    )
    pipeline = _pipeline(model, None, config)
    pipeline._epe_fpp_mesh = mesh
    results = []
    # Reuse the model/communicators for another request with independent caches.
    for seed, guidance in ((11, 6.0), (29, 1.0)):
        state = _state(
            seed,
            device=device,
            steps=steps,
            guidance=guidance,
            scheduler_kind="euler",
            dtype=model.dtype,
        )
        state.scheduler = make_wan_scheduler(solver, 3)
        state.scheduler.set_timesteps(steps, device=device)
        state.timesteps = state.scheduler.timesteps
        history = []
        run_wan_fpp(
            pipeline,
            state,
            step_callback=(
                lambda current: history.append(current.latents.cpu().clone())
            )
            if guidance > 1
            else None,
        )
        assert state.complete and not state.fpp_state.cache.entries
        if mesh.pipeline.last:
            assert state.scheduler.step_index == steps
        results.append((state.latents.cpu(), history, pipeline.last_fpp_stats))
    return results


def _worker(
    rank, port, root, pp, cp, cfg, solver, dtype, results, patches, steps, refresh
):
    torch.set_num_threads(1)
    cuda = os.environ.get("CHITU_FPP_TEST_CUDA") == "1"
    if cuda:
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl" if cuda else "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=pp * cp * cfg,
        timeout=timedelta(seconds=100),
    )
    mesh = FppMesh.create(pp, cp, cfg)
    try:
        results[rank] = _run(
            root,
            mesh,
            pp,
            cp,
            solver,
            dtype,
            patches=patches,
            steps=steps,
            refresh=refresh,
        )
    finally:
        mesh.close()
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("pp", "cp", "cfg", "solver", "dtype", "patches", "steps", "refresh"),
    [
        (2, 1, 1, "euler", "float32", 3, 8, 3),
        (3, 1, 1, "unipc", "float32", 7, 8, 3),
        (2, 2, 1, "unipc", "float32", 7, 8, 3),
        (2, 1, 2, "unipc", "float32", 3, 8, 3),
        (2, 2, 2, "unipc", "float32", 3, 8, 3),
        (2, 2, 2, "unipc", "bfloat16", 3, 8, 3),
        (2, 1, 1, "unipc", "float32", 3, 54, 50),
        (2, 1, 1, "euler", "float32", 1, 4, 0),
    ],
)
def test_stream_matches_serial_feedback_reference(
    tmp_path, pp, cp, cfg, solver, dtype, patches, steps, refresh
):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = EpeWanTransformer3DModel(**TINY).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.05)
    model.save_pretrained(tmp_path)
    reference = FppMesh(PipelineTopology((0,), 0))
    expected = _run(
        str(tmp_path),
        reference,
        pp,
        cp,
        solver,
        dtype,
        reference=True,
        patches=patches,
        steps=steps,
        refresh=refresh,
    )
    with mp.get_context("spawn").Manager() as manager:
        results = manager.dict()
        mp.spawn(
            _worker,
            args=(
                _port(),
                str(tmp_path),
                pp,
                cp,
                cfg,
                solver,
                dtype,
                results,
                patches,
                steps,
                refresh,
            ),
            nprocs=pp * cp * cfg,
            join=True,
        )
        for rank in range(pp * cp * cfg):
            for (actual, history, stats), (target, target_history, _) in zip(
                results[rank], expected, strict=True
            ):
                tolerance = 3e-3 if dtype == "bfloat16" else 3e-6
                torch.testing.assert_close(
                    actual, target, rtol=tolerance, atol=tolerance
                )
                is_last = (rank // cp) % pp == pp - 1
                assert len(history) == (len(target_history) if is_last else 0)
                for got, want in zip(
                    history, target_history if is_last else [], strict=True
                ):
                    torch.testing.assert_close(
                        got, want, rtol=tolerance, atol=tolerance
                    )
                assert stats["full_steps"] + stats["patch_steps"] == steps
                assert stats["context_degree"] == cp and stats["cfg_degree"] == cfg


def test_serial_reference_consumes_feedback_after_exactly_p_microbatches(
    tmp_path, monkeypatch
):
    from chitu_diffusion.models.wan import fpp_stream

    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = EpeWanTransformer3DModel(**TINY).eval()
    model.configure_pipeline_parallel(PipelineTopology((0,), 0))
    config = FppConfig(
        patches=3, reference_stages=2, cooldown_steps=0, rotate_patches=False
    )
    pipeline = _pipeline(model, None, config)
    pipeline._epe_fpp_mesh = FppMesh(model.pipeline_topology)
    state = _state(11, device="cpu", steps=4, guidance=1, scheduler_kind="euler")
    inputs, candidates, bases, commits = [], [], [], []
    model.patch_embedding.register_forward_pre_hook(
        lambda _, args: inputs.append(args[0].clone())
    )
    original = fpp_stream.fpp_scheduler_step

    def record(scheduler, prediction, timestep, latents, *, commit):
        candidate = original(scheduler, prediction, timestep, latents, commit=commit)
        bases.append(latents.clone())
        commits.append(commit)
        candidates.append(candidate.clone())
        return candidate

    monkeypatch.setattr(fpp_stream, "fpp_scheduler_step", record)
    run_wan_fpp(pipeline, state)
    # Input 0 is full warmup; then P patches use its result. Every later patch
    # consumes exactly the candidate P positions behind, including across steps.
    for micro, value in enumerate(inputs[1:]):
        expected = candidates[0 if micro < 2 else 1 + micro - 2]
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert commits == [True, False, False, True, False, False, True, False, False, True]
    for start in (1, 4, 7):
        for base in bases[start : start + 3]:
            torch.testing.assert_close(base, candidates[start - 1], rtol=0, atol=0)


def _public_stream_worker(rank, port, root, results):
    from chitu_diffusion import WanPipeline, WanRequest

    torch.set_num_threads(1)
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE="4",
        LOCAL_RANK=str(rank),
    )
    pipeline = WanPipeline.from_pretrained(
        root,
        pipeline_parallel_degree=2,
        context_parallel_degree=1,
        cfg_parallel=True,
        fpp_config=FppConfig(patches=3),
        torch_dtype=torch.float32,
        device="cpu",
        sample_solver="unipc",
        flow_shift=3,
        tokenizer=None,
        text_encoder=None,
        vae=None,
    )
    state = _state(11, device="cpu", steps=4, guidance=6, scheduler_kind="euler")
    try:
        output = pipeline.generate(
            WanRequest(
                prompt=None,
                width=80,
                height=48,
                num_frames=5,
                num_steps=4,
                guidance_scale=6,
                output_type="latent",
                extra_inputs={
                    "latents": state.latents,
                    "prompt_embeds": state.prompt_embeds,
                    "negative_prompt_embeds": state.negative_prompt_embeds,
                },
            )
        )
        if rank == 0:
            results["latents"] = output.frames
            results["stats"] = pipeline.last_fpp_stats
    finally:
        pipeline.close()


@pytest.mark.skipif(
    os.environ.get("CHITU_FPP_TEST_CUDA") == "1", reason="public loader smoke uses CPU"
)
def test_public_generate_routes_stream_and_cfg_groups(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = EpeWanTransformer3DModel(**TINY).eval()
    pipeline = _pipeline(model, None, FppConfig(patches=3))
    pipeline.save_pretrained(tmp_path)
    expected = _run(
        str(tmp_path / "transformer"),
        FppMesh(PipelineTopology((0,), 0)),
        2,
        1,
        "unipc",
        "float32",
        reference=True,
        patches=3,
        steps=4,
        refresh=50,
    )[0][0]
    with mp.get_context("spawn").Manager() as manager:
        results = manager.dict()
        mp.spawn(
            _public_stream_worker,
            args=(_port(), str(tmp_path), results),
            nprocs=4,
            join=True,
        )
        torch.testing.assert_close(results["latents"], expected, rtol=1e-5, atol=2e-6)
        assert results["stats"]["cfg_degree"] == 2
        assert (
            results["stats"]["full_steps"] == 2 and results["stats"]["patch_steps"] == 2
        )
