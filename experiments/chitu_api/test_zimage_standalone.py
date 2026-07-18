from __future__ import annotations

import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

from experiments.chitu_api.zimage import (
    EpeParallelContext,
    EpeZImagePipeline,
    EpeZImageTransformer2DModel,
)
from experiments.chitu_api.zimage.runtime import EpeZImageServiceRuntime


def _model_kwargs() -> dict:
    return {
        "all_patch_size": (2,),
        "all_f_patch_size": (1,),
        "in_channels": 4,
        "dim": 32,
        "n_layers": 2,
        "n_refiner_layers": 1,
        "n_heads": 2,
        "n_kv_heads": 2,
        "cap_feat_dim": 8,
        "axes_dims": [4, 6, 6],
        "axes_lens": [128, 32, 32],
    }


def _inputs() -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(11)
    image = [torch.randn((4, 1, 16, 16), generator=generator)]
    timestep = torch.tensor([0.5])
    caption = [torch.randn((32, 8), generator=generator)]
    return image, timestep, caption


def test_single_rank_context_and_native_forward() -> None:
    context = EpeParallelContext.from_torchrun(allowed_widths=(1,))
    model = EpeZImageTransformer2DModel(**_model_kwargs())
    model.configure_epe(context)
    output = model(*_inputs(), return_dict=False)[0]
    assert len(output) == 1
    assert output[0].shape == (4, 1, 16, 16)
    context.close()


def test_pipeline_is_diffusers_compatible() -> None:
    assert issubclass(EpeZImagePipeline, object)
    assert EpeZImagePipeline.__call__.__name__ == "__call__"


def _tiny_pipeline() -> EpeZImagePipeline:
    context = EpeParallelContext.from_torchrun(allowed_widths=(1,))
    transformer = EpeZImageTransformer2DModel(**_model_kwargs()).eval()
    transformer.configure_epe(context)
    pipeline = EpeZImagePipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
        vae=None,
        text_encoder=None,
        tokenizer=None,
        transformer=transformer,
    )
    pipeline._epe_parallel = context
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def test_phased_denoise_matches_native_pipeline_call() -> None:
    torch.manual_seed(17)
    pipeline = _tiny_pipeline()
    prompt_embeds = [torch.randn((8, 8))]
    native = pipeline(
        prompt=None,
        prompt_embeds=prompt_embeds,
        guidance_scale=0,
        height=128,
        width=128,
        num_inference_steps=3,
        generator=torch.Generator().manual_seed(3),
        output_type="latent",
    ).images
    state = pipeline.prepare_request(
        prompt_embeds=prompt_embeds,
        guidance_scale=0,
        height=128,
        width=128,
        num_inference_steps=3,
        generator=torch.Generator().manual_seed(3),
    )
    for _ in range(3):
        pipeline.denoise_step(state, lane_ranks=(0,))
    phased = pipeline.finalize_request(state, output_type="latent").images
    torch.testing.assert_close(phased, native, rtol=0, atol=0)
    pipeline.close()


def test_phase_planner_shrinks_existing_lane_when_concurrency_rises() -> None:
    parallel = SimpleNamespace(rank=0, world_size=4, allowed_widths=(1, 2, 4))
    pipeline = SimpleNamespace(parallel_context=parallel)
    pool = SimpleNamespace(
        policy="slo_elastic",
        phase_max_steps=1,
        switch_allowed_until_step=20,
    )
    config = SimpleNamespace(parallelism=SimpleNamespace(chitu_pool=pool))
    runtime = EpeZImageServiceRuntime(config, pipeline)
    runtime._active_order = ["first"]
    runtime._states = {
        "first": SimpleNamespace(step_index=0, timesteps=torch.arange(4))
    }

    first_plan = runtime._plan_phase()
    assert first_plan[0]["ranks"] == [0, 1, 2, 3]

    runtime._active_order.append("second")
    runtime._states["second"] = SimpleNamespace(
        step_index=0, timesteps=torch.arange(4)
    )
    second_plan = runtime._plan_phase()
    assert [item["ranks"] for item in second_plan] == [[0, 1], [2, 3]]
    assert second_plan[0]["switched"] is True


def test_static_cp_planner_serializes_on_full_world_lane() -> None:
    parallel = SimpleNamespace(rank=0, world_size=4, allowed_widths=(1, 2, 4))
    pipeline = SimpleNamespace(parallel_context=parallel)
    pool = SimpleNamespace(
        policy="static_cp",
        max_inflight_requests=4,
        phase_max_steps=50,
        switch_allowed_until_step=50,
    )
    config = SimpleNamespace(parallelism=SimpleNamespace(chitu_pool=pool))
    runtime = EpeZImageServiceRuntime(config, pipeline)
    runtime._active_order = ["first", "second"]
    runtime._states = {
        "first": SimpleNamespace(step_index=0, timesteps=torch.arange(50)),
        "second": SimpleNamespace(step_index=0, timesteps=torch.arange(50)),
    }

    plan = runtime._plan_phase()

    assert len(plan) == 1
    assert plan[0]["request_id"] == "first"
    assert plan[0]["ranks"] == [0, 1, 2, 3]
    assert plan[0]["steps"] == 50


def test_standalone_package_has_no_chitu_runtime_dependency() -> None:
    package = Path(__file__).with_name("zimage")
    forbidden = (
        "chitu_diffusion.runtime",
        "DiffusionBackend",
        "DiffusionTaskPool",
        "chitu_init",
        "chitu_generate",
    )
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(package.glob("*.py"))
    )
    for symbol in forbidden:
        assert symbol not in source


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _cp_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(1234)
    native = ZImageTransformer2DModel(**_model_kwargs()).eval()
    epe = EpeZImageTransformer2DModel(**_model_kwargs()).eval()
    epe.load_state_dict(native.state_dict())
    inputs = _inputs()
    with torch.inference_mode():
        reference = native(*inputs, return_dict=False)[0][0]
        context = EpeParallelContext.from_torchrun(allowed_widths=(1, 2))
        epe.configure_epe(context)
        actual = epe(*inputs, return_dict=False)[0][0]
    torch.testing.assert_close(actual, reference, rtol=3e-4, atol=3e-4)

    epe_pipeline = EpeZImagePipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
        vae=None,
        text_encoder=None,
        tokenizer=None,
        transformer=epe,
    )
    epe_pipeline._epe_parallel = context
    prompt_embeds = [inputs[2][0].clone()]
    state = epe_pipeline.prepare_request(
        prompt_embeds=prompt_embeds,
        guidance_scale=0,
        height=32,
        width=32,
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(19),
    )
    epe_pipeline.denoise_step(state, lane_ranks=(0, 1))
    replicas = [torch.empty_like(state.latents) for _ in range(world_size)]
    dist.all_gather(replicas, state.latents)
    torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
    if rank == 0:
        epe_pipeline.denoise_step(state, lane_ranks=(0,))
    dist.broadcast(state.latents, src=0)
    epe_pipeline.synchronize_state(state, step_index=2)
    dist.all_gather(replicas, state.latents)
    torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
    assert state.complete
    assert state.scheduler.step_index == 2
    context.close()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_two_rank_agkv_matches_native_pipeline_model() -> None:
    mp.spawn(_cp_worker, args=(2, _find_free_port()), nprocs=2, join=True)
