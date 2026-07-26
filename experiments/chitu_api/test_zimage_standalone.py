from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

from chitu_diffusers import (
    CacheConfig,
    DiffusersEngine,
    DiffusionRequest,
    EngineConfig,
    EPACPipeline,
    EPACRequest,
    EPACServeConfig,
    FullWorldLaneWorkerPool,
    LaneWorkItem,
    LaneWorkResult,
    SingletonLaneWorkerPool,
    ZImageModelAdapter,
)
from chitu_diffusers.models.zimage import (
    EpeCostModel,
    EpeRequest,
    EpeParallelContext,
    EpeZImagePipeline,
    EpeZImageTransformer2DModel,
    ZImageEpeModule,
)


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


def test_zimage_capabilities_do_not_advertise_reserved_flexcache() -> None:
    capabilities = ZImageModelAdapter.capabilities

    assert capabilities.epe
    assert capabilities.context_parallel
    assert not capabilities.flexcache


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


def test_chitu_diffusers_adapter_matches_native_pipeline_call() -> None:
    torch.manual_seed(23)
    pipeline = _tiny_pipeline()
    prompt_embeds = [torch.randn((8, 8))]
    native = pipeline(
        prompt=None,
        prompt_embeds=prompt_embeds,
        guidance_scale=0,
        height=128,
        width=128,
        num_inference_steps=3,
        generator=torch.Generator().manual_seed(5),
        output_type="latent",
    ).images
    engine = DiffusersEngine.from_pipeline(
        pipeline,
        adapter=ZImageModelAdapter(),
        config=EngineConfig(
            max_pending_requests=1,
            max_inflight_requests=1,
            pulse_steps=1,
        ),
    )
    result = engine.generate(
        DiffusionRequest(
            request_id="zimage-parity",
            inputs={
                "prompt_embeds": prompt_embeds,
                "guidance_scale": 0,
                "height": 128,
                "width": 128,
                "generator": torch.Generator().manual_seed(5),
            },
            num_inference_steps=3,
            output_type="latent",
        )
    )

    assert result.error is None
    torch.testing.assert_close(result.output.images, native, rtol=0, atol=0)
    pipeline.close()


def test_epac_pipeline_generate_matches_native_pipeline_call() -> None:
    torch.manual_seed(29)
    pipeline = _tiny_pipeline()
    prompt_embeds = [torch.randn((8, 8))]
    native = pipeline(
        prompt=None,
        prompt_embeds=prompt_embeds,
        guidance_scale=0,
        height=128,
        width=128,
        num_inference_steps=3,
        generator=torch.Generator().manual_seed(9),
        output_type="latent",
    ).images
    epac = EPACPipeline(
        pipeline,
        model_path="tiny-zimage",
        default_cache=CacheConfig(),
    )

    output = epac.generate(
        EPACRequest(
            prompt=None,
            request_id="epac-generate",
            height=128,
            width=128,
            num_steps=3,
            guidance_scale=0,
            seed=9,
            output_type="latent",
            extra_inputs={"prompt_embeds": prompt_embeds},
        )
    )

    torch.testing.assert_close(output.images, native, rtol=0, atol=0)
    epac.close()


def test_epac_pipeline_serve_reuses_loaded_pipeline(monkeypatch, tmp_path) -> None:
    pipeline = _tiny_pipeline()
    epac = EPACPipeline(
        pipeline,
        model_path="tiny-zimage",
        default_cache=CacheConfig(),
    )
    captured = {}

    def fake_run_runtime(runtime, config) -> None:
        captured["runtime"] = runtime
        captured["config"] = config

    monkeypatch.setattr(
        "chitu_diffusers.serve.torchrun.run_runtime",
        fake_run_runtime,
    )
    epac.serve(
        EPACServeConfig(
            warmup_resolutions=((128, 128),),
            warmup_steps=3,
            pulse_steps=3,
            switch_allowed_until_step=3,
            max_inflight_requests=1,
            max_pending_requests=2,
            default_num_steps=3,
            default_width=128,
            default_height=128,
            output_root=str(tmp_path),
            schedule_strategy="static_cp",
        )
    )

    assert captured["runtime"].executor.pipeline is pipeline
    assert captured["config"].parallelism.chitu_pool.warmup_resolutions == ((128, 128),)
    assert captured["config"].parallelism.chitu_pool.pulse_steps == 3
    assert captured["config"].parallelism.chitu_pool.policy == "static_cp"
    assert pipeline.transformer.epe.policy == "static_cp"
    pipeline.close()


def test_epac_cache_interface_fails_before_execution() -> None:
    request = EPACRequest(prompt="test", cache=CacheConfig(strategy="meancache"))

    with pytest.raises(NotImplementedError, match="reserved but not connected"):
        request.to_diffusion_request(device=torch.device("cpu"))


def test_single_rank_warmup_initializes_measured_cost_model() -> None:
    pipeline = _tiny_pipeline()
    report = pipeline.warmup_epe(
        resolutions=((128, 128),),
        steps=3,
        text_tokens=8,
        cfg_conditions=1,
    )
    assert report["rows"][0]["resolution"] == [128, 128]
    assert report["rows"][0]["image_tokens"] == 64
    assert pipeline.transformer.epe.cost_model.ready
    assert (
        pipeline.transformer.epe.cost_model.predict_step_ms(
            image_tokens=64,
            width=1,
            cfg_conditions=1,
        )
        > 0
    )
    pipeline.close()


def _planner(
    *,
    pulse_steps: int = 4,
    switch_allowed_until_step: int = 20,
) -> ZImageEpeModule:
    parallel = SimpleNamespace(rank=0, world_size=4, allowed_widths=(1, 2, 4))
    planner = ZImageEpeModule(
        parallel,
        policy="elastic",
        pulse_steps=pulse_steps,
        switch_allowed_until_step=switch_allowed_until_step,
        balanced_k=True,
        online_calibration=False,
    )
    planner.initialize_cost_model(_measured_profile())
    return planner


def _measured_profile() -> list[dict]:
    rows = [
        {
            "resolution": [512, 512],
            "image_tokens": 1024,
            "width": 1,
            "latency_ms": 90.0,
        },
        {
            "resolution": [512, 512],
            "image_tokens": 1024,
            "width": 2,
            "latency_ms": 100.0,
        },
        {
            "resolution": [512, 512],
            "image_tokens": 1024,
            "width": 4,
            "latency_ms": 110.0,
        },
        {
            "resolution": [1024, 1024],
            "image_tokens": 4096,
            "width": 1,
            "latency_ms": 1000.0,
        },
        {
            "resolution": [1024, 1024],
            "image_tokens": 4096,
            "width": 2,
            "latency_ms": 520.0,
        },
        {
            "resolution": [1024, 1024],
            "image_tokens": 4096,
            "width": 4,
            "latency_ms": 300.0,
        },
        {
            "resolution": [2048, 2048],
            "image_tokens": 9216,
            "width": 1,
            "latency_ms": 3200.0,
        },
        {
            "resolution": [2048, 2048],
            "image_tokens": 9216,
            "width": 2,
            "latency_ms": 1500.0,
        },
        {
            "resolution": [2048, 2048],
            "image_tokens": 9216,
            "width": 4,
            "latency_ms": 780.0,
        },
    ]
    for row in rows:
        row["cfg_conditions"] = 2
    return rows


def _request(
    request_id: str,
    *,
    image_tokens: int = 4096,
    deadline_at_ms: float | None = None,
    total_steps: int = 50,
    current_step: int = 0,
    current_ranks: tuple[int, ...] | None = None,
) -> EpeRequest:
    return EpeRequest(
        request_id=request_id,
        image_tokens=image_tokens,
        total_steps=total_steps,
        current_step=current_step,
        submitted_at_ms=0.0,
        deadline_at_ms=deadline_at_ms,
        cfg_conditions=2,
        current_ranks=current_ranks,
    )


def test_dit_owns_epe_module_and_cost_model_uses_sequence_length() -> None:
    planner = _planner()
    short = _request("short", image_tokens=1024)
    long = _request("long", image_tokens=9216)

    assert planner.predict_step_ms(long, 1) > planner.predict_step_ms(short, 1)
    assert planner.predict_step_ms(long, 4) < planner.predict_step_ms(long, 1)

    context = EpeParallelContext.from_torchrun(allowed_widths=(1,))
    model = EpeZImageTransformer2DModel(**_model_kwargs())
    model.configure_epe(context, online_calibration=False)
    assert isinstance(model.epe, ZImageEpeModule)
    assert model.epe.parallel is context
    context.close()


def test_cost_model_requires_startup_warmup_before_prediction() -> None:
    cost_model = EpeCostModel()
    with pytest.raises(RuntimeError, match="warmup"):
        cost_model.predict_step_ms(image_tokens=4096, width=2)
    cost_model.initialize(_measured_profile())
    assert cost_model.predict_step_ms(image_tokens=4096, width=2) == 520.0


def test_cost_planner_and_balanced_k_use_per_lane_step_cost() -> None:
    plan = _planner().plan_phase(
        [_request("first"), _request("second"), _request("third")],
        now_ms=0.0,
    )

    by_width = {len(item.ranks): item for item in plan}
    assert sorted(len(item.ranks) for item in plan) == [1, 1, 2]
    assert by_width[2].steps > by_width[1].steps
    phase_times = [item.predicted_phase_ms for item in plan]
    assert max(phase_times) / min(phase_times) < 1.2


def test_balanced_k_accounts_for_different_sequence_lengths() -> None:
    plan = _planner().plan_phase(
        [
            _request("short", image_tokens=1024, total_steps=100),
            _request("long", image_tokens=9216, total_steps=100),
        ],
        now_ms=0.0,
    )
    by_id = {item.request_id: item for item in plan}

    assert by_id["short"].steps > by_id["long"].steps
    assert by_id["short"].predicted_step_ms < by_id["long"].predicted_step_ms
    assert by_id["short"].steps > 4
    assert by_id["long"].steps == 4


def test_balanced_k_pulses_when_the_first_request_completes() -> None:
    planner = _planner(pulse_steps=5, switch_allowed_until_step=0)
    requests = [
        _request(
            "short-tail",
            image_tokens=4096,
            total_steps=12,
            current_ranks=(0,),
        ),
        *[
            _request(
                f"long-{rank}",
                image_tokens=9216,
                total_steps=12,
                current_step=7,
                current_ranks=(rank,),
            )
            for rank in range(1, 4)
        ],
    ]

    plan = planner.plan_phase(requests, now_ms=0.0)
    by_id = {item.request_id: item for item in plan}

    assert by_id["short-tail"].steps == 12
    assert all(by_id[f"long-{rank}"].steps == 4 for rank in range(1, 4))
    phase_times = [item.predicted_phase_ms for item in plan]
    assert max(phase_times) - min(phase_times) == pytest.approx(800.0)


def test_efficiency_layout_does_not_wait_for_starvation_to_fill_lanes() -> None:
    requests = [_request(str(index), image_tokens=9216) for index in range(4)]
    planner = _planner()

    fresh = planner.plan_phase(requests, now_ms=0.0)
    starved = planner.plan_phase(requests, now_ms=31_000.0)

    assert len(fresh) == len(requests)
    assert all(len(item.ranks) == 1 for item in fresh)
    assert len(starved) == len(requests)
    assert all(len(item.ranks) == 1 for item in starved)


def test_slo_planner_gives_tight_deadline_the_full_lane() -> None:
    now_ms = 100_000.0
    plan = _planner().plan_phase(
        [
            _request("urgent", deadline_at_ms=now_ms + 10_000.0),
            _request("normal-1"),
            _request("normal-2"),
        ],
        now_ms=now_ms,
    )

    assert len(plan) == 1
    assert plan[0].request_id == "urgent"
    assert plan[0].ranks == (0, 1, 2, 3)


def test_online_calibration_corrects_the_matching_execution_key() -> None:
    parallel = SimpleNamespace(rank=0, world_size=4, allowed_widths=(1, 2, 4))
    planner = ZImageEpeModule(
        parallel,
        online_calibration=True,
    )
    planner.initialize_cost_model(_measured_profile())
    request = _request("calibration")
    predicted = planner.predict_step_ms(request, 2)
    assignment = {
        "image_tokens": request.image_tokens,
        "ranks": [0, 1],
        "batch_size": request.batch_size,
        "cfg_conditions": request.cfg_conditions,
    }
    for _ in range(3):
        planner.observe_assignment(assignment, predicted * 2.0)

    assert planner.predict_step_ms(request, 2) == pytest.approx(predicted * 2.0)
    snapshot = planner.calibrator.snapshot()
    assert snapshot["per_key"]["4096:2:1:2"] == {
        "factor": pytest.approx(2.0),
        "samples": 3,
    }


def test_standalone_package_has_no_chitu_runtime_dependency() -> None:
    package = Path(__file__).parents[2] / "chitu_diffusers" / "models" / "zimage"
    forbidden = (
        "chitu_diffusion.runtime",
        "DiffusionBackend",
        "DiffusionTaskPool",
        "chitu_init",
        "chitu_generate",
    )
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(package.glob("*.py"))
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
    warmup_report = epe_pipeline.warmup_epe(
        resolutions=((128, 128),),
        steps=3,
        text_tokens=8,
        cfg_conditions=2,
    )
    assert sorted(row["width"] for row in warmup_report["rows"]) == [1, 2]
    assert {
        row["width"]: (row["cfp_degree"], row["cp_degree"])
        for row in warmup_report["rows"]
    } == {1: (1, 1), 2: (2, 1)}
    prompt_embeds = [inputs[2][0].clone()]
    state = epe_pipeline.prepare_request(
        prompt_embeds=prompt_embeds,
        guidance_scale=0,
        height=128,
        width=128,
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

    negative_prompt_embeds = [-prompt_embeds[0]]
    epe.epe.cfg_parallel = False
    pure_cp = epe_pipeline.prepare_request(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        guidance_scale=4.0,
        height=128,
        width=128,
        num_inference_steps=1,
        generator=torch.Generator().manual_seed(29),
    )
    epe_pipeline.denoise_step(pure_cp, lane_ranks=(0, 1))
    epe.epe.cfg_parallel = True
    cfp2 = epe_pipeline.prepare_request(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        guidance_scale=4.0,
        height=128,
        width=128,
        num_inference_steps=1,
        generator=torch.Generator().manual_seed(29),
    )
    epe_pipeline.denoise_step(cfp2, lane_ranks=(0, 1))
    torch.testing.assert_close(cfp2.latents, pure_cp.latents, rtol=4e-4, atol=4e-4)
    context.close()


def _singleton_pool_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    context = EpeParallelContext.from_torchrun(
        allowed_widths=tuple(
            width for width in range(1, world_size + 1) if world_size % width == 0
        )
    )
    context.initialize_worker_control_group()
    request_count = world_size * 3
    pending = [LaneWorkItem(f"req{index}", index) for index in range(request_count)]
    completed: list[LaneWorkResult] = []
    lock = threading.Lock()

    def claim(_lane_rank: int) -> LaneWorkItem | None:
        with lock:
            return None if not pending else pending.pop(0)

    def execute(item: LaneWorkItem, lane_rank: int) -> LaneWorkResult:
        time.sleep(0.01 if lane_rank == 0 else 0.001)
        return LaneWorkResult(
            item.request_id,
            output=item.payload,
            metadata={"lane_rank": lane_rank},
        )

    def complete(result: LaneWorkResult) -> None:
        with lock:
            completed.append(result)

    pool = SingletonLaneWorkerPool(
        rank=rank,
        world_size=world_size,
        control_group=context.worker_control_group,
        claim=claim,
        execute=execute,
        complete=complete,
        should_stop=lambda: rank == 0 and len(completed) == request_count,
        idle_sleep_s=0.001,
    )
    pool.run()
    if rank == 0:
        assert {result.request_id for result in completed} == {
            f"req{index}" for index in range(request_count)
        }
        assert {result.metadata["lane_rank"] for result in completed} == set(
            range(world_size)
        )
    dist.barrier()
    context.close()
    dist.destroy_process_group()


def _full_world_pool_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    context = EpeParallelContext.from_torchrun(
        allowed_widths=tuple(
            width for width in range(1, world_size + 1) if world_size % width == 0
        )
    )
    context.initialize_worker_control_group()
    pending = [LaneWorkItem(f"req{index}", index) for index in range(3)]
    completed: list[LaneWorkResult] = []

    def claim(_lane_rank: int) -> LaneWorkItem | None:
        return None if not pending else pending.pop(0)

    def execute(item: LaneWorkItem, lane_ranks: tuple[int, ...]) -> LaneWorkResult:
        assert lane_ranks == tuple(range(world_size))
        error = (
            "rank 3 observed req1" if rank == 3 and item.request_id == "req1" else None
        )
        return LaneWorkResult(
            item.request_id,
            output=item.payload if rank == 0 else None,
            error=error,
            metadata={"lane_rank": rank},
        )

    FullWorldLaneWorkerPool(
        rank=rank,
        world_size=world_size,
        control_group=context.worker_control_group(1),
        claim=claim,
        execute=execute,
        complete=completed.append,
        should_stop=lambda: rank == 0 and len(completed) == 3,
        idle_sleep_s=0.001,
    ).run()
    if rank == 0:
        assert [result.request_id for result in completed] == ["req0", "req1", "req2"]
        assert completed[0].output == 0
        assert completed[1].error == "rank 3 observed req1"
        assert completed[2].output == 2
    dist.barrier()
    context.close()
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_two_rank_agkv_matches_native_pipeline_model() -> None:
    mp.spawn(_cp_worker, args=(2, _find_free_port()), nprocs=2, join=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_four_rank_singleton_pool_pulls_work_independently() -> None:
    mp.spawn(
        _singleton_pool_worker,
        args=(4, _find_free_port()),
        nprocs=4,
        join=True,
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_four_rank_full_world_pool_pulls_work_collectively() -> None:
    mp.spawn(
        _full_world_pool_worker,
        args=(4, _find_free_port()),
        nprocs=4,
        join=True,
    )
