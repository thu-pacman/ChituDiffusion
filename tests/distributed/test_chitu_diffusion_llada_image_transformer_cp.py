from __future__ import annotations

import os
import socket
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from chitu_diffusion.models.llada_image.epe import LLaDAImageEpeModule
from chitu_diffusion.models.llada_image.runtime import (
    LLaDAImageDenoiseState,
    LLaDAImageRuntimeMixin,
)
from chitu_diffusion.models.llada_image.transformer import (
    EpeLLaDAImageTransformer2DModel,
)
from chitu_diffusion.parallel import EpeParallelContext


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tiny_model(device: torch.device) -> EpeLLaDAImageTransformer2DModel:
    return (
        EpeLLaDAImageTransformer2DModel(
            in_channels=2,
            dim=12,
            n_layers=1,
            n_refiner_layers=1,
            n_heads=2,
            cap_feat_dim=4,
            semantic_feat_dim=4,
            axes_dims=(2, 2, 2),
            axes_lens=(256, 32, 32),
        )
        .to(device)
        .eval()
    )


class _ContractTransformer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.dtype = torch.float32
        self.config = SimpleNamespace(in_channels=2)
        self.epe: Any = None
        self.calls: list[tuple[tuple[float, int | None, bool], ...]] = []

    def __call__(
        self,
        *,
        x: list[torch.Tensor],
        t: torch.Tensor,
        cap_feats: list[torch.Tensor],
        glm_cap_feats: list[torch.Tensor] | None,
        source_latents: list[torch.Tensor] | None,
        glm_cap_lengths: list[int] | None = None,
    ) -> SimpleNamespace:
        del t
        if glm_cap_feats is None:
            logical_glm_lengths: tuple[int | None, ...] = (None,) * len(cap_feats)
        elif glm_cap_lengths is None:
            logical_glm_lengths = tuple(len(features) for features in glm_cap_feats)
        else:
            logical_glm_lengths = tuple(glm_cap_lengths)

        has_source = source_latents is not None
        conditions = tuple(
            (float(features[0, 0]), logical_length, has_source)
            for features, logical_length in zip(
                cap_feats, logical_glm_lengths, strict=True
            )
        )
        self.calls.append(conditions)

        samples = []
        for latent, (cap_value, logical_length, uses_source) in zip(
            x, conditions, strict=True
        ):
            value = cap_value
            if logical_length is not None:
                value += logical_length / 10
            if uses_source:
                value += 0.25
            samples.append(torch.full_like(latent, value))
        return SimpleNamespace(sample=samples)


class _CaptureScheduler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_train_timesteps=1000)
        self.model_output: torch.Tensor | None = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        *,
        return_dict: bool,
    ) -> tuple[torch.Tensor]:
        del timestep
        assert return_dict is False
        self.model_output = model_output.clone()
        return (sample + model_output,)


class _ContractRuntime(LLaDAImageRuntimeMixin):
    def __init__(self, transformer: _ContractTransformer) -> None:
        self.transformer = transformer
        self.text_encoder = SimpleNamespace(device=transformer.device)
        self.vq_calls = 0

    def generate_vq_tokens(
        self,
        prompt: str | list[str],
        height: int,
        width: int,
    ) -> torch.Tensor:
        del prompt, height, width
        assert dist.get_rank() == 0, "only rank 0 may generate VQ tokens"
        self.vq_calls += 1
        return torch.randint(0, 1024, (1, 32))


def _make_state(device: torch.device, generation_mode: str) -> LLaDAImageDenoiseState:
    scheduler = _CaptureScheduler()
    cond_glm_cap_feats = None
    uncond_glm_cap_feats = None
    source_latents = None
    if generation_mode in {"vq", "editing"}:
        cond_glm_cap_feats = [torch.full((2, 4), 5.0, device=device)]
        uncond_glm_cap_feats = [torch.empty((0, 4), device=device)]
    if generation_mode == "editing":
        source_latents = [torch.full((2, 1, 2, 2), 0.5, device=device)]

    return LLaDAImageDenoiseState(
        latents=torch.zeros(1, 2, 2, 2, device=device),
        cond_cap_feats=[torch.full((1, 4), 3.0, device=device)],
        uncond_cap_feats=[torch.full((1, 4), 1.0, device=device)],
        cond_glm_cap_feats=cond_glm_cap_feats,
        uncond_glm_cap_feats=uncond_glm_cap_feats,
        source_latents=source_latents,
        timesteps=torch.tensor([500.0], device=device),
        scheduler=scheduler,
        guidance_scale=4.5,
        generation_mode=generation_mode,
        width=32,
        height=32,
        image_tokens=8 if generation_mode == "editing" else 4,
    )


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    backend = "nccl" if use_cuda else "gloo"
    device = torch.device("cuda", rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    parallel = EpeParallelContext.from_torchrun(
        allowed_widths=(1, 2),
        owns_process_group=False,
    )

    torch.manual_seed(37)
    model = _tiny_model(device)
    x = [torch.randn(2, 1, 2, 2, device=device)]
    timestep = torch.tensor([0.6], device=device)
    cap_feats = [torch.randn(3, 4, device=device)]
    glm_feats = [torch.randn(5, 4, device=device)]
    source_latents = [torch.randn(2, 1, 2, 2, device=device)]
    conditions = {
        "text": {"glm_cap_feats": None, "source_latents": None},
        "vq": {"glm_cap_feats": glm_feats, "source_latents": None},
        "editing": {
            "glm_cap_feats": glm_feats,
            "source_latents": source_latents,
        },
    }
    with torch.inference_mode():
        references = {
            name: model(
                x,
                timestep,
                cap_feats,
                glm_cap_feats=condition["glm_cap_feats"],
                source_latents=condition["source_latents"],
                return_dict=False,
            )[0][0]
            for name, condition in conditions.items()
        }

    try:
        for cfg_parallel in (False, True):
            warmup_epe = LLaDAImageEpeModule(
                parallel,
                attention_mode="agkv",
                cfg_parallel=cfg_parallel,
            )
            model.configure_epe(warmup_epe)
            report = warmup_epe.warmup_transformer(
                model,
                resolutions=((32, 32),),
                steps=3,
                vae_scale_factor=16,
                text_tokens=3,
                cfg_conditions=2,
            )
            assert {row["width"] for row in report["rows"]} == {1, 2}

        for mode in ("agkv", "ulysses"):
            model.configure_epe(
                LLaDAImageEpeModule(
                    parallel,
                    attention_mode=mode,
                    cfg_parallel=False,
                )
            )
            for name, condition in conditions.items():
                with parallel.activate((0, 1)), torch.inference_mode():
                    parallel_output = model(
                        x,
                        timestep,
                        cap_feats,
                        glm_cap_feats=condition["glm_cap_feats"],
                        source_latents=condition["source_latents"],
                        return_dict=False,
                    )[0][0]
                rtol, atol = (5e-3, 5e-4) if name == "editing" else (2e-4, 2e-5)
                torch.testing.assert_close(
                    parallel_output,
                    references[name],
                    rtol=rtol,
                    atol=atol,
                    msg=lambda message: f"{mode=} {name=}: {message}",
                )

        for generation_mode in ("text", "vq", "editing"):
            has_glm = generation_mode != "text"
            uses_source = generation_mode == "editing"
            baseline_expected = (
                (3.0, 2 if has_glm else None, uses_source),
                (1.0, 0 if has_glm else None, uses_source),
            )

            baseline_transformer = _ContractTransformer(device)
            baseline_runtime = _ContractRuntime(baseline_transformer)
            baseline_state = _make_state(device, generation_mode)
            baseline_runtime.denoise_step(baseline_state)
            baseline_scheduler = baseline_state.scheduler
            assert isinstance(baseline_scheduler, _CaptureScheduler)
            assert baseline_scheduler.model_output is not None
            assert baseline_transformer.calls == [baseline_expected]

            cfp_transformer = _ContractTransformer(device)
            cfp_transformer.epe = LLaDAImageEpeModule(
                parallel,
                attention_mode="agkv",
                cfg_parallel=True,
            )
            cfp_runtime = _ContractRuntime(cfp_transformer)
            cfp_state = _make_state(device, generation_mode)
            cfp_runtime.denoise_step(
                cfp_state,
                lane_ranks=(0, 1),
            )
            cfp_scheduler = cfp_state.scheduler
            assert isinstance(cfp_scheduler, _CaptureScheduler)
            assert cfp_scheduler.model_output is not None
            torch.testing.assert_close(
                cfp_scheduler.model_output,
                baseline_scheduler.model_output,
            )
            torch.testing.assert_close(cfp_state.latents, baseline_state.latents)
            expected_call = (baseline_expected[rank],)
            assert cfp_transformer.calls == [expected_call]

        vq_runtime = _ContractRuntime(_ContractTransformer(device))
        torch.manual_seed(700 + rank)
        rng_state = torch.random.get_rng_state()
        token_ids = vq_runtime._generate_vq_tokens_synchronized(
            "prompt",
            32,
            32,
            torch.Generator().manual_seed(42),
        )
        assert torch.equal(torch.random.get_rng_state(), rng_state)
        assert vq_runtime.vq_calls == (1 if rank == 0 else 0)
        gathered_tokens: list[torch.Tensor | None] = [None] * world_size
        dist.all_gather_object(gathered_tokens, token_ids)
        for gathered in gathered_tokens:
            assert gathered is not None
            assert torch.equal(gathered, token_ids)

        dist.barrier()
    finally:
        parallel.close()
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_llada_image_parallel_contracts_match_two_ranks() -> None:
    mp.spawn(_worker, args=(2, _free_port()), nprocs=2, join=True)
