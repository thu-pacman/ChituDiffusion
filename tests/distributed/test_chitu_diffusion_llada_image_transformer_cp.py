from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from chitu_diffusion.models.llada_image.epe import LLaDAImageEpeModule
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
    with torch.inference_mode():
        reference_t2i = model(
            x,
            timestep,
            cap_feats,
            glm_cap_feats=glm_feats,
            return_dict=False,
        )[0][0]
        reference_edit = model(
            x,
            timestep,
            cap_feats,
            glm_cap_feats=glm_feats,
            source_latents=source_latents,
            return_dict=False,
        )[0][0]

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
            with parallel.activate((0, 1)), torch.inference_mode():
                parallel_t2i = model(
                    x,
                    timestep,
                    cap_feats,
                    glm_cap_feats=glm_feats,
                    return_dict=False,
                )[0][0]
                parallel_edit = model(
                    x,
                    timestep,
                    cap_feats,
                    glm_cap_feats=glm_feats,
                    source_latents=source_latents,
                    return_dict=False,
                )[0][0]
            torch.testing.assert_close(
                parallel_t2i,
                reference_t2i,
                rtol=2e-4,
                atol=2e-5,
            )
            torch.testing.assert_close(
                parallel_edit,
                reference_edit,
                rtol=5e-3,
                atol=5e-4,
            )
        dist.barrier()
    finally:
        parallel.close()
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_llada_image_t2i_and_edit_match_two_rank_context_parallel() -> None:
    mp.spawn(_worker, args=(2, _free_port()), nprocs=2, join=True)
