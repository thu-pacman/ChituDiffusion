"""Z-Image under the shared tensor-parallel plan, against the real checkpoint.

Row-parallel layers split the contraction, so each rank rounds a partial sum to
bf16 before the all-reduce where the dense model rounds once at the end. That
makes any fixed tolerance arbitrary. The question worth asking is instead
whether sharding costs accuracy, so both bf16 results are measured against the
same fp32 forward: TP2 should be no further from the truth than dense bf16 is.

The plan itself is checked exactly, in float32, by the CPU companion test.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

CHECKPOINT = Path(
    os.environ.get("CHITU_ZIMAGE_PATH", "/cfs_cloud_code/royroychen/models/zimage/Z-Image")
)
WORLD_SIZE = 2
HEIGHT = 64
WIDTH = 64
CAPTION_TOKENS = 32

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
        reason=f"needs {WORLD_SIZE} CUDA devices",
    ),
    pytest.mark.skipif(
        not (CHECKPOINT / "transformer").is_dir(),
        reason=f"needs a Z-Image checkpoint at {CHECKPOINT}",
    ),
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _inputs(device: torch.device, dtype: torch.dtype):
    generator = torch.Generator(device="cpu").manual_seed(1234)
    latents = torch.randn(16, 1, HEIGHT, WIDTH, generator=generator)
    caption = torch.randn(CAPTION_TOKENS, 2560, generator=generator)
    return (
        [latents.to(device=device, dtype=dtype)],
        torch.tensor([0.5], device=device, dtype=dtype),
        [caption.to(device=device, dtype=dtype)],
    )


def _forward(model, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    latents, timestep, caption = _inputs(device, dtype)
    with torch.no_grad():
        output = model(latents, timestep, caption, return_dict=False)[0]
    return output[0].float().cpu()


def _worker(rank: int, port: int, results) -> None:
    from chitu_diffusion.models.zimage.transformer import EpeZImageTransformer2DModel
    from chitu_diffusion.parallel.cp import EpeParallelContext

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(WORLD_SIZE),
        LOCAL_RANK=str(rank),
    )
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dtype = torch.bfloat16

    context = EpeParallelContext.from_torchrun(
        allowed_widths=(1,),
        tensor_parallel_degree=WORLD_SIZE,
    )
    try:
        sharded = EpeZImageTransformer2DModel.from_pretrained(
            CHECKPOINT,
            subfolder="transformer",
            torch_dtype=dtype,
            tensor_parallel_degree=WORLD_SIZE,
        ).to(device)
        results[f"tp{rank}"] = _forward(sharded, device, dtype)
        del sharded
        torch.cuda.empty_cache()

        dist.barrier()
        if rank == 0:
            for label, reference_dtype in (("bf16", dtype), ("fp32", torch.float32)):
                dense = EpeZImageTransformer2DModel.from_pretrained(
                    CHECKPOINT,
                    subfolder="transformer",
                    torch_dtype=reference_dtype,
                ).to(device)
                results[f"dense_{label}"] = _forward(dense, device, reference_dtype)
                del dense
                torch.cuda.empty_cache()
        dist.barrier()
    finally:
        context.close()


def _relative_error(actual: torch.Tensor, truth: torch.Tensor) -> float:
    return ((actual - truth).norm() / truth.norm()).item()


def test_sharding_zimage_costs_no_more_accuracy_than_bf16_itself():
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(_worker, args=(_free_port(), results), nprocs=WORLD_SIZE, join=True)

    truth = results["dense_fp32"]
    dense = _relative_error(results["dense_bf16"], truth)
    sharded = [_relative_error(results[f"tp{rank}"], truth) for rank in range(WORLD_SIZE)]
    print(f"relative error vs fp32: dense bf16 {dense:.2e}, TP2 bf16 {sharded}")

    # Random latents and caption features sit far outside the training
    # distribution, so bf16 alone lands around 2% here; the cap only catches a
    # reference that has gone badly wrong.
    assert dense < 5e-2, f"dense bf16 alone is already off by {dense:.2e}"
    for rank, error in enumerate(sharded):
        assert error < 1.15 * dense, (
            f"rank {rank} sharded error {error:.2e} exceeds the dense bf16 "
            f"error {dense:.2e} by more than 15%"
        )
    torch.testing.assert_close(results["tp0"], results["tp1"], rtol=0, atol=0)
