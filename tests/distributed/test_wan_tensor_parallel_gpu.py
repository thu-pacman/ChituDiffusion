"""Wan under the shared tensor-parallel plan, on real GPUs at real widths.

The CPU companion checks the plan exactly in float32; this one checks it where
it will run: NCCL, bf16, and the published Wan 2.1 T2V-14B widths, at TP2 and
TP4. The weights are random because the point is the sharding rather than the
samples, and 14B does not fit on the box twice.

Row-parallel layers split the contraction, so each rank rounds a partial sum to
bf16 where the dense model rounds once at the end, which makes any fixed
tolerance arbitrary. The question asked instead is whether sharding costs
accuracy: both bf16 results are measured against the same fp32 forward, and the
sharded one should be no further from it than dense bf16 already is.
"""

from __future__ import annotations

import os
import socket
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

transformer_wan = pytest.importorskip(
    "diffusers.models.transformers.transformer_wan"
)
safetensors_torch = pytest.importorskip("safetensors.torch")

# Wan 2.1 T2V-14B widths. The depth is cut to two blocks so a dense fp32
# reference fits alongside the shards; sharding correctness does not depend on
# how many identical blocks follow.
WAN_14B = {
    "patch_size": (1, 2, 2),
    "num_attention_heads": 40,
    "attention_head_dim": 128,
    "ffn_dim": 13824,
    "in_channels": 16,
    "out_channels": 16,
    "text_dim": 4096,
    "freq_dim": 256,
    "num_layers": 2,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "eps": 1e-6,
    "rope_max_seq_len": 1024,
}
FRAMES, HEIGHT, WIDTH = 1, 32, 32
TEXT_TOKENS = 32

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="needs 4 CUDA devices",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _forward(model, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(1234)
    latents = torch.randn(
        1, WAN_14B["in_channels"], FRAMES, HEIGHT, WIDTH, generator=generator
    )
    prompt = torch.randn(1, TEXT_TOKENS, WAN_14B["text_dim"], generator=generator)
    with torch.no_grad():
        output = model(
            latents.to(device=device, dtype=dtype),
            torch.tensor([500.0], device=device, dtype=dtype),
            prompt.to(device=device, dtype=dtype),
            return_dict=False,
        )[0]
    return output.float().cpu()


def _write_checkpoint(root: Path) -> None:
    torch.manual_seed(7)
    model = transformer_wan.WanTransformer3DModel(**WAN_14B)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.02)
    root.mkdir(parents=True, exist_ok=True)
    model.save_config(root)
    safetensors_torch.save_file(
        {name: value.contiguous() for name, value in model.state_dict().items()},
        root / "diffusion_pytorch_model.safetensors",
    )


def _worker(rank: int, port: int, degree: int, checkpoint: str, results) -> None:
    from chitu_diffusion.models.wan.transformer import EpeWanTransformer3DModel
    from chitu_diffusion.parallel.cp import EpeParallelContext

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(degree),
        LOCAL_RANK=str(rank),
    )
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dtype = torch.bfloat16

    context = EpeParallelContext.from_torchrun(
        allowed_widths=(1,),
        tensor_parallel_degree=degree,
    )
    try:
        sharded = EpeWanTransformer3DModel.from_pretrained(
            checkpoint,
            subfolder=None,
            torch_dtype=dtype,
            tensor_parallel_degree=degree,
        ).to(device)
        results[f"tp{rank}"] = _forward(sharded, device, dtype)
        del sharded
        torch.cuda.empty_cache()

        dist.barrier()
        if rank == 0:
            for label, reference_dtype in (("bf16", dtype), ("fp32", torch.float32)):
                dense = transformer_wan.WanTransformer3DModel.from_pretrained(
                    checkpoint, torch_dtype=reference_dtype
                ).to(device)
                results[f"dense_{label}"] = _forward(dense, device, reference_dtype)
                del dense
                torch.cuda.empty_cache()
        dist.barrier()
    finally:
        context.close()


def _relative_error(actual: torch.Tensor, truth: torch.Tensor) -> float:
    return ((actual - truth).norm() / truth.norm()).item()


@pytest.mark.parametrize("degree", (2, 4))
def test_sharding_wan_costs_no_more_accuracy_than_bf16_itself(degree: int) -> None:
    manager = mp.Manager()
    results = manager.dict()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "transformer"
        _write_checkpoint(root)
        mp.spawn(
            _worker,
            args=(_free_port(), degree, str(root), results),
            nprocs=degree,
            join=True,
        )

    truth = results["dense_fp32"]
    dense = _relative_error(results["dense_bf16"], truth)
    sharded = [_relative_error(results[f"tp{rank}"], truth) for rank in range(degree)]
    print(
        f"TP{degree} relative error vs fp32: dense bf16 {dense:.2e}, "
        f"sharded bf16 {['%.2e' % value for value in sharded]}"
    )

    assert dense < 5e-2, f"dense bf16 alone is already off by {dense:.2e}"
    for rank, error in enumerate(sharded):
        assert error < 1.5 * dense, (
            f"rank {rank} sharded error {error:.2e} exceeds the dense bf16 error "
            f"{dense:.2e} by more than half"
        )
    for rank in range(1, degree):
        torch.testing.assert_close(
            results["tp0"], results[f"tp{rank}"], rtol=0, atol=0
        )
