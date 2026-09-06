"""The Z-Image plan against the dense model, small and in float32.

A column/row pairing that is wrong computes a plausible-looking wrong answer
rather than raising, and on the real checkpoint bf16 rounding hides the
difference between a bug and accumulated noise. This runs a small random model
in float32 through the production path -- meta construction, the plan, then the
rank-local safetensors loader -- where the two results should agree closely.
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

transformer_z_image = pytest.importorskip(
    "diffusers.models.transformers.transformer_z_image"
)
safetensors_torch = pytest.importorskip("safetensors.torch")

WORLD_SIZE = 2
TINY = {
    "all_patch_size": [2],
    "all_f_patch_size": [1],
    "axes_dims": [4, 6, 6],
    "axes_lens": [256, 128, 128],
    "cap_feat_dim": 32,
    "dim": 64,
    "in_channels": 16,
    "n_heads": 4,
    "n_kv_heads": 4,
    "n_layers": 2,
    "n_refiner_layers": 1,
    "norm_eps": 1e-05,
    "qk_norm": True,
    "rope_theta": 256.0,
    "siglip_feat_dim": None,
    "t_scale": 1000.0,
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _dense_model():
    torch.manual_seed(7)
    model = transformer_z_image.ZImageTransformer2DModel(**TINY)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.05)
    return model.eval()


def _inputs():
    generator = torch.Generator().manual_seed(11)
    latents = torch.randn(16, 1, 16, 16, generator=generator)
    caption = torch.randn(12, TINY["cap_feat_dim"], generator=generator)
    return [latents], torch.tensor([0.5]), [caption]


def _forward(model) -> torch.Tensor:
    latents, timestep, caption = _inputs()
    with torch.no_grad():
        return model(latents, timestep, caption, return_dict=False)[0][0]


def _write_checkpoint(model, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    model.save_config(root)
    state = {name: value.contiguous() for name, value in model.state_dict().items()}
    safetensors_torch.save_file(state, root / "diffusion_pytorch_model.safetensors")


def _worker(rank: int, port: int, checkpoint: str, results) -> None:
    from chitu_diffusion.models.zimage.tensor_parallel import (
        load_tensor_parallel_transformer,
    )
    from chitu_diffusion.parallel.tp import TensorParallelTopology
    from chitu_diffusion.parallel.tp.topology import set_tensor_parallel_topology

    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(WORLD_SIZE),
    )
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    set_tensor_parallel_topology(
        TensorParallelTopology(
            ranks=tuple(range(WORLD_SIZE)),
            rank=rank,
            rank_in_group=rank,
            degree=WORLD_SIZE,
            process_group=None,
        )
    )
    model, _ = load_tensor_parallel_transformer(
        transformer_z_image.ZImageTransformer2DModel,
        checkpoint,
        degree=WORLD_SIZE,
        subfolder=None,
        torch_dtype=torch.float32,
    )
    results[rank] = _forward(model)
    dist.barrier()
    dist.destroy_process_group()


def test_the_sharded_model_reproduces_the_dense_forward():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dense = _dense_model()
        _write_checkpoint(dense, root)
        expected = _forward(dense)

        manager = mp.Manager()
        results = manager.dict()
        mp.spawn(
            _worker,
            args=(_free_port(), str(root), results),
            nprocs=WORLD_SIZE,
            join=True,
        )

    for rank in range(WORLD_SIZE):
        torch.testing.assert_close(results[rank], expected, rtol=1e-4, atol=1e-4)
