"""The Wan plan against the dense model, small and in float32.

A column/row pairing that is wrong computes a plausible-looking wrong answer
rather than raising, and on a real checkpoint in bf16 rounding hides the
difference between a bug and accumulated noise. This runs a small random model
in float32 through the production path -- meta parameters, the plan, then the
rank-local safetensors loader -- where the two results agree to a few parts in
ten million.

The tolerance is deliberately far below what the model needs, because this is a
weak detector on its own: a wrong Q/K scale reaches the output through
attention and then through a residual stream, and on two small layers it lands
around 1e-4. The reducing norm therefore has its own test in
``test_tensor_parallel_norm_cpu.py``; this one checks that the pieces compose.

The image-conditioned variant is covered too, because ``add_k_proj`` and
``norm_added_k`` are a second column/reducing-norm pair that only that variant
instantiates.
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

WORLD_SIZE = 2
# Wan's cross-attention splits the image tokens off the front of the text
# stream by subtracting the text encoder's hardcoded 512, so the text stream
# has to be that long for the image branch to receive anything.
TEXT_TOKENS = 512
IMAGE_TOKENS = 4

TINY = {
    "patch_size": (1, 2, 2),
    "num_attention_heads": 4,
    "attention_head_dim": 12,
    "in_channels": 4,
    "out_channels": 4,
    "text_dim": 16,
    "freq_dim": 32,
    "ffn_dim": 64,
    "num_layers": 2,
    "cross_attn_norm": True,
    "qk_norm": "rms_norm_across_heads",
    "eps": 1e-6,
    "rope_max_seq_len": 64,
}
# The image embedder lifts the image context to the hidden width before the
# blocks see it, so the added K/V projections read that width, not image_dim.
IMAGE_CONDITIONED = {
    **TINY,
    "image_dim": 16,
    "added_kv_proj_dim": TINY["num_attention_heads"] * TINY["attention_head_dim"],
}
CONFIGS = {"t2v": TINY, "i2v": IMAGE_CONDITIONED}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _dense_model(variant: str):
    torch.manual_seed(7)
    model = transformer_wan.WanTransformer3DModel(**CONFIGS[variant])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.05)
    return model.eval()


def _forward(model, variant: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    latents = torch.randn(1, 4, 1, 8, 8, generator=generator)
    prompt = torch.randn(1, TEXT_TOKENS, TINY["text_dim"], generator=generator)
    image = (
        torch.randn(1, IMAGE_TOKENS, IMAGE_CONDITIONED["image_dim"], generator=generator)
        if variant == "i2v"
        else None
    )
    with torch.no_grad():
        return model(
            latents,
            torch.tensor([500.0]),
            prompt,
            encoder_hidden_states_image=image,
            return_dict=False,
        )[0]


def _write_checkpoint(model, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    model.save_config(root)
    safetensors_torch.save_file(
        {name: value.contiguous() for name, value in model.state_dict().items()},
        root / "diffusion_pytorch_model.safetensors",
    )


def _worker(rank: int, port: int, checkpoint: str, variant: str, results) -> None:
    from chitu_diffusion.models.wan.tensor_parallel import (
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
    model, rewrite = load_tensor_parallel_transformer(
        transformer_wan.WanTransformer3DModel,
        checkpoint,
        degree=WORLD_SIZE,
        subfolder=None,
        torch_dtype=torch.float32,
    )
    results[rank] = _forward(model, variant)
    if rank == 0:
        results["rewrite"] = rewrite
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("variant", sorted(CONFIGS))
def test_the_sharded_wan_transformer_reproduces_the_dense_forward(
    variant: str,
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dense = _dense_model(variant)
        _write_checkpoint(dense, root)
        expected = _forward(dense, variant)

        manager = mp.Manager()
        results = manager.dict()
        mp.spawn(
            _worker,
            args=(_free_port(), str(root), variant, results),
            nprocs=WORLD_SIZE,
            join=True,
        )

    rewrite = results["rewrite"]
    assert rewrite.degree == WORLD_SIZE
    assert len(rewrite.norm) == (10 if variant == "i2v" else 8)
    assert {local for _, _, local in rewrite.head_counts} == {
        TINY["num_attention_heads"] // WORLD_SIZE
    }

    for rank in range(WORLD_SIZE):
        torch.testing.assert_close(results[rank], expected, rtol=1e-6, atol=1e-6)
