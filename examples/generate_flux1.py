"""Generate an image with the FLUX.1 wrapper API."""

import os

import torch

from chitu_diffusion import Flux1Pipeline, Flux1Request

pipeline = Flux1Pipeline.from_pretrained(
    os.environ["FLUX1_MODEL_PATH"],
    torch_dtype=torch.bfloat16,
)
try:
    result = pipeline.generate(
        Flux1Request(
            prompt="a red cube on a white table",
            width=512,
            height=512,
            num_steps=28,
            seed=7,
        )
    )
    if pipeline.parallel_context.rank == 0:
        result.images[0].save("flux1.png")
finally:
    pipeline.close()
