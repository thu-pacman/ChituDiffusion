"""Generate an image with the LLaDA-Image wrapper API."""

import os

import torch

from chitu_diffusion import LLaDAImagePipeline, LLaDAImageRequest

pipeline = LLaDAImagePipeline.from_pretrained(
    os.environ["LLADA_IMAGE_MODEL_PATH"],
    torch_dtype=torch.bfloat16,
)
try:
    result = pipeline.generate(
        LLaDAImageRequest(
            prompt="a cinematic photograph of a red fox in the snow",
            generation_mode="text",
            width=1024,
            height=1024,
            num_inference_steps=50,
            guidance_scale=4.5,
            seed=42,
        )
    )
    if pipeline.parallel_context.rank == 0:
        result.images[0].save("llada-image.png")
finally:
    pipeline.close()
