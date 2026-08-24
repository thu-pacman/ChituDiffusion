"""Generate an image with the Qwen-Image wrapper API."""

import os

import torch

from chitu_diffusion import QwenImagePipeline, QwenImageRequest

pipeline = QwenImagePipeline.from_pretrained(
    os.environ["QWEN_IMAGE_MODEL_PATH"],
    torch_dtype=torch.bfloat16,
)
try:
    result = pipeline.generate(
        QwenImageRequest(
            prompt="a red cube on a white table",
            width=512,
            height=512,
            num_steps=50,
            seed=7,
        )
    )
    if pipeline.parallel_context.rank == 0:
        result.images[0].save("qwen-image.png")
finally:
    pipeline.close()
