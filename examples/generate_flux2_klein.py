"""Generate an image with the fixed-CP FLUX.2-klein pipeline."""

import os

import torch

from chitu_diffusion import Flux2KleinCpPipeline

device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
pipeline = Flux2KleinCpPipeline.from_pretrained(
    os.environ["FLUX2_KLEIN_MODEL_PATH"],
    torch_dtype=torch.bfloat16,
).to(device)
try:
    result = pipeline(
        prompt="a red cube on a white table",
        width=512,
        height=512,
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=torch.Generator(device=device).manual_seed(7),
    )
    if pipeline.parallel_context.rank == 0:
        result.images[0].save("flux2-klein.png")
finally:
    pipeline.close()
