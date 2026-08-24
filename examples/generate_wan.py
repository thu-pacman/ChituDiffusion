"""Generate a video with the Wan wrapper API."""

import os

import torch
from diffusers.utils import export_to_video

from chitu_diffusion import WanPipeline, WanRequest

pipeline = WanPipeline.from_pretrained(
    os.environ["WAN_MODEL_PATH"],
    torch_dtype=torch.bfloat16,
)
try:
    result = pipeline.generate(
        WanRequest(
            prompt="a cat walking on grass",
            num_frames=17,
            num_steps=30,
            seed=42,
            output_type="np",
        )
    )
    if pipeline.parallel_context.rank == 0:
        export_to_video(result.frames[0], "wan.mp4", fps=16)
finally:
    pipeline.close()
