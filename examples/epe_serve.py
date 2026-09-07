"""Serve a loaded wrapper through the blocking EPE HTTP lifecycle."""

import os

import torch

from chitu_diffusion import EPEServeConfig, ZImagePipeline

pipeline = ZImagePipeline.from_pretrained(
    os.environ["ZIMAGE_MODEL_PATH"],
    torch_dtype=torch.bfloat16,
)
try:
    pipeline.serve(
        EPEServeConfig(
            host="0.0.0.0",
            port=18200,
            warmup_resolutions=((512, 512),),
            warmup_steps=3,
            schedule_strategy="elastic",
        )
    )
finally:
    pipeline.close()
