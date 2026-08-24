"""Compare static NCCL CP with the optional Fast AGKV transport."""

import argparse
import os

import torch

from chitu_diffusion import ZImagePipeline, ZImageRequest

parser = argparse.ArgumentParser()
parser.add_argument("--transport", choices=("torch", "fast_agkv"), default="torch")
args = parser.parse_args()
world_size = int(os.environ.get("WORLD_SIZE", "1"))

pipeline = ZImagePipeline.from_pretrained(
    os.environ["ZIMAGE_MODEL_PATH"],
    torch_dtype=torch.bfloat16,
    allowed_lane_widths=(world_size,),
    attention_mode="agkv",
    agkv_transport=args.transport,
)
try:
    result = pipeline.generate(
        ZImageRequest(prompt="a red cube on a white table", seed=7)
    )
    if pipeline.parallel_context.rank == 0:
        result.images[0].save(f"zimage-{args.transport}.png")
finally:
    pipeline.close()
