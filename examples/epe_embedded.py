"""Submit one Z-Image request through the non-HTTP EPE lifecycle."""

import os

import torch

from chitu_diffusion import (
    EmbeddedDiffusionRuntime,
    HotSwitchPoolConfig,
    StageWorldSpec,
    ZImageRequest,
)
from chitu_diffusion.models.zimage import ZImageExecutorFactory

world = StageWorldSpec.from_torchrun("embedded-example")
runtime = EmbeddedDiffusionRuntime.create(
    world=world,
    pool=HotSwitchPoolConfig(
        policy="elastic",
        allowed_lane_widths=tuple(
            width
            for width in range(1, world.world_size + 1)
            if world.world_size % width == 0
        ),
        warmup_resolutions=((512, 512),),
        warmup_steps=3,
    ),
    executor_factory=ZImageExecutorFactory(
        model_path=os.environ["ZIMAGE_MODEL_PATH"],
        torch_dtype=torch.bfloat16,
        default_width=512,
        default_height=512,
    ),
)
runtime.start()
if runtime.is_leader:
    request_id = runtime.submit(
        ZImageRequest(
            request_id="embedded-example",
            prompt="a red cube on a white table",
            width=512,
            height=512,
        )
    )
    while True:
        completed = runtime.poll(timeout_s=0.1)
        match = next(
            (item for item in completed if item.request_id == request_id), None
        )
        if match is not None:
            if match.status != "completed":
                raise RuntimeError(match.error)
            match.output.save("zimage-embedded.png")
            break
    runtime.stop()
else:
    runtime.wait()
    runtime.stop(graceful=False)
