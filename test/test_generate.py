import hydra
import torch
import time
import os
import random
import logging
from logging import getLogger
import resource

from chitu_diffusion.chitu_diffusion_main import (
    chitu_init,
    chitu_generate,
    warmup_diffusion_engine,
    chitu_start,
    chitu_terminate,
    chitu_run_eval,
    chitu_is_terminated
)

# from chitu_core.task import UserRequest, TaskPool, Task
from chitu_diffusion.task import DiffusionUserRequest, DiffusionTask, DiffusionTaskPool, DiffusionUserParams

from chitu_core.global_vars import get_timers
from chitu_core.schemas import ServeConfig
from chitu_core.utils import get_config_dir_path, gen_req_id

logger = getLogger(__name__)

# T2V prompts
msgs = [
    DiffusionUserParams(
    role="Alex",
    prompt="A cat walking on grass.",
    seed=42,
    frame_num=81,
    size=(832,480), # 14b: 1280 720
    negative_prompt='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走',
    num_inference_steps=50,
    sample_solver='unipc',
    flexcache='PAB', # TODO: 统一的 quality/latency tradeoff 参数
),
    DiffusionUserParams(
    role="Bob",
    prompt="two cats walking on grass.",
    seed=42,
    frame_num=81,
    size=(832,480),
    negative_prompt='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走',
    num_inference_steps=5,
    sample_solver='unipc',
),

]

def gen_reqs(num_reqs, max_new_tokens, frequency_penalty, is_vl=False):
    reqs: list[DiffusionUserRequest] = []
    for i in range(num_reqs):
        req = DiffusionUserRequest(
            request_id = f"{gen_req_id()}",
            params=msgs[i]
        )
        reqs.append(req)
    return reqs

def run_normal(args, timers):
    rank = torch.distributed.get_rank()
    warmup_diffusion_engine(args)

    for i in range(1):
        chitu_start()
        if rank == 0:
            reqs = gen_reqs(
                num_reqs=len(msgs), # args.infer.max_reqs
                max_new_tokens=args.request.max_new_tokens,
                frequency_penalty=args.request.frequency_penalty,
                is_vl=hasattr(args.models, "vision_config"),
            )
            logger.info(f'{reqs=}')
            for req in reqs:
                DiffusionTaskPool.add(DiffusionTask(task_id=req.request_id, req=req))
            
        logger.info(f"------ batch {i} ------")
        t_start = time.time()
        timers("overall").start()

        while not chitu_is_terminated():
            chitu_generate()
            if rank == 0 and DiffusionTaskPool.all_finished():
                break
            
        if rank == 0:
            timers("overall").stop()
            t_end = time.time()
            logger.info(f"Time cost {t_end - t_start}")
        logger.info(
            f"[Final] | GPU-Alloc:{torch.cuda.memory_allocated()/1024**3:.3f} Max:{torch.cuda.max_memory_allocated()/1024**3:.3f} Rsrv:{torch.cuda.memory_reserved()/1024**3:.3f} GB  | CPU:{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.3f} GB"
        )

        timers.log()
        
    chitu_terminate()
    chitu_run_eval()


@hydra.main(
    version_base=None,
    config_path=os.getenv("CONFIG_PATH", get_config_dir_path()),
    config_name=os.getenv("CONFIG_NAME", "serve_config"),
)
def main(args: ServeConfig):
    global local_args
    local_args = args
    logger.setLevel(logging.DEBUG)
    if os.getenv("RANK") == 0:
        logger.info(f"Run with args: {args}")

    # Initialize Backend: args / distributed / load models & kernels
    chitu_init(args, logging_level=logging.INFO)
    logger.info("initialized chitu_core.")
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    timers = get_timers()
    logger.debug("finish init")
    
    run_normal(args, timers)


if __name__ == "__main__":
    main()

    # Sometimes torch.distributed will hang during destruction if CUDA graph is enabled.
    # As a workaround, we `exec` a dummy process to kill the current process, without
    # returning an error.
    logger.info("Waiting for all ranks to finish...")
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    # Don't exec bash because it loads startup scripts
    os.execl("/usr/bin/true", "true")  # /usr/bin/true does nothing but exits