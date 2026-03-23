import hydra
import torch
import time
import os
import random
import logging
import json
import copy
import re
from datetime import datetime
from logging import getLogger
import resource
from dataclasses import asdict

from omegaconf import OmegaConf

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

from chitu_core.schemas import ServeConfig
from chitu_core.utils import get_config_dir_path, gen_req_id
from chitu_diffusion.bench.timer import Timer

logger = getLogger(__name__)

PROMPT_SLUG_LEN = 32

# T2V prompts
msgs = [
    DiffusionUserParams(
    role="Alex",
    prompt="A cat walking on grass.",
    seed=42,
    frame_num=81,
    size=(832,480), # 14b: 1280 720
    negative_prompt='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走',
    num_inference_steps=40,
    sample_solver='unipc',
),
#     DiffusionUserParams(
#     role="Bob",
#     prompt="two cats walking on grass.",
#     seed=42,
#     frame_num=81,
#     size=(832,480),
#     negative_prompt='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走',
#     num_inference_steps=5,
#     sample_solver='unipc',
# ),

]

def gen_reqs(num_reqs, max_new_tokens, frequency_penalty, is_vl=False):
    reqs: list[DiffusionUserRequest] = []
    for i in range(num_reqs):
        params = copy.deepcopy(msgs[i])
        req = DiffusionUserRequest(
            request_id = f"{gen_req_id()}",
            params=params,
        )
        reqs.append(req)
    return reqs


def _slugify(raw: str, max_len: int = PROMPT_SLUG_LEN) -> str:
    if raw is None:
        raw = ""
    text = str(raw).strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    if not text:
        text = "prompt"
    return text[:max_len]


def _build_run_output_dir(args, reqs: list[DiffusionUserRequest]) -> str:
    root_dir = str(getattr(args.output, "root_dir", "outputs") or "outputs")
    root_dir = root_dir.strip() or "outputs"
    first_prompt = reqs[0].params.prompt if reqs else "prompt"
    prompt_slug = _slugify(first_prompt, max_len=PROMPT_SLUG_LEN)
    model_slug = _slugify(getattr(args.models, "name", "model"), max_len=64)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{prompt_slug}_{ts}_{model_slug}"
    run_tag = str(os.getenv("CHITU_RUN_TAG", "")).strip()
    if run_tag:
        run_tag_slug = _slugify(run_tag, max_len=PROMPT_SLUG_LEN)
        if run_tag_slug:
            run_dir_name = f"{run_tag_slug}_{run_dir_name}"
    return os.path.join(root_dir, run_dir_name)


def _attach_run_log_handler(run_output_dir: str):
    os.makedirs(run_output_dir, exist_ok=True)
    root_logger = logging.getLogger()
    run_log_path = os.path.join(run_output_dir, "run.log")
    handler = logging.FileHandler(run_log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger.addHandler(handler)
    return handler


def _dump_run_metadata(run_output_dir: str, args, reqs: list[DiffusionUserRequest]):
    os.makedirs(run_output_dir, exist_ok=True)
    req_items = []
    for req in reqs:
        req_items.append(
            {
                "request_id": req.request_id,
                "params": asdict(req.params),
            }
        )

    with open(os.path.join(run_output_dir, "request_params.json"), "w", encoding="utf-8") as f:
        json.dump({"requests": req_items}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(run_output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": str(getattr(args.models, "name", "")),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "num_requests": len(reqs),
                "output_dir": os.path.abspath(run_output_dir),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(os.path.join(run_output_dir, "run_config.yaml"), "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(args, resolve=True))

def run_normal(args):
    rank = torch.distributed.get_rank()
    warmup_diffusion_engine(args)

    for i in range(1):
        chitu_start()
        reqs = []
        run_output_dir = None
        if rank == 0:
            reqs = gen_reqs(
                num_reqs=len(msgs), # args.infer.max_reqs
                max_new_tokens=args.request.max_new_tokens,
                frequency_penalty=args.request.frequency_penalty,
                is_vl=hasattr(args.models, "vision_config"),
            )
            run_output_dir = _build_run_output_dir(args, reqs)
            for req in reqs:
                req.params.save_dir = run_output_dir

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            payload = [run_output_dir]
            torch.distributed.broadcast_object_list(payload, src=0)
            run_output_dir = payload[0]

        if run_output_dir:
            os.environ["CHITU_CURRENT_OUTPUT_DIR"] = run_output_dir

        log_handler = None
        if rank == 0 and getattr(args.output, "enable_run_log", True):
            log_handler = _attach_run_log_handler(run_output_dir)

        if rank == 0:
            _dump_run_metadata(run_output_dir, args, reqs)
            logger.info(f'{reqs=}')
            for req in reqs:
                DiffusionTaskPool.add(DiffusionTask(task_id=req.request_id, req=req))
            
        logger.info(f"------ batch {i} ------")
        t_start = time.time()
        with Timer.get_timer("overall"):
            while not chitu_is_terminated():
                chitu_generate()
                if rank == 0 and DiffusionTaskPool.all_finished():
                    break
            
        if rank == 0:
            t_end = time.time()
            logger.info(f"Time cost {t_end - t_start}")
        logger.info(
            f"[Final] | GPU-Alloc:{torch.cuda.memory_allocated()/1024**3:.3f} Max:{torch.cuda.max_memory_allocated()/1024**3:.3f} Rsrv:{torch.cuda.memory_reserved()/1024**3:.3f} GB  | CPU:{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.3f} GB"
        )

        if rank == 0 and getattr(args.output, "enable_timer_dump", False):
            Timer.print_statistics()
            Timer.save_statistics(os.path.join(run_output_dir, "timer_stats.csv"))
        
    chitu_terminate()
    chitu_run_eval()

    if rank == 0 and log_handler is not None:
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()


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

    logger.debug("finish init")
    
    run_normal(args)


if __name__ == "__main__":
    main()

    # Sometimes torch.distributed will hang during destruction if CUDA graph is enabled.
    # As a workaround, we `exec` a dummy process to kill the current process, without
    # returning an error.
    logger.info("Waiting for all ranks to finish...")
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    # Don't exec bash because it loads startup scripts
    os.execl("/usr/bin/true", "true")  # /usr/bin/true does nothing but exits