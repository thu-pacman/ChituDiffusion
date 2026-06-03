# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import sys
import math
import time
import torch
import torch.distributed as dist
from typing import Optional, List, Tuple, Dict, Any
from functools import partial
from contextlib import contextmanager
import numpy as np
import torch.amp as amp
from tqdm import tqdm

from logging import getLogger
from chitu_diffusion.core.global_vars import get_global_args, get_slot_handle
from chitu_diffusion.core.logging_utils import (
    log_stage,
    log_progress,
    log_result,
    log_perf,
    should_log_info_on_rank,
    should_log_on_rank,
)
from chitu_diffusion.runtime.backend import BackendState, CFGType, DiffusionBackend
from chitu_diffusion.flexcache.params import CubicParams, DiTangoParams, FlexCacheParams
from chitu_diffusion.runtime.task import (
    DiffusionTask,
    DiffusionTaskType,
    DiffusionTaskPool,
    DiffusionTaskStatus,
)
from chitu_diffusion.core.distributed.parallel_state import (
    get_cfg_group,
    get_cp_group,
    get_up_group,
    get_world_group
)
from chitu_diffusion.modules.samplers.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from chitu_diffusion.modules.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from chitu_diffusion.modules.utils.wan import cache_video
from chitu_diffusion.runtime.parallel_utils import SequencePadder
from chitu_diffusion.observability import Timer, MagLogger
from chitu_diffusion.runtime.output_naming import build_video_name_from_task
from chitu_diffusion.runtime.output_layout import (
    append_json_list_item,
    memory_metrics_dir,
    task_results_dir,
    timing_metrics_dir,
    write_json,
)
from chitu_diffusion.modules.utils.flux import (
    calculate_flux1_shift,
    compute_flux2_empirical_mu,
    flowmatch_sigmas,
    prepare_flux1_latents,
    prepare_flux2_latents,
    retrieve_timesteps as retrieve_flowmatch_timesteps,
    unpack_flux1_latents,
    unpack_flux2_latents_with_ids,
    unpatchify_flux2_latents,
)
from chitu_diffusion.runtime.image_output import save_image_as_png


logger = getLogger(__name__)

from contextlib import contextmanager

@contextmanager
def device_scope(model: torch.nn.Module):
    original_device = None
    
    if model is not None and torch.cuda.is_available():
        # 记录原始设备
        original_device = next(model.parameters()).device if len(list(model.parameters())) > 0 else None
        model.to(torch.cuda.current_device())
    
    try:
        DiffusionBackend.memory_used(f"Loaded model to {torch.cuda.current_device()}")
        yield model
    finally:
        if model is not None:
            # 恢复到原始设备，而不是强制CPU
            target_device = original_device if original_device is not None else "cpu"
            model.to(target_device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            DiffusionBackend.memory_used(f"Offloaded {target_device}")

class DiffusionTaskDispatcher():
    def __init__(self):
        super().__init__()
        self.group = get_world_group()
        self.rank = self.group.global_rank
        self.local_rank = self.group.local_rank

        self.main_rank = 0
        self.is_main_rank = self.group.is_first_rank
        logger.debug("init diffusion task dispatcher")


    def dispatch_metadata(self, task: Optional[DiffusionTask] = None) -> tuple[DiffusionTaskType, DiffusionTask]:
        if self.is_main_rank:
            assert task is not None
            # 发送方：序列化任务并获取大小
            task_tensor = task.serialize(
                device="cpu" if DiffusionBackend.use_gloo else self.local_rank
            )
            task_type = task.task_type
            task_size = torch.tensor([task_tensor.size()[0]], dtype=torch.int64,
                                    device="cpu" if DiffusionBackend.use_gloo else self.local_rank)
            
            # 第一阶段：广播任务大小
            logger.debug(f"Rank {self.rank}: Broadcasting task size {task_size.item()}")
            dist.broadcast(
                tensor=task_size,
                src=self.main_rank,
            )
            
        else:
            # 接收方：创建空的size tensor用于接收
            task_size = torch.zeros(1, dtype=torch.int64,
                                device="cpu" if DiffusionBackend.use_gloo else self.local_rank)
            
            # 第一阶段：接收任务大小
            dist.broadcast(
                tensor=task_size,
                src=self.main_rank,
            )
            
            # 第二阶段：根据接收到的大小创建空buffer
            logger.debug(f"Rank {self.rank}: Received task size {task_size.item()}, creating buffer")
            task_tensor = DiffusionTask.create_empty_serialization(
                size=task_size.item(),  # 注意这里要用 .item() 获取标量值
                device="cpu" if DiffusionBackend.use_gloo else self.local_rank
            )
        
        logger.debug(f"Rank {self.rank} | {task_tensor.shape=} {task_size[0]=} {task_tensor.dtype=} {task_tensor.device=}, ready to broadcast.")
        
        # 第二阶段：广播实际的任务数据
        dist.broadcast(
            tensor=task_tensor,
            src=self.main_rank,
        )
        
        if not self.is_main_rank:
            # 接收方：反序列化任务
            task = DiffusionTask.deserialize(task_tensor)
            task_type = task.task_type
        
        logger.debug(f"Rank {self.rank}: {task=}")
        return task_type, task

    def send_payload(self, *args, **kwargs):
        return
    
    def recv_payload(self, *args, **kwargs):
        return
    

class CfgDispatcher():
    def __init__(self):
        super().__init__()
        self.group = get_cfg_group()
        self.rank = self.group.global_rank
        self.local_rank = self.group.local_rank

    def all_gather_cfg_noise_preds(self, local_noise_pred: torch.Tensor):

        gathered_preds = [torch.empty_like(local_noise_pred) for _ in range(2)]
        dist.all_gather(
            tensor_list=gathered_preds,
            tensor=local_noise_pred,
            group=self.group.gpu_group,
        )
        return gathered_preds[0], gathered_preds[1]

class ContextParallelDispatcher():
    def __init__(self):
        super().__init__()
        self.group = get_cp_group()
        self.cp_size = self.group.group_size
        self.rank = self.group.global_rank
        self.local_rank = self.group.local_rank
        self.rank_in_group = self.group.rank_in_group

    def dispatch(self, tokens: torch.Tensor):
        return SequencePadder.split_sequence_padding(tokens, 
                                                     split_num=self.cp_size,
                                                     split_dim=1, 
                                                     name='x')[self.rank_in_group]
    
    def gather(self, tokens: torch.Tensor):
        tokens_list = [torch.empty_like(tokens) for _ in range(self.cp_size)]
        dist.all_gather(tensor_list=tokens_list, 
                        tensor=tokens, 
                        group=self.group.gpu_group)
        return SequencePadder.remove_sequence_padding_and_concat(tokens_list, 
                                                                 gather_dim=1,
                                                                 name='x')
    
    def wrap_model_compute_with_cp(self):
        """替换DiffusionBackend.model.model_compute方法，添加CP支持"""
        
        def create_wrapped_forward(model_instance):
            original_forward = model_instance.model_compute
            def wrapped_compute(tokens, **kwargs):
                tokens = self.dispatch(tokens)
                if "seq_lens" in kwargs.keys():
                    kwargs["seq_lens"] = torch.tensor([tokens.size(1)])
                x = original_forward(tokens, **kwargs)
                x = self.gather(x)
                return x
            return wrapped_compute

        for model in DiffusionBackend.model_pool:
            model.model_compute = create_wrapped_forward(model)

class Generator:
    @classmethod
    def build(cls, args) -> "Generator":
        return cls(args)
    
    def __init__(self, args):
        self.args = args
        self.rank = torch.distributed.get_rank()
        self.local_rank: int = self.rank % 8 
        self.task_dispatchers: List = []
        self.cp_size = args.infer.diffusion.cp_size
        self.cfg_size = get_cfg_group().group_size
        if self.cp_size > 1:
            self.cp_dispatcher = ContextParallelDispatcher()
            self.cp_dispatcher.wrap_model_compute_with_cp()
        if self.cfg_size == 2:
            self.cfg_dispatcher = CfgDispatcher()

        # Ensure FIFO
        self.current_task = None # 通过这个储存当前任务的中间状态
        self._last_logged_stage = {}
        self.denoise_progress_interval = max(1, int(os.getenv("CHITU_PROGRESS_INTERVAL", "5")))
        self.enable_stage_perf = bool(getattr(args.output, "timer", False))
        self._stage_start_time = {}
        self._dit_forward_step_elapsed_ms = {}

    def _run_dit_forward(self, task: DiffusionTask, branch: str, *args, **kwargs):
        step_index = int(task.buffer.current_step)
        timestep = None
        if task.buffer.timesteps is not None and step_index < len(task.buffer.timesteps):
            raw_timestep = task.buffer.timesteps[step_index]
            timestep = float(raw_timestep.item()) if hasattr(raw_timestep, "item") else float(raw_timestep)

        forward_call_index = getattr(task.buffer, "_dit_forward_call_index", 0)
        setattr(task.buffer, "_dit_forward_call_index", forward_call_index + 1)

        output, elapsed_ms = Timer.time_call("dit_forward", DiffusionBackend.active_model, *args, **kwargs)
        Timer.record_event(
            "dit_forward",
            {
                "task_id": task.task_id,
                "step_index": step_index,
                "timestep": timestep,
                "branch": branch,
                "call_index": forward_call_index,
                "rank": self.rank,
                "elapsed_ms": elapsed_ms,
            },
        )
        step_key = (task.task_id, step_index)
        self._dit_forward_step_elapsed_ms[step_key] = self._dit_forward_step_elapsed_ms.get(step_key, 0.0) + elapsed_ms
        return output

    def _record_dit_forward_step_summary(self, task: DiffusionTask) -> None:
        step_index = int(task.buffer.current_step)
        step_key = (task.task_id, step_index)
        elapsed_ms = self._dit_forward_step_elapsed_ms.pop(step_key, 0.0)
        timestep = None
        if task.buffer.timesteps is not None and step_index < len(task.buffer.timesteps):
            raw_timestep = task.buffer.timesteps[step_index]
            timestep = float(raw_timestep.item()) if hasattr(raw_timestep, "item") else float(raw_timestep)
        Timer.record_event(
            "dit_forward_step",
            {
                "task_id": task.task_id,
                "step_index": step_index,
                "timestep": timestep,
                "rank": self.rank,
                "elapsed_ms": elapsed_ms,
            },
        )

    def _emit_stage_start_if_needed(self, task: DiffusionTask):
        stage = task.task_type.name
        if self._last_logged_stage.get(task.task_id) == stage:
            return

        if self.enable_stage_perf:
            self._stage_start_time[(task.task_id, stage)] = time.perf_counter()

        if stage == "Denoise":
            total = task.req.params.num_inference_steps
            log_stage(
                logger,
                stage_name=stage,
                event="START",
                task_id=task.task_id,
                extra=f"step={0:>3}/{total:<3} pct={0.0:>5.1f}%",
            )
        else:
            log_stage(logger, stage_name=stage, event="START", task_id=task.task_id)
        self._last_logged_stage[task.task_id] = stage

    def _emit_stage_end(self, stage: DiffusionTaskType, task_id: str):
        log_stage(logger, stage_name=stage.name, event="END", task_id=task_id)
        if self.enable_stage_perf:
            start = self._stage_start_time.pop((task_id, stage.name), None)
            if start is not None:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                log_perf(logger, task_id=task_id, stage_name=stage.name, elapsed_ms=elapsed_ms)


    def step(self, task: Optional[DiffusionTask]) -> torch.Tensor:
        # 调度器会给generator task，翻译成kernel -> 运行 -> 正确放置输出 -> 回收对应内存
        # Prepare Payload
        if self.current_task is None: # 生成的最开始，将任务分发到workers
            task_type, task = DiffusionTaskDispatcher().dispatch_metadata(task)
            self.current_task = task
        else:
            task = self.current_task # 接续当前任务

        torch.cuda.synchronize()
        dist.barrier()
        task_type = task.task_type if task is not None else None
        assert self.current_task.task_id == task.task_id # 确保任务逐个完成，避免产生太多中间状态
        
        task.status = DiffusionTaskStatus.Running

        self._emit_stage_start_if_needed(task)

        if task_type == DiffusionTaskType.Terminate:
            self._clear_ditango_planner()
            self._clear_flexcache_strategy()
            DiffusionBackend.state = BackendState.Terminated
            task.status = DiffusionTaskStatus.Completed
            return None

        # Execute
        if task_type == DiffusionTaskType.TextEncode:
            out = self.text_encode_step(task)
        elif task_type == DiffusionTaskType.VAEEncode:
            out = self.vae_encode_step(task)
        elif task_type == DiffusionTaskType.VAEDecode:
            out = self.vae_decode_step(task)
        elif task_type == DiffusionTaskType.Denoise:
            out = self.denoise_step(task)
        else:
            raise NotImplementedError  
        
        self._update_task_stage_and_buffer(task, out)

        return out
        
    @Timer.get_timer("TextEncode")
    def text_encode_step(self, task: DiffusionTask) -> torch.Tensor:
        # payload 是本次task需要处理的数据的抽象
        if self.cfg_size == 1:
            if task.buffer.text_embeddings is None:
                payload = task.req.get_prompt()
                DiffusionBackend.cfg_type = CFGType.POS
            else:
                payload = task.req.get_n_prompt()
                DiffusionBackend.cfg_type = CFGType.NEG
        elif self.cfg_size == 2:
            if get_cfg_group().rank_in_group == 0:
                payload = task.req.get_prompt()
                DiffusionBackend.cfg_type = CFGType.POS
            else:
                payload = task.req.get_n_prompt()
                DiffusionBackend.cfg_type = CFGType.NEG

        logger.debug(f"[text_encode_step] task_id={task.task_id}, prompt_len={len(payload)}")

        if DiffusionBackend.args.models.name in ["FLUX.1-dev"]:
            with device_scope(DiffusionBackend.text_encoder):
                prompt_embeds, pooled_prompt_embeds, text_ids = DiffusionBackend.text_encoder.encode(
                    payload,
                    max_sequence_length=getattr(DiffusionBackend.args.models.encoder, "max_sequence_length", 512),
                    device=torch.device(torch.cuda.current_device()),
                )
            task.buffer.pooled_prompt_embeds = pooled_prompt_embeds
            task.buffer.text_ids = text_ids
            logger.info(
                "[text_encode_step] FLUX1 prompt_embeds=%s pooled=%s text_ids=%s",
                tuple(prompt_embeds.shape),
                tuple(pooled_prompt_embeds.shape),
                tuple(text_ids.shape),
            )
            return prompt_embeds

        if DiffusionBackend.args.models.name in ["FLUX.2-klein-4B"]:
            layers = tuple(getattr(DiffusionBackend.args.models.encoder, "hidden_states_layers", (9, 18, 27)))
            with device_scope(DiffusionBackend.text_encoder):
                prompt_embeds, text_ids = DiffusionBackend.text_encoder.encode(
                    payload,
                    max_sequence_length=getattr(DiffusionBackend.args.models.encoder, "max_sequence_length", 512),
                    hidden_states_layers=layers,
                    device=torch.device(torch.cuda.current_device()),
                )
            task.buffer.text_ids = text_ids
            logger.info(
                "[text_encode_step] FLUX2 prompt_embeds=%s text_ids=%s",
                tuple(prompt_embeds.shape),
                tuple(text_ids.shape),
            )
            return prompt_embeds
        
        with device_scope(DiffusionBackend.text_encoder.model):        
            out = DiffusionBackend.text_encoder(payload, torch.cuda.current_device())
            
        logger.debug(f"[text_encode_step] context shape: {out.shape}")
        return out
    
    def vae_encode_step(self, task: DiffusionTask):
        pass
    
    @Timer.get_timer("denoise")
    @amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def denoise_step(self, task: DiffusionTask):
        assert task.buffer.latents is not None and task.buffer.timesteps is not None
        setattr(task.buffer, "_dit_forward_call_index", 0)

        if DiffusionBackend.args.models.name in ["FLUX.1-dev"]:
            latents = task.buffer.latents
            timestep = task.buffer.timesteps[task.buffer.current_step]
            guidance = task.buffer.guidance_vec
            if guidance is not None:
                guidance = guidance.to(device=latents.device, dtype=torch.float32)
            timestep_vec = timestep.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = self._run_dit_forward(
                task,
                "pos",
                hidden_states=latents,
                timestep=timestep_vec / 1000,
                guidance=guidance,
                pooled_projections=task.buffer.pooled_prompt_embeds,
                encoder_hidden_states=task.buffer.text_embeddings,
                txt_ids=task.buffer.text_ids,
                img_ids=task.buffer.latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            sampled_latents = task.buffer.sampler.step(noise_pred, timestep, latents, return_dict=False)[0]
            self._record_dit_forward_step_summary(task)
            return sampled_latents

        if DiffusionBackend.args.models.name in ["FLUX.2-klein-4B"]:
            x = task.buffer.latents.to(DiffusionBackend.active_model.dtype)
            t_curr = task.buffer.timesteps[task.buffer.current_step]
            t_vec = t_curr.expand(x.shape[0]).to(x.dtype)

            pred = self._run_dit_forward(
                task,
                "pos",
                hidden_states=x,
                timestep=t_vec / 1000,
                guidance=None,
                encoder_hidden_states=task.buffer.text_embeddings,
                txt_ids=task.buffer.text_ids,
                img_ids=task.buffer.latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

            pred = pred[:, : task.buffer.latents.size(1)]
            sampled_latents = task.buffer.sampler.step(pred, t_curr, task.buffer.latents, return_dict=False)[0]
            self._record_dit_forward_step_summary(task)
            return sampled_latents

        latent_model_input = task.buffer.latents
        timestep = task.buffer.timesteps[task.buffer.current_step]

        if DiffusionBackend.guidance_scale > 0: # Wan的cfg是做两次
            if self.cfg_size == 2:
                if get_cfg_group().rank_in_group == 0:
                    context = task.buffer.text_embeddings
                    DiffusionBackend.cfg_type = CFGType.POS
                else:
                    context = task.buffer.negative_embeddings
                    DiffusionBackend.cfg_type = CFGType.NEG

                cfg_branch = "pos" if DiffusionBackend.cfg_type == CFGType.POS else "neg"
                cfg_partial_noise_pred = self._run_dit_forward(
                    task,
                    cfg_branch,
                    latent_model_input,
                    t=timestep,
                    context=context,
                    seq_len=task.buffer.seq_len
                )
                noise_pred_cond, noise_pred_uncond = self.cfg_dispatcher.all_gather_cfg_noise_preds(cfg_partial_noise_pred)
            else:
                DiffusionBackend.cfg_type = CFGType.POS
                noise_pred_cond = self._run_dit_forward(
                    task,
                    "pos",
                    latent_model_input,
                    t=timestep,
                    context=task.buffer.text_embeddings,
                    seq_len=task.buffer.seq_len
                )
                DiffusionBackend.cfg_type = CFGType.NEG
                noise_pred_uncond = self._run_dit_forward(
                    task,
                    "neg",
                    latent_model_input,
                    t=timestep,
                    context=task.buffer.negative_embeddings,
                    seq_len=task.buffer.seq_len
                )
            noise_pred = noise_pred_uncond + \
                DiffusionBackend.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            DiffusionBackend.cfg_type = CFGType.POS
            noise_pred = self._run_dit_forward(
                task,
                "pos",
                latent_model_input,
                t=timestep,
                context=task.buffer.text_embeddings,
                seq_len=task.buffer.seq_len
            )

        cache_strategy = getattr(DiffusionBackend.flexcache, "strategy", None)
        observe_guided_output = getattr(cache_strategy, "observe_guided_output", None)
        if observe_guided_output is not None:
            observe_guided_output(
                x=latent_model_input,
                output=noise_pred,
                t=timestep,
                step=int(task.buffer.current_step),
            )

        sampled_latents = task.buffer.sampler.step(
            noise_pred.unsqueeze(0),
            timestep,
            task.buffer.latents.unsqueeze(0),
            return_dict=False,
            generator=task.buffer.seed_g
        )[0].squeeze(0)

        self._record_dit_forward_step_summary(task)
        return sampled_latents

        
    @Timer.get_timer("VaeDecode")
    def vae_decode_step(self, task: DiffusionTask):
        target_decode_device = 0 # TODO
        logger.debug(f"task_id={task.task_id} entering VAE decode at step={task.buffer.current_step}")
        if torch.distributed.get_rank() == target_decode_device:
            payload = [task.buffer.latents]

            if DiffusionBackend.args.models.name in ["FLUX.1-dev"]:
                width, height = task.buffer.image_size or task.req.params.size
                latents = unpack_flux1_latents(
                    task.buffer.latents,
                    height=height,
                    width=width,
                    vae_scale_factor=16,
                )
                latents = (latents / DiffusionBackend.vae.config.scaling_factor) + DiffusionBackend.vae.config.shift_factor
                with device_scope(DiffusionBackend.vae):
                    image = DiffusionBackend.vae.decode(latents, return_dict=False)[0]
                self._save_image(task, image)
                return image

            if DiffusionBackend.args.models.name in ["FLUX.2-klein-4B"]:
                latents = unpack_flux2_latents_with_ids(task.buffer.latents, task.buffer.latent_image_ids)
                latents_bn_mean = DiffusionBackend.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
                latents_bn_std = torch.sqrt(
                    DiffusionBackend.vae.bn.running_var.view(1, -1, 1, 1) + DiffusionBackend.vae.config.batch_norm_eps
                ).to(latents.device, latents.dtype)
                latents = latents * latents_bn_std + latents_bn_mean
                latents = unpatchify_flux2_latents(latents)
                with device_scope(DiffusionBackend.vae):
                    img = DiffusionBackend.vae.decode(latents, return_dict=False)[0]
                self._save_image(task, img)
                return img
              
            with device_scope(DiffusionBackend.vae.model):
                video = DiffusionBackend.vae.decode(payload)[0]
            return video
        return None
    
    def _post_vae_decode(self, task: DiffusionTask, video: Optional[torch.Tensor]):
        target_decode_device = 0 # TODO: Flexible vae device
        if torch.distributed.get_rank() == target_decode_device:
            # save_video
            if video is None:
                logger.warning(f"task_id={task.task_id} VAE decode returned None; skip saving video")
                return
            if DiffusionBackend.args.models.name in ["FLUX.1-dev", "FLUX.2-klein-4B"]:
                logger.debug("task_id=%s image task already saved during VAE decode", task.task_id)
            else:
                self._save_video(task, video)
            # save_time_stats
            run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
            timing_dir = timing_metrics_dir(run_output_dir) if run_output_dir else task.req.params.save_dir
            memory_dir = memory_metrics_dir(run_output_dir) if run_output_dir else task.req.params.save_dir
            Timer.print_statistics()
            Timer.save_task_statistics_json(os.path.join(timing_dir, f"{task.task_id}.json"), task.task_id)
            rank = torch.distributed.get_rank()
            args = get_global_args()
            if DiffusionBackend.flexcache is not None and DiffusionBackend.flexcache.strategy is not None:
                DiffusionBackend.flexcache.record_compute_summary(task_id=task.task_id)
                DiffusionBackend.flexcache.flush_cache_memory_events()
            record_memory = bool(getattr(args.output, "memory", True)) if args is not None else True
            if record_memory and torch.cuda.is_available() and should_log_on_rank(rank):
                append_json_list_item(
                    os.path.join(memory_dir, f"rank{rank}.json"),
                    "events",
                    {
                        "task_id": task.task_id,
                        "stage": "task_complete",
                        "rank": rank,
                        "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                        "gpu_max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                        "gpu_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                    },
                    base_payload={"rank": rank},
                )
            # Magnitiude experiments
            # MagLogger.save_to_csv(save_dir=f"./experiments/{task.task_id}")

    # TODO: CPU/GPU overlap
    def _save_video(self, task: DiffusionTask, video: torch.Tensor):
        run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        if run_output_dir:
            task.req.params.save_dir = task_results_dir(run_output_dir, task.task_id)

        os.makedirs(task.req.params.save_dir, exist_ok=True)
        save_name = build_video_name_from_task(task)
        save_path = os.path.join(task.req.params.save_dir, save_name)
        logger.debug(f"Saving video tensor: shape={video.shape} dtype={video.dtype}")
        cache_video(
            tensor=video[None],
            save_file=save_path,
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))

        sidecar_path = os.path.splitext(save_path)[0] + ".json"
        metadata = {
            "filename": os.path.basename(save_path),
            "relative_path": os.path.join(os.path.basename(task.req.params.save_dir), os.path.basename(save_path)),
            "prompt": task.req.get_prompt(),
            "seed": getattr(task.req.params, "seed", None),
            "step": getattr(task.req.params, "num_inference_steps", None),
            "task_id": task.task_id,
        }
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        log_result(logger, task_id=task.task_id, message=f"video_saved={save_path}")
    
    # TODO: CPU/GPU overlap
    def _save_image(self, task: DiffusionTask, img: torch.Tensor):
        run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        if run_output_dir:
            task.req.params.save_dir = task_results_dir(run_output_dir, task.task_id)

        os.makedirs(task.req.params.save_dir, exist_ok=True)
        save_name = task.req.get_prompt()[:20].replace(" ", "_").replace(".", "") \
                    + f"_{task.task_id}.png"
        save_path = os.path.join(task.req.params.save_dir, save_name)
        logger.info(f"Saving image: {img.shape=} {img.dtype=}")
        save_image_as_png(img[0], save_path)
        log_result(logger, task_id=task.task_id, message=f"image_saved={save_path}")


    def _clear_flexcache_strategy(self):
        """Safely remove current flexcache strategy wrappers from active model."""
        manager = DiffusionBackend.flexcache
        if manager is None:
            return

        model = DiffusionBackend.active_model
        strategy = manager.strategy

        if model is not None and strategy is not None:
            try:
                strategy.unwrap_module(model)
            except Exception as e:
                logger.warning(f"Failed to unwrap flexcache strategy cleanly: {e}")

        manager.strategy = None
        manager.cache.clear()

    def _clear_ditango_planner(self):
        """Safely remove current DiTango planner wrappers from active model."""
        planner = DiffusionBackend.ditango
        if planner is None:
            return

        model = DiffusionBackend.active_model
        if model is not None:
            try:
                planner.unwrap_module(model)
            except Exception as e:
                logger.warning(f"Failed to unwrap DiTango planner cleanly: {e}")

        DiffusionBackend.ditango = None

    @staticmethod
    def _ditango_ase_from_ratio(cache_ratio: float) -> float:
        # 0 -> quality first (low threshold), 1 -> speed first (high threshold)
        min_thresh, max_thresh = 0.01, 0.08
        return min_thresh + cache_ratio * (max_thresh - min_thresh)

    def _resolve_flexcache_spec(self, task: DiffusionTask) -> Optional[FlexCacheParams]:
        spec = task.req.params.resolve_flexcache_params()
        if spec is None:
            return None

        total_steps = int(task.req.params.num_inference_steps)
        if spec.warmup + spec.cooldown >= total_steps:
            raise ValueError(
                "Invalid acceleration warmup/cooldown: "
                f"warmup({spec.warmup}) + cooldown({spec.cooldown}) must be < num_inference_steps({total_steps})."
            )
        return spec

    def _build_flexcache_strategy(self, task: DiffusionTask, spec: FlexCacheParams):
        strategy = spec.strategy
        warmup_steps = spec.warmup
        cooldown_steps = spec.cooldown

        if strategy == "teacache":
            from chitu_diffusion.flexcache.strategy.teacache import TeaCacheStrategy

            return TeaCacheStrategy(
                task=task,
                teacache_thresh=getattr(spec, "teacache_thresh", 0.2),
                coefficients=getattr(spec, "coefficients", None),
                warmup_steps=warmup_steps,
                cooldown_steps=cooldown_steps,
                use_ref_steps=getattr(spec, "use_ref_steps", True),
            )

        if strategy == "pab":
            from chitu_diffusion.flexcache.strategy.pab import PABStrategy

            return PABStrategy(
                task=task,
                warmup_steps=warmup_steps,
                cooldown_steps=cooldown_steps,
                skip_self_range=getattr(spec, "skip_self_range", 2),
                skip_cross_range=getattr(spec, "skip_cross_range", 3),
            )

        if strategy == "blockdance":
            from chitu_diffusion.flexcache.strategy.blockdance import BlockDanceStrategy

            return BlockDanceStrategy(
                task=task,
                warmup_steps=warmup_steps,
                cooldown_steps=cooldown_steps,
                boundary_block=getattr(spec, "boundary_block", 20),
                group_size=getattr(spec, "group_size", 2),
                start_fraction=getattr(spec, "start_fraction", 0.40),
                end_fraction=getattr(spec, "end_fraction", 0.95),
            )

        if strategy == "cubic":
            from chitu_diffusion.flexcache.strategy.cubic import CubicStrategy

            if not isinstance(spec, CubicParams):
                raise TypeError("FlexCache cubic requires CubicParams.")
            if self.cp_size != 1:
                raise ValueError("FlexCache Cubic currently supports cp_size=1 only.")
            total_steps = int(task.req.params.num_inference_steps)
            patch_size = tuple(spec.patch_size or getattr(DiffusionBackend.active_model, "patch_size", (1, 2, 2)))
            num_layers = int(
                spec.num_layers
                if spec.num_layers is not None
                else getattr(
                    DiffusionBackend.active_model,
                    "num_layers",
                    len(getattr(DiffusionBackend.active_model, "blocks", [])),
                )
            )
            return CubicStrategy(
                task=task,
                cache_ratio=spec.cache_ratio,
                warmup_steps=warmup_steps,
                cooldown_steps=cooldown_steps,
                tau_max=spec.tau_max,
                patch_size=patch_size,
                num_layers=num_layers,
                target_speedup=spec.target_speedup or max(1.0, 1.0 / max(1.0 - spec.cache_ratio, 1e-6)),
                warmup_fraction=spec.warmup_fraction if spec.warmup_fraction is not None else warmup_steps / max(total_steps, 1),
                cooldown_fraction=spec.cooldown_fraction if spec.cooldown_fraction is not None else cooldown_steps / max(total_steps, 1),
                anchor_interval=spec.anchor_interval or spec.tau_max,
                partition_mode=spec.partition_mode,
                min_block_size=spec.min_block_size,
                max_block_size=spec.max_block_size,
                cv_threshold=spec.cv_threshold,
                alpha=spec.alpha,
                beta=spec.beta,
                curvature_contrast_gamma=spec.curvature_contrast_gamma,
            )

        if strategy == "taylorseer":
            from chitu_diffusion.flexcache.strategy.taylorseer import TaylorSeerStrategy

            return TaylorSeerStrategy(
                task=task,
                fresh_threshold=getattr(spec, "fresh_threshold", 5),
                max_order=getattr(spec, "max_order", 1),
                first_enhance=getattr(spec, "first_enhance", 1),
                warmup_steps=warmup_steps,
                cooldown_steps=cooldown_steps,
            )

        raise ValueError(f"Unknown flexcache strategy '{strategy}'.")

    def _build_ditango_planner(self, task: DiffusionTask, spec: FlexCacheParams):
        if not isinstance(spec, DiTangoParams):
            raise TypeError("DiTango requires DiTangoParams.")
        from chitu_diffusion.ditango.planner import DiTangoPlanner

        return DiTangoPlanner(
            task=task,
            cache_ratio=spec.cache_ratio,
            warmup_steps=spec.warmup,
            cooldown_steps=spec.cooldown,
            tau_max=spec.tau_max,
            curvature_interval_power=spec.curvature_interval_power,
        )

    @staticmethod
    def _flexcache_strategy_name(strategy: str) -> str:
        names = {
            "blockdance": "BlockDance",
            "cubic": "FlexCacheCubic",
            "teacache": "TeaCache",
            "pab": "PAB",
            "ditango": "DiTango",
        }
        return names.get(strategy, strategy)

    def _configure_flexcache_for_task(self, task: DiffusionTask):
        spec = self._resolve_flexcache_spec(task)

        if spec is None:
            if DiffusionBackend.ditango is not None:
                self._clear_ditango_planner()
                logger.info("DiTango disabled for current task; cleared previous planner wrappers.")
            if DiffusionBackend.flexcache is not None and DiffusionBackend.flexcache.strategy is not None:
                self._clear_flexcache_strategy()
                logger.info("Flexcache disabled for current task; cleared previous strategy wrappers.")
            return

        if spec.strategy == "ditango":
            if DiffusionBackend.flexcache is not None and DiffusionBackend.flexcache.strategy is not None:
                self._clear_flexcache_strategy()
            if DiffusionBackend.ditango is not None:
                self._clear_ditango_planner()

            planner = self._build_ditango_planner(task, spec)
            DiffusionBackend.ditango = planner
            planner.wrap_module_with_strategy(DiffusionBackend.active_model)

            resolved_log: Dict[str, Any] = {
                "strategy": spec.strategy,
                "cache_ratio": spec.cache_ratio,
                "warmup": spec.warmup,
                "cooldown": spec.cooldown,
                "tau_max": planner.tau_max,
                "curvature_interval_power": planner.curvature_interval_power,
            }
            logger.info(
                f"{self._flexcache_strategy_name(spec.strategy)}: Successfully wrapped models with resolved params {resolved_log}."
            )
            return

        if DiffusionBackend.flexcache is None:
            logger.warning(
                f"Flexcache strategy '{spec.strategy}' is requested, "
                "but infer.diffusion.enable_flexcache is False. Strategy will be ignored."
            )
            return

        DiffusionBackend.flexcache.reset_compute_stats()

        if DiffusionBackend.ditango is not None:
            self._clear_ditango_planner()

        if DiffusionBackend.flexcache.strategy is not None:
            self._clear_flexcache_strategy()

        cache_strategy = self._build_flexcache_strategy(task, spec)
        DiffusionBackend.flexcache.set_strategy(cache_strategy)
        DiffusionBackend.flexcache.strategy.wrap_module_with_strategy(DiffusionBackend.active_model)

        resolved_log: Dict[str, Any] = {
            "strategy": spec.strategy,
            "warmup": spec.warmup,
            "cooldown": spec.cooldown,
        }
        if spec.strategy == "teacache":
            resolved_log["teacache_thresh"] = cache_strategy.teacache_thresh
        elif spec.strategy == "pab":
            resolved_log["skip_self_range"] = cache_strategy.skip_self_range
            resolved_log["skip_cross_range"] = cache_strategy.skip_cross_range
        elif spec.strategy == "blockdance":
            resolved_log["boundary_block"] = cache_strategy.boundary_block
            resolved_log["group_size"] = cache_strategy.group_size
            resolved_log["start_step"] = cache_strategy.start_step
            resolved_log["end_step"] = cache_strategy.end_step
        elif spec.strategy == "cubic":
            resolved_log["cache_ratio"] = spec.cache_ratio
            resolved_log["target_speedup"] = cache_strategy.config.target_speedup
            resolved_log["anchor_interval"] = cache_strategy.config.anchor_interval
            resolved_log["partition_mode"] = cache_strategy.config.partition_mode
        elif spec.strategy == "taylorseer":
            resolved_log["fresh_threshold"] = cache_strategy.fresh_threshold
            resolved_log["max_order"] = cache_strategy.max_order
            resolved_log["first_enhance"] = cache_strategy.first_enhance
        logger.info(
            f"{self._flexcache_strategy_name(spec.strategy)}: Successfully wrapped models with resolved params {resolved_log}."
        )
        
    def _pre_denoising(self, task: DiffusionTask):
        """
        Before denoising loop, prepare latents, timesteps and solver for one task.
        TODO: Control Devices
        """

        device = torch.cuda.current_device()

        if DiffusionBackend.args.models.name in ["FLUX.1-dev"]:
            from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

            width, height = task.req.params.size
            seed_g = torch.Generator(device=device)
            seed_g.manual_seed(task.req.params.seed)
            latents, latent_image_ids = prepare_flux1_latents(
                batch_size=1,
                num_channels_latents=DiffusionBackend.args.models.transformer.in_channels // 4,
                height=height,
                width=width,
                vae_scale_factor=16,
                dtype=task.buffer.text_embeddings.dtype,
                device=device,
                generator=seed_g,
            )
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                DiffusionBackend.args.models.ckpt_dir,
                subfolder="scheduler",
            )
            sigmas = flowmatch_sigmas(task.req.params.num_inference_steps)
            mu = calculate_flux1_shift(
                latents.shape[1],
                scheduler.config.base_image_seq_len,
                scheduler.config.max_image_seq_len,
                scheduler.config.base_shift,
                scheduler.config.max_shift,
            )
            timesteps, _ = retrieve_flowmatch_timesteps(
                scheduler,
                task.req.params.num_inference_steps,
                device,
                sigmas=sigmas,
                mu=mu,
            )
            guidance = DiffusionBackend.args.models.sampler.guidance_scale[0]
            guidance_vec = torch.full((latents.shape[0],), guidance, device=device, dtype=torch.float32)

            task.buffer.seed_g = seed_g
            task.buffer.sampler = scheduler
            task.buffer.latents = latents
            task.buffer.latent_image_ids = latent_image_ids
            task.buffer.timesteps = timesteps
            task.buffer.guidance_vec = guidance_vec
            task.buffer.image_size = (width, height)

            DiffusionBackend.switch_active_model(flush=True)
            logger.info(
                "[Pre Denoise FLUX1] latents=%s img_ids=%s steps=%s guidance=%s",
                tuple(latents.shape),
                tuple(latent_image_ids.shape),
                len(timesteps),
                guidance,
            )
            self._configure_flexcache_for_task(task)
            return

        if DiffusionBackend.args.models.name in ["FLUX.2-klein-4B"]:
            from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

            width, height = task.req.params.size
            seed_g = torch.Generator(device=device)
            seed_g.manual_seed(task.req.params.seed)
            latents, latent_ids = prepare_flux2_latents(
                batch_size=1,
                num_latents_channels=DiffusionBackend.args.models.transformer.in_channels // 4,
                height=height,
                width=width,
                vae_scale_factor=int(getattr(DiffusionBackend.args.models.vae, "scale_factor", 8)),
                dtype=task.buffer.text_embeddings.dtype,
                device=device,
                generator=seed_g,
            )
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                DiffusionBackend.args.models.ckpt_dir,
                subfolder="scheduler",
            )
            sigmas = flowmatch_sigmas(task.req.params.num_inference_steps)
            if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
                sigmas = None
            mu = compute_flux2_empirical_mu(
                image_seq_len=latents.shape[1],
                num_steps=task.req.params.num_inference_steps,
            )
            timesteps, _ = retrieve_flowmatch_timesteps(
                scheduler,
                task.req.params.num_inference_steps,
                device,
                sigmas=sigmas,
                mu=mu,
            )
            scheduler.set_begin_index(0)

            task.buffer.seed_g = seed_g
            task.buffer.sampler = scheduler
            task.buffer.latents = latents
            task.buffer.latent_image_ids = latent_ids
            task.buffer.timesteps = timesteps
            task.buffer.image_size = (width, height)

            DiffusionBackend.switch_active_model(flush=True)
            logger.info(
                "[Pre Denoise FLUX2] latents=%s latent_ids=%s steps=%s mu=%.4f",
                tuple(latents.shape),
                tuple(latent_ids.shape),
                len(timesteps),
                mu,
            )
            return

        # Prepare latents on rank 0, only data in rank 0 would be used.
        if task.buffer.latents is None:
            F = task.req.params.frame_num
            size = task.req.params.size
            vae_stride = DiffusionBackend.args.models.vae.stride
            target_shape = (DiffusionBackend.vae.model.z_dim, (F - 1) // vae_stride[0] + 1,
                            size[1] // vae_stride[1],
                            size[0] // vae_stride[2])

            seed_g = torch.Generator(device=device)
            seed_g.manual_seed(task.req.params.seed)

            latents = torch.randn(
                        target_shape[0],
                        target_shape[1],
                        target_shape[2],
                        target_shape[3],
                        dtype=torch.float32,
                        device=device,
                        generator=seed_g
                    )
            
            patch_size = DiffusionBackend.args.models.transformer.patch_size
            seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                        (patch_size[1] * patch_size[2]) *
                        target_shape[1] / self.cp_size) * self.cp_size
    
        # Prepare Solver and Timestep on main rank
        if task.req.params.sample_solver == 'unipc': # 求解器
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=DiffusionBackend.args.models.sampler.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                task.req.params.num_inference_steps, 
                device=device,
                shift=DiffusionBackend.args.models.sampler.sample_shift
                )
            timesteps = sample_scheduler.timesteps
        elif task.req.params.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=DiffusionBackend.args.models.sampler.num_train_timesteps, 
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(task.req.params.num_inference_steps, DiffusionBackend.args.models.sampler.sample_shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        
        task.buffer.sampler = sample_scheduler
        task.buffer.seed_g = seed_g
        task.buffer.latents = latents
        task.buffer.timesteps = timesteps
        task.buffer.seq_len = seq_len
        
        DiffusionBackend.switch_active_model(flush=True)

        self._configure_flexcache_for_task(task)

    def _update_task_stage_and_buffer(self, task: DiffusionTask, tokens: torch.Tensor):

        is_main_process = dist.get_rank() == 0
        has_img = task.req.init_image is not None
        
        if task.status == DiffusionTaskStatus.Running:
            task.status = DiffusionTaskStatus.Pending
        elif is_main_process:
            logger.warning(f"Task {task.task_id} - Status: {task.status}")
        
        # 处理Text Encode阶段
        if task.task_type == DiffusionTaskType.TextEncode:                
            if self.cfg_size == 1:
                if task.buffer.text_embeddings is None:
                    # 首次text encode (正向prompt)
                    task.buffer.text_embeddings = tokens
                    if DiffusionBackend.do_cfg:
                        # CFG模式需要第二次encode negative prompt
                        return True
                else:
                    # 第二次text encode（仅CFG模式）
                    if DiffusionBackend.do_cfg:
                        task.buffer.negative_embeddings = tokens

            elif self.cfg_size == 2:
                if get_cfg_group().rank_in_group == 0:
                    task.buffer.text_embeddings = tokens
                else:
                    task.buffer.negative_embeddings = tokens
                
            # Text Encode完成，转换到下一阶段
            self._emit_stage_end(DiffusionTaskType.TextEncode, task.task_id)
            if has_img:
                task.task_type = DiffusionTaskType.VAEEncode 
            else:
                task.task_type = DiffusionTaskType.Denoise
                self._pre_denoising(task)
            
            # 如果转换到Denoise阶段，初始化denoise相关参数
            if task.task_type == DiffusionTaskType.Denoise:
                task.buffer.current_step = 0
                
            if is_main_process:
                logger.debug(f"Task {task.task_id} transitioned to {task.task_type}")
            return True
        
        # 处理VAE Encode阶段
        elif task.task_type == DiffusionTaskType.VAEEncode:
            task.buffer.latents = tokens  # 保存编码后的latents
            
            # 转换到Denoise阶段
            self._emit_stage_end(DiffusionTaskType.VAEEncode, task.task_id)
            task.task_type = DiffusionTaskType.Denoise
            
            if is_main_process:
                logger.debug(f"Task {task.task_id} transitioned to {task.task_type}")
            return True
        
        # 处理Denoise阶段（关键：需要多次执行）
        elif task.task_type == DiffusionTaskType.Denoise:
            # 更新当前去噪后的latents
            task.buffer.latents = tokens
            task.buffer.current_step += 1
            current_step = task.buffer.current_step
            total_steps = task.req.params.num_inference_steps
            timestep = task.buffer.timesteps[current_step - 1] if task.buffer.timesteps is not None else None

            log_progress(
                logger,
                stage_name=DiffusionTaskType.Denoise.name,
                task_id=task.task_id,
                step=current_step,
                total=total_steps,
                interval=self.denoise_progress_interval,
                timestep=timestep,
            )
            

            # 检查是否完成所有denoise步骤
            if task.buffer.current_step >= task.req.params.num_inference_steps:
                # 所有denoise步骤完成，转换到VAE Decode
                self._emit_stage_end(DiffusionTaskType.Denoise, task.task_id)
                task.task_type = DiffusionTaskType.VAEDecode
                if is_main_process:
                    logger.debug(f"Task {task.task_id} completed denoising, transitioned to {task.task_type}")
                return True
            elif DiffusionBackend.boundary is not None and task.buffer.current_step >= DiffusionBackend.boundary * task.req.params.num_inference_steps:
                # 检查是否需要切换模型
                DiffusionBackend.switch_active_model(flush=False)
            else:
                # 还需要继续denoise，保持当前阶段但状态改为Pending等待下次调度
                if is_main_process:
                    logger.debug(f"Task {task.task_id} continuing denoise")
                return True
        
        # 处理VAE Decode阶段（最终阶段）
        elif task.task_type == DiffusionTaskType.VAEDecode:               
            task.buffer.generated_image = tokens  # 保存最终生成的图像
            self._post_vae_decode(task, tokens)
            self._emit_stage_end(DiffusionTaskType.VAEDecode, task.task_id)
            self._clear_ditango_planner()
            self._clear_flexcache_strategy()
            if is_main_process:
                logger.debug(f"Task {task.task_id} completed all stages")
            task.status = DiffusionTaskStatus.Completed
            self.current_task = None
            self._last_logged_stage.pop(task.task_id, None)
            if self.enable_stage_perf:
                stale_keys = [
                    key for key in self._stage_start_time.keys() if key[0] == task.task_id
                ]
                for key in stale_keys:
                    self._stage_start_time.pop(key, None)
            return False  # 完成所有阶段，不需要继续调度
        
        # 未知阶段
        else:
            if is_main_process:
                logger.error(f"Unknown task type: {task.task_type}")
            task.status = DiffusionTaskStatus.Failed
            return False
        
    
