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
from chitu_core.global_vars import get_global_args, get_slot_handle
from chitu_core.logging_utils import log_stage, log_progress, log_result, log_perf, should_log_info_on_rank
from chitu_diffusion.backend import BackendState, CFGType, DiffusionBackend
from chitu_diffusion.task import DiffusionTask, DiffusionTaskType, DiffusionTaskPool, DiffusionTaskStatus, FlexCacheParams

from chitu_core.distributed.parallel_state import (
    get_cfg_group,
    get_cp_group,
    get_fpp_group,
    get_up_group,
    get_world_group
)
from chitu_diffusion.modules.samplers.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from chitu_diffusion.modules.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from chitu_diffusion.utils.wan_utils import cache_video
from chitu_diffusion.flex_cache.strategy.FPPCache import FPPCache
from chitu_diffusion.flex_cache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.utils.shared_utils import SequencePadder, split_latent
from chitu_diffusion.bench import Timer, MagLogger
from chitu_diffusion.utils.output_naming import build_video_name_from_task
from chitu_diffusion.utils.flux_utils import (
    batched_prc_img,
    batched_prc_txt,
    get_schedule,
    scatter_ids,
    save_image_as_png,
)
from chitu_core.models.diffusion.model_wan import WanModel


logger = getLogger(__name__)


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
        self.fpp_size = args.infer.diffusion.fpp_size
        self.cfg_size = get_cfg_group().group_size
        self.fpp_size = get_fpp_group().group_size
        if self.cp_size > 1:
            self.cp_dispatcher = ContextParallelDispatcher()
            self.cp_dispatcher.wrap_model_compute_with_cp()
        if self.cfg_size == 2:
            self.cfg_dispatcher = CfgDispatcher()

        # Ensure FIFO
        self.current_task = None # 通过这个储存当前任务的中间状态
        self._last_logged_stage = {}
        self.denoise_progress_interval = max(1, int(os.getenv("CHITU_PROGRESS_INTERVAL", "5")))
        self.enable_stage_perf = bool(getattr(args.output, "enable_timer_dump", False))
        self._stage_start_time = {}

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
            debug = True
            if self.fpp_size > 1:
                self.fpp_denoise_steps(task)
                return

            elif debug:
                self.debug_denoise_steps_patch(task)
                return
            else:
                out = self.denoise_step(task)

        else:
            raise NotImplementedError  
        
        self._update_task_stage_and_buffer(task, out)

        return out


    def debug_denoise_steps_patch(self, task: DiffusionTask):
        
        warm_up_steps = task.req.params.flexcache_params.warmup 
        cool_down_steps = task.req.params.flexcache_params.cooldown
        patch_steps = task.req.params.num_inference_steps - warm_up_steps - cool_down_steps

        patch_steps = 10
        
        patch_num = 21 
        print(f"[INFO] patch_num:{patch_num}")
        for step_idx in range(warm_up_steps):
            print(f"Rank {self.rank} starting warmup step {step_idx+1}/{warm_up_steps}", flush=True)
            out = self.denoise_step(task, save_cache= step_idx == warm_up_steps - 1)
            task.buffer.latents = out
            task.buffer.current_step += 1


        for step_idx in range(patch_steps):
            # import pdb; pdb.set_trace()
            # use_std_path = os.getenv("CHITU_DEBUG_FPP", "0") == "1"
            print(f"Rank {self.rank} starting patch step {step_idx+1}/{patch_steps}", flush=True)

            self.patch_denoise_one_step(task, patch_num, no_latents_cache=False)
            

        for step_idx in range(cool_down_steps):
            print(f"Rank {self.rank} starting cooldown step {step_idx+1}/{cool_down_steps}", flush=True)
            out = self.denoise_step(task)
            task.buffer.latents = out
            task.buffer.current_step += 1

        DiffusionBackend.active_model.cache_manager.strategy.reset_state() # clear cache to free memory after FPP 

        task.task_type = DiffusionTaskType.VAEDecode
        task.status = DiffusionTaskStatus.Pending
        return  

    @Timer.get_timer("fpp_denoise_steps")
    def fpp_denoise_steps(self, task: DiffusionTask):
        # first_steps for warmup 
        warm_up_steps = task.req.params.flexcache_params.warmup 
        cool_down_steps = task.req.params.flexcache_params.cooldown

        async_steps1 = task.req.params.num_inference_steps - warm_up_steps - cool_down_steps
        print(f"warmup steps: {warm_up_steps}, cooldown steps: {cool_down_steps},  async steps: {async_steps1}", flush=True)
        model = DiffusionBackend.active_model

        if DiffusionBackend.boundary is not None:
            async_steps1 = int(DiffusionBackend.boundary * task.req.params.num_inference_steps) - warm_up_steps 

            second_syncpipe_steps = int( (1-DiffusionBackend.boundary) * task.req.params.num_inference_steps) - warm_up_steps

        else:
            async_steps1 = task.req.params.num_inference_steps - warm_up_steps - cool_down_steps
        
        for step_idx in range(warm_up_steps):
            do_cache = step_idx == warm_up_steps - 1 # only cache the last warmup step
            print(f"Rank {self.rank} starting warmup syncpipe step {step_idx+1}/{warm_up_steps}", flush=True)
            if async_steps1 == 0 and step_idx > 3 and step_idx < 8:
                with model.kv_capture_scope(noise_step=task.buffer.current_step, patch_idx=-1):
                    self.denoise_sync_pipeline(task, do_cache)
            else:
                self.denoise_sync_pipeline(task, do_cache)


        patch_num = 21
        fpp_size = get_fpp_group().group_size
        patch_offset_inteval = patch_num - fpp_size
        full_inteval = 50

        patch_travel_order = list(range(patch_num))

        for step_idx in range(async_steps1):
            is_first_step = step_idx % full_inteval == 0
            is_last_step = (step_idx % full_inteval == full_inteval - 2) or (step_idx == async_steps1 - 1)
            patch_offset = (patch_offset_inteval * step_idx) % patch_num
            if not ( (step_idx % full_inteval) == full_inteval - 1):
                self.fpp_denoise_one_step(task, is_first_step=is_first_step, is_last_step=is_last_step, patch_num = patch_num, patch_offset=patch_offset, patch_travel_order=patch_travel_order)
                patch_travel_order = patch_travel_order[patch_offset_inteval:] + patch_travel_order[:patch_offset_inteval][::-1] # rotate the patch travel order to ensure all patches are warmed up equally

            else:
                self.denoise_sync_pipeline(task, save_cache=True)

        model.cache_manager.strategy.reset_state() # clear cache to free memory after FPP

        
        for step_idx in range(cool_down_steps):
            print(f"Rank {self.rank} starting cooldown syncpipe step {step_idx+1}/{cool_down_steps}", flush=True)
            self.denoise_sync_pipeline(task)


        if DiffusionBackend.boundary is not None:
            # after boundary, run another syncpipe to warm up the second model
            DiffusionBackend.switch_active_model(flush=False)
            DiffusionBackend.flexcache.cache.clear() # clear cache to free memory after switch
            for step_idx in range(warm_up_steps):

                do_cache = step_idx == warm_up_steps - 1 # only cache the last warmup step
                self.denoise_sync_pipeline(task, do_cache)
            
            for step_idx in range(second_syncpipe_steps):
                is_first_step = (step_idx == 0)
                self.fpp_denoise_one_step(task, is_first_step=is_first_step)
        

        task.task_type = DiffusionTaskType.VAEDecode
        task.status = DiffusionTaskStatus.Pending
        return 
        
    @Timer.get_timer("fpp_sync_pipeline_step")
    @amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def denoise_sync_pipeline(self, task: DiffusionTask, save_cache: bool = False):
        latent_model_input = task.buffer.latents
        assert task.buffer.latents is not None and task.buffer.timesteps is not None


        timestep = task.buffer.timesteps[task.buffer.current_step]

        assert DiffusionBackend.guidance_scale > 0 and self.cfg_size == 1 and self.fpp_size > 1

        with Timer.get_timer("sync_pipe_forward_compute"):
            model : WanModel = DiffusionBackend.active_model
            DiffusionBackend.cfg_type = CFGType.POS
            noise_pred_cond = model.sync_pipe_forward(latent_model_input, t=timestep, context=task.buffer.text_embeddings, seq_len=task.buffer.seq_len,save_cache=save_cache)
            DiffusionBackend.cfg_type = CFGType.NEG
            noise_pred_uncond = model.sync_pipe_forward(latent_model_input, t=timestep, context=task.buffer.negative_embeddings, seq_len=task.buffer.seq_len,save_cache=save_cache)

        fpp_group = get_fpp_group()
        if fpp_group.is_last_rank:
            noise_pred = noise_pred_uncond + DiffusionBackend.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            sampled_latents = task.buffer.sampler.step(
                noise_pred.unsqueeze(0),
                timestep,
                task.buffer.latents.unsqueeze(0),
                return_dict=False,
                generator=task.buffer.seed_g
            )[0].squeeze(0) 

            fpp_group.p2p_isend(tensor=sampled_latents, dst=fpp_group.next_rank,tag=4)
            fpp_group.p2p_commit()
            fpp_group.p2p_wait()
            
            task.buffer.latents = sampled_latents 
        
        elif fpp_group.is_first_rank:

            sampled_latents = fpp_group.p2p_irecv(size=task.buffer.latents.shape,dtype=torch.float32, src=fpp_group.prev_rank, tag=4)
            fpp_group.p2p_commit()
            fpp_group.p2p_wait()

            assert sampled_latents.shape == task.buffer.latents.shape, f"Expected sampled latents shape {task.buffer.latents.shape}, but got {sampled_latents.shape}"


            task.buffer.latents = sampled_latents
        
        task.buffer.current_step += 1


    def fpp_async_tail_collection(self, task: DiffusionTask, fpp_size: Optional[int] = None):
        # collect the last step's latents from the first rank to ensure the tail syncpipe can start with updated latents
        # only run at_first_rank
        fpp_group = get_fpp_group()
        fpp_size = fpp_size if fpp_size is not None else fpp_group.fpp_size


        for patch_idx in range(fpp_size):
            recv_latent = fpp_group.p2p_irecv(task.buffer.latents.shape, torch.float32, src=fpp_group.prev_rank, tag=3)
            fpp_group.p2p_commit()
            fpp_group.p2p_wait()

        task.buffer.latents = recv_latent




    @Timer.get_timer("fpp_async_pipeline_step")
    @amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    @torch.no_grad()
    def fpp_denoise_one_step(self, task: DiffusionTask, is_first_step: bool, is_last_step: bool, patch_num: Optional[int] = None, patch_offset: int = 0, patch_travel_order: Optional[list] = None):

        fpp_group = get_fpp_group()

        dtype = task.buffer.latents.dtype
        fpp_size = get_fpp_group().group_size
        patch_num = fpp_size if patch_num is None else patch_num
        patch_seq_len = task.buffer.seq_len // patch_num

        cur_step = task.buffer.current_step
        model : WanModel = DiffusionBackend.active_model
        cache_strategy : FPPCache = DiffusionBackend.flexcache.strategy


        time_embedding = model._cal_time_embeddings(task.buffer.timesteps[cur_step])
        time_proj = model._cal_timeproj(task.buffer.timesteps[cur_step])
        context_embedding = model._cal_context_embeddings(task.buffer.text_embeddings, clip_fea=None)
        negative_context_embedding = model._cal_context_embeddings(task.buffer.negative_embeddings, clip_fea=None)

        grid_sizes = task.buffer.grid_sizes
        token_patch_shape = torch.Size([1,patch_seq_len, model.dim])


        for patch_idx in range(patch_num):
            patch_idx_cal = (patch_idx + patch_offset ) % patch_num
            if patch_travel_order is not None:
                patch_idx_cal = patch_travel_order[patch_idx]

            
            if fpp_group.is_first_rank:
                send_recv_flag = not (is_first_step and patch_idx < fpp_size - 1)
            else:
                send_recv_flag = not (  (is_last_step and patch_idx == patch_num - 1) )
            
             # only the last step's first patch and the first step's last patch can start without waiting for updated latents from last step, other patches need to wait for the data dependency to be resolved

            # pad for sending and forwarding

            if fpp_group.is_first_rank:
                
                if not (is_first_step and patch_idx < fpp_size):
                    
                    sampled_latents = fpp_group.p2p_irecv(task.buffer.latents.shape, torch.float32, src=fpp_group.prev_rank, tag=3)
                    fpp_group.p2p_commit()
                    fpp_group.p2p_wait()
                    task.buffer.latents = sampled_latents

                
    
                hidden_states : torch.Tensor = model._cal_patch_embedding(task.buffer.latents, seq_len=task.buffer.seq_len)

                # assert hidden_states.shape == torch.Size([1, task.buffer.seq_len, model.dim]), f"Expected hidden states shape {[1, task.buffer.seq_len, model.dim]}, but got {hidden_states.shape}"

                hidden_states_patch = SequencePadder.split_sequence_padding(hidden_states, patch_num, split_dim=1, name='fpp')[patch_idx_cal] # pad and split for fpp

                # assert hidden_states_patch.shape == token_patch_shape, f"Expected hidden states patch shape {token_patch_shape}, but got {hidden_states_patch.shape}"


                hidden_states_cond_patch = hidden_states_patch.clone()
                hidden_states_uncond_patch = hidden_states_patch.clone()

            else: # first rank does pre_dit
                hidden_states_cond_patch = fpp_group.p2p_irecv(token_patch_shape, dtype=torch.float32, src=fpp_group.prev_rank, tag=1)
                hidden_states_uncond_patch = fpp_group.p2p_irecv(token_patch_shape, dtype=torch.float32, src=fpp_group.prev_rank, tag=2)
                fpp_group.p2p_commit()
                fpp_group.p2p_wait()


            position_idx = patch_idx_cal * patch_seq_len
            position_idx_end = min(position_idx + patch_seq_len, task.buffer.unpad_seq_len)

            # print(f"Rank {fpp_group.global_rank} processing patch {patch_idx_cal}, position idx range [{position_idx}:{position_idx_end}]", flush=True)
            
            with Timer.get_timer(f"fpp_async_patch_compute_step_{patch_idx_cal}"):
                DiffusionBackend.cfg_type = CFGType.POS
                with model.kv_capture_scope(noise_step=cur_step, patch_idx=patch_idx_cal):
                    hidden_states_cond_patch = model.model_compute(
                        hidden_states_cond_patch,
                        time_proj,
                        context_embedding,
                        grid_sizes,
                        save_cache=True,
                        position_idx=position_idx,
                    )
                DiffusionBackend.cfg_type = CFGType.NEG
                hidden_states_uncond_patch = model.model_compute(
                        hidden_states_uncond_patch,
                        time_proj,
                        negative_context_embedding,
                        grid_sizes,
                        save_cache=True,
                        position_idx=position_idx,
                    )


            # assert hidden_states_cond_patch.dtype == torch.float32
            # assert hidden_states_uncond_patch.shape == token_patch_shape, f"Expected hidden states shape {token_patch_shape}, but got {hidden_states_uncond_patch.shape}"
            if not fpp_group.is_last_rank:
                fpp_group.p2p_isend(hidden_states_cond_patch, fpp_group.next_rank, tag=1)
                fpp_group.p2p_isend(hidden_states_uncond_patch, fpp_group.next_rank, tag=2)

            else:
                hidden_states_cond = cache_strategy.update_stale_tokens_patch(hidden_states_cond_patch, (position_idx, position_idx_end), is_pos=True)
                hidden_states_uncond = cache_strategy.update_stale_tokens_patch(hidden_states_uncond_patch, (position_idx, position_idx_end), is_pos=False)

                model: WanModel = DiffusionBackend.active_model
                latents_cond = model._post_dit(hidden_states_cond, time_embedding, grid_sizes)[0].to(torch.float32)
                latents_uncond = model._post_dit(hidden_states_uncond, time_embedding, grid_sizes)[0].to(torch.float32)

                noise_pred = latents_uncond + DiffusionBackend.guidance_scale * (latents_cond - latents_uncond)

                print(f"[Sampler] step:{cur_step}, patch_idx:{patch_idx_cal}", flush=True)

                update_step = (patch_idx == patch_num - 1)

                sampled_latents = task.buffer.sampler.step(
                    noise_pred.unsqueeze(0),
                    task.buffer.timesteps[cur_step],
                    task.buffer.latents.unsqueeze(0),
                    return_dict=False,
                    generator=task.buffer.seed_g,
                    update_step= update_step # only update sampler at the first patch to avoid redundant update, this is important for samplers with strong data dependency like DPMPP2MSampler
                )[0].squeeze(0)

                fpp_group.p2p_isend(sampled_latents, fpp_group.next_rank, tag=3)


       
            if not send_recv_flag:
                fpp_group.p2p_commit()
                fpp_group.p2p_wait()
            
    
        task.buffer.current_step += 1
        if fpp_group.is_last_rank:
            task.buffer.latents = sampled_latents

        if is_last_step and fpp_group.is_first_rank:
            self.fpp_async_tail_collection(task, fpp_size) # collect the last step's latents from the first rank to ensure the tail syncpipe can start with updated latents from last step
        # update the latents at the last rank, this buffer works only as the input for scheduler's original latent, the timing of update is subtle but carefully considered.
        # requires warmup last step stores the sampled latents to last rank's buffer

    @amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def patch_denoise_one_step(self, task: DiffusionTask, patch_num: int, no_latents_cache: bool = True):
        # for debugging fpp_denoise_one_step, run patch forward on single device.
        model: WanModel = DiffusionBackend.active_model
        cache_strategy : FlexCacheStrategy = DiffusionBackend.flexcache.strategy
        patch_seq_len = task.buffer.seq_len // patch_num
        device_num = patch_num



        time_emb = model._cal_time_embeddings(task.buffer.timesteps[task.buffer.current_step])
        time_proj = model._cal_timeproj(task.buffer.timesteps[task.buffer.current_step])
        ctx_emb = model._cal_context_embeddings(task.buffer.text_embeddings, clip_fea=None)
        neg_emb = model._cal_context_embeddings(task.buffer.negative_embeddings, clip_fea=None)
        grid_sizes = task.buffer.grid_sizes

        # hidden_states_uncond_list = [hidden_states.clone() for _ in range(patch_num)]

        # import pdb; pdb.set_trace()

        original_latents = task.buffer.latents.clone()
        # import pdb; pdb.set_trace()
        for patch_idx in range(patch_num):
            hidden_states = model._cal_patch_embedding(task.buffer.latents, seq_len=task.buffer.seq_len)
            hidden_states_list = SequencePadder.split_sequence_padding(hidden_states, patch_num, split_dim=1, name='fpp')

            hidden_states_patch = hidden_states_list[patch_idx] 

            hidden_states_cond_patch = hidden_states_patch.clone()
            hidden_states_uncond_patch = hidden_states_patch.clone()

            position_idx = patch_idx * patch_seq_len
            position_idx_end = min(position_idx + patch_seq_len, task.buffer.unpad_seq_len)

            DiffusionBackend.cfg_type = CFGType.POS
            hidden_states_cond_patch = model.model_compute(
                hidden_states_cond_patch,
                time_proj,
                ctx_emb,
                grid_sizes,
                save_cache=True,
                position_idx=position_idx,
            )
            DiffusionBackend.cfg_type = CFGType.NEG
            hidden_states_uncond_patch = model.model_compute(
                    hidden_states_uncond_patch,
                    time_proj,
                    neg_emb,
                    grid_sizes,
                    save_cache=True,
                    position_idx=position_idx,
                )

            unpad_boundary = (position_idx, position_idx_end)


            hidden_states_cond = cache_strategy.update_stale_tokens_patch(hidden_states_cond_patch, unpad_boundary, True)
            hidden_states_uncond = cache_strategy.update_stale_tokens_patch(hidden_states_uncond_patch, unpad_boundary, False)


            latents_cond = model._post_dit(hidden_states_cond, time_emb, grid_sizes)[0].to(torch.float32)
            latents_uncond = model._post_dit(hidden_states_uncond, time_emb, grid_sizes)[0].to(torch.float32)
            noise_pred = latents_uncond + DiffusionBackend.guidance_scale * (latents_cond - latents_uncond)

            sampled_latents = task.buffer.sampler.step(
                noise_pred.unsqueeze(0),
                task.buffer.timesteps[task.buffer.current_step],
                original_latents.unsqueeze(0),
                return_dict=False,
                generator=task.buffer.seed_g,
                update_step= patch_idx == (patch_num - 1) # only update the sampler at the last patch to ensure the data dependency is resolved for the next step, this is important for samplers with strong data dependency like DPMPP2MSampler
            )[0].squeeze(0)
            task.buffer.latents = sampled_latents # update latents for the next patch

        task.buffer.latents = sampled_latents
        task.buffer.current_step += 1


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

        if DiffusionBackend.args.models.name in ["FLUX.2-klein-4B"]:
            with device_scope(DiffusionBackend.text_encoder.model):
                out = DiffusionBackend.text_encoder([payload])
            ctx, ctx_ids = batched_prc_txt(out.to(torch.bfloat16))
            task.buffer.ctx_ids = ctx_ids
            logger.info(f"[text_encode_step] FLUX2 ctx shape: {ctx.shape}, ctx_ids shape: {ctx_ids.shape}")
            return ctx
        
        with device_scope(DiffusionBackend.text_encoder.model):        
            out = DiffusionBackend.text_encoder(payload, torch.cuda.current_device())
        
        logger.info(f"rank:{self.rank} [text_encode_step] context shape: {out.shape}")
        return out
    
    def vae_encode_step(self, task: DiffusionTask):
        pass
    
    @Timer.get_timer("denoise")
    @amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def denoise_step(self, task: DiffusionTask, save_cache: bool = False):
        assert task.buffer.latents is not None and task.buffer.timesteps is not None

        if DiffusionBackend.args.models.name in ["FLUX.2-klein-4B"]:
            """Single Euler denoising step for FLUX2 (guidance-distilled, no CFG)."""
            x = task.buffer.latents
            t_curr = task.buffer.timesteps[task.buffer.current_step]
            t_prev = task.buffer.timesteps[task.buffer.current_step + 1]

            t_vec = torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=x.device)

            pred = DiffusionBackend.active_model(
                x=x,
                x_ids=task.buffer.x_ids,
                timesteps=t_vec,
                ctx=task.buffer.text_embeddings,
                ctx_ids=task.buffer.ctx_ids,
                guidance=task.buffer.guidance_vec,
            )

            sampled_latents = x + (t_prev - t_curr) * pred
            return sampled_latents

        latent_model_input = task.buffer.latents
        timestep = task.buffer.timesteps[task.buffer.current_step]
        print(f"RANK:{self.rank}, Denoise step {task.buffer.current_step}/{len(task.buffer.timesteps)}, timestep: {timestep}, latents shape: {latent_model_input.shape}")
        if DiffusionBackend.guidance_scale > 0: # Wan的cfg是做两次
            if self.cfg_size == 2:
                if get_cfg_group().rank_in_group == 0:
                    context = task.buffer.text_embeddings
                    DiffusionBackend.cfg_type = CFGType.POS
                else:
                    context = task.buffer.negative_embeddings
                    DiffusionBackend.cfg_type = CFGType.NEG

                cfg_partial_noise_pred = DiffusionBackend.active_model(
                    latent_model_input,
                    t=timestep,
                    context=context,
                    seq_len=task.buffer.seq_len
                )
                noise_pred_cond, noise_pred_uncond = self.cfg_dispatcher.all_gather_cfg_noise_preds(cfg_partial_noise_pred)
            else:
                # import pdb; pdb.set_trace()
                with Timer.get_timer("cfg_full_model_forward"):
                    DiffusionBackend.cfg_type = CFGType.POS
                    noise_pred_cond = DiffusionBackend.active_model(
                        latent_model_input,
                        t=timestep,
                        context=task.buffer.text_embeddings,
                        seq_len=task.buffer.seq_len,
                        save_cache=save_cache,
                    )
                    DiffusionBackend.cfg_type = CFGType.NEG
                    noise_pred_uncond = DiffusionBackend.active_model(
                        latent_model_input,
                        t=timestep,
                        context=task.buffer.negative_embeddings,
                        seq_len=task.buffer.seq_len,
                        save_cache=save_cache,
                    )

            noise_pred = noise_pred_uncond + \
                DiffusionBackend.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            DiffusionBackend.cfg_type = CFGType.POS
            noise_pred = DiffusionBackend.active_model(
                latent_model_input,
                t=timestep,
                context=task.buffer.text_embeddings,
                seq_len=task.buffer.seq_len
            )

        sampled_latents = task.buffer.sampler.step(
            noise_pred.unsqueeze(0),
            timestep,
            task.buffer.latents.unsqueeze(0),
            return_dict=False,
            generator=task.buffer.seed_g
        )[0].squeeze(0)

        return sampled_latents

        
    @Timer.get_timer("VaeDecode")
    def vae_decode_step(self, task: DiffusionTask):
        target_decode_device = 0 # TODO
        logger.debug(f"task_id={task.task_id} entering VAE decode at step={task.buffer.current_step}")
        if torch.distributed.get_rank() == target_decode_device:
            payload = [task.buffer.latents]

            if DiffusionBackend.args.models.name in ["FLUX.2-klein-4B"]:
                x = torch.cat(scatter_ids(task.buffer.latents, task.buffer.x_ids)).squeeze(2).float()
                with device_scope(DiffusionBackend.vae.model):
                    img = DiffusionBackend.vae.decode(x).float()
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
            self._save_video(task, video)
            # save_time_stats
            output_dir = task.req.params.save_dir
            Timer.print_statistics()
            Timer.save_statistics(f"{output_dir}/time_stat_{task.task_id}.csv")
            # Magnitiude experiments
            # MagLogger.save_to_csv(save_dir=f"./experiments/{task.task_id}")

    # TODO: CPU/GPU overlap
    def _save_video(self, task: DiffusionTask, video: torch.Tensor):
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

        # Defensive cleanup in case wrapper metadata remains unexpectedly.
        if model is not None and hasattr(model, '_original_forward'):
            model.model_compute = model._original_forward
            delattr(model, '_original_forward')

        if model is not None and hasattr(model, 'blocks'):
            for block in model.blocks:
                if hasattr(block.self_attn, '_original_forward'):
                    block.self_attn.forward = block.self_attn._original_forward
                    delattr(block.self_attn, '_original_forward')
                if hasattr(block.cross_attn, '_original_forward'):
                    block.cross_attn.forward = block.cross_attn._original_forward
                    delattr(block.cross_attn, '_original_forward')

        manager.strategy = None
        manager.cache.clear()

    @staticmethod
    def _teacache_threshold_from_ratio(cache_ratio: float, model_name: str) -> float:
        # 0 -> quality first (smaller thresh), 1 -> speed first (larger thresh)
        model_name = (model_name or "").lower()
        if "14b" in model_name:
            min_thresh, max_thresh = 0.04, 0.20
        else:
            min_thresh, max_thresh = 0.08, 0.35
        return min_thresh + cache_ratio * (max_thresh - min_thresh)

    @staticmethod
    def _pab_skip_self_from_ratio(cache_ratio: float) -> int:
        # 0 -> quality first (recompute frequently), 1 -> speed first (reuse aggressively)
        return int(round(cache_ratio * 10))

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
        if spec.warmup + spec.cooldown > total_steps:
            raise ValueError(
                "Invalid flexcache warmup/cooldown: "
                f"warmup({spec.warmup}) + cooldown({spec.cooldown}) must be <= num_inference_steps({total_steps})."
            )
        return spec

    def _build_flexcache_strategy(self, task: DiffusionTask, spec: FlexCacheParams):
        strategy = spec.strategy
        cache_ratio = spec.cache_ratio
        warmup_steps = spec.warmup
        cooldown_steps = spec.cooldown

        if strategy == "teacache":
            from chitu_diffusion.flex_cache.strategy.teacache import TeaCacheStrategy

            model_name = DiffusionBackend.args.name
            teacache_thresh = self._teacache_threshold_from_ratio(cache_ratio, model_name)
            return TeaCacheStrategy(
                task=task,
                teacache_thresh=teacache_thresh,
                warmup_steps=warmup_steps,
                cooldown_steps=cooldown_steps,
            )

        if strategy == "pab":
            from chitu_diffusion.flex_cache.strategy.PAB import PABStrategy

            skip_self_range = self._pab_skip_self_from_ratio(cache_ratio)
            skip_cross_range = int(round(skip_self_range * 5 / 3))
            return PABStrategy(
                task=task,
                warmup_steps=warmup_steps,
                cooldown_steps=cooldown_steps,
                skip_self_range=skip_self_range,
                skip_cross_range=skip_cross_range,
            )

        if strategy == "ditango":
            from chitu_diffusion.flex_cache.strategy.ditango.ditango import DiTangoV3Strategy

            ase_threshold = self._ditango_ase_from_ratio(cache_ratio)
            return DiTangoV3Strategy(
                task=task,
                cache_ratio=cache_ratio,
                ase_threshold=ase_threshold,
                warmup_steps=warmup_steps,
                cooldown_steps=cooldown_steps,
            )

        if strategy == "fpp_cache":
            from chitu_diffusion.flex_cache.strategy.FPPCache import FPPCache

            return FPPCache(
                warmup_steps=warmup_steps,
                cooldown_steps=cooldown_steps,
            )

        raise ValueError(f"Unknown flexcache strategy '{strategy}'.")

    @staticmethod
    def _flexcache_strategy_name(strategy: str) -> str:
        names = {
            "teacache": "TeaCache",
            "pab": "PAB",
            "ditango": "DiTango",
        }
        return names.get(strategy, strategy)
        
    def _pre_denoising(self, task: DiffusionTask):
        """
        Before denoising loop, prepare latents, timesteps and solver for one task.
        TODO: Control Devices
        """

        device = torch.cuda.current_device()

        if DiffusionBackend.args.models.name in ["FLUX.2-klein-4B"]:
            """Prepare FLUX2 latents (packed 2D), Euler schedule, and guidance vector."""

            width, height = task.req.params.size
            shape = (1, 128, height // 16, width // 16)

            seed_g = torch.Generator(device=device)
            seed_g.manual_seed(task.req.params.seed)

            randn = torch.randn(shape, generator=seed_g, dtype=torch.bfloat16, device=device)
            x, x_ids = batched_prc_img(randn)

            timesteps = get_schedule(task.req.params.num_inference_steps, x.shape[1])

            guidance = DiffusionBackend.args.models.sampler.guidance_scale[0]
            guidance_vec = torch.full((x.shape[0],), guidance, device=device, dtype=x.dtype)

            task.buffer.seed_g = seed_g
            task.buffer.latents = x
            task.buffer.x_ids = x_ids
            task.buffer.timesteps = timesteps
            task.buffer.guidance_vec = guidance_vec

            DiffusionBackend.switch_active_model(flush=True)
            logger.info(f"[Pre Denoise FLUX2] x={x.shape}, x_ids={x_ids.shape}, "
                        f"timesteps={len(timesteps)} steps, guidance={guidance}")
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

            split_num = self.cp_size * self.fpp_size

            frames_seq_stride = target_shape[2] * target_shape[3] // (patch_size[1] * patch_size[2])

            unpad_seq_len = frames_seq_stride * target_shape[1]
            seq_len = math.ceil( unpad_seq_len / split_num) * split_num
            

            grid_sizes = torch.tensor([
                target_shape[1] // patch_size[0], 
                target_shape[2] // patch_size[1], 
                target_shape[3] // patch_size[2]], device=device, dtype=torch.long).reshape(1,3)

     

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
        task.buffer.unpad_seq_len = unpad_seq_len
        task.buffer.grid_sizes = grid_sizes
        task.buffer.frames_seq_stride = frames_seq_stride




        DiffusionBackend.switch_active_model(flush=True)

        # Configure flexcache strategy for this task.
        spec = self._resolve_flexcache_spec(task)

        if spec is None:
            if DiffusionBackend.flexcache is not None and DiffusionBackend.flexcache.strategy is not None:
                self._clear_flexcache_strategy()
                logger.info("Flexcache disabled for current task; cleared previous strategy wrappers.")
            return

        if DiffusionBackend.flexcache is None:
            logger.warning(
                f"Flexcache strategy '{spec.strategy}' is requested, "
                "but infer.diffusion.enable_flexcache is False. Strategy will be ignored."
            )
            return

        # Always clear any previous strategy before (re)applying current task strategy.
        if DiffusionBackend.flexcache.strategy is not None:
            self._clear_flexcache_strategy()

        cache_strategy = self._build_flexcache_strategy(task, spec)
        DiffusionBackend.flexcache.set_strategy(cache_strategy)

        if isinstance(cache_strategy, FlexCacheStrategy):
            DiffusionBackend.flexcache.strategy.wrap_module_with_strategy(DiffusionBackend.active_model)

        resolved_log: Dict[str, Any] = {
            "strategy": spec.strategy,
            "cache_ratio": spec.cache_ratio,
            "warmup": spec.warmup,
            "cooldown": spec.cooldown,
        }
        if spec.strategy == "teacache":
            resolved_log["teacache_thresh"] = cache_strategy.teacache_thresh
        elif spec.strategy == "pab":
            resolved_log["skip_self_range"] = cache_strategy.skip_self_range
            resolved_log["skip_cross_range"] = cache_strategy.skip_cross_range
        elif spec.strategy == "ditango":
            resolved_log["ase_threshold"] = cache_strategy.ase_threshold

        logger.info(
            f"{self._flexcache_strategy_name(spec.strategy)}: Successfully wrapped models with resolved params {resolved_log}."
        )



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
        
    
