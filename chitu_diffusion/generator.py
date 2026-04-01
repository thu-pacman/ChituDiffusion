# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import math
from models.diffusion.model_wan import WanModel
import torch
import torch.distributed as dist
from typing import Any, Callable, Optional, List, Tuple
from functools import partial
from contextlib import contextmanager
import numpy as np
import torch.amp as amp
from tqdm import tqdm

from logging import getLogger
from chitu_core.global_vars import get_global_args, get_slot_handle, get_timers
from chitu_diffusion.backend import BackendState, CFGType, DiffusionBackend
from chitu_diffusion.task import DiffusionTask, DiffusionTaskType, DiffusionTaskPool, DiffusionTaskStatus
from chitu_diffusion.task import FPPTaskState
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
from chitu_diffusion.modules.samplers.fpp_pipeline_runtime import DistributedFPPPatchPipelineRunner
from chitu_diffusion.modules.samplers.fpp_pipeline_schedule import build_patch_pipeline_schedule
from chitu_diffusion.modules.samplers.fpp_scheduler_adapter import FPPPatchSchedulerAdapter
from chitu_diffusion.modules.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
from chitu_diffusion.utils.wan_utils import cache_video
from chitu_diffusion.utils.shared_utils import (
    SequencePadder,
    split_latent,
)

from chitu_core.models.diffusion.model_wan import WanModel
from chitu_diffusion.flex_cache.strategy.FPPCache import FPPCache


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
        logger.info("init diffusion task dispatcher.")


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
            logger.info(f"Rank {self.rank}: Broadcasting task size {task_size.item()}")
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
            logger.info(f"Rank {self.rank}: Received task size {task_size.item()}, creating buffer")
            task_tensor = DiffusionTask.create_empty_serialization(
                size=task_size.item(),  # 注意这里要用 .item() 获取标量值
                device="cpu" if DiffusionBackend.use_gloo else self.local_rank
            )
        
        logger.info(f"Rank {self.rank} | {task_tensor.shape=} {task_size[0]=} {task_tensor.dtype=} {task_tensor.device=}, ready to broadcast.")
        
        # 第二阶段：广播实际的任务数据
        dist.broadcast(
            tensor=task_tensor,
            src=self.main_rank,
        )
        
        if not self.is_main_rank:
            # 接收方：反序列化任务
            task = DiffusionTask.deserialize(task_tensor)
            task_type = task.task_type
        
        logger.info(f"Rank {self.rank}: {task=}")
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
        self.timers = get_timers()
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

        if task_type == DiffusionTaskType.Terminate:
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
            if self.fpp_size > 1:
                self.fpp_denoise_steps(task)
                return
            else:
                out = self.denoise_step(task)

        else:
            raise NotImplementedError  
        
        self._update_task_stage_and_buffer(task, out)

        return out

    def fpp_denoise_steps(self, task: DiffusionTask):
        # first_steps for warmup 
        warm_up_steps = task.buffer.fpp_state.warmup_steps
        first_syncpipe_steps = task.req.params.num_inference_steps - warm_up_steps
        print(f"warmup steps: {warm_up_steps}, first syncpipe steps: {first_syncpipe_steps}", flush=True)

        if DiffusionBackend.boundary is not None:
            first_syncpipe_steps = int(DiffusionBackend.boundary * task.req.params.num_inference_steps) - warm_up_steps 

            second_syncpipe_steps = int( (1-DiffusionBackend.boundary) * task.req.params.num_inference_steps) - warm_up_steps

        else:
            first_syncpipe_steps = task.req.params.num_inference_steps - warm_up_steps

        
        for step_idx in range(warm_up_steps):
            do_cache = step_idx == warm_up_steps - 1 # only cache the last warmup step
            print(f"Rank {self.rank} starting warmup syncpipe step {step_idx+1}/{warm_up_steps}", flush=True)
            self.denoise_sync_pipeline(task, do_cache)


        for step_idx in range(first_syncpipe_steps):
            
            is_first_step = (step_idx == 0)
            is_last_step = (step_idx == first_syncpipe_steps - 1)
            print(f"Rank {self.rank} starting FPP Denoise step {step_idx+1}/{first_syncpipe_steps}", flush=True)
            self.fpp_denoise_one_step(task, is_first_step=is_first_step, is_last_step=is_last_step)
        if DiffusionBackend.boundary is not None:
            # after boundary, run another syncpipe to warm up the second model
            for step_idx in range(warm_up_steps):
                do_cache = step_idx == warm_up_steps - 1 # only cache the last warmup step
                self.denoise_sync_pipeline(task, do_cache)
            
            for step_idx in range(second_syncpipe_steps):
                is_first_step = (step_idx == 0)
                self.fpp_denoise_one_step(task, is_first_step=is_first_step)
        

        task.status = DiffusionTaskStatus.Pending
        return 
        
    @amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def denoise_sync_pipeline(self, task: DiffusionTask, save_cache: bool = False):
        latent_model_input = task.buffer.latents
        assert task.buffer.latents is not None and task.buffer.timesteps is not None


        timestep = task.buffer.timesteps[task.buffer.current_step]

        assert DiffusionBackend.guidance_scale > 0 and self.cfg_size == 1 and self.fpp_size > 1


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


    def fpp_update_step(self, task: DiffusionTask, patch_num: Optional[int] = None):
        is_main_process = dist.get_rank() == 0
        patch_num = patch_num if patch_num is not None else get_fpp_group().group_size

        task.buffer.current_step += 1

        # if get_fpp_group().is_first_rank:
        #     import pdb; pdb.set_trace()

        if task.buffer.current_step >= task.req.params.num_inference_steps:
            task.task_type = DiffusionTaskType.VAEDecode

            if get_fpp_group().is_first_rank:
                for patch_idx in range(patch_num):
                    self.fpp_receive_and_update_buffer(task)
            
            if is_main_process:
                print(f"Task {task.task_id} completed all denoise steps, moving to VAEDecode.")
        # float boundary may cause problems here
        # after switch, should clear cache and run warmup syncpipe again to fresh feature cache
        # so we need to deal with pipeline tail sync here to make sure the first rank's latent buffer is updated
        elif DiffusionBackend.boundary is not None and task.buffer.current_step == int(DiffusionBackend.boundary * task.req.params.num_inference_steps):
            DiffusionBackend.switch_active_model(flush=False)
            if get_fpp_group().is_first_rank:
                for patch_idx in range(patch_num):
                    self.fpp_receive_and_update_buffer(task)
            DiffusionBackend.flexcache.cache.clear() # clear cache to free memory after switch

        else:
            return True

    def fpp_receive_and_update_buffer(self, task: DiffusionTask):

        fpp_group = get_fpp_group()
        latent_shape = task.buffer.latents.shape

        recv_latent = fpp_group.p2p_irecv(latent_shape, torch.float32, src=fpp_group.prev_rank, tag=3)
        fpp_group.p2p_commit()
        fpp_group.p2p_wait()

        task.buffer.latents = recv_latent

    @amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    @torch.no_grad()
    def fpp_denoise_one_step(self, task: DiffusionTask, is_first_step: bool, is_last_step: bool):

        fpp_group = get_fpp_group()

        dtype = task.buffer.latents.dtype
        cur_step = task.buffer.current_step
        model : WanModel = DiffusionBackend.active_model

        patch_num = fpp_group.group_size

        time_embedding = model._cal_time_embeddings(task.buffer.timesteps[cur_step])
        time_proj = model._cal_timeproj(task.buffer.timesteps[cur_step])
        context_embedding = model._cal_context_embeddings(task.buffer.text_embeddings, clip_fea=None)
        negative_context_embedding = model._cal_context_embeddings(task.buffer.negative_embeddings, clip_fea=None)

        grid_sizes = task.buffer.grid_sizes



        for patch_idx in range(patch_num):

            patch_seq_len = task.buffer.seq_len // patch_num
            
            if fpp_group.is_first_rank:
                send_recv_flag = not (is_first_step and patch_idx < patch_num - 1)
            else:
                send_recv_flag = not ( (is_first_step and patch_idx == 0) or (is_last_step and patch_idx == patch_num - 1) )
            
             # only the last step's first patch and the first step's last patch can start without waiting for updated latents from last step, other patches need to wait for the data dependency to be resolved

            # pad for sending and forwarding
            token_patch_shape = torch.Size([1,patch_seq_len, model.dim])

            if fpp_group.is_first_rank:
                
                if not is_first_step:
                    
                    recv_latent = fpp_group.p2p_irecv(task.buffer.latents.shape, torch.float32, src=fpp_group.prev_rank, tag=3)
                    fpp_group.p2p_commit()
                    fpp_group.p2p_wait()

                    task.buffer.latents = recv_latent

                
    
                hidden_states : torch.Tensor = model._cal_patch_embedding(task.buffer.latents, seq_len=task.buffer.seq_len)
                hidden_states_patch = SequencePadder.split_sequence_padding(hidden_states, patch_num, split_dim=1, name='fpp')[patch_idx] # pad and split for fpp

                assert hidden_states_patch.shape == token_patch_shape, f"Expected hidden states patch shape {token_patch_shape}, but got {hidden_states_patch.shape}"


                hidden_states_cond_patch = hidden_states_patch.clone()
                hidden_states_uncond_patch = hidden_states_patch.clone()

            else: # first rank does pre_dit
                hidden_states_cond_patch = fpp_group.p2p_irecv(token_patch_shape, dtype=torch.float32, src=fpp_group.prev_rank, tag=1)
                hidden_states_uncond_patch = fpp_group.p2p_irecv(token_patch_shape, dtype=torch.float32, src=fpp_group.prev_rank, tag=2)
                fpp_group.p2p_commit()
                fpp_group.p2p_wait()


            position_idx = patch_idx * patch_seq_len
            position_idx_end = min(position_idx + patch_seq_len, task.buffer.unpad_seq_len)

            print(f"Rank {fpp_group.global_rank} processing patch {patch_idx}, position idx range [{position_idx}:{position_idx_end}]", flush=True)
            
             
            DiffusionBackend.cfg_type = CFGType.POS
            hidden_states_cond_patch = model.model_compute(hidden_states_cond_patch, time_proj, context_embedding, grid_sizes, save_cache=True, position_idx=position_idx)
            DiffusionBackend.cfg_type = CFGType.NEG
            hidden_states_uncond_patch = model.model_compute(hidden_states_uncond_patch, time_proj, negative_context_embedding, grid_sizes, save_cache=True, position_idx=position_idx)


            if not fpp_group.is_last_rank:

                assert hidden_states_cond_patch.dtype == torch.float32
                assert hidden_states_uncond_patch.shape == token_patch_shape, f"Expected hidden states shape {token_patch_shape}, but got {hidden_states_uncond_patch.shape}"

                fpp_group.p2p_isend(hidden_states_cond_patch, fpp_group.next_rank, tag=1)
                fpp_group.p2p_isend(hidden_states_uncond_patch, fpp_group.next_rank, tag=2)

                if not send_recv_flag:
                    fpp_group.p2p_commit()
                    fpp_group.p2p_wait()


            else:
                ## cache other patch latents,unpad
                ## hard_coding patch_grid_sizes
                unpad_boundary = (position_idx, position_idx_end)
                cache_strategy : FPPCache = DiffusionBackend.flexcache.strategy
  
                hidden_states_cond = cache_strategy.update_stale_tokens_patch(hidden_states_cond_patch, unpad_boundary, True)
                hidden_states_uncond = cache_strategy.update_stale_tokens_patch(hidden_states_uncond_patch, unpad_boundary, False)

                latents_cond = model._post_dit(hidden_states_cond, time_embedding, grid_sizes)[0].to(torch.float32)
                latents_uncond = model._post_dit(hidden_states_uncond, time_embedding, grid_sizes)[0].to(torch.float32)

                assert latents_cond.dtype == torch.float32

                noise_pred = latents_uncond + DiffusionBackend.guidance_scale * (latents_cond - latents_uncond)


                original_latent = task.buffer.latents

                print(f"[Sampler] step:{cur_step}, patch_idx:{patch_idx}")

                sampled_latents = task.buffer.sampler.step(
                    noise_pred.unsqueeze(0), 
                    task.buffer.timesteps[cur_step], 
                    original_latent.unsqueeze(0),
                    return_dict=False,
                    generator=task.buffer.seed_g,
                    update_step_index=False
                    )[0].squeeze(0)
       

                assert sampled_latents.dtype == torch.float32
                assert sampled_latents.shape == task.buffer.latents.shape, f"Expected sampled latents shape {task.buffer.latents.shape}, but got {sampled_latents.shape}"
                

                fpp_group.p2p_isend(sampled_latents, fpp_group.next_rank, tag=3)

                if not send_recv_flag:
                    fpp_group.p2p_commit()
                    fpp_group.p2p_wait()


        # after the last patch, update current_step and switch model if needed
        self.fpp_update_step(task)
        # update the latents at the last rank, this buffer works only as the input for scheduler's original latent, the timing of update is subtle but carefully considered.
        # requires warmup last step stores the sampled latents to last rank's buffer
        if fpp_group.is_last_rank:
            task.buffer.latents = sampled_latents
            task.buffer.sampler.increment_step_index()


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

        logger.info(f"[text_encode_step] task_id={task.task_id}, txt={payload}")
        
        # with device_scope(DiffusionBackend.text_encoder.model):        
        out = DiffusionBackend.text_encoder(payload, torch.cuda.current_device())
        
        logger.info(f"rank:{self.rank} [text_encode_step] context shape: {out.shape}")
        return out
    
    def vae_encode_step(self, task: DiffusionTask):
        pass
    
    @amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def denoise_step(self, task: DiffusionTask):
        assert task.buffer.latents is not None and task.buffer.timesteps is not None

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
                DiffusionBackend.cfg_type = CFGType.POS
                noise_pred_cond = DiffusionBackend.active_model(
                    latent_model_input,
                    t=timestep,
                    context=task.buffer.text_embeddings,
                    seq_len=task.buffer.seq_len
                )
                DiffusionBackend.cfg_type = CFGType.NEG
                noise_pred_uncond = DiffusionBackend.active_model(
                    latent_model_input,
                    t=timestep,
                    context=task.buffer.negative_embeddings,
                    seq_len=task.buffer.seq_len
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

        # logger.info(f"[Denoise Step] {task.buffer.current_step}/"
        #             f"{task.req.params.num_inference_steps} "
        #             f"timestep: {timestep} latents shape: {sampled_latents.shape}, {sampled_latents.dtype=}")

        return sampled_latents

        
    
    def vae_decode_step(self, task: DiffusionTask):
        target_decode_device = 0 # TODO: Flexible vae device
        if torch.distributed.get_rank() == target_decode_device:
            payload = [task.buffer.latents]
            logger.info(f"Step {task.buffer.current_step}: Enter VAE Decode Stage!")
            with device_scope(DiffusionBackend.vae.model):
                video = DiffusionBackend.vae.decode(payload)[0]
            self._save_image(task, video)
            return video
        return None


    # TODO: CPU/GPU overlap
    def _save_image(self, task: DiffusionTask, video: torch.Tensor):
        os.makedirs(task.req.params.save_dir, exist_ok=True)
        save_name = task.req.get_prompt()[:20].replace(" ", "_").replace(".", "") \
                    + f"_{task.task_id}.mp4"
        save_path = os.path.join(task.req.params.save_dir, save_name)
        logger.info(f"Saving video: {video.shape=} {video.dtype=}")
        cache_video(
            tensor=video[None],
            save_file=save_path,
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1))
        logger.info(f"[Succeed] Task {task.task_id} video saved to {save_path}")
        
    def _pre_denoising(self, task: DiffusionTask):
        """
        Before denoising loop, prepare latents, timesteps and solver for one task.
        TODO: Control Devices
        """

        device = torch.cuda.current_device()
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
            

            grid_sizes = torch.tensor([target_shape[1], target_shape[2] // patch_size[1], target_shape[3] // patch_size[2]], device=device).reshape(1,3)

     

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



        if self.fpp_size > 1:
            split_dim = 1 # split freq dim
            assert DiffusionBackend.args.models.transformer.patch_size[0] == 1, "patch size dim 0 must be 1 in our current implementation"
            task.buffer.fpp_split_idxs = split_latent(target_shape, self.fpp_size, split_dim)
            task.buffer.fpp_state = FPPTaskState(
                warmup_steps=5,
                warmup_done=False,
                num_patches=len(task.buffer.fpp_split_idxs),
                split_dim=split_dim,
            )
        
        DiffusionBackend.switch_active_model(flush=True)

        # enable flexcache
        if task.req.params.flexcache == "teacache":
            from chitu_diffusion.flex_cache.strategy.teacache import TeaCacheStrategy
            cache_strategy = TeaCacheStrategy(task=task)
            DiffusionBackend.flexcache.set_strategy(cache_strategy)
            # wrap model
            DiffusionBackend.flexcache.strategy.wrap_module_with_strategy(DiffusionBackend.active_model)
            logger.info("Teacache: Successfully wrapped models!")

        # logger.info(f"[Pre Denoise] Init {latents.shape=} {timesteps=}")
        if task.req.params.flexcache == "PAB":
            from chitu_diffusion.flex_cache.strategy.PAB import PABStrategy
            cache_strategy = PABStrategy(task=task)
            DiffusionBackend.flexcache.set_strategy(cache_strategy)
            # wrap model
            DiffusionBackend.flexcache.strategy.wrap_module_with_strategy(DiffusionBackend.active_model)
            logger.info("PAB: Successfully wrapped models!")

        if task.req.params.flexcache == "FPP":
            cache_strategy = FPPCache()
            print(f"Using FPP cache strategy with split num {self.fpp_size} and split dim {split_dim}")
            DiffusionBackend.flexcache.set_strategy(cache_strategy)
            logger.info("FPP: Successfully wrapped models!")

    def _update_task_stage_and_buffer(self, task: DiffusionTask, tokens: torch.Tensor):

        is_main_process = dist.get_rank() == 0
        has_img = task.req.init_image is not None
        
        if task.status == DiffusionTaskStatus.Running:
            task.status = DiffusionTaskStatus.Pending
        elif is_main_process:
            logger.warning(f"Task {task.task_id} - Status: {task.status}")
        
        if is_main_process:
            logger.info(f"[Task] ============ current stage: {task.task_type} ===============")

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
            task.task_type = DiffusionTaskType.Denoise
            
            if is_main_process:
                logger.debug(f"Task {task.task_id} transitioned to {task.task_type}")
            return True
        
        # 处理Denoise阶段（关键：需要多次执行）
        elif task.task_type == DiffusionTaskType.Denoise:
            # 更新当前去噪后的latents
            task.buffer.latents = tokens
            task.buffer.current_step += 1
            

            # 检查是否完成所有denoise步骤
            if task.buffer.current_step >= task.req.params.num_inference_steps:
                # 所有denoise步骤完成，转换到VAE Decode
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
            if is_main_process:
                logger.debug(f"Task {task.task_id} completed all stages")
            task.status = DiffusionTaskStatus.Completed
            self.current_task = None
            return False  # 完成所有阶段，不需要继续调度
        
        # 未知阶段
        else:
            if is_main_process:
                logger.error(f"Unknown task type: {task.task_type}")
            task.status = DiffusionTaskStatus.Failed
            return False
        
    
