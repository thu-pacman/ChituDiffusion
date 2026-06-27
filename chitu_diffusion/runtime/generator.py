# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import time
import torch
import torch.distributed as dist
from typing import Optional, List, Tuple, Dict, Any
import torch.amp as amp

from logging import getLogger
from chitu_diffusion.core.global_vars import get_global_args
from chitu_diffusion.core.logging_utils import (
    log_stage,
    log_progress,
    log_result,
    log_perf,
    should_log_on_rank,
)
from chitu_diffusion.runtime.backend import BackendState, DiffusionBackend
from chitu_diffusion.flexcache.params import (
    CubicParams,
    DiTangoParams,
    FlexCacheParams,
    FreeCacheParams,
    MeanCacheParams,
    StepTraceParams,
    TracePlannerParams,
)
from chitu_diffusion.runtime.task import (
    DiffusionTask,
    DiffusionTaskType,
    DiffusionTaskPool,
    DiffusionTaskStatus,
)
from chitu_diffusion.core.distributed.parallel_state import (
    get_cfg_group,
    get_cp_group,
    get_world_group
)
from chitu_diffusion.modules.utils.wan import cache_video
from chitu_diffusion.runtime.parallel_utils import SequencePadder
from chitu_diffusion.observability import Timer
from chitu_diffusion.runtime.output_naming import build_video_name_from_task
from chitu_diffusion.runtime.output_layout import (
    append_json_list_item,
    memory_metrics_dir,
    task_results_dir,
    timing_metrics_dir,
)
from chitu_diffusion.runtime.image_output import save_image_as_png
from chitu_diffusion.parallel import should_wrap_model_compute_for_context_parallel


logger = getLogger(__name__)

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
        if dist.is_initialized() and dist.get_world_size() == 1:
            assert task is not None
            return task.task_type, task

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
        # Parallel-sampler ("local latent") mode: when True the model_compute
        # wrapper assumes the image tokens are *already* the rank-local shard, so
        # it skips the per-call split of the latent and the final all-gather of the
        # output. The latent is split once before denoise and gathered once before
        # decode (see the adapter), keeping each rank's latent shard local across
        # all scheduler steps. Text / id sequences are still split per call (cheap,
        # comm-free, deterministic), so the math is identical to the replicated path.
        self.local_latent_mode = False

    def dispatch(self, tokens: torch.Tensor, name: str = 'x'):
        return SequencePadder.split_sequence_padding(tokens, 
                                                     split_num=self.cp_size,
                                                     split_dim=1, 
                                                     name=name)[self.rank_in_group]
    
    def gather(self, tokens: torch.Tensor, name: str = 'x'):
        tokens_list = [torch.empty_like(tokens) for _ in range(self.cp_size)]
        dist.all_gather(tensor_list=tokens_list, 
                        tensor=tokens, 
                        group=self.group.gpu_group)
        return SequencePadder.remove_sequence_padding_and_concat(tokens_list, 
                                                                 gather_dim=1,
                                                                 name=name)

    def dispatch_sequence_dim0(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        return SequencePadder.split_sequence_padding(
            tensor,
            split_num=self.cp_size,
            split_dim=0,
            name=name,
        )[self.rank_in_group]

    def dispatch_sequence_dim(self, tensor: torch.Tensor, split_dim: int, name: str) -> torch.Tensor:
        return SequencePadder.split_sequence_padding(
            tensor,
            split_num=self.cp_size,
            split_dim=split_dim,
            name=name,
        )[self.rank_in_group]

    def dispatch_id_sequence(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.ndim >= 3:
            return self.dispatch_sequence_dim(tensor, split_dim=1, name=name)
        return self.dispatch_sequence_dim(tensor, split_dim=0, name=name)

    def dispatch_flux_rotary_emb(
        self,
        rotary_emb,
        text_seq_len: int,
    ):
        if rotary_emb is None:
            return None
        if not isinstance(rotary_emb, (tuple, list)):
            return rotary_emb

        local_parts = []
        for i, tensor in enumerate(rotary_emb):
            seq_dim = 0
            total_seq_len = self._last_text_seq_len + self._last_image_seq_len
            for dim, size in enumerate(tensor.shape):
                if size == total_seq_len:
                    seq_dim = dim
                    break
            text_part = tensor.narrow(seq_dim, 0, text_seq_len)
            image_part = tensor.narrow(seq_dim, text_seq_len, tensor.shape[seq_dim] - text_seq_len)
            text_part = self.dispatch_sequence_dim(
                text_part,
                split_dim=seq_dim,
                name=f"flux_rotary_text_{i}",
            )
            image_part = self.dispatch_sequence_dim(
                image_part,
                split_dim=seq_dim,
                name=f"flux_rotary_image_{i}",
            )
            local_parts.append(torch.cat([text_part, image_part], dim=seq_dim))
        return tuple(local_parts)
    
    def wrap_model_compute_with_cp(self):
        """替换DiffusionBackend.model.model_compute方法，添加CP支持"""
        
        def create_wrapped_forward(model_instance):
            original_forward = model_instance.model_compute
            def wrapped_compute(tokens, **kwargs):
                local_mode = self.local_latent_mode
                original_image_seq_len = tokens.size(1)
                # Replicated path splits the full latent here; local-latent path
                # receives the rank-local shard directly and leaves it untouched.
                if not local_mode:
                    tokens = self.dispatch(tokens, name="hidden_states")
                if "seq_lens" in kwargs.keys():
                    kwargs["seq_lens"] = torch.tensor([tokens.size(1)])
                text_seq_len = None
                if "encoder_hidden_states" in kwargs and kwargs["encoder_hidden_states"] is not None:
                    text_seq_len = kwargs["encoder_hidden_states"].size(1)
                    self._last_text_seq_len = text_seq_len
                    self._last_image_seq_len = original_image_seq_len * self.cp_size if local_mode else original_image_seq_len
                    kwargs["encoder_hidden_states"] = self.dispatch(
                        kwargs["encoder_hidden_states"],
                        name="encoder_hidden_states",
                    )
                if "txt_ids" in kwargs and kwargs["txt_ids"] is not None:
                    kwargs["txt_ids"] = self.dispatch_id_sequence(kwargs["txt_ids"], name="txt_ids")
                if "img_ids" in kwargs and kwargs["img_ids"] is not None:
                    kwargs["img_ids"] = self.dispatch_id_sequence(kwargs["img_ids"], name="img_ids")
                if "image_rotary_emb" in kwargs and kwargs["image_rotary_emb"] is not None and text_seq_len is not None:
                    if local_mode:
                        raise NotImplementedError(
                            "Parallel-sampler local-latent mode does not support precomputed "
                            "image_rotary_emb yet; rotary must be derived from sharded img_ids."
                        )
                    kwargs["image_rotary_emb"] = self.dispatch_flux_rotary_emb(
                        kwargs["image_rotary_emb"],
                        text_seq_len=text_seq_len,
                    )
                x = original_forward(tokens, **kwargs)
                # Replicated path gathers back to the full latent; local-latent path
                # keeps the local shard so the scheduler step stays sharded too.
                if not local_mode:
                    x = self.gather(x, name="hidden_states")
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
            if should_wrap_model_compute_for_context_parallel(
                args,
                DiffusionBackend.model_adapter,
                DiffusionBackend.parallel_plan,
            ):
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

    def _release_current_task_if_stage_scheduled(self, task: DiffusionTask) -> None:
        if DiffusionBackend.model_adapter.schedule_each_stage():
            self.current_task = None

    def _run_dit_forward(self, task: DiffusionTask, branch: str, *args, **kwargs):
        step_index = int(task.buffer.current_step)
        timestep = None
        if task.buffer.timesteps is not None and step_index < len(task.buffer.timesteps):
            raw_timestep = task.buffer.timesteps[step_index]
            timestep = float(raw_timestep.item()) if hasattr(raw_timestep, "item") else float(raw_timestep)

        forward_call_index = getattr(task.buffer, "_dit_forward_call_index", 0)
        setattr(task.buffer, "_dit_forward_call_index", forward_call_index + 1)

        # When compile uses CUDA Graph (cudagraph trees), signal a new step so the
        # runtime knows the previous step's static output buffers are free to reuse.
        compile_mode = str(getattr(DiffusionBackend.args.infer.diffusion, "compile_mode", "off") or "off").lower()
        if "reduce-overhead" in compile_mode or "reduce_overhead" in compile_mode:
            torch.compiler.cudagraph_mark_step_begin()

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
                # Record per-task stage wall time so the per-task timing JSON can be
                # used to reconstruct a clean per-image end-to-end latency (warmup
                # tasks are named `warmup_*` and can be filtered out downstream).
                Timer.record_event(
                    "stage_elapsed",
                    {"task_id": task_id, "stage": stage.name, "elapsed_ms": elapsed_ms},
                )


    def step(self, task: Optional[DiffusionTask]) -> torch.Tensor:
        # 调度器会给generator task，翻译成kernel -> 运行 -> 正确放置输出 -> 回收对应内存
        # Prepare Payload
        control_override = torch.tensor(
            [1 if task is not None and task.is_control_signal() else 0],
            dtype=torch.int32,
            device="cpu" if DiffusionBackend.use_gloo else self.local_rank,
        )
        dist.broadcast(control_override, src=0)
        if bool(control_override.item()):
            self.current_task = None

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
            DiffusionTaskPool.clear_shutdown_request()
            self.current_task = None
            return None

        if task_type == DiffusionTaskType.Cancel:
            self._clear_ditango_planner()
            self._clear_flexcache_strategy()
            reason = str(task.signal_data.get("reason") or "Current generation cancelled")
            logger.info("Executing cancel signal task_id=%s reason=%s", task.task_id, reason)
            DiffusionTaskPool.cancel_active_tasks(reason)
            task.status = DiffusionTaskStatus.Completed
            DiffusionTaskPool.clear_cancel_request()
            self.current_task = None
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
        self._release_current_task_if_stage_scheduled(task)

        return out
        
    @Timer.get_timer("TextEncode")
    def text_encode_step(self, task: DiffusionTask) -> torch.Tensor:
        return DiffusionBackend.model_adapter.encode_text(task, self, DiffusionBackend)
    
    def vae_encode_step(self, task: DiffusionTask):
        pass
    
    @Timer.get_timer("denoise")
    @amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def denoise_step(self, task: DiffusionTask):
        assert task.buffer.latents is not None and task.buffer.timesteps is not None
        setattr(task.buffer, "_dit_forward_call_index", 0)
        sampled_latents = DiffusionBackend.model_adapter.denoise_step(
            task,
            self,
            DiffusionBackend,
            self._run_dit_forward,
        )
        self._record_dit_forward_step_summary(task)
        return sampled_latents

        
    @Timer.get_timer("VaeDecode")
    def vae_decode_step(self, task: DiffusionTask):
        logger.debug(f"task_id={task.task_id} entering VAE decode at step={task.buffer.current_step}")
        return DiffusionBackend.model_adapter.decode_latents(task, self, DiffusionBackend)
    
    def _post_vae_decode(self, task: DiffusionTask, video: Optional[torch.Tensor]):
        """Serving path: turn the decoded tensor into the output deliverable.

        This is the only post-decode work counted in end-to-end serving latency.
        Benchmark instrumentation (timing/memory metrics dumps) lives in
        `_record_task_metrics` and is measured separately so it does not inflate
        the reported latency.
        """
        target_decode_device = 0 # TODO: Flexible vae device
        if torch.distributed.get_rank() != target_decode_device:
            return
        if video is None:
            logger.warning(f"task_id={task.task_id} VAE decode returned None; skip saving video")
            return
        DiffusionBackend.model_adapter.save_output(task, video, self, DiffusionBackend)

    def _record_task_metrics(self, task: DiffusionTask):
        """Benchmark-only instrumentation: per-task timing/memory JSON + flexcache
        summaries. Excluded from serving latency (the caller wraps this in the
        `benchmark_overhead` timer so it can be subtracted from the wall clock).
        """
        if torch.distributed.get_rank() != 0:
            return
        run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        timing_dir = timing_metrics_dir(run_output_dir) if run_output_dir else task.req.params.save_dir
        memory_dir = memory_metrics_dir(run_output_dir) if run_output_dir else task.req.params.save_dir
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

        sidecar_path = os.path.splitext(save_path)[0] + ".json"
        metadata = {
            "filename": os.path.basename(save_path),
            "relative_path": os.path.join(os.path.basename(task.req.params.save_dir), os.path.basename(save_path)),
            "prompt": task.req.get_prompt(),
            "seed": getattr(task.req.params, "seed", None),
            "step": getattr(task.req.params, "num_inference_steps", None),
            "task_id": task.task_id,
            "model_name": getattr(getattr(getattr(DiffusionBackend, "args", None), "models", None), "name", None),
        }
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

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
        if spec.strategy == "ditango" and self.cp_size <= 1:
            raise ValueError("DiTango requires cp world size > 1.")
        if spec.strategy == "cubic" and self.cp_size != 1:
            raise ValueError("FlexCache Cubic requires cp world size = 1.")
        model_name = getattr(DiffusionBackend.args.models, "name", "")
        # MeanCache/FreeCache/StepTrace are step-level strategies: they cache the full noise prediction and
        # reuses it via a finite-difference JVP in the denoise loop, never touching the
        # model forward internals. It is therefore orthogonal to context parallelism
        # (the linear JVP commutes with sequence sharding). Validated for Flux1-dev,
        # Qwen-Image, and Wan under cp>1; other models stay gated until checked.
        if (
            spec.strategy in {"meancache", "freecache", "steptrace", "traceplanner"}
            and self.cp_size != 1
            and model_name
            not in {"Qwen-Image", "qwen-image", "Z-Image", "z-image", "Flux1-dev", "FLUX.1-dev", "Wan2.1-T2V-1.3B"}
        ):
            raise ValueError(
                "FlexCache step-level cache currently requires cp_size = 1 outside "
                "Qwen-Image / Z-Image / Flux1-dev / Wan."
            )
        if spec.strategy in {"meancache", "freecache", "steptrace", "traceplanner"} and self.cfg_size not in {1, 2}:
            raise ValueError("FlexCache step-level cache currently supports cfg_size = 1 or 2.")
        return spec

    def _build_flexcache_strategy(self, task: DiffusionTask, spec: FlexCacheParams):
        strategy = spec.strategy
        warmup_steps = spec.warmup
        cooldown_steps = spec.cooldown
        model_name = getattr(DiffusionBackend.args.models, "name", "")

        if model_name in {"Qwen-Image", "qwen-image"} and strategy in {"teacache", "taylorseer"}:
            raise NotImplementedError(
                f"FlexCache strategy '{strategy}' is not enabled for Qwen-Image; "
                "use blockdance or pab for now."
            )

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

        if strategy == "meancache":
            from chitu_diffusion.flexcache.strategy.meancache import MeanCacheStrategy

            if not isinstance(spec, MeanCacheParams):
                raise TypeError("FlexCache meancache requires MeanCacheParams.")
            if model_name not in {"Qwen-Image", "qwen-image", "Z-Image", "z-image", "Flux1-dev", "FLUX.1-dev", "Wan2.1-T2V-1.3B"}:
                raise NotImplementedError(
                    "FlexCache MeanCache is currently implemented only for Qwen-Image, Z-Image, Flux1-dev, and Wan."
                )
            return MeanCacheStrategy(
                task=task,
                fresh_steps=getattr(spec, "fresh_steps", 25),
                warmup_steps=warmup_steps,
                cooldown_steps=cooldown_steps,
                use_jvp=getattr(spec, "use_jvp", True),
            )

        if strategy == "freecache":
            from chitu_diffusion.flexcache.strategy.freecache import FreeCacheStrategy

            if not isinstance(spec, FreeCacheParams):
                raise TypeError("FlexCache freecache requires FreeCacheParams.")
            if model_name not in {"Qwen-Image", "qwen-image", "Z-Image", "z-image", "Flux1-dev", "FLUX.1-dev", "Wan2.1-T2V-1.3B"}:
                raise NotImplementedError(
                    "FlexCache FreeCache is currently implemented only for Qwen-Image, Z-Image, Flux1-dev, and Wan."
                )
            return FreeCacheStrategy(
                task=task,
                tol=getattr(spec, "tol", 0.15),
                max_gap=getattr(spec, "max_gap", 8),
                jvp_order=getattr(spec, "jvp_order", 1),
                warmup_steps=warmup_steps,
                cooldown_steps=cooldown_steps,
                anchor_interval=getattr(spec, "anchor_interval", None),
                anchor_phase=getattr(spec, "anchor_phase", 0),
                forced_compute_steps=getattr(spec, "forced_compute_steps", None),
                forced_reuse_orders=getattr(spec, "forced_reuse_orders", None),
                save_vectors=getattr(spec, "save_vectors", False),
            )

        if strategy == "steptrace":
            from chitu_diffusion.flexcache.strategy.steptrace import StepTraceStrategy

            if not isinstance(spec, StepTraceParams):
                raise TypeError("FlexCache steptrace requires StepTraceParams.")
            if model_name not in {"Qwen-Image", "qwen-image", "Z-Image", "z-image", "Flux1-dev", "FLUX.1-dev", "Wan2.1-T2V-1.3B"}:
                raise NotImplementedError(
                    "FlexCache StepTrace is currently implemented only for Qwen-Image, Z-Image, Flux1-dev, and Wan."
                )
            return StepTraceStrategy(
                task=task,
                jvp_order=getattr(spec, "jvp_order", 1),
                save_vectors=getattr(spec, "save_vectors", False),
            )

        if strategy == "traceplanner":
            from chitu_diffusion.flexcache.strategy.traceplanner import TracePlannerStrategy

            if not isinstance(spec, TracePlannerParams):
                raise TypeError("FlexCache traceplanner requires TracePlannerParams.")
            if model_name not in {"Qwen-Image", "qwen-image", "Z-Image", "z-image", "Flux1-dev", "FLUX.1-dev", "Wan2.1-T2V-1.3B"}:
                raise NotImplementedError(
                    "FlexCache TracePlanner is currently implemented only for Qwen-Image, Z-Image, Flux1-dev, and Wan."
                )
            return TracePlannerStrategy(
                task=task,
                budgets=getattr(spec, "budgets", None),
                beam_width=getattr(spec, "beam_width", 8),
                max_gap=getattr(spec, "max_gap", 8),
                warmup_steps=warmup_steps,
                cooldown_steps=cooldown_steps,
                hard_tol=getattr(spec, "hard_tol", 1.0e-2),
                alpha=getattr(spec, "alpha", 2.0),
                beta=getattr(spec, "beta", 1.0),
                checkpoint_interval=getattr(spec, "checkpoint_interval", 0),
                checkpoint_topk=getattr(spec, "checkpoint_topk", 0),
            )

        if strategy == "cubic":
            from chitu_diffusion.flexcache.strategy.cubic import (
                CubicFluxModelParams,
                CubicQwenImageModelParams,
                CubicStrategy,
                CubicWanModelParams,
            )

            if not isinstance(spec, CubicParams):
                raise TypeError("FlexCache cubic requires CubicParams.")
            if model_name in {"Flux1-dev", "FLUX.1-dev"}:
                model_params = CubicFluxModelParams()
            elif model_name in {"Qwen-Image", "qwen-image"}:
                model_params = CubicQwenImageModelParams()
            else:
                model_params = CubicWanModelParams()
            if spec.block_size is not None:
                model_params.min_block_size = int(spec.block_size)
                model_params.max_block_size = int(spec.block_size)
            if spec.uniform_square_min_splits is not None:
                model_params.uniform_square_min_splits = int(spec.uniform_square_min_splits)
            return CubicStrategy(
                task=task,
                target_speedup=spec.target_speedup,
                warmup_steps=warmup_steps,
                cooldown_steps=cooldown_steps,
                tau_max=spec.tau_max,
                model_params=model_params,
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
            anchor_interval=spec.anchor_interval,
            warmup_steps=spec.warmup,
            cooldown_steps=spec.cooldown,
            tau_max=spec.tau_max,
            intra_group_size_limit=spec.intra_group_size_limit,
            curvature_interval_power=spec.curvature_interval_power,
            locality_group_compute_boost=spec.locality_group_compute_boost,
            groupwise_stagger_period=spec.groupwise_stagger_period,
            groupwise_stagger_fresh_count=spec.groupwise_stagger_fresh_count,
            groupwise_stagger_layer_start=spec.groupwise_stagger_layer_start,
            groupwise_stagger_layer_end=spec.groupwise_stagger_layer_end,
            groupwise_keep_local=spec.groupwise_keep_local,
            groupwise_force_tail_full_layers=spec.groupwise_force_tail_full_layers,
            groupwise_reuse_stale_kv=spec.groupwise_reuse_stale_kv,
            groupwise_local_expand=spec.groupwise_local_expand,
            groupwise_fixed_anchor_steps=spec.groupwise_fixed_anchor_steps,
            groupwise_topk_mode=spec.groupwise_topk_mode,
            groupwise_extra_topk=spec.groupwise_extra_topk,
            groupwise_state_align=spec.groupwise_state_align,
            groupwise_state_align_mode=spec.groupwise_state_align_mode,
            groupwise_state_align_out_scale=spec.groupwise_state_align_out_scale,
            groupwise_state_align_lse_scale=spec.groupwise_state_align_lse_scale,
            groupwise_state_align_distance_tau=spec.groupwise_state_align_distance_tau,
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
                "intra_group_size_limit": planner.intra_group_size_limit,
                "group_num": planner.group_num,
                "curvature_interval_power": planner.curvature_interval_power,
                "locality_group_compute_boost": planner.locality_group_compute_boost,
                "anchor_interval": planner.anchor_interval,
                "groupwise_stagger_period": planner.groupwise_stagger_period,
                "groupwise_stagger_fresh_count": planner.groupwise_stagger_fresh_count,
                "groupwise_stagger_layer_start": planner.groupwise_stagger_layer_start,
                "groupwise_stagger_layer_end": planner.groupwise_stagger_layer_end,
                "groupwise_keep_local": planner.groupwise_keep_local,
                "groupwise_force_tail_full_layers": planner.groupwise_force_tail_full_layers,
                "groupwise_reuse_stale_kv": planner.groupwise_reuse_stale_kv,
                "groupwise_local_expand": planner.groupwise_local_expand,
                "groupwise_fixed_anchor_steps": planner.groupwise_fixed_anchor_steps,
                "groupwise_topk_mode": planner.groupwise_topk_mode,
                "groupwise_extra_topk": planner.groupwise_extra_topk,
                "groupwise_state_align": planner.groupwise_state_align,
                "groupwise_state_align_mode": planner.groupwise_state_align_mode,
                "groupwise_state_align_out_scale": planner.groupwise_state_align_out_scale,
                "groupwise_state_align_lse_scale": planner.groupwise_state_align_lse_scale,
                "groupwise_state_align_distance_tau": planner.groupwise_state_align_distance_tau,
            }
            logger.info(
                f"{self._flexcache_strategy_name(spec.strategy)}: Successfully wrapped models with resolved params {resolved_log}."
            )
            return

        if DiffusionBackend.flexcache is None:
            raise RuntimeError("FlexCache manager is not initialized.")

        DiffusionBackend.flexcache.reset_compute_stats()

        if DiffusionBackend.ditango is not None:
            self._clear_ditango_planner()

        if DiffusionBackend.flexcache.strategy is not None:
            self._clear_flexcache_strategy()

        cache_strategy = self._build_flexcache_strategy(task, spec)
        DiffusionBackend.flexcache.set_strategy(cache_strategy)
        if DiffusionBackend.active_model is None:
            raise ValueError(
                f"FlexCache strategy '{spec.strategy}' requires an internal transformer model; "
                f"{DiffusionBackend.args.models.name} uses an external pipeline."
            )
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
        elif spec.strategy == "meancache":
            resolved_log["fresh_steps"] = cache_strategy.fresh_steps
            resolved_log["use_jvp"] = cache_strategy.use_jvp
        elif spec.strategy == "freecache":
            resolved_log["tol"] = cache_strategy.tol
            resolved_log["max_gap"] = cache_strategy.max_gap
            resolved_log["jvp_order"] = cache_strategy.jvp_order
            resolved_log["anchor_interval"] = cache_strategy.anchor_interval
            resolved_log["anchor_phase"] = cache_strategy.anchor_phase
            if cache_strategy.forced_compute_steps is not None:
                resolved_log["forced_compute_steps"] = sorted(cache_strategy.forced_compute_steps)
            if cache_strategy.forced_reuse_orders is not None:
                resolved_log["forced_reuse_orders"] = dict(sorted(cache_strategy.forced_reuse_orders.items()))
            if cache_strategy.anchor_compute_steps is not None:
                resolved_log["anchor_compute_steps"] = sorted(cache_strategy.anchor_compute_steps)
        elif spec.strategy == "traceplanner":
            resolved_log["budgets"] = cache_strategy.budgets
            resolved_log["beam_width"] = cache_strategy.beam_width
            resolved_log["max_gap"] = cache_strategy.max_gap
            resolved_log["hard_tol"] = cache_strategy.hard_tol
            resolved_log["checkpoint_interval"] = cache_strategy.checkpoint_interval
            resolved_log["checkpoint_topk"] = cache_strategy.checkpoint_topk
        elif spec.strategy == "cubic":
            resolved_log["target_speedup"] = cache_strategy.config.target_speedup
            resolved_log["tau_max"] = cache_strategy.config.anchor_interval
            resolved_log["block_size"] = cache_strategy.config.max_block_size
            resolved_log["uniform_square_min_splits"] = cache_strategy.config.uniform_square_min_splits
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
        DiffusionBackend.model_adapter.prepare_denoise(task, self, DiffusionBackend)

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
            if DiffusionBackend.model_adapter.denoise_completes_in_single_call():
                task.buffer.current_step = task.req.params.num_inference_steps
            else:
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
            # Inference (decode) is done; close the stage timer so the VAEDecode stage
            # latency reflects compute only. Writing the PNG artifact + sidecar JSON and
            # dumping per-task metrics are benchmark-harness I/O (a real server returns
            # bytes, it does not persist files to a results dir), so they run under
            # "benchmark_overhead" and are subtracted to yield serving_elapsed_s.
            self._emit_stage_end(DiffusionTaskType.VAEDecode, task.task_id)
            with Timer.get_timer("benchmark_overhead"):
                self._post_vae_decode(task, tokens)  # save image/video artifact
                self._record_task_metrics(task)      # timing/memory metrics dumps
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
        
    
