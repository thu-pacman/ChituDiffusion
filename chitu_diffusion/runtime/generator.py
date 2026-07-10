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
from chitu_diffusion.runtime.elastic_policy import _mode_for_width
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
    get_world_group,
    dynamic_sp_degrees,
    lane_widths_available,
    initialize_lane_groups,
    dynamic_cfg_strides,
    activate_lane_topology,
    reset_active_lane_topology,
)
from chitu_diffusion.core.distributed.lane_topology import LaneTopology
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
        # Rank 0 owns scheduling decisions; metadata is broadcast so every rank
        # participating in a selected lane starts from the same task state.
        self.group = get_world_group()
        self.rank = self.group.global_rank
        self.local_rank = self.group.local_rank

        self.main_rank = 0
        self.is_main_rank = self.group.is_first_rank
        logger.debug("init diffusion task dispatcher")


    def dispatch_metadata(self, task: Optional[DiffusionTask] = None) -> tuple[DiffusionTaskType, DiffusionTask]:
        if self.group.group_size == 1:
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
            self.group.broadcast(task_size, src=self.main_rank)
            
        else:
            # 接收方：创建空的size tensor用于接收
            task_size = torch.zeros(1, dtype=torch.int64,
                                device="cpu" if DiffusionBackend.use_gloo else self.local_rank)
            
            # 第一阶段：接收任务大小
            self.group.broadcast(task_size, src=self.main_rank)
            
            # 第二阶段：根据接收到的大小创建空buffer
            logger.debug(f"Rank {self.rank}: Received task size {task_size.item()}, creating buffer")
            task_tensor = DiffusionTask.create_empty_serialization(
                size=task_size.item(),  # 注意这里要用 .item() 获取标量值
                device="cpu" if DiffusionBackend.use_gloo else self.local_rank
            )
        
        logger.debug(f"Rank {self.rank} | {task_tensor.shape=} {task_size[0]=} {task_tensor.dtype=} {task_tensor.device=}, ready to broadcast.")
        
        # 第二阶段：广播实际的任务数据
        self.group.broadcast(task_tensor, src=self.main_rank)
        
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
    

class LaneContextParallelExecutor:
    def __init__(self):
        super().__init__()

    @property
    def group(self):
        return get_cp_group()

    @property
    def cp_size(self):
        return self.group.group_size

    @property
    def rank_in_group(self):
        return self.group.rank_in_group

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
                local_mode = False
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
        if max(dynamic_sp_degrees() or [get_cp_group().group_size]) > 1:
            if should_wrap_model_compute_for_context_parallel(
                args,
                DiffusionBackend.model_adapter,
                DiffusionBackend.parallel_plan,
            ):
                # The wrapper resolves the CP group on every call. The generator
                # does not own a launch-time dispatcher or cached communicator.
                LaneContextParallelExecutor().wrap_model_compute_with_cp()

        # Ensure FIFO
        self.current_task = None # 通过这个储存当前任务的中间状态
        diffusion_args = args.infer.diffusion
        scheduler_policy = str(getattr(diffusion_args, "scheduling_policy", "fifo") or "fifo")
        # slo_elastic pool engine: request-level DP + per-request SP width across a shared
        # GPU pool. Each round rank 0 plans a full layout (which distinct request runs on
        # which contiguous GPU block at which SP width), broadcasts it, every rank runs one
        # bounded denoise phase for its lane's task(s) on its lane subgroup, then a world barrier. The
        # in-flight denoise set is ``pool_group`` (kept replicated across all ranks; see
        # ``_pool_round``).
        # slo_elastic (SLO-aware) + pure_dp / pure_sp (fixed-layout baselines) all drive
        # the same pool engine; only the layout decision in the scheduler differs.
        self.pool_engine = scheduler_policy in {"slo_elastic", "pure_dp", "pure_sp"}
        self.pool_group: List[DiffusionTask] = []
        epe = getattr(diffusion_args, "epe", None)
        # The planner and executor read the same serve-config value. ``pure_sp`` is
        # always the true static CP baseline, independent of the EPE CFP setting.
        configured_cfp = max(1, int(getattr(epe, "cfg_parallel_max", 2) or 2))
        # ``pure_sp`` is the true static CP baseline; slo_elastic uses the optimal
        # CFP-first decomposition by default.
        self.pool_cfg_parallel_max = 1 if scheduler_policy == "pure_sp" else configured_cfp
        # slo_elastic pool-engine instrumentation (stage 0 baseline). Decomposes the
        # per-round tax into per-stage timers (pool_plan_ms / pool_broadcast_lanes_ms /
        # pool_admit_ms / pool_denoise_ms / pool_barrier_ms / pool_replicate_ms /
        # pool_retire_ms / pool_round_ms) plus a per-round structured "pool_round" event
        # (layout widths, stepped/replicated counts, replicated bytes). These flow into
        # the timing summary.json via Timer.statistics_dict()/records_dict(). Toggle with
        # CHITU_POOL_INSTRUMENT (default on); disable for clean perf A/B runs.
        self._pool_instrument = str(os.getenv("CHITU_POOL_INSTRUMENT", "1")).strip().lower() in {"1", "true", "yes", "on"}
        # Online cost-model calibration (mirrors DiffusionScheduler.online_calibrate).
        # When on, each phase the engine feeds its measured clean per-step latency back to
        # the planner's calibrator via one tiny all_gather. Requires instrumentation (the
        # sample source). Enabled by default for accurate planner-owned balanced K.
        self._pool_online_calibrate = (
            self._pool_instrument
            and bool(getattr(epe, "online_calibration", True))
        )
        self._pool_round_index = 0
        # Placement preservation (stage 2). When on, canonical GPU-block assignment first
        # tries to keep each in-flight request on its previous (still width-aligned, free)
        # block instead of re-packing widest-first every phase, so a request that keeps its
        # width does not migrate. Sets up the targeted-replication savings in stage 6.
        self._pool_placement_affinity = True
        self._pool_placement_reuse = 0
        self._pool_placement_change = 0
        # Rank-0 memo of the *full* placement computed last round ({request_id: gpu block}).
        # Per-task ``_pool_gpus`` is only refreshed for the lane the local rank participates
        # in (rank 0 only runs the GPU0 lane), so reading it back on rank 0 gives stale/empty
        # blocks for every other in-flight lane -- which silently defeated placement affinity
        # for the tail width-1 lanes (needless GPU swaps within an 8-GPU node, where no
        # comm-locality exists so a move only costs a process-group re-warm + latent migration).
        # Since rank 0 already computes the concrete block for *every* lane each round, we just
        # remember it and feed it back as the affinity hint next round.
        self._pool_prev_placement: Dict[str, List[int]] = {}
        # Text-encode decoupling (stage 3). "rank0_gpu": only rank 0 runs the (GPU) text
        # encoder and broadcasts the embeddings to the world, removing the O(world_size)
        # duplicated encode compute on the synchronous admission path; "all_ranks" keeps the
        # original behavior where every rank encodes identically. An encoder cache (keyed by
        # prompt text + max_seq_len) skips re-encoding repeated prompts. World-broadcast (not
        # lane-directed) preserves the replicated-buffer invariant so a task stays reassignable.
        self._text_encode_mode = str(os.getenv("CHITU_TEXT_ENCODE", "rank0_gpu")).strip().lower()
        self._text_encode_cache: Dict[Tuple[str, int], Tuple[torch.Tensor, List[int]]] = {}
        self._text_encode_cache_cap = max(0, int(os.getenv("CHITU_TEXT_ENCODE_CACHE", "256") or 256))
        self._text_encode_cache_hits = 0
        self._text_encode_cache_miss = 0
        # Async output (stage 4). VAE decode runs concurrently within each finishing lane;
        # lane leaders return images to the rank-0 CPU writer
        # thread with a bounded queue (backpressure), so a phase boundary does not block on
        # disk I/O while other ranks wait. Per-task metrics stay on the main thread (cheap,
        # already excluded from serving latency, and avoids racing Timer's records dict).
        self._async_output = str(os.getenv("CHITU_ASYNC_OUTPUT", "on")).strip().lower() in {"1", "true", "yes", "on"}
        self._async_output_maxq = max(1, int(os.getenv("CHITU_ASYNC_OUTPUT_QUEUE", "8") or 8))
        self._async_output_queue = None
        self._async_output_thread = None
        # Targeted latent replication (stage 6). Instead of re-broadcasting every stepped
        # task's latent to the whole world after each phase, track each task's owner ranks
        # (where its current latent lives) and only migrate when the planner moves it to a
        # block whose ranks don't already hold it. A finishing task decodes directly on its
        # owner lane, so retirement does not require world replication.
        self._pool_targeted_repl = True
        # Precise targeted migration (stage 6.5). When on, a pre-phase migration sends the
        # latent ONLY to the ranks that are missing it (needed - valid_ranks) via narrow
        # point-to-point (batched isend/irecv), instead of broadcasting from an owner leader
        # to the whole world. width 1->2 moves 1 rank, 2->4 moves 2 ranks, etc. Owner is
        self.enable_step_interleave = bool(
            getattr(diffusion_args, "step_interleave", False)
            or scheduler_policy in {"step_interleave", "admission_control", "shape_aware_ratio"}
        )
        self._last_logged_stage = {}
        self.denoise_progress_interval = max(1, int(os.getenv("CHITU_PROGRESS_INTERVAL", "5")))
        self.enable_stage_perf = bool(getattr(args.output, "timer", False))
        self._stage_start_time = {}
        self._dit_forward_step_elapsed_ms = {}
        # slo_elastic pool engine: pre-warm the canonical contiguous lane CP groups
        # (widths {1,2,4,...} that divide the world) so switching a lane's width at a
        # step boundary is a pure process-group pointer swap. Collective on all ranks;
        # idempotent (no-op if dynamic-SP groups already exist).
        if self.pool_engine:
            initialize_lane_groups()
            self._pool_widths = lane_widths_available()
            if self.rank == 0:
                logger.info(
                    "[slo_elastic pool] engine enabled: lane widths=%s world_size=%d",
                    self._pool_widths, torch.distributed.get_world_size(),
                )

    def _release_current_task_if_stage_scheduled(self, task: DiffusionTask) -> None:
        if DiffusionBackend.model_adapter.schedule_each_stage():
            self.current_task = None
            return
        if (
            self.enable_step_interleave
            and task.task_type == DiffusionTaskType.Denoise
            and task.status == DiffusionTaskStatus.Pending
        ):
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


    def step(self, task: Optional[DiffusionTask] = None) -> torch.Tensor:
        """Unified per-round entry called on every rank."""
        if self.pool_engine:
            return self._pool_round()
        return self._step_single(task)

    # ------------------------------------------------------------------
    # slo_elastic pool engine (request-level DP + per-request SP width).
    #
    # Fidelity note (synchronous rounds): one engine round == every assigned lane
    # advances EXACTLY ONE denoise step on its own contiguous GPU block, then a
    # world barrier. Distinct requests run concurrently on disjoint CP subgroups
    # (true request-level DP + heterogeneous simultaneous per-request width).
    # Every in-flight task's latent is kept replicated on all ranks (re-broadcast
    # from its lane leader after each step), so a task can be reassigned to any
    # lane/width next round with a pure process-group pointer swap and zero
    # explicit migration. This is the lockstep realization of the async simulator:
    # a round costs the slowest lane's step (documented gap vs the async ground
    # truth) and retirement (VAE decode + save) is world-synchronized.
    # ------------------------------------------------------------------
    _POOL_CONTINUE = 0
    _POOL_CONTROL = 1

    def _pool_sync(self) -> None:
        """CUDA sync used only to attribute per-stage timing accurately under
        instrumentation. No-op (no added sync) when instrumentation is disabled, so
        clean perf A/B runs keep the original synchronization schedule."""
        if self._pool_instrument and torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _pool_rec(name: str, start: float) -> float:
        """Record ``perf_counter``-based elapsed ms for a segment and return ``now``
        so it can seed the next segment. Guarded by the caller's instrument flag."""
        now = time.perf_counter()
        Timer.record(name, (now - start) * 1000.0)
        return now

    def _pool_cfg_parallel_active(self) -> bool:
        """Whether this serve can run per-lane CFG parallelism (CFP): the run opted in
        (``epe.cfg_parallel_max>=2``, shared with the planner), the model uses
        classifier-free guidance (guidance_scale > 1), AND CFP pair-groups were
        pre-warmed. Guidance is a model-level config here, so this is a fixed bool."""
        if self.pool_cfg_parallel_max < 2:
            return False  # not opted in; widened lanes stay pure context-parallel.
        if not dynamic_cfg_strides():
            return False
        try:
            return bool(DiffusionBackend.model_adapter.supports_cfg(self.args))
        except Exception:
            try:
                return float(self.args.models.sampler.guidance_scale[0]) > 1.0
            except Exception:
                return False

    def _pool_lane_cfp(self, width: int) -> int:
        """CFP degree for a lane of ``width`` GPUs: 2 (cond|uncond split, CP stride
        ``width//2``) when CFP is active and the split is pre-warmed, else 1. This is
        the DP>CFP>CP realization -- a widened lane spends its first factor of two on
        (near-perfect) CFG parallelism before context parallelism."""
        width = int(width)
        if width >= 2 and width % 2 == 0 and self._pool_cfg_parallel_active() \
                and (width // 2) in dynamic_cfg_strides():
            return 2
        return 1

    def _pool_round(self) -> Optional[torch.Tensor]:
        """One slo_elastic pool-engine round (called on every rank)."""
        is_main = self.rank == 0
        device = "cpu" if DiffusionBackend.use_gloo else self.local_rank
        world = get_world_group()
        world_size = world.group_size
        instrument = self._pool_instrument
        round_t0 = time.perf_counter()

        # 1. Control signals (shutdown/cancel) fall back to the single-task path,
        #    but only when nothing is in flight (so we never orphan denoise state).
        control_task = None
        if is_main and not self.pool_group:
            if DiffusionTaskPool.has_shutdown_request():
                control_task = DiffusionTaskPool.get_shutdown_task()
            elif DiffusionTaskPool.has_cancel_request():
                control_task = DiffusionTaskPool.get_cancel_task()
        mode = self._POOL_CONTROL if control_task is not None else self._POOL_CONTINUE
        mode_t = torch.tensor([mode if is_main else 0], dtype=torch.int64, device=device)
        dist.broadcast(mode_t, src=0)
        if int(mode_t.item()) == self._POOL_CONTROL:
            return self._step_single(control_task if is_main else None)

        # 2. rank 0 plans the round (canonical lane layout + which pending to admit).
        t = time.perf_counter()
        if is_main:
            lanes, admit_tasks = self._pool_plan_layout(world_size)
            plan_blob = json.dumps({"lanes": lanes})
            self._pool_log_layout(lanes, world_size)
            if instrument:
                t = self._pool_rec("pool_plan_ms", t)
        else:
            lanes, admit_tasks = [], []
            plan_blob = None
        lanes = self._pool_broadcast_lanes(plan_blob, device)
        if instrument:
            t = self._pool_rec("pool_broadcast_lanes_ms", t)

        # 3. Admit newly-planned pending tasks (text-encode -> denoise-ready) on ALL
        #    ranks so each in-flight task's buffer is replicated everywhere.
        n_admit = len(admit_tasks) if is_main else 0
        n_t = torch.tensor([n_admit], dtype=torch.int64, device=device)
        dist.broadcast(n_t, src=0)
        n_admit = int(n_t.item())
        dispatcher = DiffusionTaskDispatcher()
        fanout_parent_ids: set[str] = set()
        for idx in range(n_admit):
            if dispatcher.group.group_size == 1:
                new_task = admit_tasks[idx]
            else:
                src = admit_tasks[idx] if (is_main and admit_tasks) else None
                _, new_task = dispatcher.dispatch_metadata(src)
            if self._pool_admit_one(new_task):
                fanout_parent_ids.add(new_task.task_id)
        if fanout_parent_ids:
            # The parent lane was only an admission slot. Its independently
            # schedulable sample children enter the denoise planner next round.
            lanes = [
                lane for lane in lanes
                if lane.get("request_id") not in fanout_parent_ids
            ]
        if instrument:
            self._pool_sync()
            t = self._pool_rec("pool_admit_ms", t)

        if not self.pool_group:
            # Nothing in flight (e.g. pending not yet schedulable) -> stay in lockstep.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.perf_counter()
            dist.barrier()
            if instrument:
                self._pool_rec("pool_barrier_ms", t)
                Timer.record("pool_round_ms", (time.perf_counter() - round_t0) * 1000.0)
                self._pool_round_index += 1
            return None

        pool_by_id = {t2.task_id: t2 for t2 in self.pool_group}

        # Phase length K is part of the rank-0 planner output and already arrived in
        # the lane-layout broadcast. Executors consume it verbatim; they never run a
        # second cost-model decision that could diverge from the scored plan.
        k_list = [max(1, int(lane.get("planned_k", 1))) for lane in lanes]
        phase_k = max(k_list) if k_list else 1  # span of the phase (for records/instrumentation)
        if is_main:
            scheduler_for_record = DiffusionBackend.scheduler
            lane_remaining_before = [int(self._pool_lane_remaining(ln, pool_by_id)) for ln in lanes]
            lane_step_ms_before = [
                float(self._pool_lane_step_ms(ln, pool_by_id, scheduler_for_record)) for ln in lanes
            ]
        else:
            lane_remaining_before = []
            lane_step_ms_before = []

        # 3b. Targeted pre-phase migration: move a task's latent to its new lane only if the
        #     lane's ranks don't already hold the current latent (placement preservation
        #     means most phases move nothing). Skipped when targeted replication is off.
        migrated_count = migrated_bytes = 0
        if self._pool_targeted_repl:
            migrated_count, migrated_bytes = self._pool_migrate_latents(lanes, pool_by_id)
            if instrument:
                self._pool_sync()
                t = self._pool_rec("pool_migrate_ms", t)

        # 4. Each rank advances its lane by K denoise steps (no per-step world barrier or
        #    latent replication); idle ranks skip straight to the boundary barrier.
        t = time.perf_counter()
        lane_steps_run = 0
        reset_active_lane_topology()
        my_lane_idx = next((i for i, ln in enumerate(lanes) if self.rank in ln["gpus"]), None)
        my_lane = lanes[my_lane_idx] if my_lane_idx is not None else None
        phase_k_local = k_list[my_lane_idx] if my_lane_idx is not None else phase_k
        if my_lane is not None:
            # DP>CFP>CP realization. cfp==2 splits the lane into a cond half and an
            # uncond half, each CP-sharded at ``width//cfp``; the CP group therefore
            # activates at the *per-condition* degree and the CFG pair-group at stride
            # ``width//cfp`` reunites cond/uncond each step. cfp==1 is the legacy path
            # (CP over the whole lane, cond+uncond batched on each rank).
            cfp = self._pool_lane_cfp(my_lane["width"])
            cp_degree = my_lane["width"] // cfp
            activate_lane_topology(
                LaneTopology(
                    offset=my_lane["gpus"][0],
                    width=my_lane["width"],
                    cfp_degree=cfp,
                    cp_degree=cp_degree,
                )
            )
            if my_lane["width"] == 1:
                rid = my_lane.get("request_id") or (my_lane.get("batch_ids") or [None])[0]
                tasks = [pool_by_id[rid]] if rid and rid in pool_by_id else []
            else:
                tasks = [pool_by_id[my_lane["request_id"]]] if my_lane["request_id"] in pool_by_id else []
            for t2 in tasks:
                t2._pool_width = my_lane["width"]
                t2._pool_gpus = list(my_lane["gpus"])
            if tasks:
                # Per-lane local runqueue (stage 8): each step only advances tasks that still
                # have remaining steps, so a lane never over-steps its own request. When the
                # phase window exceeds a lane's remaining (local-queue mode), the lane simply
                # stops early and its rank(s) idle until the boundary. Width-1 denoise is
                # rank-local and width>1 uses the lane's own CP subgroup, so lanes advancing a
                # different number of steps never desync (they only re-meet at the barrier).
                for _ in range(phase_k_local):
                    active = [
                        t2 for t2 in tasks
                        if int(t2.buffer.current_step) < int(t2.req.params.num_inference_steps)
                    ]
                    if not active:
                        break
                    self._pool_denoise_lane(active)
                    lane_steps_run += 1
        denoise_ms = 0.0
        local_compute_ms = 0.0
        if instrument:
            self._pool_sync()
            now = time.perf_counter()
            local_compute_ms = (now - t) * 1000.0
            denoise_ms = local_compute_ms
            if my_lane is not None:  # only rounds where this rank actually stepped
                Timer.record("pool_denoise_ms", denoise_ms)
                Timer.record_event("pool_lane_local", {
                    "round_index": int(self._pool_round_index),
                    "rank": int(self.rank),
                    "lane_index": int(my_lane_idx if my_lane_idx is not None else -1),
                    "request_id": str(my_lane.get("request_id") or ""),
                    "batch_ids": list(my_lane.get("batch_ids") or []),
                    "width": int(my_lane["width"]),
                    "gpus": [int(g) for g in my_lane["gpus"]],
                    "planned_k": int(phase_k_local),
                    "steps_run": int(lane_steps_run),
                    "local_compute_ms": float(local_compute_ms),
                })
            t = now
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dist.barrier()
        barrier_ms = 0.0
        if instrument:
            now = time.perf_counter()
            barrier_ms = (now - t) * 1000.0
            Timer.record("pool_barrier_ms", barrier_ms)
            t = now
        # Feed the clean per-lane compute sample to the online cost model only AFTER
        # the mandatory world barrier. The tiny fixed-size all_gather is therefore not
        # an extra synchronization point on the denoise critical path; it reuses the
        # phase boundary all ranks have already reached.
        if self._pool_online_calibrate:
            self._pool_feed_calibrator(lanes, pool_by_id, denoise_ms, lane_steps_run, device)
            if instrument:
                t = self._pool_rec("pool_calibrate_ms", t)
        # 5. Latent bookkeeping after the phase.
        if self._pool_targeted_repl:
            # Targeted mode: the ranks that stepped each task now own its updated latent; no
            # world re-broadcast of the latent (a future phase migrates on demand). We DO
            # still sync the tiny current_step int so every rank agrees on progress for
            # planning + finished detection, then record the new owners. Report migration.
            self._pool_sync_steps(lanes, pool_by_id)
            self._pool_update_owners(lanes, pool_by_id)
            replicated_count, replicated_bytes = migrated_count, migrated_bytes
        else:
            # Fallback: restore the full replicated-latent invariant every phase.
            replicated_count, replicated_bytes = self._pool_replicate_latents(lanes, pool_by_id)
        if instrument:
            self._pool_sync()
            t = self._pool_rec("pool_replicate_ms", t)

        # 6. Retire finished tasks: each owner lane decodes locally and its leader
        #    returns the image to rank 0 for asynchronous persistence.
        self._pool_retire_finished(lanes)
        if instrument:
            self._pool_sync()
            self._pool_rec("pool_retire_ms", t)
            round_ms = (time.perf_counter() - round_t0) * 1000.0
            Timer.record("pool_round_ms", round_ms)
            if is_main:
                stepped = sum(len(ln["batch_ids"]) if ln["width"] == 1 else 1 for ln in lanes)
                lane_records = []
                for idx, ln in enumerate(lanes):
                    cfp = self._pool_lane_cfp(int(ln["width"]))
                    mode = f"sp{ln['width']}" if cfp == 1 else f"cfp{cfp}xcp{int(ln['width']) // cfp}"
                    lane_k = int(k_list[idx]) if idx < len(k_list) else int(phase_k)
                    lane_remaining = (
                        int(lane_remaining_before[idx]) if idx < len(lane_remaining_before)
                        else int(self._pool_lane_remaining(ln, pool_by_id))
                    )
                    lane_steps = int(min(max(0, lane_remaining), max(0, lane_k)))
                    step_ms = (
                        float(lane_step_ms_before[idx]) if idx < len(lane_step_ms_before)
                        else float(self._pool_lane_step_ms(ln, pool_by_id, DiffusionBackend.scheduler))
                    )
                    lane_records.append({
                        "lane_index": int(idx),
                        "request_id": str(ln.get("request_id") or ""),
                        "batch_ids": list(ln.get("batch_ids") or []),
                        "mode": mode,
                        "width": int(ln["width"]),
                        "gpus": [int(g) for g in ln["gpus"]],
                        "planned_k": lane_k,
                        "steps_run": lane_steps,
                        "pred_step_ms": step_ms,
                        "pred_work_ms": float(step_ms * lane_steps),
                    })
                Timer.record_event("pool_round", {
                    "round_index": self._pool_round_index,
                    "phase_k": int(phase_k),
                    "phase_k_by_lane": [int(x) for x in k_list],
                    "n_admit": n_admit,
                    "lane_count": len(lanes),
                    "layout_widths": [int(ln["width"]) for ln in lanes],
                    "used_gpus": int(sum(int(ln["width"]) for ln in lanes)),
                    "world_size": int(world_size),
                    "stepped_task_count": int(stepped),
                    "replicated_task_count": int(replicated_count),
                    "replicated_bytes": int(replicated_bytes),
                    "rank0_denoise_ms": denoise_ms,
                    "pool_barrier_ms": float(barrier_ms),
                    "pool_round_ms": float(round_ms),
                    "lanes": lane_records,
                    "placement_reuse_total": int(self._pool_placement_reuse),
                    "placement_change_total": int(self._pool_placement_change),
                })
            self._pool_round_index += 1
        return None

    def _pool_lane_tasks(self, lane: Dict[str, Any], pool_by_id: Dict[str, DiffusionTask]) -> List[DiffusionTask]:
        rid = lane["request_id"]
        task = pool_by_id.get(rid)
        return [task] if task is not None and task.buffer is not None else []

    def _pool_lane_remaining(self, lane: Dict[str, Any], pool_by_id: Dict[str, DiffusionTask]) -> int:
        """Remaining denoise steps for the lane's request."""
        rem = 0
        for task in self._pool_lane_tasks(lane, pool_by_id):
            rem = max(rem, int(task.req.params.num_inference_steps) - int(task.buffer.current_step))
        return rem

    def _pool_lane_step_ms(self, lane: Dict[str, Any], pool_by_id: Dict[str, DiffusionTask], scheduler) -> float:
        """Predicted per-step latency recorded beside the planner-owned K."""
        tasks = self._pool_lane_tasks(lane, pool_by_id)
        if not tasks or scheduler is None or not hasattr(scheduler, "_profile_for_shape"):
            return 0.0
        task = tasks[0]
        try:
            prof = scheduler._profile_for_shape(scheduler._task_shape(task))
            mode = _mode_for_width(int(lane["width"]))
            return float(prof.step_ms(mode, int(task.buffer.current_step), batch=1))
        except Exception:
            return 0.0

    def _pool_plan_layout(self, world_size: int):
        """rank-0 planner: returns ``(lanes, admit_tasks)``.

        ``lanes`` is a list of plain dicts (JSON-broadcastable) mapping a distinct
        request to a canonical, width-aligned contiguous GPU block; ``admit_tasks``
        are the pending ``DiffusionTask`` objects the plan wants to bring in-flight
        this round (their metadata is broadcast separately)."""
        scheduler = DiffusionBackend.scheduler
        # Synchronous rounds: every in-flight task is at a denoise-step boundary, so
        # all running lanes are resizable this round (no firm/mid-step lanes).
        running = [
            (t, int(getattr(t, "_pool_width", 1)), list(getattr(t, "_pool_gpus", [])), True)
            for t in self.pool_group
        ]
        inflight_ids = {t.task_id for t in self.pool_group}
        pending = [
            DiffusionTaskPool.pool[tid]
            for tid in DiffusionTaskPool.pending_task_ids()
            if tid not in inflight_ids and not DiffusionTaskPool.pool[tid].is_control_signal()
        ]
        plan = scheduler.plan_pool_round(running, pending, world_size)
        # Affinity hint: prefer rank 0's full memo of last round's blocks; fall back to the
        # per-task ``_pool_gpus`` only for lanes rank 0 has never placed (memo cold-start).
        prev_gpus = {
            t.task_id: list(
                self._pool_prev_placement.get(t.task_id)
                or getattr(t, "_pool_gpus", [])
                or []
            )
            for t in self.pool_group
        }
        lanes = self._pool_canonical_lanes(plan, world_size, prev_gpus)
        self._pool_last_predicted_spin_ms = float(getattr(plan, "predicted_spin_ms", 0.0) or 0.0)
        # Remember the concrete block for every lane so next round's affinity is correct for
        # ALL in-flight lanes (not just the one on GPU0). Drop retired requests.
        self._pool_prev_placement = {
            ln["request_id"]: list(ln["gpus"]) for ln in lanes
        }

        assigned: set[str] = set()
        for ln in lanes:
            assigned.add(ln["request_id"])
            assigned.update(ln["batch_ids"])
        admit_tasks = [
            DiffusionTaskPool.pool[tid]
            for tid in assigned
            if tid not in inflight_ids and tid in DiffusionTaskPool.pool
        ]
        return lanes, admit_tasks

    def _pool_log_layout(self, lanes: List[Dict[str, Any]], world_size: int) -> None:
        """rank-0 observability: log the chosen pool layout whenever it changes, so
        SP-when-idle vs request-level-DP-under-burst behaviour is visible in the run
        log (used to validate against the simulator)."""
        sig = tuple(sorted((ln["request_id"], ln["width"], tuple(ln["gpus"])) for ln in lanes))
        if sig == getattr(self, "_pool_last_layout_sig", None):
            return
        self._pool_last_layout_sig = sig
        used = sum(ln["width"] for ln in lanes)

        def _fmt(ln: Dict[str, Any]) -> str:
            cfp = self._pool_lane_cfp(ln["width"])
            # w=cfp*cp: show the realized (cfp,cp) decomposition for wide lanes.
            tag = f"sp{ln['width']}" if cfp == 1 else f"cfp{cfp}xcp{ln['width'] // cfp}"
            return f"{ln['request_id']}:{tag}@{ln['gpus']}"

        desc = ", ".join(_fmt(ln) for ln in lanes) or "(empty)"
        # Predicted barrier spin the planner foresees for this layout (0 unless barrier-aware
        # scoring is on): fast lanes idling at the world barrier while a straggler drains.
        spin_ms = float(getattr(self, "_pool_last_predicted_spin_ms", 0.0) or 0.0)
        logger.info(
            "[slo_elastic pool] layout: %s | lanes=%d used=%d/%d idle=%d pred_spin=%.0fms",
            desc, len(lanes), used, world_size, world_size - used, spin_ms,
        )

    def _pool_lane_shape_tokens(self, lane: Dict[str, Any], pool_by_id: Dict[str, Any]) -> int:
        """Image-token count for a lane's request shape (0 if unknown)."""
        task = pool_by_id.get(lane.get("request_id"))
        params = getattr(getattr(task, "req", None), "params", None)
        size = getattr(params, "size", None) if params is not None else None
        if not size:
            return 0
        try:
            adapter = DiffusionBackend.model_adapter
            args = DiffusionBackend.args
            if adapter is not None and args is not None:
                return int(adapter.latent_token_count(int(size[0]), int(size[1]), args))
            from chitu_diffusion.runtime.cost_model import latent_token_count

            model_name = str(getattr(getattr(args, "models", None), "name", "") or "Z-Image")
            return int(latent_token_count(int(size[0]), int(size[1]), model_name=model_name))
        except Exception:
            return 0

    def _pool_feed_calibrator(
        self,
        lanes: List[Dict[str, Any]],
        pool_by_id: Dict[str, Any],
        denoise_ms: float,
        steps_run: int,
        device: Any,
    ) -> None:
        """Feed the planner's online calibrator with this phase's *clean* per-step
        latency. Each rank reports ITS lane leader's ``(img_tokens, sp_degree,
        cfg_parallel, ms/step)``; a single fixed-size all_gather then gives rank 0
        every lane type (not just gpu-0's lane). The sample is the ``pool_denoise_ms``
        already measured before the world barrier, so it reflects pure lane compute
        (no barrier spin) and adds no cuda sync -- only ~32 bytes of collective."""
        world = get_world_group()
        world_size = world.group_size
        my_lane = next((ln for ln in lanes if self.rank in ln["gpus"]), None)
        tokens = sp_degree = cfg_parallel = 0
        measured = 0.0
        if (
            my_lane is not None and steps_run > 0
            and my_lane["gpus"] and self.rank == my_lane["gpus"][0]
        ):
            cfp = self._pool_lane_cfp(my_lane["width"])
            cfg_parallel = int(cfp)
            sp_degree = max(1, int(my_lane["width"]) // int(cfp))
            tokens = self._pool_lane_shape_tokens(my_lane, pool_by_id)
            if tokens > 0:
                measured = float(denoise_ms) / float(steps_run)
        local = torch.tensor(
            [float(tokens), float(sp_degree), float(cfg_parallel), float(measured)],
            dtype=torch.float64, device=device,
        )
        gathered = [torch.zeros_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        if self.rank != 0:
            return
        scheduler = DiffusionBackend.scheduler
        if scheduler is None or not hasattr(scheduler, "observe_step_latency"):
            return
        for g in gathered:
            tok, sp, cfp, ms = int(g[0].item()), int(g[1].item()), int(g[2].item()), float(g[3].item())
            if tok > 0 and ms > 0.0:
                scheduler.observe_step_latency(tok, sp, cfp, ms)
        # Periodic visibility: log the live calibration table so convergence (and the
        # roofline error it is correcting) is auditable from the run log.
        cal = getattr(scheduler, "calibrator", None)
        if cal is not None and self._pool_round_index % 8 == 0:
            snap = cal.snapshot()
            keys = ", ".join(
                f"{k}->{v['factor']:.2f}(n{v['n']})" for k, v in sorted(snap["per_key"].items())
            )
            logger.info(
                "[calibrate] round=%d global_g=%.3f | %s",
                self._pool_round_index, snap["global_g"], keys or "(warming up)",
            )

    def _pool_canonical_lanes(
        self, plan, world_size: int, prev_gpus: Optional[Dict[str, List[int]]] = None
    ) -> List[Dict[str, Any]]:
        """Re-map the policy's lane widths onto canonical, width-aligned contiguous
        GPU blocks (offset % width == 0), which the pre-warmed lane subgroups require.
        The policy chooses *which* request runs at *which* width; the engine owns the
        concrete (aligned) GPU placement.

        With placement affinity on (stage 2), a request that keeps its width is kept on
        its previous block when that block is still aligned and free, so it does not
        migrate; only genuinely new/resized/displaced lanes are packed widest-first."""
        widths_avail = sorted(lane_widths_available()) or [1]
        specs: List[Tuple[str, int, int]] = []
        for lane in plan.lanes:
            w = int(lane.width)
            if w not in widths_avail:
                le = [x for x in widths_avail if x <= w]
                w = max(le) if le else 1
            specs.append((lane.request_id, w, max(1, int(lane.planned_steps))))
        blocks = self._canonical_placement(
            [w for _, w, _ in specs],
            world_size,
            prev_blocks=[(prev_gpus or {}).get(rid, []) for rid, _, _ in specs]
            if self._pool_placement_affinity
            else None,
        )
        lanes: List[Dict[str, Any]] = []
        for (rid, w, planned_k), gpus in zip(specs, blocks):
            lanes.append({
                "request_id": rid,
                "width": w,
                "gpus": gpus,
                "batch_ids": [rid],
                "planned_k": planned_k,
            })
            if self._pool_placement_affinity:
                prev = (prev_gpus or {}).get(rid, [])
                if prev and list(prev) == list(gpus):
                    self._pool_placement_reuse += 1
                elif prev:
                    self._pool_placement_change += 1
        return lanes

    @staticmethod
    def _canonical_placement(
        widths: List[int], total: int, prev_blocks: Optional[List[List[int]]] = None
    ) -> List[List[int]]:
        """Place lanes on aligned blocks: a width-``w`` lane always lands on
        ``[offset, offset+w)`` with ``offset % w == 0``. Guaranteed to succeed when widths
        are divisors of ``total`` and their sum is <= ``total`` (the policy's invariant),
        giving the canonical partitions ({4},{2,2},{2,1,1},{1,1,1,1}).

        When ``prev_blocks`` is given (placement affinity), each lane whose previous block
        is still a valid, aligned, free width-``w`` block is pinned there first; the rest
        are packed widest-first onto the remaining free aligned blocks. If affinity pinning
        fragments the free space so a remaining lane cannot be placed, it falls back to the
        pure widest-first packing (which always succeeds under the policy's invariant)."""

        def _pack(pins: Optional[List[List[int]]]) -> Optional[List[List[int]]]:
            occupied = [False] * total
            result: List[Optional[List[int]]] = [None] * len(widths)
            if pins is not None:
                for i, prev in enumerate(pins):
                    w = max(1, int(widths[i]))
                    if (
                        prev
                        and len(prev) == w
                        and all(0 <= int(g) < total for g in prev)
                        and int(prev[0]) % w == 0
                        and list(prev) == list(range(int(prev[0]), int(prev[0]) + w))
                        and not any(occupied[int(g)] for g in prev)
                    ):
                        for g in prev:
                            occupied[int(g)] = True
                        result[i] = [int(g) for g in prev]
            order = sorted(
                (i for i in range(len(widths)) if result[i] is None),
                key=lambda i: -widths[i],
            )
            for i in order:
                w = max(1, int(widths[i]))
                placed = False
                for off in range(0, total, w):  # aligned offsets only
                    if off + w <= total and not any(occupied[off + j] for j in range(w)):
                        for j in range(w):
                            occupied[off + j] = True
                        result[i] = list(range(off, off + w))
                        placed = True
                        break
                if not placed:
                    return None
            return [r for r in result if r is not None]

        placed = _pack(prev_blocks) if prev_blocks is not None else None
        if placed is None:
            placed = _pack(None)
        if placed is None:
            raise RuntimeError(
                f"cannot canonically place lane widths={widths} on {total} GPUs"
            )
        return placed

    def _pool_broadcast_lanes(self, plan_blob: Optional[str], device) -> List[Dict[str, Any]]:
        """Broadcast the JSON lane layout from rank 0 to the whole world."""
        if self.rank == 0:
            payload = plan_blob.encode("utf-8")
            size_t = torch.tensor([len(payload)], dtype=torch.int64, device=device)
        else:
            size_t = torch.zeros(1, dtype=torch.int64, device=device)
        dist.broadcast(size_t, src=0)
        n = int(size_t.item())
        if self.rank == 0:
            buf = torch.tensor(list(payload), dtype=torch.uint8, device=device)
        else:
            buf = torch.zeros(n, dtype=torch.uint8, device=device)
        dist.broadcast(buf, src=0)
        blob = bytes(buf.detach().cpu().tolist()).decode("utf-8")
        return json.loads(blob)["lanes"]

    _DTYPE_CODES = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2, torch.float64: 3}
    _CODE_DTYPES = {0: torch.float32, 1: torch.bfloat16, 2: torch.float16, 3: torch.float64}

    def _broadcast_tensor_src0(self, tensor: Optional[torch.Tensor], dev) -> torch.Tensor:
        """Broadcast a tensor of dynamic shape/dtype from rank 0 to the world group."""
        world = get_world_group()
        grp = world.gpu_group
        if self.rank == 0:
            src = tensor.to(dev).contiguous()
            dcode = self._DTYPE_CODES.get(src.dtype, 0)
            hdr = torch.tensor([src.dim(), dcode], dtype=torch.int64, device=dev)
        else:
            hdr = torch.zeros(2, dtype=torch.int64, device=dev)
        dist.broadcast(hdr, src=0, group=grp)
        ndim, dcode = int(hdr[0].item()), int(hdr[1].item())
        dtype = self._CODE_DTYPES.get(dcode, torch.float32)
        if self.rank == 0:
            shape_t = torch.tensor(list(src.shape), dtype=torch.int64, device=dev)
        else:
            shape_t = torch.zeros(ndim, dtype=torch.int64, device=dev)
        dist.broadcast(shape_t, src=0, group=grp)
        shape = [int(x) for x in shape_t.tolist()]
        if self.rank == 0:
            buf = src.to(dtype)
        else:
            buf = torch.empty(shape, dtype=dtype, device=dev)
        dist.broadcast(buf, src=0, group=grp)
        return buf

    def _broadcast_int_list_src0(self, values: Optional[List[int]], dev) -> List[int]:
        """Broadcast a small list of ints from rank 0 to the world group."""
        world = get_world_group()
        grp = world.gpu_group
        if self.rank == 0:
            cnt = torch.tensor([len(values)], dtype=torch.int64, device=dev)
        else:
            cnt = torch.zeros(1, dtype=torch.int64, device=dev)
        dist.broadcast(cnt, src=0, group=grp)
        n = int(cnt.item())
        if self.rank == 0:
            vt = torch.tensor(list(values), dtype=torch.int64, device=dev)
        else:
            vt = torch.zeros(n, dtype=torch.int64, device=dev)
        dist.broadcast(vt, src=0, group=grp)
        return [int(x) for x in vt.tolist()]

    def _pool_text_encode(self, task: DiffusionTask) -> torch.Tensor:
        """Text-encode a freshly-admitted task at admission.

        Default (``CHITU_TEXT_ENCODE=rank0_gpu``, CFP-1, world>1): only rank 0 runs
        the GPU text encoder; the padded embeddings + length metadata are broadcast to the
        world so every rank ends up with the identical replicated buffer (keeping the task
        reassignable) WITHOUT the O(world_size) duplicated encode compute. ``all_ranks`` (or
        single-rank) falls back to every rank encoding identically."""
        world = get_world_group()
        if (
            self._text_encode_mode != "rank0_gpu"
            or world.group_size == 1
            or not DiffusionBackend.model_adapter.supports_rank0_text_encode()
        ):
            return self.text_encode_step(task)

        dev = self.local_rank
        # encode_negative decision (mirrors adapter.encode_text for cfg_size==1); evaluated
        # identically on every rank because buffer state is replicated at this point.
        try:
            guidance = float(DiffusionBackend.args.models.sampler.guidance_scale[0])
        except Exception:
            guidance = 1.0
        is_neg = task.buffer.text_embeddings is not None and guidance > 1.0

        tokens = self.text_encode_step(task) if self.rank == 0 else None
        if self.rank == 0:
            lengths = (
                task.buffer._z_negative_embedding_lengths
                if is_neg
                else task.buffer._z_text_embedding_lengths
            )
            self._text_encode_cache_miss += 1
        else:
            lengths = None
        tokens = self._broadcast_tensor_src0(tokens, dev)
        lengths = self._broadcast_int_list_src0(list(lengths) if lengths else [], dev)
        if self.rank != 0:
            if is_neg:
                task.buffer._z_negative_embedding_lengths = lengths
            else:
                task.buffer._z_text_embedding_lengths = lengths
        return tokens

    def _pool_admit_one(self, task: DiffusionTask) -> bool:
        """Drive a freshly-admitted task through text-encode -> denoise-ready at step
        0 (identical, deterministic work on every rank so its buffer is replicated),
        then append it to the in-flight pool group."""
        safety = 0
        while task.task_type in (DiffusionTaskType.TextEncode, DiffusionTaskType.VAEEncode):
            task.status = DiffusionTaskStatus.Running
            self._emit_stage_start_if_needed(task)
            if task.task_type == DiffusionTaskType.TextEncode:
                out = self._pool_text_encode(task)
            else:
                out = self.vae_encode_step(task)
            self._update_task_stage_and_buffer(task, out)
            safety += 1
            if safety > 8:
                raise RuntimeError(f"Task {task.task_id} stuck during pool admission.")
        if task.task_type != DiffusionTaskType.Denoise:
            raise RuntimeError(
                f"Task {task.task_id} reached {task.task_type} during admission; expected Denoise."
            )
        ready_tasks = self._fanout_denoise_ready(task)
        fanout = len(ready_tasks) > 1
        now = time.perf_counter_ns()
        for ready in ready_tasks:
            ready.status = DiffusionTaskStatus.Pending
            ready.sched_ts = now
            ready.last_scheduled_ts = now
            ready._pool_width = 1
            ready._pool_gpus = []
            # Freshly admitted: prepare_denoise ran identically on every rank, so
            # the step-0 latent is replicated world-wide (owner = all ranks).
            ready._pool_owner = set(range(get_world_group().group_size))
            self._emit_stage_start_if_needed(ready)
            self.pool_group.append(ready)
        if self.rank == 0:
            logger.debug(
                "[slo_elastic pool] admitted request=%s samples=%d (in-flight=%d)",
                task.task_id,
                len(ready_tasks),
                len(self.pool_group),
            )
        return fanout

    @Timer.get_timer("denoise")
    @amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    @torch.no_grad()
    def _pool_denoise_lane(self, tasks: List[DiffusionTask]) -> None:
        """Advance a single lane's task(s) by EXACTLY ONE denoise step on the lane's
        active CP subgroup (width>1 => one request sequence-sharded across the lane;
        width==1 => one request-level DP replica)."""
        for task in tasks:
            assert task.buffer.latents is not None and task.buffer.timesteps is not None
            setattr(task.buffer, "_dit_forward_call_index", 0)
        DiffusionBackend.model_adapter.denoise_step_group_once(
            tasks, self, DiffusionBackend, self._run_dit_forward,
        )
        for task in tasks:
            current_step = int(task.buffer.current_step)
            total_steps = int(task.req.params.num_inference_steps)
            timestep = (
                task.buffer.timesteps[current_step - 1]
                if task.buffer.timesteps is not None and current_step - 1 < len(task.buffer.timesteps)
                else None
            )
            log_progress(
                logger,
                stage_name=DiffusionTaskType.Denoise.name,
                task_id=task.task_id,
                step=current_step,
                total=total_steps,
                interval=self.denoise_progress_interval,
                timestep=timestep,
            )

    def _pool_replicate_latents(
        self, lanes: List[Dict[str, Any]], pool_by_id: Dict[str, DiffusionTask]
    ) -> Tuple[int, int]:
        """Re-establish the replicated-latent invariant after a round: each stepped
        task's latent + current_step is broadcast from its lane leader (min GPU rank)
        to the whole world, so any rank can host the task at any width next round.

        Returns ``(replicated_task_count, replicated_bytes)`` for instrumentation.

        Documented fidelity cost of the synchronous-round realization: one latent
        broadcast per stepped task per round over the world group (no NVLink on 4090).
        Lane order is derived from the (identically broadcast) plan, so every rank
        issues the same broadcast sequence."""
        world = get_world_group()
        world_size = world.group_size
        dev = self.local_rank
        replicated_count = 0
        replicated_bytes = 0
        for lane in lanes:
            # A full-width lane (width == N) already all-gathers its updated latent to
            # every rank inside the denoise forward, so every rank holds the identical
            # latent + current_step -> the world re-broadcast would be pure waste. Skip
            # it (this is the common SP-when-idle case and the biggest overhead saver).
            if lane["width"] >= world_size:
                continue
            leader = min(lane["gpus"])
            ids = lane["batch_ids"] if lane["width"] == 1 else [lane["request_id"]]
            for rid in ids:
                task = pool_by_id.get(rid)
                if task is None or task.buffer is None or task.buffer.latents is None:
                    continue
                replicated_bytes += self._pool_broadcast_task_latent(task, leader, dev, world)
                replicated_count += 1
        return replicated_count, replicated_bytes

    def _pool_broadcast_task_latent(self, task: DiffusionTask, leader: int, dev, world) -> int:
        """Broadcast one task's current_step + latent from ``leader`` to the whole world
        group (used by both the always-replicate fallback and targeted migration). Returns
        the number of latent bytes moved."""
        step_t = torch.tensor([int(task.buffer.current_step)], dtype=torch.int64, device=dev)
        dist.broadcast(step_t, src=leader, group=world.gpu_group)
        task.buffer.current_step = int(step_t.item())
        lat = task.buffer.latents.to(dev).contiguous().float()
        dist.broadcast(lat, src=leader, group=world.gpu_group)
        task.buffer.latents = lat
        return lat.numel() * lat.element_size() + step_t.numel() * step_t.element_size()

    def _pool_migrate_latents(
        self, lanes: List[Dict[str, Any]], pool_by_id: Dict[str, DiffusionTask]
    ) -> Tuple[int, int]:
        """Targeted pre-phase migration. For each lane, if the lane's ranks do not already
        all hold the request's current latent version (``owner``/valid ranks), move it in.
        Requests that keep their block (placement preservation) are skipped -- no data moves.

        Send the latent ONLY to missing ranks (``needed - valid``) via batched
        point-to-point; ``owner`` grows to ``valid | missing``. width 1->2 moves
        1 rank, 2->4 moves 2, and 2->1 moves nothing.

        Deterministic across ranks: every rank has the same ``lanes`` and the same owner
        metadata, so all ranks compute the identical (leader, missing) pairing and issue a
        matching send/recv sequence (no deadlock)."""
        return self._pool_migrate_latents_precise(lanes, pool_by_id)

    def _pool_migrate_latents_precise(
        self, lanes: List[Dict[str, Any]], pool_by_id: Dict[str, DiffusionTask]
    ) -> Tuple[int, int]:
        """Narrow point-to-point pre-phase migration (stage 6.5). Sends each task's current
        latent (+ step) from an owner leader ONLY to the ranks that lack it, batching all
        transfers into a single ``batch_isend_irecv`` for the phase. See ``_pool_migrate_latents``.

        All ranks iterate lanes/tasks in the same order and compute the same ``leader`` and
        ``missing`` sets from the replicated owner metadata, so the isend on the leader and
        the irecv on each missing rank match up; ranks not involved in a given transfer add
        no op. Metadata (``_pool_owner``) is advanced identically on every rank."""
        world = get_world_group()
        world_size = world.group_size
        dev = self.local_rank
        all_ranks = set(range(world_size))
        migrated_count = 0
        migrated_bytes = 0
        keepalive: List[torch.Tensor] = []  # send tensors must outlive the batched wait
        pending_recv: List[Tuple[DiffusionTask, torch.Tensor, torch.Tensor]] = []
        have_ops = False
        for lane in lanes:
            needed = {int(g) for g in lane["gpus"]}
            ids = lane["batch_ids"] if lane["width"] == 1 else [lane["request_id"]]
            for rid in ids:
                task = pool_by_id.get(rid)
                if task is None or task.buffer is None or task.buffer.latents is None:
                    continue
                valid = getattr(task, "_pool_owner", None) or all_ranks
                missing = sorted(needed - valid)
                if not missing:
                    continue  # every lane rank already holds the current latent
                leader = min(valid) if valid else min(needed)
                lat_ref = task.buffer.latents
                # bytes moved is deterministic across ranks (shape is fixed per request):
                per_rank_bytes = lat_ref.numel() * 4 + 8  # float32 latent + int64 step
                migrated_bytes += len(missing) * per_rank_bytes
                migrated_count += 1
                if self.rank == leader:
                    lat = lat_ref.to(dev).contiguous().float()
                    step_t = torch.tensor([int(task.buffer.current_step)], dtype=torch.int64, device=dev)
                    keepalive.extend([lat, step_t])
                    for m in missing:
                        world.p2p_isend(step_t, dst=m)
                        world.p2p_isend(lat, dst=m)
                        have_ops = True
                elif self.rank in missing:
                    step_r = world.p2p_irecv(size=(1,), dtype=torch.int64, src=leader)
                    lat_r = world.p2p_irecv(size=tuple(lat_ref.shape), dtype=torch.float32, src=leader)
                    have_ops = True
                    pending_recv.append((task, step_r, lat_r))
                task._pool_owner = set(valid) | set(missing)
        if have_ops:
            world.p2p_commit()
            world.p2p_wait()
        for task, step_r, lat_r in pending_recv:
            task.buffer.current_step = int(step_r.item())
            task.buffer.latents = lat_r
        keepalive.clear()
        return migrated_count, migrated_bytes

    def _pool_sync_steps(self, lanes: List[Dict[str, Any]], pool_by_id: Dict[str, DiffusionTask]) -> None:
        """Broadcast each stepped task's ``current_step`` (a single int, NOT the latent)
        from its lane leader to the world after a targeted phase.

        Rationale: in targeted mode the heavy latent stays only on the lane ranks, but
        ``current_step`` is advanced ONLY on those ranks by ``_pool_denoise_lane``. Every
        rank must still agree on progress, because rank 0's next-round planning (remaining
        steps) and deterministic finished-task detection in ``_pool_retire_finished``
        depend on it -- a divergent view makes ranks issue a different communication
        sequence and deadlock. This int-only sync is negligible cost. Lane order is the
        (identically broadcast) plan, so every rank issues the same broadcast sequence."""
        world = get_world_group()
        dev = self.local_rank
        for lane in lanes:
            leader = min(lane["gpus"])
            ids = lane["batch_ids"] if lane["width"] == 1 else [lane["request_id"]]
            for rid in ids:
                task = pool_by_id.get(rid)
                if task is None or task.buffer is None:
                    continue
                step_t = torch.tensor([int(task.buffer.current_step)], dtype=torch.int64, device=dev)
                dist.broadcast(step_t, src=leader, group=world.gpu_group)
                task.buffer.current_step = int(step_t.item())

    def _pool_update_owners(self, lanes: List[Dict[str, Any]], pool_by_id: Dict[str, DiffusionTask]) -> None:
        """After a phase, the ranks that stepped a task hold its updated latent; record
        them as the task's new owner set (width==1 => the single lane GPU; width==w => all
        w lane ranks, which all-gathered the updated latent inside the denoise forward)."""
        for lane in lanes:
            ranks = {int(g) for g in lane["gpus"]}
            ids = lane["batch_ids"] if lane["width"] == 1 else [lane["request_id"]]
            for rid in ids:
                task = pool_by_id.get(rid)
                if task is not None:
                    task._pool_owner = set(ranks)

    @staticmethod
    def _pool_lane_task_ids(lane: Dict[str, Any]) -> List[str]:
        if int(lane["width"]) == 1:
            return list(lane.get("batch_ids") or [lane["request_id"]])
        return [lane["request_id"]]

    def _pool_send_decoded_to_rank0(
        self,
        output: Optional[torch.Tensor],
        leader: int,
    ) -> Optional[torch.Tensor]:
        """Move one lane leader's decoded tensor to rank 0 without a world gather."""
        if leader == 0:
            return output if self.rank == 0 else None

        use_cpu = bool(DiffusionBackend.use_gloo)
        device = torch.device("cpu") if use_cpu else torch.device("cuda", self.local_rank)
        header_len = 8
        if self.rank == leader:
            if output is None:
                header = torch.full((header_len,), -1, dtype=torch.int64, device=device)
                dist.send(header, dst=0)
                return None
            payload = output.detach().to(device).contiguous()
            dtype_code = self._DTYPE_CODES.get(payload.dtype)
            if dtype_code is None or payload.ndim > header_len - 2:
                raise ValueError(
                    f"Unsupported decoded tensor dtype/shape: {payload.dtype} {tuple(payload.shape)}"
                )
            header = torch.full((header_len,), -1, dtype=torch.int64, device=device)
            header[0] = payload.ndim
            header[1] = dtype_code
            header[2 : 2 + payload.ndim] = torch.tensor(
                payload.shape, dtype=torch.int64, device=device
            )
            dist.send(header, dst=0)
            dist.send(payload, dst=0)
            return None
        if self.rank != 0:
            return None

        header = torch.empty(header_len, dtype=torch.int64, device=device)
        dist.recv(header, src=leader)
        ndim = int(header[0].item())
        if ndim < 0:
            return None
        dtype = self._CODE_DTYPES[int(header[1].item())]
        shape = tuple(int(x) for x in header[2 : 2 + ndim].tolist())
        payload = torch.empty(shape, dtype=dtype, device=device)
        dist.recv(payload, src=leader)
        return payload

    def _pool_retire_finished(self, lanes: List[Dict[str, Any]]) -> None:
        """Decode each finished task on its final denoise lane, then retire it."""
        finished = [
            t for t in self.pool_group
            if int(t.buffer.current_step) >= int(t.req.params.num_inference_steps)
        ]
        if not finished:
            return
        survivors = [t for t in self.pool_group if t not in finished]
        finished_by_id = {task.task_id: task for task in finished}
        lane_by_task = {
            task_id: lane
            for lane in lanes
            for task_id in self._pool_lane_task_ids(lane)
            if task_id in finished_by_id
        }
        missing = sorted(set(finished_by_id) - set(lane_by_task))
        if missing:
            raise RuntimeError(f"Finished tasks are missing their final lane: {missing}")

        local_outputs: Dict[str, Optional[torch.Tensor]] = {}
        for task in finished:
            task.task_type = DiffusionTaskType.VAEDecode
            task.status = DiffusionTaskStatus.Running
            self._emit_stage_start_if_needed(task)

        # Disjoint lanes execute VAE concurrently. Decode uses cfp=1 so the whole
        # lane forms one CP group even when denoise used cfp=2.
        for task_id, lane in lane_by_task.items():
            gpus = [int(rank) for rank in lane["gpus"]]
            if self.rank not in gpus:
                continue
            width = int(lane["width"])
            activate_lane_topology(
                LaneTopology(
                    offset=gpus[0],
                    width=width,
                    cfp_degree=1,
                    cp_degree=width,
                )
            )
            output = self.vae_decode_step(finished_by_id[task_id])
            if self.rank == min(gpus):
                local_outputs[task_id] = output

        reset_active_lane_topology()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dist.barrier()

        rank0_outputs: Dict[str, Optional[torch.Tensor]] = {}
        for task in finished:
            lane = lane_by_task[task.task_id]
            leader = min(int(rank) for rank in lane["gpus"])
            received = self._pool_send_decoded_to_rank0(
                local_outputs.get(task.task_id),
                leader,
            )
            if self.rank == 0:
                rank0_outputs[task.task_id] = received
        dist.barrier()

        for task in finished:
            self._update_task_stage_and_buffer(
                task,
                rank0_outputs.get(task.task_id) if self.rank == 0 else None,
            )
        self.pool_group = survivors

    def _step_single(self, task: Optional[DiffusionTask]) -> torch.Tensor:
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
        if (
            task.task_type == DiffusionTaskType.Denoise
            and int(getattr(task.req.params, "n_sample", 1) or 1) > 1
            and not bool(getattr(task, "_sample_fanout_done", False))
        ):
            ready_tasks = self._fanout_denoise_ready(task)
            self.current_task = ready_tasks[0]
            task = ready_tasks[0]
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
    
    def _ensure_async_output_worker(self) -> None:
        """Lazily start the rank-0 background output writer (bounded queue + daemon
        thread). Registered to flush at interpreter exit so no artifact is lost."""
        if self._async_output_queue is not None:
            return
        import queue as _queue
        import threading
        import atexit

        self._async_output_queue = _queue.Queue(maxsize=self._async_output_maxq)

        def _worker() -> None:
            q = self._async_output_queue
            while True:
                item = q.get()
                if item is None:
                    q.task_done()
                    break
                task, cpu_tokens = item
                try:
                    self._post_vae_decode(task, cpu_tokens)
                except Exception as exc:  # never let a bad save kill the writer
                    logger.exception("async output writer failed for task=%s: %s", task.task_id, exc)
                finally:
                    q.task_done()

        self._async_output_thread = threading.Thread(
            target=_worker, name="chitu-async-output", daemon=True
        )
        self._async_output_thread.start()
        atexit.register(self._flush_async_output)

    def _flush_async_output(self) -> None:
        """Drain the async output queue and stop the writer (idempotent)."""
        q = self._async_output_queue
        if q is None:
            return
        q.join()
        if self._async_output_thread is not None and self._async_output_thread.is_alive():
            q.put(None)
            q.join()

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
        # Startup warmup requests run the full compute path (to warm VAE + save kernels)
        # but their artifacts are throwaway -- don't litter the results dir with them.
        if str(task.task_id).startswith("__warmup__"):
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
        task._sample_denoise_ready = True

    def _fanout_denoise_ready(self, parent: DiffusionTask) -> List[DiffusionTask]:
        """Turn one encoded request into independently schedulable ``b=1`` tasks."""
        sample_count = max(1, int(getattr(parent.req.params, "n_sample", 1) or 1))
        if sample_count == 1:
            if not bool(getattr(parent, "_sample_denoise_ready", False)):
                self._pre_denoising(parent)
            return [parent]

        seed_fallback = int(DiffusionBackend.args.infer.seed)
        children = [
            DiffusionTask.make_sample_child(parent, sample_index, seed_fallback)
            for sample_index in range(sample_count)
        ]
        for child in children:
            self._pre_denoising(child)
            child.buffer.current_step = 0
        if self.rank == 0:
            DiffusionTaskPool.replace_with_children(parent.task_id, children)
        parent._sample_fanout_done = True
        return children

    def _update_task_stage_and_buffer(self, task: DiffusionTask, tokens: torch.Tensor):

        is_main_process = dist.get_rank() == 0
        has_img = task.req.init_image is not None
        
        if task.status == DiffusionTaskStatus.Running:
            task.status = DiffusionTaskStatus.Pending
        elif is_main_process:
            logger.warning(f"Task {task.task_id} - Status: {task.status}")
        
        # 处理Text Encode阶段
        if task.task_type == DiffusionTaskType.TextEncode:
            prepares_full_cfg = bool(
                getattr(
                    DiffusionBackend.model_adapter,
                    "text_encode_prepares_cfg_state",
                    lambda: False,
                )()
            )
            # Prepare the FULL cond+neg CFG state, replicated identically on every
            # rank, independent of the live lane's CFP degree. Adapters that pack
            # both branches in one encode (``text_encode_prepares_cfg_state``) set
            # the buffers directly; the generic two-pass path encodes the positive
            # prompt first and the negative second. The cfp1/cfp2 split is a
            # denoise-time decision in the CFG executor, so a task stays reassignable
            # to any topology and no rank is ever missing a branch.
            if prepares_full_cfg:
                if task.buffer.text_embeddings is None:
                    task.buffer.text_embeddings = tokens
            else:
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
                
            # Text Encode完成，转换到下一阶段
            self._emit_stage_end(DiffusionTaskType.TextEncode, task.task_id)
            if has_img:
                task.task_type = DiffusionTaskType.VAEEncode 
            else:
                task.task_type = DiffusionTaskType.Denoise
                if int(getattr(task.req.params, "n_sample", 1) or 1) == 1:
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
                if (
                    self._async_output
                    and torch.distributed.get_rank() == 0
                    and tokens is not None
                ):
                    # Hand the (CPU-moved) image to the background writer; do not block the
                    # phase on PNG encode + disk write. put() blocks when the queue is full
                    # (backpressure) to bound pending-output memory.
                    self._ensure_async_output_worker()
                    cpu_tokens = tokens.detach().to("cpu")
                    q_t0 = time.perf_counter()
                    self._async_output_queue.put((task, cpu_tokens))
                    Timer.record("cpu_output_queue_wait_ms", (time.perf_counter() - q_t0) * 1000.0)
                    Timer.record("async_output_backlog", float(self._async_output_queue.qsize()))
                else:
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
        
    
