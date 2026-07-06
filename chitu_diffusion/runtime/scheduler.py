# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, field
from logging import getLogger
import os
import time
from typing import Any, List, Optional

from chitu_diffusion.runtime.task import (
    DiffusionTask,
    DiffusionTaskPool,
    DiffusionTaskStatus,
    DiffusionTaskType,
)

logger = getLogger(__name__)


@dataclass(frozen=True)
class ExecutionGroupProfile:
    """Static execution profile used by the serving scheduler.

    Runtime hotswitching is layered on top of these profiles later; the default
    profile keeps today's single global group behavior.
    """

    group_id: str = "world"
    ranks: tuple[int, ...] = ()
    cp_size: int = 1
    dp_replicas: int = 1
    mode: str = "single"


@dataclass(frozen=True)
class ScheduleDecision:
    task_id: str
    policy: str
    reason: str
    scheduled_ts: int
    execution_group_id: str = "world"
    shape_key: tuple = field(default_factory=tuple)
    stage: Optional[str] = None
    current_step: int = 0
    remaining_steps: Optional[int] = None
    hotswitch_action: Optional[str] = None

    def to_legacy_task_id(self) -> str:
        return self.task_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DiffusionScheduler:
    '''
    Diffusion scheduler with a backwards-compatible FIFO default.

    The new decision API carries policy, shape, step, and execution-group
    metadata while schedule() still returns the legacy list[str] expected by
    older call sites.
    '''
    
    @staticmethod
    def build(infer_args):
        logger.info(f"Building diffusion scheduler: {infer_args=}")
        return DiffusionScheduler(infer_args)
    
    def __init__(self, args):
        self.scheduling_ts = 0
        self.args = args
        self.scheduler_type = str(getattr(args, "scheduling_policy", "fifo") or "fifo")
        self.hotswitch_enabled = bool(getattr(args, "hotswitch_enabled", False))
        self.switch_allowed_until_step = int(getattr(args, "switch_allowed_until_step", 0) or 0)
        # M3 continuous batching: group same-shape, same-step, same-stage requests.
        self.continuous_batch = bool(getattr(args, "continuous_batch", False)) or (
            str(os.getenv("CHITU_CONTINUOUS_BATCH", "")).strip().lower() in {"1", "true", "yes", "on"}
        )
        self.max_batch_items = max(1, int(getattr(args, "max_batch_items", 8) or 8))
        self.execution_groups = self._build_execution_groups(args)
        logger.info(
            "Initialized DiffusionScheduler: type=%s execution_groups=%s",
            self.scheduler_type,
            [group.group_id for group in self.execution_groups],
        )

    def _build_execution_groups(self, args) -> list[ExecutionGroupProfile]:
        cp_size = max(1, int(getattr(args, "cp_size", 1) or 1))
        raw_profiles = getattr(args, "execution_profiles", None) or []
        profiles: list[ExecutionGroupProfile] = []
        for index, raw_profile in enumerate(raw_profiles):
            getter = raw_profile.get if isinstance(raw_profile, dict) else lambda key, default=None: getattr(raw_profile, key, default)
            profile_cp_size = max(1, int(getter("cp_size", cp_size) or cp_size))
            profiles.append(
                ExecutionGroupProfile(
                    group_id=str(getter("group_id", f"group_{index}")),
                    ranks=tuple(int(rank) for rank in (getter("ranks", ()) or ())),
                    cp_size=profile_cp_size,
                    dp_replicas=max(1, int(getter("dp_replicas", 1) or 1)),
                    mode=str(getter("mode", "cp" if profile_cp_size > 1 else "dp")),
                )
            )
        if profiles:
            return profiles
        return [
            ExecutionGroupProfile(
                group_id="world",
                cp_size=cp_size,
                dp_replicas=1,
                mode="cp" if cp_size > 1 else "single",
            )
        ]

    def _make_decision(self, task: DiffusionTask, *, policy: str, reason: str) -> ScheduleDecision:
        self.scheduling_ts = time.perf_counter_ns()
        task.sched_ts = self.scheduling_ts
        task.last_scheduled_ts = self.scheduling_ts
        group = self._select_execution_group_for_task(task)
        remaining_steps = task.remaining_denoise_steps()
        shape_key = task.shape_key()
        return ScheduleDecision(
            task_id=task.task_id,
            policy=policy,
            reason=reason,
            scheduled_ts=self.scheduling_ts,
            execution_group_id=group.group_id,
            shape_key=shape_key,
            stage=task.task_type.name if task.task_type is not None else None,
            current_step=int(task.buffer.current_step) if task.buffer is not None else 0,
            remaining_steps=remaining_steps,
            hotswitch_action=self._hotswitch_action(task),
        )

    def _select_execution_group_for_task(self, task: DiffusionTask) -> ExecutionGroupProfile:
        if len(self.execution_groups) == 1:
            return self.execution_groups[0]
        if self.scheduler_type in {"static_dp", "request_parallel"}:
            dp_groups = [
                group for group in self.execution_groups
                if group.mode in {"dp", "single"} or group.dp_replicas > 1
            ]
            if dp_groups:
                return min(dp_groups, key=lambda group: (group.cp_size, group.group_id))
        if task.remaining_denoise_steps() is not None:
            cp_groups = [group for group in self.execution_groups if group.cp_size > 1]
            if cp_groups:
                return max(cp_groups, key=lambda group: (group.cp_size, group.group_id))
        return self.execution_groups[0]

    def _hotswitch_action(self, task: DiffusionTask) -> Optional[str]:
        if not self.hotswitch_enabled:
            return None
        if self.scheduler_type not in {"early_hot_switch", "shape_aware_ratio"}:
            return None
        if task.task_type != DiffusionTaskType.Denoise or task.buffer is None:
            return None
        if int(task.buffer.current_step) > self.switch_allowed_until_step:
            return None
        if len(DiffusionTaskPool.pending_task_ids()) < 2:
            return None
        if task.scheduler_metadata.get("hotswitch_prepared"):
            return None
        return "cp_to_dp"

    def _pending_tasks(self) -> list[DiffusionTask]:
        return [DiffusionTaskPool.pool[task_id] for task_id in DiffusionTaskPool.pending_task_ids()]

    def _is_groupable(self, task: DiffusionTask) -> bool:
        """Whether a task may join a continuous-batch group (M3 scope: SP=1, same
        shape/step, no acceleration). Flexcache/DiTango and cp_size>1 fall back to
        the single-task path."""
        if task.is_control_signal():
            return False
        if int(getattr(self.args, "cp_size", 1) or 1) != 1:
            return False
        params = getattr(getattr(task, "req", None), "params", None)
        if params is None:
            return False
        if getattr(params, "flexcache", None) or getattr(params, "flexcache_params", None):
            return False
        return True

    @staticmethod
    def _group_key(task: DiffusionTask) -> tuple:
        """Batch key at task granularity: same shape + same denoise step + same
        stage may share one batched forward (mirrors WorkItem.batch_key + stage)."""
        return (
            task.shape_key(),
            int(task.buffer.current_step) if task.buffer is not None else 0,
            task.task_type,
        )

    def _form_group(self, primary: DiffusionTask, pending_tasks: list[DiffusionTask]) -> list[DiffusionTask]:
        key = self._group_key(primary)
        group: list[DiffusionTask] = []
        for task in pending_tasks:
            if len(group) >= self.max_batch_items:
                break
            if self._is_groupable(task) and self._group_key(task) == key:
                group.append(task)
        return group

    # ------------------------------------------------------------------
    # M4 mixed-step continuous batching: dynamic membership.
    #
    # Unlike M3 (which grouped only same-shape/same-*step* tasks that all start
    # together), M4 keeps a persistent in-flight denoise group and admits new
    # same-shape requests at step boundaries -- they join at step 0 while the
    # incumbents may be at any step. Membership is therefore decided by the
    # engine each round via ``select_cb_admissions``; the batch key is the
    # stage-INDEPENDENT request shape (``cb_shape_key``), because a fresh task
    # (TextEncode, no seq_len/image_size yet) must be comparable to an in-flight
    # task (Denoise, image_size populated). Steps are intentionally NOT part of
    # the key -- coexisting different steps in one forward is the point of M4.
    # ------------------------------------------------------------------

    @staticmethod
    def cb_shape_key(task: DiffusionTask) -> tuple:
        """Stage-independent request shape used to decide continuous-batch
        membership. Same size + frame_num + solver + num_inference_steps implies
        identical latent shape and timestep schedule after ``prepare_denoise``,
        so such tasks may share one batched forward regardless of their current
        stage or step."""
        if task.req is None or task.req.params is None:
            return ()
        params = task.req.params
        return (
            tuple(params.size) if params.size is not None else None,
            getattr(params, "frame_num", None),
            getattr(params, "sample_solver", None),
            getattr(params, "num_inference_steps", None),
        )

    def cb_is_groupable(self, task: DiffusionTask) -> bool:
        """Public alias of the M3 groupability gate (SP=1, no flexcache, not a
        control signal). Reused by the M4 engine's admission planner."""
        return self._is_groupable(task)

    def select_cb_admissions(
        self,
        cb_group: list[DiffusionTask],
        pending_tasks: list[DiffusionTask],
        capacity: int,
    ) -> list[DiffusionTask]:
        """Pick the next same-shape work to admit into the in-flight denoise
        group. If the group is non-empty its shape is the target; otherwise the
        FIFO-first groupable pending task sets the target shape. Already in-flight
        tasks (still ``Pending`` so ``can_schedule`` keeps the engine alive) are
        skipped via ``cb_group`` identity. Returns at most ``capacity`` tasks."""
        if capacity <= 0:
            return []
        inflight_ids = {task.task_id for task in cb_group}
        candidates = [
            task
            for task in pending_tasks
            if task.task_id not in inflight_ids and self.cb_is_groupable(task)
        ]
        if not candidates:
            return []
        if cb_group:
            target = self.cb_shape_key(cb_group[0])
        else:
            target = self.cb_shape_key(candidates[0])
        return [task for task in candidates if self.cb_shape_key(task) == target][:capacity]

    def _select_pending_task(self, pending_tasks: list[DiffusionTask]) -> Optional[DiffusionTask]:
        if not pending_tasks:
            return None
        if self.scheduler_type in {"step_interleave", "admission_control", "shape_aware_ratio"}:
            return min(
                pending_tasks,
                key=lambda task: (
                    task.last_scheduled_ts or -1,
                    task.arrival_ts,
                    task.task_id,
                ),
            )
        return pending_tasks[0]

    def schedule_decisions(self) -> list[ScheduleDecision]:
        shutdown_task = DiffusionTaskPool.get_shutdown_task()
        if shutdown_task is not None and shutdown_task.status == DiffusionTaskStatus.Pending:
            logger.info("Scheduling shutdown request with highest priority.")
            return [self._make_decision(shutdown_task, policy="control", reason="shutdown")]

        cancel_task = DiffusionTaskPool.get_cancel_task()
        if cancel_task is not None and cancel_task.status == DiffusionTaskStatus.Pending:
            logger.info(
                "Scheduling cancel request with high priority: task_id=%s reason=%s",
                cancel_task.task_id,
                cancel_task.signal_data.get("reason"),
            )
            return [self._make_decision(cancel_task, policy="control", reason="cancel")]

        if DiffusionTaskPool.is_empty():
            logger.info("DiffusionTaskPool is empty, returning empty task list.")
            return []

        pending_tasks = self._pending_tasks()
        if not pending_tasks:
            logger.info("No pending tasks available for scheduling.")
            return []

        selected_task = self._select_pending_task(pending_tasks)
        if selected_task is None:
            return []

        # M3: when continuous batching is on, try to form a same-shape/same-step
        # group headed by the FIFO-selected task and return one decision per group
        # member. The generator batches them into one transformer forward per step.
        if self.continuous_batch and self._is_groupable(selected_task):
            group = self._form_group(selected_task, pending_tasks)
            if len(group) >= 2:
                decisions = []
                for task in group:
                    decision = self._make_decision(
                        task,
                        policy=self.scheduler_type,
                        reason="continuous_batch_group",
                    )
                    task.scheduler_metadata["last_decision"] = decision.to_dict()
                    decisions.append(decision)
                logger.debug(
                    "Scheduled continuous-batch group of %d tasks: %s",
                    len(decisions),
                    [d.task_id for d in decisions],
                )
                return decisions

        decision = self._make_decision(
            selected_task,
            policy=self.scheduler_type,
            reason=f"{self.scheduler_type}_selected",
        )
        selected_task.scheduler_metadata["last_decision"] = decision.to_dict()
        logger.debug("Scheduled task decision: %s", decision)
        return [decision]

    def schedule(self) -> List[str]:
        """
        Backwards-compatible scheduling API.
        
        Returns:
            List[str]: task ids selected by the decision scheduler
        """
        return [decision.to_legacy_task_id() for decision in self.schedule_decisions()]

    def can_schedule(self) -> bool:
        """
        检查是否可以进行调度
        
        Returns:
            bool: 如果有待调度的任务则返回True
        """
        if DiffusionTaskPool.is_empty():
            return False
        if DiffusionTaskPool.has_shutdown_request():
            return True
        if DiffusionTaskPool.has_cancel_request():
            return True
        
        return bool(DiffusionTaskPool.pending_task_ids())

    def get_queue_length(self) -> int:
        """
        获取当前待调度任务数量
        
        Returns:
            int: 待调度任务数量
        """
        if DiffusionTaskPool.is_empty():
            return 0
        control_count = int(DiffusionTaskPool.has_shutdown_request()) + int(DiffusionTaskPool.has_cancel_request())
        
        pending_count = len(DiffusionTaskPool.pending_task_ids())
        
        return pending_count + control_count

    def get_scheduler_info(self) -> dict:
        """
        获取调度器信息
        
        Returns:
            dict: 包含调度器状态信息的字典
        """
        return {
            "scheduler_type": self.scheduler_type,
            "total_tasks": len(DiffusionTaskPool.pool),
            "pending_tasks": self.get_queue_length(),
            "can_schedule": self.can_schedule(),
            "last_scheduling_ts": self.scheduling_ts,
            "execution_groups": [asdict(group) for group in self.execution_groups],
        }
        
