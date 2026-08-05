# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, field, replace
from logging import getLogger
import os
import time
from typing import Any, List, Optional, Sequence, Tuple

from chitu_diffusion.runtime.cost_model import (
    CalibratedCostModel,
    RuntimeCalibrator,
    image_token_count,
)
from chitu_diffusion.runtime.task import (
    DiffusionTask,
    DiffusionTaskPool,
    DiffusionTaskStatus,
    DiffusionTaskType,
)
from chitu_diffusion.runtime.elastic_policy import (
    LaneAssignment,
    PoolRunning,
    RequestSpec,
    SimRequest,
    SimulationConfig,
    StepLatencyProfile,
    SwitchCostModel,
    load_cost_model,
    plan_balanced_lane_steps,
    plan_pool_layout,
    profile_from_cost_model,
)

logger = getLogger(__name__)


# One in-flight lane as seen by the pool scheduler: a task, the SP width it is currently
# running at, the GPU ranks it holds, and whether it is at a denoise-step boundary (and so
# resizable this round) or mid-step (firm, holding its GPUs).
RuntimeLane = Tuple[DiffusionTask, int, Sequence[int], bool]


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


@dataclass(frozen=True)
class SchedulingPlan:
    """A full pool layout for one denoise-step boundary (the ``slo_elastic`` runtime
    output).

    ``plan_pool_round`` produces one of these on rank 0 from the shared policy core; it
    is broadcast to the world and each rank finds its lane via :meth:`lane_for_rank`.
    Each :class:`~chitu_diffusion.runtime.elastic_policy.LaneAssignment` places one
    distinct request (``request_id``) on ``gpus`` at a given SP ``width`` (1 == a DP
    replica), so the pool runs heterogeneous, simultaneous per-request widths.
    """

    lanes: List[LaneAssignment] = field(default_factory=list)
    idle_gpus: List[int] = field(default_factory=list)
    deciding_field: Optional[str] = None
    objective: tuple = field(default_factory=tuple)
    total_gpus: int = 0
    predicted_spin_ms: float = 0.0

    def lane_for_rank(self, rank: int) -> Optional[LaneAssignment]:
        """The lane whose GPU set contains ``rank`` (None == this rank is idle)."""
        for lane in self.lanes:
            if rank in lane.gpus:
                return lane
        return None

    def lane_for_task(self, task_id: str) -> Optional[LaneAssignment]:
        for lane in self.lanes:
            if lane.request_id == task_id:
                return lane
        return None

    def assigned_task_ids(self) -> List[str]:
        return [lane.request_id for lane in self.lanes]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lanes": [asdict(lane) for lane in self.lanes],
            "idle_gpus": list(self.idle_gpus),
            "deciding_field": self.deciding_field,
            "objective": list(self.objective),
            "total_gpus": self.total_gpus,
            "predicted_spin_ms": self.predicted_spin_ms,
        }


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
        epe = getattr(args, "epe", None)
        # These are correctness properties of the synchronous EPE model, not optional
        # optimizations: the planner must see barriers and must use otherwise-idle GPUs
        # to widen an eligible tail straggler.
        self.straggler_widen = True
        self.barrier_aware = True
        self.execution_groups = self._build_execution_groups(args)

        # ---- slo_elastic pool-scheduler knobs (only used when scheduling_policy == slo_elastic) ----
        self.horizon_events = int(getattr(args, "horizon_events", 6) or 6)
        self.starvation_wait_ms = float(getattr(args, "starvation_ms", 30000.0) or 30000.0)
        self.fairness_beta = float(getattr(args, "fairness_beta", 3.0) or 3.0)
        self.enable_flexcache = bool(getattr(epe, "enable_flexcache", False))
        self.switch_total_ms = float(getattr(args, "switch_total_ms", 0.0) or 0.0)
        # Stage-7 residual runtime tax fed into the planner's completion predictions. Default
        # 0.0 => planner behaves exactly as before (and the offline simulator is unaffected);
        # set from measured instrumentation via env to make the planner account for the
        # unavoidable per-step lane sync + amortized phase-boundary tax (barrier/plan/retire).
        self.per_step_lane_cost_ms = float(os.getenv("CHITU_POOL_PER_STEP_COST_MS", "0") or 0.0)
        self.phase_boundary_cost_ms = float(os.getenv("CHITU_POOL_BOUNDARY_COST_MS", "0") or 0.0)
        # Phase length K used to amortize the boundary tax; mirrors the engine's admission bound.
        self.phase_boundary_steps = max(1, int(getattr(epe, "phase_max_steps", 5) or 5))
        self.balanced_k = bool(getattr(epe, "balanced_k", True))
        self.cost_model_name = getattr(args, "cost_model", None) or "rtx4090"
        # Per-lane CFG parallelism (CFP) pricing for the planner. When >=2 the served
        # model runs classifier-free guidance and the planner prices a widened lane's
        # first factor of two as (near-perfect) CFG parallelism before context
        # parallelism -- realizing the DP>CFP>CP scaling priority in its width choices.
        # Off (1) keeps legacy single-forward CP pricing. Fixed per model, so it is a
        # per-run env knob (guidance is a model-level config, not visible to the
        # EPE config. Only affects slo_elastic width
        # decisions; pure_dp/pure_sp are fixed-layout so their behaviour is unchanged.
        self.cfg_parallel_max = max(1, int(getattr(epe, "cfg_parallel_max", 2) or 2))
        self.planner_guidance_scale = 2.0 if self.cfg_parallel_max >= 2 else 1.0
        # Online cost-model calibration (opt-in). When on, the planner prices layouts
        # through a live EWMA correction fed by the engine's measured per-lane step
        # latencies -- avoiding roofline error (esp. small shapes) and hardware drift.
        self.online_calibrate = bool(getattr(epe, "online_calibration", True))
        # Cost model + per-shape step-latency profile cache; loaded lazily so non-pool
        # policies (fifo/etc.) pay nothing. Keyed by (width, height, num_steps).
        self._cost_model = None
        self._calibrator: Optional[RuntimeCalibrator] = None
        self._profile_cache: dict[tuple[int, int, int], StepLatencyProfile] = {}
        self._profile_cache_version = -1

        logger.info(
            "Initialized DiffusionScheduler: type=%s execution_groups=%s",
            self.scheduler_type,
            [group.group_id for group in self.execution_groups],
        )

    # Policies that drive the concurrent heterogeneous-lane pool engine. ``slo_elastic``
    # is the SLO-aware planner; ``pure_dp`` / ``pure_sp`` are fixed-layout baselines that
    # reuse the identical pool machinery (admission / per-lane denoise / world barrier /
    # latent re-replication / retire) and differ ONLY in the layout decision, so the
    # three-way comparison is apples-to-apples on one code path.
    POOL_POLICIES = frozenset({"slo_elastic", "pure_dp", "pure_sp"})

    @property
    def is_pool_policy(self) -> bool:
        """Whether this scheduler drives the concurrent heterogeneous-lane pool engine."""
        return self.scheduler_type in self.POOL_POLICIES

    @property
    def cost_model(self):
        if self._cost_model is None:
            base = load_cost_model(self.cost_model_name)
            logger.info("DiffusionScheduler loaded cost model: %s", self.cost_model_name)
            if self.online_calibrate:
                self._calibrator = RuntimeCalibrator(
                    base, guidance_scale=self.planner_guidance_scale, enabled=True,
                )
                self._cost_model = CalibratedCostModel(base, self._calibrator)
                logger.info("DiffusionScheduler online cost-model calibration ENABLED")
            else:
                self._cost_model = base
        return self._cost_model

    @property
    def calibrator(self) -> Optional[RuntimeCalibrator]:
        """Live online calibrator (or ``None`` when calibration is disabled)."""
        if self._cost_model is None:
            _ = self.cost_model  # trigger lazy init
        return self._calibrator

    def observe_step_latency(
        self, img_tokens: int, sp_degree: int, cfg_parallel: int, measured_step_ms: float
    ) -> None:
        """Feed one measured clean per-step latency to the online calibrator (no-op
        when calibration is disabled). Called by the engine on the planning rank."""
        cal = self.calibrator
        if cal is not None:
            cal.observe(img_tokens, sp_degree, cfg_parallel, measured_step_ms)

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

    # ------------------------------------------------------------------
    # slo_elastic pool round planning (request-level DP + per-request SP width)
    # ------------------------------------------------------------------
    @staticmethod
    def _task_shape(task: DiffusionTask) -> tuple[int, int, int]:
        """(width, height, num_steps) for cost-model profiling. Falls back to 1024^2/20."""
        params = getattr(getattr(task, "req", None), "params", None)
        if params is None or params.size is None:
            return (1024, 1024, 20)
        w, h = int(params.size[0]), int(params.size[1])
        steps = int(getattr(params, "num_inference_steps", None) or 20)
        return (w, h, steps)

    def _profile_for_shape(self, shape: tuple[int, int, int]) -> StepLatencyProfile:
        # When online calibration bumps its version (a correction factor materially
        # changed), drop the cached profiles so the next plan re-prices with the new
        # factors. Cheap: profiles are rebuilt from the cost model on demand.
        cal = self._calibrator
        ver = cal.version if cal is not None else 0
        if ver != self._profile_cache_version:
            self._profile_cache.clear()
            self._profile_cache_version = ver
        prof = self._profile_cache.get(shape)
        if prof is None:
            w, h, steps = shape
            prof = profile_from_cost_model(
                self.cost_model, name=f"{w}x{h}x{steps}", width=w, height=h, num_steps=steps,
                guidance_scale=self.planner_guidance_scale, cfg_parallel_max=self.cfg_parallel_max,
                model_name=str(getattr(getattr(self.args, "models", None), "name", "") or "Z-Image"),
            )
            self._profile_cache[shape] = prof
        return prof

    def _sim_request_for_task(
        self,
        task: DiffusionTask,
        now_ms: float,
        profiles: dict[str, StepLatencyProfile],
    ) -> SimRequest:
        """Build a policy-layer SimRequest mirroring one runtime task's live state."""
        shape = self._task_shape(task)
        prof = self._profile_for_shape(shape)
        shape_key = prof.name
        profiles.setdefault(shape_key, prof)
        _w, _h, steps = shape
        arrival_ms = float(task.arrival_ts) / 1e6
        # deadline_ms on the task is a *relative* budget after arrival; the policy wants an
        # absolute wall-clock deadline (compared against now_ms).
        rel_deadline = task.deadline_ms()
        abs_deadline = arrival_ms + float(rel_deadline) if rel_deadline is not None else None
        spec = RequestSpec(
            request_id=task.task_id,
            arrival_ms=arrival_ms,
            num_steps=steps,
            profile=shape_key,
            deadline_ms=abs_deadline,
        )
        current_step = int(task.buffer.current_step) if task.buffer is not None else 0
        req = SimRequest(spec=spec, current_step=current_step)
        return req

    def _pool_config(self, total_gpus: int) -> SimulationConfig:
        return SimulationConfig(
            total_gpus=max(1, int(total_gpus)),
            switch=SwitchCostModel(
                switch_cost_ms=self.switch_total_ms,
                switch_allowed_until_step=self.switch_allowed_until_step,
            ),
            horizon_events=self.horizon_events,
            starvation_wait_ms=self.starvation_wait_ms,
            fairness_beta=self.fairness_beta,
            enable_flexcache=self.enable_flexcache,
            per_step_lane_cost_ms=self.per_step_lane_cost_ms,
            phase_boundary_cost_ms=self.phase_boundary_cost_ms,
            phase_steps=self.phase_boundary_steps,
            balanced_k=self.balanced_k,
            straggler_widen=self.straggler_widen,
            barrier_aware=self.barrier_aware,
        )

    def plan_pool_round(
        self,
        running: Sequence[RuntimeLane],
        pending: Sequence[DiffusionTask],
        total_gpus: int,
        now_ms: Optional[float] = None,
    ) -> "SchedulingPlan":
        """Decide a full pool layout for the next denoise-step boundary.

        ``running`` are the in-flight lanes ``(task, width, gpus, at_boundary)``; lanes not
        at a boundary are treated as firm (they hold their GPUs this round). ``pending`` is
        the queue of not-yet-running tasks. Returns a :class:`SchedulingPlan` mapping each
        chosen task to GPU ranks + SP width (+ optional switch / FlexCache), broadcast to
        the world by the engine so every rank finds its lane.
        """
        if now_ms is None:
            now_ms = time.perf_counter_ns() / 1e6
        total = max(1, int(total_gpus))

        # Fixed-layout baselines share the whole pool engine but skip the SLO planner:
        #   pure_sp -> one request at a time on ALL GPUs (Ulysses SP=N, sequential);
        #   pure_dp -> up to N distinct requests, each on a single GPU (SP=1, concurrent).
        if self.scheduler_type in {"pure_dp", "pure_sp"}:
            return self._forced_pool_plan(running, pending, total)

        profiles: dict[str, StepLatencyProfile] = {}

        firm: list[PoolRunning] = []
        boundary: list[PoolRunning] = []
        for task, width, gpus, at_boundary in running:
            req = self._sim_request_for_task(task, now_ms, profiles)
            run = PoolRunning(
                req=req,
                width=int(width),
                gpus=[int(g) for g in gpus],
                step_end_ms=None if at_boundary else now_ms + self._solo_step_ms(req, int(width), profiles),
            )
            (boundary if at_boundary else firm).append(run)

        pending_reqs: list[SimRequest] = [
            self._sim_request_for_task(task, now_ms, profiles) for task in pending
        ]

        config = self._pool_config(total)
        layout = plan_pool_layout(firm, boundary, pending_reqs, profiles, config, now_ms=now_ms)
        return SchedulingPlan(
            lanes=list(layout.assignments),
            idle_gpus=list(layout.idle_gpus),
            deciding_field=layout.deciding_field,
            objective=tuple(layout.objective),
            total_gpus=total,
            predicted_spin_ms=float(layout.predicted_spin_ms),
        )

    def _forced_pool_plan(
        self,
        running: Sequence[RuntimeLane],
        pending: Sequence[DiffusionTask],
        total: int,
    ) -> "SchedulingPlan":
        """Fixed-layout baselines (no cost model, no SLO objective).

        ``pure_sp``: a single lane spanning all ``total`` GPUs (Ulysses SP=N). The
        in-flight request keeps the pool until it finishes, so requests are served
        strictly one-at-a-time -- the classic "everything at max SP" baseline.

        ``pure_dp``: every lane is width 1 (a DP replica); in-flight lanes always
        continue, and free GPUs are filled with the oldest pending requests, up to
        ``total`` concurrent single-GPU requests -- the classic request-level DP baseline.
        """
        inflight = [lane[0] for lane in running]
        inflight_ids = {t.task_id for t in inflight}
        pend = sorted(
            (t for t in pending if t.task_id not in inflight_ids),
            key=lambda t: (t.arrival_ts, t.task_id),
        )
        prev_width = {lane[0].task_id: int(lane[1]) for lane in running}

        lanes: list[LaneAssignment] = []
        if self.scheduler_type == "pure_sp":
            width = total
            chosen = inflight[0] if inflight else (pend[0] if pend else None)
            if chosen is not None:
                pw = prev_width.get(chosen.task_id, width)
                lanes.append(
                    LaneAssignment(
                        request_id=chosen.task_id,
                        width=width,
                        gpus=list(range(total)),
                        switched=pw != width,
                        prev_width=pw,
                    )
                )
        else:  # pure_dp
            ordered = inflight + pend
            for task in ordered[:total]:
                pw = prev_width.get(task.task_id, 1)
                lanes.append(
                    LaneAssignment(
                        request_id=task.task_id,
                        width=1,
                        gpus=[],  # engine assigns the concrete GPU via canonical placement
                        switched=pw != 1,
                        prev_width=pw,
                    )
                )

        # Fixed-layout baselines still use the planner-owned phase cadence so their
        # engine overhead is comparable to slo_elastic. Width selection remains fixed;
        # only the number of useful steps before the next heartbeat is planned here.
        task_by_id = {task.task_id: task for task in (*inflight, *pend)}
        profiles: dict[str, StepLatencyProfile] = {}
        sim_lanes: list[tuple[SimRequest, int]] = []
        now_ms = time.perf_counter_ns() / 1e6
        for lane in lanes:
            task = task_by_id.get(lane.request_id)
            if task is not None:
                try:
                    sim_lanes.append((self._sim_request_for_task(task, now_ms, profiles), int(lane.width)))
                except (AttributeError, TypeError, ValueError):
                    # Pure-logic callers may provide minimal task stubs; keep the
                    # conservative one-step default when phase inputs are unavailable.
                    sim_lanes = []
                    break
        if len(sim_lanes) == len(lanes) and lanes:
            remaining = [max(1, req.remaining_steps) for req, _width in sim_lanes]
            base_steps = min(self.phase_boundary_steps, min(remaining))
            if self.switch_allowed_until_step > 0:
                for req, _width in sim_lanes:
                    if req.current_step < self.switch_allowed_until_step:
                        base_steps = min(base_steps, self.switch_allowed_until_step - req.current_step)
            step_ms = [self._solo_step_ms(req, width, profiles) for req, width in sim_lanes]
            lane_steps = plan_balanced_lane_steps(
                step_ms,
                remaining,
                base_steps=base_steps,
                enabled=self.balanced_k,
            )
            lanes = [replace(lane, planned_steps=int(k)) for lane, k in zip(lanes, lane_steps)]

        used = sum(ln.width for ln in lanes)
        return SchedulingPlan(
            lanes=lanes,
            idle_gpus=list(range(used, total)),
            deciding_field=self.scheduler_type,
            objective=(),
            total_gpus=total,
        )

    @staticmethod
    def _solo_step_ms(req: SimRequest, width: int, profiles: dict[str, StepLatencyProfile]) -> float:
        prof = profiles[req.spec.profile]
        mode = {1: "dp1", 2: "cp2", 4: "cp4", 8: "cp8"}.get(int(width), "dp1")
        return prof.step_ms(mode, req.current_step)

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
        
