"""SLO-aware elastic CP/DP pool scheduler core (torch-free).

Single source of truth for the ``slo_elastic`` layout policy, shared by:

- the offline simulator ``experiments/cp_dp_hot_switch/simulate.py`` (which imports
  these symbols and drives them through an event loop), and
- the runtime scheduler ``chitu_diffusion.runtime.scheduler`` (which builds request
  state from live tasks, calls :func:`plan_pool_layout`, and enacts the returned
  layout on real GPUs).

The module is intentionally torch-free -- it depends only on the torch-free cost
model in :mod:`chitu_diffusion.runtime.cost_model`. Every decision is made at a
denoise-step boundary: the pool of ``total_gpus`` GPUs is partitioned into lanes of
width in {1, 2, 4} (SP degree; width 1 == a DP replica), each lane runs one distinct
request (request-level DP), and layouts may switch at step boundaries. The objective
is a lexicographic SLO-first vector scored by a rolling-horizon forward simulation
(see ``slo_elastic_scheduler_strategy.md``).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from chitu_diffusion.runtime.cost_model import (
    CostModel,
    default_cost_model,
    image_token_count,
)

# Committed per-hardware calibrations live alongside the offline experiment; the runtime
# defaults to the RTX 4090 model there (the 4-GPU no-NVLink box the scheduler targets).
_REPO_ROOT = Path(__file__).resolve().parents[2]
COST_MODELS_DIR = _REPO_ROOT / "experiments" / "cp_dp_hot_switch" / "cost_models"


def resolve_cost_model_path(name_or_path: str | Path | None) -> Path | None:
    """Resolve a cost-model name/path to a JSON file (or None to use the analytical model).

    Accepts an explicit file path, or a bare hardware name looked up in the committed
    ``experiments/cp_dp_hot_switch/cost_models/`` dir (e.g. ``rtx4090`` ->
    ``cost_models/rtx4090.json``).
    """
    if name_or_path is None:
        return None
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    named = COST_MODELS_DIR / f"{name_or_path}.json"
    if named.is_file():
        return named
    named_raw = COST_MODELS_DIR / str(name_or_path)
    if named_raw.is_file():
        return named_raw
    return None


def load_cost_model(name_or_path: str | Path | None = None) -> CostModel:
    """Load a profiled cost model by name/path, falling back to the analytical model."""
    path = resolve_cost_model_path(name_or_path)
    if path is not None and path.is_file():
        return CostModel.load(path)
    return default_cost_model()


# --------------------------------------------------------------------------- #
# Shared data structures
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StepLatencyProfile:
    """Per-step latency table for a request shape and execution mode."""

    name: str
    dp_1gpu_step_ms: list[float]
    cp_2gpu_step_ms: list[float]
    cp_4gpu_step_ms: list[float] | None = None
    dp_batch_step_ms: dict[int, list[float]] = field(default_factory=dict)
    seq_len: int = 0
    resolution: str = "unknown"

    def step_ms(self, mode: str, step_index: int, *, batch: int = 1) -> float:
        if mode == "dp1":
            values = self.dp_batch_step_ms.get(int(batch), self.dp_1gpu_step_ms) if batch > 1 else self.dp_1gpu_step_ms
        elif mode == "cp2":
            values = self.cp_2gpu_step_ms
        elif mode == "cp4":
            values = self.cp_4gpu_step_ms or self.cp_2gpu_step_ms
        else:
            raise ValueError(f"unknown execution mode {mode!r}")
        if not values:
            raise ValueError(f"profile {self.name!r} has no latency values for mode {mode}")
        return float(values[min(step_index, len(values) - 1)])


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_ms: float
    num_steps: int
    profile: str
    priority: int = 0
    deadline_ms: float | None = None


@dataclass
class SimRequest:
    spec: RequestSpec
    current_step: int = 0
    started_ms: float | None = None
    completed_ms: float | None = None
    current_mode: str | None = None
    switched: bool = False
    switch_count: int = 0
    flexcache_reduced_steps: int = 0
    continuous_batch_count: int = 0
    # ``effective_steps`` is the (possibly FlexCache-reduced) step budget; it starts
    # equal to the request's original ``num_steps`` and only ``slo_elastic`` shrinks it.
    effective_steps: int | None = None
    degree_timeline: list[tuple[int, int]] = field(default_factory=list)  # (step_index, width)

    def __post_init__(self) -> None:
        if self.effective_steps is None:
            self.effective_steps = self.spec.num_steps

    @property
    def original_steps(self) -> int:
        return self.spec.num_steps

    @property
    def remaining_steps(self) -> int:
        return max(0, int(self.effective_steps) - self.current_step)

    @property
    def is_done(self) -> bool:
        return self.current_step >= int(self.effective_steps)


@dataclass(frozen=True)
class SwitchCostModel:
    switch_cost_ms: float = 0.0
    graph_recapture_cost_ms: float = 0.0
    cache_migration_cost_ms: float = 0.0
    communicator_cost_ms: float = 0.0
    latent_migration_cost_ms: float = 0.0
    switch_allowed_until_step: int = 4

    @property
    def total_ms(self) -> float:
        return (
            self.switch_cost_ms
            + self.graph_recapture_cost_ms
            + self.cache_migration_cost_ms
            + self.communicator_cost_ms
            + self.latent_migration_cost_ms
        )


@dataclass(frozen=True)
class SimulationConfig:
    total_gpus: int = 2
    admission_wait_ms: float = 50.0
    long_request_seq_len: int = 4096
    switch: SwitchCostModel = field(default_factory=SwitchCostModel)
    # --- slo_elastic knobs (ignored by the other policies) ---
    horizon_events: int = 6  # rolling-horizon depth (step completions) for layout scoring
    starvation_wait_ms: float = 30000.0  # queue wait after which a request counts as starved
    fairness_beta: float = 3.0  # target max slowdown (soft; used only for logging)
    enable_flexcache: bool = False  # allow the simulator-only emergency step-reduction action
    enable_continuous_batch: bool = False  # let pending same-shape DP requests join at step boundaries
    max_batch_items: int = 8  # cap for one continuous-batch DP lane
    flexcache_min_steps: int = 4  # never reduce a request below this many executed steps
    flexcache_max_reduce_steps: int = 8  # max steps a single FlexCache action may drop
    flexcache_emergency_tardiness_ms: float = 1.0  # predicted tardiness above which FlexCache unlocks
    flexcache_emergency_slack_ms: float = 2000.0  # negative-slack magnitude that also unlocks FlexCache
    decision_log: bool = False  # record a per-decision explanation trail
    # --- residual runtime overhead (stage 7 cost model) ---
    # After the granularity optimizations (multi-step phase, placement preservation, targeted
    # replication) the ONLY runtime cost the planner still ignores is the *residual*,
    # unavoidable per-step lane sync and the amortized per-phase-boundary tax. These default
    # to 0.0 so the offline simulator and every existing test/scenario are byte-for-byte
    # unchanged; the runtime scheduler fills them from measured instrumentation. Migration
    # cost is NOT included here -- it is already modeled per switch via ``switch.total_ms``.
    per_step_lane_cost_ms: float = 0.0     # unavoidable per-denoise-step lane sync tax
    phase_boundary_cost_ms: float = 0.0    # plan_publish + boundary_barrier + retire_sync, per phase boundary
    phase_steps: int = 0                   # phase length K used to amortize the boundary tax (0 => no boundary term)


@dataclass
class PoolSegment:
    """One executed unit on a set of GPUs: a denoise step or a switch."""

    request_id: str
    start_ms: float
    end_ms: float
    step_index: int
    width: int
    gpus: list[int]
    event: str = "step"  # "step" | "switch"
    reason: str = ""
    batch_size: int = 1
    request_ids: list[str] = field(default_factory=list)


@dataclass
class PoolRunning:
    req: SimRequest
    width: int
    gpus: list[int]
    step_end_ms: float | None  # None == between steps (a boundary), needs (re)start
    members: list[SimRequest] = field(default_factory=list)


_WIDTH_TO_MODE = {1: "dp1", 2: "cp2", 4: "cp4"}


def _mode_for_width(width: int) -> str:
    return _WIDTH_TO_MODE.get(int(width), "dp1")


def _profile_for(req: SimRequest, profiles: dict[str, StepLatencyProfile]) -> StepLatencyProfile:
    try:
        return profiles[req.spec.profile]
    except KeyError as exc:
        raise ValueError(f"request {req.spec.request_id!r} references missing profile {req.spec.profile!r}") from exc


def _run_members(run: PoolRunning) -> list[SimRequest]:
    if not run.members:
        run.members = [run.req]
    return run.members


def _refresh_run_leader(run: PoolRunning) -> None:
    members = _run_members(run)
    if members:
        run.req = members[0]


def _largest_valid_width(cap: int, max_width: int = 4) -> int:
    """Largest width in {1,2,4} that is <= min(cap, max_width); 0 if cap < 1."""
    best = 0
    for w in (1, 2, 4):
        if w <= cap and w <= max_width:
            best = w
    return best


def _shape_aware_degree(profile: StepLatencyProfile, *, max_k: int = 4, threshold: float = 0.15) -> int:
    """Largest SP degree whose per-step latency beats the next-lower degree by > threshold.

    Uses the profile's per-mode step latency (which itself comes from the cost
    model), so the choice self-tunes per hardware: on a no-NVLink box the comm
    term keeps short/low-res shapes at k=1 automatically.
    """
    k = 1
    for nxt in (2, 4):
        if nxt > max_k:
            break
        cur = profile.step_ms(_mode_for_width(k), 0)
        upgraded = profile.step_ms(_mode_for_width(nxt), 0)
        if upgraded < cur * (1.0 - threshold):
            k = nxt
        else:
            break
    return k


def _pool_start_step(
    now_ms: float,
    run: PoolRunning,
    profiles: dict[str, StepLatencyProfile],
    segments: list[PoolSegment],
    reason: str,
    *,
    switched: bool,
    switch_total_ms: float,
) -> None:
    """Start (or restart) a denoise step for ``run`` at its current width."""
    req = run.req
    profile = _profile_for(req, profiles)
    members = _run_members(run)
    batch_members = members if run.width == 1 else [req]
    batch_size = len(batch_members)
    request_ids = [member.spec.request_id for member in batch_members]
    start_ms = now_ms
    if switched:
        req.switched = True
        req.switch_count += 1
        segments.append(
            PoolSegment(
                request_id=req.spec.request_id,
                start_ms=now_ms,
                end_ms=now_ms + switch_total_ms,
                step_index=req.current_step,
                width=run.width,
                gpus=list(run.gpus),
                event="switch",
                reason=reason,
                batch_size=1,
                request_ids=[req.spec.request_id],
            )
        )
        start_ms = now_ms + switch_total_ms
    for member in batch_members:
        if member.started_ms is None:
            member.started_ms = now_ms
        member.current_mode = _mode_for_width(run.width)
        member.degree_timeline.append((member.current_step, run.width))
    req.current_mode = _mode_for_width(run.width)
    step_ms = profile.step_ms(req.current_mode, req.current_step, batch=batch_size)
    end_ms = start_ms + step_ms
    segments.append(
        PoolSegment(
            request_id=req.spec.request_id,
            start_ms=start_ms,
            end_ms=end_ms,
            step_index=req.current_step,
            width=run.width,
            gpus=list(run.gpus),
            event="batch_step" if batch_size > 1 else "step",
            reason=reason,
            batch_size=batch_size,
            request_ids=request_ids,
        )
    )
    run.step_end_ms = end_ms


# --------------------------------------------------------------------------- #
# slo_elastic: SLO-aware, lexicographic, rolling-horizon scheduler + FlexCache
# --------------------------------------------------------------------------- #
# Lexicographic objective (compared field-by-field, earlier fields dominate). See
# slo_elastic_scheduler_strategy.md section 5.
SLO_OBJECTIVE_FIELDS = (
    "slo_miss_count",
    "max_tardiness_ms",
    "total_tardiness_ms",
    "max_slowdown",
    "starvation_count",
    "flexcache_used_count",
    "flexcache_reduced_steps_total",
    "switch_count",
    "total_switch_cost_ms",
    "total_flow_ms",  # sum of predicted (completion - arrival); minimizing this is latency/throughput-optimal
    "mean_slowdown",
)


def _solo_service_ms(req: SimRequest, profiles: dict[str, StepLatencyProfile]) -> float:
    """Single-GPU DP service time -- the fairness baseline for slowdown (strategy §3)."""
    return req.original_steps * _profile_for(req, profiles).step_ms("dp1", 0)


def _slo_shape_deg(req: SimRequest, profiles: dict[str, StepLatencyProfile], max_k: int) -> int:
    return _shape_aware_degree(_profile_for(req, profiles), max_k=max_k)


def _slo_step_ms(req: SimRequest, width: int, profiles: dict[str, StepLatencyProfile]) -> float:
    return _profile_for(req, profiles).step_ms(_mode_for_width(width), req.current_step)


def _rem_steps(req: SimRequest, override: dict[str, int] | None) -> int:
    eff = override.get(req.spec.request_id, req.effective_steps) if override else req.effective_steps
    return max(0, int(eff) - req.current_step)


def _residual_overhead_ms(config: SimulationConfig, rem_steps: int) -> float:
    """Stage-7 residual runtime tax the compute cost model does not capture: the
    unavoidable per-denoise-step lane sync plus the per-phase-boundary tax (barrier /
    plan publish / retire sync) amortized over the phase length ``phase_steps``.

    Returns 0.0 whenever the (default) config leaves these at 0, so the offline
    simulator and existing tests are unaffected. Migration cost is intentionally
    excluded -- it is already modeled per switch via ``switch.total_ms``."""
    if rem_steps <= 0:
        return 0.0
    overhead = rem_steps * config.per_step_lane_cost_ms
    if config.phase_boundary_cost_ms > 0.0 and config.phase_steps > 0:
        boundaries = math.ceil(rem_steps / config.phase_steps)
        overhead += boundaries * config.phase_boundary_cost_ms
    return overhead


def _slo_urgency(req: SimRequest, now_ms: float, profiles: dict[str, StepLatencyProfile], max_k: int,
                 override: dict[str, int] | None = None) -> tuple[int, float]:
    """Ordering key: deadline-bearing requests by smallest slack first, then SRPT."""
    prof = _profile_for(req, profiles)
    best_step = min(prof.step_ms(_mode_for_width(k), 0) for k in (1, 2, 4) if k <= max_k)
    rem_time = _rem_steps(req, override) * best_step
    if req.spec.deadline_ms is None:
        return (1, rem_time)
    return (0, req.spec.deadline_ms - now_ms - rem_time)


def _slo_key_pending(pending: list[SimRequest], now_ms: float, profiles: dict[str, StepLatencyProfile],
                     max_k: int, limit: int = 5) -> list[SimRequest]:
    ranked = sorted(pending, key=lambda r: _slo_urgency(r, now_ms, profiles, max_k))
    return ranked[:limit]


def _slo_admit_continuous_batches(now_ms: float, boundary: list[PoolRunning], pending: list[SimRequest],
                                  profiles: dict[str, StepLatencyProfile], config: SimulationConfig) -> int:
    """Greedily admit urgent same-shape pending requests into DP lanes at step boundaries.

    This models continuous batching as a fine-grained admission action: a request no
    longer has to wait for a whole in-flight request to finish, only for a compatible
    DP lane's next denoise-step boundary. We keep it conservative: same profile,
    width=1, and capped batch size.
    """
    if not config.enable_continuous_batch or config.max_batch_items <= 1:
        return 0
    total_admitted = 0
    max_k = min(4, max(1, config.total_gpus))
    for run in boundary:
        if run.width != 1:
            continue
        members = _run_members(run)
        if len(members) >= config.max_batch_items:
            continue
        profile = run.req.spec.profile
        candidates = [
            req for req in pending
            if req.spec.profile == profile and req.current_step == 0
        ]
        candidates.sort(key=lambda r: _slo_urgency(r, now_ms, profiles, max_k))
        while candidates and len(members) < config.max_batch_items:
            req = candidates.pop(0)
            if req not in pending:
                continue
            cur_batch = len(members)
            new_batch = cur_batch + 1
            prof = _profile_for(req, profiles)
            cur_step = prof.step_ms("dp1", req.current_step, batch=cur_batch)
            new_step = prof.step_ms("dp1", req.current_step, batch=new_batch)
            solo_step = prof.step_ms("dp1", req.current_step, batch=1)
            earliest_lane_free = min(member.remaining_steps for member in members) * cur_step
            before: list[tuple[SimRequest, float]] = [
                (member, now_ms + member.remaining_steps * cur_step)
                for member in members
            ]
            before.append((req, now_ms + earliest_lane_free + req.remaining_steps * solo_step))
            after = [
                (member, now_ms + member.remaining_steps * new_step)
                for member in (*members, req)
            ]

            def slo_score(items: list[tuple[SimRequest, float]]) -> tuple[int, float]:
                miss = 0
                tard = 0.0
                for item_req, comp in items:
                    dl = item_req.spec.deadline_ms
                    if dl is None:
                        continue
                    item_tard = max(0.0, comp - dl)
                    tard += item_tard
                    if item_tard > 1e-9:
                        miss += 1
                return miss, tard

            miss_before, tard_before = slo_score(before)
            miss_after, tard_after = slo_score(after)
            if miss_after > miss_before or tard_after >= tard_before - 1e-9:
                continue
            pending.remove(req)
            req.continuous_batch_count += 1
            members.append(req)
            total_admitted += 1
        run.members = members
        _refresh_run_leader(run)
    return total_admitted


def _gpu_partitions(capacity: int, max_parts: int) -> list[tuple[int, ...]]:
    """All non-increasing width multisets from {1,2,4} with sum <= ``capacity`` and at
    most ``max_parts`` parts (>=1). Sums below capacity are kept so a lone SP-negative
    request can run at width 1 and leave GPUs idle rather than being forced into SP;
    the work-conserving filter in the layout builder discards under-packing when there
    is still pending demand. Small for G<=8, so enumeration is cheap."""
    results: set[tuple[int, ...]] = set()

    def rec(remaining: int, max_w: int, parts: list[int]) -> None:
        if parts:
            results.add(tuple(parts))
        if len(parts) >= max_parts:
            return
        for w in (4, 2, 1):
            if w <= remaining and w <= max_w:
                rec(remaining - w, w, parts + [w])

    if capacity > 0 and max_parts > 0:
        rec(capacity, 4, [])
    return sorted(results, key=lambda t: (-len(t), t))


def _slo_candidate_layouts(now_ms: float, firm: list[PoolRunning], boundary: list[PoolRunning],
                           pending: list[SimRequest], profiles: dict[str, StepLatencyProfile],
                           config: SimulationConfig) -> list[dict[str, Any]]:
    total = max(1, config.total_gpus)
    max_k = min(4, total)
    window = float(config.switch.switch_allowed_until_step)
    # Batched DP lanes stay at width=1 until their current members drain; reshaping a
    # mixed-membership batch into SP would require ragged SP batching, which is out of scope.
    beyond = [r for r in boundary if r.req.current_step > window or len(_run_members(r)) > 1]
    flexb = [r for r in boundary if r not in beyond]  # width is a decision variable
    fixed_used = sum(r.width for r in firm) + sum(r.width for r in beyond)
    reassignable = total - fixed_used

    must = [{"kind": "boundary", "run": r, "req": r.req, "prev": r.width} for r in flexb]
    key_pending = _slo_key_pending(pending, now_ms, profiles, max_k, limit=5)
    kcells = [{"kind": "pending", "run": None, "req": r, "prev": 0} for r in key_pending]
    assignable = must + kcells
    max_parts = max(1, len(assignable))

    layouts: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for parts in _gpu_partitions(reassignable, max_parts):
        num = len(parts)
        if num < len(must):
            continue  # cannot run every in-flight (boundary) request
        # Work-conserving: only leave GPUs idle once every assignable request is placed
        # (otherwise a request would wait while a GPU sits idle).
        if sum(parts) < reassignable and num < len(assignable):
            continue
        chosen = must + kcells[: num - len(must)]
        widths = sorted(parts, reverse=True)
        # Give the widest SP groups to the most SP-friendly / heaviest requests; DP-friendly
        # and short requests naturally fall to width 1.
        order = sorted(
            chosen,
            key=lambda c: (-_slo_shape_deg(c["req"], profiles, max_k), -c["req"].remaining_steps, c["req"].spec.arrival_ms),
        )
        assign = [
            {"kind": c["kind"], "run": c["run"], "req": c["req"], "prev": c["prev"], "width": w}
            for c, w in zip(order, widths)
        ]
        switch_count = sum(1 for a in assign if a["kind"] == "boundary" and a["width"] != a["prev"])
        key = tuple(sorted((a["req"].spec.request_id, a["width"]) for a in assign))
        if key in seen:
            continue
        seen.add(key)
        layouts.append({"assign": assign, "beyond": beyond, "switch_count": switch_count})

    if not layouts:  # e.g. no reassignable capacity: keep flexb where they are
        assign = [{"kind": "boundary", "run": r, "req": r.req, "prev": r.width, "width": r.width} for r in flexb]
        layouts.append({"assign": assign, "beyond": beyond, "switch_count": 0})
    return layouts


def _slo_items_for_layout(now_ms: float, firm: list[PoolRunning], layout: dict[str, Any],
                          pending: list[SimRequest], profiles: dict[str, StepLatencyProfile],
                          config: SimulationConfig, override: dict[str, int] | None = None):
    """Predicted (start, completion, width) for everything running under this layout."""
    switch_total = config.switch.total_ms
    active: list[dict[str, Any]] = []
    chosen: set[str] = set()

    def add_run_members(run: PoolRunning, start: float, first_step_end: float, width: int) -> None:
        members = _run_members(run) if width == 1 else [run.req]
        batch = len(members) if width == 1 else 1
        lane = f"run:{id(run)}"
        for member in members:
            rem_after = max(0, _rem_steps(member, override) - 1)
            step_ms = _profile_for(member, profiles).step_ms(_mode_for_width(width), member.current_step, batch=batch)
            comp = first_step_end + rem_after * step_ms + _residual_overhead_ms(config, _rem_steps(member, override))
            active.append({
                "id": member.spec.request_id,
                "req": member,
                "width": width,
                "lane": lane,
                "start": member.started_ms if member.started_ms is not None else start,
                "comp": comp,
            })
            chosen.add(member.spec.request_id)

    for run in firm:
        add_run_members(run, now_ms, run.step_end_ms, run.width)
    for run in layout["beyond"]:
        batch = len(_run_members(run)) if run.width == 1 else 1
        first_step_end = now_ms + _profile_for(run.req, profiles).step_ms(_mode_for_width(run.width), run.req.current_step, batch=batch)
        add_run_members(run, now_ms, first_step_end, run.width)
    for a in layout["assign"]:
        req = a["req"]
        switched = a["kind"] == "boundary" and a["width"] != a["prev"]
        start = now_ms + (switch_total if switched else 0.0)
        comp = (
            start
            + _rem_steps(req, override) * _slo_step_ms(req, a["width"], profiles)
            + _residual_overhead_ms(config, _rem_steps(req, override))
        )
        active.append({
            "id": req.spec.request_id,
            "req": req,
            "width": a["width"],
            "lane": f"req:{req.spec.request_id}",
            "start": start,
            "comp": comp,
        })
        chosen.add(req.spec.request_id)
    waiting = [r for r in pending if r.spec.request_id not in chosen]
    return active, waiting


def _slo_forward(now_ms: float, active: list[dict[str, Any]], waiting: list[SimRequest], total: int,
                 profiles: dict[str, StepLatencyProfile], config: SimulationConfig,
                 override: dict[str, int] | None = None) -> dict[str, tuple[float, float, int]]:
    """List-schedule the whole remaining queue to completion with a load-aware fallback.

    Fallback: when GPUs free, place the most urgent waiting request; if the queue is at
    least as deep as the free GPUs, place it as DP (width 1) to maximize concurrency
    (this is what makes the objective prefer DP under saturation and SP when idle)."""
    max_k = min(4, total)
    completions: dict[str, tuple[float, float, int]] = {
        it["id"]: (it["start"], it["comp"], it["width"]) for it in active
    }
    lanes: dict[str, tuple[float, int]] = {}
    for it in active:
        lane = str(it.get("lane", it["id"]))
        comp, width = lanes.get(lane, (0.0, int(it["width"])))
        lanes[lane] = (max(comp, float(it["comp"])), width)
    free = total - sum(width for _comp, width in lanes.values())
    heap = [(comp, width) for comp, width in lanes.values()]
    heapq.heapify(heap)
    queue = sorted(waiting, key=lambda r: _slo_urgency(r, now_ms, profiles, max_k, override))
    t = now_ms
    guard = 0
    while queue:
        guard += 1
        if guard > 200000:
            break
        while queue and free > 0:
            width = 1 if len(queue) >= free else _largest_valid_width(min(_slo_shape_deg(queue[0], profiles, max_k), free), max_width=max_k)
            if width < 1:
                break
            req = queue.pop(0)
            comp = (
                t
                + _rem_steps(req, override) * _slo_step_ms(req, width, profiles)
                + _residual_overhead_ms(config, _rem_steps(req, override))
            )
            completions[req.spec.request_id] = (t, comp, width)
            heapq.heappush(heap, (comp, width))
            free -= width
        if not queue or not heap:
            break
        comp, width = heapq.heappop(heap)
        t = max(t, comp)
        free += width
    return completions


def _slo_objective(now_ms: float, completions: dict[str, tuple[float, float, int]],
                   by_id: dict[str, SimRequest], config: SimulationConfig,
                   profiles: dict[str, StepLatencyProfile], switch_count: int,
                   flex_used: int, flex_steps: int) -> tuple[float, ...]:
    slo_miss = 0
    tardiness: list[float] = []
    slowdowns: list[float] = []
    starvation = 0
    total_flow = 0.0
    for rid, (start, comp, _width) in completions.items():
        req = by_id[rid]
        dl = req.spec.deadline_ms
        tard = max(0.0, comp - dl) if dl is not None else 0.0
        tardiness.append(tard)
        if dl is not None and comp > dl + 1e-9:
            slo_miss += 1
        solo = _solo_service_ms(req, profiles)
        slowdowns.append((comp - req.spec.arrival_ms) / solo if solo > 0 else 0.0)
        if start - req.spec.arrival_ms > config.starvation_wait_ms:
            starvation += 1
        total_flow += comp - req.spec.arrival_ms
    # Quantize the continuous SLO/fairness fields that rank *above* switch_count so that
    # only a *meaningful* predicted improvement (not myopic prediction noise) is worth a
    # switch. Big wins (e.g. saturated 1024: DP cuts tardiness by seconds) survive; sub-
    # bucket differences collapse to a tie and the switch-count field then keeps things put.
    def _q(value: float, bucket: float) -> float:
        return round(value / bucket) * bucket

    return (
        float(slo_miss),
        _q(max(tardiness, default=0.0), 500.0),
        _q(sum(tardiness), 500.0),
        _q(max(slowdowns, default=0.0), 0.25),
        float(starvation),
        float(flex_used),
        float(flex_steps),
        float(switch_count),
        float(switch_count) * config.switch.total_ms,
        total_flow,
        mean(slowdowns) if slowdowns else 0.0,
    )


def _slo_extreme_risk(completions: dict[str, tuple[float, float, int]], by_id: dict[str, SimRequest],
                      config: SimulationConfig) -> bool:
    for rid, (start, comp, _w) in completions.items():
        dl = by_id[rid].spec.deadline_ms
        if dl is None:
            continue
        if comp - dl > config.flexcache_emergency_tardiness_ms:
            return True
        if (dl - comp) < -config.flexcache_emergency_slack_ms:
            return True
    return False


def _slo_flexcache_augment(now_ms: float, firm: list[PoolRunning], layout: dict[str, Any],
                           pending: list[SimRequest], profiles: dict[str, StepLatencyProfile],
                           config: SimulationConfig, by_id: dict[str, SimRequest]):
    """Emergency-only step reduction. Greedily shrink the worst-missing requests by the
    smallest amount that helps, re-scoring each time. Returns (obj, layout, comps, actions)
    or ``None`` if FlexCache cannot improve the objective."""
    override: dict[str, int] = {}
    actions: dict[str, int] = {}

    def score(ovr: dict[str, int]):
        active, waiting = _slo_items_for_layout(now_ms, firm, layout, pending, profiles, config, ovr)
        comps = _slo_forward(now_ms, active, waiting, max(1, config.total_gpus), profiles, config, ovr)
        flex_used = sum(1 for _ in actions)
        flex_steps = sum(actions.values())
        return _slo_objective(now_ms, comps, by_id, config, profiles, layout["switch_count"], flex_used, flex_steps), comps

    best_obj, best_comps = score(override)
    improved = True
    while improved:
        improved = False
        # worst-missing request first
        misses = sorted(
            (
                (rid, comp)
                for rid, (_s, comp, _w) in best_comps.items()
                if by_id[rid].spec.deadline_ms is not None and comp - by_id[rid].spec.deadline_ms > config.flexcache_emergency_tardiness_ms
            ),
            key=lambda x: x[1] - by_id[x[0]].spec.deadline_ms,
            reverse=True,
        )
        for rid, _comp in misses:
            req = by_id[rid]
            floor = max(config.flexcache_min_steps, req.current_step)
            current_eff = override.get(rid, req.effective_steps)
            if current_eff <= floor:
                continue
            best_local = None
            for reduce in range(1, config.flexcache_max_reduce_steps + 1):
                new_eff = max(floor, current_eff - reduce)
                if new_eff >= current_eff:
                    break
                trial = dict(override)
                trial[rid] = new_eff
                trial_actions = dict(actions)
                trial_actions[rid] = trial_actions.get(rid, 0) + (current_eff - new_eff)
                active, waiting = _slo_items_for_layout(now_ms, firm, layout, pending, profiles, config, trial)
                comps = _slo_forward(now_ms, active, waiting, max(1, config.total_gpus), profiles, config, trial)
                obj = _slo_objective(now_ms, comps, by_id, config, profiles, layout["switch_count"],
                                     sum(1 for _ in trial_actions), sum(trial_actions.values()))
                if obj < best_obj:
                    best_local = (obj, comps, new_eff, current_eff - new_eff)
                    break  # smallest reduction that helps this request
            if best_local is not None:
                obj, comps, new_eff, reduced = best_local
                override[rid] = new_eff
                actions[rid] = actions.get(rid, 0) + reduced
                best_obj, best_comps = obj, comps
                improved = True
                break
    if not actions:
        return None
    action_list = [{"req": by_id[rid], "new_effective": override[rid], "reduced": actions[rid]} for rid in actions]
    return best_obj, layout, best_comps, action_list


def _build_start_plan(layout: dict[str, Any], running: list[PoolRunning],
                      pending: list[SimRequest], *, persist: bool) -> list[dict[str, Any]]:
    """Build the ordered (re)start plan for a chosen layout.

    When ``persist`` is True (simulator), newly-assigned pending requests become
    persistent ``PoolRunning`` lanes appended to ``running`` and removed from
    ``pending``. When False (runtime planning query), the new lanes are ephemeral and
    the caller's ``running``/``pending`` are left untouched.
    """
    to_start: list[dict[str, Any]] = []
    for run in layout["beyond"]:
        to_start.append({"run": run, "width": run.width, "prev": run.width, "new": False})
    for a in layout["assign"]:
        if a["kind"] == "boundary":
            to_start.append({"run": a["run"], "width": a["width"], "prev": a["prev"], "new": False})
        else:
            run = PoolRunning(req=a["req"], width=a["width"], gpus=[], step_end_ms=None)
            if persist:
                running.append(run)
                if a["req"] in pending:
                    pending.remove(a["req"])
            to_start.append({"run": run, "width": a["width"], "prev": 0, "new": True})
    return to_start


def _assign_gpu_indices(to_start: list[dict[str, Any]], running: list[PoolRunning], total: int) -> None:
    """Assign concrete GPU indices to each lane in ``to_start`` (mutates run.gpus/width).

    Firm runs (mid-step, ``step_end_ms`` not None) keep their GPUs reserved; boundary
    lanes reuse their previous GPUs when still free (affinity) before taking new ones.
    """
    reserved = {i for r in running if r.step_end_ms is not None for i in r.gpus}  # firm hold their GPUs
    free_idx = [i for i in range(total) if i not in reserved]
    for entry in to_start:
        run = entry["run"]
        width = entry["width"]
        chosen: list[int] = []
        if not entry["new"]:
            for i in run.gpus:
                if i in free_idx and len(chosen) < width:
                    chosen.append(i)
                    free_idx.remove(i)
        while len(chosen) < width and free_idx:
            chosen.append(free_idx.pop(0))
        run.gpus = chosen
        run.width = width


def _slo_apply_layout(now_ms: float, layout: dict[str, Any], profiles: dict[str, StepLatencyProfile],
                      config: SimulationConfig, segments: list[PoolSegment],
                      running: list[PoolRunning], pending: list[SimRequest]) -> None:
    total = max(1, config.total_gpus)
    switch_total = config.switch.total_ms
    to_start = _build_start_plan(layout, running, pending, persist=True)
    _assign_gpu_indices(to_start, running, total)
    for entry in to_start:
        switched = (not entry["new"]) and (entry["width"] != entry["prev"])
        _pool_start_step(now_ms, entry["run"], profiles, segments, "slo_elastic",
                         switched=switched, switch_total_ms=switch_total)


@dataclass
class PlanDecision:
    """Result of scoring candidate layouts (before any state mutation)."""

    layout: dict[str, Any]
    comps: dict[str, tuple[float, float, int]]
    obj: tuple[float, ...]
    runner_up: tuple[float, ...] | None
    flex_actions: list[dict[str, Any]]
    cb_admitted: int
    deciding_field: str | None


def _slo_plan(now_ms: float, running: list[PoolRunning], firm: list[PoolRunning],
              boundary: list[PoolRunning], pending: list[SimRequest],
              profiles: dict[str, StepLatencyProfile], config: SimulationConfig) -> PlanDecision:
    """Enumerate candidate layouts, score each by rolling-horizon forward simulation, and
    pick the lexicographic best. Continuous-batch admission (which mutates ``boundary``
    membership and ``pending``) is applied when the pool is otherwise saturated. FlexCache
    is evaluated as data only -- no ``effective_steps`` mutation happens here."""
    total = max(1, config.total_gpus)
    free_capacity = total - sum(r.width for r in firm) - sum(r.width for r in boundary)
    cb_admitted = 0
    if free_capacity <= 0:
        cb_admitted = _slo_admit_continuous_batches(now_ms, boundary, pending, profiles, config)
    layouts = _slo_candidate_layouts(now_ms, firm, boundary, pending, profiles, config)
    by_id = {r.spec.request_id: r for r in (*(member for run in running for member in _run_members(run)), *pending)}

    scored: list[tuple[tuple[float, ...], dict[str, Any], dict[str, tuple[float, float, int]]]] = []
    for layout in layouts:
        active, waiting = _slo_items_for_layout(now_ms, firm, layout, pending, profiles, config)
        comps = _slo_forward(now_ms, active, waiting, total, profiles, config)
        obj = _slo_objective(now_ms, comps, by_id, config, profiles, layout["switch_count"], 0, 0)
        scored.append((obj, layout, comps))
    scored.sort(key=lambda x: x[0])
    best_obj, best_layout, best_comps = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else None

    flex_actions: list[dict[str, Any]] = []
    if config.enable_flexcache and _slo_extreme_risk(best_comps, by_id, config):
        aug = _slo_flexcache_augment(now_ms, firm, best_layout, pending, profiles, config, by_id)
        if aug is not None and aug[0] < best_obj:
            best_obj, best_layout, best_comps, flex_actions = aug

    deciding = None
    if runner_up is not None:
        for idx, (a, b) in enumerate(zip(best_obj, runner_up)):
            if abs(a - b) > 1e-9:
                deciding = SLO_OBJECTIVE_FIELDS[idx]
                break
    return PlanDecision(best_layout, best_comps, best_obj, runner_up, flex_actions, cb_admitted, deciding)


def _slo_elastic_schedule(now_ms: float, running: list[PoolRunning], firm: list[PoolRunning],
                          boundary: list[PoolRunning], pending: list[SimRequest],
                          profiles: dict[str, StepLatencyProfile], config: SimulationConfig,
                          segments: list[PoolSegment], decision_log: list[dict[str, Any]] | None) -> None:
    """Simulator entry point: choose a layout, log it, apply FlexCache + start steps."""
    decision = _slo_plan(now_ms, running, firm, boundary, pending, profiles, config)

    if decision_log is not None and config.decision_log:
        decision_log.append({
            "now_ms": now_ms,
            "chosen_layout": [(a["req"].spec.request_id, a["width"]) for a in decision.layout["assign"]],
            "chosen_objective": list(decision.obj),
            "runner_up_objective": list(decision.runner_up) if decision.runner_up is not None else None,
            "deciding_field": decision.deciding_field,
            "switch_count": decision.layout["switch_count"],
            "continuous_batch_admitted": decision.cb_admitted,
            "flexcache_actions": [(a["req"].spec.request_id, a["reduced"]) for a in decision.flex_actions],
        })

    for act in decision.flex_actions:
        req = act["req"]
        req.effective_steps = int(act["new_effective"])
        req.flexcache_reduced_steps += int(act["reduced"])

    _slo_apply_layout(now_ms, decision.layout, profiles, config, segments, running, pending)


# --------------------------------------------------------------------------- #
# Cost-model derived step-latency profiles
# --------------------------------------------------------------------------- #
def profile_from_cost_model(
    model: CostModel,
    *,
    name: str,
    width: int,
    height: int,
    num_steps: int,
) -> StepLatencyProfile:
    """Build a step-latency profile for one shape by querying the cost model.

    Each execution mode maps to a sequence-parallel degree: dp1 -> k=1 (a whole
    request on one GPU), cp2 -> k=2, cp4 -> k=4. Steps are near-uniform for DiT so
    the per-step value is repeated across ``num_steps``.
    """
    img_tokens = image_token_count(width, height)
    dp1 = model.predict_step_ms(img_tokens=img_tokens, batch=1, sp_degree=1)
    cp2 = model.predict_step_ms(img_tokens=img_tokens, batch=1, sp_degree=2)
    cp4 = model.predict_step_ms(img_tokens=img_tokens, batch=1, sp_degree=4)
    dp_batch = {
        batch: [model.predict_step_ms(img_tokens=img_tokens, batch=batch, sp_degree=1)] * num_steps
        for batch in range(2, 17)
    }
    return StepLatencyProfile(
        name=name,
        seq_len=img_tokens,
        resolution=f"{width}x{height}",
        dp_1gpu_step_ms=[dp1] * num_steps,
        cp_2gpu_step_ms=[cp2] * num_steps,
        cp_4gpu_step_ms=[cp4] * num_steps,
        dp_batch_step_ms=dp_batch,
    )


# --------------------------------------------------------------------------- #
# Runtime-facing pure planner
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LaneAssignment:
    """One lane the runtime should execute this round: a distinct request on a set of
    GPU indices at a given SP width (1 == DP replica)."""

    request_id: str
    width: int
    gpus: list[int]
    switched: bool
    prev_width: int
    batch_request_ids: list[str]  # >1 only for a continuous-batch DP lane (width==1)
    flexcache_new_effective: int | None = None
    flexcache_reduced: int = 0


@dataclass(frozen=True)
class PoolLayoutPlan:
    """A full pool layout decision for one denoise-step boundary."""

    assignments: list[LaneAssignment]
    idle_gpus: list[int]
    deciding_field: str | None
    objective: tuple[float, ...]
    continuous_batch_admitted: int


def plan_pool_layout(firm: list[PoolRunning], boundary: list[PoolRunning],
                     pending: list[SimRequest], profiles: dict[str, StepLatencyProfile],
                     config: SimulationConfig, now_ms: float = 0.0) -> PoolLayoutPlan:
    """Pure query used by the runtime: given current firm (mid-step, GPU-holding) lanes,
    boundary (resizable) lanes, and the pending queue, return the chosen layout as plain
    data (which request runs on which GPUs at which width, plus optional switch/FlexCache).

    Does not start steps or emit segments. ``boundary``/``pending`` may be mutated only by
    continuous-batch admission (which is itself an action the runtime enacts); the runtime
    passes per-round snapshot lists so this is safe.
    """
    total = max(1, config.total_gpus)
    running = list(firm) + list(boundary)
    decision = _slo_plan(now_ms, running, firm, boundary, pending, profiles, config)
    to_start = _build_start_plan(decision.layout, running, pending, persist=False)
    _assign_gpu_indices(to_start, running, total)

    flex_by_id = {a["req"].spec.request_id: a for a in decision.flex_actions}
    assignments: list[LaneAssignment] = []
    used: set[int] = set()
    for entry in to_start:
        run = entry["run"]
        rid = run.req.spec.request_id
        members = _run_members(run) if run.width == 1 else [run.req]
        fa = flex_by_id.get(rid)
        assignments.append(LaneAssignment(
            request_id=rid,
            width=run.width,
            gpus=list(run.gpus),
            switched=(not entry["new"]) and entry["width"] != entry["prev"],
            prev_width=entry["prev"],
            batch_request_ids=[m.spec.request_id for m in members],
            flexcache_new_effective=int(fa["new_effective"]) if fa else None,
            flexcache_reduced=int(fa["reduced"]) if fa else 0,
        ))
        used.update(run.gpus)

    reserved = {g for r in firm for g in r.gpus}
    idle = [i for i in range(total) if i not in used and i not in reserved]
    return PoolLayoutPlan(
        assignments=assignments,
        idle_gpus=idle,
        deciding_field=decision.deciding_field,
        objective=decision.obj,
        continuous_batch_admitted=decision.cb_admitted,
    )
