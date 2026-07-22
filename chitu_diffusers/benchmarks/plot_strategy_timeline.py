from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PHASE_RE = re.compile(
    r"(?:EPE phase layout|elastic pulse \d+ layout|static_cp phase|static_dp phase): "
    r"(?P<layout>.*)$"
)
ASSIGNMENT_RE = re.compile(
    r"(?P<request>[^, ]+):(?P<parallel>cp|dp)(?P<width>\d+)@"
    r"\[(?P<ranks>[^]]+)] step=(?P<start>\d+)\+(?P<steps>\d+)"
    r"(?: predicted=(?P<predicted>[0-9.]+)ms/step)?"
)
OBSERVATION_RE = re.compile(
    r"EPE observation: (?P<request>\S+) cp(?P<width>\d+) "
    r"predicted=(?P<predicted>[0-9.]+)ms measured=(?P<measured>[0-9.]+)ms"
)
STATIC_DP_COMPLETE_RE = re.compile(
    r"static_dp lane complete: request=(?P<request>\S+) rank=(?P<rank>\d+) "
    r"start_unix_ms=(?P<start>[0-9.]+) "
    r"denoise_start_unix_ms=(?P<denoise_start>[0-9.]+) "
    r"end_unix_ms=(?P<end>[0-9.]+) steps=(?P<steps>\d+) "
    r"predicted=(?P<predicted>[0-9.]+)ms/step "
    r"measured=(?P<measured>[0-9.]+)ms/step"
)
STATIC_CP_COMPLETE_RE = re.compile(
    r"static_cp lane complete: request=(?P<request>\S+) "
    r"ranks=\[(?P<ranks>[^]]+)] start_unix_ms=(?P<start>[0-9.]+) "
    r"denoise_start_unix_ms=(?P<denoise_start>[0-9.]+) "
    r"end_unix_ms=(?P<end>[0-9.]+) steps=(?P<steps>\d+) "
    r"predicted=(?P<predicted>[0-9.]+)ms/step "
    r"measured=(?P<measured>[0-9.]+)ms/step"
)
FINALIZE_PROFILE_RE = re.compile(
    r"finalize=(?P<finalize>[0-9.]+)ms "
    r"vae=(?P<vae>[0-9.]+)ms "
    r"postprocess=(?P<postprocess>[0-9.]+)ms "
    r"png=(?P<png>[0-9.]+)ms"
)


@dataclass
class Assignment:
    request_id: str
    width: int
    ranks: tuple[int, ...]
    start_step: int
    steps: int
    predicted_step_ms: float | None
    measured_step_ms: float | None = None

    @property
    def duration_ms(self) -> float:
        step_ms = self.measured_step_ms or self.predicted_step_ms
        if step_ms is None:
            raise ValueError(f"{self.request_id} has no measured or predicted cost")
        return self.steps * step_ms


@dataclass
class Phase:
    index: int
    assignments: list[Assignment]
    start_ms: float = 0.0
    end_ms: float = 0.0


@dataclass(frozen=True)
class Segment:
    gpu: int
    start_ms: float
    end_ms: float
    kind: str
    request_id: str | None
    width: int
    phase: int


@dataclass
class Run:
    name: str
    phases: list[Phase]
    segments: list[Segment]
    schedule_summary: dict[str, float | int | str]
    metrics: dict
    step_cost_ms: dict[tuple[int, int], float]


def parse_phases(
    log_path: Path,
    *,
    request_tokens: dict[str, int],
    fallback_cost_ms: dict[tuple[int, int], float],
) -> list[Phase]:
    phases: list[Phase] = []
    current: Phase | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        phase_match = PHASE_RE.search(line)
        if phase_match:
            assignments: list[Assignment] = []
            for match in ASSIGNMENT_RE.finditer(phase_match.group("layout")):
                request_id = match.group("request")
                if request_id.startswith("benchmark-warmup-"):
                    continue
                assignments.append(
                    Assignment(
                        request_id=request_id,
                        width=int(match.group("width")),
                        ranks=tuple(
                            int(rank.strip())
                            for rank in match.group("ranks").split(",")
                        ),
                        start_step=int(match.group("start")),
                        steps=int(match.group("steps")),
                        predicted_step_ms=(
                            float(match.group("predicted"))
                            if match.group("predicted")
                            else fallback_cost_ms.get(
                                (request_tokens[request_id], int(match.group("width")))
                            )
                        ),
                    )
                )
            current = None
            if assignments:
                current = Phase(index=len(phases), assignments=assignments)
                phases.append(current)
            continue

        observation = OBSERVATION_RE.search(line)
        if observation is None or current is None:
            continue
        request_id = observation.group("request")
        width = int(observation.group("width"))
        for assignment in current.assignments:
            if assignment.request_id == request_id and assignment.width == width:
                assignment.measured_step_ms = float(observation.group("measured"))
                break

    if not phases:
        raise ValueError(f"no scheduling phases found in {log_path}")
    progress: dict[str, int] = {}
    for index, phase in enumerate(phases):
        for assignment in tuple(phase.assignments):
            known_step = progress.get(assignment.request_id, 0)
            if assignment.start_step > known_step and index > 0:
                phases[index - 1].assignments.append(
                    Assignment(
                        request_id=assignment.request_id,
                        width=assignment.width,
                        ranks=assignment.ranks,
                        start_step=known_step,
                        steps=assignment.start_step - known_step,
                        predicted_step_ms=assignment.predicted_step_ms,
                    )
                )
            progress[assignment.request_id] = assignment.start_step + assignment.steps
    return phases


def build_segments(
    phases: list[Phase], world_size: int
) -> tuple[list[Segment], dict[str, float | int | str]]:
    segments: list[Segment] = []
    now_ms = 0.0
    compute_gpu_ms = 0.0
    barrier_idle_gpu_ms = 0.0
    unused_idle_gpu_ms = 0.0

    for phase in phases:
        phase.start_ms = now_ms
        phase_duration_ms = max(item.duration_ms for item in phase.assignments)
        phase.end_ms = now_ms + phase_duration_ms
        occupied: set[int] = set()
        for assignment in phase.assignments:
            compute_end_ms = now_ms + assignment.duration_ms
            for gpu in assignment.ranks:
                occupied.add(gpu)
                segments.append(
                    Segment(
                        gpu=gpu,
                        start_ms=now_ms,
                        end_ms=compute_end_ms,
                        kind="compute",
                        request_id=assignment.request_id,
                        width=assignment.width,
                        phase=phase.index,
                    )
                )
                compute_gpu_ms += assignment.duration_ms
                if compute_end_ms < phase.end_ms - 0.5:
                    segments.append(
                        Segment(
                            gpu=gpu,
                            start_ms=compute_end_ms,
                            end_ms=phase.end_ms,
                            kind="pulse_wait",
                            request_id=assignment.request_id,
                            width=assignment.width,
                            phase=phase.index,
                        )
                    )
                    barrier_idle_gpu_ms += phase.end_ms - compute_end_ms
        for gpu in sorted(set(range(world_size)) - occupied):
            segments.append(
                Segment(
                    gpu=gpu,
                    start_ms=now_ms,
                    end_ms=phase.end_ms,
                    kind="unused",
                    request_id=None,
                    width=0,
                    phase=phase.index,
                )
            )
            unused_idle_gpu_ms += phase_duration_ms
        now_ms = phase.end_ms

    capacity_gpu_ms = now_ms * world_size
    return segments, {
        "world_size": world_size,
        "phase_count": len(phases),
        "denoise_wall_ms": now_ms,
        "compute_gpu_ms": compute_gpu_ms,
        "pulse_wait_gpu_ms": barrier_idle_gpu_ms,
        "unused_gpu_ms": unused_idle_gpu_ms,
        "denoise_gpu_utilization": (
            compute_gpu_ms / capacity_gpu_ms if capacity_gpu_ms else 0.0
        ),
    }


def parse_rank_timeline(
    run_dir: Path,
    world_size: int,
) -> tuple[list[Segment], dict[str, float | int | str]] | None:
    paths = sorted(run_dir.glob("timeline-rank*.jsonl"))
    if len(paths) != world_size:
        return None
    events: list[dict] = []
    seen_ranks: set[int] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            rank = int(event["rank"])
            if not 0 <= rank < world_size:
                raise ValueError(f"invalid timeline rank {rank} in {path}")
            seen_ranks.add(rank)
            events.append(event)
    if seen_ranks != set(range(world_size)) or not events:
        return None

    submissions = [
        int(event["start_unix_ns"])
        for event in events
        if event["stage"] == "request_submit"
    ]
    origin_ns = min(submissions or [int(event["start_unix_ns"]) for event in events])
    stage_kinds = {
        "prepare": "prepare",
        "denoise": "compute",
        "pulse_wait": "pulse_wait",
        "control_wait": "control_wait",
        "state_transfer": "state_transfer",
        "vae": "vae",
        "d2h": "d2h",
        "postprocess": "postprocess",
        "png": "png",
        "cpu_postprocess": "cpu_postprocess",
        "cpu_png": "cpu_png",
    }
    segments: list[Segment] = []
    typed_gpu_ms: dict[str, float] = {kind: 0.0 for kind in set(stage_kinds.values())}
    typed_cpu_ms: dict[str, float] = {
        "cpu_postprocess": 0.0,
        "cpu_png": 0.0,
    }
    phase_counter = 0
    for event in sorted(events, key=lambda item: int(item["start_unix_ns"])):
        kind = stage_kinds.get(str(event["stage"]))
        if kind is None:
            continue
        start_ms = (int(event["start_unix_ns"]) - origin_ns) / 1_000_000.0
        end_ms = (int(event["end_unix_ns"]) - origin_ns) / 1_000_000.0
        if end_ms <= start_ms + 0.001:
            continue
        metadata = event.get("metadata") or {}
        phase = int(metadata.get("epoch", phase_counter))
        phase_counter += 1
        lane_ranks = tuple(int(rank) for rank in event.get("lane_ranks", ()))
        resource = str(event.get("resource", "gpu"))
        segment_rank = world_size if resource == "cpu" else int(event["rank"])
        segments.append(
            Segment(
                gpu=segment_rank,
                start_ms=start_ms,
                end_ms=end_ms,
                kind=kind,
                request_id=event.get("request_id"),
                width=max(1, len(lane_ranks)),
                phase=phase,
            )
        )
        if resource == "cpu":
            typed_cpu_ms[kind] = typed_cpu_ms.get(kind, 0.0) + end_ms - start_ms
        else:
            typed_gpu_ms[kind] += end_ms - start_ms

    if not segments:
        return None
    wall_ms = max(segment.end_ms for segment in segments)
    idle_gpu_ms = 0.0
    for gpu in range(world_size):
        cursor_ms = 0.0
        gpu_segments = sorted(
            (segment for segment in segments if segment.gpu == gpu),
            key=lambda segment: segment.start_ms,
        )
        for segment in gpu_segments:
            if segment.start_ms > cursor_ms + 0.5:
                segments.append(
                    Segment(
                        gpu=gpu,
                        start_ms=cursor_ms,
                        end_ms=segment.start_ms,
                        kind="idle",
                        request_id=None,
                        width=1,
                        phase=segment.phase,
                    )
                )
                idle_gpu_ms += segment.start_ms - cursor_ms
            cursor_ms = max(cursor_ms, segment.end_ms)
        if cursor_ms < wall_ms - 0.5:
            segments.append(
                Segment(
                    gpu=gpu,
                    start_ms=cursor_ms,
                    end_ms=wall_ms,
                    kind="idle",
                    request_id=None,
                    width=1,
                    phase=phase_counter,
                )
            )
            idle_gpu_ms += wall_ms - cursor_ms

    compute_gpu_ms = typed_gpu_ms.get("compute", 0.0)
    return segments, {
        "timeline_source": "rank_events",
        "world_size": world_size,
        "phase_count": phase_counter,
        "denoise_wall_ms": wall_ms,
        "compute_gpu_ms": compute_gpu_ms,
        "pulse_wait_gpu_ms": typed_gpu_ms.get("pulse_wait", 0.0),
        "control_wait_gpu_ms": typed_gpu_ms.get("control_wait", 0.0),
        "state_transfer_gpu_ms": typed_gpu_ms.get("state_transfer", 0.0),
        "d2h_gpu_ms": typed_gpu_ms.get("d2h", 0.0),
        "unused_gpu_ms": idle_gpu_ms,
        "outside_dit_gpu_ms": 0.0,
        "prepare_gpu_ms": typed_gpu_ms.get("prepare", 0.0),
        "vae_gpu_ms": typed_gpu_ms.get("vae", 0.0),
        "postprocess_gpu_ms": typed_gpu_ms.get("postprocess", 0.0),
        "png_gpu_ms": typed_gpu_ms.get("png", 0.0),
        "cpu_postprocess_ms": typed_cpu_ms.get("cpu_postprocess", 0.0),
        "cpu_png_ms": typed_cpu_ms.get("cpu_png", 0.0),
        "denoise_gpu_utilization": compute_gpu_ms / (wall_ms * world_size),
    }


def parse_fixed_lane_segments(
    log_path: Path, world_size: int
) -> tuple[list[Segment], dict[str, float | int | str]] | None:
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        dp_match = STATIC_DP_COMPLETE_RE.search(line)
        cp_match = STATIC_CP_COMPLETE_RE.search(line)
        finalize_match = FINALIZE_PROFILE_RE.search(line)
        if dp_match is not None:
            row = dp_match.groupdict()
            row["lane_ranks"] = (int(row["rank"]),)
            row["finalize_profile"] = (
                finalize_match.groupdict() if finalize_match is not None else None
            )
            rows.append(row)
        elif cp_match is not None:
            row = cp_match.groupdict()
            row["lane_ranks"] = tuple(
                int(rank.strip()) for rank in row["ranks"].split(",")
            )
            row["finalize_profile"] = (
                finalize_match.groupdict() if finalize_match is not None else None
            )
            rows.append(row)
    if not rows:
        return None

    origin_ms = min(float(row["start"]) for row in rows)
    segments: list[Segment] = []
    typed_gpu_ms: dict[str, float] = {
        "prepare": 0.0,
        "vae": 0.0,
        "postprocess": 0.0,
        "png": 0.0,
        "lane_wait": 0.0,
    }
    for phase, row in enumerate(rows):
        request_start_ms = float(row["start"]) - origin_ms
        start_ms = float(row["denoise_start"]) - origin_ms
        end_ms = start_ms + int(row["steps"]) * float(row["measured"])
        for gpu in row["lane_ranks"]:
            if start_ms > request_start_ms + 0.5:
                segments.append(
                    Segment(
                        gpu=gpu,
                        start_ms=request_start_ms,
                        end_ms=start_ms,
                        kind="prepare",
                        request_id=row["request"],
                        width=len(row["lane_ranks"]),
                        phase=phase,
                    )
                )
                typed_gpu_ms["prepare"] += start_ms - request_start_ms
            segments.append(
                Segment(
                    gpu=gpu,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    kind="compute",
                    request_id=row["request"],
                    width=len(row["lane_ranks"]),
                    phase=phase,
                )
            )
        completed_ms = float(row["end"]) - origin_ms
        if completed_ms <= end_ms + 0.5:
            continue
        leader = row["lane_ranks"][0]
        finalize_profile = row["finalize_profile"]
        if finalize_profile is not None:
            cursor_ms = end_ms
            for kind in ("vae", "postprocess", "png"):
                duration_ms = float(finalize_profile[kind])
                typed_end_ms = min(cursor_ms + duration_ms, completed_ms)
                if typed_end_ms > cursor_ms + 0.5:
                    segments.append(
                        Segment(
                            gpu=leader,
                            start_ms=cursor_ms,
                            end_ms=typed_end_ms,
                            kind=kind,
                            request_id=row["request"],
                            width=1,
                            phase=phase,
                        )
                    )
                    typed_gpu_ms[kind] += typed_end_ms - cursor_ms
                cursor_ms = typed_end_ms
        for gpu in row["lane_ranks"][1:]:
            segments.append(
                Segment(
                    gpu=gpu,
                    start_ms=end_ms,
                    end_ms=completed_ms,
                    kind="lane_wait",
                    request_id=row["request"],
                    width=len(row["lane_ranks"]),
                    phase=phase,
                )
            )
            typed_gpu_ms["lane_wait"] += completed_ms - end_ms

    wall_ms = max(float(row["end"]) - origin_ms for row in rows)
    compute_gpu_ms = sum(
        segment.end_ms - segment.start_ms
        for segment in segments
        if segment.kind == "compute"
    )
    unclassified_gpu_ms = 0.0
    for gpu in range(world_size):
        cursor_ms = 0.0
        gpu_segments = sorted(
            (segment for segment in segments if segment.gpu == gpu),
            key=lambda segment: segment.start_ms,
        )
        for segment in gpu_segments:
            if segment.start_ms > cursor_ms + 0.5:
                segments.append(
                    Segment(
                        gpu=gpu,
                        start_ms=cursor_ms,
                        end_ms=segment.start_ms,
                        kind="outside_dit",
                        request_id=None,
                        width=1,
                        phase=segment.phase,
                    )
                )
                unclassified_gpu_ms += segment.start_ms - cursor_ms
            cursor_ms = max(cursor_ms, segment.end_ms)
        if cursor_ms < wall_ms - 0.5:
            segments.append(
                Segment(
                    gpu=gpu,
                    start_ms=cursor_ms,
                    end_ms=wall_ms,
                    kind="outside_dit",
                    request_id=None,
                    width=1,
                    phase=len(rows),
                )
            )
            unclassified_gpu_ms += wall_ms - cursor_ms
    return segments, {
        "world_size": world_size,
        "phase_count": len(rows),
        "denoise_wall_ms": wall_ms,
        "compute_gpu_ms": compute_gpu_ms,
        "pulse_wait_gpu_ms": 0.0,
        "unused_gpu_ms": 0.0,
        "outside_dit_gpu_ms": unclassified_gpu_ms,
        **{f"{kind}_gpu_ms": value for kind, value in typed_gpu_ms.items()},
        "denoise_gpu_utilization": compute_gpu_ms / (wall_ms * world_size),
    }


def load_run(
    name: str,
    log_path: Path,
    metrics_path: Path,
    world_size: int,
    request_tokens: dict[str, int],
) -> Run:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    warmup_path = log_path.parent / "epe_warmup.json"
    step_cost_ms = {}
    if warmup_path.exists():
        warmup = json.loads(warmup_path.read_text(encoding="utf-8"))
        step_cost_ms = {
            (int(row["image_tokens"]), int(row["width"])): float(row["latency_ms"])
            for row in warmup["rows"]
        }
    rank_timeline = parse_rank_timeline(log_path.parent, world_size)
    if rank_timeline is not None:
        segments, schedule_summary = rank_timeline
        return Run(name, [], segments, schedule_summary, metrics, step_cost_ms)
    fixed_lane = parse_fixed_lane_segments(log_path, world_size)
    if fixed_lane is not None:
        segments, schedule_summary = fixed_lane
        return Run(name, [], segments, schedule_summary, metrics, step_cost_ms)
    phases = parse_phases(
        log_path,
        request_tokens=request_tokens,
        fallback_cost_ms=step_cost_ms,
    )
    segments, schedule_summary = build_segments(phases, world_size)
    return Run(name, phases, segments, schedule_summary, metrics, step_cost_ms)


def request_colors(request_ids: list[str]) -> dict[str, tuple[float, ...]]:
    palette = plt.colormaps.get_cmap("tab20")
    return {
        request_id: palette(index % palette.N)
        for index, request_id in enumerate(sorted(set(request_ids)))
    }


def derive_deadlines_ms(
    requests: list[dict],
    runs: list[Run],
    *,
    default_factor: float,
) -> tuple[dict[str, float], dict[str, str]]:
    """Return absolute trace deadlines and whether each came from trace or cp1 cost."""
    if default_factor <= 0:
        raise ValueError("default deadline factor must be positive")
    deadlines = {}
    sources = {}
    for request in requests:
        request_id = str(request["request_id"])
        arrival_ms = float(request["arrival_ms"])
        explicit_ms = request.get("deadline_ms")
        if explicit_ms is not None:
            deadlines[request_id] = arrival_ms + float(explicit_ms)
            sources[request_id] = "explicit"
            continue
        image_tokens = (int(request["height"]) // 16) * (int(request["width"]) // 16)
        cp1_costs = [
            run.step_cost_ms[(image_tokens, 1)]
            for run in runs
            if (image_tokens, 1) in run.step_cost_ms
        ]
        if not cp1_costs:
            raise ValueError(
                f"request {request_id!r} needs a cp1 warmup row to derive its "
                "default deadline"
            )
        single_gpu_ms = statistics.median(cp1_costs) * int(request["num_steps"])
        deadlines[request_id] = arrival_ms + default_factor * single_gpu_ms
        sources[request_id] = "cp1_default"
    return deadlines, sources


def strategy_colors(runs: list[Run]) -> dict[str, tuple[float, ...]]:
    palette = plt.colormaps.get_cmap("tab10")
    return {run.name: palette(index % palette.N) for index, run in enumerate(runs)}


def draw_arrivals(
    ax,
    requests: list[dict],
    runs: list[Run],
    colors: dict[str, tuple[float, ...]],
    run_colors: dict[str, tuple[float, ...]],
    deadlines_ms: dict[str, float],
    deadline_sources: dict[str, str],
    default_deadline_factor: float,
    shared_end_ms: float,
) -> None:
    markers = {512: "o", 1024: "s", 2048: "^"}
    run_count = len(runs)
    offsets = [
        (index - (run_count - 1) / 2.0) * min(0.16, 0.48 / max(1, run_count))
        for index in range(run_count)
    ]
    for row, request in enumerate(requests):
        request_id = str(request["request_id"])
        arrival_s = float(request["arrival_ms"]) / 1000.0
        size = int(request["width"])
        ax.scatter(
            arrival_s,
            row,
            s=55,
            marker=markers.get(size, "o"),
            color=colors[request_id],
            edgecolor="#101828",
            linewidth=0.6,
            zorder=3,
        )
        deadline_s = deadlines_ms[request_id] / 1000.0
        ax.hlines(
            row,
            arrival_s,
            deadline_s,
            color="#d92d20",
            linewidth=0.9,
            linestyle=(0, (3, 2)),
            alpha=0.6,
        )
        ax.scatter(
            deadline_s,
            row,
            s=62,
            marker="|",
            color="#b42318",
            linewidth=1.8,
            zorder=5,
        )
        for offset, run in zip(offsets, runs, strict=True):
            metrics = run.metrics["per_request"].get(request_id)
            if metrics is None:
                continue
            start_s = (
                float(metrics["trace_arrival_ms"]) + float(metrics["queue_delay_ms"])
            ) / 1000.0
            completion_s = float(metrics["finish_ms"]) / 1000.0
            y = row + offset
            color = run_colors[run.name]
            ax.hlines(
                y,
                start_s,
                completion_s,
                color=color,
                linewidth=1.2,
                alpha=0.8,
                zorder=3,
            )
            ax.scatter(
                start_s,
                y,
                s=27,
                marker=">",
                facecolor="white",
                edgecolor=color,
                linewidth=1.0,
                zorder=4,
            )
            ax.scatter(
                completion_s,
                y,
                s=29,
                marker="x",
                color=color,
                linewidth=1.1,
                zorder=4,
            )
    ax.set_xlim(0.0, shared_end_ms / 1000.0)
    ax.set_ylim(-0.7, len(requests) - 0.3)
    ax.set_yticks(
        range(len(requests)),
        [
            (
                f"{request['request_id']}  {int(request['width'])}²"
                + (
                    f"  SLO {float(request['deadline_ms']) / 1000.0:g}s"
                    if deadline_sources[str(request["request_id"])] == "explicit"
                    else (
                        f"  DDL {default_deadline_factor:g}×cp1 "
                        f"{(deadlines_ms[str(request['request_id'])] - float(request['arrival_ms'])) / 1000.0:.1f}s"
                    )
                )
            )
            for request in requests
        ],
        fontsize=7,
    )
    ax.invert_yaxis()
    ax.set_title(
        "Request arrivals (shared time axis)",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(axis="x", color="#e4e7ec", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=run_colors[run.name],
                marker=">",
                markerfacecolor="white",
                markeredgecolor=run_colors[run.name],
                linewidth=1.2,
                label=f"{run.name}: start → completion",
            )
            for run in runs
        ],
        loc="upper right",
        frameon=False,
        fontsize=7.5,
    )


def draw_strategy(
    ax,
    run: Run,
    colors: dict[str, tuple[float, ...]],
    requests: list[dict],
    world_size: int,
    shared_end_ms: float,
) -> None:
    lane_height = 0.68
    has_cpu_lane = any(segment.gpu == world_size for segment in run.segments)
    lane_count = world_size + int(has_cpu_lane)
    for gpu in range(lane_count):
        ax.barh(
            gpu,
            shared_end_ms / 1000.0,
            height=lane_height,
            color="#f2f4f7",
            edgecolor="#d0d5dd",
            linewidth=0.6,
            zorder=0,
        )

    labeled: set[str] = set()
    for segment in run.segments:
        left_s = segment.start_ms / 1000.0
        width_s = (segment.end_ms - segment.start_ms) / 1000.0
        if segment.kind == "compute":
            request_id = segment.request_id or ""
            ax.barh(
                segment.gpu,
                width_s,
                left=left_s,
                height=lane_height,
                color=colors[request_id],
                alpha=0.9,
                edgecolor="#101828",
                linewidth=0.35,
                zorder=2,
            )
            if request_id not in labeled and width_s >= 0.45:
                ax.text(
                    left_s + min(width_s / 2, 1.7),
                    segment.gpu,
                    request_id.removeprefix("req"),
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color="#101828",
                    zorder=4,
                )
                labeled.add(request_id)
        elif segment.kind == "pulse_wait":
            ax.barh(
                segment.gpu,
                width_s,
                left=left_s,
                height=lane_height,
                color="#fda29b",
                alpha=0.65,
                edgecolor="#b42318",
                linewidth=0.45,
                hatch="////",
                zorder=3,
            )
        elif segment.kind in {"unused", "idle"}:
            ax.barh(
                segment.gpu,
                width_s,
                left=left_s,
                height=lane_height,
                color="#d0d5dd",
                edgecolor="#98a2b3",
                linewidth=0.35,
                zorder=1,
            )
        elif segment.kind in {
            "prepare",
            "vae",
            "postprocess",
            "png",
            "lane_wait",
            "control_wait",
            "state_transfer",
            "d2h",
            "cpu_postprocess",
            "cpu_png",
        }:
            styles = {
                "prepare": ("#d1e9ff", "#1570ef", ""),
                "vae": ("#d1fadf", "#039855", ""),
                "postprocess": ("#e9d7fe", "#7f56d9", ""),
                "png": ("#fedf89", "#dc6803", ""),
                "lane_wait": ("#e4e7ec", "#667085", "////"),
                "control_wait": ("#fee4e2", "#d92d20", "xxxx"),
                "state_transfer": ("#cff9fe", "#088ab2", "\\\\"),
                "d2h": ("#a5f0fc", "#0e7090", ""),
                "cpu_postprocess": ("#e9d7fe", "#7f56d9", "...."),
                "cpu_png": ("#fedf89", "#dc6803", "...."),
            }
            facecolor, edgecolor, hatch = styles[segment.kind]
            ax.barh(
                segment.gpu,
                width_s,
                left=left_s,
                height=lane_height,
                color=facecolor,
                edgecolor=edgecolor,
                linewidth=0.35,
                hatch=hatch,
                zorder=2,
            )
        else:
            ax.barh(
                segment.gpu,
                width_s,
                left=left_s,
                height=lane_height,
                color="#fef0c7",
                edgecolor="#f79009",
                linewidth=0.35,
                hatch="....",
                zorder=1,
            )

    for phase in run.phases:
        ax.axvline(
            phase.start_ms / 1000.0,
            color="#667085",
            linewidth=0.35,
            alpha=0.25,
            zorder=1,
        )
    for request in requests:
        ax.axvline(
            float(request["arrival_ms"]) / 1000.0,
            color=colors[str(request["request_id"])],
            linewidth=0.35,
            alpha=0.22,
            zorder=1,
        )
        if request.get("deadline_ms") is not None:
            deadline_s = (
                float(request["arrival_ms"]) + float(request["deadline_ms"])
            ) / 1000.0
            ax.axvline(
                deadline_s,
                color="#b42318",
                linewidth=0.8,
                linestyle=(0, (3, 2)),
                alpha=0.7,
                zorder=3,
            )

    benchmark = run.metrics["summary"]
    makespan_s = float(benchmark["makespan_ms"]) / 1000.0
    ax.axvline(
        makespan_s,
        color="#101828",
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        alpha=0.8,
        zorder=4,
    )
    utilization = float(run.schedule_summary["denoise_gpu_utilization"]) * 100.0
    utilization_label = (
        "measured DiT utilization"
        if run.schedule_summary.get("timeline_source") == "rank_events"
        else "reconstructed denoise utilization"
    )
    slo_total = int(benchmark.get("num_slo_requests", 0))
    slo_label = (
        f"  |  SLO {int(benchmark['num_slo_met'])}/{slo_total}" if slo_total else ""
    )
    ax.set_title(
        f"{run.name}  |  makespan {makespan_s:.1f}s  |  "
        f"throughput {benchmark['throughput_req_per_s']:.3f} req/s  |  "
        f"{utilization_label} {utilization:.1f}%{slo_label}",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_xlim(0.0, shared_end_ms / 1000.0)
    labels = [f"GPU {gpu}" for gpu in range(world_size)]
    if has_cpu_lane:
        labels.append("CPU encode")
    ax.set_yticks(range(lane_count), labels)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e4e7ec", linewidth=0.65)
    ax.set_axisbelow(True)


def render(
    runs: list[Run],
    trace: dict,
    output: Path,
    world_size: int,
    *,
    default_deadline_factor: float = 1.2,
) -> dict[str, dict[str, float | int]]:
    requests = trace["requests"]
    request_ids = [str(request["request_id"]) for request in requests]
    colors = request_colors(request_ids)
    run_colors = strategy_colors(runs)
    arrival_runs = [run for run in runs if run.name.lower() == "elastic"]
    if len(arrival_runs) != 1:
        raise ValueError("request arrival panel requires exactly one Elastic run")
    deadlines_ms, deadline_sources = derive_deadlines_ms(
        requests,
        runs,
        default_factor=default_deadline_factor,
    )
    shared_end_ms = (
        max(
            max(
                float(request["finish_ms"])
                for run in runs
                for request in run.metrics["per_request"].values()
            ),
            max(deadlines_ms.values()),
        )
        * 1.01
    )

    row_count = len(runs) + 1
    fig = plt.figure(
        figsize=(18, max(8.5 + 1.8 * len(runs), 6.0 + len(requests) * 0.2))
    )
    grid = fig.add_gridspec(
        row_count,
        1,
        height_ratios=(2.4, *(1.8 for _ in runs)),
        left=0.065,
        right=0.985,
        top=0.89,
        bottom=0.17,
        hspace=0.52,
    )
    arrival_ax = fig.add_subplot(grid[0])
    strategy_axes = [fig.add_subplot(grid[index]) for index in range(1, row_count)]
    draw_arrivals(
        arrival_ax,
        requests,
        arrival_runs,
        colors,
        run_colors,
        deadlines_ms,
        deadline_sources,
        default_deadline_factor,
        shared_end_ms,
    )
    arrival_ax.tick_params(labelbottom=False)
    for ax, run in zip(strategy_axes, runs, strict=True):
        draw_strategy(ax, run, colors, requests, world_size, shared_end_ms)
    for ax in strategy_axes[:-1]:
        ax.tick_params(labelbottom=False)
    all_measured = all(
        run.schedule_summary.get("timeline_source") == "rank_events" for run in runs
    )
    strategy_axes[-1].set_xlabel(
        "measured wall-clock time (s), shared scale"
        if all_measured
        else "reconstructed denoise time (s), shared scale"
    )

    fig.suptitle(
        str(
            trace.get("meta", {}).get(
                "title", "EPAC mixed-resolution scheduling timeline"
            )
        ),
        x=0.065,
        y=0.975,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    sizes = sorted({int(request["width"]) for request in requests})
    steps = sorted({int(request["num_steps"]) for request in requests})
    size_text = " / ".join(f"{size}²" for size in sizes)
    step_text = "/".join(str(value) for value in steps)
    fig.text(
        0.065,
        0.94,
        f"{len(requests)} requests · {size_text} · {step_text} denoise steps · "
        f"{world_size}× GPU",
        fontsize=10,
        color="#475467",
    )
    request_legend = [
        Patch(
            facecolor=colors[request_id],
            edgecolor="#101828",
            label=request_id,
        )
        for request_id in request_ids
    ]
    state_legend = [
        Patch(
            facecolor="#fda29b",
            edgecolor="#b42318",
            hatch="////",
            label="pulse wait",
        ),
        Patch(
            facecolor="#d0d5dd",
            edgecolor="#98a2b3",
            label="unused / measured idle",
        ),
        Patch(facecolor="#d1e9ff", edgecolor="#1570ef", label="request prepare"),
        Patch(facecolor="#d1fadf", edgecolor="#039855", label="VAE decode"),
        Patch(facecolor="#a5f0fc", edgecolor="#0e7090", label="D2H handoff"),
        Patch(facecolor="#e9d7fe", edgecolor="#7f56d9", label="postprocess"),
        Patch(facecolor="#fedf89", edgecolor="#dc6803", label="PNG encode"),
        Patch(
            facecolor="#e9d7fe",
            edgecolor="#7f56d9",
            hatch="....",
            label="async CPU postprocess",
        ),
        Patch(
            facecolor="#fedf89",
            edgecolor="#dc6803",
            hatch="....",
            label="async CPU PNG",
        ),
        Patch(
            facecolor="#e4e7ec",
            edgecolor="#667085",
            hatch="////",
            label="lane waits for leader finalize",
        ),
        Patch(
            facecolor="#fee4e2",
            edgecolor="#d92d20",
            hatch="xxxx",
            label="control wait",
        ),
        Patch(
            facecolor="#cff9fe",
            edgecolor="#088ab2",
            hatch="\\\\",
            label="state transfer",
        ),
        Patch(
            facecolor="#fef0c7",
            edgecolor="#f79009",
            hatch="....",
            label="outside DiT / unclassified",
        ),
        Line2D(
            [0],
            [0],
            color="#101828",
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            label="benchmark completion",
        ),
        Line2D(
            [0],
            [0],
            color="#b42318",
            linewidth=0.9,
            linestyle=(0, (3, 2)),
            label="request deadline",
        ),
    ]
    fig.legend(
        handles=request_legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.088),
        ncol=8,
        frameon=False,
        fontsize=7.8,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.legend(
        handles=state_legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=7,
        frameon=False,
        fontsize=7.8,
        columnspacing=1.1,
        handletextpad=0.45,
    )
    fig.text(
        0.985,
        0.012,
        (
            "All panels share one x-axis and use independently recorded rank-local "
            "wall-clock spans."
            if all_measured
            else "All panels share one x-axis; runs without rank traces fall back to "
            "log reconstruction."
        ),
        ha="right",
        fontsize=8.5,
        color="#667085",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return {run.name: run.schedule_summary for run in runs}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare request arrivals and named EPAC GPU timelines"
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=RUN_DIR",
        help="Named run containing service.log, metrics.json and rank timelines",
    )
    parser.add_argument("--arrival-rate", type=float)
    parser.add_argument("--static-dp-log", type=Path)
    parser.add_argument("--static-dp-metrics", type=Path)
    parser.add_argument("--static-cp-log", type=Path)
    parser.add_argument("--static-cp-metrics", type=Path)
    parser.add_argument("--elastic-log", type=Path)
    parser.add_argument("--elastic-metrics", type=Path)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument(
        "--default-deadline-factor",
        type=float,
        default=1.2,
        help=(
            "For requests without deadline_ms, derive DDL as this factor times "
            "the measured cp1 denoise time (default: 1.2)."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    if args.arrival_rate is not None:
        from .benchmark_client import scale_trace_arrivals

        trace["requests"], _ = scale_trace_arrivals(trace, args.arrival_rate)
    request_tokens = {
        str(request["request_id"]): (int(request["height"]) // 16)
        * (int(request["width"]) // 16)
        for request in trace["requests"]
    }
    if args.run:
        named_runs = []
        for value in args.run:
            if "=" not in value:
                parser.error(f"invalid --run {value!r}; expected NAME=RUN_DIR")
            name, raw_directory = value.split("=", 1)
            directory = Path(raw_directory)
            named_runs.append(
                (name, directory / "service.log", directory / "metrics.json")
            )
    else:
        legacy = (
            ("Static DP", args.static_dp_log, args.static_dp_metrics),
            ("Static CP", args.static_cp_log, args.static_cp_metrics),
            ("Elastic", args.elastic_log, args.elastic_metrics),
        )
        if any(log is None or metrics is None for _, log, metrics in legacy):
            parser.error("provide --run NAME=RUN_DIR or all legacy strategy paths")
        named_runs = legacy
    runs = [
        load_run(name, log, metrics, args.world_size, request_tokens)
        for name, log, metrics in named_runs
    ]
    summary = render(
        runs,
        trace,
        args.output,
        args.world_size,
        default_deadline_factor=args.default_deadline_factor,
    )
    args.output.with_name(f"{args.output.stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
