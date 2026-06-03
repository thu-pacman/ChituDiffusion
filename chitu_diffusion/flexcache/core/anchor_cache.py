from __future__ import annotations

import os
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, Hashable, Iterable, Optional

import torch

from chitu_diffusion.runtime.backend import CFGType, DiffusionBackend

logger = getLogger(__name__)


@dataclass
class AnchorCacheDecision:
    should_compute: bool
    is_anchor: bool
    key: Hashable
    interval: int
    reason: str


class CurvatureState:
    def __init__(self):
        self.previous: Dict[Hashable, torch.Tensor] = {}
        self.curvature: Dict[Hashable, float] = {}

    def update(self, key: Hashable, value: torch.Tensor) -> Optional[float]:
        value = value.detach()
        previous = self.previous.get(key)
        self.previous[key] = value
        if previous is None or previous.shape != value.shape:
            return None

        with torch.no_grad():
            diff = (value.float() - previous.to(value.device).float()).square().mean().sqrt()
            denom = previous.to(value.device).float().square().mean().sqrt().clamp_min(1e-8)
            curvature = float((diff / denom).detach().cpu().item())
        self.curvature[key] = curvature
        return curvature

    def reset(self):
        self.previous.clear()
        self.curvature.clear()


class AnchorCachePlanner:
    def __init__(
        self,
        *,
        cache_ratio: float,
        warmup_steps: int,
        cooldown_steps: int,
        total_steps: int,
        tau_max: int = 8,
        curvature_interval_power: float = 1.0 / 3.0,
        mode: str = "ranked_interval",
    ):
        self.cache_ratio = float(max(0.0, min(1.0, cache_ratio)))
        self.warmup_steps = max(0, int(warmup_steps))
        self.cooldown_steps = max(0, int(cooldown_steps))
        self.total_steps = max(1, int(total_steps))
        self.mode = str(mode)
        self.tau_max = max(1, int(tau_max))
        self.curvature_interval_power = max(0.0, float(curvature_interval_power))
        self.curvature = CurvatureState()
        self.intervals: Dict[Hashable, int] = {}
        self.last_compute_step: Dict[Hashable, int] = {}
        self.next_anchor_step: Dict[str, int] = {}
        self.decision_records: Dict[tuple[int, str], int] = {}
        self.max_step = -1

    def reset(self):
        self.curvature.reset()
        self.intervals.clear()
        self.last_compute_step.clear()
        self.next_anchor_step.clear()
        self.decision_records.clear()
        self.max_step = -1

    def branch_key(self) -> str:
        return "pos" if DiffusionBackend.cfg_type == CFGType.POS else "neg"

    def current_step(self) -> int:
        task = DiffusionBackend.generator.current_task
        return int(task.buffer.current_step)

    def in_warmup_or_cooldown(self, step: Optional[int] = None) -> bool:
        step = self.current_step() if step is None else int(step)
        cooldown_start = max(0, self.total_steps - self.cooldown_steps)
        return step < self.warmup_steps or step >= cooldown_start

    def is_anchor_step(self, step: Optional[int] = None) -> bool:
        step = self.current_step() if step is None else int(step)
        if self.in_warmup_or_cooldown(step):
            return True
        branch_key = self.branch_key()
        if branch_key not in self.next_anchor_step:
            self.next_anchor_step[branch_key] = max(self.warmup_steps, step)
        if step >= self.next_anchor_step[branch_key]:
            return True
        return False

    def decide(self, key: Hashable) -> AnchorCacheDecision:
        step = self.current_step()
        if self.in_warmup_or_cooldown(step):
            decision = AnchorCacheDecision(True, True, key, 1, "warmup_cooldown")
            self.record_decision(key, decision)
            return decision
        if self.is_anchor_step(step):
            decision = AnchorCacheDecision(True, True, key, self.intervals.get(key, 1), "anchor")
            self.record_decision(key, decision)
            return decision

        interval = max(1, int(self.intervals.get(key, 1)))
        last_step = self.last_compute_step.get(key)
        if last_step is None:
            decision = AnchorCacheDecision(True, False, key, interval, "cache_miss")
            self.record_decision(key, decision)
            return decision
        if (step - last_step) >= interval:
            decision = AnchorCacheDecision(True, False, key, interval, "interval_due")
            self.record_decision(key, decision)
            return decision
        decision = AnchorCacheDecision(False, False, key, interval, "reuse")
        self.record_decision(key, decision)
        return decision

    def mark_computed(self, key: Hashable, value: torch.Tensor, is_anchor: bool = False):
        step = self.current_step()
        self.last_compute_step[key] = step
        curvature = self.curvature.update(key, value)
        if curvature is not None:
            self._refresh_intervals(self.curvature.curvature.keys())
        if is_anchor and not self.in_warmup_or_cooldown(step):
            self._schedule_next_anchor(step)
        if is_anchor and curvature is not None:
            logger.info(
                "[FlexCache anchor] mode=%s branch=%s step=%d key=%s curvature=%.6e interval=%d next_anchor=%s",
                self.mode,
                self.branch_key(),
                step,
                self._label_for_key(key),
                curvature,
                int(self.intervals.get(key, 1)),
                self.next_anchor_step.get(self.branch_key()),
            )

    def _refresh_intervals(self, keys: Iterable[Hashable]):
        keys = list(keys)
        if not keys:
            return

        curvatures = torch.tensor(
            [max(float(self.curvature.curvature.get(key, 0.0)), 1e-8) for key in keys],
            dtype=torch.float64,
        )
        if self.curvature_interval_power > 0:
            raw = torch.pow(curvatures, -self.curvature_interval_power)
        else:
            raw = torch.ones_like(curvatures)

        # cache_ratio controls the mean interval target. 0 => all compute,
        # 1 => approach tau_max, while preserving curvature ordering.
        target_mean = 1.0 + self.cache_ratio * (self.tau_max - 1.0)
        scale = target_mean / raw.mean().clamp_min(1e-8)
        intervals = torch.round(raw * scale).clamp(1, self.tau_max).to(torch.int64)
        for key, interval in zip(keys, intervals.tolist()):
            self.intervals[key] = int(interval)

    def _schedule_next_anchor(self, step: int):
        branch_key = self.branch_key()
        branch_intervals = [
            interval
            for key, interval in self.intervals.items()
            if not isinstance(key, tuple) or str(key[0]) == branch_key
        ]
        anchor_gap = max(1, 2 * max(branch_intervals, default=1))
        self._set_next_anchor_step(branch_key, step, anchor_gap)

    def _set_next_anchor_step(self, branch_key: str, step: int, anchor_gap: int):
        cooldown_start = max(0, self.total_steps - self.cooldown_steps)
        next_step = step + anchor_gap
        if next_step >= cooldown_start:
            self.next_anchor_step[branch_key] = self.total_steps + 1
        else:
            self.next_anchor_step[branch_key] = next_step

    def record_decision(self, key: Hashable, decision: AnchorCacheDecision):
        step = self.current_step()
        code = self._decision_code(decision)
        self.decision_records[(step, self._label_for_key(key))] = code
        self.max_step = max(self.max_step, step)

    @staticmethod
    def _decision_code(decision: AnchorCacheDecision) -> int:
        if decision.reason == "warmup_cooldown":
            return 0
        if decision.is_anchor:
            return 1
        if decision.should_compute:
            return 2
        return 3

    @staticmethod
    def _label_for_key(key: Hashable) -> str:
        if isinstance(key, tuple):
            return "/".join(str(part) for part in key[1:])
        return str(key)

    def save_decision_ppm(self, output_dir: str, filename: str):
        if self.max_step < 0 or not self.decision_records:
            return

        os.makedirs(output_dir, exist_ok=True)
        labels = sorted({label for _, label in self.decision_records.keys()})
        label_to_row = {label: idx for idx, label in enumerate(labels)}
        cell = 8
        width = (self.max_step + 1) * cell
        height = max(1, len(labels)) * cell
        rgb = bytearray(width * height * 3)

        for idx in range(0, len(rgb), 3):
            rgb[idx] = 20
            rgb[idx + 1] = 20
            rgb[idx + 2] = 20

        for (step, label), code in self.decision_records.items():
            row = label_to_row[label]
            color = self._color_for_code(code)
            for yy in range(row * cell, (row + 1) * cell):
                for xx in range(step * cell, (step + 1) * cell):
                    idx = (yy * width + xx) * 3
                    rgb[idx] = color[0]
                    rgb[idx + 1] = color[1]
                    rgb[idx + 2] = color[2]

        ppm_path = os.path.join(output_dir, filename)
        with open(ppm_path, "wb") as f:
            f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            f.write(bytes(rgb))

    @staticmethod
    def _color_for_code(code: int) -> tuple[int, int, int]:
        if code == 0:
            return (150, 150, 150)  # warmup/cooldown
        if code == 1:
            return (30, 180, 80)    # anchor: compute + decision
        if code == 2:
            return (40, 140, 255)   # compute
        if code == 3:
            return (255, 180, 40)   # reuse
        return (0, 0, 0)
