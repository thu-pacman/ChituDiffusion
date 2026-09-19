"""Fine-grained pipeline settings, independent of the model and scheduler."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FppConfig:
    patches: int = 4
    warmup_steps: int = 1
    cooldown_steps: int = 1
    refresh_interval: int = 50
    rotate_patches: bool = True
    schedule: str = "stream"
    reference_stages: int | None = None
    reference_context_degree: int = 1

    def __post_init__(self) -> None:
        for name in ("patches", "warmup_steps", "cooldown_steps", "refresh_interval"):
            value = getattr(self, name)
            minimum = 1 if name in {"patches", "warmup_steps"} else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"FPP {name} must be an integer >= {minimum}")
        if type(self.rotate_patches) is not bool:
            raise ValueError("FPP rotate_patches must be a boolean")
        if self.schedule not in {"stream", "step"}:
            raise ValueError("FPP schedule must be 'stream' or 'step'")
        if self.reference_context_degree is None:
            raise ValueError("FPP reference_context_degree must be a positive integer")
        for value in (self.reference_stages, self.reference_context_degree):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError("FPP reference degrees must be positive integers")
        if self.schedule == "step" and (
            self.reference_stages is not None or self.reference_context_degree != 1
        ):
            raise ValueError("FPP virtual reference degrees require stream scheduling")

    def full_step(self, step: int, total_steps: int) -> bool:
        if not 0 <= step < total_steps:
            raise ValueError("FPP step must be within the request's timestep schedule")
        # Short requests may consist entirely of full steps. Warmup/cooldown
        # never add steps to, or truncate, the user's schedule.
        return (
            self.patches == 1
            or step < self.warmup_steps
            or step >= total_steps - self.cooldown_steps
            or (
                self.refresh_interval > 0
                and (step if self.schedule == "step" else step - self.warmup_steps + 1)
                % self.refresh_interval
                == 0
            )
        )

    def order(self, step: int, stages: int = 1) -> tuple[int, ...]:
        if self.schedule == "stream":
            order = tuple(range(self.patches))
            count = max(0, step - self.warmup_steps)
            if self.refresh_interval:
                count -= count // self.refresh_interval
            if self.rotate_patches:
                for _ in range(count):
                    cut = self.patches - stages
                    order = order[cut:] + order[:cut][::-1]
            return order
        start = (
            max(0, step - self.warmup_steps) % self.patches
            if self.rotate_patches
            else 0
        )
        return tuple((start + index) % self.patches for index in range(self.patches))


def partition_bounds(length: int, parts: int) -> tuple[tuple[int, int], ...]:
    """Nonempty, balanced ranges with no padding or omitted tail tokens."""
    if type(length) is not int or type(parts) is not int or not 1 <= parts <= length:
        raise ValueError(f"partition needs 1 <= parts <= length, got {parts}, {length}")
    size, remainder = divmod(length, parts)
    bounds = []
    start = 0
    for index in range(parts):
        stop = start + size + (index < remainder)
        bounds.append((start, stop))
        start = stop
    return tuple(bounds)
