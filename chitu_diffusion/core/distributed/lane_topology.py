from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
from typing import Any, Iterable, Literal, Optional


logger = getLogger(__name__)

TopologyMode = Literal["fixed", "elastic"]


def _read(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _positive_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer, got {result}")
    return result


def _divisors(value: int) -> tuple[int, ...]:
    return tuple(d for d in range(1, value + 1) if value % d == 0)


@dataclass(frozen=True)
class FixedTopology:
    dp: int
    cfp: int
    cp: int

    @property
    def lane_width(self) -> int:
        return self.cfp * self.cp

    def validate(self, world_size: int) -> None:
        world_size = _positive_int(world_size, "world_size")
        dp = _positive_int(self.dp, "fixed_topology.dp")
        cfp = _positive_int(self.cfp, "fixed_topology.cfp")
        cp = _positive_int(self.cp, "fixed_topology.cp")
        if cfp not in (1, 2):
            raise ValueError(f"fixed_topology.cfp must be 1 or 2, got {cfp}")
        if dp * cfp * cp != world_size:
            raise ValueError(
                "fixed_topology must cover the world exactly: "
                f"dp({dp}) * cfp({cfp}) * cp({cp}) != world_size({world_size})"
            )


@dataclass(frozen=True)
class LaneTopology:
    offset: int
    width: int
    cfp_degree: int
    cp_degree: int
    active: bool = True

    @classmethod
    def idle(cls, rank: int) -> "LaneTopology":
        return cls(
            offset=int(rank),
            width=1,
            cfp_degree=1,
            cp_degree=1,
            active=False,
        )

    def validate(self, *, world_size: int, rank: Optional[int] = None) -> None:
        world_size = _positive_int(world_size, "world_size")
        offset = int(self.offset)
        width = _positive_int(self.width, "lane.width")
        cfp = _positive_int(self.cfp_degree, "lane.cfp_degree")
        cp = _positive_int(self.cp_degree, "lane.cp_degree")
        if cfp not in (1, 2):
            raise ValueError(f"lane.cfp_degree must be 1 or 2, got {cfp}")
        if cfp * cp != width:
            raise ValueError(
                f"lane width {width} must equal cfp_degree({cfp}) * cp_degree({cp})"
            )
        if offset < 0 or offset + width > world_size:
            raise ValueError(
                f"lane [{offset}, {offset + width}) is outside world_size={world_size}"
            )
        if offset % width != 0:
            raise ValueError(f"lane offset {offset} must be aligned to width {width}")
        if rank is not None and self.active and not (offset <= rank < offset + width):
            raise ValueError(
                f"rank {rank} is not in active lane [{offset}, {offset + width})"
            )


@dataclass(frozen=True)
class LaneTopologyConfig:
    mode: TopologyMode
    fixed: Optional[FixedTopology] = None
    elastic_allowed_widths: tuple[int, ...] = ()
    elastic_cfg_parallel_max: int = 2

    def validate(self, world_size: int) -> None:
        world_size = _positive_int(world_size, "world_size")
        if self.mode not in ("fixed", "elastic"):
            raise ValueError(f"topology_mode must be fixed or elastic, got {self.mode!r}")
        if self.mode == "fixed":
            if self.fixed is None:
                raise ValueError("fixed topology mode requires fixed_topology")
            self.fixed.validate(world_size)
            return
        if self.fixed is not None:
            raise ValueError("elastic topology mode cannot set fixed_topology")
        if self.elastic_cfg_parallel_max not in (1, 2):
            raise ValueError("elastic_cfg_parallel_max must be 1 or 2")
        for width in self.elastic_allowed_widths:
            width = _positive_int(width, "elastic_allowed_width")
            if world_size % width != 0:
                raise ValueError(
                    f"elastic lane width {width} must divide world_size {world_size}"
                )

    def allowed_live_cp_degrees(self) -> tuple[int, ...]:
        if self.mode == "fixed":
            assert self.fixed is not None
            return (self.fixed.cp,)
        degrees = set(self.elastic_allowed_widths)
        if self.elastic_cfg_parallel_max >= 2:
            degrees.update(width // 2 for width in self.elastic_allowed_widths if width % 2 == 0)
        return tuple(sorted(degrees))

    def allowed_live_cfg_strides(self) -> tuple[int, ...]:
        if self.mode == "fixed":
            assert self.fixed is not None
            return (self.fixed.cp,) if self.fixed.cfp == 2 else ()
        if self.elastic_cfg_parallel_max < 2:
            return ()
        return tuple(sorted({width // 2 for width in self.elastic_allowed_widths if width % 2 == 0}))


def _parse_widths(values: Any, world_size: int) -> tuple[int, ...]:
    if values in (None, ""):
        return _divisors(world_size)
    if isinstance(values, str):
        values = [item.strip() for item in values.split(",") if item.strip()]
    widths = tuple(sorted({_positive_int(value, "elastic_allowed_width") for value in values}))
    return widths or _divisors(world_size)


def resolve_lane_topology_config(
    diffusion_config: Any,
    *,
    world_size: int,
    legacy_dp_size: int = 1,
    legacy_cfp_size: int = 1,
) -> LaneTopologyConfig:
    world_size = _positive_int(world_size, "world_size")
    mode = _read(diffusion_config, "topology_mode", None)
    if mode in ("", None):
        mode = "elastic" if bool(_read(diffusion_config, "dynamic_sp", False)) else "fixed"
        logger.warning(
            "topology_mode is not set; mapping legacy launch fields "
            "(dp_size/cfg_size/cp_size/dynamic_sp) to topology_mode=%s. This "
            "compatibility shim is deprecated -- set infer.diffusion.topology_mode "
            "(with fixed_topology or elastic_allowed_widths) explicitly.",
            mode,
        )
    mode = str(mode).lower()

    if mode == "fixed":
        raw = _read(diffusion_config, "fixed_topology", None)
        if raw is None:
            fixed = FixedTopology(
                dp=_positive_int(legacy_dp_size, "dp_size"),
                cfp=_positive_int(legacy_cfp_size, "cfg_size"),
                cp=_positive_int(_read(diffusion_config, "cp_size", 1), "cp_size"),
            )
        else:
            fixed = FixedTopology(
                dp=_positive_int(_read(raw, "dp", 1), "fixed_topology.dp"),
                cfp=_positive_int(_read(raw, "cfp", 1), "fixed_topology.cfp"),
                cp=_positive_int(_read(raw, "cp", 1), "fixed_topology.cp"),
            )
        result = LaneTopologyConfig(mode="fixed", fixed=fixed)
    elif mode == "elastic":
        epe_config = _read(diffusion_config, "epe", None)
        result = LaneTopologyConfig(
            mode="elastic",
            elastic_allowed_widths=_parse_widths(
                _read(diffusion_config, "elastic_allowed_widths", None),
                world_size,
            ),
            elastic_cfg_parallel_max=_positive_int(
                _read(epe_config, "cfg_parallel_max", 2),
                "epe.cfg_parallel_max",
            ),
        )
    else:
        raise ValueError(f"topology_mode must be fixed or elastic, got {mode!r}")

    result.validate(world_size)
    return result


def up_rank_lists(
    *,
    cp_degree: int,
    up_degree: int,
    world_size: int,
) -> list[list[int]]:
    cp_degree = _positive_int(cp_degree, "cp_degree")
    up_degree = _positive_int(up_degree, "up_degree")
    world_size = _positive_int(world_size, "world_size")
    if world_size % cp_degree != 0:
        raise ValueError(f"cp_degree {cp_degree} must divide world_size {world_size}")
    if cp_degree % up_degree != 0:
        raise ValueError(f"up_degree {up_degree} must divide cp_degree {cp_degree}")
    groups: list[list[int]] = []
    for base in range(0, world_size, cp_degree):
        for shard in range(cp_degree // up_degree):
            start = base + shard * up_degree
            groups.append(list(range(start, start + up_degree)))
    return groups
