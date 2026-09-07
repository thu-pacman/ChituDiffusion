"""Declarative tensor parallelism for regular transformer graphs.

Splitting a transformer by weight is mechanical once someone states which
projection splits along its output rows and which splits along its input
columns. This module takes that statement as data, so a model contributes a
table instead of a hand-written parallel copy of its module graph.

Every linear layer must be named by exactly one rule. An unclassified layer is
an error rather than a silent replication, because getting a column/row pair
wrong does not raise -- it quietly computes the wrong answer, and an upstream
release that introduces a new projection would otherwise slip through.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Literal

from torch import nn

from .norm import TensorParallelRMSNorm
from .plain_linear import PlainColumnParallelLinear, PlainRowParallelLinear
from .topology import get_tp_world_size

Role = Literal["column", "row", "replicated"]


def _match_segments(name: list[str], pattern: list[str]) -> bool:
    if not pattern:
        return not name
    head, rest = pattern[0], pattern[1:]
    if head == "**":
        return any(_match_segments(name[index:], rest) for index in range(len(name) + 1))
    if not name:
        return False
    return fnmatchcase(name[0], head) and _match_segments(name[1:], rest)


def matches(name: str, pattern: str) -> bool:
    """Match a qualified module name one path segment at a time.

    ``*`` stays inside a segment and ``**`` spans any number of them, so
    ``**.attention`` selects the attention modules without also selecting the
    projections underneath them.
    """

    return _match_segments(name.split("."), pattern.split("."))


@dataclass(frozen=True, slots=True)
class TensorParallelPlan:
    """Which linear layers shard, how, and which head counts shrink with them.

    Patterns are segment-wise globs over qualified module names; see
    :func:`matches`. A layer whose output rows are split is ``column``; a layer
    whose input columns are split, and which therefore all-reduces, is ``row``.
    A column layer must feed a row layer for the pair to reconstruct the full
    activation.

    ``norm`` names normalization layers whose normalized dimension is one of
    the dimensions being split, so they must reduce across ranks instead of
    normalizing over a shard. Unlike the linear roles these patterns may match
    nothing, because the layers they cover are often conditional.

    ``head_counts`` names attributes holding an attention head count. Sharding
    the projections shrinks the heads each rank holds, so an attention
    implementation that reshapes by this attribute needs it divided too.
    """

    column: tuple[str, ...] = ()
    row: tuple[str, ...] = ()
    replicated: tuple[str, ...] = ()
    norm: tuple[str, ...] = ()
    head_counts: tuple[tuple[str, str], ...] = ()

    def role(self, name: str) -> Role | None:
        matched = [
            role
            for role, patterns in (
                ("column", self.column),
                ("row", self.row),
                ("replicated", self.replicated),
            )
            if any(matches(name, pattern) for pattern in patterns)
        ]
        if len(matched) > 1:
            raise ValueError(
                f"{name} matches more than one tensor-parallel role: "
                f"{', '.join(matched)}"
            )
        return matched[0] if matched else None  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class TensorParallelRewrite:
    """What a plan did, for logging and for tests to assert against."""

    degree: int
    column: tuple[str, ...] = ()
    row: tuple[str, ...] = ()
    replicated: tuple[str, ...] = ()
    norm: tuple[str, ...] = ()
    head_counts: tuple[tuple[str, str, int], ...] = ()

    @property
    def sharded(self) -> int:
        return len(self.column) + len(self.row)

    def describe(self) -> str:
        return (
            f"TP{self.degree}: {len(self.column)} column, {len(self.row)} row, "
            f"{len(self.replicated)} replicated linears, "
            f"{len(self.norm)} reducing norms"
        )


def _normalized_width(module: nn.Module) -> int | None:
    """The width of a per-feature scale, or None if this is not such a layer."""

    if isinstance(module, nn.Linear):
        return None
    weight = module._parameters.get("weight")
    if weight is None or weight.ndim != 1:
        return None
    return int(weight.shape[0])


def _reject_undeclared_sharded_norms(
    modules: dict[str, nn.Module],
    roles: dict[str, Role | None],
    declared: set[str],
) -> None:
    """Refuse a scale that lands on a column-parallel output but was not named.

    Such a layer normalizes over a dimension the shards divide, so each rank
    would reduce over its own slice and quietly produce the wrong scale. Wan's
    qk-norm is exactly this shape. The test is local: a per-feature scale whose
    width matches a column-parallel sibling's output width.
    """

    widths: dict[str, set[int]] = {}
    for name, role in roles.items():
        if role != "column":
            continue
        parent, _, _ = name.rpartition(".")
        widths.setdefault(parent, set()).add(int(modules[name].out_features))

    suspects = sorted(
        name
        for name, module in modules.items()
        if name not in declared
        and _normalized_width(module) in widths.get(name.rpartition(".")[0], ())
    )
    if suspects:
        raise ValueError(
            "these layers scale a column-parallel output but the plan does not "
            f"list them under norm: {', '.join(suspects)}"
        )


def _replace(parent: nn.Module, leaf: str, source: nn.Linear, role: Role) -> None:
    kind = (
        PlainColumnParallelLinear if role == "column" else PlainRowParallelLinear
    )
    replacement = kind(
        source.in_features,
        source.out_features,
        bias=source.bias is not None,
        params_dtype=source.weight.dtype,
        device=source.weight.device,
    )
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and leaf.isdigit():
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def validate_tensor_parallel_plan(
    model: nn.Module,
    plan: TensorParallelPlan,
    *,
    degree: int,
) -> TensorParallelRewrite:
    """Report what ``plan`` would do to ``model`` at ``degree``, changing nothing.

    Needs no process group, so a test can check that a table still covers its
    model's graph and divides evenly at every degree the model claims to
    support, on a meta-device instance and in milliseconds.
    """

    return _apply(model, plan, degree=int(degree), rewrite=False)


def apply_tensor_parallel_plan(
    model: nn.Module,
    plan: TensorParallelPlan,
    *,
    degree: int | None = None,
) -> TensorParallelRewrite:
    """Shard ``model`` in place according to ``plan`` and report what changed."""

    resolved = get_tp_world_size() if degree is None else int(degree)
    return _apply(model, plan, degree=resolved, rewrite=True)


def _apply(
    model: nn.Module,
    plan: TensorParallelPlan,
    *,
    degree: int,
    rewrite: bool,
) -> TensorParallelRewrite:
    resolved = degree
    if resolved < 1:
        raise ValueError("tensor-parallel degree must be positive")

    modules = dict(model.named_modules())
    linears = [
        (name, module)
        for name, module in modules.items()
        if isinstance(module, nn.Linear)
    ]
    roles = {name: plan.role(name) for name, _ in linears}
    unclassified = sorted(name for name, role in roles.items() if role is None)
    if unclassified:
        raise ValueError(
            "the tensor-parallel plan does not name these linear layers: "
            f"{', '.join(unclassified[:10])}"
            + (f" (and {len(unclassified) - 10} more)" if len(unclassified) > 10 else "")
        )

    for name, module in linears:
        role = roles[name]
        axis, extent = (
            ("out_features", module.out_features)
            if role == "column"
            else ("in_features", module.in_features)
        )
        if role != "replicated" and extent % resolved:
            raise ValueError(
                f"{name}.{axis}={extent} is not divisible by tensor-parallel "
                f"degree {resolved}"
            )

    norms = [
        name
        for name in modules
        if any(matches(name, pattern) for pattern in plan.norm)
    ]
    for name in norms:
        size = _normalized_width(modules[name])
        if size is None:
            raise ValueError(f"{name} is not a normalization layer with a scale")
        if size % resolved:
            raise ValueError(
                f"{name} normalizes over {size}, which is not divisible by "
                f"tensor-parallel degree {resolved}"
            )
    _reject_undeclared_sharded_norms(modules, roles, set(norms))

    head_counts: list[tuple[str, str, int]] = []
    for pattern, attribute in plan.head_counts:
        selected = [name for name in modules if matches(name, pattern)]
        if not selected:
            raise ValueError(f"no module matches the head-count pattern {pattern}")
        for name in selected:
            module = modules[name]
            if not hasattr(module, attribute):
                raise ValueError(f"{name} has no attribute {attribute}")
            total = int(getattr(module, attribute))
            if total % resolved:
                raise ValueError(
                    f"{name}.{attribute}={total} is not divisible by "
                    f"tensor-parallel degree {resolved}"
                )
            head_counts.append((name, attribute, total // resolved))

    if rewrite and resolved > 1:
        active = get_tp_world_size()
        if resolved != active:
            # The replacements size themselves from the active topology, so a
            # mismatch would build shards for a degree nobody is running.
            raise ValueError(
                f"tensor-parallel degree {resolved} does not match the active "
                f"topology degree {active}"
            )
        for name, module in linears:
            if roles[name] == "replicated":
                continue
            parent_name, _, leaf = name.rpartition(".")
            _replace(modules[parent_name], leaf, module, roles[name])  # type: ignore[arg-type]
        for name in norms:
            parent_name, _, leaf = name.rpartition(".")
            setattr(
                modules[parent_name],
                leaf,
                TensorParallelRMSNorm.from_dense(modules[name]),
            )
        for name, attribute, local in head_counts:
            setattr(modules[name], attribute, local)

    by_role = {
        role: tuple(
            sorted(name for name, value in roles.items() if value == role)
        )
        for role in ("column", "row", "replicated")
    }
    return TensorParallelRewrite(
        degree=resolved,
        column=by_role["column"],
        row=by_role["row"],
        replicated=by_role["replicated"],
        norm=tuple(sorted(norms)),
        head_counts=tuple(head_counts),
    )


__all__ = [
    "TensorParallelPlan",
    "TensorParallelRewrite",
    "apply_tensor_parallel_plan",
    "matches",
    "validate_tensor_parallel_plan",
]
